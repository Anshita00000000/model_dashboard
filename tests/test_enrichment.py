"""Tests for app/core/enrichment/ — adapter interface, stub adapters, and the
orchestration runner. fetch() is stubbed everywhere; these tests verify the
mock behaves realistically (configurable match rate, status mix, phone-less
records are NOT_SENT) and that transform()/the runner handle it correctly.
"""

from __future__ import annotations

import time
from collections import Counter

import pandas as pd
import pytest

from app.core import storage
from app.core.enrichment import ADAPTER_REGISTRY
from app.core.enrichment.base import (
    CanonicalRecord,
    EnrichmentAdapter,
    RawResponse,
    ResponseStatus,
    check_enrichment_timing_leakage,
    mock_fetch,
    new_batch_id,
)
from app.core.enrichment.crif import CrifAdapter, _parse_duration_months
from app.core.enrichment.equifax import EquifaxAdapter
from app.core.enrichment.runner import build_canonical_records, run_enrichment, run_source


def _records(n: int, *, with_phone_fraction: float = 1.0) -> list[CanonicalRecord]:
    out = []
    for i in range(n):
        has_phone = (i / n) < with_phone_fraction
        out.append(CanonicalRecord(lead_id=str(i), phone=f"900000{i:04d}" if has_phone else None))
    return out


# ---------------------------------------------------------------------------
# mock_fetch — shared status-mix behaviour
# ---------------------------------------------------------------------------


def test_mock_fetch_phone_less_records_are_always_not_sent():
    records = _records(20, with_phone_fraction=0.0)
    resp = mock_fetch(records, source_name="x", batch_id=new_batch_id(), payload_factory=lambda r, rng: {})
    assert all(r.status == ResponseStatus.NOT_SENT.value for r in resp.values())


def test_mock_fetch_match_rate_is_approximately_respected_at_scale():
    records = _records(5000)
    resp = mock_fetch(
        records, source_name="x", batch_id=new_batch_id(), match_rate=0.40, error_rate=0.0, not_sent_rate=0.0,
        payload_factory=lambda r, rng: {"hit": True}, seed=7,
    )
    matched_fraction = sum(1 for r in resp.values() if r.status == ResponseStatus.SUCCESS_MATCHED.value) / len(records)
    assert 0.35 < matched_fraction < 0.45


def test_mock_fetch_every_record_gets_a_response():
    records = _records(37)
    resp = mock_fetch(records, source_name="x", batch_id=new_batch_id(), payload_factory=lambda r, rng: {}, seed=1)
    assert set(resp.keys()) == {r.lead_id for r in records}


def test_mock_fetch_is_deterministic_given_a_seed():
    records = _records(50)
    resp1 = mock_fetch(records, source_name="x", batch_id="b", payload_factory=lambda r, rng: {"v": rng.random()}, seed=99)
    resp2 = mock_fetch(records, source_name="x", batch_id="b", payload_factory=lambda r, rng: {"v": rng.random()}, seed=99)
    assert {k: v.status for k, v in resp1.items()} == {k: v.status for k, v in resp2.items()}


def test_mock_fetch_rejects_invalid_rate_config():
    with pytest.raises(ValueError):
        mock_fetch(_records(1), source_name="x", batch_id="b", match_rate=0.8, error_rate=0.5, payload_factory=lambda r, rng: {})


def test_mock_fetch_carries_fetched_at_and_batch_id_on_every_response():
    records = _records(10)
    batch_id = new_batch_id()
    resp = mock_fetch(records, source_name="crif", batch_id=batch_id, payload_factory=lambda r, rng: {}, seed=1)
    for r in resp.values():
        assert r.batch_id == batch_id
        assert r.fetched_at
        assert r.source == "crif"


# ---------------------------------------------------------------------------
# Adapter registry — every adapter conforms to the interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_name", list(ADAPTER_REGISTRY.keys()))
def test_every_registered_adapter_fetches_and_transforms_without_error(source_name):
    adapter = ADAPTER_REGISTRY[source_name](seed=1)
    records = _records(200)
    responses = adapter.fetch(records, batch_id=new_batch_id())
    assert len(responses) == len(records)
    feature_names = adapter.feature_names()
    for resp in responses.values():
        features = adapter.transform(resp)
        assert isinstance(features, dict)
        assert set(features.keys()) <= set(feature_names) or features == {}


# ---------------------------------------------------------------------------
# Equifax transform — sentinel handling
# ---------------------------------------------------------------------------


def test_equifax_transform_maps_sentinel_score_to_none():
    resp = RawResponse(
        lead_id="1", source="equifax", status=ResponseStatus.SUCCESS_MATCHED.value,
        payload={"hit": True, "score": -1, "score_band": None, "total_accounts": 2, "total_balance": 100.0,
                 "enquiries_last_6m": 1, "delinquent_accounts": 0},
        fetched_at="2024-01-01T00:00:00+00:00", batch_id="b",
    )
    features = EquifaxAdapter().transform(resp)
    assert features["equifax_score"] is None
    assert features["equifax_total_accounts"] == 2  # a real value alongside the sentinel score is preserved


def test_equifax_transform_keeps_real_score():
    resp = RawResponse(
        lead_id="1", source="equifax", status=ResponseStatus.SUCCESS_MATCHED.value,
        payload={"hit": True, "score": 720, "score_band": "B", "total_accounts": 2, "total_balance": 100.0,
                 "enquiries_last_6m": 1, "delinquent_accounts": 0},
        fetched_at="2024-01-01T00:00:00+00:00", batch_id="b",
    )
    features = EquifaxAdapter().transform(resp)
    assert features["equifax_score"] == 720


def test_equifax_transform_returns_all_null_for_no_hit_never_zero_filled():
    resp = RawResponse(
        lead_id="1", source="equifax", status=ResponseStatus.SUCCESS_NO_HIT.value,
        payload={"hit": False, "score": -1, "total_accounts": 0, "total_balance": 0.0,
                 "enquiries_last_6m": 0, "delinquent_accounts": 0, "score_band": None},
        fetched_at="2024-01-01T00:00:00+00:00", batch_id="b",
    )
    features = EquifaxAdapter().transform(resp)
    # "no bureau data" must stay null, never look like an observed zero (CLAUDE.md)
    assert all(v is None for v in features.values())


def test_equifax_transform_returns_all_null_for_error_and_not_sent():
    for status in (ResponseStatus.ERROR.value, ResponseStatus.NOT_SENT.value):
        resp = RawResponse(lead_id="1", source="equifax", status=status, payload={}, fetched_at="t", batch_id="b")
        features = EquifaxAdapter().transform(resp)
        assert all(v is None for v in features.values())


# ---------------------------------------------------------------------------
# CRIF transform — sentinel handling + text-duration parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel_score", [0, 10, 15, 18])
def test_crif_transform_maps_sentinel_reason_codes_to_none(sentinel_score):
    resp = RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_MATCHED.value,
        payload={"hit": True, "score": sentinel_score, "total_accounts": 3, "active_accounts": 2,
                 "overdue_accounts": 0, "total_outstanding": 5000.0, "credit_history_length_text": "2yrs 3mon"},
        fetched_at="t", batch_id="b",
    )
    features = CrifAdapter().transform(resp)
    assert features["crif_score"] is None


def test_crif_transform_keeps_real_score_starting_around_300():
    resp = RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_MATCHED.value,
        payload={"hit": True, "score": 650, "total_accounts": 3, "active_accounts": 2,
                 "overdue_accounts": 0, "total_outstanding": 5000.0, "credit_history_length_text": "2yrs 3mon"},
        fetched_at="t", batch_id="b",
    )
    features = CrifAdapter().transform(resp)
    assert features["crif_score"] == 650


def test_crif_parses_text_encoded_duration_to_months():
    assert _parse_duration_months("6yrs 0mon") == 72
    assert _parse_duration_months("0yrs 6mon") == 6
    assert _parse_duration_months("2yrs 3mon") == 27


def test_crif_parse_duration_returns_none_for_unparseable_text():
    assert _parse_duration_months(None) is None
    assert _parse_duration_months("") is None
    assert _parse_duration_months("garbage") is None


def test_crif_transform_no_hit_is_all_null():
    resp = RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_NO_HIT.value,
        payload={"hit": False, "score": 0, "total_accounts": 0, "active_accounts": 0,
                 "overdue_accounts": 0, "total_outstanding": 0.0, "credit_history_length_text": "0yrs 0mon"},
        fetched_at="t", batch_id="b",
    )
    features = CrifAdapter().transform(resp)
    assert all(v is None for v in features.values())


# ---------------------------------------------------------------------------
# Stub-only adapters (epfo, salary_estimator, lead_credit_engine, payu)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_name", ["epfo", "salary_estimator", "lead_credit_engine", "payu"])
def test_unimplemented_adapters_return_empty_transform_and_feature_list(source_name):
    adapter = ADAPTER_REGISTRY[source_name](seed=1)
    records = _records(20)
    responses = adapter.fetch(records, batch_id=new_batch_id())
    matched = next(r for r in responses.values() if r.status == ResponseStatus.SUCCESS_MATCHED.value)
    assert adapter.transform(matched) == {}
    assert adapter.feature_names() == []
    # fetch() itself still returns a structurally-real mock payload, even though
    # transform() doesn't consume it yet
    assert isinstance(matched.payload, dict)
    assert matched.payload.get("hit") is True


# ---------------------------------------------------------------------------
# Timing-leakage check
# ---------------------------------------------------------------------------


def test_timing_leakage_flags_fetch_after_outcome():
    responses = [RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_MATCHED.value, payload={},
        fetched_at="2024-06-01T00:00:00+00:00", batch_id="b",
    )]
    warnings = check_enrichment_timing_leakage(outcome_dates={"1": "2024-01-01"}, responses=responses)
    assert len(warnings) == 1
    assert warnings[0].lead_id == "1"


def test_timing_leakage_silent_when_fetch_precedes_outcome():
    responses = [RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_MATCHED.value, payload={},
        fetched_at="2024-01-01T00:00:00+00:00", batch_id="b",
    )]
    warnings = check_enrichment_timing_leakage(outcome_dates={"1": "2024-06-01"}, responses=responses)
    assert warnings == []


def test_timing_leakage_skips_leads_with_no_known_outcome_date():
    responses = [RawResponse(
        lead_id="1", source="crif", status=ResponseStatus.SUCCESS_MATCHED.value, payload={},
        fetched_at="2024-06-01T00:00:00+00:00", batch_id="b",
    )]
    assert check_enrichment_timing_leakage(outcome_dates={}, responses=responses) == []


# ---------------------------------------------------------------------------
# build_canonical_records
# ---------------------------------------------------------------------------


def test_build_canonical_records_blank_phone_becomes_none():
    df = pd.DataFrame({"lead_id": ["1", "2"], "phone": ["9000000001", None]})
    records = build_canonical_records(df)
    assert records[0].phone == "9000000001"
    assert records[1].phone is None


def test_build_canonical_records_carries_extra_fields():
    df = pd.DataFrame({"lead_id": ["1"], "phone": ["9000000001"], "disposition": ["Lead"]})
    records = build_canonical_records(df)
    assert records[0].fields["disposition"] == "Lead"


# ---------------------------------------------------------------------------
# Orchestration — persistence, idempotency, retries
# ---------------------------------------------------------------------------


def _seed_canonical_dataset(store: storage.MetadataStore, canonical_dataset_id: str = "cd1") -> None:
    store.insert_cleaned_dataset(
        cleaned_dataset_id="cln1", merchant="Evoke", purpose="train", original_filename="f.csv",
        stored_path="p", parquet_path="p2", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=1, col_count=1, content_hash="h", source_raw_dataset_id=None, uploaded_by="t",
    )
    store.insert_field_mapping(
        mapping_id="map1", merchant="Evoke", name="m1", version=1,
        mapping_json="{}", unmapped_columns_json="[]", created_by="t",
    )
    store.insert_canonical_dataset(
        canonical_dataset_id=canonical_dataset_id, merchant="Evoke", purpose="train",
        cleaned_dataset_id="cln1", mapping_id="map1", row_count=1, col_count=1,
        artifact_path="p", validation_json="{}", created_by="t",
    )


def test_run_source_persists_every_response_and_records_the_run(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)
    records = _records(30)
    adapter = ADAPTER_REGISTRY["equifax"](seed=1)
    summary = run_source(adapter, records, store=store, canonical_dataset_id="cd1")

    run_row = store.get_enrichment_run(summary.run_id)
    assert run_row is not None
    assert run_row["rows_attempted"] == 30
    assert run_row["matched"] + run_row["no_hit"] + run_row["error"] + run_row["not_sent"] == 30

    responses = store.list_enrichment_responses(run_id=summary.run_id)
    assert len(responses) == 30
    for r in responses:
        assert r["batch_id"] == summary.batch_id
        assert r["fetched_at"]


def test_run_source_is_idempotent_rerun_creates_new_batch_not_overwrite(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)
    records = _records(10)
    adapter = ADAPTER_REGISTRY["equifax"](seed=1)
    summary1 = run_source(adapter, records, store=store, canonical_dataset_id="cd1")
    summary2 = run_source(adapter, records, store=store, canonical_dataset_id="cd1")
    assert summary1.batch_id != summary2.batch_id
    assert summary1.run_id != summary2.run_id
    assert len(store.list_enrichment_runs("cd1")) == 2


def test_run_source_retries_error_status_and_eventually_succeeds(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)

    class FlakyThenSucceedsAdapter(EnrichmentAdapter):
        source_name = "flaky"

        def __init__(self):
            self.calls = 0

        def fetch(self, records, *, batch_id):
            self.calls += 1
            status = ResponseStatus.ERROR.value if self.calls < 3 else ResponseStatus.SUCCESS_MATCHED.value
            return {r.lead_id: RawResponse(r.lead_id, self.source_name, status, {}, "t", batch_id) for r in records}

        def transform(self, raw):
            return {}

        def feature_names(self):
            return []

    adapter = FlakyThenSucceedsAdapter()
    records = _records(5)
    summary = run_source(adapter, records, store=store, canonical_dataset_id="cd1", max_retries=5, backoff_seconds=0)
    assert summary.matched == 5
    assert summary.error == 0
    assert adapter.calls == 3


def test_run_source_gives_up_after_max_retries_status_stays_error(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)

    class AlwaysErrorsAdapter(EnrichmentAdapter):
        source_name = "broken"

        def fetch(self, records, *, batch_id):
            return {r.lead_id: RawResponse(r.lead_id, self.source_name, ResponseStatus.ERROR.value, {}, "t", batch_id) for r in records}

        def transform(self, raw):
            return {}

        def feature_names(self):
            return []

    summary = run_source(AlwaysErrorsAdapter(), _records(3), store=store, canonical_dataset_id="cd1", max_retries=2, backoff_seconds=0)
    assert summary.error == 3
    assert summary.matched == 0


def test_run_source_no_hit_and_not_sent_are_never_retried(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)

    class TerminalStatusAdapter(EnrichmentAdapter):
        source_name = "terminal"

        def __init__(self):
            self.calls = 0

        def fetch(self, records, *, batch_id):
            self.calls += 1
            return {r.lead_id: RawResponse(r.lead_id, self.source_name, ResponseStatus.SUCCESS_NO_HIT.value, {}, "t", batch_id) for r in records}

        def transform(self, raw):
            return {}

        def feature_names(self):
            return []

    adapter = TerminalStatusAdapter()
    run_source(adapter, _records(4), store=store, canonical_dataset_id="cd1", max_retries=5, backoff_seconds=0)
    assert adapter.calls == 1  # no retries triggered for a terminal status


def test_run_enrichment_runs_every_selected_source_and_reports_progress(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)
    adapters = [ADAPTER_REGISTRY["equifax"](seed=1), ADAPTER_REGISTRY["crif"](seed=2)]
    progress_calls = []
    summaries = run_enrichment(
        store=store, canonical_dataset_id="cd1", records=_records(20), adapters=adapters,
        progress_callback=lambda name, summary: progress_calls.append(name),
    )
    assert {s.source for s in summaries} == {"equifax", "crif"}
    assert progress_calls == ["equifax", "crif"]
    assert len(store.list_enrichment_runs("cd1")) == 2


def test_run_source_backoff_uses_injected_sleep_fn(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    _seed_canonical_dataset(store)

    class AlwaysErrorsAdapter(EnrichmentAdapter):
        source_name = "broken"

        def fetch(self, records, *, batch_id):
            return {r.lead_id: RawResponse(r.lead_id, self.source_name, ResponseStatus.ERROR.value, {}, "t", batch_id) for r in records}

        def transform(self, raw):
            return {}

        def feature_names(self):
            return []

    sleeps = []
    run_source(
        AlwaysErrorsAdapter(), _records(2), store=store, canonical_dataset_id="cd1",
        max_retries=3, backoff_seconds=1.0, sleep_fn=sleeps.append,
    )
    assert sleeps == [1.0, 2.0, 4.0]  # exponential backoff, never a real time.sleep in tests
