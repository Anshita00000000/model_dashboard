"""Tests for app/core/assemble.py — canonical data is the spine; every
enrichment source is left-joined onto it, never the reverse. See CLAUDE.md:
enrichment coverage is only 35-43%, so most leads have no match, and that
must stay a legitimate, informative (null) state rather than shrinking the
dataset.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.core import assemble, schema, storage
from app.core.enrichment import ADAPTER_REGISTRY
from app.core.enrichment import runner as enrichment_runner
from app.core.enrichment.base import CanonicalRecord, EnrichmentAdapter, RawResponse, ResponseStatus


def _seed_canonical_dataset(store: storage.MetadataStore, root, canonical_df: pd.DataFrame, canonical_dataset_id: str = "cd1", merchant: str = "Evoke") -> None:
    parquet_path = storage.save_dataframe(canonical_df, "canonical", merchant, f"{canonical_dataset_id}_canonical", root=root)
    store.insert_cleaned_dataset(
        cleaned_dataset_id="cln1", merchant=merchant, purpose="train", original_filename="f.csv",
        stored_path="p", parquet_path="p2", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=len(canonical_df), col_count=len(canonical_df.columns), content_hash="h",
        source_raw_dataset_id=None, uploaded_by="t",
    )
    store.insert_field_mapping(
        mapping_id="map1", merchant=merchant, name="m1", version=1,
        mapping_json="{}", unmapped_columns_json="[]", created_by="t",
    )
    store.insert_canonical_dataset(
        canonical_dataset_id=canonical_dataset_id, merchant=merchant, purpose="train",
        cleaned_dataset_id="cln1", mapping_id="map1", row_count=len(canonical_df), col_count=len(canonical_df.columns),
        artifact_path=str(parquet_path), validation_json="{}", created_by="t",
    )


def _canonical_df(n: int = 5) -> pd.DataFrame:
    dispositions = ["Lead", "Engaged", "Converted"]
    return pd.DataFrame({
        "lead_id": [str(i) for i in range(n)],
        "phone": [f"9000000{i:03d}" for i in range(n)],
        "created_at": ["2024-01-01"] * n,
        "merchant": ["Evoke"] * n,
        "disposition": [dispositions[i % len(dispositions)] for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Row-count invariant — the core guarantee of this module
# ---------------------------------------------------------------------------


def test_assemble_preserves_row_count_with_no_enrichment_runs(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(5)
    _seed_canonical_dataset(store, tmp_path, df)
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert result.row_count == 5
    assert result.sources == []


def test_assemble_preserves_row_count_with_partial_enrichment_coverage(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(20)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    # low match rate on purpose: most leads get no bureau match
    enrichment_runner.run_enrichment(
        store=store, canonical_dataset_id="cd1", records=records,
        adapters=[ADAPTER_REGISTRY["equifax"](match_rate=0.1, seed=3)],
    )
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert result.row_count == 20  # unchanged regardless of match rate
    assert len(result.df) == 20


def test_unmatched_leads_stay_in_dataset_with_nulls_not_dropped(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(3)
    _seed_canonical_dataset(store, tmp_path, df)

    class AllNoHitAdapter(EnrichmentAdapter):
        source_name = "equifax"
        def fetch(self, records, *, batch_id):
            return {r.lead_id: RawResponse(r.lead_id, "equifax", ResponseStatus.SUCCESS_NO_HIT.value, {}, "t", batch_id) for r in records}
        def transform(self, raw):
            return {"equifax_score": None}
        def feature_names(self):
            return ["equifax_score"]

    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[AllNoHitAdapter()])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert len(result.df) == 3
    assert result.df["equifax_score"].isna().all()
    assert result.df["lead_id"].tolist() == ["0", "1", "2"]  # every lead still present


def test_lead_with_no_phone_is_not_sent_and_still_present(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(3)
    df.loc[1, "phone"] = None
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["equifax"](seed=1)])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert len(result.df) == 3
    row = result.df[result.df["lead_id"] == "1"].iloc[0]
    assert row["equifax_match_status"] == "NOT_SENT"
    assert row["equifax_is_enriched"] == False  # noqa: E712 - explicit bool check


# ---------------------------------------------------------------------------
# Prefixing / provenance columns
# ---------------------------------------------------------------------------


def test_source_features_are_prefixed_by_source_name(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(5)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["equifax"](seed=1)])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert all(c.startswith("equifax_") for c in result.sources[0].feature_columns)
    assert "equifax_score" in result.df.columns


def test_provenance_columns_carried_through_per_source(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(5)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["crif"](seed=2)])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    for col in ["crif_match_status", "crif_fetched_at", "crif_batch_id", "crif_is_enriched"]:
        assert col in result.df.columns


def test_multiple_sources_all_joined(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(10)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    adapters = [ADAPTER_REGISTRY["equifax"](seed=1), ADAPTER_REGISTRY["crif"](seed=2)]
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=adapters)
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert {s.source for s in result.sources} == {"equifax", "crif"}
    assert result.row_count == 10


# ---------------------------------------------------------------------------
# is_enriched: computed from actual data presence, never trusted from a flag
# ---------------------------------------------------------------------------


def test_is_enriched_true_only_when_transform_produces_real_values(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(5)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["equifax"](seed=1)])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    matched_rows = result.df[result.df["equifax_match_status"] == "SUCCESS_MATCHED"]
    non_matched_rows = result.df[result.df["equifax_match_status"] != "SUCCESS_MATCHED"]
    assert (matched_rows["equifax_is_enriched"] == True).all()  # noqa: E712
    assert (non_matched_rows["equifax_is_enriched"] == False).all()  # noqa: E712


def test_is_enriched_does_not_trust_status_when_payload_is_actually_empty(tmp_path):
    # A matched response whose payload is malformed/empty must still show as
    # NOT enriched -- this is exactly the "never from a provided flag" rule.
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(1)
    _seed_canonical_dataset(store, tmp_path, df)

    class LiesAboutMatchingAdapter(EnrichmentAdapter):
        source_name = "equifax"
        def fetch(self, records, *, batch_id):
            return {r.lead_id: RawResponse(r.lead_id, "equifax", ResponseStatus.SUCCESS_MATCHED.value, {"malformed": True}, "t", batch_id) for r in records}
        def transform(self, raw):
            return {"equifax_score": None, "equifax_total_accounts": None}  # a real adapter returns null on unparseable payload
        def feature_names(self):
            return ["equifax_score", "equifax_total_accounts"]

    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[LiesAboutMatchingAdapter()])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert result.df.iloc[0]["equifax_match_status"] == "SUCCESS_MATCHED"
    assert result.df.iloc[0]["equifax_is_enriched"] == False  # noqa: E712


def test_is_enriched_falls_back_to_status_for_sources_with_no_features_yet(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(5)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["epfo"](seed=1)])
    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    matched_rows = result.df[result.df["epfo_match_status"] == "SUCCESS_MATCHED"]
    assert (matched_rows["epfo_is_enriched"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Idempotent re-runs: assembly always uses the LATEST batch per source
# ---------------------------------------------------------------------------


def test_assemble_uses_latest_batch_when_source_was_rerun(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(2)
    _seed_canonical_dataset(store, tmp_path, df)
    records = enrichment_runner.build_canonical_records(df)

    # First run: force everyone matched
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["equifax"](match_rate=1.0, error_rate=0.0, not_sent_rate=0.0, seed=1)])
    # Second run (rerun, new batch): force everyone no-hit
    enrichment_runner.run_enrichment(store=store, canonical_dataset_id="cd1", records=records, adapters=[ADAPTER_REGISTRY["equifax"](match_rate=0.0, error_rate=0.0, not_sent_rate=0.0, seed=1)])

    result = assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")
    assert (result.df["equifax_match_status"] == "SUCCESS_NO_HIT").all()  # the later run wins
    assert len(result.sources) == 1  # still just one SourceAssembly for equifax (its latest run)


# ---------------------------------------------------------------------------
# Missing canonical spine / lead_id column
# ---------------------------------------------------------------------------


def test_assemble_raises_for_unknown_canonical_dataset(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    with pytest.raises(ValueError):
        assemble.assemble_dataset(store=store, canonical_dataset_id="does-not-exist")


def test_assemble_raises_when_canonical_has_no_lead_id_column(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = pd.DataFrame({"phone": ["9000000001"]})
    _seed_canonical_dataset(store, tmp_path, df)
    with pytest.raises(ValueError):
        assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")


def test_duplicate_lead_id_within_a_run_raises(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = _canonical_df(2)
    _seed_canonical_dataset(store, tmp_path, df)
    # Manually insert a run with a duplicated lead_id response, simulating an adapter bug.
    store.insert_enrichment_run(
        run_id="run1", canonical_dataset_id="cd1", source="equifax", batch_id="b1",
        rows_attempted=2, matched=2, no_hit=0, error=0, not_sent=0, started_at="t", finished_at="t",
    )
    for _ in range(2):
        store.insert_enrichment_response(
            response_id=f"r{_}", run_id="run1", lead_id="0", source="equifax",
            status="SUCCESS_MATCHED", payload_json="{}", fetched_at="t", batch_id="b1",
        )
    with pytest.raises(ValueError):
        assemble.assemble_dataset(store=store, canonical_dataset_id="cd1")


# ---------------------------------------------------------------------------
# Export / registration
# ---------------------------------------------------------------------------


def test_default_role_overrides_excludes_only_provenance_not_is_enriched():
    class FakeSource:
        def __init__(self, source):
            self.source = source
            self.provenance_columns = [f"{source}_match_status", f"{source}_fetched_at", f"{source}_batch_id", f"{source}_is_enriched"]
            self.feature_columns = [f"{source}_score"]

    result = assemble.AssembleResult(df=pd.DataFrame(), row_count=0, canonical_dataset_id="x", sources=[FakeSource("crif")])
    overrides = assemble.default_role_overrides(result, target_column="disposition", metadata_columns=("merchant",))
    assert overrides["lead_id"]["role"] == "id"
    assert overrides["disposition"]["role"] == "target"
    assert overrides["merchant"]["role"] == "metadata"
    assert overrides["crif_match_status"]["role"] == "excluded"
    assert overrides["crif_fetched_at"]["role"] == "excluded"
    assert overrides["crif_batch_id"]["role"] == "excluded"
    assert "crif_is_enriched" not in overrides  # left to default to "feature"


def test_register_dataset_writes_a_parseable_phase1_compatible_artifact(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = pd.DataFrame({"lead_id": ["1", "2"], "disposition": ["Lead", "Converted"], "equifax_score": [700, None]})
    overrides = {"lead_id": {"role": "id"}, "disposition": {"role": "target"}}
    row = assemble.register_dataset(
        store=store, root=tmp_path, df=df, merchant="Evoke", purpose="train",
        source_file="assembled from cd1", role_overrides=overrides,
    )
    assert row["row_count"] == 2
    loaded_df = storage.load_dataframe(row["artifact_path"])
    pd.testing.assert_frame_equal(loaded_df.reset_index(drop=True), df.reset_index(drop=True))
    ds_schema = schema.DatasetSchema.from_dict(json.loads(row["schema_json"]))
    assert ds_schema.columns_with_role("id") == ["lead_id"]
    assert ds_schema.columns_with_role("target") == ["disposition"]


def test_register_dataset_is_append_only(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    df = pd.DataFrame({"lead_id": ["1"], "disposition": ["Lead"]})
    overrides = {"lead_id": {"role": "id"}, "disposition": {"role": "target"}}
    row1 = assemble.register_dataset(store=store, root=tmp_path, df=df, merchant="Evoke", purpose="train", source_file="s", role_overrides=overrides)
    row2 = assemble.register_dataset(store=store, root=tmp_path, df=df, merchant="Evoke", purpose="train", source_file="s", role_overrides=overrides)
    assert row1["dataset_id"] != row2["dataset_id"]
    assert len(store.list_datasets(merchant="Evoke")) == 2
