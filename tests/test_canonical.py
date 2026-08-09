"""Tests for app/core/canonical.py — the cross-merchant schema layer.

Phone normalisation gets its own focused section: a bug here silently
destroys enrichment match rates (see CLAUDE.md).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.core import canonical, storage


# ---------------------------------------------------------------------------
# Phone normalisation — get this exactly right
# ---------------------------------------------------------------------------


def test_normalize_phone_strips_float_artifact_before_extracting_digits():
    # The exact incident from CLAUDE.md: extracting digits first would keep
    # the ".0"'s trailing zero as a real digit and slice off the leading one,
    # producing 1 match instead of 16,073.
    assert canonical.normalize_phone(7889358744.0) == "7889358744"
    assert canonical.normalize_phone("7889358744.0") == "7889358744"


def test_normalize_phone_strips_non_digits():
    assert canonical.normalize_phone("+91-7889358744") == "7889358744"
    assert canonical.normalize_phone("(788) 935-8744") == "7889358744"


def test_normalize_phone_keeps_last_ten_digits_with_country_code():
    assert canonical.normalize_phone("917889358744") == "7889358744"


def test_normalize_phone_returns_none_for_too_short():
    assert canonical.normalize_phone("12345") is None
    assert canonical.normalize_phone("") is None


def test_normalize_phone_returns_none_for_blank():
    assert canonical.normalize_phone(None) is None
    assert canonical.normalize_phone(float("nan")) is None
    assert canonical.normalize_phone("   ") is None


def test_normalize_phone_does_not_confuse_a_genuine_trailing_zero_pair():
    # A 10-digit number that happens to end in "00" is not a float artifact —
    # only an exact ".0" suffix is stripped.
    assert canonical.normalize_phone("7889358700") == "7889358700"


# ---------------------------------------------------------------------------
# Canonical field configuration
# ---------------------------------------------------------------------------


def test_seed_canonical_fields_is_idempotent(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    canonical.seed_canonical_fields(store)
    first = store.list_canonical_fields()
    canonical.seed_canonical_fields(store)
    second = store.list_canonical_fields()
    assert len(first) == len(canonical.CANONICAL_FIELD_SEEDS)
    assert len(second) == len(first)


def test_seed_canonical_fields_marks_disposition_required_for_train_only(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    canonical.seed_canonical_fields(store)
    disposition = store.get_canonical_field_by_name("disposition")
    lead_id = store.get_canonical_field_by_name("lead_id")
    assert disposition["required_level"] == canonical.REQUIRED_FOR_TRAIN
    assert lead_id["required_level"] == canonical.REQUIRED
    assert lead_id["unique_required"] == 1


def test_add_canonical_field_inline(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    field_id = canonical.add_canonical_field(store, name="insurance_provider", dtype="categorical")
    field = store.get_canonical_field_by_name("insurance_provider")
    assert field["field_id"] == field_id
    assert field["required_level"] == canonical.OPTIONAL


def test_add_canonical_field_is_idempotent_by_name(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    id1 = canonical.add_canonical_field(store, name="new_field", dtype="text")
    id2 = canonical.add_canonical_field(store, name="new_field", dtype="text")
    assert id1 == id2
    assert len(store.list_canonical_fields()) == 1


def test_add_canonical_field_rejects_invalid_required_level(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    with pytest.raises(ValueError):
        canonical.add_canonical_field(store, name="x", dtype="text", required_level="sometimes")


# ---------------------------------------------------------------------------
# Mapping suggestion — ranked, never auto-applied
# ---------------------------------------------------------------------------


def test_suggest_mapping_finds_alias_matches_with_high_confidence():
    fields = [{"name": "phone"}, {"name": "created_at"}]
    suggestions = canonical.suggest_mapping(["Mobile Number", "Lead Created Date", "unrelated_col"], fields)
    assert suggestions["phone"][0].source_column == "Mobile Number"
    assert suggestions["phone"][0].confidence == canonical.ALIAS_CONFIDENCE


def test_suggest_mapping_is_report_only_does_not_mutate_inputs():
    fields = [{"name": "phone"}]
    cols = ["Mobile Number"]
    before = list(cols)
    canonical.suggest_mapping(cols, fields)
    assert cols == before


def test_suggest_mapping_ranks_best_match_first():
    fields = [{"name": "lead_id"}]
    suggestions = canonical.suggest_mapping(["lead_id", "unrelated"], fields)
    assert suggestions["lead_id"][0].source_column == "lead_id"
    assert suggestions["lead_id"][0].confidence == canonical.ALIAS_CONFIDENCE


# ---------------------------------------------------------------------------
# Applying a confirmed mapping
# ---------------------------------------------------------------------------


def test_apply_mapping_simple_rename():
    df = pd.DataFrame({"Mobile": ["9000000001", "9000000002"]})
    out = canonical.apply_mapping(df, {"phone": ["Mobile"]})
    assert out["phone"].tolist() == ["9000000001", "9000000002"]


def test_apply_mapping_many_to_one_first_non_null_wins():
    df = pd.DataFrame({
        "phone_primary": ["9000000001", None, ""],
        "phone_secondary": [None, "9000000002", "9000000003"],
    })
    out = canonical.apply_mapping(df, {"phone": ["phone_primary", "phone_secondary"]})
    assert out["phone"].tolist() == ["9000000001", "9000000002", "9000000003"]


def test_apply_mapping_precedence_order_matters():
    df = pd.DataFrame({"a": ["from_a", "from_a"], "b": ["from_b", "from_b"]})
    out_a_first = canonical.apply_mapping(df, {"field": ["a", "b"]})
    out_b_first = canonical.apply_mapping(df, {"field": ["b", "a"]})
    assert out_a_first["field"].tolist() == ["from_a", "from_a"]
    assert out_b_first["field"].tolist() == ["from_b", "from_b"]


def test_apply_mapping_preserves_unmapped_columns_under_extra_prefix():
    df = pd.DataFrame({"phone": ["9000000001"], "some_random_col": ["value"]})
    out = canonical.apply_mapping(df, {"phone": ["phone"]})
    assert "extra__some_random_col" in out.columns
    assert out["extra__some_random_col"].tolist() == ["value"]


def test_apply_mapping_never_drops_a_column():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = canonical.apply_mapping(df, {"x": ["a"]})
    # a -> x (mapped), b and c -> extra__b, extra__c (unmapped, preserved)
    assert set(out.columns) == {"x", "extra__b", "extra__c"}


def test_apply_mapping_does_not_mutate_input_df():
    df = pd.DataFrame({"a": ["1", "2"]})
    before = df.copy()
    canonical.apply_mapping(df, {"x": ["a"]})
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Named, versioned mapping profiles
# ---------------------------------------------------------------------------


def test_save_field_mapping_versions_increment_per_merchant(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    id1 = canonical.save_field_mapping(store, merchant="Evoke", name="v1", mapping={"phone": ["Mobile"]}, unmapped_columns=[], created_by="t")
    id2 = canonical.save_field_mapping(store, merchant="Evoke", name="v2", mapping={"phone": ["Contact"]}, unmapped_columns=[], created_by="t")
    row1, row2 = store.get_field_mapping(id1), store.get_field_mapping(id2)
    assert row1["version"] == 1
    assert row2["version"] == 2


def test_save_field_mapping_is_append_only_a_new_save_is_a_new_version(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    canonical.save_field_mapping(store, merchant="Evoke", name="v1", mapping={}, unmapped_columns=[], created_by="t")
    canonical.save_field_mapping(store, merchant="Evoke", name="v2", mapping={}, unmapped_columns=[], created_by="t")
    assert len(store.list_field_mappings(merchant="Evoke")) == 2


def test_get_latest_field_mapping_returns_most_recent(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    canonical.save_field_mapping(store, merchant="Evoke", name="v1", mapping={"a": ["x"]}, unmapped_columns=[], created_by="t")
    id2 = canonical.save_field_mapping(store, merchant="Evoke", name="v2", mapping={"a": ["y"]}, unmapped_columns=[], created_by="t")
    latest = canonical.get_latest_field_mapping(store, "Evoke")
    assert latest["mapping_id"] == id2
    assert canonical.load_mapping_dict(latest) == {"a": ["y"]}


def test_get_latest_field_mapping_none_when_no_mappings_saved(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    assert canonical.get_latest_field_mapping(store, "Evoke") is None


# ---------------------------------------------------------------------------
# Column diff (lineage)
# ---------------------------------------------------------------------------


def test_column_diff_detects_added_removed_unchanged():
    diff = canonical.column_diff(["a", "b", "c"], ["a", "b", "d"])
    assert diff.unchanged == ["a", "b"]
    assert diff.removed == ["c"]
    assert diff.added == ["d"]


def test_column_diff_heuristic_rename_detection():
    diff = canonical.column_diff(["Mobile Number"], ["mobile_number"])
    assert diff.renamed == [("Mobile Number", "mobile_number")]
    assert diff.added == []
    assert diff.removed == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _fields():
    return canonical.CANONICAL_FIELD_SEEDS


def test_validate_canonical_flags_missing_required_field():
    df = pd.DataFrame({"phone": ["9000000001"], "created_at": ["2024-01-01"], "merchant": ["Evoke"]})
    report = canonical.validate_canonical(df, _fields(), purpose="predict")
    kinds = {i.kind for i in report.issues}
    assert "missing_required_field" in kinds
    assert not report.is_valid


def test_validate_canonical_disposition_only_required_for_train():
    df = pd.DataFrame({
        "lead_id": ["1", "2"], "phone": ["9000000001", "9000000002"],
        "created_at": ["2024-01-01", "2024-01-02"], "merchant": ["Evoke", "Evoke"],
    })
    predict_report = canonical.validate_canonical(df, _fields(), purpose="predict")
    train_report = canonical.validate_canonical(df, _fields(), purpose="train")
    assert not any(i.field_name == "disposition" for i in predict_report.issues)
    assert any(i.field_name == "disposition" and i.kind == "missing_required_field" for i in train_report.issues)


def test_validate_canonical_duplicate_lead_id_is_an_error():
    df = pd.DataFrame({
        "lead_id": ["1", "1", "2"], "phone": ["9000000001", "9000000002", "9000000003"],
        "created_at": ["2024-01-01"] * 3, "merchant": ["Evoke"] * 3,
    })
    report = canonical.validate_canonical(df, _fields(), purpose="predict")
    assert any(i.kind == "duplicate_value" and i.field_name == "lead_id" for i in report.issues)
    assert not report.is_valid


def test_validate_canonical_phone_match_rate_below_threshold_is_an_error():
    df = pd.DataFrame({
        "lead_id": [str(i) for i in range(10)],
        "phone": ["9000000001"] + ["bad"] * 9,  # 1/10 = 10% match rate
        "created_at": ["2024-01-01"] * 10, "merchant": ["Evoke"] * 10,
    })
    report = canonical.validate_canonical(df, _fields(), purpose="predict", min_phone_match_rate=0.90)
    assert math.isclose(report.phone_match_rate, 0.1)
    assert report.phone_failed_count == 9
    assert any(i.kind == "low_phone_match_rate" for i in report.issues)
    assert not report.is_valid


def test_validate_canonical_phone_match_rate_above_threshold_passes():
    df = pd.DataFrame({
        "lead_id": [str(i) for i in range(10)],
        "phone": [f"900000000{i}" for i in range(10)],
        "created_at": ["2024-01-01"] * 10, "merchant": ["Evoke"] * 10,
    })
    report = canonical.validate_canonical(df, _fields(), purpose="predict", min_phone_match_rate=0.90)
    assert report.phone_match_rate == 1.0
    assert not any(i.kind == "low_phone_match_rate" for i in report.issues)


def test_validate_canonical_ambiguous_created_at_requires_confirmation():
    df = pd.DataFrame({
        "lead_id": ["1"], "phone": ["9000000001"], "created_at": ["05/06/2024"], "merchant": ["Evoke"],
    })
    unconfirmed = canonical.validate_canonical(df, _fields(), purpose="predict", created_at_ambiguous=True)
    confirmed = canonical.validate_canonical(
        df, _fields(), purpose="predict", created_at_ambiguous=True, created_at_day_first=True,
    )
    assert any(i.kind == "ambiguous_date_needs_confirmation" for i in unconfirmed.issues)
    assert not any(i.kind == "ambiguous_date_needs_confirmation" for i in confirmed.issues)
    assert confirmed.created_at_parsed_count == 1


def test_validate_canonical_reports_disposition_value_counts():
    df = pd.DataFrame({
        "lead_id": ["1", "2", "3"], "phone": ["9000000001", "9000000002", "9000000003"],
        "created_at": ["2024-01-01"] * 3, "merchant": ["Evoke"] * 3,
        "disposition": ["Lead", "Lead", "Converted"],
    })
    report = canonical.validate_canonical(df, _fields(), purpose="train")
    assert dict(report.disposition_value_counts) == {"Lead": 2, "Converted": 1}


def test_validate_canonical_unparsed_created_at_is_a_warning_not_an_error():
    df = pd.DataFrame({
        "lead_id": ["1", "2"], "phone": ["9000000001", "9000000002"],
        "created_at": ["2024-01-01", "not a date"], "merchant": ["Evoke", "Evoke"],
    })
    report = canonical.validate_canonical(df, _fields(), purpose="predict")
    issue = next(i for i in report.issues if i.kind == "unparsed_created_at")
    assert issue.severity == "warning"
    assert report.is_valid  # a warning alone doesn't block
