"""Tests for app/core/derive.py — derived feature engineering applied after
assembly. Every function is independently callable, configurable, and never
mutates its input; unparseable/unmapped values stay null rather than being
guessed at or coerced to a default (CLAUDE.md's fail-loudly bias).
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core import derive


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


def test_temporal_features_known_dates():
    # 15/03/2024 is a Friday, 20/01/2024 is a Saturday.
    df = pd.DataFrame({"created_at": ["15/03/2024", "20/01/2024"]})
    out, report = derive.derive_temporal_features(df, day_first=True)
    assert out["created_at_month"].tolist() == [3, 1]
    assert out["created_at_day_of_week"].tolist() == [4, 5]  # Monday=0
    assert out["created_at_is_weekend"].tolist() == [0.0, 1.0]
    assert out["created_at_day_of_month"].tolist() == [15, 20]
    assert out["created_at_week_of_month"].tolist() == [3, 3]


def test_temporal_features_unparsed_rows_are_null_not_guessed():
    df = pd.DataFrame({"created_at": ["15/03/2024", "not a date", None]})
    out, report = derive.derive_temporal_features(df, day_first=True)
    assert report.parsed_count == 1
    assert report.unparsed_count == 1  # "not a date" counted; None (blank) is not
    assert pd.isna(out.loc[1, "created_at_month"])
    assert pd.isna(out.loc[1, "created_at_is_weekend"])  # never silently False
    assert pd.isna(out.loc[2, "created_at_month"])


def test_temporal_features_does_not_mutate_input():
    df = pd.DataFrame({"created_at": ["15/03/2024"]})
    before = df.copy()
    derive.derive_temporal_features(df, day_first=True)
    pd.testing.assert_frame_equal(df, before)


def test_temporal_features_missing_column_raises():
    with pytest.raises(ValueError):
        derive.derive_temporal_features(pd.DataFrame({"x": [1]}), column="created_at")


# ---------------------------------------------------------------------------
# Funnel target derivation
# ---------------------------------------------------------------------------


STAGE_ORDER = ["Lead", "Engaged", "Consulted", "Converted"]
MAPPING = {
    "No Appointment": "Lead", "Appointment Booked": "Engaged",
    "Consulted": "Consulted", "Converted": "Converted",
}


def test_funnel_stages_binary_flags_are_cumulative():
    df = pd.DataFrame({"disposition": ["No Appointment", "Appointment Booked", "Consulted", "Converted"]})
    out, report = derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping=MAPPING)
    # A converted lead reached every stage.
    converted_row = out.iloc[3]
    assert converted_row["reached_Engaged"] == 1.0
    assert converted_row["reached_Consulted"] == 1.0
    assert converted_row["reached_Converted"] == 1.0
    # A lead who only booked an appointment did not reach Consulted/Converted.
    engaged_row = out.iloc[1]
    assert engaged_row["reached_Engaged"] == 1.0
    assert engaged_row["reached_Consulted"] == 0.0
    assert engaged_row["reached_Converted"] == 0.0


def test_funnel_stages_k_levels_produce_k_minus_1_flags():
    df = pd.DataFrame({"disposition": ["Converted"]})
    out, report = derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping=MAPPING)
    reached_cols = [c for c in out.columns if c.startswith("reached_")]
    assert len(reached_cols) == len(STAGE_ORDER) - 1


def test_funnel_stages_unmapped_value_is_reported_not_dropped():
    df = pd.DataFrame({"disposition": ["No Appointment", "Some New Status Nobody Mapped"]})
    out, report = derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping=MAPPING)
    assert report.unmapped_values == [("Some New Status Nobody Mapped", 1)]
    assert pd.isna(out.loc[1, "funnel_stage"])
    assert pd.isna(out.loc[1, "reached_Engaged"])  # unknown, not "did not reach" (0)


def test_funnel_stages_blank_disposition_is_null_not_reported_as_unmapped():
    df = pd.DataFrame({"disposition": [None, ""]})
    out, report = derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping=MAPPING)
    assert report.unmapped_values == []
    assert pd.isna(out.loc[0, "funnel_stage"])


def test_funnel_stages_rejects_mapping_target_outside_stage_order():
    df = pd.DataFrame({"disposition": ["X"]})
    with pytest.raises(ValueError):
        derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping={"X": "NotAStage"})


def test_funnel_stages_counts_per_stage():
    df = pd.DataFrame({"disposition": ["No Appointment", "No Appointment", "Converted"]})
    out, report = derive.derive_funnel_stages(df, stage_order=STAGE_ORDER, mapping=MAPPING)
    assert report.stage_counts == {"Lead": 2, "Engaged": 0, "Consulted": 0, "Converted": 1}


# ---------------------------------------------------------------------------
# Multi-valued categorical normalisation
# ---------------------------------------------------------------------------


def test_multi_valued_order_independent_collapse():
    df = pd.DataFrame({"pain_site": ["Back; Leg", "Leg; Back"]})
    out, report = derive.normalize_multi_valued_column(df, "pain_site")
    assert out["pain_site_normalized"].nunique() == 1
    assert out.loc[0, "pain_site_normalized"] == out.loc[1, "pain_site_normalized"]


def test_multi_valued_component_count():
    df = pd.DataFrame({"pain_site": ["Back; Leg; Neck", "Face", None]})
    out, report = derive.normalize_multi_valued_column(df, "pain_site")
    assert out["pain_site_component_count"].tolist()[:2] == [3, 1]
    assert pd.isna(out.loc[2, "pain_site_component_count"])


def test_multi_valued_binary_flags_per_top_component():
    df = pd.DataFrame({"pain_site": ["Back; Leg", "Face", "Back"]})
    out, report = derive.normalize_multi_valued_column(df, "pain_site", top_n_components=5)
    assert out["pain_site_has_back"].tolist() == [True, False, True]
    assert out["pain_site_has_leg"].tolist() == [True, False, False]


def test_multi_valued_respects_top_n_limit():
    df = pd.DataFrame({"tags": [f"c{i}" for i in range(20)]})  # 20 distinct singleton components
    out, report = derive.normalize_multi_valued_column(df, "tags", top_n_components=3)
    has_cols = [c for c in out.columns if c.startswith("tags_has_")]
    assert len(has_cols) == 3


# ---------------------------------------------------------------------------
# Free-text duration parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_months", [
    ("6yrs 0mon", 72), ("0yrs 6mon", 6), ("2 years 3 months", 27),
    ("1yr", 12), ("6mon", 6),
])
def test_parse_duration_to_months(text, expected_months):
    assert derive.parse_duration_to_months(text) == expected_months


def test_parse_duration_bare_number_is_ambiguous_never_guessed():
    assert derive.parse_duration_to_months("5") is None


def test_parse_duration_garbage_and_blank_return_none():
    assert derive.parse_duration_to_months("garbage") is None
    assert derive.parse_duration_to_months(None) is None
    assert derive.parse_duration_to_months("") is None


def test_band_duration_months_boundaries():
    assert derive.band_duration_months(0) == "0-6"
    assert derive.band_duration_months(5) == "0-6"
    assert derive.band_duration_months(6) == "6-12"
    assert derive.band_duration_months(72) == "60+"


def test_band_duration_months_none_stays_none():
    assert derive.band_duration_months(None) is None


def test_derive_duration_features_end_to_end_null_never_banded():
    # Regression: a NaN (not None) months value must not fall through to the
    # final "60+" bucket by accident (band_duration_months must check for
    # NaN, not just `is None` -- pandas upcasts None to NaN once mixed with
    # real numbers in a Series).
    df = pd.DataFrame({"history": ["6yrs 0mon", "garbage", None]})
    out, report = derive.derive_duration_features(df, "history")
    assert out.loc[0, "history_band"] == "60+"
    assert pd.isna(out.loc[1, "history_band"])
    assert pd.isna(out.loc[2, "history_band"])
    assert report.parsed_count == 1
    assert report.unparsed_count == 1


# ---------------------------------------------------------------------------
# Near-duplicate category consolidation
# ---------------------------------------------------------------------------


def test_suggest_category_merges_case_whitespace_default_confirmed():
    df = pd.DataFrame({"city": ["Gurugram"] * 5 + ["gurugram "] * 3})
    suggestions = derive.suggest_category_merges(df, ["city"])
    assert len(suggestions) == 1
    assert suggestions[0].kind == "case_or_whitespace"
    assert suggestions[0].default_confirmed is True


def test_suggest_category_merges_fuzzy_not_default_confirmed():
    df = pd.DataFrame({"city": ["Gurugram"] * 5 + ["Gurgaon"] * 3})
    suggestions = derive.suggest_category_merges(df, ["city"])
    assert any(s.kind == "possible_spelling_variant" and not s.default_confirmed for s in suggestions)


def test_apply_category_merges_only_applies_confirmed():
    df = pd.DataFrame({"city": ["Gurugram", "gurugram ", "Gurgaon"]})
    all_suggestions = derive.suggest_category_merges(df, ["city"])
    confirmed = [s for s in all_suggestions if s.default_confirmed]
    out = derive.apply_category_merges(df, confirmed)
    assert out["city"].tolist() == ["Gurugram", "Gurugram", "Gurgaon"]  # only the whitespace variant merged


def test_apply_category_merges_empty_list_is_a_no_op():
    df = pd.DataFrame({"city": ["Gurugram", "Gurgaon"]})
    out = derive.apply_category_merges(df, [])
    pd.testing.assert_frame_equal(out, df)


def test_apply_category_merges_does_not_mutate_input():
    df = pd.DataFrame({"city": ["Gurugram", "gurugram "]})
    before = df.copy()
    suggestions = derive.suggest_category_merges(df, ["city"])
    derive.apply_category_merges(df, suggestions)
    pd.testing.assert_frame_equal(df, before)
