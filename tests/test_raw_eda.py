"""Tests for app/core/raw_eda.py — every finding maps to a real bug from
CLAUDE.md's hard-won rules. All are report-only; none mutate the input df.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.core import raw_eda


# ---------------------------------------------------------------------------
# Overview / per-column profile
# ---------------------------------------------------------------------------


def test_profile_overview():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [None, None, None], "c": ["x", "y", "y"]})
    ov = raw_eda.profile_overview(df)
    assert ov.row_count == 3
    assert ov.col_count == 3
    assert ov.fully_duplicate_row_count == 0  # every row distinct once column c is considered
    assert ov.fully_empty_columns == ["b"]


def test_profile_overview_detects_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    ov = raw_eda.profile_overview(df)
    assert ov.fully_duplicate_row_count == 1


def test_profile_columns_null_pct_and_distinct():
    df = pd.DataFrame({"col": ["a", "a", "b", None, ""]})
    profiles = raw_eda.profile_columns(df)
    p = profiles[0]
    assert p.null_pct == pytest.approx(40.0)  # 2 of 5 blank
    assert p.distinct_count == 2  # "a", "b"
    assert p.most_frequent[0] == ("a", 2)


def test_categorical_and_numeric_column_filters():
    df = pd.DataFrame({"n": ["1", "2", "3"], "c": ["x", "y", "z"]})
    profiles = raw_eda.profile_columns(df)
    assert raw_eda.numeric_columns(profiles) == ["n"]
    assert raw_eda.categorical_columns(profiles) == ["c"]


# ---------------------------------------------------------------------------
# FINDING — near-duplicate categories
# ---------------------------------------------------------------------------


def test_near_duplicate_case_and_whitespace_variants():
    df = pd.DataFrame({"source": ["FB Ads"] * 5 + ["Fb Ads"] * 3 + ["Weight Loss "] * 2 + ["Weight Loss"] * 4})
    findings = raw_eda.find_near_duplicate_categories(df, ["source"])
    kinds = {f.canonical: f for f in findings}
    assert "FB Ads" in kinds
    assert kinds["FB Ads"].kind == "case_or_whitespace"
    assert dict(kinds["FB Ads"].variants) == {"FB Ads": 5, "Fb Ads": 3}


def test_near_duplicate_fuzzy_spelling_variant():
    df = pd.DataFrame({"city": ["Gurugram"] * 15 + ["Gurgaon"] * 8 + ["Mumbai"] * 50 + ["Delhi"] * 27})
    findings = raw_eda.find_near_duplicate_categories(df, ["city"])
    fuzzy = [f for f in findings if f.kind == "possible_spelling_variant"]
    assert any({v for v, _ in f.variants} == {"Gurugram", "Gurgaon"} for f in fuzzy)
    for f in fuzzy:
        assert f.similarity is not None
        assert f.similarity >= raw_eda.FUZZY_SIMILARITY_THRESHOLD


def test_near_duplicate_skips_free_text_columns():
    # every value distinct -> should not flood fuzzy matches (e.g. "note 9" vs "note 99")
    df = pd.DataFrame({"notes": [f"unique note number {i} about the visit" for i in range(100)]})
    findings = raw_eda.find_near_duplicate_categories(df, ["notes"])
    assert findings == []


def test_near_duplicate_skips_date_like_columns():
    dates = pd.date_range("2024-01-01", periods=50, freq="D").strftime("%d/%m/%Y").tolist()
    df = pd.DataFrame({"created_at": dates})
    findings = raw_eda.find_near_duplicate_categories(df, ["created_at"])
    assert findings == []


def test_near_duplicate_report_only_does_not_mutate_input():
    df = pd.DataFrame({"source": ["FB Ads", "Fb Ads"]})
    before = df.copy()
    raw_eda.find_near_duplicate_categories(df, ["source"])
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# FINDING — sentinel / disguised-code values
# ---------------------------------------------------------------------------


def test_sentinel_literal_zero_detected():
    rng = np.random.default_rng(0)
    values = [0] * 30 + list(rng.integers(300, 850, size=200))
    df = pd.DataFrame({"crif_score": values})
    findings = raw_eda.detect_sentinel_values(df, ["crif_score"])
    literal = [f for f in findings if f.kind == "literal_value" and "0" in f.message.split()[3]]
    assert any(f.count == 30 for f in findings if f.kind == "literal_value")


def test_sentinel_literal_negative_one_detected():
    rng = np.random.default_rng(0)
    values = [-1] * 20 + list(rng.integers(300, 850, size=200))
    df = pd.DataFrame({"equifax_score": values})
    findings = raw_eda.detect_sentinel_values(df, ["equifax_score"])
    assert any(f.kind == "literal_value" and f.count == 20 for f in findings)


def test_sentinel_low_cluster_matches_claude_md_example_shape():
    rng = np.random.default_rng(0)
    # reason codes 10-18 well below real scores 510-885, matching CLAUDE.md's example shape
    low = rng.integers(10, 19, size=23008).tolist()
    high = rng.integers(510, 886, size=70000).tolist()
    df = pd.DataFrame({"crif_score": low + high})
    findings = raw_eda.detect_sentinel_values(df, ["crif_score"])
    cluster = next(f for f in findings if f.kind == "low_cluster")
    assert cluster.count == 23008
    assert "reason codes" in cluster.message
    assert "510" in cluster.message or "511" in cluster.message


def test_sentinel_no_false_positive_on_clean_numeric_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"age": rng.integers(18, 65, size=500)})
    findings = raw_eda.detect_sentinel_values(df, ["age"])
    assert findings == []


def test_sentinel_all_zero_column_not_flagged_as_sentinel():
    # legitimately all-zero column has nothing to contrast against -> not a "masquerading" sentinel
    df = pd.DataFrame({"always_zero": [0] * 100})
    findings = raw_eda.detect_sentinel_values(df, ["always_zero"])
    assert findings == []


# ---------------------------------------------------------------------------
# FINDING — ambiguous date formats
# ---------------------------------------------------------------------------


def test_date_format_day_first_evidence():
    df = pd.DataFrame({"dob": ["25/01/2024", "15/03/2024", "05/06/2024"] * 10})
    finding = raw_eda.detect_date_format(df, "dob")
    assert finding.inference == "day_first"
    assert "25" in finding.evidence or "15" in finding.evidence


def test_date_format_month_first_evidence():
    df = pd.DataFrame({"dob": ["02/28/2024", "11/05/2024", "01/02/2024"] * 10})
    finding = raw_eda.detect_date_format(df, "dob")
    assert finding.inference == "month_first"


def test_date_format_ambiguous_when_no_component_exceeds_12():
    df = pd.DataFrame({"dob": ["01/02/2024", "03/04/2024", "05/06/2024"] * 10})
    finding = raw_eda.detect_date_format(df, "dob")
    assert finding.inference == "ambiguous"
    assert "cannot be determined" in finding.evidence


def test_date_format_mixed_when_both_signals_present():
    df = pd.DataFrame({"dob": ["25/01/2024"] * 10 + ["01/25/2024"] * 10})
    finding = raw_eda.detect_date_format(df, "dob")
    assert finding.inference == "mixed"


def test_date_format_text_month():
    df = pd.DataFrame({"dob": ["30 Apr 2026", "1 Jan 2024", "15 Dec 2023"] * 10})
    finding = raw_eda.detect_date_format(df, "dob")
    assert finding.inference == "text_month"


def test_date_format_none_for_non_date_column():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"] * 10})
    assert raw_eda.detect_date_format(df, "name") is None


def test_date_formats_checked_independently_per_column():
    df = pd.DataFrame(
        {
            "day_first_col": ["25/01/2024"] * 20,
            "month_first_col": ["01/25/2024"] * 20,
            "not_a_date": ["Alice", "Bob"] * 10,
        }
    )
    findings = {f.column: f.inference for f in raw_eda.detect_date_formats(df)}
    assert findings["day_first_col"] == "day_first"
    assert findings["month_first_col"] == "month_first"
    assert "not_a_date" not in findings


# ---------------------------------------------------------------------------
# FINDING — float-formatted identifiers
# ---------------------------------------------------------------------------


def test_float_formatted_identifier_detected():
    df = pd.DataFrame({"phone": [f"{7889358744 + i}.0" for i in range(50)]})
    findings = raw_eda.detect_float_formatted_identifiers(df)
    assert len(findings) == 1
    assert findings[0].column == "phone"
    assert findings[0].match_fraction == 1.0


def test_float_formatted_identifier_not_flagged_for_clean_ids():
    df = pd.DataFrame({"phone": [str(7889358744 + i) for i in range(50)]})
    assert raw_eda.detect_float_formatted_identifiers(df) == []


def test_float_formatted_identifier_ignores_small_decimals():
    # "1.0", "2.0" etc. as small numbers shouldn't be mistaken for corrupted long IDs
    df = pd.DataFrame({"rating": ["1.0", "2.0", "3.0", "4.0", "5.0"] * 10})
    assert raw_eda.detect_float_formatted_identifiers(df) == []


# ---------------------------------------------------------------------------
# FINDING — multi-valued cells
# ---------------------------------------------------------------------------


def test_multi_valued_cells_detects_delimiter_and_order_variance():
    df = pd.DataFrame({"areas": ["Back; Leg"] * 10 + ["Leg; Back"] * 5 + ["Face"] * 20})
    findings = raw_eda.detect_multi_valued_cells(df, ["areas"])
    assert len(findings) == 1
    f = findings[0]
    assert f.delimiter == ";"
    assert f.distinct_component_count == 2
    assert f.order_varies is True


def test_multi_valued_cells_no_order_variance_when_consistent():
    df = pd.DataFrame({"areas": ["Back; Leg"] * 10 + ["Face; Arm"] * 10 + ["Face"] * 20})
    findings = raw_eda.detect_multi_valued_cells(df, ["areas"])
    f = findings[0]
    assert f.order_varies is False


def test_multi_valued_cells_none_for_single_valued_column():
    df = pd.DataFrame({"city": ["Mumbai", "Delhi", "Bangalore"] * 10})
    assert raw_eda.detect_multi_valued_cells(df, ["city"]) == []


# ---------------------------------------------------------------------------
# FINDING — free-text in categorical-looking fields
# ---------------------------------------------------------------------------


def test_free_text_field_detected():
    df = pd.DataFrame({"notes": [f"patient comment number {i}" for i in range(50)]})
    findings = raw_eda.detect_free_text_fields(df, ["notes"])
    assert len(findings) == 1
    assert findings[0].distinct_ratio == 1.0


def test_free_text_field_not_flagged_for_real_categorical():
    df = pd.DataFrame({"source": (["FB Ads", "Google", "Organic"] * 20)})
    assert raw_eda.detect_free_text_fields(df, ["source"]) == []


def test_free_text_field_requires_minimum_rows():
    df = pd.DataFrame({"notes": [f"note {i}" for i in range(5)]})  # below FREE_TEXT_MIN_ROWS
    assert raw_eda.detect_free_text_fields(df, ["notes"]) == []


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def test_numeric_distribution_stats():
    df = pd.DataFrame({"score": list(range(1, 101))})
    nd = raw_eda.numeric_distribution(df, "score")
    assert nd.count == 100
    assert nd.min == 1
    assert nd.max == 100
    assert nd.mean == pytest.approx(50.5)
    assert sum(nd.histogram_counts) == 100


def test_numeric_distribution_none_when_no_numeric_values():
    df = pd.DataFrame({"score": ["x", "y", None]})
    assert raw_eda.numeric_distribution(df, "score") is None


def test_categorical_distribution_long_tail():
    values = ["A"] * 50 + ["B"] * 30 + [f"rare_{i}" for i in range(40)]
    df = pd.DataFrame({"cat": values})
    cd = raw_eda.categorical_distribution(df, "cat", top_n=2)
    assert cd.top_values == [("A", 50), ("B", 30)]
    assert cd.long_tail_count == 40
    assert cd.long_tail_distinct == 40


def test_date_distribution_skips_when_ambiguous():
    df = pd.DataFrame({"dob": ["01/02/2024"] * 20})
    assert raw_eda.date_distribution(df, "dob", day_first=None) is None


def test_date_distribution_builds_when_format_known():
    dates = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%d/%m/%Y").tolist()
    df = pd.DataFrame({"created_at": dates})
    dd = raw_eda.date_distribution(df, "created_at", day_first=True)
    assert dd is not None
    assert dd.min_date == "2024-01-01"
    assert dd.max_date == "2024-01-10"
    assert len(dd.records_per_day) == 10


# ---------------------------------------------------------------------------
# Outcome exploration
# ---------------------------------------------------------------------------


def test_outcome_exploration_ordinal_funnel():
    df = pd.DataFrame({"disposition": ["Lead"] * 70 + ["Engaged"] * 25 + ["Converted"] * 5})
    oc = raw_eda.explore_outcome_column(df, "disposition")
    assert oc.looks_ordinal is True
    assert oc.implied_funnel == ["Lead", "Engaged", "Converted"]
    assert oc.value_counts[0] == ("Lead", 70)


def test_outcome_exploration_not_ordinal_when_too_many_levels():
    df = pd.DataFrame({"category": [f"level_{i}" for i in range(20)] * 5})
    oc = raw_eda.explore_outcome_column(df, "category")
    assert oc.looks_ordinal is False
    assert oc.implied_funnel is None


def test_outcome_exploration_requires_explicit_column_selection():
    # build_profile only explores an outcome when the caller explicitly passes one
    df = pd.DataFrame({"disposition": ["Lead", "Converted"] * 10})
    profile = raw_eda.build_profile(df, raw_dataset_id="raw1")
    assert profile["outcome_exploration"] is None


# ---------------------------------------------------------------------------
# Full profile export
# ---------------------------------------------------------------------------


def test_build_profile_is_json_serializable():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "source": ["FB Ads"] * 40 + ["Fb Ads"] * 10 + ["Google"] * 50,
            "crif_score": [0] * 20 + list(rng.integers(510, 886, size=80)),
            "phone": [f"{9000000000 + i}.0" for i in range(100)],
            "dob": ["15/03/2024"] * 100,
            "disposition": ["Lead"] * 70 + ["Engaged"] * 25 + ["Converted"] * 5,
        }
    )
    profile = raw_eda.build_profile(df, raw_dataset_id="raw-123", outcome_column="disposition")
    json.dumps(profile)  # must not raise

    assert profile["raw_dataset_id"] == "raw-123"
    assert profile["overview"]["row_count"] == 100
    assert len(profile["columns"]) == 5
    assert profile["outcome_exploration"]["column"] == "disposition"
    assert "near_duplicate_categories" in profile["findings"]
    assert "sentinel_values" in profile["findings"]
    assert "date_formats" in profile["findings"]
    assert "float_formatted_identifiers" in profile["findings"]
    assert "multi_valued_cells" in profile["findings"]
    assert "free_text_fields" in profile["findings"]


def test_render_markdown_report_contains_key_sections():
    df = pd.DataFrame({"source": ["FB Ads", "Fb Ads"] * 5, "disposition": ["Lead", "Converted"] * 5})
    profile = raw_eda.build_profile(df, raw_dataset_id="raw-456", outcome_column="disposition")
    report = raw_eda.render_markdown_report(profile)

    assert "raw-456" in report
    assert "## Overview" in report
    assert "## Per-column profile" in report
    assert "### Near-duplicate categories" in report
    assert "### Sentinel / disguised-code values" in report
    assert "### Ambiguous date formats" in report
    assert "### Float-formatted identifiers" in report
    assert "### Multi-valued cells" in report
    assert "### Free-text in categorical-looking fields" in report
    assert "## Outcome exploration" in report


def test_raw_eda_never_mutates_input_dataframe():
    df = pd.DataFrame(
        {
            "source": ["FB Ads", "Fb Ads"] * 5,
            "crif_score": [0, 650] * 5,
            "phone": ["9000000001.0", "9000000002.0"] * 5,
            "areas": ["Back; Leg", "Face"] * 5,
            "disposition": ["Lead", "Converted"] * 5,
        }
    )
    before = df.copy()
    raw_eda.build_profile(df, raw_dataset_id="raw-789", outcome_column="disposition")
    pd.testing.assert_frame_equal(df, before)
