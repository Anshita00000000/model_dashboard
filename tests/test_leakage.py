from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import leakage


# ---------------------------------------------------------------------------
# GATE 1 — target leakage
# ---------------------------------------------------------------------------


def test_gate_target_leakage_passes_when_clean():
    result = leakage.gate_target_leakage(["converted"], ["lead_source", "city"])
    assert result.passed


def test_gate_target_leakage_raises_when_target_in_features():
    with pytest.raises(leakage.LeakageGateFailure) as exc_info:
        leakage.gate_target_leakage(["converted"], ["lead_source", "converted"])
    assert exc_info.value.gate == "target_leakage"
    assert "converted" in exc_info.value.details["offending_columns"]


def test_gate_target_leakage_raises_for_target_derived_columns():
    with pytest.raises(leakage.LeakageGateFailure):
        leakage.gate_target_leakage(["converted"], ["converted_flag"], target_derived_columns=["converted_flag"])


# ---------------------------------------------------------------------------
# GATE 2 — implausible performance
# ---------------------------------------------------------------------------


def test_gate_implausible_performance_blocks_by_default():
    with pytest.raises(leakage.LeakageGateFailure) as exc_info:
        leakage.gate_implausible_performance(roc_auc_value=0.99, base_rate=0.03)
    assert exc_info.value.gate == "implausible_performance"


def test_gate_implausible_performance_allows_override():
    result = leakage.gate_implausible_performance(roc_auc_value=0.99, base_rate=0.03, allow_override=True)
    assert result.passed is False
    assert result.details["overridden"] is True


def test_gate_implausible_performance_passes_realistic_auc():
    result = leakage.gate_implausible_performance(roc_auc_value=0.72, base_rate=0.03)
    assert result.passed is True


def test_gate_implausible_performance_high_auc_on_common_event_is_fine():
    # not a rare event (base rate 40%) -> high AUC isn't flagged
    result = leakage.gate_implausible_performance(roc_auc_value=0.95, base_rate=0.40)
    assert result.passed is True


# ---------------------------------------------------------------------------
# GATE 3 — provenance detection
# ---------------------------------------------------------------------------


def test_gate_provenance_detection_returns_none_when_not_configured():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert leakage.gate_provenance_detection(df, ["x"], None) is None


def test_gate_provenance_detection_flags_high_auc():
    rng = np.random.default_rng(0)
    n = 400
    source = rng.choice(["batch_a", "batch_b"], size=n)
    # a feature that near-perfectly encodes source
    leaky_feature = np.where(source == "batch_a", rng.normal(0, 0.1, n), rng.normal(10, 0.1, n))
    df = pd.DataFrame({"leaky_feature": leaky_feature, "source": source})

    result = leakage.gate_provenance_detection(df, ["leaky_feature"], "source")
    assert result is not None
    assert result.passed is False
    assert result.details["auc"] > leakage.PROVENANCE_AUC_THRESHOLD
    assert result.details["top_features_by_gain"][0][0] == "leaky_feature"


def test_gate_provenance_detection_passes_when_no_signal():
    rng = np.random.default_rng(0)
    n = 400
    source = rng.choice(["batch_a", "batch_b"], size=n)
    unrelated_feature = rng.normal(size=n)  # independent of source
    df = pd.DataFrame({"unrelated_feature": unrelated_feature, "source": source})

    result = leakage.gate_provenance_detection(df, ["unrelated_feature"], "source")
    assert result is not None
    assert result.passed is True


def test_gate_provenance_detection_single_source_value_is_trivially_fine():
    df = pd.DataFrame({"x": [1, 2, 3], "source": ["only_one"] * 3})
    result = leakage.gate_provenance_detection(df, ["x"], "source")
    assert result.passed is True


# ---------------------------------------------------------------------------
# GATE 4 — post-event features
# ---------------------------------------------------------------------------


def test_gate_post_event_features_flags_known_patterns():
    result = leakage.gate_post_event_features(["lead_source", "turnaround_time_days", "procedure_performed"])
    assert result.passed is False
    assert "turnaround_time_days" in result.details["offending_columns"]
    assert "procedure_performed" in result.details["offending_columns"]


def test_gate_post_event_features_passes_when_clean():
    result = leakage.gate_post_event_features(["lead_source", "city", "crif_score"])
    assert result.passed is True


# ---------------------------------------------------------------------------
# GATE 5 — formula consistency
# ---------------------------------------------------------------------------


def test_make_sum_invariant_and_gate_reports_overall_pass_rate():
    df = pd.DataFrame({"total": [10, 20, 999], "part_a": [4, 10, 5], "part_b": [6, 10, 5]})
    inv = leakage.make_sum_invariant("total_check", "total", ["part_a", "part_b"])
    results = leakage.gate_formula_consistency(df, [inv], source_column=None)
    assert len(results) == 1
    assert results[0].details["overall_pass_rate"] == pytest.approx(2 / 3)
    assert results[0].passed is False  # below default min_pass_rate=0.95


def test_gate_formula_consistency_flags_per_source_spread():
    # batch A always satisfies total == sum(parts); batch B never does (different formula)
    df = pd.DataFrame(
        {
            "total": [10, 10, 10, 999, 999, 999],
            "part_a": [4, 4, 4, 4, 4, 4],
            "part_b": [6, 6, 6, 6, 4, 4],
            "source": ["A", "A", "A", "B", "B", "B"],
        }
    )
    inv = leakage.make_sum_invariant("total_check", "total", ["part_a", "part_b"])
    results = leakage.gate_formula_consistency(df, [inv], source_column="source")
    assert len(results) == 1
    result = results[0]
    assert result.passed is False
    assert result.details["per_source_pass_rate"]["A"] == pytest.approx(1.0)
    assert result.details["per_source_pass_rate"]["B"] == pytest.approx(0.0)
    assert "spread" in result.message


def test_make_le_invariant():
    df = pd.DataFrame({"active": [1, 5, 10], "total": [10, 5, 8]})
    inv = leakage.make_le_invariant("active_le_total", "active", "total")
    results = leakage.gate_formula_consistency(df, [inv], source_column=None)
    # 2 of 3 rows satisfy active <= total (row 2: 5<=5 ok, row 3: 10<=8 fails)
    assert results[0].details["overall_pass_rate"] == pytest.approx(2 / 3)


def test_gate_formula_consistency_reports_missing_columns():
    df = pd.DataFrame({"total": [1, 2]})
    inv = leakage.make_sum_invariant("bad", "total", ["does_not_exist"])
    results = leakage.gate_formula_consistency(df, [inv], source_column=None)
    assert results[0].passed is False
    assert "does_not_exist" in results[0].details["missing_columns"]
