from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from app.core import evaluate


def test_roc_auc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert evaluate.roc_auc(y, score) == pytest.approx(1.0)


def test_roc_auc_random_is_about_half():
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.random(n) < 0.5).astype(int)
    score = rng.random(n)  # independent of y
    auc = evaluate.roc_auc(y, score)
    assert 0.47 < auc < 0.53


def test_roc_auc_single_class_is_nan():
    y = np.array([1, 1, 1])
    score = np.array([0.1, 0.5, 0.9])
    assert np.isnan(evaluate.roc_auc(y, score))


def test_pr_auc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert evaluate.pr_auc(y, score) == pytest.approx(1.0)


def test_log_loss_matches_hand_computation():
    y = np.array([1, 0])
    score = np.array([0.8, 0.2])
    expected = -(np.log(0.8) + np.log(0.8)) / 2
    assert evaluate.log_loss(y, score) == pytest.approx(expected)


def test_brier_score_matches_hand_computation():
    y = np.array([1, 0])
    score = np.array([0.8, 0.3])
    expected = ((0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 2
    assert evaluate.brier_score(y, score) == pytest.approx(expected)


def test_benchmark_metrics_basic_shape():
    y = np.array([0, 1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    m = evaluate.benchmark_metrics("stage_1", y, score)
    assert m.stage == "stage_1"
    assert m.n_rows == 5
    assert m.n_positive == 2
    assert m.base_rate == pytest.approx(0.4)
    assert m.roc_auc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tier table
# ---------------------------------------------------------------------------


def test_tier_table_tier_1_is_highest_score():
    n = 100
    y = np.array([1] * 10 + [0] * 90)
    score = np.linspace(1.0, 0.0, n)  # highest score first, perfectly separating
    table = evaluate.tier_table(y, score, n_tiles=10)
    assert table.loc[0, "tier"] == 1
    assert table.loc[0, "actual_rate"] == pytest.approx(1.0)
    assert table.loc[0, "pct_of_positives_captured"] == pytest.approx(100.0)
    assert table.loc[9, "actual_rate"] == pytest.approx(0.0)


def test_tier_table_n_rows_sums_to_total():
    rng = np.random.default_rng(0)
    n = 997  # not evenly divisible by 10
    y = (rng.random(n) < 0.1).astype(int)
    score = rng.random(n)
    table = evaluate.tier_table(y, score, n_tiles=10)
    assert table["n"].sum() == n
    assert len(table) == 10


def test_tier_table_calibration_gap_sign():
    # overconfident: predicted always higher than actual
    y = np.array([0, 0, 0, 0, 1])
    score = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    table = evaluate.tier_table(y, score, n_tiles=1)
    assert table.loc[0, "calibration_gap_pp"] > 0


# ---------------------------------------------------------------------------
# Tier cutoffs — pinned, reused at predict time instead of re-ranking a batch
# ---------------------------------------------------------------------------


def test_tier_cutoffs_length_and_descending():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    cutoffs = evaluate.tier_cutoffs(scores, n_tiles=10)
    assert len(cutoffs) == 9
    assert cutoffs == sorted(cutoffs, reverse=True)


def test_assign_tiers_matches_tier_table_on_same_population():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.1).astype(int)
    score = rng.random(n)

    cutoffs = evaluate.tier_cutoffs(score, n_tiles=10)
    tiers = evaluate.assign_tiers(score, cutoffs)

    assert tiers.min() >= 1
    assert tiers.max() <= 10
    # tier 1 should be (approximately) the top 10% of scores
    top_tier_scores = score[tiers == 1]
    other_scores = score[tiers != 1]
    assert top_tier_scores.min() >= other_scores.max() - 1e-9


def test_assign_tiers_stable_across_batch_composition():
    """The whole point of pinned cutoffs: the same score lands in the same tier
    regardless of what else is in the batch being scored."""
    cutoffs = [0.9, 0.7, 0.5, 0.3, 0.1]
    score = np.array([0.95])
    tier_in_small_batch = evaluate.assign_tiers(score, cutoffs)[0]
    # same score, now surrounded by a very different batch composition
    big_batch = np.concatenate([score, np.full(500, 0.99)])
    tier_in_big_batch = evaluate.assign_tiers(big_batch, cutoffs)[0]
    assert tier_in_small_batch == tier_in_big_batch


def test_tier_cutoffs_empty_input():
    assert evaluate.tier_cutoffs(np.array([]), n_tiles=10) == []


def test_assign_tiers_no_cutoffs_returns_all_tier_1():
    tiers = evaluate.assign_tiers(np.array([0.1, 0.9]), [])
    assert list(tiers) == [1, 1]


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


def test_compute_lift_stable_metric_always_defined():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    score = rng.random(n)
    lift = evaluate.compute_lift(y, score, n_tiles=10)
    assert not np.isnan(lift.lift_vs_base_rate)


def test_compute_lift_bottom_tier_zero_positives_returns_string_not_inf():
    n = 100
    y = np.zeros(n, dtype=int)
    y[:5] = 1  # only in the top scores
    score = np.linspace(1.0, 0.0, n)
    lift = evaluate.compute_lift(y, score, n_tiles=10)
    assert lift.top_bottom_ratio == "bottom tier has zero positives"


def test_compute_lift_small_bottom_tier_gets_ci():
    rng = np.random.default_rng(3)
    n = 500
    # construct scores/labels so the bottom tier has a handful (but not zero) positives
    score = np.concatenate([np.full(50, 0.9), np.full(450, 0.1)])
    y = np.zeros(n, dtype=int)
    y[:20] = 1  # positives concentrated in top tier
    y[480:485] = 1  # a few positives land in the bottom tier (< 20)
    lift = evaluate.compute_lift(y, score, n_tiles=10, random_seed=1, n_bootstrap=200)
    assert isinstance(lift.top_bottom_ratio, float)
    assert lift.top_bottom_ratio_ci is not None
    lo, hi = lift.top_bottom_ratio_ci
    assert lo <= hi


def test_compute_lift_uses_unrounded_rates():
    # a case where rounding at display precision would change the ratio
    n = 300
    score = np.concatenate([np.full(30, 0.99), np.full(270, 0.01)])
    y = np.zeros(n, dtype=int)
    y[:29] = 1  # top tier rate = 29/30, not a round number
    y[290:293] = 1  # bottom tier: 3/30
    lift = evaluate.compute_lift(y, score, n_tiles=10)
    expected_top_rate = 29 / 30
    expected_bottom_rate = 3 / 30
    assert lift.top_bottom_ratio == pytest.approx(expected_top_rate / expected_bottom_rate)


# ---------------------------------------------------------------------------
# Capture thresholds
# ---------------------------------------------------------------------------


def test_capture_thresholds_perfect_model_needs_exactly_base_rate_population():
    n = 1000
    y = np.array([1] * 100 + [0] * 900)
    score = np.linspace(1.0, 0.0, n)  # perfectly ranks positives first
    thresholds = evaluate.capture_thresholds(y, score, targets=(0.60, 0.70))
    # to capture 60% of 100 positives = 60 positives, need exactly the top 60 rows = 6% of population
    assert thresholds[0.60] == pytest.approx(6.0, abs=0.2)
    assert thresholds[0.70] == pytest.approx(7.0, abs=0.2)


def test_capture_thresholds_no_positives_returns_nan():
    y = np.zeros(50, dtype=int)
    score = np.random.default_rng(0).random(50)
    thresholds = evaluate.capture_thresholds(y, score, targets=(0.6,))
    assert np.isnan(thresholds[0.6])


# ---------------------------------------------------------------------------
# Feature gain
# ---------------------------------------------------------------------------


def test_feature_gain_table_reports_source_and_shares_sum_to_100():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "core_feat": rng.normal(size=n),
            "crif_score": rng.normal(size=n),
        }
    )
    y = (df["core_feat"] + df["crif_score"] > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5},
        lgb.Dataset(df, label=y),
        num_boost_round=20,
    )
    sources = {"core_feat": "core", "crif_score": "enrichment"}
    table = evaluate.feature_gain_table("stage_1", booster, sources)

    assert set(table["feature"]) == {"core_feat", "crif_score"}
    assert set(table["source"]) == {"core", "enrichment"}
    assert table["gain_share_pct"].sum() == pytest.approx(100.0, abs=0.01)
    # sorted descending by gain
    assert list(table["gain"]) == sorted(table["gain"], reverse=True)
