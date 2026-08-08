"""Evaluation: benchmark metrics, tier tables, lift, capture thresholds, feature gain.

All metrics are implemented from scratch on numpy/pandas (no scikit-learn
dependency, consistent with the rest of app/core/).

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_N_TILES = 10
DEFAULT_CAPTURE_TARGETS: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75)
LOW_BOTTOM_TIER_POSITIVES_THRESHOLD = 20
DEFAULT_BOOTSTRAP_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-aware. NaN if only one class present."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[y_true == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision: sum over thresholds of (recall_n - recall_{n-1}) * precision_n."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted)
    n_seen = np.arange(1, len(y_sorted) + 1)
    precision = tp_cum / n_seen
    recall = tp_cum / n_pos

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    delta_recall = recall - recall_prev
    return float(np.sum(delta_recall * precision))


def log_loss(y_true: np.ndarray, y_score: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


@dataclass(frozen=True)
class BenchmarkMetrics:
    stage: str
    n_rows: int
    n_positive: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier: float

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "base_rate": self.base_rate,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "log_loss": self.log_loss,
            "brier": self.brier,
        }


def benchmark_metrics(stage_name: str, y_true: np.ndarray, y_score: np.ndarray) -> BenchmarkMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    return BenchmarkMetrics(
        stage=stage_name,
        n_rows=n,
        n_positive=n_pos,
        base_rate=(n_pos / n) if n else float("nan"),
        roc_auc=roc_auc(y_true, y_score),
        pr_auc=pr_auc(y_true, y_score),
        log_loss=log_loss(y_true, y_score) if n else float("nan"),
        brier=brier_score(y_true, y_score) if n else float("nan"),
    )


# ---------------------------------------------------------------------------
# Tier table — Tier 1 = HIGHEST predicted probability
# ---------------------------------------------------------------------------


def _tile_boundaries(n: int, n_tiles: int) -> np.ndarray:
    return np.linspace(0, n, n_tiles + 1).astype(int)


def tier_table(y_true: np.ndarray, y_score: np.ndarray, n_tiles: int = DEFAULT_N_TILES) -> pd.DataFrame:
    """Columns: tier, n, actual_rate, pct_of_positives_captured, avg_predicted,
    calibration_gap_pp (predicted - actual, in percentage points). Tier 1 is
    the highest-predicted-probability slice.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return pd.DataFrame(
            columns=["tier", "n", "actual_rate", "pct_of_positives_captured", "avg_predicted", "calibration_gap_pp"]
        )

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    boundaries = _tile_boundaries(n, n_tiles)
    total_positives = y_true.sum()

    rows = []
    for tier in range(n_tiles):
        start, end = boundaries[tier], boundaries[tier + 1]
        tier_y = y_sorted[start:end]
        tier_score = score_sorted[start:end]
        n_tier = len(tier_y)
        n_pos_tier = float(tier_y.sum())
        actual_rate = (n_pos_tier / n_tier) if n_tier else float("nan")
        pct_captured = (100 * n_pos_tier / total_positives) if total_positives else float("nan")
        avg_pred = float(tier_score.mean()) if n_tier else float("nan")
        calibration_gap_pp = (100 * (avg_pred - actual_rate)) if n_tier else float("nan")
        rows.append(
            {
                "tier": tier + 1,
                "n": n_tier,
                "actual_rate": actual_rate,
                "pct_of_positives_captured": pct_captured,
                "avg_predicted": avg_pred,
                "calibration_gap_pp": calibration_gap_pp,
            }
        )
    return pd.DataFrame(rows)


def tier_cutoffs(y_score: np.ndarray, n_tiles: int = DEFAULT_N_TILES) -> list[float]:
    """Score thresholds separating tiers, computed once (normally against the
    held-out evaluation set) and then pinned for reuse at predict time — see
    app/core/predict.py. Re-ranking each future batch independently would put
    the same lead in a different tier depending on that batch's composition,
    which makes tier assignments non-comparable across prediction runs.

    Returns n_tiles-1 boundaries, descending. Tier 1 (highest scores) is
    score >= cutoffs[0]; tier k is cutoffs[k-1] <= score < cutoffs[k-2].
    """
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    if n == 0:
        return []
    scores_desc = np.sort(y_score)[::-1]
    boundaries = _tile_boundaries(n, n_tiles)
    return [float(scores_desc[boundaries[tier]]) for tier in range(1, n_tiles) if boundaries[tier] < n]


def assign_tiers(scores: np.ndarray, cutoffs: list[float]) -> np.ndarray:
    """Assign each score a 1-indexed tier using PINNED cutoffs (as returned by
    tier_cutoffs), not by re-ranking `scores` against itself. Tier 1 = highest.
    """
    scores = np.asarray(scores, dtype=float)
    if not cutoffs:
        return np.full(len(scores), 1, dtype=int)
    ascending_cutoffs = np.sort(np.asarray(cutoffs, dtype=float))
    n_tiles = len(cutoffs) + 1
    return n_tiles - np.searchsorted(ascending_cutoffs, scores, side="right")


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiftResult:
    lift_vs_base_rate: float  # headline: top-tier rate / overall rate. Stable, always defined.
    top_bottom_ratio: Any  # float, or the string "bottom tier has zero positives"
    top_bottom_ratio_ci: Optional[tuple[float, float]] = None  # bootstrap 95% CI when bottom tier has < 20 positives


def _bootstrap_ratio_ci(
    top_y: np.ndarray, bottom_y: np.ndarray, random_seed: int, n_bootstrap: int
) -> Optional[tuple[float, float]]:
    n_top, n_bottom = len(top_y), len(bottom_y)
    if n_top == 0 or n_bottom == 0:
        return None
    rng = np.random.default_rng(random_seed)
    ratios = []
    for _ in range(n_bootstrap):
        top_sample = top_y[rng.integers(0, n_top, n_top)]
        bottom_sample = bottom_y[rng.integers(0, n_bottom, n_bottom)]
        bottom_rate = bottom_sample.mean()
        if bottom_rate == 0:
            continue
        ratios.append(top_sample.mean() / bottom_rate)
    if not ratios:
        return None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return (float(lo), float(hi))


def compute_lift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_tiles: int = DEFAULT_N_TILES,
    random_seed: int = 42,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> LiftResult:
    """Lift is computed from UNROUNDED rates, not from the (rounded, for display)
    tier_table.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return LiftResult(lift_vs_base_rate=float("nan"), top_bottom_ratio="bottom tier has zero positives")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    boundaries = _tile_boundaries(n, n_tiles)

    top_y = y_sorted[boundaries[0] : boundaries[1]]
    bottom_y = y_sorted[boundaries[-2] : boundaries[-1]]

    overall_rate = y_true.mean()
    top_rate = top_y.mean() if len(top_y) else float("nan")
    bottom_rate = bottom_y.mean() if len(bottom_y) else float("nan")
    bottom_positives = int(bottom_y.sum())

    lift_vs_base = (top_rate / overall_rate) if overall_rate else float("nan")

    ratio: Any
    ci: Optional[tuple[float, float]] = None
    if bottom_positives == 0:
        ratio = "bottom tier has zero positives"
    else:
        ratio = float(top_rate / bottom_rate)
        if bottom_positives < LOW_BOTTOM_TIER_POSITIVES_THRESHOLD:
            ci = _bootstrap_ratio_ci(top_y, bottom_y, random_seed, n_bootstrap)

    return LiftResult(lift_vs_base_rate=float(lift_vs_base), top_bottom_ratio=ratio, top_bottom_ratio_ci=ci)


# ---------------------------------------------------------------------------
# Capture thresholds
# ---------------------------------------------------------------------------


def capture_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, targets: tuple[float, ...] = DEFAULT_CAPTURE_TARGETS
) -> dict[float, float]:
    """% of population (0-100), ranked by descending score, needed to capture
    at least `target` fraction of all positives.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    total_positives = y_true.sum()
    if n == 0 or total_positives == 0:
        return {t: float("nan") for t in targets}

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    cum_positives = np.cumsum(y_sorted)
    cum_capture_frac = cum_positives / total_positives

    result = {}
    for t in targets:
        idx = np.searchsorted(cum_capture_frac, t, side="left")
        idx = min(idx, n - 1)
        result[t] = float(100 * (idx + 1) / n)
    return result


# ---------------------------------------------------------------------------
# Feature gain
# ---------------------------------------------------------------------------


def feature_gain_table(stage_label: str, booster: lgb.Booster, feature_sources: dict[str, str]) -> pd.DataFrame:
    """Per feature: gain, gain share %, and a core/enrichment source tag."""
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    total = float(np.sum(gains))

    rows = [
        {
            "stage": stage_label,
            "feature": name,
            "gain": float(gain),
            "gain_share_pct": (100 * float(gain) / total) if total else 0.0,
            "source": feature_sources.get(name, "core"),
        }
        for name, gain in zip(names, gains)
    ]
    return pd.DataFrame(rows, columns=["stage", "feature", "gain", "gain_share_pct", "source"]).sort_values(
        "gain", ascending=False
    ).reset_index(drop=True)
