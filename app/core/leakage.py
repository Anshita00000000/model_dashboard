"""Leakage gates. Run automatically, block on failure.

Every gate here exists because it caught a real bug in this project — see
CLAUDE.md, "Leakage — the dominant risk in this domain". Only GATE 1 (hard
assert) and GATE 2 (halt + explicit override) actually block; GATE 3/4/5 are
advisory and always return a report for the caller to display.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import evaluate

RARE_EVENT_BASE_RATE = 0.10
IMPLAUSIBLE_AUC_THRESHOLD = 0.90
PROVENANCE_AUC_THRESHOLD = 0.70

# Post-event fields: only knowable AFTER the outcome (turnaround time,
# procedure/treatment-performed, status restatements). Configurable blocklist.
DEFAULT_POST_EVENT_PATTERNS: tuple[str, ...] = (
    r"turnaround",
    r"\btat\b",
    r"_tat\b",
    r"procedure.*perform",
    r"treatment.*perform",
    r"status.*restat",
    r"final_status",
    r"closure",
    r"resolved_at",
    r"completed_at",
    r"disbursed_at",
)


class LeakageGateFailure(Exception):
    """Raised by a blocking gate. The caller may catch this and, only with
    explicit user consent, re-run with an override flag set.
    """

    def __init__(self, gate: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.gate = gate
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GATE 1 — target leakage (hard assert, always blocks)
# ---------------------------------------------------------------------------


def gate_target_leakage(
    target_columns: list[str],
    feature_columns: list[str],
    target_derived_columns: Optional[list[str]] = None,
) -> GateResult:
    """Assert no target column, and no known target-derived column, appears in
    the feature list. (A target column was once accidentally left in the
    feature set; it produced a perfect 1.0000 AUC.)
    """
    banned = set(target_columns) | set(target_derived_columns or [])
    hits = [c for c in feature_columns if c in banned]
    if hits:
        raise LeakageGateFailure(
            "target_leakage",
            f"target (or target-derived) column(s) found in the feature list: {hits}",
            details={"offending_columns": hits},
        )
    return GateResult("target_leakage", True, "no target or target-derived column present in the feature list")


# ---------------------------------------------------------------------------
# GATE 2 — implausible performance (blocks unless explicitly overridden)
# ---------------------------------------------------------------------------


def gate_implausible_performance(
    roc_auc_value: float,
    base_rate: float,
    *,
    allow_override: bool = False,
    rare_event_threshold: float = RARE_EVENT_BASE_RATE,
    auc_threshold: float = IMPLAUSIBLE_AUC_THRESHOLD,
) -> GateResult:
    """If ROC-AUC > 0.90 on a rare-event target (base rate < 10%), halt and
    report suspected leakage. Realistic performance on this kind of problem
    is 0.65-0.80 (see CLAUDE.md) — anything above 0.90 here is almost always
    leakage, not skill.
    """
    is_rare_event = base_rate < rare_event_threshold
    suspicious = bool(is_rare_event and roc_auc_value > auc_threshold)

    if suspicious and not allow_override:
        raise LeakageGateFailure(
            "implausible_performance",
            f"ROC-AUC {roc_auc_value:.4f} on a rare-event target (base rate {base_rate:.2%}) exceeds "
            f"the {auc_threshold:.2f} plausibility ceiling — this is almost always leakage, not skill. "
            f"Re-run with allow_override=True only after you've specifically ruled out leakage.",
            details={"roc_auc": roc_auc_value, "base_rate": base_rate},
        )

    if suspicious:
        message = f"ROC-AUC {roc_auc_value:.4f} exceeds the plausibility ceiling but was explicitly overridden"
    else:
        message = f"ROC-AUC {roc_auc_value:.4f} is within the plausible range for base rate {base_rate:.2%}"

    return GateResult(
        "implausible_performance",
        not suspicious,
        message,
        details={"roc_auc": roc_auc_value, "base_rate": base_rate, "overridden": suspicious and allow_override},
    )


# ---------------------------------------------------------------------------
# GATE 3 — provenance detection (report only)
# ---------------------------------------------------------------------------


def gate_provenance_detection(
    df: pd.DataFrame,
    feature_columns: list[str],
    provenance_column: Optional[str],
    *,
    auc_threshold: float = PROVENANCE_AUC_THRESHOLD,
    random_seed: int = 42,
) -> Optional[GateResult]:
    """Train a quick model to predict the provenance column from the feature
    set alone, evaluated OUT OF SAMPLE (a held-out random slice) — an
    in-sample fit overfits noise readily enough with a handful of features
    that it produces false positives here. AUC > 0.70 means the features
    encode data provenance — if provenance also correlates with the target,
    the model can identify outcomes without learning anything real. Returns
    None if no provenance_column is configured (nothing to check).
    """
    if not provenance_column:
        return None
    if provenance_column not in df.columns:
        raise ValueError(f"provenance_column {provenance_column!r} not found in dataframe")
    if not feature_columns:
        raise ValueError("provenance detection needs a non-empty feature_columns list")

    y = df[provenance_column]
    distinct = y.dropna().unique()
    if len(distinct) < 2:
        return GateResult(
            "provenance_detection", True, "provenance column has fewer than 2 distinct values; nothing to detect"
        )

    if len(distinct) == 2:
        positive_label = distinct[0]
    else:
        positive_label = y.value_counts().idxmax()
    y_bin = (y == positive_label).astype(int).to_numpy()

    X = df[feature_columns].copy()
    cat_cols = [c for c in feature_columns if str(X[c].dtype) in ("object", "category")]
    for c in cat_cols:
        X[c] = X[c].astype("category")

    rng = np.random.default_rng(random_seed)
    n = len(df)
    order = rng.permutation(n)
    split = int(n * 0.7)
    train_idx, test_idx = order[:split], order[split:]

    if len(test_idx) == 0 or len(np.unique(y_bin[train_idx])) < 2:
        return GateResult(
            "provenance_detection", True, "not enough data for a reliable out-of-sample provenance check"
        )

    dataset = lgb.Dataset(X.iloc[train_idx], label=y_bin[train_idx], categorical_feature=cat_cols or "auto", free_raw_data=False)
    params = {"objective": "binary", "verbosity": -1, "seed": random_seed, "min_data_in_leaf": 20, "num_leaves": 15}
    booster = lgb.train(params, dataset, num_boost_round=50)
    preds = booster.predict(X.iloc[test_idx])
    auc = evaluate.roc_auc(y_bin[test_idx], preds)

    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    top_features = sorted(
        ((n, float(g)) for n, g in zip(names, gains) if g > 0), key=lambda t: -t[1]
    )[:10]

    if np.isnan(auc):
        # degenerate fit (e.g. every row landed in one leaf); nothing to flag
        passed = True
        message = f"could not fit a provenance-detection model for {provenance_column!r}"
    else:
        passed = bool(auc <= auc_threshold)
        message = f"features predict {provenance_column!r} with AUC {auc:.4f} (threshold {auc_threshold:.2f})"
        message += " — no meaningful provenance signal" if passed else " — features encode data provenance; check for a leakage pathway"

    return GateResult(
        "provenance_detection",
        passed,
        message,
        details={"auc": auc, "provenance_column": provenance_column, "top_features_by_gain": top_features},
    )


# ---------------------------------------------------------------------------
# GATE 4 — post-event features (warn only)
# ---------------------------------------------------------------------------


def gate_post_event_features(
    feature_columns: list[str], patterns: tuple[str, ...] = DEFAULT_POST_EVENT_PATTERNS
) -> GateResult:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    hits = [c for c in feature_columns if any(p.search(c) for p in compiled)]
    passed = not hits
    message = (
        "no feature names match known post-event patterns"
        if passed
        else f"{len(hits)} feature(s) look post-event (only knowable after the outcome): {hits}"
    )
    return GateResult("post_event_features", passed, message, details={"offending_columns": hits})


# ---------------------------------------------------------------------------
# GATE 5 — formula consistency across sources (report only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulaInvariant:
    name: str
    check: str  # human description, e.g. "total == sum(parts)"
    columns: tuple[str, ...]
    evaluator: Callable[[pd.DataFrame], pd.Series]  # -> boolean Series, True = invariant holds for that row


def make_sum_invariant(name: str, total_col: str, part_cols: list[str], tolerance: float = 1e-6) -> FormulaInvariant:
    """total_col == sum(part_cols), for rows where total_col is present."""

    def _check(df: pd.DataFrame) -> pd.Series:
        total = pd.to_numeric(df[total_col], errors="coerce")
        parts_sum = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in part_cols)
        return total.isna() | (total - parts_sum).abs().le(tolerance)

    return FormulaInvariant(name=name, check=f"{total_col} == sum({part_cols})", columns=(total_col, *part_cols), evaluator=_check)


def make_le_invariant(name: str, lesser_col: str, greater_col: str, tolerance: float = 1e-6) -> FormulaInvariant:
    """lesser_col <= greater_col, for rows where both are present."""

    def _check(df: pd.DataFrame) -> pd.Series:
        lesser = pd.to_numeric(df[lesser_col], errors="coerce")
        greater = pd.to_numeric(df[greater_col], errors="coerce")
        both_present = lesser.notna() & greater.notna()
        return (~both_present) | (lesser <= greater + tolerance)

    return FormulaInvariant(name=name, check=f"{lesser_col} <= {greater_col}", columns=(lesser_col, greater_col), evaluator=_check)


def gate_formula_consistency(
    df: pd.DataFrame,
    invariants: list[FormulaInvariant],
    source_column: Optional[str],
    *,
    min_pass_rate: float = 0.95,
    max_source_spread: float = 0.15,
) -> list[GateResult]:
    """Per invariant: overall pass rate, and (if source_column is given) pass
    rate PER SOURCE. A pass rate that differs sharply between sources means
    the same field was computed by a different formula per source — a bug,
    not a signal (see CLAUDE.md: total_sanctioned was Summary-derived for one
    batch and account-summed for another, failing its own invariant 76% of
    the time in one batch).
    """
    results: list[GateResult] = []
    for inv in invariants:
        missing = [c for c in inv.columns if c not in df.columns]
        if missing:
            results.append(
                GateResult(f"formula_consistency:{inv.name}", False, f"columns not found: {missing}", details={"missing_columns": missing})
            )
            continue

        holds = inv.evaluator(df)
        overall_rate = float(holds.mean()) if len(holds) else float("nan")
        details: dict = {"overall_pass_rate": overall_rate, "check": inv.check}
        passed = overall_rate >= min_pass_rate
        message_parts = [f"{inv.check}: {overall_rate:.1%} of rows satisfy the invariant"]

        if source_column and source_column in df.columns:
            per_source = holds.groupby(df[source_column], dropna=False).mean()
            details["per_source_pass_rate"] = {str(k): float(v) for k, v in per_source.items()}
            if len(per_source) > 1:
                spread = float(per_source.max() - per_source.min())
                details["source_spread"] = spread
                if spread > max_source_spread:
                    passed = False
                    worst = per_source.idxmin()
                    message_parts.append(
                        f"pass rate varies sharply by source (spread {spread:.1%}, worst: {worst!r} at "
                        f"{per_source.min():.1%}) — likely a different formula per source, not real data"
                    )

        results.append(GateResult(f"formula_consistency:{inv.name}", passed, "; ".join(message_parts), details=details))
    return results
