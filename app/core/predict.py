"""Batch prediction: pre-flight validation, scoring, and tier assignment.

Scoring uses ONLY a loaded registry.ModelBundle's pinned state — see
registry.py's ModelBundle.predict(), which never refits. Pre-flight
validation runs BEFORE any scoring and reports every problem it finds in one
pass, so a caller (app/ui/tab_prediction.py) can refuse to score a
partially-valid file rather than silently scoring around a hole in the data.

Tier assignment uses the CUTOFFS pinned in the bundle at training time
(bundle.metrics["tier_cutoffs"], written by app/core/train.py /
app/ui/tab_training.py), never by re-ranking the batch being scored — see
evaluate.tier_cutoffs()'s docstring for why that matters.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from . import evaluate, registry, storage
from . import schema as schema_mod

NULL_RATE_SHIFT_WARN_THRESHOLD = 0.20  # 20 percentage points
TOP_OFFENDING_VALUES_TO_REPORT = 10


def new_run_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    column: str
    kind: str  # "missing_feature" | "dtype_not_coercible" | "unseen_categories" | "null_rate_shift"
    severity: str  # "error" | "warning"
    message: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_scoreable(self) -> bool:
        return not self.errors


def _all_pinned_features(bundle: registry.ModelBundle) -> list[str]:
    seen: list[str] = []
    for feature_list in bundle.preprocessing.stage_features.values():
        for f in feature_list:
            if f not in seen:
                seen.append(f)
    return seen


def validate_for_prediction(df: pd.DataFrame, bundle: registry.ModelBundle) -> ValidationReport:
    """Check df against everything the bundle's predict() will assume, without
    scoring anything. Reports every issue found, in one pass:

      - missing required features                                (error, blocks scoring)
      - numeric values that can't be coerced to the pinned dtype  (error, blocks scoring)
      - categorical values not seen during training                (warning: map to null)
      - per-feature null rate shifted > 20pp vs training time       (warning)
    """
    issues: list[ValidationIssue] = []
    all_features = _all_pinned_features(bundle)

    missing = [f for f in all_features if f not in df.columns]
    for f in missing:
        issues.append(
            ValidationIssue(f, "missing_feature", "error", f"required feature {f!r} is absent from this dataset")
        )

    present_features = [f for f in all_features if f not in missing]

    for col in present_features:
        series = df[col]

        if col in bundle.preprocessing.numeric_dtypes:
            dtype = bundle.preprocessing.numeric_dtypes[col]
            coerced = pd.to_numeric(series, errors="coerce")
            originally_non_blank = ~series.map(schema_mod.is_blank)
            failed = originally_non_blank & coerced.isna()
            n_failed = int(failed.sum())
            if n_failed:
                examples = series.loc[failed].astype(str).unique().tolist()[:TOP_OFFENDING_VALUES_TO_REPORT]
                issues.append(
                    ValidationIssue(
                        col,
                        "dtype_not_coercible",
                        "error",
                        f"{n_failed} value(s) in {col!r} cannot be coerced to {dtype}",
                        details={"n_failed": n_failed, "examples": examples},
                    )
                )

        if col in bundle.preprocessing.categorical_vocab:
            vocab = set(bundle.preprocessing.categorical_vocab[col])
            non_blank = series[~series.map(schema_mod.is_blank)]
            unseen = non_blank[~non_blank.astype(str).isin(vocab)]
            if len(unseen):
                top_values = unseen.astype(str).value_counts().head(TOP_OFFENDING_VALUES_TO_REPORT)
                issues.append(
                    ValidationIssue(
                        col,
                        "unseen_categories",
                        "warning",
                        f"{len(unseen)} row(s) in {col!r} have a category not seen during training — "
                        f"these will score as null for this feature",
                        details={"n_unseen": int(len(unseen)), "top_values": {str(k): int(v) for k, v in top_values.items()}},
                    )
                )

        training_rate = bundle.preprocessing.training_null_rates.get(col)
        if training_rate is not None:
            current_rate = schema_mod.blank_rate(series)
            shift = current_rate - training_rate
            if abs(shift) > NULL_RATE_SHIFT_WARN_THRESHOLD:
                issues.append(
                    ValidationIssue(
                        col,
                        "null_rate_shift",
                        "warning",
                        f"{col!r} null rate shifted from {training_rate:.1%} at training time to "
                        f"{current_rate:.1%} now ({shift:+.1%} points)",
                        details={"training_null_rate": training_rate, "current_null_rate": current_rate, "shift_pp": shift},
                    )
                )

    return ValidationReport(issues=issues)


# ---------------------------------------------------------------------------
# Scoring + tier assignment
# ---------------------------------------------------------------------------


def score_batch(df: pd.DataFrame, bundle: registry.ModelBundle, *, id_columns: list[str], bundle_id: str) -> pd.DataFrame:
    """Score df using ONLY the bundle's pinned state. Tiers use the bundle's
    training-time cutoffs (bundle.metrics["tier_cutoffs"]) — never re-ranked
    within this batch, so tier assignment stays comparable across runs.
    """
    scored = bundle.predict(df)  # {stage}_prob columns + cumulative "score"; never refits
    tier_cutoffs_by_stage = bundle.metrics.get("tier_cutoffs", {})

    output = pd.DataFrame(index=df.index)
    for col in id_columns:
        if col not in df.columns:
            raise ValueError(f"id column {col!r} not found in dataframe")
        output[col] = df[col]

    for spec in bundle.manifest.stages:
        prob_col = f"{spec.name}_prob"
        output[prob_col] = scored[prob_col]
        cutoffs = tier_cutoffs_by_stage.get(spec.name)
        output[f"{spec.name}_tier"] = evaluate.assign_tiers(scored[prob_col].to_numpy(), cutoffs or [])

    output["score"] = scored["score"]
    output["bundle_id"] = bundle_id
    output["scored_at"] = storage.utcnow_iso()
    return output


def save_prediction_run(
    *,
    store: storage.MetadataStore,
    root: Path | str,
    merchant: str,
    bundle_id: str,
    dataset_id: str,
    scored_df: pd.DataFrame,
    name: str = "predictions",
) -> tuple[str, Path]:
    run_id = new_run_id()
    path = storage.save_dataframe(scored_df, "prediction", merchant, name, root=root)
    store.insert_prediction_run(run_id=run_id, bundle_id=bundle_id, dataset_id=dataset_id, row_count=len(scored_df), output_path=str(path))
    return run_id, path
