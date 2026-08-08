"""Model architectures: which features go in, and what hyperparameters to use.

Three architectures, numbered to match the project's existing convention
(ARCH 3, quantile-binned, is deliberately out of scope for this phase):

  ARCH 1 "baseline_anchor"           — core/CRM features only. The floor
                                        benchmark that proves what enrichment
                                        actually adds.
  ARCH 2 "unified_native_sparse"     — all features, raw, with native NaN.
                                        Never impute — LightGBM routes around
                                        missingness natively, and imputation
                                        destroys the "no bureau data" signal.
  ARCH 4 "dual_route_cascade"        — model A: enriched rows only, all
                                        features. Model B: all rows, core
                                        features only. Blend for enriched
                                        rows, B alone for unenriched.

Every architecture gets cat_smooth=50.0 and min_data_per_group=50 by default
(user-overridable): a category with few training rows otherwise gets an
unreliable, overconfident estimate — see CLAUDE.md, the 115-row clinic that
learned a ~2x-inflated rate and then grew 9x in volume.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

ArchitectureName = Literal["baseline_anchor", "unified_native_sparse", "dual_route_cascade"]

ARCHITECTURES: tuple[ArchitectureName, ...] = ("baseline_anchor", "unified_native_sparse", "dual_route_cascade")

ARCHITECTURE_LABELS: dict[ArchitectureName, str] = {
    "baseline_anchor": "ARCH 1 — Baseline Anchor",
    "unified_native_sparse": "ARCH 2 — Unified Native Sparse",
    "dual_route_cascade": "ARCH 4 — Dual-Route Cascade",
}

MANDATORY_DEFAULTS: dict[str, float] = {
    "cat_smooth": 50.0,
    "min_data_per_group": 50,
}

_ARCHITECTURE_DEFAULT_HYPERPARAMS: dict[ArchitectureName, dict] = {
    "baseline_anchor": {
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "max_bin": 255,
        "n_estimators": 300,
    },
    "unified_native_sparse": {
        "max_bin": 128,
        "min_child_samples": 100,
        "colsample_bytree": 0.6,
        "reg_lambda": 10.0,
        "n_estimators": 300,
    },
    "dual_route_cascade": {
        "n_estimators": 300,
    },
}

DEFAULT_BLEND_WEIGHT = 0.65
MIN_ENRICHED_ROWS_FOR_MODEL_A = 30


def default_hyperparams(architecture: ArchitectureName) -> dict:
    """Sensible defaults for this architecture, mandatory settings included.
    Always user-overridable — this is just the pre-filled starting point.
    """
    if architecture not in _ARCHITECTURE_DEFAULT_HYPERPARAMS:
        raise ValueError(f"unknown architecture {architecture!r}, must be one of {ARCHITECTURES}")
    return {**MANDATORY_DEFAULTS, **_ARCHITECTURE_DEFAULT_HYPERPARAMS[architecture]}


def with_mandatory_defaults(hyperparams: dict) -> dict:
    """Ensure cat_smooth / min_data_per_group are present, without clobbering
    an explicit user override of either.
    """
    return {**MANDATORY_DEFAULTS, **hyperparams}


@dataclass(frozen=True)
class ArchitectureSpec:
    name: ArchitectureName
    hyperparams: dict
    blend_weight: float = DEFAULT_BLEND_WEIGHT  # only meaningful for dual_route_cascade


def select_features(
    architecture: ArchitectureName,
    *,
    all_features: list[str],
    feature_sources: dict[str, str],
    route: Optional[Literal["a", "b"]] = None,
) -> list[str]:
    """Which feature columns this architecture (and, for dual-route, which
    route) actually trains on. Order is preserved from all_features.
    """
    if architecture == "baseline_anchor":
        return [c for c in all_features if feature_sources.get(c, "core") == "core"]
    if architecture == "unified_native_sparse":
        return list(all_features)
    if architecture == "dual_route_cascade":
        if route == "a":
            return list(all_features)
        if route == "b":
            return [c for c in all_features if feature_sources.get(c, "core") == "core"]
        raise ValueError("dual_route_cascade requires route='a' or route='b'")
    raise ValueError(f"unknown architecture {architecture!r}, must be one of {ARCHITECTURES}")


def enriched_row_mask(df: pd.DataFrame, enrichment_features: list[str]) -> pd.Series:
    """A row counts as 'enriched' if at least one enrichment feature is non-null
    (i.e. this lead has real bureau/enrichment data on file, per CLAUDE.md).
    """
    if not enrichment_features:
        return pd.Series(False, index=df.index)
    present = [c for c in enrichment_features if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].notna().any(axis=1)
