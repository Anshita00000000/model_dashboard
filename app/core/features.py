"""Feature typing: numeric vs categorical, and core vs enrichment source tagging.

Two independent classifications, both fully user-overridable:
  - numeric/categorical: same >=90%-parse-rate rule as app/core/schema.py.
  - core/enrichment: which columns come from the six enrichment sources in
    CLAUDE.md (CRIF, Equifax, EPFO, salary estimator, PayU,
    LeadCreditEngineOutput) vs core/CRM fields. This drives ARCH 1's "core
    only" feature set, ARCH 4's dual-route split, and the feature-gain
    source tag in app/core/evaluate.py.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from .schema import classify_dtype

FeatureSource = Literal["core", "enrichment"]

# Column-name patterns matching CLAUDE.md's enrichment sources. A sensible
# default, always overridable per column via `overrides`.
DEFAULT_ENRICHMENT_PATTERNS: tuple[str, ...] = (
    r"crif",
    r"equifax",
    r"epfo",
    r"salary_est",
    r"est_salary",
    r"income_est",
    r"payu",
    r"credit_engine",
    r"leadcreditengine",
    r"bureau",
)


@dataclass(frozen=True)
class FeatureTypes:
    numeric: list[str]
    categorical: list[str]

    def all_features(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)


def classify_numeric_categorical(
    df: pd.DataFrame,
    columns: list[str],
    overrides: Optional[dict[str, str]] = None,
) -> FeatureTypes:
    """Numeric if >=90% of non-null, non-empty values parse as numbers, else categorical."""
    overrides = overrides or {}
    numeric: list[str] = []
    categorical: list[str] = []
    for col in columns:
        forced = overrides.get(col)
        if forced is not None:
            if forced not in ("numeric", "categorical"):
                raise ValueError(f"override for {col!r} must be 'numeric' or 'categorical', got {forced!r}")
            dtype = forced
        else:
            dtype = classify_dtype(df[col])
        (numeric if dtype == "numeric" else categorical).append(col)
    return FeatureTypes(numeric=numeric, categorical=categorical)


def classify_feature_source(
    column_name: str,
    overrides: Optional[dict[str, FeatureSource]] = None,
    patterns: tuple[str, ...] = DEFAULT_ENRICHMENT_PATTERNS,
) -> FeatureSource:
    if overrides and column_name in overrides:
        return overrides[column_name]
    lowered = column_name.lower()
    for pattern in patterns:
        if re.search(pattern, lowered):
            return "enrichment"
    return "core"


def classify_feature_sources(
    columns: list[str],
    overrides: Optional[dict[str, FeatureSource]] = None,
    patterns: tuple[str, ...] = DEFAULT_ENRICHMENT_PATTERNS,
) -> dict[str, FeatureSource]:
    """{column: "core"|"enrichment"} for every column in `columns`."""
    return {c: classify_feature_source(c, overrides, patterns) for c in columns}
