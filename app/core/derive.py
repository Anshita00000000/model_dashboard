"""Derived feature engineering — the modelling-ready transformations applied
after assembly (app/core/assemble.py) and before export.

Every function here is independently callable and configurable from the UI;
none of them mutate their input dataframe. Each returns (new_df, report) so
the caller (app/ui/tab_enriched_eda.py) can show what happened rather than
silently applying a transformation — consistent with this codebase's
"fail loudly, report don't silently drop" bias (CLAUDE.md).

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import raw_eda
from . import schema as schema_mod

# ---------------------------------------------------------------------------
# Temporal features from created_at
# ---------------------------------------------------------------------------

TEMPORAL_SUFFIXES = ("month", "day_of_week", "hour", "is_weekend", "day_of_month", "week_of_month")


@dataclass(frozen=True)
class TemporalDeriveReport:
    column: str
    parsed_count: int
    unparsed_count: int
    produced_columns: list[str]


def derive_temporal_features(
    df: pd.DataFrame, *, column: str = "created_at", day_first: Optional[bool] = None
) -> tuple[pd.DataFrame, TemporalDeriveReport]:
    """month, day_of_week (Monday=0), hour, is_weekend, day_of_month,
    week_of_month (1-5). day_first should come from Tab 2's ambiguity finding
    for this column, confirmed by the user — never guessed here. Rows whose
    value doesn't parse get null for every produced column, never a guessed
    default.
    """
    if column not in df.columns:
        raise ValueError(f"column {column!r} not present in dataframe")
    out = df.copy()
    non_blank = ~df[column].map(schema_mod.is_blank)
    parsed = pd.to_datetime(df[column], errors="coerce", dayfirst=bool(day_first))

    day_of_week = parsed.dt.dayofweek
    is_weekend = day_of_week.isin([5, 6]).astype("float64").mask(day_of_week.isna())

    out[f"{column}_month"] = parsed.dt.month
    out[f"{column}_day_of_week"] = day_of_week
    out[f"{column}_hour"] = parsed.dt.hour
    out[f"{column}_is_weekend"] = is_weekend
    out[f"{column}_day_of_month"] = parsed.dt.day
    out[f"{column}_week_of_month"] = ((parsed.dt.day - 1) // 7) + 1

    report = TemporalDeriveReport(
        column=column,
        parsed_count=int((parsed.notna() & non_blank).sum()),
        unparsed_count=int((parsed.isna() & non_blank).sum()),
        produced_columns=[f"{column}_{suffix}" for suffix in TEMPORAL_SUFFIXES],
    )
    return out, report


# ---------------------------------------------------------------------------
# Funnel target derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelDeriveReport:
    disposition_column: str
    stage_order: list[str]
    unmapped_values: list[tuple[str, int]]  # raw disposition values with no mapping, most frequent first
    stage_counts: dict[str, int]
    produced_columns: list[str]


def derive_funnel_stages(
    df: pd.DataFrame, *, disposition_column: str = "disposition", stage_order: list[str], mapping: dict[str, str]
) -> tuple[pd.DataFrame, FunnelDeriveReport]:
    """User maps raw disposition values to an ordered stage list (e.g. Evoke's
    "No Appointment"/"Appointment Booked"/"Consulted"/"Converted"). Produces:
      - funnel_stage: the raw value normalised onto stage_order's vocabulary
      - reached_{stage} for every stage after the first (K stages -> K-1
        binary flags, per CLAUDE.md — "K outcome levels require K-1
        conditional stages, not K"): 1 if this lead's stage is at or past
        `stage`, 0 if it fell short, null if the raw value was blank or
        unmapped (never coerced to 0 — an unknown outcome is not a negative).
    A raw value with no entry in `mapping` is left out of funnel_stage/every
    reached_ flag and reported back, never silently dropped or guessed at.
    """
    if disposition_column not in df.columns:
        raise ValueError(f"column {disposition_column!r} not present in dataframe")
    if not stage_order:
        raise ValueError("stage_order must not be empty")
    unknown_targets = sorted(set(mapping.values()) - set(stage_order))
    if unknown_targets:
        raise ValueError(f"mapping targets not present in stage_order: {unknown_targets}")

    out = df.copy()
    raw = df[disposition_column]
    non_blank = ~raw.map(schema_mod.is_blank)

    def _map_value(v):
        if schema_mod.is_blank(v):
            return None
        return mapping.get(str(v).strip())

    normalized_stage = raw.map(_map_value)
    out["funnel_stage"] = normalized_stage

    stage_index = {stage: i for i, stage in enumerate(stage_order)}
    stage_rank = normalized_stage.map(lambda s: stage_index.get(s) if s is not None else None)

    produced = ["funnel_stage"]
    for stage in stage_order[1:]:
        col = f"reached_{stage}"
        flag = pd.Series(np.nan, index=df.index, dtype="float64")
        known = stage_rank.notna()
        flag.loc[known] = (stage_rank.loc[known] >= stage_index[stage]).astype(float)
        out[col] = flag
        produced.append(col)

    unmapped_mask = non_blank & normalized_stage.isna()
    unmapped_counts = raw[unmapped_mask].value_counts()
    unmapped_values = [(str(k), int(v)) for k, v in unmapped_counts.items()]
    stage_counts = {stage: int((normalized_stage == stage).sum()) for stage in stage_order}

    report = FunnelDeriveReport(
        disposition_column=disposition_column, stage_order=list(stage_order),
        unmapped_values=unmapped_values, stage_counts=stage_counts, produced_columns=produced,
    )
    return out, report


# ---------------------------------------------------------------------------
# Multi-valued categorical normalisation
# ---------------------------------------------------------------------------

DEFAULT_MULTI_VALUE_DELIMITER = ";"
DEFAULT_TOP_N_COMPONENTS = 10


@dataclass(frozen=True)
class MultiValuedDeriveReport:
    column: str
    delimiter: str
    distinct_component_count: int
    flagged_components: list[str]
    produced_columns: list[str]


def _split_components(value, delimiter: str) -> Optional[list[str]]:
    if schema_mod.is_blank(value):
        return None
    parts = [p.strip() for p in str(value).split(delimiter)]
    return [p for p in parts if p]


def normalize_multi_valued_column(
    df: pd.DataFrame, column: str, *, delimiter: str = DEFAULT_MULTI_VALUE_DELIMITER,
    top_n_components: int = DEFAULT_TOP_N_COMPONENTS,
) -> tuple[pd.DataFrame, MultiValuedDeriveReport]:
    """"Back; Leg" and "Leg; Back" collapse to the same normalised category
    (components sorted before rejoining) so order-of-entry doesn't fragment
    an otherwise-identical combination into two categories. Also emits a
    component count and a binary has_<component> flag per common component
    (top_n_components by frequency, to avoid a combinatorial explosion of
    near-unique flags on a near-free-text field).
    """
    if column not in df.columns:
        raise ValueError(f"column {column!r} not present in dataframe")
    out = df.copy()

    join_sep = f"{delimiter} " if not delimiter.endswith(" ") else delimiter
    parsed = df[column].map(lambda v: _split_components(v, delimiter))
    out[f"{column}_normalized"] = parsed.map(lambda comps: join_sep.join(sorted(comps)) if comps is not None else None)
    out[f"{column}_component_count"] = parsed.map(lambda comps: len(comps) if comps is not None else None)

    component_counter: Counter = Counter()
    for comps in parsed.dropna():
        component_counter.update(comps)
    top_components = [c for c, _ in component_counter.most_common(top_n_components)]

    produced = [f"{column}_normalized", f"{column}_component_count"]
    for component in top_components:
        safe_name = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_") or "component"
        col_name = f"{column}_has_{safe_name}"
        out[col_name] = parsed.map(lambda comps: (component in comps) if comps is not None else None)
        produced.append(col_name)

    report = MultiValuedDeriveReport(
        column=column, delimiter=delimiter, distinct_component_count=len(component_counter),
        flagged_components=top_components, produced_columns=produced,
    )
    return out, report


# ---------------------------------------------------------------------------
# Free-text duration parsing
# ---------------------------------------------------------------------------

DEFAULT_DURATION_BAND_EDGES: tuple[int, ...] = (6, 12, 24, 60)

# Years and months are each optional, but at least one unit must be present —
# a bare number with no unit ("5") is genuinely ambiguous and is never guessed.
_DURATION_RE = re.compile(
    r"(?:(?P<years>\d+)\s*y(?:r|ea?r)?s?)?\s*(?:(?P<months>\d+)\s*mo?n?(?:th)?s?)?", re.IGNORECASE
)


def parse_duration_to_months(value) -> Optional[int]:
    """"6yrs 0mon" -> 72, "0yrs 6mon" -> 6, "2 years 3 months" -> 27.
    Returns None for blank or unparseable text — never guesses.
    """
    if schema_mod.is_blank(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _DURATION_RE.search(text)
    if not match:
        return None
    years_str, months_str = match.group("years"), match.group("months")
    if years_str is None and months_str is None:
        return None
    return int(years_str or 0) * 12 + int(months_str or 0)


def band_duration_months(months: Optional[int], *, edges: tuple[int, ...] = DEFAULT_DURATION_BAND_EDGES) -> Optional[str]:
    # `months` may arrive as a genuine Python None (called directly) or as a
    # pandas/numpy NaN (called via .map() on a Series that's upcast to
    # float64 once None is mixed with real numbers) — both mean "unparsed".
    if months is None or (isinstance(months, float) and pd.isna(months)):
        return None
    boundaries = sorted(edges)
    lower = 0
    for edge in boundaries:
        if months < edge:
            return f"{lower}-{edge}"
        lower = edge
    return f"{lower}+"


@dataclass(frozen=True)
class DurationDeriveReport:
    column: str
    parsed_count: int
    unparsed_count: int
    produced_columns: list[str]


def derive_duration_features(
    df: pd.DataFrame, column: str, *, band_edges: tuple[int, ...] = DEFAULT_DURATION_BAND_EDGES
) -> tuple[pd.DataFrame, DurationDeriveReport]:
    if column not in df.columns:
        raise ValueError(f"column {column!r} not present in dataframe")
    out = df.copy()
    months = df[column].map(parse_duration_to_months)
    out[f"{column}_months"] = months
    out[f"{column}_band"] = months.map(lambda m: band_duration_months(m, edges=band_edges))

    non_blank = ~df[column].map(schema_mod.is_blank)
    report = DurationDeriveReport(
        column=column,
        parsed_count=int((months.notna() & non_blank).sum()),
        unparsed_count=int((months.isna() & non_blank).sum()),
        produced_columns=[f"{column}_months", f"{column}_band"],
    )
    return out, report


# ---------------------------------------------------------------------------
# Near-duplicate category consolidation
#
# Tab 2 (app/ui/tab_raw_eda.py) is deliberately read-only and never writes
# back a modified dataset (see its module docstring) — it only surfaces
# raw_eda.find_near_duplicate_categories() as a report. This is where those
# findings become an actionable, user-confirmed transformation: the caller
# (app/ui/tab_enriched_eda.py) re-runs the same detector on the assembled
# dataframe, presents each suggestion for the user to confirm or reject, and
# only CONFIRMED merges are ever applied — nothing here auto-applies.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeSuggestion:
    column: str
    variant: str
    canonical: str
    kind: str  # "case_or_whitespace" | "possible_spelling_variant"
    similarity: Optional[float]
    count: int
    # case/whitespace merges carry zero false-positive risk (raw_eda.py);
    # fuzzy spelling-variant merges need a human to actually look at them.
    default_confirmed: bool


def suggest_category_merges(df: pd.DataFrame, columns: list[str]) -> list[MergeSuggestion]:
    groups = raw_eda.find_near_duplicate_categories(df, columns)
    suggestions: list[MergeSuggestion] = []
    for group in groups:
        for variant, count in group.variants:
            if variant == group.canonical:
                continue
            suggestions.append(MergeSuggestion(
                column=group.column, variant=variant, canonical=group.canonical, kind=group.kind,
                similarity=group.similarity, count=count,
                default_confirmed=(group.kind == "case_or_whitespace"),
            ))
    return suggestions


def apply_category_merges(df: pd.DataFrame, confirmed: list[MergeSuggestion]) -> pd.DataFrame:
    """Applies only the merges the caller passes in — build `confirmed` from
    whichever suggest_category_merges() results the user actually checked.
    """
    out = df.copy()
    by_column: dict[str, dict[str, str]] = {}
    for s in confirmed:
        by_column.setdefault(s.column, {})[s.variant] = s.canonical

    for column, replacements in by_column.items():
        if column not in out.columns:
            continue
        out[column] = out[column].map(lambda v: replacements.get(v, v) if not schema_mod.is_blank(v) else v)

    return out
