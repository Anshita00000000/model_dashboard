"""Automated profiling of a raw, as-uploaded file.

Read-only: nothing here ever writes back a modified dataset. The output is a
report plus flagged findings for the team's OFFLINE cleaning pass — this
module diagnoses, it never fixes. No canonical schema, no renaming, no
coercion of the underlying data (only of throwaway copies used to compute a
statistic, e.g. parsing a numeric-looking string to float to describe its
distribution).

Every "finding" here maps to a real bug documented in CLAUDE.md's hard-won
rules. All of them are report-only: they surface evidence and a plain-language
explanation, never an auto-fix.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import schema as schema_mod

_NUMERIC_DATE_RE = re.compile(r"^(\d{1,4})[/\-.](\d{1,4})[/\-.](\d{1,4})$")
_TEXT_MONTH_RE = re.compile(
    r"^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}$", re.IGNORECASE
)
DATE_LIKE_MIN_FRACTION = 0.7


def _date_match_masks(non_blank: pd.Series) -> tuple[pd.Series, pd.Series]:
    return non_blank.str.match(_NUMERIC_DATE_RE), non_blank.str.match(_TEXT_MONTH_RE)


def _is_date_like_column(non_blank: pd.Series) -> bool:
    """Shared with detect_date_format: date strings are near-identical to each
    other character-for-character (same separators, similar digit runs), which
    otherwise floods near-duplicate-category fuzzy matching with false positives
    like '30/01/2024' vs '31/01/2024'.
    """
    if non_blank.empty:
        return False
    numeric_matches, text_matches = _date_match_masks(non_blank)
    return bool((numeric_matches | text_matches).mean() >= DATE_LIKE_MIN_FRACTION)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Overview:
    row_count: int
    col_count: int
    memory_bytes: int
    fully_duplicate_row_count: int
    fully_empty_columns: list[str]


def profile_overview(df: pd.DataFrame) -> Overview:
    return Overview(
        row_count=len(df),
        col_count=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        fully_duplicate_row_count=int(df.duplicated(keep="first").sum()),
        fully_empty_columns=[c for c in df.columns if bool(df[c].map(schema_mod.is_blank).all())],
    )


# ---------------------------------------------------------------------------
# Per-column profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    column: str
    inferred_dtype: str  # "numeric" | "categorical" — INFERRED FOR PROFILING, never applied
    null_pct: float
    distinct_count: int
    sample_values: list[str]
    most_frequent: list[tuple[str, int]]


def profile_columns(df: pd.DataFrame, top_n: int = 5, sample_n: int = 5) -> list[ColumnProfile]:
    profiles = []
    for col in df.columns:
        series = df[col]
        non_blank = series[~series.map(schema_mod.is_blank)].astype(str)
        value_counts = non_blank.value_counts()
        profiles.append(
            ColumnProfile(
                column=col,
                inferred_dtype=schema_mod.classify_dtype(series),
                null_pct=100 * schema_mod.blank_rate(series),
                distinct_count=int(value_counts.shape[0]),
                sample_values=non_blank.unique()[:sample_n].tolist(),
                most_frequent=[(str(v), int(c)) for v, c in value_counts.head(top_n).items()],
            )
        )
    return profiles


def categorical_columns(profiles: list[ColumnProfile]) -> list[str]:
    return [p.column for p in profiles if p.inferred_dtype == "categorical"]


def numeric_columns(profiles: list[ColumnProfile]) -> list[str]:
    return [p.column for p in profiles if p.inferred_dtype == "numeric"]


# ---------------------------------------------------------------------------
# FINDING — near-duplicate categories
# ---------------------------------------------------------------------------

MIN_GROUP_SIZE = 2
FUZZY_MIN_LENGTH = 5
FUZZY_SIMILARITY_THRESHOLD = 0.65
MAX_DISTINCT_FOR_FUZZY = 300
MAX_FUZZY_PAIRS_PER_COLUMN = 15
# Free-text columns (most values distinct) produce a flood of coincidentally-similar
# strings under fuzzy matching (e.g. "note 9" vs "note 99") — skip fuzzy matching
# there; find_free_text_fields separately flags those columns on their own terms.
FUZZY_MAX_DISTINCT_RATIO = 0.3


@dataclass(frozen=True)
class NearDuplicateGroup:
    column: str
    kind: str  # "case_or_whitespace" (high confidence) | "possible_spelling_variant" (needs review)
    canonical: str  # suggested canonical form — the most frequent literal spelling
    variants: list[tuple[str, int]]  # (raw_value, count), most frequent first
    similarity: Optional[float] = None  # set for "possible_spelling_variant" only


def _normalize_case_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def find_near_duplicate_categories(df: pd.DataFrame, categorical_columns: list[str]) -> list[NearDuplicateGroup]:
    """Two tiers, both report-only:
      - case_or_whitespace: exact match after case-folding + whitespace collapse.
        Zero false-positive risk — e.g. "FB Ads"/"Fb Ads", "Weight Loss"/"Weight Loss ".
      - possible_spelling_variant: distinct normalized forms that are still highly
        similar strings (e.g. "Gurugram"/"Gurgaon"). This is a genuine heuristic
        and WILL surface occasional false positives (e.g. "Hyderabad"/"Ahmedabad"
        can score similarly) — that's why it's report-only with the similarity
        score shown, for a human to judge, never auto-merged.
    """
    findings: list[NearDuplicateGroup] = []
    for col in categorical_columns:
        series = df[col]
        non_blank = series[~series.map(schema_mod.is_blank)].astype(str)
        if non_blank.empty or _is_date_like_column(non_blank):
            continue

        exact_groups: dict[str, Counter] = defaultdict(Counter)
        for raw in non_blank:
            exact_groups[_normalize_case_whitespace(raw)][raw] += 1

        for norm_key, counter in exact_groups.items():
            if len(counter) < MIN_GROUP_SIZE:
                continue
            variants = counter.most_common()
            findings.append(NearDuplicateGroup(column=col, kind="case_or_whitespace", canonical=variants[0][0], variants=variants))

        representatives = [counter.most_common(1)[0][0] for counter in exact_groups.values()]
        distinct_ratio = len(representatives) / len(non_blank)
        if 2 <= len(representatives) <= MAX_DISTINCT_FOR_FUZZY and distinct_ratio <= FUZZY_MAX_DISTINCT_RATIO:
            fuzzy_pairs: list[tuple[float, str, str]] = []
            for i in range(len(representatives)):
                for j in range(i + 1, len(representatives)):
                    a, b = representatives[i], representatives[j]
                    if len(a) < FUZZY_MIN_LENGTH or len(b) < FUZZY_MIN_LENGTH:
                        continue
                    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
                    if ratio >= FUZZY_SIMILARITY_THRESHOLD:
                        fuzzy_pairs.append((ratio, a, b))
            fuzzy_pairs.sort(reverse=True)
            for ratio, a, b in fuzzy_pairs[:MAX_FUZZY_PAIRS_PER_COLUMN]:
                count_a = exact_groups[_normalize_case_whitespace(a)][a]
                count_b = exact_groups[_normalize_case_whitespace(b)][b]
                variants = sorted([(a, count_a), (b, count_b)], key=lambda t: -t[1])
                findings.append(
                    NearDuplicateGroup(
                        column=col, kind="possible_spelling_variant", canonical=variants[0][0],
                        variants=variants, similarity=round(ratio, 3),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# FINDING — sentinel / disguised-code detection
# ---------------------------------------------------------------------------

SENTINEL_LITERAL_VALUES = (0.0, -1.0)
SENTINEL_LITERAL_MIN_FRACTION = 0.01
SENTINEL_CLUSTER_MIN_LOW_FRACTION = 0.005
SENTINEL_CLUSTER_MAX_LOW_FRACTION = 0.95
SENTINEL_CLUSTER_MIN_GAP_RATIO = 0.15


@dataclass(frozen=True)
class SentinelFinding:
    column: str
    kind: str  # "literal_value" | "low_cluster"
    message: str
    count: int
    fraction: float


def _numeric_values(series: pd.Series) -> pd.Series:
    non_blank = series[~series.map(schema_mod.is_blank)]
    return pd.to_numeric(non_blank, errors="coerce").dropna()


def _detect_literal_sentinels(column: str, numeric: pd.Series) -> list[SentinelFinding]:
    findings = []
    for sentinel in SENTINEL_LITERAL_VALUES:
        mask = numeric == sentinel
        fraction = float(mask.mean())
        count = int(mask.sum())
        # only worth flagging if it's a real concentration AND there's a distinct
        # non-sentinel population to contrast it with (else it's just a legitimately
        # all-zero column, not a masquerading sentinel)
        if fraction >= SENTINEL_LITERAL_MIN_FRACTION and fraction < 1.0:
            others = numeric[~mask]
            findings.append(
                SentinelFinding(
                    column=column, kind="literal_value", count=count, fraction=fraction,
                    message=(
                        f"{count:,} values ({fraction:.1%}) are exactly {sentinel:g}, while the remaining "
                        f"{len(others):,} range {others.min():g}–{others.max():g} — check whether {sentinel:g} "
                        f"is a sentinel/placeholder rather than a real value."
                    ),
                )
            )
    return findings


def _detect_low_cluster(column: str, numeric: pd.Series) -> Optional[SentinelFinding]:
    sorted_vals = np.sort(numeric.unique())
    if len(sorted_vals) < 3:
        return None
    overall_range = sorted_vals[-1] - sorted_vals[0]
    if overall_range <= 0:
        return None
    gaps = np.diff(sorted_vals)
    idx = int(np.argmax(gaps))
    gap_ratio = gaps[idx] / overall_range
    if gap_ratio < SENTINEL_CLUSTER_MIN_GAP_RATIO:
        return None

    low_cutoff = sorted_vals[idx]
    high_start = sorted_vals[idx + 1]
    low_mask = numeric <= low_cutoff
    low_fraction = float(low_mask.mean())
    if not (SENTINEL_CLUSTER_MIN_LOW_FRACTION <= low_fraction <= SENTINEL_CLUSTER_MAX_LOW_FRACTION):
        return None

    low_count = int(low_mask.sum())
    high_vals = numeric[~low_mask]
    return SentinelFinding(
        column=column, kind="low_cluster", count=low_count, fraction=low_fraction,
        message=(
            f"{low_count:,} values are below {high_start:g} while the remaining values range "
            f"{high_start:g}–{high_vals.max():g} — likely reason codes rather than real scores."
        ),
    )


def detect_sentinel_values(df: pd.DataFrame, numeric_columns: list[str]) -> list[SentinelFinding]:
    findings: list[SentinelFinding] = []
    for col in numeric_columns:
        numeric = _numeric_values(df[col])
        if len(numeric) < 30:
            continue
        findings.extend(_detect_literal_sentinels(col, numeric))
        cluster = _detect_low_cluster(col, numeric)
        if cluster is not None:
            findings.append(cluster)
    return findings


# ---------------------------------------------------------------------------
# FINDING — ambiguous date formats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DateFormatFinding:
    column: str
    inference: str  # "day_first" | "month_first" | "ambiguous" | "mixed" | "text_month"
    evidence: str
    sample_values: list[str]


def detect_date_format(df: pd.DataFrame, column: str) -> Optional[DateFormatFinding]:
    series = df[column]
    non_blank = series[~series.map(schema_mod.is_blank)].astype(str).str.strip()
    if non_blank.empty:
        return None

    numeric_matches, text_matches = _date_match_masks(non_blank)
    date_like = numeric_matches | text_matches
    if date_like.mean() < DATE_LIKE_MIN_FRACTION:
        return None

    if text_matches.any() and not numeric_matches.any():
        examples = non_blank[text_matches].head(5).tolist()
        return DateFormatFinding(
            column=column, inference="text_month",
            evidence=f"{int(text_matches.sum())} value(s) match a 'D Mon YYYY' text format, e.g. {examples[0]!r}",
            sample_values=examples,
        )

    day_first_evidence: list[str] = []
    month_first_evidence: list[str] = []
    for v in non_blank[numeric_matches]:
        m = _NUMERIC_DATE_RE.match(v)
        c1, c2 = int(m.group(1)), int(m.group(2))
        if c1 > 12:
            day_first_evidence.append(v)
        if c2 > 12:
            month_first_evidence.append(v)

    examples = non_blank[numeric_matches].head(5).tolist()
    if day_first_evidence and month_first_evidence:
        return DateFormatFinding(
            column=column, inference="mixed",
            evidence=(
                f"found BOTH day>12 evidence (e.g. {day_first_evidence[0]!r}) AND month>12 evidence "
                f"(e.g. {month_first_evidence[0]!r}) in the same column — likely mixed formats, not one format"
            ),
            sample_values=examples,
        )
    if day_first_evidence:
        return DateFormatFinding(
            column=column, inference="day_first",
            evidence=(
                f"{len(day_first_evidence)} value(s) have a first component > 12 (e.g. "
                f"{day_first_evidence[0]!r}), so the first component must be the day"
            ),
            sample_values=examples,
        )
    if month_first_evidence:
        return DateFormatFinding(
            column=column, inference="month_first",
            evidence=(
                f"{len(month_first_evidence)} value(s) have a second component > 12 (e.g. "
                f"{month_first_evidence[0]!r}), so the first component must be the month"
            ),
            sample_values=examples,
        )
    return DateFormatFinding(
        column=column, inference="ambiguous",
        evidence=(
            "no value has a component > 12 in either the first or second position — day-first vs "
            "month-first cannot be determined empirically from this column alone"
        ),
        sample_values=examples,
    )


def detect_date_formats(df: pd.DataFrame) -> list[DateFormatFinding]:
    """Checked on EVERY column, independently — a single file may use different
    date formats in different columns."""
    findings = []
    for col in df.columns:
        finding = detect_date_format(df, col)
        if finding is not None:
            findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# FINDING — float-formatted identifiers
# ---------------------------------------------------------------------------

_FLOAT_ID_RE = re.compile(r"^\d{5,}\.0$")
FLOAT_ID_MIN_FRACTION = 0.5


@dataclass(frozen=True)
class FloatIdFinding:
    column: str
    match_fraction: float
    sample_values: list[str]


def detect_float_formatted_identifiers(df: pd.DataFrame) -> list[FloatIdFinding]:
    findings = []
    for col in df.columns:
        series = df[col]
        non_blank = series[~series.map(schema_mod.is_blank)].astype(str).str.strip()
        if non_blank.empty:
            continue
        matches = non_blank.str.match(_FLOAT_ID_RE)
        fraction = float(matches.mean())
        if fraction >= FLOAT_ID_MIN_FRACTION:
            findings.append(FloatIdFinding(column=col, match_fraction=fraction, sample_values=non_blank[matches].head(5).tolist()))
    return findings


# ---------------------------------------------------------------------------
# FINDING — multi-valued cells
# ---------------------------------------------------------------------------

MULTI_VALUE_DELIMITERS = (";", "|", ",")
MULTI_VALUE_MIN_FRACTION = 0.05


@dataclass(frozen=True)
class MultiValueFinding:
    column: str
    delimiter: str
    match_fraction: float
    distinct_component_count: int
    order_varies: bool
    sample_values: list[str]


def detect_multi_valued_cells(df: pd.DataFrame, categorical_columns: list[str]) -> list[MultiValueFinding]:
    findings = []
    for col in categorical_columns:
        series = df[col]
        non_blank = series[~series.map(schema_mod.is_blank)].astype(str).str.strip()
        if non_blank.empty:
            continue

        for delim in MULTI_VALUE_DELIMITERS:
            parts_per_row = non_blank.str.split(delim).apply(lambda parts: tuple(p.strip() for p in parts if p.strip()))
            has_multi = parts_per_row.apply(len) >= 2
            fraction = float(has_multi.mean())
            if fraction < MULTI_VALUE_MIN_FRACTION:
                continue

            matching = parts_per_row[has_multi]
            distinct_components: set[str] = set()
            orders_by_set: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
            for tup in matching:
                distinct_components.update(tup)
                orders_by_set[tuple(sorted(tup))].add(tup)
            order_varies = any(len(orders) > 1 for orders in orders_by_set.values())

            findings.append(
                MultiValueFinding(
                    column=col, delimiter=delim, match_fraction=fraction,
                    distinct_component_count=len(distinct_components), order_varies=order_varies,
                    sample_values=non_blank[has_multi].head(5).tolist(),
                )
            )
            break  # most relevant delimiter found for this column; avoid redundant reports
    return findings


# ---------------------------------------------------------------------------
# FINDING — free-text in a categorical-looking field
# ---------------------------------------------------------------------------

FREE_TEXT_MIN_ROWS = 20
FREE_TEXT_MIN_DISTINCT_RATIO = 0.5
FREE_TEXT_MIN_SINGLETON_RATIO = 0.5


@dataclass(frozen=True)
class FreeTextFinding:
    column: str
    distinct_count: int
    row_count: int
    distinct_ratio: float
    singleton_ratio: float
    sample_values: list[str]


def detect_free_text_fields(df: pd.DataFrame, categorical_columns: list[str]) -> list[FreeTextFinding]:
    findings = []
    for col in categorical_columns:
        series = df[col]
        non_blank = series[~series.map(schema_mod.is_blank)].astype(str)
        n = len(non_blank)
        if n < FREE_TEXT_MIN_ROWS:
            continue
        value_counts = non_blank.value_counts()
        distinct_ratio = len(value_counts) / n
        singleton_ratio = float((value_counts == 1).sum()) / len(value_counts) if len(value_counts) else 0.0
        if distinct_ratio >= FREE_TEXT_MIN_DISTINCT_RATIO and singleton_ratio >= FREE_TEXT_MIN_SINGLETON_RATIO:
            findings.append(
                FreeTextFinding(
                    column=col, distinct_count=len(value_counts), row_count=n,
                    distinct_ratio=float(distinct_ratio), singleton_ratio=singleton_ratio,
                    sample_values=non_blank.drop_duplicates().head(5).tolist(),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericDistribution:
    column: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    percentiles: dict[str, float]
    histogram_bins: list[float]
    histogram_counts: list[int]


def numeric_distribution(df: pd.DataFrame, column: str, n_bins: int = 30) -> Optional[NumericDistribution]:
    numeric = _numeric_values(df[column])
    if numeric.empty:
        return None
    counts, bin_edges = np.histogram(numeric, bins=n_bins)
    percentiles = {f"p{p}": float(np.percentile(numeric, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    return NumericDistribution(
        column=column, count=int(len(numeric)), min=float(numeric.min()), max=float(numeric.max()),
        mean=float(numeric.mean()), median=float(numeric.median()), std=float(numeric.std()) if len(numeric) > 1 else 0.0,
        percentiles=percentiles, histogram_bins=[float(b) for b in bin_edges], histogram_counts=[int(c) for c in counts],
    )


@dataclass(frozen=True)
class CategoricalDistribution:
    column: str
    top_values: list[tuple[str, int]]
    long_tail_count: int
    long_tail_distinct: int


def categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 30) -> Optional[CategoricalDistribution]:
    non_blank = df[column][~df[column].map(schema_mod.is_blank)].astype(str)
    if non_blank.empty:
        return None
    value_counts = non_blank.value_counts()
    top = value_counts.head(top_n)
    tail = value_counts.iloc[top_n:]
    return CategoricalDistribution(
        column=column, top_values=[(str(v), int(c)) for v, c in top.items()],
        long_tail_count=int(tail.sum()), long_tail_distinct=int(len(tail)),
    )


@dataclass(frozen=True)
class DateDistribution:
    column: str
    min_date: str
    max_date: str
    records_per_day: list[tuple[str, int]]


def date_distribution(df: pd.DataFrame, column: str, day_first: Optional[bool]) -> Optional[DateDistribution]:
    """day_first: from detect_date_format's inference (True for 'day_first',
    False for 'month_first'). None (ambiguous/mixed/no finding) skips the
    distribution rather than silently guessing a parse order.
    """
    if day_first is None:
        return None
    series = df[column]
    non_blank = series[~series.map(schema_mod.is_blank)].astype(str).str.strip()
    if non_blank.empty:
        return None
    parsed = pd.to_datetime(non_blank, dayfirst=day_first, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None
    per_day = valid.dt.date.value_counts().sort_index()
    return DateDistribution(
        column=column, min_date=str(valid.min().date()), max_date=str(valid.max().date()),
        records_per_day=[(str(d), int(c)) for d, c in per_day.items()],
    )


# ---------------------------------------------------------------------------
# Outcome exploration — only for a column the user explicitly selects
# ---------------------------------------------------------------------------

MAX_ORDINAL_LEVELS = 8


@dataclass(frozen=True)
class OutcomeExploration:
    column: str
    value_counts: list[tuple[str, int]]
    looks_ordinal: bool
    implied_funnel: Optional[list[str]]  # value_counts ordered by count descending — a heuristic, not verified semantics


def explore_outcome_column(df: pd.DataFrame, column: str) -> OutcomeExploration:
    non_blank = df[column][~df[column].map(schema_mod.is_blank)].astype(str)
    value_counts = non_blank.value_counts()
    looks_ordinal = 2 <= len(value_counts) <= MAX_ORDINAL_LEVELS
    return OutcomeExploration(
        column=column,
        value_counts=[(str(v), int(c)) for v, c in value_counts.items()],
        looks_ordinal=looks_ordinal,
        implied_funnel=list(value_counts.index) if looks_ordinal else None,
    )


# ---------------------------------------------------------------------------
# Full profile assembly + export
# ---------------------------------------------------------------------------


def build_profile(df: pd.DataFrame, *, raw_dataset_id: str, outcome_column: Optional[str] = None) -> dict[str, Any]:
    """Assemble the complete, JSON-serializable profile — the artifact that gets
    exported and handed to whoever does the offline cleaning.
    """
    overview = profile_overview(df)
    columns = profile_columns(df)
    categorical_cols = categorical_columns(columns)
    numeric_cols = numeric_columns(columns)

    date_findings = detect_date_formats(df)
    day_first_by_column = {
        f.column: (True if f.inference == "day_first" else False if f.inference == "month_first" else None)
        for f in date_findings
    }

    distributions: dict[str, Any] = {"numeric": [], "categorical": [], "date": []}
    for col in numeric_cols:
        d = numeric_distribution(df, col)
        if d is not None:
            distributions["numeric"].append(asdict(d))
    for col in categorical_cols:
        d = categorical_distribution(df, col)
        if d is not None:
            distributions["categorical"].append(asdict(d))
    for col, day_first in day_first_by_column.items():
        d = date_distribution(df, col, day_first)
        if d is not None:
            distributions["date"].append(asdict(d))

    outcome = explore_outcome_column(df, outcome_column) if outcome_column else None

    return {
        "raw_dataset_id": raw_dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": asdict(overview),
        "columns": [asdict(c) for c in columns],
        "findings": {
            "near_duplicate_categories": [asdict(f) for f in find_near_duplicate_categories(df, categorical_cols)],
            "sentinel_values": [asdict(f) for f in detect_sentinel_values(df, numeric_cols)],
            "date_formats": [asdict(f) for f in date_findings],
            "float_formatted_identifiers": [asdict(f) for f in detect_float_formatted_identifiers(df)],
            "multi_valued_cells": [asdict(f) for f in detect_multi_valued_cells(df, categorical_cols)],
            "free_text_fields": [asdict(f) for f in detect_free_text_fields(df, categorical_cols)],
        },
        "distributions": distributions,
        "outcome_exploration": asdict(outcome) if outcome else None,
    }


def _md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    header = "| " + " | ".join(columns) + " |\n" + "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |\n" for r in rows)
    return header + body


def render_markdown_report(profile: dict[str, Any]) -> str:
    """Human-readable cleaning worklist, structured to mirror the JSON profile."""
    lines: list[str] = []
    lines.append(f"# Raw data profile — {profile['raw_dataset_id']}")
    lines.append(f"_Generated {profile['generated_at']}_\n")

    ov = profile["overview"]
    lines.append("## Overview")
    lines.append(f"- Shape: {ov['row_count']:,} rows × {ov['col_count']} columns")
    lines.append(f"- Memory: {ov['memory_bytes'] / 1e6:.1f} MB")
    lines.append(f"- Fully duplicate rows: {ov['fully_duplicate_row_count']:,}")
    lines.append(f"- Fully empty columns: {', '.join(ov['fully_empty_columns']) or 'none'}\n")

    lines.append("## Per-column profile (dtype INFERRED for profiling only)")
    lines.append(
        _md_table(
            [
                {
                    "column": c["column"], "inferred_dtype": c["inferred_dtype"],
                    "null_%": f"{c['null_pct']:.1f}", "distinct": c["distinct_count"],
                    "samples": ", ".join(c["sample_values"][:3]),
                }
                for c in profile["columns"]
            ],
            ["column", "inferred_dtype", "null_%", "distinct", "samples"],
        )
    )

    findings = profile["findings"]

    lines.append("## Findings — cleaning worklist\n")

    lines.append("### Near-duplicate categories")
    for f in findings["near_duplicate_categories"]:
        variants = ", ".join(f"{v!r} ({c})" for v, c in f["variants"])
        conf = "high confidence" if f["kind"] == "case_or_whitespace" else f"possible, similarity={f['similarity']}"
        lines.append(f"- **{f['column']}** ({conf}): {variants} → suggested canonical: {f['canonical']!r}")
    if not findings["near_duplicate_categories"]:
        lines.append("_none found_")

    lines.append("\n### Sentinel / disguised-code values")
    for f in findings["sentinel_values"]:
        lines.append(f"- **{f['column']}**: {f['message']}")
    if not findings["sentinel_values"]:
        lines.append("_none found_")

    lines.append("\n### Ambiguous date formats")
    for f in findings["date_formats"]:
        lines.append(f"- **{f['column']}** → {f['inference']}: {f['evidence']}")
    if not findings["date_formats"]:
        lines.append("_no date-like columns detected_")

    lines.append("\n### Float-formatted identifiers")
    for f in findings["float_formatted_identifiers"]:
        lines.append(f"- **{f['column']}**: {f['match_fraction']:.0%} of values look like a float-formatted ID, e.g. {f['sample_values'][0]!r}")
    if not findings["float_formatted_identifiers"]:
        lines.append("_none found_")

    lines.append("\n### Multi-valued cells")
    for f in findings["multi_valued_cells"]:
        order_note = "order VARIES between rows" if f["order_varies"] else "order is consistent"
        lines.append(
            f"- **{f['column']}** (delimiter {f['delimiter']!r}): {f['distinct_component_count']} distinct "
            f"component(s) across {f['match_fraction']:.0%} of rows, {order_note}"
        )
    if not findings["multi_valued_cells"]:
        lines.append("_none found_")

    lines.append("\n### Free-text in categorical-looking fields")
    for f in findings["free_text_fields"]:
        lines.append(f"- **{f['column']}**: {f['distinct_count']}/{f['row_count']} distinct values, {f['singleton_ratio']:.0%} appear once")
    if not findings["free_text_fields"]:
        lines.append("_none found_")

    if profile.get("outcome_exploration"):
        oc = profile["outcome_exploration"]
        lines.append(f"\n## Outcome exploration — {oc['column']}")
        for v, c in oc["value_counts"]:
            lines.append(f"- {v}: {c:,}")
        if oc["looks_ordinal"]:
            lines.append(f"\nImplied funnel (by count, largest first — verify this matches real stage order): {' → '.join(oc['implied_funnel'])}")

    return "\n".join(lines) + "\n"
