"""Canonical field mapping — the cross-merchant schema layer.

Purpose: consistency ACROSS MERCHANTS, so everything downstream of this module
is merchant-agnostic (see CLAUDE.md, "The platform must be merchant-agnostic").
This module never guesses a merchant's schema; it only offers ranked
suggestions that a human confirms or overrides, and coalesces per an explicit,
user-chosen precedence.

The canonical field LIST is configuration, not code — see CANONICAL_FIELD_SEEDS
and seed_canonical_fields()/add_canonical_field(). It is seeded, not hardcoded
into validation logic, so a new field can be added from the UI without a
deploy.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from . import schema, storage

# ---------------------------------------------------------------------------
# Canonical field configuration (seeded, not hardcoded into logic)
# ---------------------------------------------------------------------------

REQUIRED = "required"
REQUIRED_FOR_TRAIN = "required_for_train"
OPTIONAL = "optional"
_VALID_REQUIRED_LEVELS = {REQUIRED, REQUIRED_FOR_TRAIN, OPTIONAL}

CANONICAL_FIELD_SEEDS: list[dict[str, Any]] = [
    {"name": "lead_id", "dtype": "id", "required_level": REQUIRED, "unique_required": True,
     "description": "Unique identifier for the lead."},
    {"name": "phone", "dtype": "text", "required_level": REQUIRED, "unique_required": False,
     "description": "Enrichment join key. Normalised to the last 10 digits."},
    {"name": "created_at", "dtype": "date", "required_level": REQUIRED, "unique_required": False,
     "description": "When the lead was created."},
    {"name": "merchant", "dtype": "categorical", "required_level": REQUIRED, "unique_required": False,
     "description": "Clinic chain this lead belongs to."},
    {"name": "disposition", "dtype": "categorical", "required_level": REQUIRED_FOR_TRAIN, "unique_required": False,
     "description": "Funnel stage/outcome reached. Required for purpose=train."},
    {"name": "full_name", "dtype": "text", "required_level": OPTIONAL, "unique_required": False, "description": ""},
    {"name": "clinic_location", "dtype": "categorical", "required_level": OPTIONAL, "unique_required": False, "description": ""},
    {"name": "treatment_type", "dtype": "categorical", "required_level": OPTIONAL, "unique_required": False, "description": ""},
    {"name": "lead_source", "dtype": "categorical", "required_level": OPTIONAL, "unique_required": False, "description": ""},
    {"name": "gender", "dtype": "categorical", "required_level": OPTIONAL, "unique_required": False, "description": ""},
    {"name": "age", "dtype": "numeric", "required_level": OPTIONAL, "unique_required": False, "description": ""},
]


def seed_canonical_fields(store: storage.MetadataStore) -> None:
    """Idempotent: only inserts seeds not already present by name."""
    existing = {f["name"] for f in store.list_canonical_fields()}
    for spec in CANONICAL_FIELD_SEEDS:
        if spec["name"] in existing:
            continue
        store.insert_canonical_field(field_id=uuid.uuid4().hex, **spec)


def add_canonical_field(
    store: storage.MetadataStore,
    *,
    name: str,
    dtype: str,
    required_level: str = OPTIONAL,
    unique_required: bool = False,
    description: str = "",
) -> str:
    """Adds a new canonical field. Idempotent by name — re-adding a field that
    already exists returns its existing field_id rather than erroring, since
    the mapping UI may call this speculatively while a user types.
    """
    name = name.strip()
    if not name:
        raise ValueError("canonical field name must not be empty")
    if required_level not in _VALID_REQUIRED_LEVELS:
        raise ValueError(f"invalid required_level {required_level!r}, must be one of {sorted(_VALID_REQUIRED_LEVELS)}")
    existing = store.get_canonical_field_by_name(name)
    if existing:
        return existing["field_id"]
    field_id = uuid.uuid4().hex
    store.insert_canonical_field(
        field_id=field_id, name=name, dtype=dtype, required_level=required_level,
        unique_required=unique_required, description=description,
    )
    return field_id


def list_canonical_fields(store: storage.MetadataStore) -> list[dict]:
    return store.list_canonical_fields()


# ---------------------------------------------------------------------------
# Phone normalisation — the enrichment join key, get this exactly right
# ---------------------------------------------------------------------------

_FLOAT_SUFFIX = ".0"
_NON_DIGIT_RE = re.compile(r"[^0-9]")


def normalize_phone(value: Any) -> Optional[str]:
    """Strip a float-formatted artifact BEFORE extracting digits.

    Clinic files often store phone as a float (7889358744.0 instead of the
    string "7889358744"). Extracting digits first treats the trailing zero as
    a real digit and slices off the leading one when truncating to 10 —
    this produced 1 match instead of 16,073 once (see CLAUDE.md). The order
    here is not incidental: strip ".0" first, extract digits second.
    """
    if schema.is_blank(value):
        return None
    s = str(value).strip()
    if s.endswith(_FLOAT_SUFFIX):
        s = s[: -len(_FLOAT_SUFFIX)]
    digits = _NON_DIGIT_RE.sub("", s)
    return digits[-10:] if len(digits) >= 10 else None


# ---------------------------------------------------------------------------
# Mapping suggestion — ranked, never auto-applied
# ---------------------------------------------------------------------------

# Common CRM-field synonyms per canonical field. These only raise a
# suggestion's confidence; they never bypass the user's confirm/override step.
_ALIASES: dict[str, list[str]] = {
    "lead_id": ["lead_id", "leadid", "id", "enquiry_id", "lead_no", "lead_number"],
    "phone": ["phone", "mobile", "contact_number", "phone_number", "mobile_number", "contact", "mobile_no"],
    "created_at": ["created_at", "lead_created", "enquiry_date", "created_date", "date_created", "lead_date", "created"],
    "merchant": ["merchant", "clinic", "clinic_name", "brand"],
    "disposition": ["disposition", "status", "stage", "funnel_stage", "outcome", "lead_status"],
    "full_name": ["full_name", "name", "patient_name", "lead_name", "customer_name"],
    "clinic_location": ["clinic_location", "location", "city", "branch"],
    "treatment_type": ["treatment_type", "treatment", "procedure", "service"],
    "lead_source": ["lead_source", "source", "utm_source", "channel"],
    "gender": ["gender", "sex"],
    "age": ["age", "patient_age"],
}

ALIAS_CONFIDENCE = 0.95
FUZZY_MIN_CONFIDENCE = 0.55


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class MappingSuggestion:
    source_column: str
    confidence: float


def suggest_mapping(
    source_columns: list[str], canonical_fields: list[dict]
) -> dict[str, list[MappingSuggestion]]:
    """For each canonical field, rank source columns by name-similarity, best
    first. Report-only: the mapping UI must show every suggestion for the
    user to confirm or override — nothing here is applied automatically.
    """
    normalized_sources = {col: _normalize_name(col) for col in source_columns}
    suggestions: dict[str, list[MappingSuggestion]] = {}
    for cfield in canonical_fields:
        field_name = cfield["name"]
        aliases = {_normalize_name(a) for a in _ALIASES.get(field_name, [])}
        aliases.add(_normalize_name(field_name))
        scored: list[MappingSuggestion] = []
        for col, norm_col in normalized_sources.items():
            if norm_col in aliases:
                scored.append(MappingSuggestion(col, ALIAS_CONFIDENCE))
                continue
            ratio = max(
                (difflib.SequenceMatcher(None, norm_col, alias).ratio() for alias in aliases),
                default=0.0,
            )
            if ratio >= FUZZY_MIN_CONFIDENCE:
                scored.append(MappingSuggestion(col, round(ratio, 3)))
        scored.sort(key=lambda s: s.confidence, reverse=True)
        suggestions[field_name] = scored
    return suggestions


# ---------------------------------------------------------------------------
# Applying a confirmed mapping
# ---------------------------------------------------------------------------

EXTRA_PREFIX = "extra__"

# mapping: {canonical_field_name: [source_col_in_precedence_order, ...]}
MappingSpec = dict[str, list[str]]


def apply_mapping(df: pd.DataFrame, mapping: MappingSpec) -> pd.DataFrame:
    """Many-to-one coalesces left-to-right per the user's chosen precedence —
    first non-null wins. Every source column not referenced by any canonical
    field is preserved verbatim under an `extra__` prefix so it stays
    available to Tab 4 feature extraction later. Nothing is ever dropped.
    """
    mapped_source_cols = {col for cols in mapping.values() for col in cols}
    out = pd.DataFrame(index=df.index)
    for field_name, source_cols in mapping.items():
        present = [c for c in source_cols if c in df.columns]
        if not present:
            continue
        coalesced = df[present[0]].copy()
        for c in present[1:]:
            still_blank = coalesced.map(schema.is_blank)
            coalesced = coalesced.where(~still_blank, df[c])
        out[field_name] = coalesced
    for col in df.columns:
        if col not in mapped_source_cols:
            out[f"{EXTRA_PREFIX}{col}"] = df[col]
    return out


# ---------------------------------------------------------------------------
# Named, versioned mapping profiles
# ---------------------------------------------------------------------------


def save_field_mapping(
    store: storage.MetadataStore,
    *,
    merchant: str,
    name: str,
    mapping: MappingSpec,
    unmapped_columns: list[str],
    created_by: str,
) -> str:
    """Append-only: every save is a new version, never an overwrite."""
    prior = store.list_field_mappings(merchant=merchant)
    version = max((m["version"] for m in prior), default=0) + 1
    mapping_id = uuid.uuid4().hex
    store.insert_field_mapping(
        mapping_id=mapping_id, merchant=merchant, name=name, version=version,
        mapping_json=json.dumps(mapping), unmapped_columns_json=json.dumps(unmapped_columns),
        created_by=created_by,
    )
    return mapping_id


def get_latest_field_mapping(store: storage.MetadataStore, merchant: str) -> Optional[dict]:
    """Most recent saved mapping profile for this merchant, to pre-fill the UI."""
    rows = store.list_field_mappings(merchant=merchant)
    return rows[0] if rows else None


def load_mapping_dict(row: dict) -> MappingSpec:
    return json.loads(row["mapping_json"])


# ---------------------------------------------------------------------------
# Lineage: column diff between a raw dataset and its cleaned successor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnDiff:
    added: list[str]
    removed: list[str]
    renamed: list[tuple[str, str]]  # (old_name, new_name) — heuristic, informational only
    unchanged: list[str]


def column_diff(raw_columns: list[str], cleaned_columns: list[str]) -> ColumnDiff:
    """Purely informational — never blocks or drives anything automatically.
    A "rename" is a heuristic guess (removed name and added name normalise to
    the same string); it is shown to a human, not acted on.
    """
    raw_set, cleaned_set = set(raw_columns), set(cleaned_columns)
    added = sorted(cleaned_set - raw_set)
    removed = sorted(raw_set - cleaned_set)

    removed_by_norm: dict[str, list[str]] = {}
    for c in removed:
        removed_by_norm.setdefault(_normalize_name(c), []).append(c)

    renamed: list[tuple[str, str]] = []
    consumed_removed: set[str] = set()
    consumed_added: set[str] = set()
    for c in added:
        candidates = [r for r in removed_by_norm.get(_normalize_name(c), []) if r not in consumed_removed]
        if candidates:
            old = candidates[0]
            renamed.append((old, c))
            consumed_removed.add(old)
            consumed_added.add(c)

    return ColumnDiff(
        added=[c for c in added if c not in consumed_added],
        removed=[c for c in removed if c not in consumed_removed],
        renamed=renamed,
        unchanged=sorted(raw_set & cleaned_set),
    )


# ---------------------------------------------------------------------------
# Validation before proceeding to enrichment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    kind: str
    field_name: Optional[str]
    message: str
    severity: str  # "error" | "warning"


@dataclass(frozen=True)
class ValidationReport:
    issues: list[ValidationIssue]
    phone_normalized_count: int
    phone_failed_count: int
    phone_failed_samples: list[str]
    phone_match_rate: float  # NaN if no phone column / no non-blank values
    created_at_parsed_count: int
    created_at_unparsed_count: int
    created_at_inferred_format: Optional[str]
    disposition_value_counts: list[tuple[str, int]]

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


_PHONE_FAILED_SAMPLE_LIMIT = 10


def validate_canonical(
    df: pd.DataFrame,
    canonical_fields: list[dict],
    *,
    purpose: str,
    min_phone_match_rate: float = 0.90,
    created_at_ambiguous: bool = False,
    created_at_day_first: Optional[bool] = None,
) -> ValidationReport:
    """created_at_ambiguous should be passed as True when Tab 2's raw-EDA
    profile flagged this column's date format as ambiguous — in that case a
    parse is only attempted once created_at_day_first has been explicitly
    confirmed by the user (never guessed here).
    """
    issues: list[ValidationIssue] = []

    for cfield in canonical_fields:
        level = cfield["required_level"]
        needed = level == REQUIRED or (level == REQUIRED_FOR_TRAIN and purpose == "train")
        if needed and cfield["name"] not in df.columns:
            issues.append(ValidationIssue(
                "missing_required_field", cfield["name"],
                f"required canonical field {cfield['name']!r} is not mapped", "error",
            ))
        if cfield.get("unique_required") and cfield["name"] in df.columns:
            series = df[cfield["name"]]
            non_blank = series[~series.map(schema.is_blank)]
            dupes = non_blank[non_blank.duplicated(keep=False)]
            if len(dupes) > 0:
                issues.append(ValidationIssue(
                    "duplicate_value", cfield["name"],
                    f"{dupes.nunique()} {cfield['name']} value(s) appear more than once ({len(dupes)} rows total)",
                    "error",
                ))

    phone_normalized_count = phone_failed_count = 0
    phone_failed_samples: list[str] = []
    phone_match_rate = float("nan")
    if "phone" in df.columns:
        raw_phone = df["phone"]
        non_blank_mask = ~raw_phone.map(schema.is_blank)
        non_blank_count = int(non_blank_mask.sum())
        normalized = raw_phone.map(normalize_phone)
        phone_normalized_count = int(normalized.notna().sum())
        phone_failed_count = non_blank_count - phone_normalized_count
        if non_blank_count > 0:
            phone_match_rate = phone_normalized_count / non_blank_count
        failed_mask = non_blank_mask & normalized.isna()
        phone_failed_samples = raw_phone[failed_mask].astype(str).head(_PHONE_FAILED_SAMPLE_LIMIT).tolist()
        if non_blank_count > 0 and phone_match_rate < min_phone_match_rate:
            issues.append(ValidationIssue(
                "low_phone_match_rate", "phone",
                f"only {phone_match_rate:.1%} of non-blank phone values normalised successfully "
                f"(threshold {min_phone_match_rate:.0%}); {phone_failed_count} failure(s)",
                "error",
            ))

    created_at_parsed_count = created_at_unparsed_count = 0
    created_at_inferred_format: Optional[str] = None
    if "created_at" in df.columns:
        if created_at_ambiguous and created_at_day_first is None:
            issues.append(ValidationIssue(
                "ambiguous_date_needs_confirmation", "created_at",
                "Raw EDA flagged this column's date format as ambiguous (day-first vs month-first) — "
                "confirm the format before proceeding.",
                "error",
            ))
        else:
            day_first = bool(created_at_day_first) if created_at_day_first is not None else False
            series = df["created_at"]
            non_blank_mask = ~series.map(schema.is_blank)
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=day_first)
            created_at_parsed_count = int((parsed.notna() & non_blank_mask).sum())
            created_at_unparsed_count = int((parsed.isna() & non_blank_mask).sum())
            created_at_inferred_format = "day-first" if day_first else "month-first / ISO"
            if created_at_unparsed_count > 0:
                issues.append(ValidationIssue(
                    "unparsed_created_at", "created_at",
                    f"{created_at_unparsed_count} non-blank created_at value(s) failed to parse as a date",
                    "warning",
                ))

    disposition_value_counts: list[tuple[str, int]] = []
    if "disposition" in df.columns:
        vc = df["disposition"].value_counts(dropna=True)
        disposition_value_counts = [(str(k), int(v)) for k, v in vc.items()]

    return ValidationReport(
        issues=issues,
        phone_normalized_count=phone_normalized_count,
        phone_failed_count=phone_failed_count,
        phone_failed_samples=phone_failed_samples,
        phone_match_rate=phone_match_rate,
        created_at_parsed_count=created_at_parsed_count,
        created_at_unparsed_count=created_at_unparsed_count,
        created_at_inferred_format=created_at_inferred_format,
        disposition_value_counts=disposition_value_counts,
    )
