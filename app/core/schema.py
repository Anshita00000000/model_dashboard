"""Dataset contract for data arriving at the modelling stage.

Datasets reaching this layer are already enriched and feature-extracted (see
CLAUDE.md — Phase 2 produces the enriched CSV, Phase 1 consumes it). This
module only describes and validates the *shape* of that data; it does not
enrich or feature-extract anything.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import pandas as pd

Purpose = Literal["train", "predict"]
Role = Literal["id", "target", "feature", "metadata", "excluded"]

_VALID_ROLES = {"id", "target", "feature", "metadata", "excluded"}
_VALID_PURPOSES = {"train", "predict"}
_VALID_DTYPES = {"numeric", "categorical"}

# A column is numeric if >=90% of its non-null, non-empty values parse as numbers.
NUMERIC_PARSE_THRESHOLD = 0.90


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str  # "numeric" | "categorical"
    nullable: bool
    role: Role

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"column {self.name!r}: invalid role {self.role!r}, must be one of {sorted(_VALID_ROLES)}")
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"column {self.name!r}: invalid dtype {self.dtype!r}, must be one of {sorted(_VALID_DTYPES)}")


@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    merchant_name: str
    purpose: Purpose
    source_file: str
    row_count: int
    column_count: int
    columns: list[ColumnSchema]
    created_at: str  # UTC ISO-8601 timestamp
    content_hash: str  # SHA256 of the dataframe bytes

    def __post_init__(self) -> None:
        if self.purpose not in _VALID_PURPOSES:
            raise ValueError(f"invalid purpose {self.purpose!r}, must be one of {sorted(_VALID_PURPOSES)}")
        if self.column_count != len(self.columns):
            raise ValueError(f"column_count ({self.column_count}) does not match len(columns) ({len(self.columns)})")

    def column(self, name: str) -> ColumnSchema:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"no column {name!r} in schema {self.dataset_id!r}")

    def columns_with_role(self, role: Role) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSchema":
        cols = [ColumnSchema(**c) for c in d["columns"]]
        return cls(**{**d, "columns": cols})


@dataclass(frozen=True)
class SchemaViolation:
    column: str
    kind: str  # "missing_column" | "unexpected_column" | "dtype_mismatch" | "null_violation"
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.column}: {self.detail}"


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA256 of the dataframe bytes — proves which exact data a model saw."""
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _infer_dtype(series: pd.Series) -> str:
    non_blank = series[~series.map(_is_blank)]
    if len(non_blank) == 0:
        # An all-null/all-empty column carries no numeric evidence; treat as categorical
        # rather than silently guessing, matching this codebase's "fail loudly" bias
        # (an empty numeric column would otherwise be indistinguishable from a bug).
        return "categorical"
    parsed = pd.to_numeric(non_blank, errors="coerce")
    numeric_ratio = parsed.notna().mean()
    return "numeric" if numeric_ratio >= NUMERIC_PARSE_THRESHOLD else "categorical"


def classify_dtype(series: pd.Series) -> str:
    """Public entry point for the numeric/categorical rule (>=90% parse rate) used
    across app/core/ — e.g. app/core/features.py reuses this rather than
    re-implementing it, so the rule only lives in one place.
    """
    return _infer_dtype(series)


def is_blank(value: Any) -> bool:
    """Public entry point for the blank-value rule (None, NaN, or an
    empty/whitespace-only string) used across app/core/ — e.g.
    app/core/predict.py's null-rate drift check reuses this.
    """
    return _is_blank(value)


def blank_rate(series: pd.Series) -> float:
    """Fraction of values in series that are blank. NaN if series is empty."""
    if len(series) == 0:
        return float("nan")
    return float(series.map(_is_blank).mean())


def _infer_nullable(series: pd.Series) -> bool:
    return bool(series.map(_is_blank).any())


def infer_schema(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    merchant_name: str,
    purpose: Purpose,
    source_file: str,
    overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> DatasetSchema:
    """Infer a DatasetSchema from a dataframe.

    overrides: optional {column_name: {"dtype": ..., "nullable": ..., "role": ...}}.
    Anything not present in overrides is inferred; role defaults to "feature"
    when not overridden (the caller is expected to mark id/target/metadata/excluded
    columns explicitly — nothing about a column name implies its role).
    """
    overrides = overrides or {}
    columns: list[ColumnSchema] = []
    for name in df.columns:
        col_override = overrides.get(name, {})
        series = df[name]
        dtype = col_override.get("dtype") or _infer_dtype(series)
        nullable = col_override.get("nullable")
        if nullable is None:
            nullable = _infer_nullable(series)
        role = col_override.get("role", "feature")
        columns.append(ColumnSchema(name=name, dtype=dtype, nullable=nullable, role=role))

    return DatasetSchema(
        dataset_id=dataset_id,
        merchant_name=merchant_name,
        purpose=purpose,
        source_file=source_file,
        row_count=len(df),
        column_count=len(columns),
        columns=columns,
        created_at=datetime.now(timezone.utc).isoformat(),
        content_hash=hash_dataframe(df),
    )


def validate(df: pd.DataFrame, schema: DatasetSchema) -> list[SchemaViolation]:
    """Check a dataframe against a previously-established schema.

    This is a drift check: e.g. validating a fresh prediction batch against the
    schema pinned at training time. Excluded columns are not required to be
    present. Returns an empty list when the dataframe conforms.
    """
    violations: list[SchemaViolation] = []
    df_columns = set(df.columns)
    schema_columns = {c.name for c in schema.columns}

    for col in schema.columns:
        if col.role == "excluded":
            continue
        if col.name not in df_columns:
            violations.append(SchemaViolation(col.name, "missing_column", "required by schema but absent from dataframe"))
            continue

        series = df[col.name]
        actual_dtype = _infer_dtype(series)
        if actual_dtype != col.dtype:
            violations.append(
                SchemaViolation(col.name, "dtype_mismatch", f"schema expects {col.dtype!r}, observed {actual_dtype!r}")
            )

        if not col.nullable and _infer_nullable(series):
            violations.append(
                SchemaViolation(col.name, "null_violation", "schema marks column non-nullable but nulls/blanks are present")
            )

    for name in sorted(df_columns - schema_columns):
        violations.append(SchemaViolation(name, "unexpected_column", "present in dataframe but not declared in schema"))

    return violations
