"""Raw file ingestion.

SCOPE BOUNDARY — read this before touching this file: this layer applies NO
canonical schema, NO field renaming, NO normalisation, and NO cleaning. The
modelling schema is still being explored and will change. Uploaded files are
stored EXACTLY as received and profiled as-is. Cleaning happens offline by
the team; field mapping happens later, at Tab 3 (Phase 2, not built yet).

The only intelligence here is reading the file CORRECTLY — detecting its
delimiter and encoding, and reading every value as a string so nothing is
silently coerced — not changing what it contains. Do not add dtype coercion,
column renaming, or row-dropping to this module.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import csv
import hashlib
import io
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from . import storage

SUPPORTED_EXTENSIONS = ("csv", "tsv", "txt", "xlsx", "xls")
DELIMITED_EXTENSIONS = ("csv", "tsv", "txt")
EXCEL_EXTENSIONS = ("xlsx", "xls")

CANDIDATE_DELIMITERS: dict[str, str] = {",": "comma", "|": "pipe", "\t": "tab", ";": "semicolon"}

_UTF8_BOM = b"\xef\xbb\xbf"
_MALFORMED_ROW_EXAMPLES_LIMIT = 10
_DELIMITER_SAMPLE_LINES = 500


def new_raw_dataset_id() -> str:
    return uuid.uuid4().hex


def content_hash_of(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------


def detect_encoding(raw_bytes: bytes) -> tuple[str, str]:
    """Try utf-8, then utf-8-sig (BOM — common in bureau exports), then latin-1.

    A leading BOM is checked for explicitly rather than relying on try/except
    alone: plain "utf-8" decoding SUCCEEDS on BOM'd bytes (the BOM is valid
    UTF-8), it just leaves a stray U+FEFF character glued onto the first
    column's name — so a naive try-in-order approach would never reach the
    utf-8-sig branch for the exact files it exists to handle.

    Returns (encoding_used, decoded_text). latin-1 never raises (every byte
    maps to a code point), so this always succeeds.
    """
    if raw_bytes.startswith(_UTF8_BOM):
        return "utf-8-sig", raw_bytes.decode("utf-8-sig")
    try:
        return "utf-8", raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return "utf-8-sig", raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    return "latin-1", raw_bytes.decode("latin-1")


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelimiterDetection:
    delimiter: str
    label: str
    consistency: float  # fraction of sampled rows matching the most common field count
    most_common_field_count: int


def _field_counts(text: str, delimiter: str, sample_lines: int) -> list[int]:
    lines = text.splitlines()[:sample_lines]
    reader = csv.reader(lines, delimiter=delimiter)
    return [len(row) for row in reader if row]


def detect_delimiter(text: str, sample_lines: int = _DELIMITER_SAMPLE_LINES) -> DelimiterDetection:
    """Try each candidate delimiter; pick the one giving the most consistent
    field count across sampled rows. A delimiter that never appears in the
    text trivially gives a "perfectly consistent" count of 1 field per row —
    that degenerate case is only used as a last resort, so a real multi-column
    delimiter is preferred whenever one produces a consistent split.
    """
    candidates: list[DelimiterDetection] = []
    for delim, label in CANDIDATE_DELIMITERS.items():
        counts = _field_counts(text, delim, sample_lines)
        if not counts:
            continue
        mode_count, mode_freq = Counter(counts).most_common(1)[0]
        candidates.append(DelimiterDetection(delim, label, mode_freq / len(counts), mode_count))

    if not candidates:
        raise ValueError("file has no non-empty lines to detect a delimiter from")

    qualifying = [c for c in candidates if c.most_common_field_count > 1]
    pool = qualifying or candidates
    pool.sort(key=lambda c: (c.consistency, c.most_common_field_count), reverse=True)
    return pool[0]


# ---------------------------------------------------------------------------
# Reading (as strings — profiling dtypes are inferred separately, never applied)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    df: pd.DataFrame  # every value read as a string (or NaN for blanks); never coerced
    original_bytes: bytes
    original_filename: str
    file_type: str  # "csv" | "tsv" | "txt" | "xlsx" | "xls"
    delimiter: Optional[str]  # None for excel
    delimiter_label: Optional[str]
    delimiter_consistency: Optional[float]
    encoding: str
    sheet_name: Optional[str]  # None for delimited files
    available_sheets: list[str] = field(default_factory=list)  # empty for delimited files
    malformed_row_count: int = 0
    malformed_row_examples: list[str] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0


def _read_delimited_text(text: str, delimiter: str) -> tuple[pd.DataFrame, int, list[str]]:
    all_counts = _field_counts(text, delimiter, sample_lines=10**9)
    if not all_counts:
        raise ValueError("file has no data rows")
    header_count = all_counts[0]
    data_counts = all_counts[1:]

    raw_lines = text.splitlines()
    malformed_line_numbers = [i + 2 for i, c in enumerate(data_counts) if c != header_count]  # 1-indexed, +1 for header
    examples = [raw_lines[n - 1] for n in malformed_line_numbers[:_MALFORMED_ROW_EXAMPLES_LIMIT] if n - 1 < len(raw_lines)]

    # Rows with too MANY fields can't be represented in a rectangular frame and are
    # skipped by pandas (already counted above as malformed); rows with too FEW
    # fields are kept and NaN-padded by pandas — also already counted above, so
    # nothing here is silently dropped without being reported.
    df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, engine="python", on_bad_lines="skip")
    return df, len(malformed_line_numbers), examples


def list_excel_sheets(raw_bytes: bytes) -> list[str]:
    """Cheap: reads workbook structure only, not cell data."""
    with pd.ExcelFile(io.BytesIO(raw_bytes)) as excel_file:
        return list(excel_file.sheet_names)


def load_raw_file(raw_bytes: bytes, original_filename: str, *, sheet_name: Optional[str] = None) -> IngestResult:
    """Read a file exactly as uploaded. Never renames columns, coerces values,
    or drops rows without counting and reporting them.
    """
    ext = Path(original_filename).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported file type {ext!r} for {original_filename!r}; "
            f"expected one of {SUPPORTED_EXTENSIONS}"
        )

    if ext in DELIMITED_EXTENSIONS:
        encoding, text = detect_encoding(raw_bytes)
        detection = detect_delimiter(text)
        df, malformed_count, examples = _read_delimited_text(text, detection.delimiter)
        return IngestResult(
            df=df,
            original_bytes=raw_bytes,
            original_filename=original_filename,
            file_type=ext,
            delimiter=detection.delimiter,
            delimiter_label=detection.label,
            delimiter_consistency=detection.consistency,
            encoding=encoding,
            sheet_name=None,
            available_sheets=[],
            malformed_row_count=malformed_count,
            malformed_row_examples=examples,
            row_count=len(df),
            col_count=len(df.columns),
        )

    # Excel: binary format, no delimiter/text-encoding concept.
    sheets = list_excel_sheets(raw_bytes)
    if not sheets:
        raise ValueError(f"{original_filename!r} has no sheets")
    chosen_sheet = sheet_name if sheet_name in sheets else sheets[0]
    df = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=chosen_sheet, dtype=str)
    return IngestResult(
        df=df,
        original_bytes=raw_bytes,
        original_filename=original_filename,
        file_type=ext,
        delimiter=None,
        delimiter_label=None,
        delimiter_consistency=None,
        encoding="n/a (binary format)",
        sheet_name=chosen_sheet,
        available_sheets=sheets,
        malformed_row_count=0,
        malformed_row_examples=[],
        row_count=len(df),
        col_count=len(df.columns),
    )


# ---------------------------------------------------------------------------
# Immutable storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaveResult:
    raw_dataset_id: str
    stored_path: Path
    parquet_path: Path
    content_hash: str
    duplicate_of: list[dict]  # prior raw_datasets rows with identical bytes; empty if none


def save_raw_dataset(
    *,
    store: storage.MetadataStore,
    root: Path | str,
    merchant: str,
    purpose: str,
    result: IngestResult,
    uploaded_by: str,
    notes: str = "",
) -> SaveResult:
    """Write the original bytes byte-for-byte, a parquet copy for fast
    profiling, and record the append-only audit row. Re-uploading identical
    bytes is detected via content_hash and reported back for the caller to
    warn about — it still gets a new row (append-only, never deduplicated
    away).
    """
    raw_dataset_id = new_raw_dataset_id()
    content_hash = content_hash_of(result.original_bytes)
    duplicate_of = store.find_raw_datasets_by_content_hash(content_hash)

    stored_path = storage.timestamped_raw_file_path(root, merchant, result.original_filename)
    storage.write_bytes_atomic(stored_path, result.original_bytes)

    parquet_path = storage.save_dataframe(result.df, "raw", merchant, f"{raw_dataset_id}_profile", root=root)

    store.insert_raw_dataset(
        raw_dataset_id=raw_dataset_id,
        merchant=merchant,
        purpose=purpose,
        original_filename=result.original_filename,
        stored_path=str(stored_path),
        parquet_path=str(parquet_path),
        delimiter=result.delimiter,
        encoding=result.encoding,
        sheet_name=result.sheet_name,
        row_count=result.row_count,
        col_count=result.col_count,
        content_hash=content_hash,
        uploaded_by=uploaded_by,
        notes=notes,
    )

    return SaveResult(
        raw_dataset_id=raw_dataset_id,
        stored_path=stored_path,
        parquet_path=parquet_path,
        content_hash=content_hash,
        duplicate_of=duplicate_of,
    )


@dataclass(frozen=True)
class CleanedSaveResult:
    cleaned_dataset_id: str
    stored_path: Path
    parquet_path: Path
    content_hash: str
    duplicate_of: list[dict]  # prior cleaned_datasets rows with identical bytes; empty if none


def save_cleaned_dataset(
    *,
    store: storage.MetadataStore,
    root: Path | str,
    merchant: str,
    purpose: str,
    result: IngestResult,
    uploaded_by: str,
    source_raw_dataset_id: Optional[str] = None,
    notes: str = "",
) -> CleanedSaveResult:
    """Same robust loader, same byte-exact-plus-parquet storage pattern as
    save_raw_dataset — this is the second upload point (Tab 3): a file the
    team has already cleaned offline. Still no schema is applied here; that
    happens later, in app.core.canonical's field mapping.
    """
    cleaned_dataset_id = new_raw_dataset_id()
    content_hash = content_hash_of(result.original_bytes)
    duplicate_of = store.find_cleaned_datasets_by_content_hash(content_hash)

    stored_path = storage.timestamped_cleaned_file_path(root, merchant, result.original_filename)
    storage.write_bytes_atomic(stored_path, result.original_bytes)

    parquet_path = storage.save_dataframe(result.df, "cleaned", merchant, f"{cleaned_dataset_id}_profile", root=root)

    store.insert_cleaned_dataset(
        cleaned_dataset_id=cleaned_dataset_id,
        merchant=merchant,
        purpose=purpose,
        original_filename=result.original_filename,
        stored_path=str(stored_path),
        parquet_path=str(parquet_path),
        delimiter=result.delimiter,
        encoding=result.encoding,
        sheet_name=result.sheet_name,
        row_count=result.row_count,
        col_count=result.col_count,
        content_hash=content_hash,
        source_raw_dataset_id=source_raw_dataset_id,
        uploaded_by=uploaded_by,
        notes=notes,
    )

    return CleanedSaveResult(
        cleaned_dataset_id=cleaned_dataset_id,
        stored_path=stored_path,
        parquet_path=parquet_path,
        content_hash=content_hash,
        duplicate_of=duplicate_of,
    )
