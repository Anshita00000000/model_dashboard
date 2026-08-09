"""Storage abstraction.

Two responsibilities, kept in one file today so that swapping either backend
later (filesystem -> S3, SQLite -> Postgres) is a single-file change:

1. Artifact storage: dataframes and JSON blobs written to timestamped,
   never-overwritten paths under a storage root.
2. Metadata storage: an append-only SQLite audit trail (datasets, splits,
   model_bundles, prediction_runs) that answers "which model, trained on
   which data, produced this score, and when".

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

ArtifactKind = Literal["dataset", "split", "prediction", "bundle"]

DEFAULT_STORAGE_ROOT = Path("data")
DEFAULT_DB_PATH = DEFAULT_STORAGE_ROOT / "meta.db"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_segment(value: str) -> str:
    """Filesystem-safe path segment. Fails loudly rather than silently mangling."""
    if not value or not value.strip():
        raise ValueError("path segment must not be empty")
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value.strip())


# ---------------------------------------------------------------------------
# Path construction (append-only: never returns a path that already exists)
# ---------------------------------------------------------------------------


def _timestamped_candidate(directory: Path, stamp: str, stem: str, ext: Optional[str]) -> Path:
    """Shared collision-avoidance: append a numeric suffix if the exact path is
    already taken (two writes in the same second), so storage is genuinely
    append-only. ext=None leaves the stem as the whole filename (no dot added) —
    used when the name itself must be preserved verbatim, e.g. an original
    uploaded filename that already carries its own extension.
    """
    suffix = f".{ext}" if ext else ""
    candidate = directory / f"{stamp}_{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stamp}_{stem}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def timestamped_path(root: Path, kind: str, merchant: str, name: str, ext: str) -> Path:
    """{root}/{kind}/{merchant}/{YYYYMMDD_HHMMSS}_{name}.{ext}"""
    directory = Path(root) / kind / _safe_segment(merchant)
    return _timestamped_candidate(directory, _utcnow_stamp(), _safe_segment(name), ext)


def unique_dir_path(root: Path, kind: str, merchant: str, name: str) -> Path:
    """Like timestamped_path but for a directory (used for model bundles)."""
    directory = Path(root) / kind / _safe_segment(merchant)
    return _timestamped_candidate(directory, _utcnow_stamp(), _safe_segment(name), None)


def timestamped_raw_file_path(root: Path | str, merchant: str, original_filename: str) -> Path:
    """{root}/datasets/raw/{merchant}/{YYYYMMDD_HHMMSS}_{original_filename}

    The original filename (extension included) is preserved as a single unit —
    this is the byte-exact record of an uploaded file, so it keeps its own name.
    """
    directory = Path(root) / "datasets" / "raw" / _safe_segment(merchant)
    return _timestamped_candidate(directory, _utcnow_stamp(), _safe_segment(original_filename), None)


def timestamped_cleaned_file_path(root: Path | str, merchant: str, original_filename: str) -> Path:
    """{root}/datasets/cleaned/{merchant}/{YYYYMMDD_HHMMSS}_{original_filename}"""
    directory = Path(root) / "datasets" / "cleaned" / _safe_segment(merchant)
    return _timestamped_candidate(directory, _utcnow_stamp(), _safe_segment(original_filename), None)


# ---------------------------------------------------------------------------
# Low-level atomic writers (exact path, used for files inside a bundle dir)
# ---------------------------------------------------------------------------


def write_bytes_atomic(path: Path | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def write_json_atomic(path: Path | str, obj: Any) -> None:
    write_bytes_atomic(path, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


# ---------------------------------------------------------------------------
# Artifact storage: dataframes and JSON, timestamped, never overwritten
# ---------------------------------------------------------------------------


def save_dataframe(
    df: pd.DataFrame,
    kind: ArtifactKind,
    merchant: str,
    name: str,
    root: Path | str = DEFAULT_STORAGE_ROOT,
) -> Path:
    """Persist df as parquet at a new timestamped path. Never overwrites."""
    path = timestamped_path(Path(root), kind, merchant, name, "parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


def load_dataframe(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_json(
    obj: Any,
    kind: ArtifactKind,
    merchant: str,
    name: str,
    root: Path | str = DEFAULT_STORAGE_ROOT,
) -> Path:
    """Persist obj as a new timestamped JSON artifact. Never overwrites."""
    path = timestamped_path(Path(root), kind, merchant, name, "json")
    write_json_atomic(path, obj)
    return path


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metadata storage: append-only SQLite audit trail
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id      TEXT PRIMARY KEY,
    merchant        TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    row_count       INTEGER NOT NULL,
    col_count       INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    schema_json     TEXT NOT NULL,
    artifact_path   TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS splits (
    split_id        TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(dataset_id),
    strategy        TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    train_ids_path  TEXT NOT NULL,
    test_ids_path   TEXT NOT NULL,
    train_rows      INTEGER NOT NULL,
    test_rows       INTEGER NOT NULL,
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS model_bundles (
    bundle_id           TEXT PRIMARY KEY,
    merchant            TEXT NOT NULL,
    dataset_id          TEXT NOT NULL REFERENCES datasets(dataset_id),
    split_id            TEXT NOT NULL REFERENCES splits(split_id),
    architecture        TEXT NOT NULL,
    target_config_json  TEXT NOT NULL,
    feature_list_json   TEXT NOT NULL,
    hyperparams_json    TEXT NOT NULL,
    metrics_json        TEXT NOT NULL,
    bundle_path         TEXT NOT NULL,
    created_at          TIMESTAMP NOT NULL,
    created_by          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id          TEXT PRIMARY KEY,
    bundle_id       TEXT NOT NULL REFERENCES model_bundles(bundle_id),
    dataset_id      TEXT NOT NULL REFERENCES datasets(dataset_id),
    row_count       INTEGER NOT NULL,
    output_path     TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_datasets (
    raw_dataset_id    TEXT PRIMARY KEY,
    merchant          TEXT NOT NULL,
    purpose           TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path       TEXT NOT NULL,
    parquet_path      TEXT NOT NULL,
    delimiter         TEXT,
    encoding          TEXT NOT NULL,
    sheet_name        TEXT,
    row_count         INTEGER NOT NULL,
    col_count         INTEGER NOT NULL,
    content_hash      TEXT NOT NULL,
    uploaded_by       TEXT NOT NULL,
    notes             TEXT NOT NULL DEFAULT '',
    created_at        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS cleaned_datasets (
    cleaned_dataset_id     TEXT PRIMARY KEY,
    merchant               TEXT NOT NULL,
    purpose                TEXT NOT NULL,
    original_filename      TEXT NOT NULL,
    stored_path            TEXT NOT NULL,
    parquet_path           TEXT NOT NULL,
    delimiter              TEXT,
    encoding                TEXT NOT NULL,
    sheet_name              TEXT,
    row_count               INTEGER NOT NULL,
    col_count                INTEGER NOT NULL,
    content_hash             TEXT NOT NULL,
    source_raw_dataset_id     TEXT REFERENCES raw_datasets(raw_dataset_id),
    uploaded_by                TEXT NOT NULL,
    notes                       TEXT NOT NULL DEFAULT '',
    created_at                   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_fields (
    field_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    dtype           TEXT NOT NULL,
    required_level  TEXT NOT NULL,
    unique_required INTEGER NOT NULL DEFAULT 0,
    description     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS field_mappings (
    mapping_id              TEXT PRIMARY KEY,
    merchant                TEXT NOT NULL,
    name                    TEXT NOT NULL,
    version                 INTEGER NOT NULL,
    mapping_json             TEXT NOT NULL,
    unmapped_columns_json     TEXT NOT NULL,
    created_by                 TEXT NOT NULL,
    created_at                  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_datasets (
    canonical_dataset_id   TEXT PRIMARY KEY,
    merchant               TEXT NOT NULL,
    purpose                TEXT NOT NULL,
    cleaned_dataset_id      TEXT NOT NULL REFERENCES cleaned_datasets(cleaned_dataset_id),
    mapping_id               TEXT NOT NULL REFERENCES field_mappings(mapping_id),
    row_count                 INTEGER NOT NULL,
    col_count                  INTEGER NOT NULL,
    artifact_path                TEXT NOT NULL,
    validation_json               TEXT NOT NULL,
    created_by                     TEXT NOT NULL,
    created_at                       TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS enrichment_runs (
    run_id                  TEXT PRIMARY KEY,
    canonical_dataset_id    TEXT NOT NULL REFERENCES canonical_datasets(canonical_dataset_id),
    source                  TEXT NOT NULL,
    batch_id                 TEXT NOT NULL,
    rows_attempted            INTEGER NOT NULL,
    matched                    INTEGER NOT NULL,
    no_hit                      INTEGER NOT NULL,
    error                         INTEGER NOT NULL,
    not_sent                       INTEGER NOT NULL,
    started_at                       TIMESTAMP NOT NULL,
    finished_at                        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS enrichment_responses (
    response_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES enrichment_runs(run_id),
    lead_id         TEXT NOT NULL,
    source           TEXT NOT NULL,
    status            TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    fetched_at           TIMESTAMP NOT NULL,
    batch_id              TEXT NOT NULL,
    created_at              TIMESTAMP NOT NULL
);
"""

# Every table is append-only: block UPDATE/DELETE at the database level so a
# bug elsewhere in the app can't silently rewrite the audit trail. canonical_fields
# is a growing config list (fields are only ever added, never renamed/removed here
# either — see app/core/canonical.py, which treats "already exists" as a no-op)
# so it gets the same protection as the audit-trail tables.
_APPEND_ONLY_TABLES = (
    "datasets", "splits", "model_bundles", "prediction_runs", "raw_datasets",
    "cleaned_datasets", "canonical_fields", "field_mappings", "canonical_datasets",
    "enrichment_runs", "enrichment_responses",
)

_TRIGGER_SQL_TEMPLATE = """
CREATE TRIGGER IF NOT EXISTS {table}_no_update
BEFORE UPDATE ON {table}
BEGIN
    SELECT RAISE(ABORT, '{table} is append-only: UPDATE is not permitted');
END;

CREATE TRIGGER IF NOT EXISTS {table}_no_delete
BEFORE DELETE ON {table}
BEGIN
    SELECT RAISE(ABORT, '{table} is append-only: DELETE is not permitted');
END;
"""


class MetadataStore:
    """Append-only metadata store backed by SQLite (meta.db)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA_SQL)
            for table in _APPEND_ONLY_TABLES:
                conn.executescript(_TRIGGER_SQL_TEMPLATE.format(table=table))
            conn.commit()

    # -- datasets ----------------------------------------------------------

    def insert_dataset(
        self,
        *,
        dataset_id: str,
        merchant: str,
        purpose: str,
        source_file: str,
        row_count: int,
        col_count: int,
        content_hash: str,
        schema_json: str,
        artifact_path: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO datasets
                   (dataset_id, merchant, purpose, source_file, row_count, col_count,
                    content_hash, schema_json, artifact_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dataset_id, merchant, purpose, source_file, row_count, col_count,
                 content_hash, schema_json, artifact_path, utcnow_iso()),
            )
            conn.commit()

    def get_dataset(self, dataset_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            return dict(row) if row else None

    def list_datasets(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM datasets WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- splits --------------------------------------------------------------

    def insert_split(
        self,
        *,
        split_id: str,
        dataset_id: str,
        strategy: str,
        config_json: str,
        train_ids_path: str,
        test_ids_path: str,
        train_rows: int,
        test_rows: int,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO splits
                   (split_id, dataset_id, strategy, config_json, train_ids_path,
                    test_ids_path, train_rows, test_rows, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (split_id, dataset_id, strategy, config_json, train_ids_path,
                 test_ids_path, train_rows, test_rows, utcnow_iso()),
            )
            conn.commit()

    def get_split(self, split_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM splits WHERE split_id = ?", (split_id,)).fetchone()
            return dict(row) if row else None

    def list_splits(self, dataset_id: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if dataset_id is None:
                rows = conn.execute("SELECT * FROM splits ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM splits WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- model_bundles ---------------------------------------------------------

    def insert_model_bundle(
        self,
        *,
        bundle_id: str,
        merchant: str,
        dataset_id: str,
        split_id: str,
        architecture: str,
        target_config_json: str,
        feature_list_json: str,
        hyperparams_json: str,
        metrics_json: str,
        bundle_path: str,
        created_by: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO model_bundles
                   (bundle_id, merchant, dataset_id, split_id, architecture,
                    target_config_json, feature_list_json, hyperparams_json,
                    metrics_json, bundle_path, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (bundle_id, merchant, dataset_id, split_id, architecture,
                 target_config_json, feature_list_json, hyperparams_json,
                 metrics_json, bundle_path, utcnow_iso(), created_by),
            )
            conn.commit()

    def get_model_bundle(self, bundle_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM model_bundles WHERE bundle_id = ?", (bundle_id,)).fetchone()
            return dict(row) if row else None

    def list_model_bundles(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM model_bundles ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM model_bundles WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- prediction_runs -----------------------------------------------------

    def insert_prediction_run(
        self,
        *,
        run_id: str,
        bundle_id: str,
        dataset_id: str,
        row_count: int,
        output_path: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO prediction_runs
                   (run_id, bundle_id, dataset_id, row_count, output_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, bundle_id, dataset_id, row_count, output_path, utcnow_iso()),
            )
            conn.commit()

    def get_prediction_run(self, run_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_prediction_runs(self, bundle_id: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if bundle_id is None:
                rows = conn.execute("SELECT * FROM prediction_runs ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM prediction_runs WHERE bundle_id = ? ORDER BY created_at DESC", (bundle_id,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- raw_datasets ----------------------------------------------------------

    def insert_raw_dataset(
        self,
        *,
        raw_dataset_id: str,
        merchant: str,
        purpose: str,
        original_filename: str,
        stored_path: str,
        parquet_path: str,
        delimiter: Optional[str],
        encoding: str,
        sheet_name: Optional[str],
        row_count: int,
        col_count: int,
        content_hash: str,
        uploaded_by: str,
        notes: str = "",
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO raw_datasets
                   (raw_dataset_id, merchant, purpose, original_filename, stored_path, parquet_path,
                    delimiter, encoding, sheet_name, row_count, col_count, content_hash, uploaded_by,
                    notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (raw_dataset_id, merchant, purpose, original_filename, stored_path, parquet_path,
                 delimiter, encoding, sheet_name, row_count, col_count, content_hash, uploaded_by,
                 notes, utcnow_iso()),
            )
            conn.commit()

    def get_raw_dataset(self, raw_dataset_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM raw_datasets WHERE raw_dataset_id = ?", (raw_dataset_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_raw_datasets(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM raw_datasets ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM raw_datasets WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    def find_raw_datasets_by_content_hash(self, content_hash: str) -> list[dict]:
        """Prior uploads with identical bytes — used to warn on re-upload.
        Never blocks the new insert; append-only means it's recorded anyway.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM raw_datasets WHERE content_hash = ? ORDER BY created_at DESC", (content_hash,)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- cleaned_datasets ----------------------------------------------------

    def insert_cleaned_dataset(
        self,
        *,
        cleaned_dataset_id: str,
        merchant: str,
        purpose: str,
        original_filename: str,
        stored_path: str,
        parquet_path: str,
        delimiter: Optional[str],
        encoding: str,
        sheet_name: Optional[str],
        row_count: int,
        col_count: int,
        content_hash: str,
        source_raw_dataset_id: Optional[str],
        uploaded_by: str,
        notes: str = "",
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO cleaned_datasets
                   (cleaned_dataset_id, merchant, purpose, original_filename, stored_path, parquet_path,
                    delimiter, encoding, sheet_name, row_count, col_count, content_hash,
                    source_raw_dataset_id, uploaded_by, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cleaned_dataset_id, merchant, purpose, original_filename, stored_path, parquet_path,
                 delimiter, encoding, sheet_name, row_count, col_count, content_hash,
                 source_raw_dataset_id, uploaded_by, notes, utcnow_iso()),
            )
            conn.commit()

    def get_cleaned_dataset(self, cleaned_dataset_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM cleaned_datasets WHERE cleaned_dataset_id = ?", (cleaned_dataset_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_cleaned_datasets(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM cleaned_datasets ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cleaned_datasets WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    def find_cleaned_datasets_by_content_hash(self, content_hash: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM cleaned_datasets WHERE content_hash = ? ORDER BY created_at DESC", (content_hash,)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- canonical_fields ------------------------------------------------------

    def insert_canonical_field(
        self,
        *,
        field_id: str,
        name: str,
        dtype: str,
        required_level: str,
        unique_required: bool = False,
        description: str = "",
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO canonical_fields
                   (field_id, name, dtype, required_level, unique_required, description, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (field_id, name, dtype, required_level, int(unique_required), description, utcnow_iso()),
            )
            conn.commit()

    def get_canonical_field_by_name(self, name: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM canonical_fields WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_canonical_fields(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM canonical_fields ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    # -- field_mappings --------------------------------------------------------

    def insert_field_mapping(
        self,
        *,
        mapping_id: str,
        merchant: str,
        name: str,
        version: int,
        mapping_json: str,
        unmapped_columns_json: str,
        created_by: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO field_mappings
                   (mapping_id, merchant, name, version, mapping_json, unmapped_columns_json, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (mapping_id, merchant, name, version, mapping_json, unmapped_columns_json, created_by, utcnow_iso()),
            )
            conn.commit()

    def get_field_mapping(self, mapping_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM field_mappings WHERE mapping_id = ?", (mapping_id,)).fetchone()
            return dict(row) if row else None

    def list_field_mappings(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM field_mappings ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM field_mappings WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- canonical_datasets ------------------------------------------------------

    def insert_canonical_dataset(
        self,
        *,
        canonical_dataset_id: str,
        merchant: str,
        purpose: str,
        cleaned_dataset_id: str,
        mapping_id: str,
        row_count: int,
        col_count: int,
        artifact_path: str,
        validation_json: str,
        created_by: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO canonical_datasets
                   (canonical_dataset_id, merchant, purpose, cleaned_dataset_id, mapping_id,
                    row_count, col_count, artifact_path, validation_json, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (canonical_dataset_id, merchant, purpose, cleaned_dataset_id, mapping_id,
                 row_count, col_count, artifact_path, validation_json, created_by, utcnow_iso()),
            )
            conn.commit()

    def get_canonical_dataset(self, canonical_dataset_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM canonical_datasets WHERE canonical_dataset_id = ?", (canonical_dataset_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_canonical_datasets(self, merchant: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if merchant is None:
                rows = conn.execute("SELECT * FROM canonical_datasets ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM canonical_datasets WHERE merchant = ? ORDER BY created_at DESC", (merchant,)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- enrichment_runs ----------------------------------------------------------

    def insert_enrichment_run(
        self,
        *,
        run_id: str,
        canonical_dataset_id: str,
        source: str,
        batch_id: str,
        rows_attempted: int,
        matched: int,
        no_hit: int,
        error: int,
        not_sent: int,
        started_at: str,
        finished_at: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO enrichment_runs
                   (run_id, canonical_dataset_id, source, batch_id, rows_attempted,
                    matched, no_hit, error, not_sent, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, canonical_dataset_id, source, batch_id, rows_attempted,
                 matched, no_hit, error, not_sent, started_at, finished_at),
            )
            conn.commit()

    def get_enrichment_run(self, run_id: str) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM enrichment_runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_enrichment_runs(self, canonical_dataset_id: Optional[str] = None) -> list[dict]:
        with closing(self._connect()) as conn:
            if canonical_dataset_id is None:
                rows = conn.execute("SELECT * FROM enrichment_runs ORDER BY started_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM enrichment_runs WHERE canonical_dataset_id = ? ORDER BY started_at DESC",
                    (canonical_dataset_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # -- enrichment_responses ----------------------------------------------------

    def insert_enrichment_response(
        self,
        *,
        response_id: str,
        run_id: str,
        lead_id: str,
        source: str,
        status: str,
        payload_json: str,
        fetched_at: str,
        batch_id: str,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO enrichment_responses
                   (response_id, run_id, lead_id, source, status, payload_json, fetched_at, batch_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (response_id, run_id, lead_id, source, status, payload_json, fetched_at, batch_id, utcnow_iso()),
            )
            conn.commit()

    def list_enrichment_responses(
        self, run_id: Optional[str] = None, lead_id: Optional[str] = None
    ) -> list[dict]:
        with closing(self._connect()) as conn:
            clauses, params = [], []
            if run_id is not None:
                clauses.append("run_id = ?")
                params.append(run_id)
            if lead_id is not None:
                clauses.append("lead_id = ?")
                params.append(lead_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM enrichment_responses {where} ORDER BY created_at DESC", params
            ).fetchall()
            return [dict(r) for r in rows]

    # -- cross-table -------------------------------------------------------

    def list_merchants(self) -> list[str]:
        """Every distinct merchant name seen anywhere in the audit trail, for
        the Upload tab's autocomplete — new merchants are expected, so this is
        never a fixed list.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT merchant FROM raw_datasets
                   UNION SELECT merchant FROM datasets
                   UNION SELECT merchant FROM model_bundles
                   UNION SELECT merchant FROM cleaned_datasets
                   UNION SELECT merchant FROM canonical_datasets
                   ORDER BY merchant"""
            ).fetchall()
            return [r["merchant"] for r in rows]
