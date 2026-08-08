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


def timestamped_path(root: Path, kind: str, merchant: str, name: str, ext: str) -> Path:
    """{root}/{kind}/{merchant}/{YYYYMMDD_HHMMSS}_{name}.{ext}

    Appends a numeric suffix if that exact path is already taken (two writes
    in the same second), so storage is genuinely append-only.
    """
    directory = Path(root) / kind / _safe_segment(merchant)
    stamp = _utcnow_stamp()
    safe_name = _safe_segment(name)
    candidate = directory / f"{stamp}_{safe_name}.{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stamp}_{safe_name}_{counter:03d}.{ext}"
        counter += 1
    return candidate


def unique_dir_path(root: Path, kind: str, merchant: str, name: str) -> Path:
    """Like timestamped_path but for a directory (used for model bundles)."""
    directory = Path(root) / kind / _safe_segment(merchant)
    stamp = _utcnow_stamp()
    safe_name = _safe_segment(name)
    candidate = directory / f"{stamp}_{safe_name}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stamp}_{safe_name}_{counter:03d}"
        counter += 1
    return candidate


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
"""

# Every table is append-only: block UPDATE/DELETE at the database level so a
# bug elsewhere in the app can't silently rewrite the audit trail.
_APPEND_ONLY_TABLES = ("datasets", "splits", "model_bundles", "prediction_runs")

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
