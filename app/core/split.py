"""Train/test split strategies.

Every strategy below is a pure function over an already-loaded dataframe: it
returns row *positions*, never mutates or refits anything, and never depends
on library-version-sensitive RNG behaviour for reproducibility — the actual
train/test membership is captured as explicit row-ID lists in SplitResult,
and save_split() is what makes that durable (see CLAUDE.md: "Seeds are not
stable across library versions.").

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from . import storage

if TYPE_CHECKING:
    from .storage import MetadataStore

SplitStrategy = Literal["random", "stratified", "temporal", "grouped"]
TemporalMode = Literal["cutoff", "periods"]

SPLIT_STRATEGIES: tuple[SplitStrategy, ...] = ("random", "stratified", "temporal", "grouped")
TEMPORAL_MODES: tuple[TemporalMode, ...] = ("cutoff", "periods")

# Any stratum (joint key of the chosen stratification columns) with fewer than
# this many members breaks a stratified split — one side would get zero rows
# for that stratum for at least half of them. Collapse instead of crashing.
MIN_STRATUM_SIZE = 2
RARE_STRATUM_LABEL = "__RARE__"
MISSING_VALUE_LABEL = "<missing>"

# Relative shift in base rate (test vs train) beyond which a temporal split
# gets a warning — a strong signal of outcome immaturity, not just split noise.
DEFAULT_BASE_RATE_SHIFT_THRESHOLD = 0.30


def new_split_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class SplitResult:
    strategy: SplitStrategy
    config: dict[str, Any]
    train_ids: list[Any]
    test_ids: list[Any]
    train_rows: int
    test_rows: int
    composition: pd.DataFrame  # long-format: column, value, train_count, train_pct, test_count, test_pct
    warnings: list[str]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _row_ids(df: pd.DataFrame, row_id_column: Optional[str]) -> np.ndarray:
    """Stable per-row identifiers. Falls back to the dataframe's own index.

    Fails loudly on nulls/duplicates: an ambiguous ID would silently corrupt
    which physical row a persisted split refers to.
    """
    if row_id_column is None:
        return df.index.to_numpy()

    if row_id_column not in df.columns:
        raise ValueError(f"row_id_column {row_id_column!r} not found in dataframe")
    ids = df[row_id_column]
    if ids.isna().any():
        raise ValueError(f"row_id_column {row_id_column!r} contains null values; every row needs a stable id")
    if ids.duplicated().any():
        raise ValueError(f"row_id_column {row_id_column!r} contains duplicate values; every row needs a unique id")
    return ids.to_numpy()


def _composition_report(
    df: pd.DataFrame, train_positions: np.ndarray, test_positions: np.ndarray, columns: list[str]
) -> pd.DataFrame:
    """Per stratification column, per value: count and % share of train vs test.

    This is the table a user eyeballs to confirm composition was preserved.
    """
    train_df = df.iloc[train_positions]
    test_df = df.iloc[test_positions]
    rows: list[dict[str, Any]] = []

    for col in columns:
        train_counts = train_df[col].fillna(MISSING_VALUE_LABEL).astype(str).value_counts()
        test_counts = test_df[col].fillna(MISSING_VALUE_LABEL).astype(str).value_counts()
        for value in sorted(set(train_counts.index) | set(test_counts.index)):
            train_n = int(train_counts.get(value, 0))
            test_n = int(test_counts.get(value, 0))
            rows.append(
                {
                    "column": col,
                    "value": value,
                    "train_count": train_n,
                    "train_pct": round(100 * train_n / len(train_df), 2) if len(train_df) else 0.0,
                    "test_count": test_n,
                    "test_pct": round(100 * test_n / len(test_df), 2) if len(test_df) else 0.0,
                }
            )

    return pd.DataFrame(rows, columns=["column", "value", "train_count", "train_pct", "test_count", "test_pct"])


def _check_base_rate_shift(train_rate: float, test_rate: float, threshold: float, warnings: list[str]) -> None:
    if train_rate is None or test_rate is None or math.isnan(train_rate) or math.isnan(test_rate):
        return
    if train_rate == 0:
        if test_rate > 0:
            warnings.append(
                f"train base rate is 0% but test base rate is {test_rate:.2%} — check for outcome "
                f"immaturity or a bad cutoff choice"
            )
        return
    relative_shift = abs(test_rate - train_rate) / train_rate
    if relative_shift > threshold:
        warnings.append(
            f"test base rate ({test_rate:.2%}) differs from train base rate ({train_rate:.2%}) by "
            f"{relative_shift:.0%} relative — this often signals outcome immaturity (recent rows "
            f"haven't had time to convert yet and look like negatives) rather than a genuine shift"
        )


# ---------------------------------------------------------------------------
# Strategy implementations — each returns (train_positions, test_positions, config, warnings)
# ---------------------------------------------------------------------------


def _random_positions(n: int, test_size: float, rng: np.random.Generator):
    positions = np.arange(n)
    rng.shuffle(positions)
    n_test = min(max(round(n * test_size), 0), n)
    test_positions = positions[:n_test]
    train_positions = positions[n_test:]
    config = {"test_size": test_size}
    return train_positions, test_positions, config, []


def _joint_stratification_key(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = [df[c].fillna(MISSING_VALUE_LABEL).astype(str) for c in columns]
    key = parts[0]
    for p in parts[1:]:
        key = key + "\x1f" + p  # unit separator: avoids ambiguity between column values containing "|" etc.
    return key.to_numpy()


def _stratified_positions(df: pd.DataFrame, stratify_columns: list[str], test_size: float, rng: np.random.Generator):
    key = _joint_stratification_key(df, stratify_columns)
    positions = np.arange(len(df))
    warnings: list[str] = []

    counts = pd.Series(key).value_counts()
    rare_values = counts[counts < MIN_STRATUM_SIZE].index
    n_collapsed_strata = len(rare_values)
    n_collapsed_rows = int(counts.loc[rare_values].sum()) if n_collapsed_strata else 0
    if n_collapsed_strata:
        key = np.where(np.isin(key, rare_values), RARE_STRATUM_LABEL, key)
        warnings.append(
            f"Collapsed {n_collapsed_strata} rare strata ({n_collapsed_rows} rows, each with fewer than "
            f"{MIN_STRATUM_SIZE} members) into a single '{RARE_STRATUM_LABEL}' stratum before splitting."
        )

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    n_singleton_rows = 0
    for stratum in pd.unique(key):
        stratum_positions = positions[key == stratum].copy()
        rng.shuffle(stratum_positions)
        if len(stratum_positions) >= MIN_STRATUM_SIZE:
            n_test = round(len(stratum_positions) * test_size)
            n_test = min(max(n_test, 1), len(stratum_positions) - 1)  # guarantee both sides get >=1
        else:
            n_test = 0
            n_singleton_rows += len(stratum_positions)
        test_parts.append(stratum_positions[:n_test])
        train_parts.append(stratum_positions[n_test:])

    if n_singleton_rows:
        warnings.append(
            f"{n_singleton_rows} row(s) belonged to a stratum with exactly 1 member after collapsing and "
            f"could not be split between train and test; assigned entirely to train."
        )

    train_positions = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_positions = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    config = {
        "stratify_columns": list(stratify_columns),
        "test_size": test_size,
        "n_collapsed_strata": n_collapsed_strata,
        "n_collapsed_rows": n_collapsed_rows,
    }
    return train_positions, test_positions, config, warnings


def _temporal_positions(
    df: pd.DataFrame,
    *,
    date_column: str,
    mode: TemporalMode,
    cutoff: Optional[str],
    test_period_values: Optional[list[Any]],
):
    if date_column not in df.columns:
        raise ValueError(f"date_column {date_column!r} not found in dataframe")

    positions = np.arange(len(df))

    if mode == "cutoff":
        if not cutoff:
            raise ValueError("temporal split in 'cutoff' mode requires a cutoff value")
        parsed = pd.to_datetime(df[date_column], errors="raise")
        cutoff_ts = pd.to_datetime(cutoff)
        test_mask = (parsed >= cutoff_ts).to_numpy()
    elif mode == "periods":
        if not test_period_values:
            raise ValueError("temporal split in 'periods' mode requires test_period_values")
        test_mask = df[date_column].isin(test_period_values).to_numpy()
    else:
        raise ValueError(f"unknown temporal split mode {mode!r}, must be 'cutoff' or 'periods'")

    test_positions = positions[test_mask]
    train_positions = positions[~test_mask]

    config = {
        "date_column": date_column,
        "mode": mode,
        "cutoff": cutoff,
        "test_period_values": list(test_period_values) if test_period_values else None,
    }
    return train_positions, test_positions, config, []


def _grouped_positions(df: pd.DataFrame, group_column: str, test_size: float, rng: np.random.Generator):
    if group_column not in df.columns:
        raise ValueError(f"group_column {group_column!r} not found in dataframe")
    groups = df[group_column]
    if groups.isna().any():
        raise ValueError(f"group_column {group_column!r} contains null values; every row needs a group")

    group_values = groups.to_numpy()
    unique_groups = pd.unique(group_values)
    rng.shuffle(unique_groups)

    warnings: list[str] = []
    if len(unique_groups) < 2:
        n_test_groups = 0
        warnings.append("only one distinct group value is present; the entire dataset was assigned to train")
    else:
        n_test_groups = round(len(unique_groups) * test_size)
        n_test_groups = min(max(n_test_groups, 1), len(unique_groups) - 1)

    test_groups = set(unique_groups[:n_test_groups])
    test_mask = np.isin(group_values, list(test_groups))
    positions = np.arange(len(df))
    test_positions = positions[test_mask]
    train_positions = positions[~test_mask]

    config = {
        "group_column": group_column,
        "test_size": test_size,
        "n_groups": int(len(unique_groups)),
        "n_test_groups": int(n_test_groups),
    }
    return train_positions, test_positions, config, warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
    *,
    strategy: SplitStrategy,
    row_id_column: Optional[str] = None,
    test_size: float = 0.2,
    stratify_columns: Optional[list[str]] = None,
    date_column: Optional[str] = None,
    temporal_mode: TemporalMode = "cutoff",
    cutoff: Optional[str] = None,
    test_period_values: Optional[list[Any]] = None,
    group_column: Optional[str] = None,
    target_column: Optional[str] = None,
    composition_columns: Optional[list[str]] = None,
    random_seed: int = 42,
    base_rate_shift_threshold: float = DEFAULT_BASE_RATE_SHIFT_THRESHOLD,
) -> SplitResult:
    """Split df into train/test row-ID sets using the chosen strategy.

    random_seed only controls this in-memory split; reproducibility comes
    from persisting the resulting IDs via save_split(), not from the seed.
    """
    if len(df) == 0:
        raise ValueError("cannot split an empty dataframe")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}, must be one of {SPLIT_STRATEGIES}")

    rng = np.random.default_rng(random_seed)

    if strategy == "random":
        train_pos, test_pos, config, warnings = _random_positions(len(df), test_size, rng)

    elif strategy == "stratified":
        if not stratify_columns:
            raise ValueError("stratified split requires at least one column in stratify_columns")
        missing = [c for c in stratify_columns if c not in df.columns]
        if missing:
            raise ValueError(f"stratify_columns not found in dataframe: {missing}")
        train_pos, test_pos, config, warnings = _stratified_positions(df, stratify_columns, test_size, rng)

    elif strategy == "temporal":
        if not date_column:
            raise ValueError("temporal split requires date_column")
        train_pos, test_pos, config, warnings = _temporal_positions(
            df, date_column=date_column, mode=temporal_mode, cutoff=cutoff, test_period_values=test_period_values
        )
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"target_column {target_column!r} not found in dataframe")
            train_rate = (
                pd.to_numeric(df.iloc[train_pos][target_column], errors="coerce").mean() if len(train_pos) else float("nan")
            )
            test_rate = (
                pd.to_numeric(df.iloc[test_pos][target_column], errors="coerce").mean() if len(test_pos) else float("nan")
            )
            _check_base_rate_shift(train_rate, test_rate, base_rate_shift_threshold, warnings)

    else:  # grouped
        if not group_column:
            raise ValueError("grouped split requires group_column")
        train_pos, test_pos, config, warnings = _grouped_positions(df, group_column, test_size, rng)

    warnings = list(warnings)
    if len(train_pos) == 0:
        warnings.append("train split is empty")
    if len(test_pos) == 0:
        warnings.append("test split is empty")

    ids = _row_ids(df, row_id_column)
    train_ids = [_jsonable(v) for v in ids[train_pos]]
    test_ids = [_jsonable(v) for v in ids[test_pos]]

    report_columns = composition_columns if composition_columns is not None else list(stratify_columns or [])
    report_columns = [c for c in report_columns if c in df.columns]
    composition = _composition_report(df, train_pos, test_pos, report_columns)

    config = {**config, "row_id_column": row_id_column, "random_seed": random_seed}

    return SplitResult(
        strategy=strategy,
        config=config,
        train_ids=train_ids,
        test_ids=test_ids,
        train_rows=len(train_pos),
        test_rows=len(test_pos),
        composition=composition,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Persistence — explicit row-ID lists, never a seed (see module docstring)
# ---------------------------------------------------------------------------


def save_split(
    *,
    store: "MetadataStore",
    root: Path | str,
    merchant: str,
    dataset_id: str,
    split_id: str,
    result: SplitResult,
) -> tuple[Path, Path]:
    """Persist train/test IDs as new timestamped artifacts and record the split."""
    train_artifact = {
        "split_id": split_id,
        "side": "train",
        "row_id_column": result.config.get("row_id_column"),
        "ids": result.train_ids,
    }
    test_artifact = {
        "split_id": split_id,
        "side": "test",
        "row_id_column": result.config.get("row_id_column"),
        "ids": result.test_ids,
    }
    train_path = storage.save_json(train_artifact, "split", merchant, f"{split_id}_train_ids", root=root)
    test_path = storage.save_json(test_artifact, "split", merchant, f"{split_id}_test_ids", root=root)

    store.insert_split(
        split_id=split_id,
        dataset_id=dataset_id,
        strategy=result.strategy,
        config_json=json.dumps(result.config, sort_keys=True),
        train_ids_path=str(train_path),
        test_ids_path=str(test_path),
        train_rows=result.train_rows,
        test_rows=result.test_rows,
    )
    return train_path, test_path


def load_split_ids(split_row: dict) -> tuple[list[Any], list[Any]]:
    """Load back the (train_ids, test_ids) persisted for a splits-table row."""
    train_ids = storage.load_json(split_row["train_ids_path"])["ids"]
    test_ids = storage.load_json(split_row["test_ids_path"])["ids"]
    return train_ids, test_ids


def apply_split(
    df: pd.DataFrame, row_id_column: Optional[str], train_ids: list[Any], test_ids: list[Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize train/test dataframes from persisted IDs. Fails loudly if any ID is missing."""
    ids = _row_ids(df, row_id_column)
    id_to_position = {v: i for i, v in enumerate(ids)}

    missing_train = [i for i in train_ids if i not in id_to_position]
    missing_test = [i for i in test_ids if i not in id_to_position]
    if missing_train or missing_test:
        raise ValueError(
            f"split ids not found in dataframe: {len(missing_train)} train id(s), {len(missing_test)} test id(s) missing"
        )

    train_df = df.iloc[[id_to_position[i] for i in train_ids]]
    test_df = df.iloc[[id_to_position[i] for i in test_ids]]
    return train_df, test_df
