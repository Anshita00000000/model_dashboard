"""Chained multi-stage scoring.

The user configures a single ordinal outcome column with K distinct levels in
funnel order (e.g. target_column="disposition", level_order=["No Appointment",
"Appointment Booked", "Consulted", "Converted"]). This module turns that into
K-1 conditional binary stages:

  stage i (0-indexed): eligible rows = reached level i (level_index >= i).
                        target = did they progress past level i (level_index > i)?

Stage 1 is eligible = all rows (level_index >= 0 always holds), matching
"stage 1 trains on all rows; stage N trains ONLY on rows where stage N-1
occurred" — subsetting uses the ACTUAL historical outcome, not a predicted
probability.

At prediction time (app/core/registry.py), every row is scored through every
stage's booster unconditionally, and the final score is the cumulative
product of stage probabilities. That product is mathematically guaranteed to
be non-increasing stage over stage as long as every stage probability is a
valid value in [0, 1] — assert_no_ordinal_violations() is the sanity check
that catches it if that guarantee is ever broken by a bug upstream (e.g. a
raw margin score slipping in instead of a probability).

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_STAGE_POSITIVES_WARN = 50


@dataclass(frozen=True)
class FunnelConfig:
    target_column: str
    level_order: list[str]  # earliest -> final, K distinct levels
    stage_names: list[str]  # K-1 names, one per conditional stage

    @property
    def n_stages(self) -> int:
        return len(self.level_order) - 1


def _default_stage_name(level_label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", level_label.strip().lower()).strip("_")
    return f"reached_{slug}" if slug else "reached_unnamed_stage"


def build_funnel_config(
    df: pd.DataFrame,
    *,
    target_column: str,
    level_order: list[str],
    stage_names: list[str] | None = None,
) -> FunnelConfig:
    """Validate the user's stage config against the target column's actual
    distinct values, and report how many conditional stages will be built.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not found in dataframe")
    if len(level_order) != len(set(level_order)):
        raise ValueError(f"level_order contains duplicate values: {level_order}")
    if len(level_order) < 2:
        raise ValueError(
            f"a funnel needs at least 2 outcome levels to build any conditional stage, got {level_order}"
        )

    actual_levels = set(df[target_column].dropna().unique().tolist())
    configured_levels = set(level_order)
    in_data_not_configured = actual_levels - configured_levels
    configured_not_in_data = configured_levels - actual_levels
    if in_data_not_configured or configured_not_in_data:
        raise ValueError(
            f"level_order does not match the distinct values of {target_column!r}: "
            f"in data but not in level_order: {sorted(in_data_not_configured)}; "
            f"in level_order but not in data: {sorted(configured_not_in_data)}"
        )

    n_stages = len(level_order) - 1
    if stage_names is None:
        stage_names = [_default_stage_name(level_order[i + 1]) for i in range(n_stages)]
    if len(stage_names) != n_stages:
        raise ValueError(
            f"{len(level_order)} outcome levels require {n_stages} conditional stage(s), "
            f"got {len(stage_names)} stage_names: {stage_names}"
        )
    if len(stage_names) != len(set(stage_names)):
        raise ValueError(f"stage_names must be unique: {stage_names}")

    return FunnelConfig(target_column=target_column, level_order=list(level_order), stage_names=list(stage_names))


def describe_funnel_config(funnel: FunnelConfig) -> str:
    levels = " -> ".join(funnel.level_order)
    stages = ", ".join(funnel.stage_names)
    return f"{len(funnel.level_order)} outcome levels ({levels}) -> {funnel.n_stages} conditional stage(s): {stages}"


@dataclass(frozen=True)
class StageData:
    name: str
    level_from: str
    level_to: str
    positions: np.ndarray  # positions into whatever dataframe was passed to build_stage_data
    y: np.ndarray  # binary target, aligned to positions
    n_rows: int
    n_positive: int


def build_stage_data(df: pd.DataFrame, funnel: FunnelConfig) -> list[StageData]:
    level_to_index = {level: i for i, level in enumerate(funnel.level_order)}
    level_index = df[funnel.target_column].map(level_to_index).to_numpy()

    stages: list[StageData] = []
    for i, name in enumerate(funnel.stage_names):
        eligible_mask = level_index >= i
        positions = np.where(eligible_mask)[0]
        y = (level_index[positions] > i).astype(int)
        stages.append(
            StageData(
                name=name,
                level_from=funnel.level_order[i],
                level_to=funnel.level_order[i + 1],
                positions=positions,
                y=y,
                n_rows=len(positions),
                n_positive=int(y.sum()),
            )
        )
    return stages


def undertrained_warnings(stages: list[StageData], min_positive: int = MIN_STAGE_POSITIVES_WARN) -> list[str]:
    warnings: list[str] = []
    for s in stages:
        if s.n_positive < min_positive:
            warnings.append(
                f"stage {s.name!r} has only {s.n_positive} positive example(s) (< {min_positive}); "
                f"this stage's model is likely undertrained"
            )
    return warnings


def assert_no_ordinal_violations(
    cumulative_probs_by_stage: dict[str, np.ndarray], stage_order: list[str], tolerance: float = 1e-9
) -> None:
    """cumulative_probs_by_stage: {stage_name: chained cumulative probability array},
    covering the SAME rows for every stage. Raises AssertionError if any row's
    cumulative probability increases moving to a later stage, or if any stage
    produced a value outside [0, 1].
    """
    prev: np.ndarray | None = None
    prev_name: str | None = None
    for name in stage_order:
        if name not in cumulative_probs_by_stage:
            raise ValueError(f"missing cumulative probabilities for stage {name!r}")
        probs = np.asarray(cumulative_probs_by_stage[name])

        out_of_range = (probs < -tolerance) | (probs > 1 + tolerance)
        if np.any(out_of_range):
            raise AssertionError(
                f"stage {name!r} produced {int(out_of_range.sum())} probability value(s) outside [0, 1]"
            )

        if prev is not None and np.any(probs > prev + tolerance):
            n_violations = int(np.sum(probs > prev + tolerance))
            raise AssertionError(
                f"ordinal violation: {n_violations} row(s) have a higher cumulative probability at "
                f"stage {name!r} than at the previous stage {prev_name!r} — chained scoring must be "
                f"non-increasing by construction"
            )
        prev, prev_name = probs, name
