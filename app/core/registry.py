"""Model bundle registry.

A bundle is ONE immutable directory containing everything needed to
reproduce a prediction: the boosters, the exact fitted preprocessing state,
and the manifest that ties it back to the dataset and split it was trained
on. This is the most important design constraint in the system — see
CLAUDE.md, "Model bundles are immutable and self-contained."

    bundle/
      manifest.json       bundle_id, architecture, created_at, dataset content_hash,
                           split_id, target config, ordered feature list, hyperparams
      stage_1.txt          LightGBM booster (native text format)
      stage_2.txt          additional stages for chained models
      stage_N_model_a.txt  dual-route stage, route a
      stage_N_model_b.txt  dual-route stage, route b
      preprocessing.json   FITTED preprocessing state (see PreprocessingState)
      metrics.json

Rules enforced here:
  - save_bundle() writes atomically; bundles are never mutated after write.
  - load_bundle() returns an object whose predict() uses ONLY the pinned state.
  - predict() never refits. A category unseen at training time maps to null
    (matching LightGBM's own training-time missing-value handling), never to
    a new category.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import storage

StageKind = Literal["single", "dual_route"]


def new_bundle_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Fitted preprocessing state — the contract predict() is not allowed to break
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingState:
    ordered_features: list[str]
    categorical_vocab: dict[str, list[str]]  # column -> exact training-time category vocabulary
    numeric_dtypes: dict[str, str]  # column -> dtype coercion rule (e.g. "float64")
    blend_weights: dict[str, float] = field(default_factory=dict)  # stage_name -> weight on route a (dual-route only)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingState":
        return cls(
            ordered_features=list(d["ordered_features"]),
            categorical_vocab={k: list(v) for k, v in d["categorical_vocab"].items()},
            numeric_dtypes=dict(d["numeric_dtypes"]),
            blend_weights=dict(d.get("blend_weights", {})),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: StageKind
    file: Optional[str] = None  # "single"
    file_a: Optional[str] = None  # "dual_route"
    file_b: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind == "single" and not self.file:
            raise ValueError(f"stage {self.name!r}: kind 'single' requires 'file'")
        if self.kind == "dual_route" and not (self.file_a and self.file_b):
            raise ValueError(f"stage {self.name!r}: kind 'dual_route' requires 'file_a' and 'file_b'")


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    architecture: str
    created_at: str
    dataset_content_hash: str
    split_id: str
    target_config: dict
    feature_list: list[str]
    hyperparams: dict
    stages: list[StageSpec]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# save_bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageInput:
    """One stage's fitted booster(s), ready to be written into a bundle."""

    name: str
    booster: Optional[lgb.Booster] = None  # single-model stage
    booster_a: Optional[lgb.Booster] = None  # dual-route stage, route a
    booster_b: Optional[lgb.Booster] = None  # dual-route stage, route b


def save_bundle(
    *,
    root: Path | str,
    merchant: str,
    bundle_id: str,
    architecture: str,
    stages: list[StageInput],
    preprocessing: PreprocessingState,
    dataset_content_hash: str,
    split_id: str,
    target_config: dict,
    hyperparams: dict,
    metrics: dict,
) -> Path:
    """Write a new immutable bundle directory. Never overwrites an existing one."""
    if not stages:
        raise ValueError("a bundle must have at least one stage")

    root = Path(root)
    bundle_merchant_dir = root / "bundle" / merchant
    bundle_merchant_dir.mkdir(parents=True, exist_ok=True)
    # Scratch dir on the same filesystem as the final location, so the commit
    # below is a single atomic rename rather than a copy that can fail partway.
    tmp_dir = Path(tempfile.mkdtemp(dir=bundle_merchant_dir, prefix=".tmp_bundle_"))

    try:
        stage_specs: list[StageSpec] = []
        for i, stage in enumerate(stages, start=1):
            if stage.booster is not None:
                filename = f"stage_{i}.txt"
                stage.booster.save_model(str(tmp_dir / filename))
                stage_specs.append(StageSpec(name=stage.name, kind="single", file=filename))
            elif stage.booster_a is not None and stage.booster_b is not None:
                filename_a = f"stage_{i}_model_a.txt"
                filename_b = f"stage_{i}_model_b.txt"
                stage.booster_a.save_model(str(tmp_dir / filename_a))
                stage.booster_b.save_model(str(tmp_dir / filename_b))
                stage_specs.append(StageSpec(name=stage.name, kind="dual_route", file_a=filename_a, file_b=filename_b))
            else:
                raise ValueError(f"stage {stage.name!r} must set either 'booster', or both 'booster_a' and 'booster_b'")

        manifest = BundleManifest(
            bundle_id=bundle_id,
            architecture=architecture,
            created_at=storage.utcnow_iso(),
            dataset_content_hash=dataset_content_hash,
            split_id=split_id,
            target_config=target_config,
            feature_list=list(preprocessing.ordered_features),
            hyperparams=hyperparams,
            stages=stage_specs,
        )
        storage.write_json_atomic(tmp_dir / "manifest.json", manifest.to_dict())
        storage.write_json_atomic(tmp_dir / "preprocessing.json", preprocessing.to_dict())
        storage.write_json_atomic(tmp_dir / "metrics.json", metrics)

        final_path = storage.unique_dir_path(root, "bundle", merchant, bundle_id)
        os.rename(tmp_dir, final_path)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return final_path


# ---------------------------------------------------------------------------
# load_bundle / ModelBundle
# ---------------------------------------------------------------------------


@dataclass
class _LoadedStage:
    name: str
    kind: StageKind
    booster: Optional[lgb.Booster] = None
    booster_a: Optional[lgb.Booster] = None
    booster_b: Optional[lgb.Booster] = None


class ModelBundle:
    """A loaded, immutable bundle. predict() uses ONLY the pinned state and never refits."""

    def __init__(
        self,
        path: Path,
        manifest: BundleManifest,
        preprocessing: PreprocessingState,
        metrics: dict,
        stages: list[_LoadedStage],
    ):
        self.path = path
        self.manifest = manifest
        self.preprocessing = preprocessing
        self.metrics = metrics
        self._stages = stages

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = self.preprocessing.ordered_features
        missing = [c for c in ordered if c not in df.columns]
        if missing:
            raise ValueError(f"prediction dataframe is missing required features: {missing}")

        # Select in the pinned order — this is what makes scoring invariant to
        # the incoming dataframe's column order (positional order is never used).
        X = df.loc[:, ordered].copy()

        for col, vocab in self.preprocessing.categorical_vocab.items():
            # Any value not in the training-time vocabulary becomes NaN here,
            # i.e. an unseen category maps to "missing", never to a new category.
            X[col] = X[col].astype("category").cat.set_categories(vocab)

        for col, dtype in self.preprocessing.numeric_dtypes.items():
            X[col] = X[col].astype(dtype)

        return X

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score df. Returns a per-stage probability column for each stage plus
        an overall chained 'score' (the cumulative product across stages, i.e.
        P(reach final stage) = product of each stage's conditional probability).
        """
        X = self._prepare_features(df)
        result = pd.DataFrame(index=df.index)
        cumulative = np.ones(len(df))

        for spec, loaded in zip(self.manifest.stages, self._stages):
            if loaded.kind == "single":
                stage_prob = loaded.booster.predict(X)
            else:
                weight = self.preprocessing.blend_weights.get(spec.name)
                if weight is None:
                    raise ValueError(f"dual-route stage {spec.name!r} has no pinned blend weight")
                prob_a = loaded.booster_a.predict(X)
                prob_b = loaded.booster_b.predict(X)
                stage_prob = weight * prob_a + (1.0 - weight) * prob_b

            result[f"{spec.name}_prob"] = stage_prob
            cumulative = cumulative * stage_prob

        result["score"] = cumulative
        return result


def load_bundle(path: Path | str) -> ModelBundle:
    path = Path(path)
    manifest_dict = dict(storage.load_json(path / "manifest.json"))
    stage_dicts = manifest_dict.pop("stages")
    manifest = BundleManifest(stages=[StageSpec(**s) for s in stage_dicts], **manifest_dict)
    preprocessing = PreprocessingState.from_dict(storage.load_json(path / "preprocessing.json"))
    metrics = storage.load_json(path / "metrics.json")

    loaded_stages: list[_LoadedStage] = []
    for spec in manifest.stages:
        if spec.kind == "single":
            booster = lgb.Booster(model_file=str(path / spec.file))
            loaded_stages.append(_LoadedStage(name=spec.name, kind="single", booster=booster))
        else:
            booster_a = lgb.Booster(model_file=str(path / spec.file_a))
            booster_b = lgb.Booster(model_file=str(path / spec.file_b))
            loaded_stages.append(_LoadedStage(name=spec.name, kind="dual_route", booster_a=booster_a, booster_b=booster_b))

    return ModelBundle(path=path, manifest=manifest, preprocessing=preprocessing, metrics=metrics, stages=loaded_stages)
