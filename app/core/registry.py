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

Different stages — and, within a dual-route stage, different routes — can be
trained on different feature subsets (e.g. ARCH 1 uses core features only;
ARCH 4's route B uses core features while route A uses all features). So the
ordered feature list is pinned PER stage/route, keyed by "{stage_name}" for a
single-model stage or "{stage_name}::a" / "{stage_name}::b" for a dual-route
stage's two routes — see PreprocessingState.stage_features.

Rules enforced here:
  - save_bundle() writes atomically; bundles are never mutated after write.
  - load_bundle() returns an object whose predict() uses ONLY the pinned state.
  - predict() never refits. A category unseen at training time maps to null
    (matching LightGBM's own training-time missing-value handling), never to
    a new category.
  - A dual-route stage blends blend_weight*A + (1-blend_weight)*B only for
    rows with real enrichment data (per the pinned enrichment_features list);
    unenriched rows get route B alone, since route A was never trained on
    (and has no business scoring) unenriched rows.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import storage

StageKind = Literal["single", "dual_route"]


def new_bundle_id() -> str:
    return uuid.uuid4().hex


def route_key(stage_name: str, route: Literal["a", "b"]) -> str:
    return f"{stage_name}::{route}"


# ---------------------------------------------------------------------------
# Fitted preprocessing state — the contract predict() is not allowed to break
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingState:
    # "{stage_name}" (single) or "{stage_name}::a" / "{stage_name}::b" (dual-route)
    # -> the exact ordered feature list that model was fit on.
    stage_features: dict[str, list[str]]
    categorical_vocab: dict[str, list[str]]  # column -> exact training-time category vocabulary
    numeric_dtypes: dict[str, str]  # column -> dtype coercion rule (e.g. "float64")
    blend_weights: dict[str, float] = field(default_factory=dict)  # stage_name -> weight on route a (dual-route only)
    enrichment_features: list[str] = field(default_factory=list)  # pinned list used to decide enriched vs not at predict time
    training_null_rates: dict[str, float] = field(default_factory=dict)  # column -> blank rate at training time (drift check)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingState":
        return cls(
            stage_features={k: list(v) for k, v in d["stage_features"].items()},
            categorical_vocab={k: list(v) for k, v in d["categorical_vocab"].items()},
            numeric_dtypes=dict(d["numeric_dtypes"]),
            blend_weights=dict(d.get("blend_weights", {})),
            enrichment_features=list(d.get("enrichment_features", [])),
            training_null_rates=dict(d.get("training_null_rates", {})),
        )

    def features_for(self, key: str) -> list[str]:
        if key not in self.stage_features:
            raise KeyError(f"no pinned feature list for {key!r}")
        return self.stage_features[key]


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
    feature_list: list[str]  # union of every feature used anywhere in the bundle (informational)
    hyperparams: dict
    stages: list[StageSpec]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# save_bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageInput:
    """One stage's fitted booster(s) plus the exact feature list(s) it was fit
    on, ready to be written into a bundle.
    """

    name: str
    booster: Optional[lgb.Booster] = None  # single-model stage
    features: Optional[list[str]] = None  # required when booster is set
    booster_a: Optional[lgb.Booster] = None  # dual-route stage, route a
    features_a: Optional[list[str]] = None  # required when booster_a is set
    booster_b: Optional[lgb.Booster] = None  # dual-route stage, route b
    features_b: Optional[list[str]] = None  # required when booster_b is set

    def __post_init__(self) -> None:
        if self.booster is not None and not self.features:
            raise ValueError(f"stage {self.name!r}: 'features' is required when 'booster' is set")
        if self.booster_a is not None and not self.features_a:
            raise ValueError(f"stage {self.name!r}: 'features_a' is required when 'booster_a' is set")
        if self.booster_b is not None and not self.features_b:
            raise ValueError(f"stage {self.name!r}: 'features_b' is required when 'booster_b' is set")


def _validate_pinned_features(preprocessing: PreprocessingState, key: str, features: list[str]) -> None:
    pinned = preprocessing.stage_features.get(key)
    if pinned is None:
        raise ValueError(f"preprocessing.stage_features is missing an entry for {key!r}")
    if list(pinned) != list(features):
        raise ValueError(f"preprocessing.stage_features[{key!r}] does not match the features that model was fit on")


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
        seen_features: list[str] = []

        def _track(features: list[str]) -> None:
            for f in features:
                if f not in seen_features:
                    seen_features.append(f)

        for i, stage in enumerate(stages, start=1):
            if stage.booster is not None:
                filename = f"stage_{i}.txt"
                stage.booster.save_model(str(tmp_dir / filename))
                stage_specs.append(StageSpec(name=stage.name, kind="single", file=filename))
                _validate_pinned_features(preprocessing, stage.name, stage.features)
                _track(stage.features)
            elif stage.booster_a is not None and stage.booster_b is not None:
                filename_a = f"stage_{i}_model_a.txt"
                filename_b = f"stage_{i}_model_b.txt"
                stage.booster_a.save_model(str(tmp_dir / filename_a))
                stage.booster_b.save_model(str(tmp_dir / filename_b))
                stage_specs.append(StageSpec(name=stage.name, kind="dual_route", file_a=filename_a, file_b=filename_b))
                _validate_pinned_features(preprocessing, route_key(stage.name, "a"), stage.features_a)
                _validate_pinned_features(preprocessing, route_key(stage.name, "b"), stage.features_b)
                _track(stage.features_a)
                _track(stage.features_b)
            else:
                raise ValueError(f"stage {stage.name!r} must set either 'booster', or both 'booster_a' and 'booster_b'")

        manifest = BundleManifest(
            bundle_id=bundle_id,
            architecture=architecture,
            created_at=storage.utcnow_iso(),
            dataset_content_hash=dataset_content_hash,
            split_id=split_id,
            target_config=target_config,
            feature_list=seen_features,
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

    @classmethod
    def from_stage_inputs(cls, preprocessing: PreprocessingState, stages: list[StageInput]) -> "ModelBundle":
        """Build an in-memory bundle directly from freshly-fit stage inputs,
        without touching disk. Used by app/core/train.py to evaluate a model
        immediately after fitting (ordinal-violation check, per-stage metrics)
        via the exact same predict() path a saved bundle would use, before
        deciding whether to persist it via save_bundle().
        """
        stage_specs: list[StageSpec] = []
        loaded_stages: list[_LoadedStage] = []
        for stage in stages:
            if stage.booster is not None:
                _validate_pinned_features(preprocessing, stage.name, stage.features)
                stage_specs.append(StageSpec(name=stage.name, kind="single", file="<in-memory>"))
                loaded_stages.append(_LoadedStage(name=stage.name, kind="single", booster=stage.booster))
            elif stage.booster_a is not None and stage.booster_b is not None:
                _validate_pinned_features(preprocessing, route_key(stage.name, "a"), stage.features_a)
                _validate_pinned_features(preprocessing, route_key(stage.name, "b"), stage.features_b)
                stage_specs.append(StageSpec(name=stage.name, kind="dual_route", file_a="<in-memory>", file_b="<in-memory>"))
                loaded_stages.append(
                    _LoadedStage(name=stage.name, kind="dual_route", booster_a=stage.booster_a, booster_b=stage.booster_b)
                )
            else:
                raise ValueError(f"stage {stage.name!r} must set either 'booster', or both 'booster_a' and 'booster_b'")

        manifest = BundleManifest(
            bundle_id="<in-memory>",
            architecture="<in-memory>",
            created_at=storage.utcnow_iso(),
            dataset_content_hash="<in-memory>",
            split_id="<in-memory>",
            target_config={},
            feature_list=[],
            hyperparams={},
            stages=stage_specs,
        )
        return cls(path=Path("<in-memory>"), manifest=manifest, preprocessing=preprocessing, metrics={}, stages=loaded_stages)

    def _prepare_features(self, df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
        missing = [c for c in feature_list if c not in df.columns]
        if missing:
            raise ValueError(f"prediction dataframe is missing required features: {missing}")

        # Select in the pinned order — this is what makes scoring invariant to
        # the incoming dataframe's column order (positional order is never used).
        X = df.loc[:, feature_list].copy()

        for col in feature_list:
            if col in self.preprocessing.categorical_vocab:
                vocab = self.preprocessing.categorical_vocab[col]
                # Any value not in the training-time vocabulary becomes NaN here,
                # i.e. an unseen category maps to "missing", never to a new category.
                X[col] = X[col].astype("category").cat.set_categories(vocab)
            elif col in self.preprocessing.numeric_dtypes:
                X[col] = X[col].astype(self.preprocessing.numeric_dtypes[col])
            else:
                raise ValueError(f"feature {col!r} has no pinned dtype rule (categorical_vocab or numeric_dtypes)")

        return X

    def _enriched_mask(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.preprocessing.enrichment_features if c in df.columns]
        if not cols:
            return np.ones(len(df), dtype=bool)
        return df[cols].notna().any(axis=1).to_numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score df. Returns a per-stage probability column for each stage plus
        an overall chained 'score' (the cumulative product across stages, i.e.
        P(reach final stage) = product of each stage's conditional probability).
        """
        result = pd.DataFrame(index=df.index)
        cumulative = np.ones(len(df))

        has_dual_route = any(s.kind == "dual_route" for s in self._stages)
        if has_dual_route and not self.preprocessing.enrichment_features:
            raise ValueError("this bundle has a dual-route stage but no enrichment_features are pinned")
        enriched_mask = self._enriched_mask(df) if has_dual_route else None

        for spec, loaded in zip(self.manifest.stages, self._stages):
            if loaded.kind == "single":
                X = self._prepare_features(df, self.preprocessing.features_for(spec.name))
                stage_prob = loaded.booster.predict(X)
            else:
                X_a = self._prepare_features(df, self.preprocessing.features_for(route_key(spec.name, "a")))
                X_b = self._prepare_features(df, self.preprocessing.features_for(route_key(spec.name, "b")))
                weight = self.preprocessing.blend_weights.get(spec.name)
                if weight is None:
                    raise ValueError(f"dual-route stage {spec.name!r} has no pinned blend weight")
                prob_a = loaded.booster_a.predict(X_a)
                prob_b = loaded.booster_b.predict(X_b)
                blended = weight * prob_a + (1.0 - weight) * prob_b
                # Route A was only ever trained on enriched rows; unenriched rows get route B alone.
                stage_prob = np.where(enriched_mask, blended, prob_b)

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
