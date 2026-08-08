"""Training orchestrator.

Ties app/core/{architectures,chain,leakage,evaluate,registry}.py together
into one run: given a chosen architecture and a training dataframe,
train_architecture() fits every conditional stage from chain.py (running the
static leakage gates automatically along the way) and returns everything
registry.save_bundle() needs. evaluate_run() then scores a held-out dataframe
(normally the split's test side) through the freshly-fit model — via an
in-memory registry.ModelBundle, so evaluation exercises the exact same
predict() path a saved bundle would — computing per-stage benchmark metrics,
tier tables, lift, capture thresholds, and running GATE 2 (implausible
performance) against real held-out numbers.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import architectures, chain, evaluate, leakage, registry
from . import schema as schema_mod


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageOutcome:
    name: str
    n_rows: int
    n_positive: int
    route_b_only: bool  # dual-route only: True if route A was skipped and this stage fell back to B alone
    feature_gain: pd.DataFrame


@dataclass(frozen=True)
class TrainRunResult:
    architecture: architectures.ArchitectureName
    hyperparams: dict
    blend_weight: Optional[float]
    stage_inputs: list[registry.StageInput]
    preprocessing: registry.PreprocessingState
    stage_outcomes: list[StageOutcome]
    pre_flight_gates: list[leakage.GateResult]
    warnings: list[str]


def _build_categorical_vocab(df: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    return {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in categorical_features}


def _prepare_fit_frame(
    df: pd.DataFrame, features: list[str], categorical_vocab: dict[str, list[str]], numeric_dtypes: dict[str, str]
) -> pd.DataFrame:
    X = df.loc[:, features].copy()
    for col in features:
        if col in categorical_vocab:
            X[col] = X[col].astype("category").cat.set_categories(categorical_vocab[col])
        elif col in numeric_dtypes:
            X[col] = X[col].astype(numeric_dtypes[col])
    return X


def _fit_booster(X: pd.DataFrame, y: np.ndarray, categorical_features: list[str], hyperparams: dict, random_seed: int) -> lgb.Booster:
    cat_cols = [c for c in X.columns if c in categorical_features]
    params = dict(hyperparams)
    n_estimators = int(params.pop("n_estimators", 300))
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    params.setdefault("seed", random_seed)
    dataset = lgb.Dataset(X, label=y, categorical_feature=cat_cols or "auto", free_raw_data=False)
    return lgb.train(params, dataset, num_boost_round=n_estimators)


def train_architecture(
    train_df: pd.DataFrame,
    *,
    funnel: chain.FunnelConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    feature_sources: dict[str, str],
    architecture: architectures.ArchitectureName,
    hyperparams: dict,
    blend_weight: float = architectures.DEFAULT_BLEND_WEIGHT,
    target_derived_columns: Optional[list[str]] = None,
    provenance_column: Optional[str] = None,
    formula_invariants: Optional[list[leakage.FormulaInvariant]] = None,
    source_column: Optional[str] = None,
    post_event_patterns: tuple[str, ...] = leakage.DEFAULT_POST_EVENT_PATTERNS,
    min_stage_positives: int = chain.MIN_STAGE_POSITIVES_WARN,
    random_seed: int = 42,
) -> TrainRunResult:
    """Fit one architecture across every conditional stage of `funnel`.

    Raises leakage.LeakageGateFailure if GATE 1 (target leakage) fails — this
    gate has no override. Raises AssertionError if the resulting chained
    scores have any ordinal violation.
    """
    all_candidate_features = list(numeric_features) + list(categorical_features)
    hyperparams = architectures.with_mandatory_defaults(hyperparams)

    pre_flight: list[leakage.GateResult] = [
        leakage.gate_target_leakage([funnel.target_column], all_candidate_features, target_derived_columns),
        leakage.gate_post_event_features(all_candidate_features, post_event_patterns),
    ]
    if formula_invariants:
        pre_flight.extend(leakage.gate_formula_consistency(train_df, formula_invariants, source_column))
    provenance_result = leakage.gate_provenance_detection(
        train_df, all_candidate_features, provenance_column, random_seed=random_seed
    )
    if provenance_result is not None:
        pre_flight.append(provenance_result)

    enrichment_features = [c for c in all_candidate_features if feature_sources.get(c) == "enrichment"]

    stages = chain.build_stage_data(train_df, funnel)
    run_warnings = list(chain.undertrained_warnings(stages, min_stage_positives))

    used_features: set[str] = set()
    if architecture == "dual_route_cascade":
        used_features |= set(architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources, route="a"))
        used_features |= set(architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources, route="b"))
    else:
        used_features |= set(architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources))

    categorical_vocab = _build_categorical_vocab(train_df, [c for c in categorical_features if c in used_features])
    numeric_dtypes = {c: "float64" for c in numeric_features if c in used_features}
    training_null_rates = {c: schema_mod.blank_rate(train_df[c]) for c in used_features}

    stage_inputs: list[registry.StageInput] = []
    stage_features_map: dict[str, list[str]] = {}
    blend_weights: dict[str, float] = {}
    stage_outcomes: list[StageOutcome] = []

    for stage in stages:
        stage_df = train_df.iloc[stage.positions]
        y = stage.y
        route_b_only = False

        if architecture in ("baseline_anchor", "unified_native_sparse"):
            features = architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources)
            X = _prepare_fit_frame(stage_df, features, categorical_vocab, numeric_dtypes)
            booster = _fit_booster(X, y, categorical_features, hyperparams, random_seed)

            stage_inputs.append(registry.StageInput(name=stage.name, booster=booster, features=features))
            stage_features_map[stage.name] = features
            gain_df = evaluate.feature_gain_table(stage.name, booster, feature_sources)

        else:  # dual_route_cascade
            features_b = architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources, route="b")
            features_a = architectures.select_features(architecture, all_features=all_candidate_features, feature_sources=feature_sources, route="a")

            X_b = _prepare_fit_frame(stage_df, features_b, categorical_vocab, numeric_dtypes)
            booster_b = _fit_booster(X_b, y, categorical_features, hyperparams, random_seed)

            enriched_mask = architectures.enriched_row_mask(stage_df, enrichment_features).to_numpy()
            n_enriched = int(enriched_mask.sum())
            y_enriched = y[enriched_mask]
            enough_rows = n_enriched >= architectures.MIN_ENRICHED_ROWS_FOR_MODEL_A
            enough_classes = len(np.unique(y_enriched)) >= 2 if n_enriched else False
            route_b_only = not (enough_rows and enough_classes)

            if route_b_only:
                reason = (
                    f"only {n_enriched} enriched row(s) (< {architectures.MIN_ENRICHED_ROWS_FOR_MODEL_A})"
                    if not enough_rows
                    else "the enriched subset has only one class present"
                )
                run_warnings.append(f"stage {stage.name!r}: skipping route A ({reason}); falling back to route B alone")
                stage_inputs.append(registry.StageInput(name=stage.name, booster=booster_b, features=features_b))
                stage_features_map[stage.name] = features_b
                gain_df = evaluate.feature_gain_table(stage.name, booster_b, feature_sources)
            else:
                X_a = _prepare_fit_frame(stage_df[enriched_mask], features_a, categorical_vocab, numeric_dtypes)
                booster_a = _fit_booster(X_a, y_enriched, categorical_features, hyperparams, random_seed)

                stage_inputs.append(
                    registry.StageInput(
                        name=stage.name, booster_a=booster_a, features_a=features_a, booster_b=booster_b, features_b=features_b
                    )
                )
                stage_features_map[registry.route_key(stage.name, "a")] = features_a
                stage_features_map[registry.route_key(stage.name, "b")] = features_b
                blend_weights[stage.name] = blend_weight
                gain_df = pd.concat(
                    [
                        evaluate.feature_gain_table(f"{stage.name} (route A)", booster_a, feature_sources),
                        evaluate.feature_gain_table(f"{stage.name} (route B)", booster_b, feature_sources),
                    ],
                    ignore_index=True,
                )

        stage_outcomes.append(
            StageOutcome(
                name=stage.name, n_rows=stage.n_rows, n_positive=stage.n_positive, route_b_only=route_b_only, feature_gain=gain_df
            )
        )

    preprocessing = registry.PreprocessingState(
        stage_features=stage_features_map,
        categorical_vocab=categorical_vocab,
        numeric_dtypes=numeric_dtypes,
        blend_weights=blend_weights,
        enrichment_features=enrichment_features,
        training_null_rates=training_null_rates,
    )

    # Sanity check: score the training population itself and assert the chained
    # scores are non-increasing stage over stage (see chain.py docstring).
    bundle = registry.ModelBundle.from_stage_inputs(preprocessing, stage_inputs)
    scored = bundle.predict(train_df)
    cumulative_by_stage: dict[str, np.ndarray] = {}
    cumulative = np.ones(len(train_df))
    for stage_name in funnel.stage_names:
        cumulative = cumulative * scored[f"{stage_name}_prob"].to_numpy()
        cumulative_by_stage[stage_name] = cumulative.copy()
    chain.assert_no_ordinal_violations(cumulative_by_stage, funnel.stage_names)

    return TrainRunResult(
        architecture=architecture,
        hyperparams=hyperparams,
        blend_weight=blend_weight if architecture == "dual_route_cascade" else None,
        stage_inputs=stage_inputs,
        preprocessing=preprocessing,
        stage_outcomes=stage_outcomes,
        pre_flight_gates=pre_flight,
        warnings=run_warnings,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageEvaluation:
    stage: str
    metrics: evaluate.BenchmarkMetrics
    tier_table: pd.DataFrame
    lift: evaluate.LiftResult
    capture_thresholds: dict[float, float]
    gate2: leakage.GateResult
    tier_cutoffs: list[float]


@dataclass(frozen=True)
class EvaluationResult:
    stage_evaluations: list[StageEvaluation]


def evaluate_run(
    result: TrainRunResult,
    eval_df: pd.DataFrame,
    funnel: chain.FunnelConfig,
    *,
    n_tiles: int = evaluate.DEFAULT_N_TILES,
    capture_targets: tuple[float, ...] = evaluate.DEFAULT_CAPTURE_TARGETS,
    allow_implausible_override: bool = False,
    random_seed: int = 42,
) -> EvaluationResult:
    """Score eval_df (normally the split's held-out test side) through the
    freshly-fit model and compute per-stage metrics. Each stage is evaluated
    on its OWN eligible population (the same rows it would have trained on,
    per chain.py) using its marginal stage probability — this is what "actual
    rate per stage" means: conditional on having reached that stage.

    Raises leakage.LeakageGateFailure (GATE 2) if any stage's ROC-AUC is
    implausibly high for its base rate and allow_implausible_override=False.
    """
    bundle = registry.ModelBundle.from_stage_inputs(result.preprocessing, result.stage_inputs)
    scored = bundle.predict(eval_df)

    cumulative_by_stage: dict[str, np.ndarray] = {}
    cumulative = np.ones(len(eval_df))
    for stage_name in funnel.stage_names:
        cumulative = cumulative * scored[f"{stage_name}_prob"].to_numpy()
        cumulative_by_stage[stage_name] = cumulative.copy()
    chain.assert_no_ordinal_violations(cumulative_by_stage, funnel.stage_names)

    eval_stages = chain.build_stage_data(eval_df, funnel)

    stage_evals: list[StageEvaluation] = []
    for eval_stage in eval_stages:
        full_stage_score = scored[f"{eval_stage.name}_prob"].to_numpy()
        y_true = eval_stage.y
        y_score = full_stage_score[eval_stage.positions]

        metrics = evaluate.benchmark_metrics(eval_stage.name, y_true, y_score)
        tiers = evaluate.tier_table(y_true, y_score, n_tiles=n_tiles)
        lift = evaluate.compute_lift(y_true, y_score, n_tiles=n_tiles, random_seed=random_seed)
        capture = evaluate.capture_thresholds(y_true, y_score, targets=capture_targets)
        gate2 = leakage.gate_implausible_performance(metrics.roc_auc, metrics.base_rate, allow_override=allow_implausible_override)
        # Cutoffs are computed on the UNCONDITIONAL score across all of eval_df, not just
        # this stage's eligible subset: a future prediction batch has no ground-truth
        # eligibility to condition on, so cutoffs must match the population predict()
        # actually scores at serving time.
        cutoffs = evaluate.tier_cutoffs(full_stage_score, n_tiles=n_tiles)

        stage_evals.append(
            StageEvaluation(
                stage=eval_stage.name, metrics=metrics, tier_table=tiers, lift=lift,
                capture_thresholds=capture, gate2=gate2, tier_cutoffs=cutoffs,
            )
        )

    return EvaluationResult(stage_evaluations=stage_evals)


# ---------------------------------------------------------------------------
# Serialization for storage (registry.save_bundle()'s metrics.json)
# ---------------------------------------------------------------------------


def serialize_evaluation(evaluation: EvaluationResult) -> dict:
    return {
        se.stage: {
            "metrics": se.metrics.to_dict(),
            "tier_table": se.tier_table.to_dict(orient="records"),
            "lift": {
                "lift_vs_base_rate": se.lift.lift_vs_base_rate,
                "top_bottom_ratio": se.lift.top_bottom_ratio,
                "top_bottom_ratio_ci": list(se.lift.top_bottom_ratio_ci) if se.lift.top_bottom_ratio_ci else None,
            },
            "capture_thresholds": {str(k): v for k, v in se.capture_thresholds.items()},
            "gate2": {"passed": se.gate2.passed, "message": se.gate2.message, "details": se.gate2.details},
        }
        for se in evaluation.stage_evaluations
    }


def serialize_stage_outcomes(result: TrainRunResult) -> dict:
    return {
        o.name: {
            "n_rows": o.n_rows,
            "n_positive": o.n_positive,
            "route_b_only": o.route_b_only,
            "feature_gain": o.feature_gain.to_dict(orient="records"),
        }
        for o in result.stage_outcomes
    }


def serialize_tier_cutoffs(evaluation: EvaluationResult) -> dict[str, list[float]]:
    """Top-level, stable lookup for app/core/predict.py: {stage_name: cutoffs}.
    Stored as bundle metrics["tier_cutoffs"] at save time (see app/ui/tab_training.py).
    """
    return {se.stage: se.tier_cutoffs for se in evaluation.stage_evaluations}
