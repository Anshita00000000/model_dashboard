"""Integration tests for app/core/train.py: the orchestrator tying together
architectures, chained scoring, leakage gates, evaluation, and the bundle
registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import chain, features, leakage, registry, train

CANDIDATE_FEATURES = ["lead_source", "city", "days_since_creation", "crif_score", "equifax_flag"]


def _make_synthetic_df(n: int = 2000, seed: int = 0, enriched_frac: float = 0.45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lead_source = rng.choice(["FB Ads", "Google", "Organic", "Referral"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    city = rng.choice(["Mumbai", "Delhi", "Bangalore"], size=n)
    days_since_creation = rng.integers(0, 90, size=n).astype(float)

    enriched_mask = rng.random(n) < enriched_frac
    crif_score = np.where(enriched_mask, rng.normal(600, 80, n), np.nan)
    equifax_flag = np.where(enriched_mask, rng.choice(["good", "fair", "poor"], size=n), None)

    source_boost = np.select([lead_source == "Referral", lead_source == "Google"], [0.6, 0.2], default=0.0)
    engaged_logit = -0.4 + source_boost + 0.01 * (90 - days_since_creation) + rng.normal(scale=1.2, size=n)
    engaged = (engaged_logit > 0.4).astype(int)

    credit_boost = np.where(enriched_mask, (np.nan_to_num(crif_score, nan=600.0) - 600) / 200, 0.0)
    converted_logit = -0.5 + credit_boost + source_boost * 0.3 + rng.normal(scale=1.3, size=n)
    converted = np.where(engaged == 1, (converted_logit > 0.4).astype(int), 0)

    disposition = np.where(converted == 1, "Converted", np.where(engaged == 1, "Engaged", "Lead"))

    return pd.DataFrame(
        {
            "lead_id": np.arange(n),
            "lead_source": lead_source,
            "city": city,
            "days_since_creation": days_since_creation,
            "crif_score": crif_score,
            "equifax_flag": equifax_flag,
            "disposition": disposition,
        }
    )


def _funnel(df: pd.DataFrame) -> chain.FunnelConfig:
    return chain.build_funnel_config(df, target_column="disposition", level_order=["Lead", "Engaged", "Converted"])


def _feature_types(df: pd.DataFrame):
    types = features.classify_numeric_categorical(df, CANDIDATE_FEATURES)
    sources = features.classify_feature_sources(CANDIDATE_FEATURES)
    return types, sources


def test_baseline_anchor_uses_core_features_only(tmp_path):
    df = _make_synthetic_df()
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df,
        funnel=funnel,
        numeric_features=types.numeric,
        categorical_features=types.categorical,
        feature_sources=sources,
        architecture="baseline_anchor",
        hyperparams={},
    )

    assert len(result.stage_inputs) == funnel.n_stages == 2
    for stage_input in result.stage_inputs:
        assert stage_input.booster is not None  # single-model stage, no dual route
        assert "crif_score" not in stage_input.features
        assert "equifax_flag" not in stage_input.features

    # round-trips through the real bundle registry and scores without error
    bundle_path = registry.save_bundle(
        root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="baseline_anchor",
        stages=result.stage_inputs, preprocessing=result.preprocessing, dataset_content_hash="x" * 8,
        split_id="split-1", target_config={"target_column": "disposition"}, hyperparams=result.hyperparams,
        metrics={},
    )
    bundle = registry.load_bundle(bundle_path)
    scored = bundle.predict(df)
    for stage_name in funnel.stage_names:
        assert (scored[f"{stage_name}_prob"] >= 0).all() and (scored[f"{stage_name}_prob"] <= 1).all()
    assert (scored["score"] <= scored[f"{funnel.stage_names[0]}_prob"] + 1e-9).all()


def test_unified_native_sparse_uses_all_features():
    df = _make_synthetic_df()
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="unified_native_sparse", hyperparams={},
    )
    for stage_input in result.stage_inputs:
        assert "crif_score" in stage_input.features
        assert "equifax_flag" in stage_input.features
        assert stage_input.booster is not None


def test_dual_route_cascade_trains_both_routes_when_enough_enriched_rows(tmp_path):
    df = _make_synthetic_df(n=3000, enriched_frac=0.45)
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="dual_route_cascade", hyperparams={}, blend_weight=0.65,
    )

    assert result.blend_weight == 0.65
    for stage_input, outcome in zip(result.stage_inputs, result.stage_outcomes):
        assert stage_input.booster_a is not None
        assert stage_input.booster_b is not None
        assert "crif_score" in stage_input.features_a
        assert "crif_score" not in stage_input.features_b
        assert outcome.route_b_only is False

    bundle_path = registry.save_bundle(
        root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="dual_route_cascade",
        stages=result.stage_inputs, preprocessing=result.preprocessing, dataset_content_hash="x" * 8,
        split_id="split-1", target_config={}, hyperparams=result.hyperparams, metrics={},
    )
    bundle = registry.load_bundle(bundle_path)
    scored = bundle.predict(df)
    assert scored["score"].between(0, 1).all()


def test_dual_route_cascade_falls_back_to_b_when_enriched_subset_too_small():
    df = _make_synthetic_df(n=500, enriched_frac=0.02)  # ~10 enriched rows, well under the 30-row floor
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="dual_route_cascade", hyperparams={},
    )

    stage_1_input = result.stage_inputs[0]
    stage_1_outcome = result.stage_outcomes[0]
    assert stage_1_outcome.route_b_only is True
    assert stage_1_input.booster_a is None
    assert stage_1_input.booster is not None  # fell back to a plain single-model stage using route B
    assert any("skipping route A" in w for w in result.warnings)


def test_gate1_blocks_when_target_column_is_a_feature():
    df = _make_synthetic_df()
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    with pytest.raises(leakage.LeakageGateFailure) as exc_info:
        train.train_architecture(
            df, funnel=funnel, numeric_features=types.numeric,
            categorical_features=types.categorical + ["disposition"],  # target leaked into features
            feature_sources=sources, architecture="baseline_anchor", hyperparams={},
        )
    assert exc_info.value.gate == "target_leakage"


def test_undertrained_stage_warns():
    # a tiny second-stage positive count triggers chain.py's undertrained warning
    df = _make_synthetic_df(n=300, seed=7)
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="baseline_anchor", hyperparams={},
    )
    n_positive_by_stage = {o.name: o.n_positive for o in result.stage_outcomes}
    if any(n < chain.MIN_STAGE_POSITIVES_WARN for n in n_positive_by_stage.values()):
        assert any("undertrained" in w for w in result.warnings)


def test_evaluate_run_produces_per_stage_report():
    df = _make_synthetic_df(n=3000)
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    train_df = df.iloc[:2400]
    test_df = df.iloc[2400:]

    result = train.train_architecture(
        train_df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="unified_native_sparse", hyperparams={},
    )
    evaluation = train.evaluate_run(result, test_df, funnel, allow_implausible_override=True)

    assert len(evaluation.stage_evaluations) == funnel.n_stages
    for stage_eval in evaluation.stage_evaluations:
        assert stage_eval.metrics.n_rows > 0
        assert len(stage_eval.tier_table) == 10
        assert stage_eval.tier_table["n"].sum() == stage_eval.metrics.n_rows
        assert stage_eval.capture_thresholds  # non-empty dict
        assert stage_eval.gate2.gate == "implausible_performance"


def test_serialize_evaluation_and_stage_outcomes_are_json_safe():
    import json

    df = _make_synthetic_df(n=3000)
    funnel = _funnel(df)
    types, sources = _feature_types(df)
    train_df, test_df = df.iloc[:2400], df.iloc[2400:]

    result = train.train_architecture(
        train_df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="baseline_anchor", hyperparams={},
    )
    evaluation = train.evaluate_run(result, test_df, funnel, allow_implausible_override=True)

    payload = {
        "evaluation": train.serialize_evaluation(evaluation),
        "training": train.serialize_stage_outcomes(result),
    }
    json.dumps(payload)  # must not raise
    assert set(payload["evaluation"].keys()) == set(funnel.stage_names)
    assert set(payload["training"].keys()) == set(funnel.stage_names)


def test_feature_gain_reports_core_vs_enrichment_source():
    df = _make_synthetic_df(n=3000)
    funnel = _funnel(df)
    types, sources = _feature_types(df)

    result = train.train_architecture(
        df, funnel=funnel, numeric_features=types.numeric, categorical_features=types.categorical,
        feature_sources=sources, architecture="unified_native_sparse", hyperparams={},
    )
    for outcome in result.stage_outcomes:
        assert set(outcome.feature_gain["source"]) <= {"core", "enrichment"}
        assert "crif_score" in outcome.feature_gain["feature"].tolist()
