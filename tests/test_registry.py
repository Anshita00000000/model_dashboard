"""Tests for app/core/registry.py — the model bundle contract.

These prove the guarantees the bundle format exists for:
  1. save -> load reproduces byte-identical predictions (no drift on reload).
  2. a category unseen at training time scores without error (maps to null,
     never to a new category, never a refit).
  3. the ordered feature list is applied per stage/route, not positional order.
  4. dual-route stages blend A+B only for enriched rows; unenriched rows get
     route B alone, even though route A was never trained on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from app.core import registry

FEATURES = ["numeric_feat", "cat_feat"]


def _make_training_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    categories = ["A", "B", "C"]
    numeric = rng.normal(size=n)
    cat = rng.choice(categories, size=n)
    signal = numeric + (cat == "A").astype(float) + rng.normal(scale=0.5, size=n)
    target = (signal > 0.5).astype(int)
    df = pd.DataFrame({"numeric_feat": numeric, "cat_feat": cat, "target": target})
    return df, categories


def _train_booster(df: pd.DataFrame, features: list[str], categories: list[str]) -> lgb.Booster:
    X = df[features].copy()
    if "cat_feat" in features:
        X["cat_feat"] = X["cat_feat"].astype("category").cat.set_categories(categories)
    dataset = lgb.Dataset(X, label=df["target"], categorical_feature=["cat_feat"] if "cat_feat" in features else "auto", free_raw_data=False)
    params = {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, "cat_smooth": 10, "seed": 0}
    return lgb.train(params, dataset, num_boost_round=20)


def _make_preprocessing(categories: list[str]) -> registry.PreprocessingState:
    return registry.PreprocessingState(
        stage_features={"stage_1": FEATURES},
        categorical_vocab={"cat_feat": categories},
        numeric_dtypes={"numeric_feat": "float64"},
    )


def _save_test_bundle(tmp_path, booster, preprocessing):
    stage = registry.StageInput(name="stage_1", booster=booster, features=FEATURES)
    return registry.save_bundle(
        root=tmp_path,
        merchant="acme",
        bundle_id=registry.new_bundle_id(),
        architecture="single_stage",
        stages=[stage],
        preprocessing=preprocessing,
        dataset_content_hash="deadbeef" * 8,
        split_id="split-1",
        target_config={"stage_1": {"target_column": "target", "positive_label": 1}},
        hyperparams={"num_boost_round": 20},
        metrics={"stage_1": {"auc": 0.7}},
    )


def test_save_then_load_gives_byte_identical_predictions(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    bundle = registry.load_bundle(bundle_path)

    predict_df = df[FEATURES].head(30)

    # What predicting "by hand" with the original in-memory booster looks like.
    hand_rolled = predict_df.copy()
    hand_rolled["cat_feat"] = hand_rolled["cat_feat"].astype("category").cat.set_categories(categories)
    direct_preds = booster.predict(hand_rolled)

    loaded_preds = bundle.predict(predict_df)["stage_1_prob"].to_numpy()

    assert np.array_equal(direct_preds, loaded_preds)


def test_unseen_category_scores_without_error(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    bundle = registry.load_bundle(bundle_path)

    predict_df = df[FEATURES].head(10).copy()
    predict_df.loc[predict_df.index[0], "cat_feat"] = "NEVER_SEEN_AT_TRAINING_TIME"

    result = bundle.predict(predict_df)

    assert len(result) == 10
    assert result["score"].notna().all()
    assert np.isfinite(result["score"].to_numpy()).all()

    # The unseen value must map to null (not to a new/nearby category): scoring
    # that row through the pinned vocab directly should agree with the bundle.
    manual = predict_df.copy()
    manual["cat_feat"] = manual["cat_feat"].astype("category").cat.set_categories(categories)
    assert manual["cat_feat"].isna().iloc[0]
    manual_preds = booster.predict(manual)
    assert np.array_equal(manual_preds, result["stage_1_prob"].to_numpy())


def test_column_order_invariance(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    bundle = registry.load_bundle(bundle_path)

    predict_df = df[FEATURES].head(15).copy()
    # Reversed column order, plus an extra column the bundle never asked for.
    reordered_df = predict_df[["cat_feat", "numeric_feat"]].copy()
    reordered_df["extra_untrained_col"] = "ignored"

    preds_canonical = bundle.predict(predict_df)["score"].to_numpy()
    preds_reordered = bundle.predict(reordered_df)["score"].to_numpy()

    assert np.array_equal(preds_canonical, preds_reordered)


def test_missing_required_feature_raises(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    bundle = registry.load_bundle(bundle_path)

    incomplete_df = df[["numeric_feat"]].head(5)
    try:
        bundle.predict(incomplete_df)
        assert False, "expected a ValueError for a missing required feature"
    except ValueError as exc:
        assert "cat_feat" in str(exc)


def test_bundle_directory_is_immutable_on_disk(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    manifest_before = (bundle_path / "manifest.json").read_bytes()

    registry.load_bundle(bundle_path).predict(df[FEATURES].head(5))

    manifest_after = (bundle_path / "manifest.json").read_bytes()
    assert manifest_before == manifest_after


def test_save_bundle_rejects_feature_list_mismatch(tmp_path):
    df, categories = _make_training_frame()
    booster = _train_booster(df, FEATURES, categories)
    preprocessing = registry.PreprocessingState(
        stage_features={"stage_1": ["numeric_feat"]},  # deliberately wrong: missing cat_feat
        categorical_vocab={"cat_feat": categories},
        numeric_dtypes={"numeric_feat": "float64"},
    )
    stage = registry.StageInput(name="stage_1", booster=booster, features=FEATURES)
    try:
        registry.save_bundle(
            root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="single_stage",
            stages=[stage], preprocessing=preprocessing, dataset_content_hash="x" * 8, split_id="split-1",
            target_config={}, hyperparams={}, metrics={},
        )
        assert False, "expected a ValueError for mismatched pinned features"
    except ValueError as exc:
        assert "stage_1" in str(exc)


# ---------------------------------------------------------------------------
# Dual-route: enriched rows blend A+B, unenriched rows get B alone
# ---------------------------------------------------------------------------


def _make_dual_route_frame(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    core_numeric = rng.normal(size=n)
    # enrichment feature: null for ~50% of rows ("no bureau data")
    enriched_mask = rng.random(n) < 0.5
    enrichment_feat = np.where(enriched_mask, rng.normal(loc=1.0, size=n), np.nan)
    target = (core_numeric + np.nan_to_num(enrichment_feat, nan=0.0) + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    df = pd.DataFrame({"core_numeric": core_numeric, "enrichment_feat": enrichment_feat, "target": target})
    return df, enriched_mask


def test_dual_route_blends_only_enriched_rows(tmp_path):
    df, enriched_mask = _make_dual_route_frame()

    features_b = ["core_numeric"]
    features_a = ["core_numeric", "enrichment_feat"]

    enriched_df = df[enriched_mask]
    # Train route A on enriched rows only, all features; route B on all rows, core only.
    X_a = enriched_df[features_a]
    dataset_a = lgb.Dataset(X_a, label=enriched_df["target"], free_raw_data=False)
    booster_a = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, "cat_smooth": 10}, dataset_a, num_boost_round=20)

    X_b = df[features_b]
    dataset_b = lgb.Dataset(X_b, label=df["target"], free_raw_data=False)
    booster_b = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, "cat_smooth": 10}, dataset_b, num_boost_round=20)

    blend_weight = 0.65
    preprocessing = registry.PreprocessingState(
        stage_features={
            registry.route_key("stage_1", "a"): features_a,
            registry.route_key("stage_1", "b"): features_b,
        },
        categorical_vocab={},
        numeric_dtypes={"core_numeric": "float64", "enrichment_feat": "float64"},
        blend_weights={"stage_1": blend_weight},
        enrichment_features=["enrichment_feat"],
    )
    stage = registry.StageInput(name="stage_1", booster_a=booster_a, features_a=features_a, booster_b=booster_b, features_b=features_b)
    bundle_path = registry.save_bundle(
        root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="dual_route_cascade",
        stages=[stage], preprocessing=preprocessing, dataset_content_hash="x" * 8, split_id="split-1",
        target_config={}, hyperparams={}, metrics={},
    )
    bundle = registry.load_bundle(bundle_path)

    result = bundle.predict(df)

    prob_a_all = booster_a.predict(df[features_a])
    prob_b_all = booster_b.predict(df[features_b])
    expected_blended = blend_weight * prob_a_all + (1 - blend_weight) * prob_b_all

    expected = np.where(enriched_mask, expected_blended, prob_b_all)
    assert np.allclose(result["stage_1_prob"].to_numpy(), expected)

    # unenriched rows must exactly equal route B alone (not touched by route A at all)
    unenriched_result = result.loc[~enriched_mask, "stage_1_prob"].to_numpy()
    unenriched_b_only = prob_b_all[~enriched_mask]
    assert np.array_equal(unenriched_result, unenriched_b_only)


def test_dual_route_requires_enrichment_features_pinned(tmp_path):
    df, enriched_mask = _make_dual_route_frame()
    features_b = ["core_numeric"]
    features_a = ["core_numeric", "enrichment_feat"]
    X_a = df[enriched_mask][features_a]
    booster_a = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X_a, label=df[enriched_mask]["target"]), num_boost_round=5)
    X_b = df[features_b]
    booster_b = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X_b, label=df["target"]), num_boost_round=5)

    preprocessing = registry.PreprocessingState(
        stage_features={
            registry.route_key("stage_1", "a"): features_a,
            registry.route_key("stage_1", "b"): features_b,
        },
        categorical_vocab={},
        numeric_dtypes={"core_numeric": "float64", "enrichment_feat": "float64"},
        blend_weights={"stage_1": 0.5},
        enrichment_features=[],  # deliberately not pinned
    )
    stage = registry.StageInput(name="stage_1", booster_a=booster_a, features_a=features_a, booster_b=booster_b, features_b=features_b)
    bundle_path = registry.save_bundle(
        root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="dual_route_cascade",
        stages=[stage], preprocessing=preprocessing, dataset_content_hash="x" * 8, split_id="split-1",
        target_config={}, hyperparams={}, metrics={},
    )
    bundle = registry.load_bundle(bundle_path)
    try:
        bundle.predict(df)
        assert False, "expected a ValueError when enrichment_features is not pinned"
    except ValueError as exc:
        assert "enrichment_features" in str(exc)
