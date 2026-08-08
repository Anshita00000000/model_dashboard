"""Tests for app/core/registry.py — the model bundle contract.

These prove the three guarantees the bundle format exists for:
  1. save -> load reproduces byte-identical predictions (no drift on reload).
  2. a category unseen at training time scores without error (maps to null,
     never to a new category, never a refit).
  3. the ordered feature list is applied, not positional column order.
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


def _train_booster(df: pd.DataFrame, categories: list[str]) -> lgb.Booster:
    X = df[FEATURES].copy()
    X["cat_feat"] = X["cat_feat"].astype("category").cat.set_categories(categories)
    dataset = lgb.Dataset(X, label=df["target"], categorical_feature=["cat_feat"], free_raw_data=False)
    params = {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, "cat_smooth": 10, "seed": 0}
    return lgb.train(params, dataset, num_boost_round=20)


def _make_preprocessing(categories: list[str]) -> registry.PreprocessingState:
    return registry.PreprocessingState(
        ordered_features=FEATURES,
        categorical_vocab={"cat_feat": categories},
        numeric_dtypes={"numeric_feat": "float64"},
    )


def _save_test_bundle(tmp_path, booster, preprocessing):
    stage = registry.StageInput(name="stage_1", booster=booster)
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
    booster = _train_booster(df, categories)
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
    booster = _train_booster(df, categories)
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
    booster = _train_booster(df, categories)
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
    booster = _train_booster(df, categories)
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
    booster = _train_booster(df, categories)
    preprocessing = _make_preprocessing(categories)

    bundle_path = _save_test_bundle(tmp_path, booster, preprocessing)
    manifest_before = (bundle_path / "manifest.json").read_bytes()

    registry.load_bundle(bundle_path).predict(df[FEATURES].head(5))

    manifest_after = (bundle_path / "manifest.json").read_bytes()
    assert manifest_before == manifest_after
