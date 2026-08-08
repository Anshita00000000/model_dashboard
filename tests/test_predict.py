"""Tests for app/core/predict.py: pre-flight validation, scoring, tier stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from app.core import predict, registry
from app.core.storage import MetadataStore

FEATURES = ["numeric_feat", "cat_feat"]


def _train_and_bundle(tmp_path, n=600, seed=0):
    rng = np.random.default_rng(seed)
    categories = ["A", "B", "C"]
    numeric = rng.normal(size=n)
    cat = rng.choice(categories, size=n)
    target = ((numeric + (cat == "A").astype(float) + rng.normal(scale=0.5, size=n)) > 0.5).astype(int)
    df = pd.DataFrame({"numeric_feat": numeric, "cat_feat": cat, "target": target})

    X = df[FEATURES].copy()
    X["cat_feat"] = X["cat_feat"].astype("category").cat.set_categories(categories)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, "cat_smooth": 10},
        lgb.Dataset(X, label=df["target"], categorical_feature=["cat_feat"], free_raw_data=False),
        num_boost_round=20,
    )
    scores = booster.predict(X)

    preprocessing = registry.PreprocessingState(
        stage_features={"stage_1": FEATURES},
        categorical_vocab={"cat_feat": categories},
        numeric_dtypes={"numeric_feat": "float64"},
        training_null_rates={"numeric_feat": 0.0, "cat_feat": 0.0},
    )
    stage = registry.StageInput(name="stage_1", booster=booster, features=FEATURES)
    from app.core import evaluate as evaluate_mod

    cutoffs = evaluate_mod.tier_cutoffs(scores, n_tiles=10)
    bundle_path = registry.save_bundle(
        root=tmp_path, merchant="acme", bundle_id=registry.new_bundle_id(), architecture="baseline_anchor",
        stages=[stage], preprocessing=preprocessing, dataset_content_hash="x" * 8, split_id="split-1",
        target_config={}, hyperparams={}, metrics={"tier_cutoffs": {"stage_1": cutoffs}},
    )
    return registry.load_bundle(bundle_path), df, categories


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def test_validate_clean_batch_has_no_issues(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    report = predict.validate_for_prediction(df[FEATURES], bundle)
    assert report.issues == []
    assert report.is_scoreable


def test_validate_reports_missing_feature(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    incomplete = df[["numeric_feat"]]
    report = predict.validate_for_prediction(incomplete, bundle)
    assert not report.is_scoreable
    assert any(i.kind == "missing_feature" and i.column == "cat_feat" for i in report.errors)


def test_validate_reports_non_coercible_numeric(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    bad = df[FEATURES].copy()
    bad["numeric_feat"] = bad["numeric_feat"].astype(object)
    bad.loc[bad.index[:3], "numeric_feat"] = "not_a_number"
    report = predict.validate_for_prediction(bad, bundle)
    assert not report.is_scoreable
    hit = next(i for i in report.errors if i.kind == "dtype_not_coercible")
    assert hit.column == "numeric_feat"
    assert hit.details["n_failed"] == 3


def test_validate_reports_unseen_categories_as_warning_not_error(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    fresh = df[FEATURES].copy()
    fresh.loc[fresh.index[:5], "cat_feat"] = "NEW_CATEGORY"
    report = predict.validate_for_prediction(fresh, bundle)
    assert report.is_scoreable  # warning only, doesn't block
    hit = next(i for i in report.warnings if i.kind == "unseen_categories")
    assert hit.column == "cat_feat"
    assert hit.details["n_unseen"] == 5
    assert "NEW_CATEGORY" in hit.details["top_values"]


def test_validate_reports_null_rate_shift(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    drifted = df[FEATURES].copy()
    drifted.loc[drifted.index[: len(drifted) // 2], "numeric_feat"] = None  # 50% null, vs 0% at training
    report = predict.validate_for_prediction(drifted, bundle)
    hit = next(i for i in report.warnings if i.kind == "null_rate_shift" and i.column == "numeric_feat")
    assert hit.details["shift_pp"] == pytest.approx(0.5, abs=0.02)


def test_validate_reports_all_problems_at_once(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    bad = df[FEATURES].copy()
    bad["numeric_feat"] = bad["numeric_feat"].astype(object)
    bad.loc[bad.index[:3], "numeric_feat"] = "garbage"
    bad.loc[bad.index[3:8], "cat_feat"] = "UNSEEN"
    report = predict.validate_for_prediction(bad, bundle)
    kinds = {i.kind for i in report.issues}
    assert "dtype_not_coercible" in kinds
    assert "unseen_categories" in kinds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_batch_uses_only_pinned_state_and_includes_ids(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    predict_df = df[FEATURES].head(20).copy()
    predict_df["lead_id"] = range(20)

    output = predict.score_batch(predict_df, bundle, id_columns=["lead_id"], bundle_id="bundle-123")

    assert list(output["lead_id"]) == list(range(20))
    assert "stage_1_prob" in output.columns
    assert "stage_1_tier" in output.columns
    assert "score" in output.columns
    assert (output["bundle_id"] == "bundle-123").all()
    assert "scored_at" in output.columns
    assert output["stage_1_tier"].between(1, 10).all()


def test_score_batch_tier_assignment_stable_across_batch_composition(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    one_row = df[FEATURES].iloc[[0]].copy()
    one_row["lead_id"] = [1]

    big_batch = df[FEATURES].copy()
    big_batch["lead_id"] = range(len(big_batch))

    small_output = predict.score_batch(one_row, bundle, id_columns=["lead_id"], bundle_id="b1")
    big_output = predict.score_batch(big_batch, bundle, id_columns=["lead_id"], bundle_id="b1")

    tier_alone = small_output["stage_1_tier"].iloc[0]
    tier_in_big_batch = big_output.loc[big_output["lead_id"] == 0, "stage_1_tier"].iloc[0]
    assert tier_alone == tier_in_big_batch


def test_score_batch_missing_id_column_raises(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    with pytest.raises(ValueError):
        predict.score_batch(df[FEATURES], bundle, id_columns=["nonexistent_id"], bundle_id="b1")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_prediction_run_persists_artifact_and_metadata(tmp_path):
    bundle, df, categories = _train_and_bundle(tmp_path)
    predict_df = df[FEATURES].head(10).copy()
    predict_df["lead_id"] = range(10)
    scored = predict.score_batch(predict_df, bundle, id_columns=["lead_id"], bundle_id="bundle-123")

    store = MetadataStore(tmp_path / "meta.db")
    # prediction_runs has FK references to datasets/model_bundles; satisfy them first.
    store.insert_dataset(
        dataset_id="ds-1", merchant="acme", purpose="predict", source_file="x.csv", row_count=10, col_count=2,
        content_hash="h", schema_json="{}", artifact_path="dataset/acme/x.parquet",
    )
    store.insert_dataset(
        dataset_id="ds-train", merchant="acme", purpose="train", source_file="train.csv", row_count=600, col_count=3,
        content_hash="h2", schema_json="{}", artifact_path="dataset/acme/train.parquet",
    )
    store.insert_split(
        split_id="split-1", dataset_id="ds-train", strategy="random", config_json="{}",
        train_ids_path="x", test_ids_path="y", train_rows=480, test_rows=120,
    )
    store.insert_model_bundle(
        bundle_id="bundle-123", merchant="acme", dataset_id="ds-train", split_id="split-1", architecture="baseline_anchor",
        target_config_json="{}", feature_list_json="[]", hyperparams_json="{}", metrics_json="{}",
        bundle_path=str(bundle.path), created_by="test",
    )

    run_id, path = predict.save_prediction_run(
        store=store, root=tmp_path, merchant="acme", bundle_id="bundle-123", dataset_id="ds-1", scored_df=scored
    )
    assert path.exists()
    row = store.get_prediction_run(run_id)
    assert row["bundle_id"] == "bundle-123"
    assert row["row_count"] == 10
    assert row["output_path"] == str(path)
