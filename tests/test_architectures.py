from __future__ import annotations

import pandas as pd
import pytest

from app.core import architectures


def test_default_hyperparams_include_mandatory_defaults():
    for arch in architectures.ARCHITECTURES:
        params = architectures.default_hyperparams(arch)
        assert params["cat_smooth"] == 50.0
        assert params["min_data_per_group"] == 50


def test_default_hyperparams_match_spec_values():
    baseline = architectures.default_hyperparams("baseline_anchor")
    assert baseline["learning_rate"] == 0.05
    assert baseline["max_depth"] == 6
    assert baseline["num_leaves"] == 31
    assert baseline["max_bin"] == 255
    assert baseline["n_estimators"] == 300

    sparse = architectures.default_hyperparams("unified_native_sparse")
    assert sparse["max_bin"] == 128
    assert sparse["min_child_samples"] == 100
    assert sparse["colsample_bytree"] == 0.6
    assert sparse["reg_lambda"] == 10.0


def test_with_mandatory_defaults_does_not_clobber_user_override():
    merged = architectures.with_mandatory_defaults({"cat_smooth": 5.0})
    assert merged["cat_smooth"] == 5.0
    assert merged["min_data_per_group"] == 50


def test_select_features_baseline_anchor_is_core_only():
    all_features = ["lead_source", "city", "crif_score", "equifax_score"]
    sources = {"lead_source": "core", "city": "core", "crif_score": "enrichment", "equifax_score": "enrichment"}
    selected = architectures.select_features("baseline_anchor", all_features=all_features, feature_sources=sources)
    assert selected == ["lead_source", "city"]


def test_select_features_unified_native_sparse_is_everything():
    all_features = ["lead_source", "crif_score"]
    sources = {"lead_source": "core", "crif_score": "enrichment"}
    selected = architectures.select_features("unified_native_sparse", all_features=all_features, feature_sources=sources)
    assert selected == all_features


def test_select_features_dual_route_cascade_routes():
    all_features = ["lead_source", "city", "crif_score"]
    sources = {"lead_source": "core", "city": "core", "crif_score": "enrichment"}
    route_a = architectures.select_features("dual_route_cascade", all_features=all_features, feature_sources=sources, route="a")
    route_b = architectures.select_features("dual_route_cascade", all_features=all_features, feature_sources=sources, route="b")
    assert route_a == all_features
    assert route_b == ["lead_source", "city"]


def test_select_features_dual_route_cascade_requires_route():
    with pytest.raises(ValueError):
        architectures.select_features("dual_route_cascade", all_features=["a"], feature_sources={})


def test_enriched_row_mask():
    df = pd.DataFrame({"crif_score": [300, None, 500], "equifax_score": [None, None, 700]})
    mask = architectures.enriched_row_mask(df, ["crif_score", "equifax_score"])
    assert mask.tolist() == [True, False, True]


def test_enriched_row_mask_no_enrichment_columns():
    df = pd.DataFrame({"x": [1, 2, 3]})
    mask = architectures.enriched_row_mask(df, [])
    assert mask.tolist() == [False, False, False]
