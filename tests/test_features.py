from __future__ import annotations

import pandas as pd

from app.core import features


def test_classify_numeric_categorical_matches_schema_rule():
    df = pd.DataFrame(
        {
            "mostly_numeric": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "not_a_number"],
            "mostly_text": ["1", "2", "x", "y", "z", "a", "b", "c", "d", "e"],
        }
    )
    result = features.classify_numeric_categorical(df, list(df.columns))
    assert result.numeric == ["mostly_numeric"]
    assert result.categorical == ["mostly_text"]
    assert result.all_features() == ["mostly_numeric", "mostly_text"]


def test_classify_numeric_categorical_override_wins():
    df = pd.DataFrame({"looks_numeric": ["1", "2", "3"]})
    result = features.classify_numeric_categorical(df, ["looks_numeric"], overrides={"looks_numeric": "categorical"})
    assert result.categorical == ["looks_numeric"]
    assert result.numeric == []


def test_classify_feature_source_default_patterns():
    assert features.classify_feature_source("crif_credit_score") == "enrichment"
    assert features.classify_feature_source("equifax_score") == "enrichment"
    assert features.classify_feature_source("epfo_status") == "enrichment"
    assert features.classify_feature_source("payu_success_rate") == "enrichment"
    assert features.classify_feature_source("lead_source") == "core"
    assert features.classify_feature_source("city") == "core"


def test_classify_feature_source_override_wins():
    assert features.classify_feature_source("weird_column", overrides={"weird_column": "enrichment"}) == "enrichment"
    assert features.classify_feature_source("crif_score", overrides={"crif_score": "core"}) == "core"


def test_classify_feature_sources_batch():
    cols = ["crif_score", "lead_source", "epfo_flag"]
    result = features.classify_feature_sources(cols)
    assert result == {"crif_score": "enrichment", "lead_source": "core", "epfo_flag": "enrichment"}
