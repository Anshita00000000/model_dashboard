from __future__ import annotations

import pandas as pd
import pytest

from app.core import schema as schema_mod


def test_infer_schema_numeric_vs_categorical_threshold():
    # 9/10 numeric-parseable -> numeric (>= 90%)
    df = pd.DataFrame(
        {
            "mostly_numeric": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "not_a_number"],
            "mostly_text": ["1", "2", "x", "y", "z", "a", "b", "c", "d", "e"],
        }
    )
    result = schema_mod.infer_schema(
        df, dataset_id="ds1", merchant_name="evoke", purpose="train", source_file="leads.csv"
    )
    assert result.column("mostly_numeric").dtype == "numeric"
    assert result.column("mostly_text").dtype == "categorical"


def test_infer_schema_blank_and_null_are_excluded_from_ratio():
    # blanks/nulls shouldn't count against the numeric ratio, only parse failures should
    df = pd.DataFrame({"col": ["1", "2", "3", None, "", "  "]})
    result = schema_mod.infer_schema(
        df, dataset_id="ds1", merchant_name="evoke", purpose="train", source_file="leads.csv"
    )
    assert result.column("col").dtype == "numeric"
    assert result.column("col").nullable is True


def test_infer_schema_overrides_win():
    df = pd.DataFrame({"lead_id": [1, 2, 3], "converted": [0, 1, 0]})
    result = schema_mod.infer_schema(
        df,
        dataset_id="ds1",
        merchant_name="misya",
        purpose="train",
        source_file="leads.csv",
        overrides={"lead_id": {"role": "id"}, "converted": {"role": "target"}},
    )
    assert result.column("lead_id").role == "id"
    assert result.column("converted").role == "target"
    # anything not overridden defaults to "feature"
    assert result.row_count == 3
    assert result.column_count == 2


def test_hash_dataframe_is_deterministic_and_sensitive_to_content():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})
    df3 = pd.DataFrame({"a": [1, 2, 4]})
    assert schema_mod.hash_dataframe(df1) == schema_mod.hash_dataframe(df2)
    assert schema_mod.hash_dataframe(df1) != schema_mod.hash_dataframe(df3)


def test_validate_detects_drift():
    train_df = pd.DataFrame({"lead_id": [1, 2, 3], "score_feat": [1.0, 2.0, 3.0], "converted": [0, 1, 0]})
    train_schema = schema_mod.infer_schema(
        train_df,
        dataset_id="ds-train",
        merchant_name="nivaan",
        purpose="train",
        source_file="leads.csv",
        overrides={"lead_id": {"role": "id"}, "converted": {"role": "target"}},
    )

    # predict-time frame: score_feat became categorical garbage, converted (target) is
    # absent as expected at predict time, and there's a brand-new unexpected column
    predict_df = pd.DataFrame({"lead_id": [4, 5], "score_feat": ["x", "y"], "unexpected": [1, 2]})
    violations = schema_mod.validate(predict_df, train_schema)
    kinds = {(v.column, v.kind) for v in violations}
    assert ("converted", "missing_column") in kinds
    assert ("score_feat", "dtype_mismatch") in kinds
    assert ("unexpected", "unexpected_column") in kinds


def test_validate_clean_dataframe_has_no_violations():
    df = pd.DataFrame({"lead_id": [1, 2, 3], "score_feat": [1.0, 2.0, 3.0]})
    s = schema_mod.infer_schema(
        df,
        dataset_id="ds1",
        merchant_name="nivaan",
        purpose="predict",
        source_file="leads.csv",
        overrides={"lead_id": {"role": "id"}},
    )
    assert schema_mod.validate(df, s) == []


def test_is_blank():
    assert schema_mod.is_blank(None) is True
    assert schema_mod.is_blank(float("nan")) is True
    assert schema_mod.is_blank("") is True
    assert schema_mod.is_blank("   ") is True
    assert schema_mod.is_blank(0) is False
    assert schema_mod.is_blank("x") is False


def test_blank_rate():
    series = pd.Series([1, None, "", "x", float("nan")])
    assert schema_mod.blank_rate(series) == pytest.approx(3 / 5)


def test_blank_rate_empty_series_is_nan():
    import math

    assert math.isnan(schema_mod.blank_rate(pd.Series([], dtype=object)))
