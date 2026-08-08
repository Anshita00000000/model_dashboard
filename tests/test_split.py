from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import split as split_mod
from app.core.storage import MetadataStore


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sources = rng.choice(["FB Ads", "Google", "Organic"], size=n, p=[0.5, 0.3, 0.2])
    outcome = (rng.random(n) < 0.1).astype(int)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "lead_id": np.arange(1, n + 1),
            "source": sources,
            "converted": outcome,
            "created_at": dates,
            "phone": rng.integers(1000, 1010, size=n),  # repeat leads: only 10 distinct phones
        }
    )


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------


def test_random_split_disjoint_and_covers_all_rows():
    df = _make_df()
    result = split_mod.split_dataset(df, strategy="random", row_id_column="lead_id", test_size=0.25)

    assert result.train_rows + result.test_rows == len(df)
    assert set(result.train_ids).isdisjoint(set(result.test_ids))
    assert set(result.train_ids) | set(result.test_ids) == set(df["lead_id"].tolist())
    assert 0.20 < result.test_rows / len(df) < 0.30
    assert result.warnings == []


def test_random_split_uses_index_when_no_row_id_column():
    df = _make_df().drop(columns=["lead_id"])
    result = split_mod.split_dataset(df, strategy="random", test_size=0.2)
    assert set(result.train_ids) | set(result.test_ids) == set(df.index.tolist())


# ---------------------------------------------------------------------------
# stratified
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_joint_composition():
    df = _make_df(n=2000)
    result = split_mod.split_dataset(
        df,
        strategy="stratified",
        row_id_column="lead_id",
        test_size=0.2,
        stratify_columns=["source", "converted"],
    )

    comp = result.composition
    for col in ("source", "converted"):
        sub = comp[comp["column"] == col]
        for _, row in sub.iterrows():
            # with 2000 rows and well-populated strata, train/test % should be close
            assert abs(row["train_pct"] - row["test_pct"]) < 5.0


def test_stratified_split_collapses_rare_strata_without_crashing():
    # a handful of joint (source, converted) combinations with a single member
    df = pd.DataFrame(
        {
            "lead_id": range(1, 21),
            "source": ["A"] * 9 + ["B"] * 9 + ["RareSource1", "RareSource2"],
            "converted": [0] * 9 + [1] * 9 + [1, 0],
        }
    )
    result = split_mod.split_dataset(
        df,
        strategy="stratified",
        row_id_column="lead_id",
        test_size=0.3,
        stratify_columns=["source", "converted"],
    )

    assert result.train_rows + result.test_rows == len(df)
    assert set(result.train_ids) | set(result.test_ids) == set(df["lead_id"].tolist())
    assert any("Collapsed" in w for w in result.warnings)
    assert result.config["n_collapsed_strata"] == 2
    assert result.config["n_collapsed_rows"] == 2


def test_stratified_split_requires_columns():
    df = _make_df()
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="stratified", stratify_columns=[])
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="stratified", stratify_columns=["does_not_exist"])


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------


def test_temporal_split_cutoff_mode():
    df = _make_df(n=100)
    cutoff = "2024-03-01"
    result = split_mod.split_dataset(
        df, strategy="temporal", row_id_column="lead_id", date_column="created_at", cutoff=cutoff
    )

    train_df, test_df = split_mod.apply_split(df, "lead_id", result.train_ids, result.test_ids)
    assert (train_df["created_at"] < pd.Timestamp(cutoff)).all()
    assert (test_df["created_at"] >= pd.Timestamp(cutoff)).all()


def test_temporal_split_periods_mode():
    df = _make_df(n=100)
    df["month"] = df["created_at"].dt.strftime("%b")  # Jan, Feb, Mar, Apr
    result = split_mod.split_dataset(
        df,
        strategy="temporal",
        row_id_column="lead_id",
        date_column="month",
        temporal_mode="periods",
        test_period_values=["Apr"],
    )
    train_df, test_df = split_mod.apply_split(df, "lead_id", result.train_ids, result.test_ids)
    assert (test_df["month"] == "Apr").all()
    assert (train_df["month"] != "Apr").all()


def test_temporal_split_warns_on_base_rate_shift():
    n = 400
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # outcome immaturity: last 30 days have ~0 conversions (too recent to have converted)
    converted = np.where(np.arange(n) < n - 30, (np.arange(n) % 5 == 0).astype(int), 0)
    df = pd.DataFrame({"lead_id": range(n), "created_at": dates, "converted": converted})

    result = split_mod.split_dataset(
        df,
        strategy="temporal",
        row_id_column="lead_id",
        date_column="created_at",
        cutoff=str(dates[-30].date()),
        target_column="converted",
    )
    assert any("base rate" in w for w in result.warnings)


def test_temporal_split_no_warning_when_rates_are_similar():
    n = 400
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    converted = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({"lead_id": range(n), "created_at": dates, "converted": converted})

    result = split_mod.split_dataset(
        df,
        strategy="temporal",
        row_id_column="lead_id",
        date_column="created_at",
        cutoff=str(dates[300].date()),
        target_column="converted",
    )
    assert not any("base rate" in w for w in result.warnings)


def test_temporal_split_requires_date_column():
    df = _make_df()
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="temporal")


# ---------------------------------------------------------------------------
# grouped
# ---------------------------------------------------------------------------


def test_grouped_split_no_group_in_both_sides():
    df = _make_df(n=500)
    result = split_mod.split_dataset(
        df, strategy="grouped", row_id_column="lead_id", group_column="phone", test_size=0.3
    )

    train_df, test_df = split_mod.apply_split(df, "lead_id", result.train_ids, result.test_ids)
    train_phones = set(train_df["phone"].tolist())
    test_phones = set(test_df["phone"].tolist())
    assert train_phones.isdisjoint(test_phones)
    assert result.train_rows + result.test_rows == len(df)


def test_grouped_split_requires_group_column():
    df = _make_df()
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="grouped")


# ---------------------------------------------------------------------------
# row-id validation
# ---------------------------------------------------------------------------


def test_row_id_column_with_duplicates_raises():
    df = pd.DataFrame({"id": [1, 1, 2], "x": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="random", row_id_column="id")


def test_row_id_column_with_nulls_raises():
    df = pd.DataFrame({"id": [1, None, 3], "x": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_mod.split_dataset(df, strategy="random", row_id_column="id")


# ---------------------------------------------------------------------------
# persistence: explicit ID lists, not seeds
# ---------------------------------------------------------------------------


def test_save_and_reload_split_reconstructs_same_rows(tmp_path):
    df = _make_df(n=200)
    result = split_mod.split_dataset(df, strategy="random", row_id_column="lead_id", test_size=0.2)

    store = MetadataStore(tmp_path / "meta.db")
    store.insert_dataset(
        dataset_id="ds1", merchant="evoke", purpose="train", source_file="leads.csv",
        row_count=len(df), col_count=len(df.columns), content_hash="hash", schema_json="{}",
        artifact_path="dataset/evoke/x.parquet",
    )
    split_id = split_mod.new_split_id()
    train_path, test_path = split_mod.save_split(
        store=store, root=tmp_path, merchant="evoke", dataset_id="ds1", split_id=split_id, result=result
    )
    assert train_path.exists() and test_path.exists()

    split_row = store.get_split(split_id)
    assert split_row["train_rows"] == result.train_rows
    assert split_row["test_rows"] == result.test_rows

    train_ids, test_ids = split_mod.load_split_ids(split_row)
    assert train_ids == result.train_ids
    assert test_ids == result.test_ids

    train_df, test_df = split_mod.apply_split(df, "lead_id", train_ids, test_ids)
    assert len(train_df) == result.train_rows
    assert len(test_df) == result.test_rows
    assert set(train_df["lead_id"]).isdisjoint(set(test_df["lead_id"]))


def test_apply_split_raises_on_missing_ids():
    df = _make_df(n=10)
    with pytest.raises(ValueError):
        split_mod.apply_split(df, "lead_id", [9999], [])
