from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from app.core import storage


def test_save_dataframe_round_trips_and_never_overwrites(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})

    path1 = storage.save_dataframe(df, "dataset", "evoke", "leads", root=tmp_path)
    path2 = storage.save_dataframe(df, "dataset", "evoke", "leads", root=tmp_path)

    assert path1 != path2
    assert path1.exists() and path2.exists()
    assert "dataset" in path1.parts and "evoke" in path1.parts

    loaded = storage.load_dataframe(path1)
    pd.testing.assert_frame_equal(loaded, df)


def test_save_json_round_trips_and_never_overwrites(tmp_path):
    obj = {"hello": "world", "n": 3}
    path1 = storage.save_json(obj, "split", "misya", "config", root=tmp_path)
    path2 = storage.save_json(obj, "split", "misya", "config", root=tmp_path)

    assert path1 != path2
    assert storage.load_json(path1) == obj
    assert storage.load_json(path2) == obj


def test_write_json_atomic_exact_path(tmp_path):
    target = tmp_path / "nested" / "manifest.json"
    storage.write_json_atomic(target, {"x": 1})
    assert target.exists()
    assert storage.load_json(target) == {"x": 1}
    # no leftover temp files
    assert list(target.parent.glob(".*")) == []


def test_metadata_store_insert_and_query(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")

    store.insert_dataset(
        dataset_id="ds1",
        merchant="evoke",
        purpose="train",
        source_file="leads.csv",
        row_count=100,
        col_count=5,
        content_hash="abc123",
        schema_json="{}",
        artifact_path="dataset/evoke/x.parquet",
    )

    fetched = store.get_dataset("ds1")
    assert fetched["merchant"] == "evoke"
    assert fetched["row_count"] == 100
    assert fetched["created_at"]  # populated server-side

    all_for_merchant = store.list_datasets(merchant="evoke")
    assert len(all_for_merchant) == 1
    assert store.list_datasets(merchant="misya") == []
    assert store.get_dataset("nonexistent") is None


def test_metadata_store_is_append_only(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_dataset(
        dataset_id="ds1",
        merchant="evoke",
        purpose="train",
        source_file="leads.csv",
        row_count=100,
        col_count=5,
        content_hash="abc123",
        schema_json="{}",
        artifact_path="dataset/evoke/x.parquet",
    )

    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE datasets SET row_count = 999 WHERE dataset_id = 'ds1'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM datasets WHERE dataset_id = 'ds1'")

    # unchanged after both blocked attempts
    assert store.get_dataset("ds1")["row_count"] == 100


def test_metadata_store_full_chain(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_dataset(
        dataset_id="ds1", merchant="evoke", purpose="train", source_file="leads.csv",
        row_count=100, col_count=5, content_hash="abc123", schema_json="{}",
        artifact_path="dataset/evoke/x.parquet",
    )
    store.insert_split(
        split_id="sp1", dataset_id="ds1", strategy="random", config_json="{}",
        train_ids_path="split/evoke/train.json", test_ids_path="split/evoke/test.json",
        train_rows=80, test_rows=20,
    )
    store.insert_model_bundle(
        bundle_id="b1", merchant="evoke", dataset_id="ds1", split_id="sp1",
        architecture="single_stage", target_config_json="{}", feature_list_json="[]",
        hyperparams_json="{}", metrics_json="{}", bundle_path="bundle/evoke/b1",
        created_by="test-user",
    )
    store.insert_prediction_run(
        run_id="r1", bundle_id="b1", dataset_id="ds1", row_count=20,
        output_path="prediction/evoke/r1.parquet",
    )

    assert store.get_split("sp1")["dataset_id"] == "ds1"
    assert store.get_model_bundle("b1")["split_id"] == "sp1"
    assert store.get_prediction_run("r1")["bundle_id"] == "b1"
    assert len(store.list_splits(dataset_id="ds1")) == 1
    assert len(store.list_model_bundles(merchant="evoke")) == 1
    assert len(store.list_prediction_runs(bundle_id="b1")) == 1
