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


# ---------------------------------------------------------------------------
# raw_datasets
# ---------------------------------------------------------------------------


def test_timestamped_raw_file_path_preserves_original_filename_and_never_collides(tmp_path):
    p1 = storage.timestamped_raw_file_path(tmp_path, "Evoke", "leads export (final).csv")
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(b"x")
    p2 = storage.timestamped_raw_file_path(tmp_path, "Evoke", "leads export (final).csv")

    assert p1 != p2
    assert p1.name.endswith("_leads_export__final_.csv")
    assert p1.parts[-4:-1] == ("datasets", "raw", "Evoke")


def test_insert_and_get_raw_dataset(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_raw_dataset(
        raw_dataset_id="raw1", merchant="Evoke", purpose="train", original_filename="leads.csv",
        stored_path="datasets/raw/Evoke/x.csv", parquet_path="raw/Evoke/x.parquet",
        delimiter=",", encoding="utf-8", sheet_name=None, row_count=100, col_count=5,
        content_hash="abc123", uploaded_by="tester", notes="first batch",
    )
    row = store.get_raw_dataset("raw1")
    assert row["merchant"] == "Evoke"
    assert row["delimiter"] == ","
    assert row["notes"] == "first batch"
    assert row["created_at"]
    assert store.get_raw_dataset("nonexistent") is None


def test_list_raw_datasets_filters_by_merchant(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    for i, merchant in enumerate(["Evoke", "Misya", "Evoke"]):
        store.insert_raw_dataset(
            raw_dataset_id=f"raw{i}", merchant=merchant, purpose="train", original_filename="x.csv",
            stored_path="p", parquet_path="q", delimiter=",", encoding="utf-8", sheet_name=None,
            row_count=1, col_count=1, content_hash=f"hash{i}", uploaded_by="tester",
        )
    assert len(store.list_raw_datasets(merchant="Evoke")) == 2
    assert len(store.list_raw_datasets(merchant="Misya")) == 1
    assert len(store.list_raw_datasets()) == 3


def test_find_raw_datasets_by_content_hash(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_raw_dataset(
        raw_dataset_id="raw1", merchant="Evoke", purpose="train", original_filename="x.csv",
        stored_path="p", parquet_path="q", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=1, col_count=1, content_hash="dup-hash", uploaded_by="tester",
    )
    # append-only: re-uploading identical bytes still inserts a new row
    store.insert_raw_dataset(
        raw_dataset_id="raw2", merchant="Evoke", purpose="train", original_filename="x.csv",
        stored_path="p2", parquet_path="q2", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=1, col_count=1, content_hash="dup-hash", uploaded_by="tester",
    )
    matches = store.find_raw_datasets_by_content_hash("dup-hash")
    assert len(matches) == 2
    assert store.find_raw_datasets_by_content_hash("no-such-hash") == []


def test_raw_datasets_is_append_only(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_raw_dataset(
        raw_dataset_id="raw1", merchant="Evoke", purpose="train", original_filename="x.csv",
        stored_path="p", parquet_path="q", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=1, col_count=1, content_hash="h", uploaded_by="tester",
    )
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE raw_datasets SET row_count = 999 WHERE raw_dataset_id = 'raw1'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM raw_datasets WHERE raw_dataset_id = 'raw1'")


def test_list_merchants_aggregates_across_tables(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    store.insert_raw_dataset(
        raw_dataset_id="raw1", merchant="Nivaan", purpose="train", original_filename="x.csv",
        stored_path="p", parquet_path="q", delimiter=",", encoding="utf-8", sheet_name=None,
        row_count=1, col_count=1, content_hash="h", uploaded_by="tester",
    )
    store.insert_dataset(
        dataset_id="ds1", merchant="Evoke", purpose="train", source_file="leads.csv",
        row_count=100, col_count=5, content_hash="abc123", schema_json="{}",
        artifact_path="dataset/evoke/x.parquet",
    )
    assert store.list_merchants() == ["Evoke", "Nivaan"]
