"""Run History panel — the audit trail from CLAUDE.md, made visible.

UI only: reads the append-only SQLite tables via app.core.storage.MetadataStore
and displays them. No writes happen here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core import storage

_DATASET_COLUMNS = ["dataset_id", "purpose", "source_file", "row_count", "col_count", "content_hash", "created_at"]
_SPLIT_COLUMNS = ["split_id", "dataset_id", "strategy", "train_rows", "test_rows", "created_at"]
_BUNDLE_COLUMNS = ["bundle_id", "dataset_id", "split_id", "architecture", "created_by", "created_at"]
_RUN_COLUMNS = ["run_id", "bundle_id", "dataset_id", "row_count", "output_path", "created_at"]


def _table(rows: list[dict], columns: list[str], empty_message: str) -> None:
    if not rows:
        st.caption(empty_message)
        return
    df = pd.DataFrame(rows)[columns].sort_values("created_at", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)


def render(store: storage.MetadataStore, merchant: str) -> None:
    """Every dataset, split, model bundle, and prediction run for `merchant`,
    newest first — answers "which model, trained on which data, produced
    this score, and when" straight from the append-only audit trail.
    """
    st.caption(f"Everything registered for {merchant}, most recent first.")

    st.markdown("**Datasets**")
    _table(store.list_datasets(merchant=merchant), _DATASET_COLUMNS, "No datasets registered yet.")

    st.markdown("**Splits**")
    dataset_ids = {d["dataset_id"] for d in store.list_datasets(merchant=merchant)}
    splits = [s for s in store.list_splits() if s["dataset_id"] in dataset_ids]
    _table(splits, _SPLIT_COLUMNS, "No splits saved yet.")

    st.markdown("**Model bundles**")
    _table(store.list_model_bundles(merchant=merchant), _BUNDLE_COLUMNS, "No model bundles trained yet.")

    st.markdown("**Prediction runs**")
    bundle_ids = {b["bundle_id"] for b in store.list_model_bundles(merchant=merchant)}
    runs = [r for r in store.list_prediction_runs() if r["bundle_id"] in bundle_ids]
    _table(runs, _RUN_COLUMNS, "No prediction runs yet.")
