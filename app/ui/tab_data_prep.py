"""Tab 5 — Data Prep (train/test split).

UI only: every decision made here is delegated to app.core.split. This tab
is skipped for purpose="predict" datasets — a prediction batch has no
train/test split to make.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.core import schema as schema_mod
from app.core import split as split_mod
from app.core import storage

ROW_INDEX_LABEL = "(row index)"
NONE_LABEL = "(none)"

_STRATEGY_LABELS = {
    "random": "Random",
    "stratified": "Stratified (joint, multi-column)",
    "temporal": "Temporal (by date/period)",
    "grouped": "Grouped (no group in both sides)",
}
_TEMPORAL_MODE_LABELS = {
    "cutoff": "Cutoff date",
    "periods": "Explicit period values",
}


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataframe(path: str):
    return storage.load_dataframe(path)


def _dataset_schema(dataset_row: dict) -> schema_mod.DatasetSchema | None:
    try:
        return schema_mod.DatasetSchema.from_dict(json.loads(dataset_row["schema_json"]))
    except (KeyError, ValueError, TypeError):
        return None


def _suggest_role_column(ds_schema: schema_mod.DatasetSchema | None, role: str) -> str | None:
    if ds_schema is None:
        return None
    candidates = ds_schema.columns_with_role(role)
    return candidates[0] if candidates else None


def render(store: storage.MetadataStore, merchant: str, storage_root: Path | str = storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Data Prep — Train / Test Split")

    all_datasets = store.list_datasets(merchant=merchant)
    train_datasets = [d for d in all_datasets if d["purpose"] == "train"]

    if not train_datasets:
        st.info(
            "No training-purpose datasets registered for this merchant yet. This tab only "
            "applies to purpose='train' datasets — prediction batches don't need a split."
        )
        return

    options = {
        f"{d['dataset_id']}  ·  {d['source_file']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d
        for d in train_datasets
    }
    selected_label = st.selectbox("Dataset", list(options.keys()), key="data_prep_dataset_select")
    dataset_row = options[selected_label]

    # Belt-and-suspenders: this tab is a no-op for anything but purpose="train",
    # even though the selector above already filters to train-purpose datasets.
    if dataset_row["purpose"] == "predict":
        st.info("Data prep is skipped for prediction-purpose datasets.")
        return

    df = _load_dataframe(dataset_row["artifact_path"])
    ds_schema = _dataset_schema(dataset_row)
    columns = list(df.columns)

    suggested_id_col = _suggest_role_column(ds_schema, "id")
    suggested_target_col = _suggest_role_column(ds_schema, "target")

    st.divider()
    col_strategy, col_row_id = st.columns(2)
    with col_strategy:
        strategy = st.selectbox(
            "Split strategy",
            split_mod.SPLIT_STRATEGIES,
            format_func=lambda s: _STRATEGY_LABELS[s],
        )
    with col_row_id:
        row_id_options = [ROW_INDEX_LABEL] + columns
        default_idx = row_id_options.index(suggested_id_col) if suggested_id_col in row_id_options else 0
        row_id_choice = st.selectbox(
            "Row ID column", row_id_options, index=default_idx,
            help="Used to persist train/test membership as explicit IDs, not a random seed.",
        )
    row_id_column = None if row_id_choice == ROW_INDEX_LABEL else row_id_choice

    split_kwargs: dict = {"strategy": strategy, "row_id_column": row_id_column}
    stratify_columns: list[str] = []

    if strategy == "random":
        split_kwargs["test_size"] = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)

    elif strategy == "stratified":
        split_kwargs["test_size"] = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
        default_strat = [c for c in [suggested_target_col] if c]
        stratify_columns = st.multiselect(
            "Stratify jointly on",
            columns,
            default=default_strat,
            help="All selected columns are combined into a single joint stratification key.",
        )
        split_kwargs["stratify_columns"] = stratify_columns

    elif strategy == "temporal":
        date_column = st.selectbox("Date / period column", columns)
        split_kwargs["date_column"] = date_column
        mode = st.radio(
            "Mode", split_mod.TEMPORAL_MODES, format_func=lambda m: _TEMPORAL_MODE_LABELS[m], horizontal=True
        )
        split_kwargs["temporal_mode"] = mode
        if mode == "cutoff":
            cutoff = st.text_input(
                "Cutoff (rows on/after this go to test)", placeholder="YYYY-MM-DD",
            )
            split_kwargs["cutoff"] = cutoff or None
        else:
            period_values = sorted(df[date_column].dropna().astype(str).unique().tolist())
            split_kwargs["test_period_values"] = st.multiselect(
                "Period value(s) that go to test (everything else goes to train)", period_values,
            )
        target_options = [NONE_LABEL] + columns
        default_target_idx = target_options.index(suggested_target_col) if suggested_target_col in target_options else 0
        target_choice = st.selectbox(
            "Target column (for outcome-immaturity / base-rate check)", target_options, index=default_target_idx,
        )
        split_kwargs["target_column"] = None if target_choice == NONE_LABEL else target_choice

    else:  # grouped
        split_kwargs["test_size"] = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
        split_kwargs["group_column"] = st.selectbox(
            "Group column (e.g. phone / customer id)", columns,
            help="No group's rows will appear on both sides of the split.",
        )

    default_composition = stratify_columns or ([suggested_target_col] if suggested_target_col else [])
    composition_columns = st.multiselect(
        "Columns to show in the composition comparison report", columns, default=default_composition
    )
    split_kwargs["composition_columns"] = composition_columns

    st.divider()
    session_key = f"data_prep_result::{merchant}::{dataset_row['dataset_id']}"

    if st.button("Run split", type="primary"):
        try:
            result = split_mod.split_dataset(df, **split_kwargs)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[session_key] = result

    result = st.session_state.get(session_key)
    if result is not None:
        for w in result.warnings:
            st.warning(w)

        c1, c2 = st.columns(2)
        c1.metric("Train rows", result.train_rows)
        c2.metric("Test rows", result.test_rows)

        if not result.composition.empty:
            st.caption("Composition — train vs test, by column and value")
            st.dataframe(result.composition, width="stretch", hide_index=True)

        if st.button("Save split"):
            split_id = split_mod.new_split_id()
            split_mod.save_split(
                store=store,
                root=storage_root,
                merchant=merchant,
                dataset_id=dataset_row["dataset_id"],
                split_id=split_id,
                result=result,
            )
            st.success(f"Saved split_id = {split_id}")
            del st.session_state[session_key]
