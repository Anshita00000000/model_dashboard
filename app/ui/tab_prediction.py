"""Tab 7 — Predict.

UI only: every decision made here is delegated to app.core.{predict,
registry, schema}. Merchant selection happens in app/main.py's sidebar;
this tab picks a trained bundle within that merchant, then a purpose="predict"
dataset to score against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.core import predict as predict_mod
from app.core import registry
from app.core import schema as schema_mod
from app.core import storage

_ISSUE_KIND_LABELS = {
    "missing_feature": "Missing feature",
    "dtype_not_coercible": "Values not coercible to numeric",
    "unseen_categories": "Categories not seen at training time",
    "null_rate_shift": "Null rate shifted vs training",
}


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataframe(path: str) -> pd.DataFrame:
    return storage.load_dataframe(path)


@st.cache_resource(show_spinner="Loading model bundle...")
def _load_bundle(path: str) -> registry.ModelBundle:
    return registry.load_bundle(path)


def _dataset_schema(dataset_row: dict) -> Optional[schema_mod.DatasetSchema]:
    try:
        return schema_mod.DatasetSchema.from_dict(json.loads(dataset_row["schema_json"]))
    except (KeyError, ValueError, TypeError):
        return None


def _bundle_metrics_summary(bundle_row: dict) -> pd.DataFrame:
    try:
        metrics = json.loads(bundle_row["metrics_json"])
    except (KeyError, ValueError, TypeError):
        return pd.DataFrame()
    evaluation = metrics.get("evaluation", {})
    rows = []
    for stage, payload in evaluation.items():
        m = payload.get("metrics", {})
        rows.append(
            {
                "stage": stage,
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "base_rate": m.get("base_rate"),
                "n_rows": m.get("n_rows"),
            }
        )
    return pd.DataFrame(rows)


def render(store: storage.MetadataStore, merchant: str, storage_root: Path | str = storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Predict")

    bundles = store.list_model_bundles(merchant=merchant)
    if not bundles:
        st.info("No trained model bundles for this merchant yet. Train one in the Train & Test tab first.")
        return

    bundle_options = {f"{b['bundle_id']}  ·  {b['architecture']}  ·  {b['created_at']}": b for b in bundles}
    bundle_row = bundle_options[st.selectbox("Model bundle", list(bundle_options.keys()))]

    training_dataset = store.get_dataset(bundle_row["dataset_id"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Architecture", bundle_row["architecture"])
    c2.metric("Trained", bundle_row["created_at"][:19])
    c3.metric("Training dataset hash", (training_dataset["content_hash"][:12] + "…") if training_dataset else "unknown")

    metrics_summary = _bundle_metrics_summary(bundle_row)
    if not metrics_summary.empty:
        st.caption("Held-out metrics at training time")
        st.dataframe(metrics_summary, width="stretch", hide_index=True)

    st.divider()

    predict_datasets = [d for d in store.list_datasets(merchant=merchant) if d["purpose"] == "predict"]
    if not predict_datasets:
        st.info("No prediction-purpose datasets registered for this merchant yet.")
        return
    dataset_options = {
        f"{d['dataset_id']}  ·  {d['source_file']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d
        for d in predict_datasets
    }
    dataset_row = dataset_options[st.selectbox("Dataset to score", list(dataset_options.keys()))]

    df = _load_dataframe(dataset_row["artifact_path"])
    bundle = _load_bundle(bundle_row["bundle_path"])
    ds_schema = _dataset_schema(dataset_row)

    st.divider()
    st.markdown("#### Pre-flight validation")
    report = predict_mod.validate_for_prediction(df, bundle)

    if not report.issues:
        st.success("No issues found — every pinned feature is present, coercible, and in-distribution.")
    else:
        if report.errors:
            st.error(f"{len(report.errors)} blocking issue(s) — scoring is disabled until these are fixed.")
        if report.warnings:
            st.warning(f"{len(report.warnings)} advisory issue(s) — review before trusting the scores.")

        issue_rows = [
            {
                "severity": "🔴 error" if i.severity == "error" else "🟡 warning",
                "column": i.column,
                "issue": _ISSUE_KIND_LABELS.get(i.kind, i.kind),
                "detail": i.message,
            }
            for i in report.issues
        ]
        st.dataframe(pd.DataFrame(issue_rows), width="stretch", hide_index=True)

    st.divider()

    default_id_col = None
    if ds_schema is not None:
        id_cols = ds_schema.columns_with_role("id")
        default_id_col = id_cols[0] if id_cols else None
    id_columns = st.multiselect(
        "Identifier column(s) to carry through to the output",
        list(df.columns),
        default=[default_id_col] if default_id_col else [],
    )

    session_key = f"prediction_result::{merchant}::{bundle_row['bundle_id']}::{dataset_row['dataset_id']}"

    can_score = report.is_scoreable and bool(id_columns)
    if not id_columns:
        st.caption("Pick at least one identifier column before scoring.")

    if st.button("Score", type="primary", disabled=not can_score):
        scored = predict_mod.score_batch(df, bundle, id_columns=id_columns, bundle_id=bundle_row["bundle_id"])
        st.session_state[session_key] = scored

    scored = st.session_state.get(session_key)
    if scored is None:
        return

    st.markdown("#### Scored output")
    st.dataframe(scored.head(200), width="stretch", hide_index=True)
    st.caption(f"{len(scored)} row(s) scored")

    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"predictions_{bundle_row['bundle_id']}.csv",
        mime="text/csv",
    )

    if st.button("Save prediction run"):
        run_id, path = predict_mod.save_prediction_run(
            store=store,
            root=storage_root,
            merchant=merchant,
            bundle_id=bundle_row["bundle_id"],
            dataset_id=dataset_row["dataset_id"],
            scored_df=scored,
        )
        st.success(f"Saved run_id = {run_id} → {path}")
