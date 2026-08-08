"""Streamlit entrypoint — UI layer only.

Every piece of actual logic lives in app/core/ (schema, storage, registry) and
is imported here, never re-implemented. This file is free to be thrown away
and replaced by a FastAPI service without touching app/core/.

Phase 1 (this build): foundation for tabs 5 (data prep), 6 (train/test), and
7 (predict). Phase 2 (tabs 1-4: upload, EDA, field mapping, enrichment) is not
built yet — see CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.core.storage import DEFAULT_STORAGE_ROOT, MetadataStore
from app.ui import tab_data_prep

MERCHANTS = ["Evoke", "Misya", "Nivaan"]

st.set_page_config(page_title="CarePay Lead Scoring", layout="wide")


@st.cache_resource
def get_metadata_store() -> MetadataStore:
    return MetadataStore(Path(DEFAULT_STORAGE_ROOT) / "meta.db")


def main() -> None:
    st.title("CarePay Lead Scoring Dashboard")
    st.caption("Phase 1 foundation — data prep, training/testing, and prediction.")

    with st.sidebar:
        merchant = st.selectbox("Merchant", MERCHANTS)

    store = get_metadata_store()

    tab_prep, tab_train, tab_predict = st.tabs(
        ["5 · Data Prep", "6 · Train & Test", "7 · Predict"]
    )

    with tab_prep:
        tab_data_prep.render(store, merchant, storage_root=DEFAULT_STORAGE_ROOT)

    with tab_train:
        st.subheader(f"Model bundles — {merchant}")
        bundles = store.list_model_bundles(merchant=merchant)
        if bundles:
            st.dataframe(bundles, width="stretch")
        else:
            st.info("No model bundles trained yet for this merchant.")

    with tab_predict:
        st.subheader(f"Prediction runs — {merchant}")
        bundle_ids = [b["bundle_id"] for b in store.list_model_bundles(merchant=merchant)]
        if not bundle_ids:
            st.info("Train a model bundle before running predictions.")
        else:
            selected_bundle = st.selectbox("Model bundle", bundle_ids)
            runs = store.list_prediction_runs(bundle_id=selected_bundle)
            if runs:
                st.dataframe(runs, width="stretch")
            else:
                st.info("No prediction runs yet for this bundle.")


if __name__ == "__main__":
    main()
