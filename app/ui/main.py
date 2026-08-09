"""Streamlit entrypoint.

Launch with `streamlit run app/ui/main.py` (or `make run`). All logic lives
in app/core/ — this file, and the tab_*.py / run_history.py modules it wires
together, only turn widgets into calls into that logic. See CLAUDE.md,
"All logic in app/core/ — importable, testable, no Streamlit import."

Tabs 1-3 (raw file upload, raw EDA, canonical mapping + enrichment) are Phase
2's ingestion layer — enrichment API calls are stubbed (mock data), see
app/core/enrichment/. Tab 4 (feature extraction) is Phase 2 — not built yet.
Tabs 5-7 (data prep, train/test, predict) are Phase 1, driven by an
already-enriched CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable regardless of the caller's cwd/PYTHONPATH —
# `streamlit run` does not reliably put it there on its own (observed to work
# in some environments and fail in others, e.g. a fresh Colab shell), and this
# file can be launched directly (`streamlit run app/ui/main.py`, `make run`),
# not just via the app/main.py shim.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402 (must follow the sys.path fix above)

from app.core.storage import DEFAULT_STORAGE_ROOT, MetadataStore  # noqa: E402
from app.ui import run_history, tab_data_prep, tab_enrichment, tab_prediction, tab_raw_eda, tab_training, tab_upload  # noqa: E402

# Seed list so the sidebar isn't empty before any data exists; the merchant
# picker is otherwise fully dynamic (store.list_merchants()) — Tab 1 accepts
# any new merchant name, and it appears here as soon as something is uploaded
# or trained for it.
SEED_MERCHANTS = ["Evoke", "Misya", "Nivaan"]
PURPOSES = ["train", "predict"]

st.set_page_config(page_title="CarePay Lead Scoring", layout="wide")


@st.cache_resource
def get_metadata_store() -> MetadataStore:
    return MetadataStore(Path(DEFAULT_STORAGE_ROOT) / "meta.db")


def _render_phase2_placeholder(title: str, description: str) -> None:
    st.subheader(title)
    st.info(f"**Phase 2 — not built yet.** {description} See CLAUDE.md for the build plan.")


_FEATURE_EXTRACTION_TAB = (
    "4 · Feature Extraction",
    "Build model-ready features from canonical + enriched fields.",
)


def main() -> None:
    st.title("CarePay Lead Scoring Dashboard")

    store = get_metadata_store()

    with st.sidebar:
        # Tab 1 (Upload) can only stage a merchant switch via this key BEFORE the
        # selectbox below is instantiated — writing to st.session_state["merchant"]
        # after that point raises (Streamlit forbids mutating an active widget's key).
        if "_pending_merchant_switch" in st.session_state:
            st.session_state["merchant"] = st.session_state.pop("_pending_merchant_switch")

        merchant_options = sorted(set(SEED_MERCHANTS) | set(store.list_merchants()))
        if st.session_state.get("merchant") not in merchant_options:
            st.session_state["merchant"] = merchant_options[0]
        # Value comes from st.session_state alone (pre-seeded above) — passing index=
        # as well would make Streamlit warn about two competing sources of truth.
        merchant = st.selectbox("Merchant", merchant_options, key="merchant")
        purpose = st.radio(
            "Purpose", PURPOSES, key="purpose", horizontal=True,
            help="'predict' disables Data Prep and Train & Test — nothing to prepare or train on a prediction batch.",
        )
        st.caption("Phase 1 foundation — data prep, training/testing, and prediction, driven by an already-enriched CSV.")

    predict_mode = purpose == "predict"

    tab_labels = [
        "1 · Upload", "2 · EDA", "3 · Enrichment", _FEATURE_EXTRACTION_TAB[0],
        "5 · Data Prep" + (" 🔒" if predict_mode else ""),
        "6 · Train & Test" + (" 🔒" if predict_mode else ""),
        "7 · Predict",
    ]
    tabs = st.tabs(tab_labels)
    tab_upload_, tab_eda_, tab_enrich, tab_feat, tab_prep, tab_train, tab_predict = tabs

    with tab_upload_:
        tab_upload.render(store, storage_root=DEFAULT_STORAGE_ROOT)

    with tab_eda_:
        tab_raw_eda.render(store, merchant)

    with tab_enrich:
        tab_enrichment.render(store, merchant, storage_root=DEFAULT_STORAGE_ROOT)

    with tab_feat:
        _render_phase2_placeholder(*_FEATURE_EXTRACTION_TAB)

    with tab_prep:
        if predict_mode:
            st.info("Data Prep is disabled while Purpose = 'predict'. Switch to 'train' in the sidebar to use it.")
        else:
            tab_data_prep.render(store, merchant, storage_root=DEFAULT_STORAGE_ROOT)

    with tab_train:
        if predict_mode:
            st.info("Train & Test is disabled while Purpose = 'predict'. Switch to 'train' in the sidebar to use it.")
        else:
            tab_training.render(store, merchant, storage_root=DEFAULT_STORAGE_ROOT)

    with tab_predict:
        tab_prediction.render(store, merchant, storage_root=DEFAULT_STORAGE_ROOT)

    st.divider()
    with st.expander("📜 Run History", expanded=False):
        run_history.render(store, merchant)


if __name__ == "__main__":
    main()
