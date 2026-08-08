"""Streamlit entrypoint.

Launch with `streamlit run app/ui/main.py` (or `make run`). All logic lives
in app/core/ — this file, and the tab_*.py / run_history.py modules it wires
together, only turn widgets into calls into that logic. See CLAUDE.md,
"All logic in app/core/ — importable, testable, no Streamlit import."

Tabs 1-4 (clinic file upload, EDA, field mapping to the canonical CRM schema,
enrichment orchestration) are Phase 2 — not built yet. Tabs 5-7 (data prep,
train/test, predict) are Phase 1, driven by an already-enriched CSV.
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
from app.ui import run_history, tab_data_prep, tab_prediction, tab_training  # noqa: E402

MERCHANTS = ["Evoke", "Misya", "Nivaan"]
PURPOSES = ["train", "predict"]

_PHASE2_TABS = [
    ("1 · Upload", "Clinic lead file upload."),
    ("2 · EDA", "Exploratory data analysis on the uploaded file."),
    ("3 · Field Mapping", "Map clinic-specific fields to the canonical CRM schema."),
    ("4 · Enrichment", "Orchestrate CRIF / Equifax / EPFO / salary estimator / PayU / LeadCreditEngineOutput."),
]

st.set_page_config(page_title="CarePay Lead Scoring", layout="wide")


@st.cache_resource
def get_metadata_store() -> MetadataStore:
    return MetadataStore(Path(DEFAULT_STORAGE_ROOT) / "meta.db")


def _render_phase2_placeholder(title: str, description: str) -> None:
    st.subheader(title)
    st.info(f"**Phase 2 — not built yet.** {description} See CLAUDE.md for the build plan.")


def main() -> None:
    st.title("CarePay Lead Scoring Dashboard")

    with st.sidebar:
        merchant = st.selectbox("Merchant", MERCHANTS, key="merchant")
        purpose = st.radio(
            "Purpose", PURPOSES, key="purpose", horizontal=True,
            help="'predict' disables Data Prep and Train & Test — nothing to prepare or train on a prediction batch.",
        )
        st.caption("Phase 1 foundation — data prep, training/testing, and prediction, driven by an already-enriched CSV.")

    predict_mode = purpose == "predict"
    store = get_metadata_store()

    tab_labels = [label for label, _ in _PHASE2_TABS] + [
        "5 · Data Prep" + (" 🔒" if predict_mode else ""),
        "6 · Train & Test" + (" 🔒" if predict_mode else ""),
        "7 · Predict",
    ]
    tabs = st.tabs(tab_labels)
    phase2_tabs, tab_prep, tab_train, tab_predict = tabs[:4], tabs[4], tabs[5], tabs[6]

    for tab, (label, description) in zip(phase2_tabs, _PHASE2_TABS):
        with tab:
            _render_phase2_placeholder(label, description)

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
