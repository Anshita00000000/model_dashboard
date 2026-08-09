"""Tab 1 — Upload.

UI only: every decision made here is delegated to app.core.{ingest, raw_eda}.
No canonical schema, no renaming, no cleaning happens here — see
app/core/ingest.py's module docstring for the scope boundary. This tab's job
is only to get the file stored byte-for-byte and give the uploader immediate
feedback on how it was read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.core import ingest
from app.core import raw_eda
from app.core import storage

UPLOADED_BY = "streamlit-ui"
NEW_MERCHANT_LABEL = "+ New merchant..."
ACCEPTED_TYPES = ["csv", "tsv", "xlsx", "xls"]


def _merchant_picker(store: storage.MetadataStore) -> str:
    existing = store.list_merchants()
    default_merchant = st.session_state.get("merchant")

    if not existing:
        return st.text_input("Merchant", value=default_merchant or "", placeholder="e.g. Evoke", key="upload_merchant_new_only").strip()

    options = existing + [NEW_MERCHANT_LABEL]
    default_idx = options.index(default_merchant) if default_merchant in existing else 0
    choice = st.selectbox("Merchant", options, index=default_idx, key="upload_merchant_select")
    if choice == NEW_MERCHANT_LABEL:
        return st.text_input("New merchant name", key="upload_merchant_new").strip()
    return choice


def render(store: storage.MetadataStore, storage_root: Path | str = storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Upload")
    st.caption(
        "Stored exactly as received — no canonical schema, no renaming, no cleaning. "
        "Cleaning happens offline by the team; field mapping happens later, at Tab 3 (Phase 2)."
    )

    # Saving may trigger st.rerun() (to switch the sidebar's merchant) — a
    # confirmation shown right before that rerun would be discarded before the
    # user ever sees it, so it's staged here and shown on the run that follows.
    just_saved = st.session_state.pop("_upload_just_saved", None)
    if just_saved:
        st.success(f"Saved raw_dataset_id = {just_saved}")

    col_merchant, col_purpose = st.columns(2)
    with col_merchant:
        merchant = _merchant_picker(store)
    with col_purpose:
        purpose = st.radio("Purpose", ["train", "predict"], key="upload_purpose", horizontal=True)

    uploaded = st.file_uploader("Lead file", type=ACCEPTED_TYPES)
    notes = st.text_area("Notes (optional)", placeholder="e.g. source system, date range, known issues", key="upload_notes")

    if uploaded is None:
        return
    if not merchant:
        st.warning("Enter a merchant name to continue.")
        return

    raw_bytes = uploaded.getvalue()
    filename = uploaded.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    sheet_name = None
    if ext in ("xlsx", "xls"):
        try:
            sheets = ingest.list_excel_sheets(raw_bytes)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the app on a bad workbook
            st.error(f"Could not read this workbook: {exc}")
            return
        sheet_name = st.selectbox("Sheet", sheets, index=0, key="upload_sheet_select") if len(sheets) > 1 else sheets[0]

    try:
        result = ingest.load_raw_file(raw_bytes, filename, sheet_name=sheet_name)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.divider()
    st.success(f"Read {filename} — {result.row_count:,} rows × {result.col_count} columns")

    c1, c2, c3 = st.columns(3)
    c1.metric("Delimiter", f"{result.delimiter_label} ({result.delimiter!r})" if result.delimiter else "n/a (Excel)")
    c2.metric("Encoding", result.encoding)
    c3.metric("Malformed rows", result.malformed_row_count)
    if result.delimiter_consistency is not None:
        st.caption(
            f"Delimiter detection confidence: {result.delimiter_consistency:.0%} of sampled rows agreed on "
            f"{result.col_count} fields (tried comma, pipe, tab, semicolon)."
        )
    if result.available_sheets:
        st.caption(f"Sheets in this workbook: {', '.join(result.available_sheets)} (using {result.sheet_name!r})")
    if result.malformed_row_examples:
        with st.expander(f"{result.malformed_row_count} malformed row(s) — never silently dropped, shown here"):
            for example in result.malformed_row_examples:
                st.code(example, language=None)

    st.markdown("#### Raw preview (first 20 rows, exactly as read)")
    st.dataframe(result.df.head(20), width="stretch")

    st.markdown("#### Column summary — INFERRED FOR PROFILING ONLY, not applied to the stored data")
    profiles = raw_eda.profile_columns(result.df)
    st.dataframe(
        pd.DataFrame(
            [{"column": p.column, "inferred_dtype": p.inferred_dtype, "null_%": round(p.null_pct, 1), "distinct": p.distinct_count} for p in profiles]
        ),
        width="stretch",
        hide_index=True,
    )

    content_hash = ingest.content_hash_of(raw_bytes)
    duplicates = store.find_raw_datasets_by_content_hash(content_hash)
    if duplicates:
        st.warning(
            f"This exact file (identical bytes) was already uploaded {len(duplicates)} time(s) before — most "
            f"recently as raw_dataset_id={duplicates[0]['raw_dataset_id']} on {duplicates[0]['created_at']}. "
            f"Saving again will still record a new, separate entry."
        )

    if st.button("Save raw dataset", type="primary"):
        save_result = ingest.save_raw_dataset(
            store=store, root=storage_root, merchant=merchant, purpose=purpose, result=result,
            uploaded_by=UPLOADED_BY, notes=notes,
        )
        st.session_state["raw_dataset_id"] = save_result.raw_dataset_id
        st.session_state["_upload_just_saved"] = save_result.raw_dataset_id
        if st.session_state.get("merchant") != merchant:
            # The sidebar's merchant selectbox owns st.session_state["merchant"] directly (via its
            # own `key=`) and Streamlit forbids writing to a widget's key after it's instantiated
            # this run — the sidebar renders before this tab. Stage the switch; main.py consumes
            # "_pending_merchant_switch" before the selectbox is created on the next run.
            st.session_state["_pending_merchant_switch"] = merchant
        # Always rerun so the confirmation above renders from a clean run (and, when the
        # merchant changed, so the sidebar picks up the staged switch) rather than leaving
        # a stale file-upload widget and preview on screen after a successful save.
        st.rerun()
