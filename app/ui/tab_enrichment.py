"""Tab 3 — Enrichment.

Three panels, in order: (1) cleaned-file upload with an optional lineage link
back to a Tab 1 raw file, (2) canonical field mapping across merchants, and
(3) enrichment orchestration. API calls are STUBBED — the adapter interface
and orchestration are real, network calls return mock data (see
app/core/enrichment/).

UI only: every decision is delegated to app.core.{ingest, raw_eda, canonical,
enrichment}. Nothing here applies a mapping automatically — every suggestion
is shown for the user to confirm or override.
"""

from __future__ import annotations

import json
import traceback
import uuid
from typing import Optional

import pandas as pd
import streamlit as st

from app.core import canonical, ingest, raw_eda, storage
from app.core.enrichment import ADAPTER_REGISTRY
from app.core.enrichment import runner as enrichment_runner

UPLOADED_BY = "streamlit-ui"
NONE_LABEL = "(none)"
ACCEPTED_TYPES = ["csv", "tsv", "xlsx", "xls"]
EXPECTED_MATCH_RATE_LOW = 0.35
EXPECTED_MATCH_RATE_HIGH = 0.43
ALIAS_AUTO_PREFILL_CONFIDENCE = 0.90  # only pre-check a suggestion this confident; still just a default, never final


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataframe(path: str) -> pd.DataFrame:
    return storage.load_dataframe(path)


# ---------------------------------------------------------------------------
# Panel 1 — cleaned-file upload
# ---------------------------------------------------------------------------


def _panel_upload(store: storage.MetadataStore, merchant: str, storage_root) -> Optional[dict]:
    st.markdown("### 1 · Cleaned-file upload")
    st.caption(
        "The team cleans offline; upload the result here. Optionally link it back to a raw file "
        "from Tab 1 for lineage — a cleaned file may also be assembled from several raw files, "
        "so this is optional."
    )

    just_saved = st.session_state.pop("_enrich_upload_just_saved", None)
    if just_saved:
        st.success(f"Saved cleaned_dataset_id = {just_saved}")

    cleaned_datasets = store.list_cleaned_datasets(merchant=merchant)
    selected: Optional[dict] = None
    if cleaned_datasets:
        options = {
            f"{d['cleaned_dataset_id']}  ·  {d['original_filename']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d
            for d in cleaned_datasets
        }
        labels = list(options.keys())
        default_idx = next(
            (i for i, l in enumerate(labels) if options[l]["cleaned_dataset_id"] == st.session_state.get("cleaned_dataset_id")),
            0,
        )
        choice = st.selectbox("Existing cleaned dataset", labels, index=default_idx, key="enrich_cleaned_select")
        selected = options[choice]
        st.session_state["cleaned_dataset_id"] = selected["cleaned_dataset_id"]
    else:
        st.info("No cleaned datasets uploaded yet for this merchant.")

    with st.expander("Upload a new cleaned file", expanded=not cleaned_datasets):
        col_purpose, col_lineage = st.columns(2)
        with col_purpose:
            purpose = st.radio("Purpose", ["train", "predict"], key="enrich_upload_purpose", horizontal=True)
        with col_lineage:
            raw_datasets = store.list_raw_datasets(merchant=merchant)
            raw_options = {NONE_LABEL: None}
            raw_options.update({
                f"{d['raw_dataset_id']}  ·  {d['original_filename']}  ·  {d['created_at']}": d for d in raw_datasets
            })
            raw_choice = st.selectbox(
                "Lineage: link to a raw dataset (optional)", list(raw_options.keys()), key="enrich_lineage_select"
            )
            source_raw = raw_options[raw_choice]

        uploaded = st.file_uploader("Cleaned lead file", type=ACCEPTED_TYPES, key="enrich_cleaned_uploader")
        notes = st.text_area("Notes (optional)", key="enrich_cleaned_notes")

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            filename = uploaded.name
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            sheet_name = None
            if ext in ("xlsx", "xls"):
                try:
                    sheets = ingest.list_excel_sheets(raw_bytes)
                except Exception as exc:  # noqa: BLE001 - surface, don't crash on a bad workbook
                    st.error(f"Could not read this workbook: {exc}")
                    return selected
                sheet_name = st.selectbox("Sheet", sheets, index=0, key="enrich_cleaned_sheet") if len(sheets) > 1 else sheets[0]

            try:
                result = ingest.load_raw_file(raw_bytes, filename, sheet_name=sheet_name)
            except ValueError as exc:
                st.error(str(exc))
                return selected

            st.divider()
            st.success(f"Read {filename} — {result.row_count:,} rows × {result.col_count} columns")
            c1, c2, c3 = st.columns(3)
            c1.metric("Delimiter", f"{result.delimiter_label} ({result.delimiter!r})" if result.delimiter else "n/a (Excel)")
            c2.metric("Encoding", result.encoding)
            c3.metric("Malformed rows", result.malformed_row_count)
            st.markdown("###### Preview (first 20 rows)")
            st.dataframe(result.df.head(20), width="stretch")

            if source_raw is not None:
                raw_df = _load_dataframe(source_raw["parquet_path"])
                diff = canonical.column_diff(list(raw_df.columns), list(result.df.columns))
                st.markdown("##### Lineage: column diff vs the linked raw file")
                st.caption("Purely informational — a 'renamed' pairing is a heuristic guess, not applied anywhere.")
                d1, d2, d3 = st.columns(3)
                d1.metric("Added", len(diff.added))
                d2.metric("Removed", len(diff.removed))
                d3.metric("Renamed (guess)", len(diff.renamed))
                with st.expander("Column diff detail"):
                    if diff.added:
                        st.write("**Added:** " + ", ".join(diff.added))
                    if diff.removed:
                        st.write("**Removed:** " + ", ".join(diff.removed))
                    if diff.renamed:
                        st.write("**Renamed (heuristic guess):** " + ", ".join(f"{o} → {n}" for o, n in diff.renamed))
                    if diff.unchanged:
                        st.write("**Unchanged:** " + ", ".join(diff.unchanged))

                st.markdown("##### Did cleaning resolve the flagged findings?")
                raw_profiles = raw_eda.profile_columns(raw_df)
                cleaned_profiles = raw_eda.profile_columns(result.df)
                raw_cat, raw_num = raw_eda.categorical_columns(raw_profiles), raw_eda.numeric_columns(raw_profiles)
                cln_cat, cln_num = raw_eda.categorical_columns(cleaned_profiles), raw_eda.numeric_columns(cleaned_profiles)
                comparison = [
                    ("Near-duplicate categories",
                     len(raw_eda.find_near_duplicate_categories(raw_df, raw_cat)),
                     len(raw_eda.find_near_duplicate_categories(result.df, cln_cat))),
                    ("Sentinel / disguised-code values",
                     len(raw_eda.detect_sentinel_values(raw_df, raw_num)),
                     len(raw_eda.detect_sentinel_values(result.df, cln_num))),
                    ("Ambiguous date formats",
                     sum(1 for f in raw_eda.detect_date_formats(raw_df) if f.inference == "ambiguous"),
                     sum(1 for f in raw_eda.detect_date_formats(result.df) if f.inference == "ambiguous")),
                    ("Float-formatted identifiers",
                     len(raw_eda.detect_float_formatted_identifiers(raw_df)),
                     len(raw_eda.detect_float_formatted_identifiers(result.df))),
                    ("Multi-valued cells",
                     len(raw_eda.detect_multi_valued_cells(raw_df, raw_cat)),
                     len(raw_eda.detect_multi_valued_cells(result.df, cln_cat))),
                    ("Free-text in categorical fields",
                     len(raw_eda.detect_free_text_fields(raw_df, raw_cat)),
                     len(raw_eda.detect_free_text_fields(result.df, cln_cat))),
                ]
                st.dataframe(
                    pd.DataFrame(comparison, columns=["Finding", "Raw (before)", "Cleaned (after)"]),
                    width="stretch", hide_index=True,
                )

            if st.button("Save cleaned dataset", type="primary", key="enrich_save_cleaned_btn"):
                save_result = ingest.save_cleaned_dataset(
                    store=store, root=storage_root, merchant=merchant, purpose=purpose, result=result,
                    uploaded_by=UPLOADED_BY,
                    source_raw_dataset_id=source_raw["raw_dataset_id"] if source_raw else None,
                    notes=notes,
                )
                st.session_state["cleaned_dataset_id"] = save_result.cleaned_dataset_id
                st.session_state["_enrich_upload_just_saved"] = save_result.cleaned_dataset_id
                st.rerun()

    return selected


# ---------------------------------------------------------------------------
# Panel 2 — canonical field mapping
# ---------------------------------------------------------------------------


def _report_to_dict(report: canonical.ValidationReport) -> dict:
    return {
        "is_valid": report.is_valid,
        "issues": [{"kind": i.kind, "field": i.field_name, "message": i.message, "severity": i.severity} for i in report.issues],
        "phone_match_rate": None if pd.isna(report.phone_match_rate) else report.phone_match_rate,
        "phone_normalized_count": report.phone_normalized_count,
        "phone_failed_count": report.phone_failed_count,
        "created_at_parsed_count": report.created_at_parsed_count,
        "created_at_unparsed_count": report.created_at_unparsed_count,
        "created_at_inferred_format": report.created_at_inferred_format,
        "disposition_value_counts": report.disposition_value_counts,
    }


def _panel_mapping(store: storage.MetadataStore, merchant: str, cleaned_row: Optional[dict], storage_root) -> Optional[dict]:
    st.markdown("### 2 · Canonical field mapping")
    st.caption("Consistency across merchants — nothing downstream should know which clinic a row came from.")

    if cleaned_row is None:
        st.info("Select or upload a cleaned dataset above first.")
        return None

    canonical.seed_canonical_fields(store)
    fields = canonical.list_canonical_fields(store)

    df = _load_dataframe(cleaned_row["parquet_path"])
    source_columns = list(df.columns)

    with st.expander("Add a new canonical field", expanded=False):
        c1, c2, c3 = st.columns(3)
        new_name = c1.text_input("Field name", key="enrich_new_field_name")
        new_dtype = c2.selectbox("Dtype", ["text", "categorical", "numeric", "date", "id"], key="enrich_new_field_dtype")
        new_required = c3.selectbox(
            "Required", [canonical.OPTIONAL, canonical.REQUIRED, canonical.REQUIRED_FOR_TRAIN], key="enrich_new_field_required"
        )
        new_desc = st.text_input("Description", key="enrich_new_field_desc")
        if st.button("Add field", key="enrich_add_field_btn"):
            if new_name.strip():
                canonical.add_canonical_field(
                    store, name=new_name.strip(), dtype=new_dtype, required_level=new_required, description=new_desc
                )
                st.session_state["_enrich_field_added"] = new_name.strip()
                st.rerun()
            else:
                st.warning("Enter a field name.")

    just_added = st.session_state.pop("_enrich_field_added", None)
    if just_added:
        st.success(f"Added canonical field {just_added!r}")

    latest_mapping_row = canonical.get_latest_field_mapping(store, merchant)
    prefill = canonical.load_mapping_dict(latest_mapping_row) if latest_mapping_row else {}
    suggestions = canonical.suggest_mapping(source_columns, fields)

    st.caption(
        "Auto-suggested by name similarity — confirm or override every field yourself; nothing is applied "
        "automatically. Selecting more than one source column coalesces them left-to-right (first non-null wins)."
    )

    mapping: dict[str, list[str]] = {}
    for cfield in fields:
        name = cfield["name"]
        if name in prefill:
            default = [c for c in prefill[name] if c in source_columns]
        else:
            top = [s.source_column for s in suggestions.get(name, []) if s.confidence >= ALIAS_AUTO_PREFILL_CONFIDENCE]
            default = top[:1]
        badge = {
            canonical.REQUIRED: " (required)",
            canonical.REQUIRED_FOR_TRAIN: " (required for train)",
        }.get(cfield["required_level"], "")
        chosen = st.multiselect(
            f"{name}{badge}", source_columns, default=default, key=f"enrich_map_{name}",
            help=cfield.get("description") or None,
        )
        if chosen:
            mapping[name] = chosen

    mapped_cols = {c for cols in mapping.values() for c in cols}
    unmapped = [c for c in source_columns if c not in mapped_cols]
    if unmapped:
        st.caption(f"Unmapped columns (preserved under `extra__`, never dropped): {', '.join(unmapped)}")

    mapping_name = st.text_input("Mapping profile name", value=f"{merchant} mapping", key="enrich_mapping_name")
    if st.button("Save mapping profile", key="enrich_save_mapping_btn"):
        mapping_id = canonical.save_field_mapping(
            store, merchant=merchant, name=mapping_name, mapping=mapping, unmapped_columns=unmapped,
            created_by=UPLOADED_BY,
        )
        st.session_state["mapping_id"] = mapping_id
        st.session_state["_enrich_mapping_just_saved"] = mapping_id
        st.rerun()

    just_saved_mapping = st.session_state.pop("_enrich_mapping_just_saved", None)
    if just_saved_mapping:
        st.success(f"Saved mapping profile version, mapping_id = {just_saved_mapping}")

    mapping_row = store.get_field_mapping(st.session_state["mapping_id"]) if st.session_state.get("mapping_id") else None
    if mapping_row is None or mapping_row["merchant"] != merchant:
        return None

    st.divider()
    st.markdown("##### Validation")
    saved_mapping = canonical.load_mapping_dict(mapping_row)
    canonical_df = canonical.apply_mapping(df, saved_mapping)
    purpose = cleaned_row["purpose"]

    created_at_source_cols = saved_mapping.get("created_at", [])
    created_at_ambiguous = False
    if created_at_source_cols:
        finding = raw_eda.detect_date_format(df, created_at_source_cols[0])
        created_at_ambiguous = finding is not None and finding.inference == "ambiguous"

    created_at_day_first: Optional[bool] = None
    if created_at_ambiguous:
        confirm = st.radio(
            "created_at's format was flagged ambiguous (day-first vs month-first) in Tab 2 — confirm before validating:",
            ["Not yet confirmed", "Day-first (DD/MM)", "Month-first (MM/DD)"], key="enrich_created_at_confirm",
        )
        created_at_day_first = {"Day-first (DD/MM)": True, "Month-first (MM/DD)": False}.get(confirm)

    min_phone_rate = st.slider("Minimum phone normalisation rate", 0.0, 1.0, 0.90, key="enrich_min_phone_rate")

    report = canonical.validate_canonical(
        canonical_df, fields, purpose=purpose, min_phone_match_rate=min_phone_rate,
        created_at_ambiguous=created_at_ambiguous, created_at_day_first=created_at_day_first,
    )

    if report.issues:
        for issue in report.issues:
            (st.error if issue.severity == "error" else st.warning)(f"**{issue.field_name or '—'}**: {issue.message}")
    else:
        st.success("No validation issues.")

    if not pd.isna(report.phone_match_rate):
        st.metric(
            "Phone normalisation rate", f"{report.phone_match_rate:.1%}",
            f"{report.phone_normalized_count} ok / {report.phone_failed_count} failed",
        )
        if report.phone_failed_samples:
            with st.expander("Sample phone normalisation failures"):
                for s in report.phone_failed_samples:
                    st.code(s, language=None)

    if report.disposition_value_counts:
        st.markdown("###### disposition value counts")
        st.bar_chart(pd.DataFrame(report.disposition_value_counts, columns=["value", "count"]).set_index("value"))

    if report.is_valid:
        if st.button("Save canonical dataset", type="primary", key="enrich_save_canonical_btn"):
            artifact_path = storage.save_dataframe(
                canonical_df, "canonical", merchant, f"{mapping_row['mapping_id']}_canonical", root=storage_root
            )
            canonical_dataset_id = uuid.uuid4().hex
            store.insert_canonical_dataset(
                canonical_dataset_id=canonical_dataset_id, merchant=merchant, purpose=purpose,
                cleaned_dataset_id=cleaned_row["cleaned_dataset_id"], mapping_id=mapping_row["mapping_id"],
                row_count=len(canonical_df), col_count=len(canonical_df.columns), artifact_path=str(artifact_path),
                validation_json=json.dumps(_report_to_dict(report)), created_by=UPLOADED_BY,
            )
            st.session_state["canonical_dataset_id"] = canonical_dataset_id
            st.session_state["_enrich_canonical_just_saved"] = canonical_dataset_id
            st.rerun()
    else:
        st.info("Resolve the error(s) above before saving a canonical dataset.")

    just_saved_canonical = st.session_state.pop("_enrich_canonical_just_saved", None)
    if just_saved_canonical:
        st.success(f"Saved canonical_dataset_id = {just_saved_canonical}")

    canonical_row = (
        store.get_canonical_dataset(st.session_state["canonical_dataset_id"])
        if st.session_state.get("canonical_dataset_id") else None
    )
    if canonical_row and canonical_row["merchant"] == merchant:
        return canonical_row
    return None


# ---------------------------------------------------------------------------
# Panel 3 — enrichment orchestration
# ---------------------------------------------------------------------------


def _panel_enrichment(store: storage.MetadataStore, merchant: str, canonical_row: Optional[dict]) -> None:
    st.markdown("### 3 · Enrichment")
    st.caption(
        "API calls are stubbed — this is the real adapter interface and orchestration, but fetch() returns "
        "mock data rather than calling CRIF/Equifax/EPFO/etc. over the network."
    )

    if canonical_row is None:
        canonical_datasets = store.list_canonical_datasets(merchant=merchant)
        if not canonical_datasets:
            st.info("Save a canonical dataset in panel 2 above first.")
            return
        options = {
            f"{d['canonical_dataset_id']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d for d in canonical_datasets
        }
        choice = st.selectbox("Canonical dataset", list(options.keys()), key="enrich_canonical_select")
        canonical_row = options[choice]
        st.session_state["canonical_dataset_id"] = canonical_row["canonical_dataset_id"]

    st.caption(f"Enriching canonical_dataset_id = {canonical_row['canonical_dataset_id']} ({canonical_row['row_count']:,} rows)")
    canonical_df = _load_dataframe(canonical_row["artifact_path"])

    source_names = list(ADAPTER_REGISTRY.keys())
    selected_sources = st.multiselect("Sources to run", source_names, default=source_names, key="enrich_sources_select")

    configs: dict[str, dict] = {}
    if selected_sources:
        with st.expander("Per-source configuration", expanded=False):
            for source in selected_sources:
                st.markdown(f"**{source}**")
                c1, c2, c3, c4 = st.columns(4)
                match_rate = c1.slider("Match rate", 0.0, 1.0, 0.40, key=f"enrich_cfg_match_{source}")
                error_rate = c2.slider("Error rate", 0.0, 0.30, 0.03, key=f"enrich_cfg_error_{source}")
                not_sent_rate = c3.slider("Not-sent rate", 0.0, 0.30, 0.02, key=f"enrich_cfg_notsent_{source}")
                seed_text = c4.text_input("Seed (optional)", key=f"enrich_cfg_seed_{source}")
                seed = int(seed_text) if seed_text.strip().isdigit() else None
                configs[source] = dict(match_rate=match_rate, error_rate=error_rate, not_sent_rate=not_sent_rate, seed=seed)

    summary_key = f"_enrich_summaries_{canonical_row['canonical_dataset_id']}"

    if st.button("Run enrichment", type="primary", disabled=not selected_sources, key="enrich_run_btn"):
        records = enrichment_runner.build_canonical_records(canonical_df)
        adapters = [ADAPTER_REGISTRY[s](**configs[s]) for s in selected_sources]
        progress_bar = st.progress(0.0, text=f"0/{len(adapters)} source(s) complete")
        done: list = []

        def _on_progress(source_name: str, summary) -> None:
            done.append(summary)
            progress_bar.progress(len(done) / len(adapters), text=f"{len(done)}/{len(adapters)} source(s) complete — {source_name} done")

        try:
            summaries = enrichment_runner.run_enrichment(
                store=store, canonical_dataset_id=canonical_row["canonical_dataset_id"], records=records,
                adapters=adapters, progress_callback=_on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - never swallow; surface the full traceback
            st.error(f"Enrichment run failed: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc(), language="python")
        else:
            st.session_state[summary_key] = summaries

    summaries = st.session_state.get(summary_key, [])
    if summaries:
        st.markdown("##### Coverage summary")
        rows = []
        for s in summaries:
            total = s.rows_attempted or 1
            match_rate = s.matched / total
            rows.append({
                "source": s.source, "attempted": s.rows_attempted,
                "matched": s.matched, "matched_%": round(100 * match_rate, 1),
                "no_hit": s.no_hit, "no_hit_%": round(100 * s.no_hit / total, 1),
                "error": s.error, "error_%": round(100 * s.error / total, 1),
                "not_sent": s.not_sent, "not_sent_%": round(100 * s.not_sent / total, 1),
                "batch_id": s.batch_id,
            })
            if not (EXPECTED_MATCH_RATE_LOW <= match_rate <= EXPECTED_MATCH_RATE_HIGH):
                st.warning(
                    f"**{s.source}**: match rate {match_rate:.1%} falls outside the expected "
                    f"{EXPECTED_MATCH_RATE_LOW:.0%}–{EXPECTED_MATCH_RATE_HIGH:.0%} coverage band — investigate "
                    f"before trusting this run."
                )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Every response — matched, no-hit, error, or not-sent — is persisted with its own fetched_at and "
            "batch_id (see enrichment_responses). A timing-leakage check "
            "(app.core.enrichment.base.check_enrichment_timing_leakage) is available once a canonical outcome-date "
            "field is mapped; it warns when a source's fetched_at falls after the lead's outcome date."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(store: storage.MetadataStore, merchant: str, storage_root=storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Enrichment")
    st.caption(
        "Cleaned-file upload, canonical field mapping, and enrichment orchestration. API calls are stubbed — "
        "the adapter interface and orchestration are real; network calls return mock data."
    )
    canonical.seed_canonical_fields(store)

    cleaned_row = _panel_upload(store, merchant, storage_root)
    st.divider()
    canonical_row = _panel_mapping(store, merchant, cleaned_row, storage_root)
    st.divider()
    _panel_enrichment(store, merchant, canonical_row)
