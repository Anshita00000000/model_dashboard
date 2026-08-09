"""Tab 4 — Feature Extraction: assemble the modelling dataset and profile it.

UI only: every computation is delegated to app.core.{assemble, derive,
raw_eda, schema}. This is the final Phase 2 stage — its export makes the
dataset ready for Phase 1's Tab 5 (Data Prep).

Note on the "invariant checks per batch" panel: CLAUDE.md's build spec
references an "Appendix A" for exact per-batch invariants that was not
included in the prompt this tab was built from. The panel below runs a
best-effort, generic set of structural checks (no duplicate lead_id within a
batch, stored run-summary counts matching the actual persisted responses,
fetched_at spread) rather than guessing at source-specific invariants that
were never specified.
"""

from __future__ import annotations

import json
import traceback
from collections import Counter
from typing import Optional

import pandas as pd
import streamlit as st

from app.core import assemble, derive, raw_eda
from app.core import schema as schema_mod
from app.core import storage

UPLOADED_BY = "streamlit-ui"
NONE_LABEL = "(none)"
EXPECTED_MATCH_RATE_LOW = 0.35
EXPECTED_MATCH_RATE_HIGH = 0.43
PROVENANCE_LEAK_SPREAD_THRESHOLD = 0.15
DEGENERATE_MIN_POSITIVES = 50
DEGENERATE_MAX_PASS_THROUGH = 0.95
IMMATURITY_BINS = [0, 7, 14, 30, 60, 90, 10**6]
IMMATURITY_LABELS = ["0-7", "7-14", "14-30", "30-60", "60-90", "90+"]


def _pick_outcome_column(df: pd.DataFrame) -> Optional[str]:
    """The last-derived `reached_<stage>` column corresponds to the final
    funnel stage (derive.derive_funnel_stages preserves stage_order), so it's
    the natural default binary outcome for immaturity/signal panels.
    """
    reached_cols = [c for c in df.columns if c.startswith("reached_")]
    return reached_cols[-1] if reached_cols else None


# ---------------------------------------------------------------------------
# Panel 1 — select a canonical dataset and assemble
# ---------------------------------------------------------------------------


def _panel_assemble(store: storage.MetadataStore, merchant: str) -> Optional[dict]:
    st.markdown("### 1 · Assemble")
    st.caption("Canonical data is the spine — every enrichment source is left-joined onto it. Enrichment never controls row count.")

    canonical_datasets = store.list_canonical_datasets(merchant=merchant)
    if not canonical_datasets:
        st.info("No canonical datasets for this merchant yet. Build one in the Enrichment tab first.")
        return None

    options = {
        f"{d['canonical_dataset_id']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d for d in canonical_datasets
    }
    labels = list(options.keys())
    default_idx = next(
        (i for i, l in enumerate(labels) if options[l]["canonical_dataset_id"] == st.session_state.get("canonical_dataset_id")),
        0,
    )
    choice = st.selectbox("Canonical dataset", labels, index=default_idx, key="feat_canonical_select")
    canonical_row = options[choice]
    st.session_state["canonical_dataset_id"] = canonical_row["canonical_dataset_id"]

    just_assembled = st.session_state.pop("_feat_just_assembled", None)
    if just_assembled:
        st.success(just_assembled)

    if st.button("Assemble (join canonical + enrichment)", type="primary", key="feat_assemble_btn"):
        try:
            result = assemble.assemble_dataset(store=store, canonical_dataset_id=canonical_row["canonical_dataset_id"])
        except Exception as exc:  # noqa: BLE001 - never swallow, surface the full traceback
            st.error(f"Assembly failed: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc(), language="python")
            return canonical_row

        st.session_state["feat_working_df"] = result.df
        st.session_state["feat_source_canonical_dataset_id"] = canonical_row["canonical_dataset_id"]
        st.session_state["feat_sources"] = [
            {"source": s.source, "feature_columns": s.feature_columns, "provenance_columns": s.provenance_columns}
            for s in result.sources
        ]
        st.session_state["feat_provenance_columns"] = result.provenance_columns()
        st.session_state["_feat_just_assembled"] = (
            f"Assembled {result.row_count:,} rows — {len(result.sources)} enrichment source(s) joined "
            f"({', '.join(s.source for s in result.sources) or 'none run yet'})."
        )
        st.rerun()

    if st.session_state.get("feat_source_canonical_dataset_id") != canonical_row["canonical_dataset_id"]:
        st.info("Selected canonical dataset differs from the last assembled one — click Assemble to (re)build.")

    return canonical_row


# ---------------------------------------------------------------------------
# Panel 2 — coverage per source
# ---------------------------------------------------------------------------


def _panel_coverage(df: pd.DataFrame, sources: list[dict]) -> None:
    st.markdown("### 2 · Coverage per source")
    if not sources:
        st.info("No enrichment sources have been run for this canonical dataset yet.")
        return

    rows = []
    for s in sources:
        source = s["source"]
        counts = df[f"{source}_match_status"].value_counts()
        total = len(df) or 1
        row = {"source": source, "rows": len(df)}
        for status, label in [("SUCCESS_MATCHED", "matched"), ("SUCCESS_NO_HIT", "no_hit"), ("ERROR", "error"), ("NOT_SENT", "not_sent")]:
            n = int(counts.get(status, 0))
            row[label] = n
            row[f"{label}_%"] = round(100 * n / total, 1)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for r in rows:
        rate = r["matched_%"] / 100
        if not (EXPECTED_MATCH_RATE_LOW <= rate <= EXPECTED_MATCH_RATE_HIGH):
            st.warning(
                f"**{r['source']}**: match rate {r['matched_%']}% falls outside the expected "
                f"{EXPECTED_MATCH_RATE_LOW:.0%}-{EXPECTED_MATCH_RATE_HIGH:.0%} coverage band."
            )


# ---------------------------------------------------------------------------
# Panel 3 — enrichment coverage by disposition (provenance leak detector)
# ---------------------------------------------------------------------------


def _panel_coverage_by_disposition(df: pd.DataFrame, sources: list[dict]) -> None:
    st.markdown("### 3 · Enrichment coverage by disposition")
    st.caption(
        "If coverage differs sharply by outcome, that's the signature of a provenance leak (CLAUDE.md — one "
        "merchant's converters were enriched at loan origination while non-converters were enriched at lead "
        "capture, and a model learned to detect the pull rather than the lead). Far cheaper to catch here."
    )
    if not sources or "disposition" not in df.columns:
        st.info("Nothing to show yet (no sources run, or no disposition column).")
        return

    for s in sources:
        source = s["source"]
        is_enriched_col = f"{source}_is_enriched"
        rate_by_disposition = df.groupby("disposition", observed=True)[is_enriched_col].mean()
        st.markdown(f"**{source}**")
        st.bar_chart(rate_by_disposition)
        st.dataframe(
            rate_by_disposition.rename("enrichment_rate").reset_index(), width="stretch", hide_index=True,
        )
        spread = float(rate_by_disposition.max() - rate_by_disposition.min()) if len(rate_by_disposition) else 0.0
        if spread > PROVENANCE_LEAK_SPREAD_THRESHOLD:
            st.error(
                f"🚨 **{source}**: enrichment-rate spread across dispositions is {spread:.1%} "
                f"(min {rate_by_disposition.min():.1%}, max {rate_by_disposition.max():.1%}) — investigate "
                f"before training. This is the exact signature of a provenance leak."
            )
        else:
            st.caption(f"Spread across dispositions: {spread:.1%} (below the {PROVENANCE_LEAK_SPREAD_THRESHOLD:.0%} flag threshold).")


# ---------------------------------------------------------------------------
# Panel 4 — invariant checks per batch (multi-batch sources only)
# ---------------------------------------------------------------------------


def _panel_invariants(store: storage.MetadataStore, canonical_dataset_id: str, sources: list[dict]) -> None:
    st.markdown("### 4 · Invariant checks per batch")
    all_runs = store.list_enrichment_runs(canonical_dataset_id=canonical_dataset_id)
    runs_by_source: dict[str, list[dict]] = {}
    for run in all_runs:
        runs_by_source.setdefault(run["source"], []).append(run)
    multi_batch_sources = {src: runs for src, runs in runs_by_source.items() if len(runs) > 1}

    if not multi_batch_sources:
        st.info("No source has been run more than once yet (no multi-batch source active) — nothing to check.")
        return

    st.caption(
        "Appendix A (referenced in this build's spec) wasn't provided, so these are best-effort generic "
        "batch-consistency checks rather than source-specific invariants."
    )
    for source, runs in multi_batch_sources.items():
        st.markdown(f"**{source}** — {len(runs)} batch(es)")
        rows = []
        for run in runs:
            responses = store.list_enrichment_responses(run_id=run["run_id"])
            lead_ids = [r["lead_id"] for r in responses]
            no_duplicates = len(lead_ids) == len(set(lead_ids))
            status_counts = Counter(r["status"] for r in responses)
            counts_match = (
                status_counts.get("SUCCESS_MATCHED", 0) == run["matched"]
                and status_counts.get("SUCCESS_NO_HIT", 0) == run["no_hit"]
                and status_counts.get("ERROR", 0) == run["error"]
                and status_counts.get("NOT_SENT", 0) == run["not_sent"]
            )
            fetched_ats = pd.to_datetime([r["fetched_at"] for r in responses], errors="coerce")
            spread = (fetched_ats.max() - fetched_ats.min()) if len(fetched_ats.dropna()) else pd.Timedelta(0)
            rows.append({
                "batch_id": run["batch_id"], "rows": len(responses),
                "no_duplicate_lead_ids": no_duplicates, "stored_counts_match_run_summary": counts_match,
                "fetched_at_spread": str(spread),
            })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, width="stretch", hide_index=True)
        failing = result_df[~(result_df["no_duplicate_lead_ids"] & result_df["stored_counts_match_run_summary"])]
        if len(failing) > 0:
            st.error(f"{source}: {len(failing)} batch(es) failed an invariant check.")


# ---------------------------------------------------------------------------
# Panel 5 — derived feature engineering
# ---------------------------------------------------------------------------


def _derive_temporal_panel(df: pd.DataFrame) -> None:
    with st.expander("Temporal features from created_at", expanded=True):
        date_like_cols = [c for c in df.columns if "date" in c.lower() or "created_at" in c.lower()] or list(df.columns)
        default_idx = date_like_cols.index("created_at") if "created_at" in date_like_cols else 0
        column = st.selectbox("Date column", date_like_cols, index=default_idx, key="feat_temporal_col")

        finding = raw_eda.detect_date_format(df, column)
        day_first: Optional[bool] = None
        if finding is not None and finding.inference == "ambiguous":
            confirm = st.radio(
                f"{column!r}'s format is ambiguous — confirm:", ["Not confirmed", "Day-first (DD/MM)", "Month-first (MM/DD)"],
                key="feat_temporal_dayfirst",
            )
            day_first = {"Day-first (DD/MM)": True, "Month-first (MM/DD)": False}.get(confirm)
        elif finding is not None:
            day_first = finding.inference == "day_first"
            st.caption(f"Detected format: {finding.inference} ({finding.evidence}).")

        if st.button("Apply temporal features", key="feat_apply_temporal"):
            if finding is not None and finding.inference == "ambiguous" and day_first is None:
                st.warning("Confirm day-first vs month-first before applying.")
            else:
                new_df, report = derive.derive_temporal_features(df, column=column, day_first=day_first)
                st.session_state["feat_working_df"] = new_df
                st.session_state["_feat_last_report"] = (
                    f"Added {len(report.produced_columns)} temporal column(s) from {column!r}: "
                    f"{report.parsed_count} parsed, {report.unparsed_count} unparsed."
                )
                st.rerun()


def _derive_funnel_panel(df: pd.DataFrame) -> None:
    with st.expander("Funnel target derivation", expanded=True):
        if "disposition" not in df.columns:
            st.info("No 'disposition' column in the assembled dataset.")
            return
        stage_text = st.text_input(
            "Ordered funnel stages (comma-separated, earliest first)",
            value=", ".join(st.session_state.get("feat_stage_order", ["Lead", "Engaged", "Consulted", "Converted"])),
            key="feat_funnel_stage_order",
        )
        stage_order = [s.strip() for s in stage_text.split(",") if s.strip()]

        distinct_values = sorted(v for v in df["disposition"].dropna().unique().tolist())
        st.caption("Map each raw disposition value to a stage (or leave unmapped — unmapped rows get a null outcome, never a guess).")
        mapping: dict[str, str] = {}
        for val in distinct_values:
            default_stage = val if val in stage_order else "(unmapped)"
            choice = st.selectbox(str(val), ["(unmapped)"] + stage_order, index=(["(unmapped)"] + stage_order).index(default_stage) if default_stage in (["(unmapped)"] + stage_order) else 0, key=f"feat_funnel_map_{val}")
            if choice != "(unmapped)":
                mapping[val] = choice

        if st.button("Apply funnel derivation", key="feat_apply_funnel"):
            if not stage_order:
                st.warning("Enter at least one stage.")
            else:
                try:
                    new_df, report = derive.derive_funnel_stages(df, stage_order=stage_order, mapping=mapping)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["feat_working_df"] = new_df
                    st.session_state["feat_stage_order"] = stage_order
                    msg = f"Added funnel_stage + {len(report.produced_columns) - 1} reached_<stage> column(s)."
                    if report.unmapped_values:
                        msg += f" {len(report.unmapped_values)} unmapped value(s): {report.unmapped_values}."
                    st.session_state["_feat_last_report"] = msg
                    st.rerun()


def _derive_multivalued_panel(df: pd.DataFrame) -> None:
    with st.expander("Multi-valued categorical normalisation", expanded=False):
        profiles = raw_eda.profile_columns(df)
        cat_cols = raw_eda.categorical_columns(profiles)
        findings = raw_eda.detect_multi_valued_cells(df, cat_cols)
        if not findings:
            st.caption("No multi-valued columns detected.")
            return
        for f in findings:
            enabled = st.checkbox(f"{f.column} (delimiter {f.delimiter!r}, {f.match_fraction:.0%} of rows)", value=True, key=f"feat_mv_enable_{f.column}")
            if not enabled:
                continue
            top_n = st.slider(f"Top N components to flag — {f.column}", 1, 30, 10, key=f"feat_mv_topn_{f.column}")
            if st.button(f"Apply to {f.column}", key=f"feat_mv_apply_{f.column}"):
                new_df, report = derive.normalize_multi_valued_column(df, f.column, delimiter=f.delimiter, top_n_components=top_n)
                st.session_state["feat_working_df"] = new_df
                st.session_state["_feat_last_report"] = f"Normalised {f.column!r}: {len(report.produced_columns)} column(s) added, {report.distinct_component_count} distinct component(s) found."
                st.rerun()


def _derive_duration_panel(df: pd.DataFrame) -> None:
    with st.expander("Free-text duration parsing", expanded=False):
        column = st.selectbox("Column to parse as a duration", [NONE_LABEL] + list(df.columns), key="feat_duration_col")
        if column == NONE_LABEL:
            return
        preview = df[column].map(derive.parse_duration_to_months)
        non_blank = ~df[column].map(schema_mod.is_blank)
        denom = max(1, int(non_blank.sum()))
        st.caption(f"Preview parse rate: {preview.notna().sum() / denom:.0%} of non-blank values.")
        if st.button("Apply duration parsing", key="feat_apply_duration"):
            new_df, report = derive.derive_duration_features(df, column)
            st.session_state["feat_working_df"] = new_df
            st.session_state["_feat_last_report"] = f"Parsed {column!r}: {report.parsed_count} parsed, {report.unparsed_count} unparsed."
            st.rerun()


def _derive_merge_panel(df: pd.DataFrame) -> None:
    with st.expander("Near-duplicate category consolidation", expanded=False):
        st.caption(
            "Re-runs Tab 2's near-duplicate detector on the assembled dataset. Case/whitespace merges are "
            "pre-checked (zero false-positive risk); fuzzy spelling-variant merges need your judgment."
        )
        profiles = raw_eda.profile_columns(df)
        cat_cols = raw_eda.categorical_columns(profiles)
        suggestions = derive.suggest_category_merges(df, cat_cols)
        if not suggestions:
            st.caption("No near-duplicate categories detected.")
            return
        confirmed = []
        for i, s in enumerate(suggestions):
            label = f"{s.column}: {s.variant!r} → {s.canonical!r}  ({s.kind}, n={s.count}" + (f", similarity={s.similarity})" if s.similarity else ")")
            if st.checkbox(label, value=s.default_confirmed, key=f"feat_merge_{i}"):
                confirmed.append(s)
        if st.button("Apply confirmed merges", key="feat_apply_merges"):
            new_df = derive.apply_category_merges(df, confirmed)
            st.session_state["feat_working_df"] = new_df
            st.session_state["_feat_last_report"] = f"Applied {len(confirmed)} confirmed merge(s)."
            st.rerun()


def _panel_derive(df: pd.DataFrame) -> None:
    st.markdown("### 5 · Derived feature engineering")
    last_report = st.session_state.pop("_feat_last_report", None)
    if last_report:
        st.success(last_report)
    _derive_temporal_panel(df)
    _derive_funnel_panel(df)
    _derive_multivalued_panel(df)
    _derive_duration_panel(df)
    _derive_merge_panel(df)


# ---------------------------------------------------------------------------
# Panel 6 — outcome rate vs days-since-creation (outcome immaturity)
# ---------------------------------------------------------------------------


def _panel_outcome_immaturity(df: pd.DataFrame) -> None:
    st.markdown("### 6 · Outcome rate vs days-since-creation")
    st.caption("Recently-created leads haven't had time to convert yet — a suppressed rate here is immaturity, not a real negative signal (CLAUDE.md).")
    if "created_at" not in df.columns:
        st.info("No created_at column.")
        return
    outcome_col = _pick_outcome_column(df)
    if outcome_col is None:
        st.info("Derive funnel stages above first to see this chart against a binary outcome.")
        return

    parsed = pd.to_datetime(df["created_at"], errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    days_since = (pd.Timestamp.now() - parsed).dt.days
    bucket = pd.cut(days_since, bins=IMMATURITY_BINS, labels=IMMATURITY_LABELS, right=False)
    rate = df.groupby(bucket, observed=True)[outcome_col].mean()
    st.bar_chart(rate)
    st.caption(f"Outcome column: {outcome_col}")


# ---------------------------------------------------------------------------
# Panel 7 — funnel summary
# ---------------------------------------------------------------------------


def _panel_funnel_summary(df: pd.DataFrame) -> None:
    st.markdown("### 7 · Funnel summary")
    stage_order = st.session_state.get("feat_stage_order", [])
    if "funnel_stage" not in df.columns or not stage_order:
        st.info("Derive funnel stages above to see this summary.")
        return

    reached_counts = {stage_order[0]: int(df["funnel_stage"].notna().sum())}
    for stage in stage_order[1:]:
        col = f"reached_{stage}"
        reached_counts[stage] = int(df[col].sum()) if col in df.columns else 0

    rows = []
    for i, stage in enumerate(stage_order):
        count = reached_counts[stage]
        if i == 0:
            rate = None
        else:
            prev_count = reached_counts[stage_order[i - 1]]
            rate = (count / prev_count) if prev_count > 0 else None
        degenerate = i > 0 and (count < DEGENERATE_MIN_POSITIVES or (rate is not None and rate > DEGENERATE_MAX_PASS_THROUGH))
        rows.append({
            "stage": stage, "count_reached": count,
            "pass_through_rate_from_prev": round(rate, 4) if rate is not None else None,
            "degenerate": degenerate,
        })

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, width="stretch", hide_index=True)
    degenerate_stages = result_df[result_df["degenerate"]]["stage"].tolist()
    if degenerate_stages:
        st.warning(
            f"Degenerate stage(s) — fewer than {DEGENERATE_MIN_POSITIVES} positives or a pass-through rate "
            f"above {DEGENERATE_MAX_PASS_THROUGH:.0%}, unsuitable for their own model (merge with a neighbour "
            f"instead): {', '.join(degenerate_stages)}"
        )


# ---------------------------------------------------------------------------
# Panel 8 — feature-level profiling
# ---------------------------------------------------------------------------


def _panel_feature_profiling(df: pd.DataFrame, sources: list[dict]) -> None:
    st.markdown("### 8 · Feature-level profiling")
    profiles = raw_eda.profile_columns(df)
    st.dataframe(
        pd.DataFrame([
            {"column": p.column, "dtype": p.inferred_dtype, "null_%": round(p.null_pct, 1), "distinct": p.distinct_count}
            for p in profiles
        ]),
        width="stretch", hide_index=True,
    )

    enrichment_cols = [c for s in sources for c in s["feature_columns"]]
    if not enrichment_cols:
        return
    st.markdown("###### Enriched vs unenriched split (enrichment features only)")
    col = st.selectbox("Feature", enrichment_cols, key="feat_profile_enrich_col")
    source = next(s["source"] for s in sources if col in s["feature_columns"])
    is_enriched_col = f"{source}_is_enriched"
    enriched_vals = df.loc[df[is_enriched_col] == True, col]  # noqa: E712
    unenriched_vals = df.loc[df[is_enriched_col] == False, col]  # noqa: E712
    dtype = next((p.inferred_dtype for p in profiles if p.column == col), "categorical")
    if dtype == "numeric":
        numeric_enriched = pd.to_numeric(enriched_vals, errors="coerce")
        stats_df = pd.DataFrame({
            "enriched": [len(enriched_vals), numeric_enriched.mean(), numeric_enriched.median()],
            "unenriched": [len(unenriched_vals), float("nan"), float("nan")],
        }, index=["count", "mean", "median"])
    else:
        stats_df = pd.DataFrame({
            "enriched": [len(enriched_vals), enriched_vals.nunique()],
            "unenriched": [len(unenriched_vals), unenriched_vals.nunique()],
        }, index=["count", "distinct"])
    st.dataframe(stats_df, width="stretch")


# ---------------------------------------------------------------------------
# Panel 9 — univariate signal
# ---------------------------------------------------------------------------


def _panel_univariate_signal(df: pd.DataFrame) -> None:
    st.markdown("### 9 · Univariate signal")
    st.caption("Outcome rate by quartile (numeric) or category — a first read on which features carry signal.")
    outcome_col = _pick_outcome_column(df)
    if outcome_col is None:
        st.info("Derive funnel stages above to see univariate signal against a binary outcome.")
        return

    profiles = raw_eda.profile_columns(df)
    numeric_cols = [c for c in raw_eda.numeric_columns(profiles) if c != outcome_col and not c.startswith("reached_")]
    categorical_cols = [c for c in raw_eda.categorical_columns(profiles) if c != outcome_col]
    feature = st.selectbox("Feature", [NONE_LABEL] + numeric_cols + categorical_cols, key="feat_univariate_feature")
    if feature == NONE_LABEL:
        return

    if feature in numeric_cols:
        numeric_series = pd.to_numeric(df[feature], errors="coerce")
        try:
            quartile = pd.qcut(numeric_series, 4, duplicates="drop")
        except ValueError:
            st.caption("Not enough distinct values to form quartiles.")
            return
        rate = df.groupby(quartile, observed=True)[outcome_col].mean()
    else:
        rate = df.groupby(feature, observed=True)[outcome_col].mean().sort_values(ascending=False).head(20)
    st.bar_chart(rate)


# ---------------------------------------------------------------------------
# Panel 10 — export
# ---------------------------------------------------------------------------


def _panel_export(store: storage.MetadataStore, merchant: str, storage_root) -> None:
    st.markdown("### 10 · Export")
    st.caption("Registers this dataset as a Phase-1-compatible artifact, ready for Tab 5 · Data Prep.")
    df = st.session_state["feat_working_df"]
    columns = list(df.columns)

    just_exported = st.session_state.pop("_feat_just_exported", None)
    if just_exported:
        st.success(f"Registered dataset_id = {just_exported}. Ready for Tab 5 · Data Prep.")

    id_col = st.selectbox(
        "ID column", columns, index=columns.index("lead_id") if "lead_id" in columns else 0, key="feat_export_id_col",
    )
    target_candidates = [c for c in columns if c == "funnel_stage" or c.startswith("reached_") or c == "disposition"]
    target_options = [NONE_LABEL] + target_candidates
    target_col = st.selectbox("Target column", target_options, key="feat_export_target_col")
    metadata_defaults = [c for c in ("merchant", "created_at") if c in columns]
    metadata_cols = st.multiselect(
        "Metadata columns (carried through, never trained on)", [c for c in columns if c != id_col],
        default=metadata_defaults, key="feat_export_metadata_cols",
    )
    purpose = st.radio("Purpose", ["train", "predict"], key="feat_export_purpose")

    if st.button("Register dataset", type="primary", key="feat_export_btn"):
        overrides: dict[str, dict] = {id_col: {"role": "id"}}
        if target_col != NONE_LABEL:
            overrides[target_col] = {"role": "target"}
        for col in metadata_cols:
            overrides[col] = {"role": "metadata"}
        for col in st.session_state.get("feat_provenance_columns", []):
            if col in columns:
                overrides[col] = {"role": "excluded"}

        source_canonical_id = st.session_state.get("feat_source_canonical_dataset_id", "")
        row = assemble.register_dataset(
            store=store, root=storage_root, df=df, merchant=merchant, purpose=purpose,
            source_file=f"assembled from canonical_dataset_id={source_canonical_id}",
            role_overrides=overrides,
        )
        st.session_state["dataset_id"] = row["dataset_id"]
        st.session_state["_feat_just_exported"] = row["dataset_id"]
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(store: storage.MetadataStore, merchant: str, storage_root=storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Feature Extraction")
    st.caption(
        "Assemble the modelling dataset (canonical spine + enrichment) and profile it. This is the final "
        "Phase 2 stage — its export is what Tab 5 · Data Prep consumes."
    )

    canonical_row = _panel_assemble(store, merchant)
    if canonical_row is None:
        return

    working_df = st.session_state.get("feat_working_df")
    assembled_for = st.session_state.get("feat_source_canonical_dataset_id")
    if working_df is None or assembled_for != canonical_row["canonical_dataset_id"]:
        return

    sources = st.session_state.get("feat_sources", [])

    st.divider()
    _panel_coverage(working_df, sources)
    st.divider()
    _panel_coverage_by_disposition(working_df, sources)
    st.divider()
    _panel_invariants(store, canonical_row["canonical_dataset_id"], sources)
    st.divider()
    _panel_derive(working_df)

    # Re-fetch: derive panels may have just replaced the working df via st.rerun()
    working_df = st.session_state.get("feat_working_df", working_df)

    st.divider()
    _panel_outcome_immaturity(working_df)
    st.divider()
    _panel_funnel_summary(working_df)
    st.divider()
    _panel_feature_profiling(working_df, sources)
    st.divider()
    _panel_univariate_signal(working_df)
    st.divider()
    _panel_export(store, merchant, storage_root)
