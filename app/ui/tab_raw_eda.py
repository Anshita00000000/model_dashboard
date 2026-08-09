"""Tab 2 — Raw EDA.

UI only: every computation is delegated to app.core.raw_eda. Read-only — this
tab NEVER writes back a modified dataset. Its output is a report plus flagged
findings the team can act on during their offline cleaning; the report is
structured so it doubles as a cleaning worklist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.core import raw_eda
from app.core import storage

NONE_LABEL = "(none)"


@st.cache_data(show_spinner="Loading raw dataset...")
def _load_dataframe(path: str) -> pd.DataFrame:
    return storage.load_dataframe(path)


def _day_first_from_inference(inference: str) -> Optional[bool]:
    if inference == "day_first":
        return True
    if inference == "month_first":
        return False
    return None


def render(store: storage.MetadataStore, merchant: str) -> None:
    st.subheader("Raw EDA")
    st.caption(
        "Automated profiling of the raw file — read-only. This tab never writes back a modified "
        "dataset; its output is a report plus findings for the team's offline cleaning pass."
    )

    raw_datasets = store.list_raw_datasets(merchant=merchant)
    if not raw_datasets:
        st.info("No raw datasets uploaded yet for this merchant. Upload one in the Upload tab first.")
        return

    options = {
        f"{d['raw_dataset_id']}  ·  {d['original_filename']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d
        for d in raw_datasets
    }
    labels = list(options.keys())
    default_idx = next(
        (i for i, l in enumerate(labels) if options[l]["raw_dataset_id"] == st.session_state.get("raw_dataset_id")), 0
    )
    raw_row = options[st.selectbox("Raw dataset", labels, index=default_idx, key="raw_eda_dataset_select")]
    st.session_state["raw_dataset_id"] = raw_row["raw_dataset_id"]

    df = _load_dataframe(raw_row["parquet_path"])
    st.caption(
        f"Source: {raw_row['original_filename']}  ·  delimiter={raw_row['delimiter']!r}  ·  "
        f"encoding={raw_row['encoding']}  ·  uploaded by {raw_row['uploaded_by']}"
    )
    if raw_row.get("notes"):
        st.caption(f"Notes: {raw_row['notes']}")

    st.divider()

    # ---- overview ----
    overview = raw_eda.profile_overview(df)
    st.markdown("#### Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview.row_count:,}")
    c2.metric("Columns", overview.col_count)
    c3.metric("Duplicate rows", f"{overview.fully_duplicate_row_count:,}")
    c4.metric("Empty columns", len(overview.fully_empty_columns))
    if overview.fully_empty_columns:
        st.caption(f"Fully empty: {', '.join(overview.fully_empty_columns)}")

    # ---- per-column profile ----
    columns = raw_eda.profile_columns(df)
    cat_cols = raw_eda.categorical_columns(columns)
    num_cols = raw_eda.numeric_columns(columns)

    st.markdown("#### Per-column profile")
    st.caption("dtype is INFERRED FOR PROFILING ONLY — never applied to the stored data.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "column": c.column, "inferred_dtype": c.inferred_dtype, "null_%": round(c.null_pct, 1),
                    "distinct": c.distinct_count, "samples": ", ".join(c.sample_values[:3]),
                    "most_frequent": ", ".join(f"{v} ({n})" for v, n in c.most_frequent[:3]),
                }
                for c in columns
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    # ---- findings ----
    st.markdown("#### Findings — cleaning worklist")

    with st.expander("Near-duplicate categories", expanded=False):
        findings = raw_eda.find_near_duplicate_categories(df, cat_cols)
        if findings:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "column": f.column,
                            "confidence": "high (case/whitespace)" if f.kind == "case_or_whitespace" else f"possible (similarity={f.similarity})",
                            "canonical": f.canonical,
                            "variants": ", ".join(f"{v!r} ({c})" for v, c in f.variants),
                        }
                        for f in findings
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("None found.")

    with st.expander("Sentinel / disguised-code values", expanded=False):
        findings = raw_eda.detect_sentinel_values(df, num_cols)
        if findings:
            for f in findings:
                st.write(f"**{f.column}**: {f.message}")
        else:
            st.caption("None found.")

    with st.expander("Ambiguous date formats", expanded=False):
        findings = raw_eda.detect_date_formats(df)
        if findings:
            st.dataframe(
                pd.DataFrame([{"column": f.column, "inference": f.inference, "evidence": f.evidence} for f in findings]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No date-like columns detected.")

    with st.expander("Float-formatted identifiers", expanded=False):
        findings = raw_eda.detect_float_formatted_identifiers(df)
        if findings:
            for f in findings:
                st.write(f"**{f.column}**: {f.match_fraction:.0%} of values look like a float-formatted ID/phone number, e.g. {f.sample_values[0]!r} — strip the trailing '.0' before extracting digits.")
        else:
            st.caption("None found.")

    with st.expander("Multi-valued cells", expanded=False):
        findings = raw_eda.detect_multi_valued_cells(df, cat_cols)
        if findings:
            for f in findings:
                order_note = "order VARIES between rows" if f.order_varies else "order is consistent"
                st.write(
                    f"**{f.column}** (delimiter {f.delimiter!r}): {f.distinct_component_count} distinct "
                    f"component(s) across {f.match_fraction:.0%} of rows, {order_note}"
                )
        else:
            st.caption("None found.")

    with st.expander("Free-text in categorical-looking fields", expanded=False):
        findings = raw_eda.detect_free_text_fields(df, cat_cols)
        if findings:
            for f in findings:
                st.write(f"**{f.column}**: {f.distinct_count}/{f.row_count} distinct values, {f.singleton_ratio:.0%} appear exactly once — candidate for parsing/bucketing.")
        else:
            st.caption("None found.")

    # ---- distributions ----
    st.markdown("#### Distributions")
    dist_column = st.selectbox("Column", list(df.columns), key="raw_eda_dist_column")
    dtype = next(c.inferred_dtype for c in columns if c.column == dist_column)
    date_finding = raw_eda.detect_date_format(df, dist_column)

    if date_finding is not None:
        st.caption(f"Detected as date-like → {date_finding.inference}: {date_finding.evidence}")
        dd = raw_eda.date_distribution(df, dist_column, _day_first_from_inference(date_finding.inference))
        if dd is not None:
            chart_df = pd.DataFrame(dd.records_per_day, columns=["date", "count"]).set_index("date")
            st.bar_chart(chart_df)
            st.caption(f"{dd.min_date} to {dd.max_date}")
        else:
            st.caption("Format is ambiguous or mixed — skipping the chart rather than guessing a parse order.")
    elif dtype == "numeric":
        nd = raw_eda.numeric_distribution(df, dist_column)
        if nd is not None:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Min", f"{nd.min:g}")
            m2.metric("Max", f"{nd.max:g}")
            m3.metric("Mean", f"{nd.mean:.2f}")
            m4.metric("Median", f"{nd.median:.2f}")
            m5.metric("Std", f"{nd.std:.2f}")
            bin_labels = [f"{nd.histogram_bins[i]:.2g}" for i in range(len(nd.histogram_counts))]
            st.bar_chart(pd.DataFrame({"bin_start": bin_labels, "count": nd.histogram_counts}).set_index("bin_start"))
        else:
            st.caption("No numeric values to show.")
    else:
        cdist = raw_eda.categorical_distribution(df, dist_column)
        if cdist is not None:
            st.bar_chart(pd.DataFrame(cdist.top_values, columns=["value", "count"]).set_index("value"))
            st.caption(f"Top {len(cdist.top_values)} shown; {cdist.long_tail_distinct} more distinct value(s) account for {cdist.long_tail_count:,} row(s) in the long tail.")
        else:
            st.caption("No values to show.")

    # ---- outcome exploration ----
    st.markdown("#### Outcome exploration")
    st.caption("Pick which column is the outcome — this is never guessed.")
    outcome_choice = st.selectbox("Outcome / disposition column", [NONE_LABEL] + list(df.columns), key="raw_eda_outcome_column")
    outcome_column = None if outcome_choice == NONE_LABEL else outcome_choice
    if outcome_column:
        oc = raw_eda.explore_outcome_column(df, outcome_column)
        st.bar_chart(pd.DataFrame(oc.value_counts, columns=["value", "count"]).set_index("value"))
        if oc.looks_ordinal:
            st.info(
                "Small distinct-value count — looks ordinal. Implied funnel (by count, largest first; "
                "verify this matches the real stage order): " + " → ".join(oc.implied_funnel)
            )

    # ---- export ----
    st.divider()
    st.markdown("#### Export")
    profile = raw_eda.build_profile(df, raw_dataset_id=raw_row["raw_dataset_id"], outcome_column=outcome_column)
    report_md = raw_eda.render_markdown_report(profile)
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download profile (JSON)", data=json.dumps(profile, indent=2),
        file_name=f"{raw_row['raw_dataset_id']}_profile.json", mime="application/json",
    )
    dl2.download_button(
        "Download report (Markdown)", data=report_md,
        file_name=f"{raw_row['raw_dataset_id']}_report.md", mime="text/markdown",
    )
