"""Tab 6 — Train & Test.

UI only: every decision made here is delegated to app.core.{chain, features,
architectures, leakage, evaluate, train, registry}. This tab is skipped for
purpose="predict" datasets — there's nothing to train on a prediction batch.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.core import architectures as arch_mod
from app.core import chain as chain_mod
from app.core import features as features_mod
from app.core import leakage as leakage_mod
from app.core import registry
from app.core import schema as schema_mod
from app.core import split as split_mod
from app.core import storage
from app.core import train as train_mod

NONE_LABEL = "(none)"
CREATED_BY = "streamlit-ui"
MAX_FUNNEL_LEVELS = 20

_INT_HYPERPARAMS = {"max_depth", "num_leaves", "max_bin", "n_estimators", "min_child_samples", "min_data_per_group"}


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataframe(path: str) -> pd.DataFrame:
    return storage.load_dataframe(path)


def _dataset_schema(dataset_row: dict) -> Optional[schema_mod.DatasetSchema]:
    try:
        return schema_mod.DatasetSchema.from_dict(json.loads(dataset_row["schema_json"]))
    except (KeyError, ValueError, TypeError):
        return None


def _suggest_target_column(ds_schema: Optional[schema_mod.DatasetSchema]) -> Optional[str]:
    if ds_schema is None:
        return None
    candidates = ds_schema.columns_with_role("target")
    return candidates[0] if candidates else None


def _default_feature_columns(df: pd.DataFrame, ds_schema: Optional[schema_mod.DatasetSchema], target_column: str) -> list[str]:
    candidates = [c for c in df.columns if c != target_column]
    if ds_schema is None:
        return candidates
    excluded_roles = {"id", "target", "metadata", "excluded"}
    schema_features = {c.name for c in ds_schema.columns if c.role not in excluded_roles}
    default_on = [c for c in candidates if c in schema_features]
    return default_on or candidates


def _render_level_order_picker(distinct_values: list[str]) -> list[str]:
    """Explicit level ordering: one selectbox per position, each narrowing the
    remaining pool. Safer than relying on multiselect click-order.
    """
    level_order: list[str] = []
    remaining = list(distinct_values)
    for i in range(len(distinct_values)):
        choice = st.selectbox(f"Level {i + 1}{' (earliest)' if i == 0 else ''}{' (final)' if i == len(distinct_values) - 1 else ''}", remaining, key=f"funnel_level_{i}")
        level_order.append(choice)
        remaining = [v for v in remaining if v != choice]
    return level_order


def _render_hyperparams(architecture: arch_mod.ArchitectureName) -> dict:
    defaults = arch_mod.default_hyperparams(architecture)
    edited: dict = {}
    cols = st.columns(3)
    for i, (key, default_value) in enumerate(defaults.items()):
        with cols[i % 3]:
            if key in _INT_HYPERPARAMS:
                edited[key] = st.number_input(key, value=int(default_value), step=1, key=f"hp_{architecture}_{key}")
            else:
                edited[key] = st.number_input(key, value=float(default_value), step=0.01, format="%.4f", key=f"hp_{architecture}_{key}")
    return edited


def _gate_icon(passed: bool) -> str:
    return "🟢" if passed else "🔴"


def _render_gate_report(pre_flight: list[leakage_mod.GateResult], gate2_by_stage: dict[str, leakage_mod.GateResult]) -> None:
    rows = [{"gate": g.gate, "status": _gate_icon(g.passed), "message": g.message} for g in pre_flight]
    for stage, g in gate2_by_stage.items():
        rows.append({"gate": f"implausible_performance ({stage})", "status": _gate_icon(g.passed), "message": g.message})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render(store: storage.MetadataStore, merchant: str, storage_root: Path | str = storage.DEFAULT_STORAGE_ROOT) -> None:
    st.subheader("Train & Test")

    all_datasets = store.list_datasets(merchant=merchant)
    train_datasets = [d for d in all_datasets if d["purpose"] == "train"]
    if not train_datasets:
        st.info(
            "No training-purpose datasets registered for this merchant yet. This tab only "
            "applies to purpose='train' datasets — prediction batches aren't trained on."
        )
        return

    dataset_options = {
        f"{d['dataset_id']}  ·  {d['source_file']}  ·  {d['row_count']} rows  ·  {d['created_at']}": d
        for d in train_datasets
    }
    dataset_labels = list(dataset_options.keys())
    dataset_default_idx = next(
        (i for i, l in enumerate(dataset_labels) if dataset_options[l]["dataset_id"] == st.session_state.get("dataset_id")), 0
    )
    dataset_row = dataset_options[
        st.selectbox("Dataset", dataset_labels, index=dataset_default_idx, key="training_dataset_select")
    ]
    st.session_state["dataset_id"] = dataset_row["dataset_id"]
    if dataset_row["purpose"] == "predict":
        st.info("Training is skipped for prediction-purpose datasets.")
        return

    splits = store.list_splits(dataset_id=dataset_row["dataset_id"])
    if not splits:
        st.info("No splits saved for this dataset yet. Create one in the Data Prep tab first.")
        return
    split_options = {
        f"{s['split_id']}  ·  {s['strategy']}  ·  {s['train_rows']} train / {s['test_rows']} test  ·  {s['created_at']}": s
        for s in splits
    }
    split_labels = list(split_options.keys())
    split_default_idx = next(
        (i for i, l in enumerate(split_labels) if split_options[l]["split_id"] == st.session_state.get("split_id")), 0
    )
    split_row = split_options[st.selectbox("Split", split_labels, index=split_default_idx, key="training_split_select")]
    st.session_state["split_id"] = split_row["split_id"]

    df = _load_dataframe(dataset_row["artifact_path"])
    ds_schema = _dataset_schema(dataset_row)
    train_ids, test_ids = split_mod.load_split_ids(split_row)
    row_id_column = json.loads(split_row["config_json"]).get("row_id_column")
    try:
        train_df, test_df = split_mod.apply_split(df, row_id_column, train_ids, test_ids)
    except ValueError as exc:
        st.error(f"Could not apply this split to the current dataset: {exc}")
        return

    st.caption(f"{len(train_df)} train rows, {len(test_df)} test rows")
    st.divider()

    # ---- funnel configuration ----
    st.markdown("#### Funnel stages")
    columns_list = list(df.columns)
    suggested_target = _suggest_target_column(ds_schema)
    target_index = columns_list.index(suggested_target) if suggested_target in columns_list else 0
    target_column = st.selectbox("Target / disposition column", columns_list, index=target_index)
    distinct_values = sorted(df[target_column].dropna().astype(str).unique().tolist())
    if len(distinct_values) < 2:
        st.error(f"{target_column!r} has fewer than 2 distinct values; a funnel needs at least 2 outcome levels.")
        return
    if len(distinct_values) > MAX_FUNNEL_LEVELS:
        st.error(
            f"{target_column!r} has {len(distinct_values)} distinct values — a funnel/disposition column should "
            f"have a small number of ordinal levels (at most {MAX_FUNNEL_LEVELS}). Pick a different column."
        )
        return
    st.caption(f"{len(distinct_values)} distinct value(s) found — order them from earliest to final:")
    level_order = _render_level_order_picker(distinct_values)

    custom_names_raw = st.text_input(
        "Stage names (comma-separated, leave blank for defaults)", placeholder="reached_appointment, reached_consulted, converted"
    )
    stage_names = [s.strip() for s in custom_names_raw.split(",") if s.strip()] or None

    try:
        funnel = chain_mod.build_funnel_config(train_df, target_column=target_column, level_order=level_order, stage_names=stage_names)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.info(chain_mod.describe_funnel_config(funnel))

    st.divider()

    # ---- feature checklist ----
    st.markdown("#### Features")
    candidate_columns = [c for c in df.columns if c != target_column]
    default_features = _default_feature_columns(df, ds_schema, target_column)
    selected_features = st.multiselect("Feature columns", candidate_columns, default=default_features)
    if not selected_features:
        st.warning("Select at least one feature to continue.")
        return

    inferred_types = features_mod.classify_numeric_categorical(train_df, selected_features)
    inferred_sources = features_mod.classify_feature_sources(selected_features)
    editor_df = pd.DataFrame(
        {
            "feature": selected_features,
            "type": [("numeric" if f in inferred_types.numeric else "categorical") for f in selected_features],
            "source": [inferred_sources[f] for f in selected_features],
        }
    )
    with st.expander("Override feature typing (numeric/categorical, core/enrichment)", expanded=False):
        edited = st.data_editor(
            editor_df,
            column_config={
                "feature": st.column_config.TextColumn(disabled=True),
                "type": st.column_config.SelectboxColumn("Type", options=["numeric", "categorical"]),
                "source": st.column_config.SelectboxColumn("Source", options=["core", "enrichment"]),
            },
            hide_index=True,
            width="stretch",
            key="feature_type_editor",
        )
    dtype_overrides = dict(zip(edited["feature"], edited["type"]))
    source_overrides = dict(zip(edited["feature"], edited["source"]))

    feature_types = features_mod.classify_numeric_categorical(train_df, selected_features, overrides=dtype_overrides)
    feature_sources = features_mod.classify_feature_sources(selected_features, overrides=source_overrides)

    st.divider()

    # ---- architectures + hyperparameters ----
    st.markdown("#### Architectures")
    selected_architectures = st.multiselect(
        "Architectures to run", arch_mod.ARCHITECTURES, default=list(arch_mod.ARCHITECTURES),
        format_func=lambda a: arch_mod.ARCHITECTURE_LABELS[a],
    )
    if not selected_architectures:
        st.warning("Select at least one architecture to continue.")
        return

    hyperparams_by_arch: dict[str, dict] = {}
    blend_weight = arch_mod.DEFAULT_BLEND_WEIGHT
    for architecture in selected_architectures:
        with st.expander(f"Hyperparameters — {arch_mod.ARCHITECTURE_LABELS[architecture]}", expanded=False):
            hyperparams_by_arch[architecture] = _render_hyperparams(architecture)
            if architecture == "dual_route_cascade":
                blend_weight = st.slider(
                    "Blend weight (share on route A, enriched rows only)", 0.0, 1.0, arch_mod.DEFAULT_BLEND_WEIGHT, 0.05
                )

    st.divider()

    # ---- leakage gate configuration ----
    with st.expander("Leakage gate configuration", expanded=False):
        provenance_options = [NONE_LABEL] + candidate_columns
        provenance_choice = st.selectbox(
            "Provenance / batch / vintage column (GATE 3)", provenance_options,
            help="If set, a quick model checks whether the feature set can predict this column — a sign features encode data provenance rather than a real signal.",
        )
        provenance_column = None if provenance_choice == NONE_LABEL else provenance_choice

        st.caption("Formula consistency (GATE 5) — optional, checked per source below")
        source_choice = st.selectbox("Source column for per-source breakdown", [NONE_LABEL] + candidate_columns)
        source_column = None if source_choice == NONE_LABEL else source_choice

        formula_invariants = []
        sum_total = st.selectbox("Sum invariant: total column", [NONE_LABEL] + candidate_columns, key="sum_total")
        sum_parts = st.multiselect("...must equal the sum of these part columns", candidate_columns, key="sum_parts")
        if sum_total != NONE_LABEL and sum_parts:
            formula_invariants.append(leakage_mod.make_sum_invariant(f"{sum_total}_eq_sum", sum_total, sum_parts))

        le_lesser = st.selectbox("Order invariant: this column...", [NONE_LABEL] + candidate_columns, key="le_lesser")
        le_greater = st.selectbox("...must be <= this column", [NONE_LABEL] + candidate_columns, key="le_greater")
        if le_lesser != NONE_LABEL and le_greater != NONE_LABEL and le_lesser != le_greater:
            formula_invariants.append(leakage_mod.make_le_invariant(f"{le_lesser}_le_{le_greater}", le_lesser, le_greater))

        allow_override = st.checkbox(
            "Allow implausible performance (ROC-AUC > 0.90 on a rare event) — I have specifically ruled out leakage",
            value=False,
        )

    st.divider()

    session_key = f"training_results::{merchant}::{dataset_row['dataset_id']}::{split_row['split_id']}"

    if st.button("Run training", type="primary"):
        run_results: dict[str, dict] = {}
        n_total = len(selected_architectures)
        progress_bar = st.progress(0.0, text=f"0/{n_total} architecture(s) complete")
        for idx, architecture in enumerate(selected_architectures):
            label = arch_mod.ARCHITECTURE_LABELS[architecture]
            with st.status(f"Training {label}...", expanded=True) as status:
                try:
                    result = train_mod.train_architecture(
                        train_df,
                        funnel=funnel,
                        numeric_features=feature_types.numeric,
                        categorical_features=feature_types.categorical,
                        feature_sources=feature_sources,
                        architecture=architecture,
                        hyperparams=hyperparams_by_arch[architecture],
                        blend_weight=blend_weight,
                        provenance_column=provenance_column,
                        formula_invariants=formula_invariants or None,
                        source_column=source_column,
                    )
                    for outcome in result.stage_outcomes:
                        st.write(f"stage {outcome.name!r}: {outcome.n_rows} rows, {outcome.n_positive} positive" + (" — fell back to route B alone" if outcome.route_b_only else ""))

                    evaluation = train_mod.evaluate_run(result, test_df, funnel, allow_implausible_override=allow_override)
                    run_results[architecture] = {"result": result, "evaluation": evaluation, "error": None, "traceback": None}
                    status.update(label=f"{label} — done", state="complete")
                except leakage_mod.LeakageGateFailure as exc:
                    run_results[architecture] = {
                        "result": None, "evaluation": None, "error": str(exc), "traceback": traceback.format_exc(),
                    }
                    status.update(label=f"{label} — blocked by a leakage gate", state="error")
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001 - never swallow; every failure is surfaced, run continues to the next architecture
                    message = f"{type(exc).__name__}: {exc}"
                    run_results[architecture] = {
                        "result": None, "evaluation": None, "error": message, "traceback": traceback.format_exc(),
                    }
                    status.update(label=f"{label} — failed unexpectedly", state="error")
                    st.error(message)
            progress_bar.progress((idx + 1) / n_total, text=f"{idx + 1}/{n_total} architecture(s) complete")

        st.session_state[session_key] = run_results

    run_results = st.session_state.get(session_key)
    if not run_results:
        return

    st.divider()
    st.markdown("### Results")

    comparison_rows = []
    for architecture, payload in run_results.items():
        if payload["error"] is not None:
            continue
        for se in payload["evaluation"].stage_evaluations:
            comparison_rows.append(
                {
                    "architecture": arch_mod.ARCHITECTURE_LABELS[architecture],
                    "stage": se.stage,
                    "n_rows": se.metrics.n_rows,
                    "base_rate": se.metrics.base_rate,
                    "roc_auc": se.metrics.roc_auc,
                    "pr_auc": se.metrics.pr_auc,
                    "log_loss": se.metrics.log_loss,
                    "brier": se.metrics.brier,
                    "lift_vs_base_rate": se.lift.lift_vs_base_rate,
                }
            )
    if comparison_rows:
        st.markdown("#### Benchmark comparison")
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    for architecture, payload in run_results.items():
        label = arch_mod.ARCHITECTURE_LABELS[architecture]
        with st.expander(label, expanded=payload["error"] is not None):
            if payload["error"] is not None:
                st.error(payload["error"])
                if payload.get("traceback"):
                    with st.expander("Full traceback"):
                        st.code(payload["traceback"], language="python")
                continue

            result: train_mod.TrainRunResult = payload["result"]
            evaluation: train_mod.EvaluationResult = payload["evaluation"]

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

            st.markdown("**Leakage gates**")
            gate2_by_stage = {se.stage: se.gate2 for se in evaluation.stage_evaluations}
            _render_gate_report(result.pre_flight_gates, gate2_by_stage)

            stage_choice = st.selectbox("Stage", funnel.stage_names, key=f"stage_choice_{architecture}")
            se = next(s for s in evaluation.stage_evaluations if s.stage == stage_choice)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROC-AUC", f"{se.metrics.roc_auc:.4f}")
            m2.metric("PR-AUC", f"{se.metrics.pr_auc:.4f}")
            m3.metric("Log Loss", f"{se.metrics.log_loss:.4f}")
            m4.metric("Brier", f"{se.metrics.brier:.4f}")

            st.markdown("**Lift**")
            lc1, lc2 = st.columns(2)
            lc1.metric("Lift vs base rate", f"{se.lift.lift_vs_base_rate:.2f}x")
            ratio_display = (
                f"{se.lift.top_bottom_ratio:.2f}x" if isinstance(se.lift.top_bottom_ratio, float) else se.lift.top_bottom_ratio
            )
            lc2.metric("Top/bottom ratio", ratio_display)
            if se.lift.top_bottom_ratio_ci:
                lo, hi = se.lift.top_bottom_ratio_ci
                st.caption(f"bottom tier has few positives — bootstrap 95% CI on the ratio: [{lo:.2f}, {hi:.2f}]")

            st.markdown("**Tier table**")
            st.dataframe(se.tier_table, width="stretch", hide_index=True)

            st.markdown("**Capture thresholds**")
            capture_df = pd.DataFrame(
                [{"target_capture": f"{t:.0%}", "pct_of_population_needed": v} for t, v in se.capture_thresholds.items()]
            )
            st.dataframe(capture_df, width="stretch", hide_index=True)

            st.markdown("**Feature gain**")
            outcome = next(o for o in result.stage_outcomes if o.name == stage_choice)
            st.dataframe(outcome.feature_gain, width="stretch", hide_index=True)

            if st.button(f"Save bundle — {label}", key=f"save_{architecture}"):
                metrics_payload = {
                    "evaluation": train_mod.serialize_evaluation(evaluation),
                    "training": train_mod.serialize_stage_outcomes(result),
                    "tier_cutoffs": train_mod.serialize_tier_cutoffs(evaluation),
                }
                bundle_id = registry.new_bundle_id()
                bundle_path = registry.save_bundle(
                    root=storage_root,
                    merchant=merchant,
                    bundle_id=bundle_id,
                    architecture=architecture,
                    stages=result.stage_inputs,
                    preprocessing=result.preprocessing,
                    dataset_content_hash=dataset_row["content_hash"],
                    split_id=split_row["split_id"],
                    target_config={
                        "target_column": funnel.target_column,
                        "level_order": funnel.level_order,
                        "stage_names": funnel.stage_names,
                    },
                    hyperparams=result.hyperparams,
                    metrics=metrics_payload,
                )
                store.insert_model_bundle(
                    bundle_id=bundle_id,
                    merchant=merchant,
                    dataset_id=dataset_row["dataset_id"],
                    split_id=split_row["split_id"],
                    architecture=architecture,
                    target_config_json=json.dumps(
                        {"target_column": funnel.target_column, "level_order": funnel.level_order, "stage_names": funnel.stage_names}
                    ),
                    feature_list_json=json.dumps(selected_features),
                    hyperparams_json=json.dumps(result.hyperparams),
                    metrics_json=json.dumps(metrics_payload),
                    bundle_path=str(bundle_path),
                    created_by=CREATED_BY,
                )
                st.session_state["bundle_id"] = bundle_id
                st.success(f"Saved bundle_id = {bundle_id}")
