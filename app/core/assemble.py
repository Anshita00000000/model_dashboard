"""Assemble the modelling dataset — the final Phase 2 stage.

CANONICAL DATA IS THE SPINE. Every enrichment source is left-joined onto it,
never the other way round: a lead with no bureau match stays in the dataset
with nulls, it is not dropped, and enrichment never controls row count (see
CLAUDE.md — enrichment coverage is only 35-43%, so most leads have no match
at all, and that is a legitimate, informative state).

Each source's transform() already prefixes its own feature keys (e.g.
"crif_score", "equifax_total_accounts" — see app/core/enrichment/*.py), so
this module doesn't re-prefix anything; it only adds the three provenance
columns per source that transform() doesn't produce:
{source}_match_status, {source}_fetched_at, {source}_batch_id, plus an
is_enriched flag computed from actual data presence (never trusted from the
status flag alone) — the leakage incident in CLAUDE.md (provenance
detection scoring AUC 0.99) is exactly why every one of these columns exists
and is carried through rather than discarded after use.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import schema as schema_mod
from . import storage
from .enrichment import ADAPTER_REGISTRY
from .enrichment.base import RawResponse

LEAD_ID_COLUMN = "lead_id"
MATCHED_STATUS = "SUCCESS_MATCHED"


@dataclass(frozen=True)
class SourceAssembly:
    source: str
    run_id: str
    batch_id: str
    feature_columns: list[str]  # this source's transform()-produced columns, already prefixed
    provenance_columns: list[str]  # {source}_match_status/_fetched_at/_batch_id/_is_enriched
    responses_in_run: int


@dataclass(frozen=True)
class AssembleResult:
    df: pd.DataFrame
    row_count: int
    canonical_dataset_id: str
    sources: list[SourceAssembly]

    def enrichment_feature_columns(self) -> list[str]:
        return [c for s in self.sources for c in s.feature_columns]

    def provenance_columns(self) -> list[str]:
        """{source}_match_status/_fetched_at/_batch_id only — diagnostic
        columns that must never be trained on (see default_role_overrides).
        is_enriched is deliberately NOT included here: unlike the other
        three, it's a legitimate feature (CLAUDE.md — "no bureau data" is
        informative signal, not noise), not raw provenance.
        """
        return [c for s in self.sources for c in s.provenance_columns if not c.endswith("_is_enriched")]

    def is_enriched_columns(self) -> list[str]:
        return [f"{s.source}_is_enriched" for s in self.sources]


def _latest_run_per_source(runs: list[dict]) -> dict[str, dict]:
    """`runs` is already ordered started_at DESC by MetadataStore.list_enrichment_runs,
    so the first occurrence of each source is its most recent run/batch — a
    source can be re-run (idempotent, new batch each time; see
    app/core/enrichment/runner.py), and assembly always uses the latest one.
    """
    latest: dict[str, dict] = {}
    for run in runs:
        latest.setdefault(run["source"], run)
    return latest


def _source_frame(store: storage.MetadataStore, run: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per lead_id for this source's most recent run: transform()ed
    features (already prefixed by the adapter) plus the four provenance
    columns. Returns (frame, feature_columns, provenance_columns).
    """
    source = run["source"]
    adapter_cls = ADAPTER_REGISTRY.get(source)
    if adapter_cls is None:
        raise ValueError(f"no adapter registered for enrichment source {source!r}")
    adapter = adapter_cls()
    feature_names = adapter.feature_names()

    provenance_cols = [f"{source}_match_status", f"{source}_fetched_at", f"{source}_batch_id", f"{source}_is_enriched"]

    responses = store.list_enrichment_responses(run_id=run["run_id"])
    seen_lead_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in responses:
        lead_id = row["lead_id"]
        if lead_id in seen_lead_ids:
            raise ValueError(
                f"duplicate lead_id {lead_id!r} within a single enrichment run "
                f"(run_id={run['run_id']!r}, source={source!r}) — a run must produce at most one response per lead"
            )
        seen_lead_ids.add(lead_id)

        raw = RawResponse(
            lead_id=lead_id, source=row["source"], status=row["status"],
            payload=json.loads(row["payload_json"]), fetched_at=row["fetched_at"], batch_id=row["batch_id"],
        )
        features = adapter.transform(raw)

        # "Compute is_enriched from ACTUAL data presence, per source, never
        # from a provided flag": check the transformed values themselves.
        # Sources with no implemented transform() yet (feature_names() == [])
        # have no data to check presence of, so fall back to the match
        # status — there's genuinely nothing else to verify against there.
        if feature_names:
            is_enriched = any(v is not None for v in features.values())
        else:
            is_enriched = raw.status == MATCHED_STATUS

        record: dict[str, Any] = {LEAD_ID_COLUMN: lead_id, **features}
        record[f"{source}_match_status"] = raw.status
        record[f"{source}_fetched_at"] = raw.fetched_at
        record[f"{source}_batch_id"] = raw.batch_id
        record[f"{source}_is_enriched"] = is_enriched
        rows.append(record)

    columns = [LEAD_ID_COLUMN, *feature_names, *provenance_cols]
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    return frame, list(feature_names), provenance_cols


def assemble_dataset(*, store: storage.MetadataStore, canonical_dataset_id: str) -> AssembleResult:
    """Left-joins every enrichment source (its latest run) onto the canonical
    dataset. Raises AssertionError if any join changes the row count —
    that would mean enrichment silently controlled which leads survive,
    which must never happen (see module docstring).
    """
    canonical_row = store.get_canonical_dataset(canonical_dataset_id)
    if canonical_row is None:
        raise ValueError(f"no canonical_dataset with id {canonical_dataset_id!r}")

    canonical_df = storage.load_dataframe(canonical_row["artifact_path"])
    if LEAD_ID_COLUMN not in canonical_df.columns:
        raise ValueError(
            f"canonical dataset {canonical_dataset_id!r} has no {LEAD_ID_COLUMN!r} column to join enrichment on"
        )

    spine_row_count = len(canonical_df)
    assembled = canonical_df.copy()

    runs = store.list_enrichment_runs(canonical_dataset_id=canonical_dataset_id)
    latest_by_source = _latest_run_per_source(runs)

    sources: list[SourceAssembly] = []
    for source, run in latest_by_source.items():
        source_frame, feature_cols, provenance_cols = _source_frame(store, run)

        before = len(assembled)
        assembled = assembled.merge(source_frame, on=LEAD_ID_COLUMN, how="left")
        if len(assembled) != before:
            raise AssertionError(
                f"row count changed after joining source {source!r}: {before} -> {len(assembled)}. "
                f"Enrichment must never control row count — this indicates a duplicate lead_id in that run's responses."
            )
        if len(assembled) != spine_row_count:
            raise AssertionError(
                f"row count drifted from the canonical spine after joining source {source!r}: "
                f"{spine_row_count} -> {len(assembled)}"
            )

        sources.append(SourceAssembly(
            source=source, run_id=run["run_id"], batch_id=run["batch_id"],
            feature_columns=feature_cols, provenance_columns=provenance_cols,
            responses_in_run=len(source_frame),
        ))

    if len(assembled) != spine_row_count:
        raise AssertionError(f"final assembled row count {len(assembled)} != canonical row count {spine_row_count}")

    return AssembleResult(df=assembled, row_count=len(assembled), canonical_dataset_id=canonical_dataset_id, sources=sources)


# ---------------------------------------------------------------------------
# Export — register the assembled (and, typically, derived-feature-augmented)
# dataframe as a Phase-1-compatible `datasets` table artifact.
# ---------------------------------------------------------------------------


def default_role_overrides(
    result: AssembleResult,
    *,
    id_column: str = LEAD_ID_COLUMN,
    target_column: Optional[str] = None,
    metadata_columns: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Builds the `overrides` dict for schema.infer_schema(): role=id for the
    lead identifier, role=target for the caller-chosen outcome column,
    role=metadata for anything explicitly flagged as descriptive-not-feature,
    and role=excluded for every provenance column — match_status, fetched_at
    and batch_id are diagnostic (Phase 1's leakage Gate 3 reads them) but
    must never be trained on. Everything else defaults to role=feature.
    """
    overrides: dict[str, dict[str, Any]] = {id_column: {"role": "id"}}
    if target_column:
        overrides[target_column] = {"role": "target"}
    for col in metadata_columns:
        overrides[col] = {"role": "metadata"}
    for col in result.provenance_columns():
        overrides[col] = {"role": "excluded"}
    return overrides


def register_dataset(
    *,
    store: storage.MetadataStore,
    root: Path | str,
    df: pd.DataFrame,
    merchant: str,
    purpose: str,
    source_file: str,
    role_overrides: dict[str, dict[str, Any]],
) -> dict:
    """Persists df as a new `datasets` table row — the Phase-1-compatible
    artifact Tab 5 onward consumes. Always a new row (append-only); never
    overwrites a prior export.
    """
    dataset_id = uuid.uuid4().hex
    artifact_path = storage.save_dataframe(df, "dataset", merchant, dataset_id, root=root)
    ds_schema = schema_mod.infer_schema(
        df, dataset_id=dataset_id, merchant_name=merchant, purpose=purpose,
        source_file=source_file, overrides=role_overrides,
    )
    store.insert_dataset(
        dataset_id=dataset_id, merchant=merchant, purpose=purpose, source_file=source_file,
        row_count=ds_schema.row_count, col_count=ds_schema.column_count, content_hash=ds_schema.content_hash,
        schema_json=json.dumps(ds_schema.to_dict()), artifact_path=str(artifact_path),
    )
    return store.get_dataset(dataset_id)
