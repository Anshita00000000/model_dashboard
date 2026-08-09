"""Enrichment orchestration.

Runs each selected adapter over a canonical dataset's leads, persisting
EVERY raw response — never just the successful ones — with its status,
fetched_at and batch_id (see base.py's module docstring for why that's
non-negotiable). Idempotent by construction: each invocation of run_source()
mints a fresh batch_id and inserts a new enrichment_runs row; it never
updates or reuses a prior run.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .. import schema, storage
from .base import (
    CanonicalRecord,
    EnrichmentAdapter,
    RawResponse,
    RETRYABLE_STATUSES,
    ResponseStatus,
    new_batch_id,
    utcnow_iso,
)


def build_canonical_records(
    df: pd.DataFrame, *, lead_id_col: str = "lead_id", phone_col: str = "phone",
    extra_cols: Optional[list[str]] = None,
) -> list[CanonicalRecord]:
    """One CanonicalRecord per row. `phone` is expected to already be the
    normalised join key (see app.core.canonical.normalize_phone) — this
    function doesn't normalise anything itself, only reads what's there.
    """
    cols = extra_cols if extra_cols is not None else [c for c in df.columns if c not in (lead_id_col, phone_col)]
    records = []
    for _, row in df.iterrows():
        raw_phone = row.get(phone_col)
        records.append(CanonicalRecord(
            lead_id=str(row[lead_id_col]),
            phone=None if schema.is_blank(raw_phone) else str(raw_phone),
            fields={c: row[c] for c in cols if c in df.columns},
        ))
    return records


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    source: str
    batch_id: str
    rows_attempted: int
    matched: int
    no_hit: int
    error: int
    not_sent: int
    started_at: str
    finished_at: str


def run_source(
    adapter: EnrichmentAdapter,
    records: list[CanonicalRecord],
    *,
    store: storage.MetadataStore,
    canonical_dataset_id: str,
    max_retries: int = 3,
    backoff_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Fetch, retrying ERROR responses (with exponential backoff) up to
    max_retries times; NOT_SENT and SUCCESS_NO_HIT are terminal and never
    retried. Persists every final response, then records the run.
    """
    started_at = utcnow_iso()
    batch_id = new_batch_id()
    run_id = uuid.uuid4().hex

    final: dict[str, RawResponse] = {}
    pending = list(records)
    attempt = 0
    while pending and attempt <= max_retries:
        result = adapter.fetch(pending, batch_id=batch_id)
        retry_leads = []
        for rec in pending:
            resp = result.get(rec.lead_id)
            if resp is None:
                continue
            final[rec.lead_id] = resp
            if resp.status in RETRYABLE_STATUSES:
                retry_leads.append(rec)
        pending = retry_leads
        attempt += 1
        if pending and attempt <= max_retries and backoff_seconds:
            sleep_fn(backoff_seconds * (2 ** (attempt - 1)))

    counts = Counter(r.status for r in final.values())
    finished_at = utcnow_iso()
    matched = counts.get(ResponseStatus.SUCCESS_MATCHED.value, 0)
    no_hit = counts.get(ResponseStatus.SUCCESS_NO_HIT.value, 0)
    error = counts.get(ResponseStatus.ERROR.value, 0)
    not_sent = counts.get(ResponseStatus.NOT_SENT.value, 0)

    # The run row must exist before responses referencing it via FK.
    store.insert_enrichment_run(
        run_id=run_id, canonical_dataset_id=canonical_dataset_id, source=adapter.source_name, batch_id=batch_id,
        rows_attempted=len(records), matched=matched, no_hit=no_hit, error=error, not_sent=not_sent,
        started_at=started_at, finished_at=finished_at,
    )
    for lead_id, resp in final.items():
        store.insert_enrichment_response(
            response_id=uuid.uuid4().hex, run_id=run_id, lead_id=lead_id, source=resp.source,
            status=resp.status, payload_json=json.dumps(resp.payload), fetched_at=resp.fetched_at,
            batch_id=resp.batch_id,
        )

    return RunSummary(
        run_id=run_id, source=adapter.source_name, batch_id=batch_id, rows_attempted=len(records),
        matched=matched, no_hit=no_hit, error=error, not_sent=not_sent,
        started_at=started_at, finished_at=finished_at,
    )


def run_enrichment(
    *,
    store: storage.MetadataStore,
    canonical_dataset_id: str,
    records: list[CanonicalRecord],
    adapters: list[EnrichmentAdapter],
    max_retries: int = 3,
    backoff_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress_callback: Optional[Callable[[str, RunSummary], None]] = None,
) -> list[RunSummary]:
    """Runs every selected source in turn, reporting progress as each source
    finishes. A prior run for the same source is untouched — re-running always
    creates a new batch (see run_source).
    """
    summaries = []
    for adapter in adapters:
        summary = run_source(
            adapter, records, store=store, canonical_dataset_id=canonical_dataset_id,
            max_retries=max_retries, backoff_seconds=backoff_seconds, sleep_fn=sleep_fn,
        )
        summaries.append(summary)
        if progress_callback:
            progress_callback(adapter.source_name, summary)
    return summaries
