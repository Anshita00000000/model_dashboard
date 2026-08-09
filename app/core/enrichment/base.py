"""The enrichment adapter interface shared by every source.

The fetch/transform split is deliberate: fetch() is network-bound and STUBBED
for now (mock data only — see the mock_fetch() helper below, used by every
stub adapter). transform() is a pure, deterministic function with no network
call, so it can be unit-tested against the exact mock payload shapes without
a live API.

STORE THE PULL TIMESTAMP AND BATCH — non-negotiable, see CLAUDE.md. A pull
that happens after the outcome leaks: one merchant's converters were enriched
at loan origination while non-converters were enriched at lead capture, and
the model learned to identify which pull a row came from (provenance
detection scored AUC 0.99). Every RawResponse carries fetched_at and
batch_id so this is diagnosable, and check_enrichment_timing_leakage() below
catches the direct case automatically instead of leaving it to be found by
accident.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_batch_id() -> str:
    return uuid.uuid4().hex


class ResponseStatus(str, Enum):
    SUCCESS_MATCHED = "SUCCESS_MATCHED"
    SUCCESS_NO_HIT = "SUCCESS_NO_HIT"
    ERROR = "ERROR"
    NOT_SENT = "NOT_SENT"


TERMINAL_STATUSES = (ResponseStatus.SUCCESS_MATCHED.value, ResponseStatus.SUCCESS_NO_HIT.value, ResponseStatus.NOT_SENT.value)
RETRYABLE_STATUSES = (ResponseStatus.ERROR.value,)


@dataclass(frozen=True)
class CanonicalRecord:
    """The minimal per-lead input every adapter needs: an id to key the
    response, a normalised phone to query with (None means the adapter can't
    be called — see mock_fetch()'s NOT_SENT handling), and whatever other
    canonical fields a real adapter might key off (e.g. name for a fuzzy
    bureau match). Extra canonical/`extra__` columns are passed through in
    `fields` without the adapter needing to know the full schema.
    """

    lead_id: str
    phone: Optional[str]
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawResponse:
    lead_id: str
    source: str
    status: str  # a ResponseStatus value
    payload: dict[str, Any]
    fetched_at: str  # UTC ISO-8601
    batch_id: str


class EnrichmentAdapter(ABC):
    """Every source implements this same two-part interface."""

    source_name: str

    @abstractmethod
    def fetch(self, records: list[CanonicalRecord], *, batch_id: str) -> dict[str, RawResponse]:
        """Network call. Returns lead_id -> raw payload. STUBBED for now."""

    @abstractmethod
    def transform(self, raw: RawResponse) -> dict[str, Any]:
        """Pure function: raw payload -> flat feature dict. NO network."""

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Stable ordered list of features this adapter produces."""


# ---------------------------------------------------------------------------
# Shared mock-fetch helper — used by every stub adapter in this package so
# the status mix and match-rate configuration only lives in one place.
# ---------------------------------------------------------------------------


def mock_fetch(
    records: list[CanonicalRecord],
    *,
    source_name: str,
    batch_id: str,
    match_rate: float = 0.40,
    error_rate: float = 0.03,
    not_sent_rate: float = 0.02,
    payload_factory: Callable[[CanonicalRecord, random.Random], dict[str, Any]],
    no_hit_payload_factory: Optional[Callable[[CanonicalRecord], dict[str, Any]]] = None,
    seed: Optional[int] = None,
) -> dict[str, RawResponse]:
    """Mock network call: draws a realistic status per record and builds a
    structurally-real payload for SUCCESS_MATCHED (via payload_factory) so
    transform() can be exercised against it. A record with no phone number is
    always NOT_SENT — a real API call genuinely can't be made without one.
    """
    if not (0.0 <= match_rate + error_rate + not_sent_rate <= 1.0):
        raise ValueError("match_rate + error_rate + not_sent_rate must be within [0, 1]")
    no_hit_rate = 1.0 - match_rate - error_rate - not_sent_rate
    rng = random.Random(seed)
    fetched_at = utcnow_iso()
    responses: dict[str, RawResponse] = {}

    for rec in records:
        if rec.phone is None:
            status = ResponseStatus.NOT_SENT.value
        else:
            draw = rng.random()
            if draw < match_rate:
                status = ResponseStatus.SUCCESS_MATCHED.value
            elif draw < match_rate + error_rate:
                status = ResponseStatus.ERROR.value
            elif draw < match_rate + error_rate + not_sent_rate:
                status = ResponseStatus.NOT_SENT.value
            else:
                status = ResponseStatus.SUCCESS_NO_HIT.value

        if status == ResponseStatus.SUCCESS_MATCHED.value:
            payload = payload_factory(rec, rng)
        elif status == ResponseStatus.SUCCESS_NO_HIT.value:
            payload = no_hit_payload_factory(rec) if no_hit_payload_factory else {"hit": False}
        else:
            payload = {}

        responses[rec.lead_id] = RawResponse(
            lead_id=rec.lead_id, source=source_name, status=status, payload=payload,
            fetched_at=fetched_at, batch_id=batch_id,
        )

    _ = no_hit_rate  # documents the implied share; not otherwise consumed
    return responses


# ---------------------------------------------------------------------------
# Timing-leakage check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingLeakWarning:
    lead_id: str
    source: str
    fetched_at: str
    outcome_at: str
    message: str


def check_enrichment_timing_leakage(
    *, outcome_dates: dict[str, str], responses: list[RawResponse]
) -> list[TimingLeakWarning]:
    """Warns when a lead's enrichment fetched_at is later than its
    disposition/outcome date — a direct leak (the model would be seeing
    information from after the thing it's trying to predict).
    outcome_dates: {lead_id: outcome_date_string}. Leads without a known
    outcome date, or timestamps that fail to parse, are silently skipped —
    this is a best-effort check, not a validator that blocks on parse issues.
    """
    warnings: list[TimingLeakWarning] = []
    for r in responses:
        outcome_at = outcome_dates.get(r.lead_id)
        if not outcome_at:
            continue
        fetched_dt = pd.to_datetime(r.fetched_at, errors="coerce")
        outcome_dt = pd.to_datetime(outcome_at, errors="coerce")
        if pd.isna(fetched_dt) or pd.isna(outcome_dt):
            continue
        if fetched_dt.tzinfo is not None:
            fetched_dt = fetched_dt.tz_localize(None)
        if outcome_dt.tzinfo is not None:
            outcome_dt = outcome_dt.tz_localize(None)
        if fetched_dt > outcome_dt:
            warnings.append(TimingLeakWarning(
                lead_id=r.lead_id, source=r.source, fetched_at=r.fetched_at, outcome_at=outcome_at,
                message=(
                    f"{r.source} pulled for lead {r.lead_id} at {r.fetched_at}, AFTER its outcome date "
                    f"{outcome_at} — this is a direct leak."
                ),
            ))
    return warnings
