"""Equifax adapter — credit bureau, alternative provider to CRIF.

fetch() is STUBBED (mock data, see base.mock_fetch). transform() is real: it
maps Equifax's -1 "no score" sentinel to None rather than letting a model read
a reason code as a credit score (CLAUDE.md — "Sentinel values masquerade as
numbers"), and returns an all-null feature dict for anything that isn't a
confirmed match, since "no bureau data" must stay a legitimate, informative
state rather than being zero-filled.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .base import CanonicalRecord, EnrichmentAdapter, RawResponse, ResponseStatus, mock_fetch

_SCORE_SENTINEL = -1
_SCORE_BANDS = ["A", "B", "C", "D", "E"]

_FEATURE_NAMES = [
    "equifax_score",
    "equifax_score_band",
    "equifax_total_accounts",
    "equifax_total_balance",
    "equifax_enquiries_last_6m",
    "equifax_delinquent_accounts",
]


class EquifaxAdapter(EnrichmentAdapter):
    source_name = "equifax"

    def __init__(
        self,
        *,
        match_rate: float = 0.40,
        error_rate: float = 0.03,
        not_sent_rate: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.match_rate = match_rate
        self.error_rate = error_rate
        self.not_sent_rate = not_sent_rate
        self.seed = seed

    def fetch(self, records: list[CanonicalRecord], *, batch_id: str) -> dict[str, RawResponse]:
        return mock_fetch(
            records,
            source_name=self.source_name,
            batch_id=batch_id,
            match_rate=self.match_rate,
            error_rate=self.error_rate,
            not_sent_rate=self.not_sent_rate,
            payload_factory=self._mock_matched_payload,
            no_hit_payload_factory=self._mock_no_hit_payload,
            seed=self.seed,
        )

    @staticmethod
    def _mock_matched_payload(rec: CanonicalRecord, rng: random.Random) -> dict[str, Any]:
        # A small fraction of real matches still come back with the -1 "no
        # score" sentinel (e.g. a thin file); the mock reproduces that on
        # purpose so transform()'s sentinel handling has something to catch.
        score = _SCORE_SENTINEL if rng.random() < 0.05 else rng.randint(300, 900)
        total_accounts = rng.randint(0, 12)
        return {
            "hit": True,
            "score": score,
            "score_band": None if score == _SCORE_SENTINEL else rng.choice(_SCORE_BANDS),
            "total_accounts": total_accounts,
            "total_balance": round(rng.uniform(0, 500_000), 2),
            "enquiries_last_6m": rng.randint(0, 8),
            "delinquent_accounts": rng.randint(0, min(3, total_accounts)),
        }

    @staticmethod
    def _mock_no_hit_payload(rec: CanonicalRecord) -> dict[str, Any]:
        # A real no-hit response often comes back with zeroed behavioural
        # fields rather than nulls — transform() must not treat these as
        # observations (see CLAUDE.md), so it discards the payload entirely
        # for any non-matched status rather than reading these zeros.
        return {"hit": False, "score": _SCORE_SENTINEL, "score_band": None, "total_accounts": 0,
                "total_balance": 0.0, "enquiries_last_6m": 0, "delinquent_accounts": 0}

    def transform(self, raw: RawResponse) -> dict[str, Any]:
        if raw.status != ResponseStatus.SUCCESS_MATCHED.value:
            return {name: None for name in _FEATURE_NAMES}
        payload = raw.payload
        score = payload.get("score")
        score = None if score in (None, _SCORE_SENTINEL) else score
        return {
            "equifax_score": score,
            "equifax_score_band": payload.get("score_band"),
            "equifax_total_accounts": payload.get("total_accounts"),
            "equifax_total_balance": payload.get("total_balance"),
            "equifax_enquiries_last_6m": payload.get("enquiries_last_6m"),
            "equifax_delinquent_accounts": payload.get("delinquent_accounts"),
        }

    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)
