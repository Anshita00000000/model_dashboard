"""CRIF adapter — credit bureau (primary provider).

fetch() is STUBBED (mock data, see base.mock_fetch). transform() is real: it
maps CRIF's 0 and 10-18 "insufficient data" reason codes to None — real CRIF
scores start around 300, so left unmapped a reason code reads as a credit
score (CLAUDE.md) — and parses the text-encoded credit-history duration
("6yrs 3mon") into numeric months rather than leaving it as free text.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from .base import CanonicalRecord, EnrichmentAdapter, RawResponse, ResponseStatus, mock_fetch

_SENTINEL_SCORES = {0} | set(range(10, 19))  # 0, and the 10-18 reason-code band

_FEATURE_NAMES = [
    "crif_score",
    "crif_total_accounts",
    "crif_active_accounts",
    "crif_overdue_accounts",
    "crif_total_outstanding",
    "crif_credit_history_months",
]

_DURATION_RE = re.compile(r"^\s*(\d+)\s*yrs?\s*(\d+)\s*mon\s*$", re.IGNORECASE)


def _parse_duration_months(text: Optional[str]) -> Optional[int]:
    """"6yrs 0mon" -> 72. Returns None if the text doesn't match — never
    raises and never guesses a partial parse.
    """
    if not text:
        return None
    m = _DURATION_RE.match(text)
    if not m:
        return None
    years, months = int(m.group(1)), int(m.group(2))
    return years * 12 + months


class CrifAdapter(EnrichmentAdapter):
    source_name = "crif"

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
        if rng.random() < 0.05:
            score = rng.choice([0] + list(range(10, 19)))
        else:
            score = rng.randint(300, 900)
        years, months = rng.randint(0, 15), rng.randint(0, 11)
        return {
            "hit": True,
            "score": score,
            "reason_code": f"R{rng.randint(1, 9)}" if score in _SENTINEL_SCORES else None,
            "total_accounts": rng.randint(0, 15),
            "active_accounts": rng.randint(0, 10),
            "overdue_accounts": rng.randint(0, 3),
            "total_outstanding": round(rng.uniform(0, 800_000), 2),
            "credit_history_length_text": f"{years}yrs {months}mon",
        }

    @staticmethod
    def _mock_no_hit_payload(rec: CanonicalRecord) -> dict[str, Any]:
        return {"hit": False, "score": 0, "reason_code": "R1", "total_accounts": 0, "active_accounts": 0,
                "overdue_accounts": 0, "total_outstanding": 0.0, "credit_history_length_text": "0yrs 0mon"}

    def transform(self, raw: RawResponse) -> dict[str, Any]:
        if raw.status != ResponseStatus.SUCCESS_MATCHED.value:
            return {name: None for name in _FEATURE_NAMES}
        payload = raw.payload
        score = payload.get("score")
        score = None if score is None or score in _SENTINEL_SCORES else score
        return {
            "crif_score": score,
            "crif_total_accounts": payload.get("total_accounts"),
            "crif_active_accounts": payload.get("active_accounts"),
            "crif_overdue_accounts": payload.get("overdue_accounts"),
            "crif_total_outstanding": payload.get("total_outstanding"),
            "crif_credit_history_months": _parse_duration_months(payload.get("credit_history_length_text")),
        }

    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)
