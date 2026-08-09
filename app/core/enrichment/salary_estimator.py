"""Salary estimator adapter — modelled income.

fetch() is STUBBED (mock data, structurally identical to the real response
shape). transform() is NOT implemented yet — returns {} so the pipeline stays
end-to-end runnable, but produces no features. Implement against the mock
payload shape below in a future pass.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .base import CanonicalRecord, EnrichmentAdapter, RawResponse, mock_fetch


class SalaryEstimatorAdapter(EnrichmentAdapter):
    source_name = "salary_estimator"

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
            seed=self.seed,
        )

    @staticmethod
    def _mock_matched_payload(rec: CanonicalRecord, rng: random.Random) -> dict[str, Any]:
        income = round(rng.uniform(12_000, 200_000), 2)
        return {
            "hit": True,
            "estimated_monthly_income": income,
            "income_band": rng.choice(["<25k", "25k-50k", "50k-100k", "100k+"]),
            "confidence": round(rng.uniform(0.4, 0.95), 2),
        }

    def transform(self, raw: RawResponse) -> dict[str, Any]:
        # TODO: implement in a future pass, against the mock payload shape above.
        return {}

    def feature_names(self) -> list[str]:
        # TODO: populate once transform() is implemented.
        return []
