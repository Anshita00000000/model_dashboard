"""EPFO adapter — employment / provident fund status.

fetch() is STUBBED (mock data, structurally identical to the real response
shape). transform() is NOT implemented yet — returns {} so the pipeline stays
end-to-end runnable, but produces no features. Implement against the mock
payload shape below in a future pass.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .base import CanonicalRecord, EnrichmentAdapter, RawResponse, mock_fetch


class EpfoAdapter(EnrichmentAdapter):
    source_name = "epfo"

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
        return {
            "hit": True,
            "employment_status": rng.choice(["employed", "self_employed", "unemployed"]),
            "current_employer": rng.choice(["Acme Pvt Ltd", "Globex Corp", None]),
            "uan_active": rng.random() < 0.7,
            "monthly_contribution": round(rng.uniform(500, 15_000), 2),
            "employment_tenure_months": rng.randint(0, 240),
        }

    def transform(self, raw: RawResponse) -> dict[str, Any]:
        # TODO: implement in a future pass, against the mock payload shape above.
        return {}

    def feature_names(self) -> list[str]:
        # TODO: populate once transform() is implemented.
        return []
