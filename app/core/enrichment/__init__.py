"""Enrichment adapters — one module per external source.

See base.py for the shared interface and CLAUDE.md for why every response
carries a fetched_at/batch_id (provenance leakage is the dominant risk here).
"""

from __future__ import annotations

from .base import EnrichmentAdapter
from .crif import CrifAdapter
from .epfo import EpfoAdapter
from .equifax import EquifaxAdapter
from .lead_credit_engine import LeadCreditEngineAdapter
from .payu import PayUAdapter
from .salary_estimator import SalaryEstimatorAdapter

# Source name -> adapter class. The UI's source multiselect and the runner
# both drive off this — adding a seventh source is a one-line addition here.
ADAPTER_REGISTRY: dict[str, type[EnrichmentAdapter]] = {
    "equifax": EquifaxAdapter,
    "crif": CrifAdapter,
    "epfo": EpfoAdapter,
    "salary_estimator": SalaryEstimatorAdapter,
    "lead_credit_engine": LeadCreditEngineAdapter,
    "payu": PayUAdapter,
}
