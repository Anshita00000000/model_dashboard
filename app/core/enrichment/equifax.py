"""Equifax adapter — credit bureau, alternative provider to CRIF.

fetch() is STUBBED (mock data, see base.mock_fetch) but the mock payload is
structurally identical to the real Decentro-wrapped Equifax report, so
transform() below runs unmodified against either.

transform() is a pure function verified against real production payloads:
no network, fully deterministic given a RawResponse. It shares its credit
domain rules (account-type bucketing, active-status definition, institution
classification) with bureau_common.py rather than reimplementing them here —
CRIF will reuse the same rules unchanged once its own transform() lands.

Score sentinel: a scoreDetails value of -1 means "no score" and is mapped to
None, same rationale as everywhere else in this codebase (CLAUDE.md —
"Sentinel values masquerade as numbers"). A non-MATCHED status, OR a matched
response whose payload is missing the nested report path entirely, both
return an all-null feature dict — "no bureau data" must stay a legitimate,
informative state, never zero-filled.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from . import bureau_common as bc
from .base import CanonicalRecord, EnrichmentAdapter, RawResponse, ResponseStatus, mock_fetch

logger = logging.getLogger(__name__)

_SCORE_SENTINEL = -1

# ---------------------------------------------------------------------------
# Feature-key short names (bucket / segment -> the suffix used in output keys)
# ---------------------------------------------------------------------------

_BUCKET_SHORT_NAME: dict[str, str] = {
    "microfinance_loans": "microfinance",
    "credit_cards": "credit_card",
    "gold_loans": "gold",
    "home_loans": "home",
    "education_loans": "education",
    "consumer_loans": "consumer",
    "personal_loans": "personal",
    "property_loans": "property",
    "overdraft_accounts": "overdraft",
    "business_loans": "business",
    "vehicle_loans": "vehicle",
    "other_loans": "other",
}

_SEGMENT_SHORT_NAME: dict[str, str] = {
    "Premium": "premium",
    "Mainstream": "mainstream",
    "Distressed": "distressed",
    "Mass Market - Digital": "mass_market_digital",
    "Mass Market - Secured": "mass_market_secured",
    "Mass Market - General NBFC": "mass_market_general_nbfc",
    "Unclassified": "unclassified",
}

# "10 non-card loan types": every bucket except credit_cards (a card, not a
# loan) and other_loans (a catch-all, not a specific type).
_SANCTIONED_PCT_BUCKETS = [
    b for b in bc.ACCOUNT_TYPE_BUCKETS if b not in ("credit_cards", "other_loans")
]
_BALANCE_ACTIVE_PCT_BUCKETS = ["consumer_loans", "overdraft_accounts", "other_loans"]
_INSTALLMENT_ACTIVE_PCT_BUCKETS = [
    "gold_loans", "consumer_loans", "personal_loans", "property_loans",
    "business_loans", "vehicle_loans", "other_loans",
]

# ---------------------------------------------------------------------------
# scoringElements[].description -> reason bucket
# ---------------------------------------------------------------------------

_REASON_BUCKET_DESCRIPTIONS: dict[str, list[str]] = {
    "utilization": [
        "Total Utilization", "Credit Card Utilization", "Kisan Credit Card Utilization",
    ],
    "delinquency_default": [
        "Delinquency Presence", "Overdue Amount", "Occurances of default account",
        "Overdue Amount on Unsecured Trades", "Overdue amount Credit Card",
        "Occurances of delinquent account", "Sanction Amount on Derog trades",
    ],
    "credit_exposure_unsecured_debt": [
        "Total Credit Exposure", "Recent Credit exposure", "Balance On Unsecured Trade",
        "Percentage of Secure Trades", "Total credit exposure of Unsecured Loan",
    ],
    "portfolio_composition": [
        "Non Retail Trades", "Number of Home Loans", "Number of Personal Loans",
        "Number of Gold Loans", "Number of Credit Cards", "Number of live Consumer Durables loans",
    ],
    "credit_history_vintage_activity": [
        "Vintage in Bureau", "No Live Accounts", "Limited Credit History",
        "Recent Closed Accounts", "No Active Accounts", "Insufficient information to score",
        "Earlier Closed Accounts", "Not Applicable", "Installment loan opened recently",
    ],
}
_REASON_BUCKETS = list(_REASON_BUCKET_DESCRIPTIONS.keys())
_REASON_DESCRIPTION_TO_BUCKET: dict[str, str] = {
    desc.strip().lower(): bucket
    for bucket, descriptions in _REASON_BUCKET_DESCRIPTIONS.items()
    for desc in descriptions
}

# ---------------------------------------------------------------------------
# Full ordered feature-name list — built from the same constants transform()
# uses, so the two can never drift apart.
# ---------------------------------------------------------------------------

_PORTFOLIO_FEATURES = [
    "equifax_score", "equifax_total_accounts", "equifax_active_accounts", "equifax_active_pct",
    "equifax_closed_accounts", "equifax_closed_pct", "equifax_settled_accounts",
    "equifax_writtenoff_or_settled_pct",
]
_AGE_FEATURES = [
    "equifax_oldest_account_age_months", "equifax_newest_account_age_months",
    "equifax_average_account_age_months",
]
_TOTALS_FEATURES = [
    "equifax_total_credit_limit", "equifax_total_active_credit_limit",
    "equifax_total_sanction_amount", "equifax_total_active_sanction_amount",
    "equifax_total_balance_amount", "equifax_total_monthly_payment_amount",
]
_IDENTITY_FEATURES = [
    "equifax_gender", "equifax_age_years", "equifax_has_pan", "equifax_has_passport",
    "equifax_has_voter_id",
]

_FEATURE_NAMES: list[str] = (
    _PORTFOLIO_FEATURES
    + _AGE_FEATURES
    + _TOTALS_FEATURES
    + [f"equifax_num_{b}" for b in bc.ACCOUNT_TYPE_BUCKETS]
    + [f"equifax_active_count_{b}" for b in bc.ACCOUNT_TYPE_BUCKETS]
    + [f"equifax_total_count_institution_segment_{_SEGMENT_SHORT_NAME[s]}" for s in bc.INSTITUTION_SEGMENTS]
    + [f"equifax_active_count_institution_segment_{_SEGMENT_SHORT_NAME[s]}" for s in bc.INSTITUTION_SEGMENTS]
    + ["equifax_total_count_institution_category_psu_bank"]
    + ["equifax_is_premium", "equifax_credit_diversity_index"]
    + [f"equifax_pct_of_total_sanctioned_amount_{_BUCKET_SHORT_NAME[b]}" for b in _SANCTIONED_PCT_BUCKETS]
    + [f"equifax_pct_of_total_sanctioned_amount_active_{_BUCKET_SHORT_NAME[b]}" for b in _SANCTIONED_PCT_BUCKETS]
    + [f"equifax_pct_of_total_balance_amount_active_{_BUCKET_SHORT_NAME[b]}" for b in _BALANCE_ACTIVE_PCT_BUCKETS]
    + [f"equifax_pct_of_total_installment_amount_active_{_BUCKET_SHORT_NAME[b]}" for b in _INSTALLMENT_ACTIVE_PCT_BUCKETS]
    + ["equifax_suit_filed_status_values"]
    + ["equifax_summary_most_severe_status_with_in24_months", "equifax_summary_no_of_past_due_accounts"]
    + [f"equifax_reason_count_{b}" for b in _REASON_BUCKETS]
    + _IDENTITY_FEATURES
)


# ---------------------------------------------------------------------------
# Small numeric/date helpers
# ---------------------------------------------------------------------------


def _to_number(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return default if pd.isna(value) else float(value)
    text = str(value).strip().replace(",", "")
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed


def _month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _safe_pct(numerator: float, denominator: Optional[float]) -> Optional[float]:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 4)


# ---------------------------------------------------------------------------
# Mock fetch — structurally identical to the real Decentro-wrapped envelope
# ---------------------------------------------------------------------------

_MOCK_ACCOUNT_TYPES = [
    "Credit Card", "Gold Loan", "Housing Loan", "Personal Loan", "Two Wheeler Loan",
    "Kisan Credit Card", "Business Loan", "Consumer Loan", "Microfinance Loan",
    "Overdraft Facility", "Education Loan", "Loan Against Property",
]
_MOCK_INSTITUTIONS = [
    "HDFC Bank", "State Bank Of India", "Bajaj Finance Limited", "MPOKKET",
    "Standard Chartered Bank", "Muthoot Finance", "ICICI Bank", "Kreditbee",
]
_MOCK_STATUSES = ["Active", "Closed", "Settled", "Delinquent", "Written Off", "Current"]
_MOCK_REASON_DESCRIPTIONS = [desc for descs in _REASON_BUCKET_DESCRIPTIONS.values() for desc in descs]


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
        n_accounts = rng.randint(1, 5)
        accounts = []
        for _ in range(n_accounts):
            opened = (datetime.now(timezone.utc) - timedelta(days=rng.randint(30, 3650))).strftime("%d-%m-%Y")
            reported = (datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 60))).strftime("%d-%m-%Y")
            accounts.append({
                "accountType": rng.choice(_MOCK_ACCOUNT_TYPES),
                "institution": rng.choice(_MOCK_INSTITUTIONS),
                "creditLimit": round(rng.uniform(0, 200_000), 2),
                "sanctionAmount": round(rng.uniform(0, 500_000), 2),
                "installmentAmount": round(rng.uniform(0, 20_000), 2),
                "balance": round(rng.uniform(0, 300_000), 2),
                "accountStatus": rng.choice(_MOCK_STATUSES),
                "dateReported": reported,
                "dateOpened": opened,
                "open": True,
                "history48Months": [
                    {"key": str(j), "paymentStatus": "0",
                     "suitFiledStatus": rng.choice(["*", "*", "*", "Suit Filed"]),
                     "assetClassificationStatus": "STD"}
                    for j in range(3)
                ],
            })
        score = _SCORE_SENTINEL if rng.random() < 0.05 else rng.randint(300, 900)
        scoring_elements = [{"description": rng.choice(_MOCK_REASON_DESCRIPTIONS)} for _ in range(rng.randint(2, 5))]
        return {
            "data": {"cCRResponse": {"cIRReportDataLst": [{"cIRReportData": {
                "retailAccountDetails": accounts,
                "retailAccountsSummary": {
                    "mostSevereStatusWithIn24Months": rng.choice(["STD", "SUB", "DBT"]),
                    "noOfPastDueAccounts": rng.randint(0, 2),
                    "totalBalanceAmount": sum(a["balance"] for a in accounts),
                    "totalSanctionAmount": sum(a["sanctionAmount"] for a in accounts),
                    "totalCreditLimit": sum(a["creditLimit"] for a in accounts),
                    "totalMonthlyPaymentAmount": sum(a["installmentAmount"] for a in accounts),
                    "noOfAccounts": len(accounts),
                    "noOfActiveAccounts": sum(1 for a in accounts if bc.is_active_account(a["accountStatus"])),
                },
                "scoreDetails": [{"value": score, "scoringElements": scoring_elements}],
                "iDAndContactInfo": {
                    "personalInfo": {"gender": rng.choice(["M", "F"]), "dateOfBirth": "15-08-1990"},
                    "identityInfo": {
                        "pANId": [{"idNumber": "ABCDE1234F"}], "nationalIDCard": [],
                        "otherId": [{"idNumber": "XYZ123"}],
                    },
                },
            }}]}}
        }

    @staticmethod
    def _mock_no_hit_payload(rec: CanonicalRecord) -> dict[str, Any]:
        # Structurally realistic: the envelope exists but carries no report.
        return {"data": {"cCRResponse": {"cIRReportDataLst": []}}}

    # -- transform ----------------------------------------------------------

    def transform(self, raw: RawResponse) -> dict[str, Any]:
        if raw.status != ResponseStatus.SUCCESS_MATCHED.value:
            return {name: None for name in _FEATURE_NAMES}

        root = bc.safe_get(raw.payload, "data", "cCRResponse", "cIRReportDataLst", 0, "cIRReportData")
        if root is None:
            # Matched, but the report body itself is missing/malformed — treat
            # exactly like "no bureau data" rather than raising or zero-filling.
            return {name: None for name in _FEATURE_NAMES}

        accounts = bc.safe_get(root, "retailAccountDetails", default=[])
        if not isinstance(accounts, list):
            accounts = []
        summary = bc.safe_get(root, "retailAccountsSummary", default={}) or {}

        score = _to_number(bc.safe_get(root, "scoreDetails", 0, "value"), default=None)
        if score == _SCORE_SENTINEL:
            score = None

        scoring_elements = bc.safe_get(root, "scoreDetails", 0, "scoringElements", default=[])
        if not isinstance(scoring_elements, list):
            scoring_elements = []

        active_flags = [bc.is_active_account(a.get("accountStatus")) for a in accounts]
        buckets = [bc.classify_account_type(a.get("accountType")) for a in accounts]
        categories = [bc.classify_institution(a.get("institution")) for a in accounts]
        segments = [bc.classify_institution_segment(cat, a.get("institution")) for a, cat in zip(accounts, categories)]

        total_accounts = len(accounts)
        active_accounts = sum(active_flags)
        closed_accounts = total_accounts - active_accounts
        statuses_normalized = [bc.normalize_status(a.get("accountStatus")) for a in accounts]
        settled_accounts = sum(1 for s in statuses_normalized if s in bc.SETTLED_STATUSES)
        writtenoff_or_settled = sum(1 for s in statuses_normalized if s in bc.WRITTENOFF_OR_SETTLED_STATUSES)

        features: dict[str, Any] = {
            "equifax_score": score,
            "equifax_total_accounts": total_accounts,
            "equifax_active_accounts": active_accounts,
            "equifax_active_pct": _safe_pct(active_accounts, total_accounts),
            "equifax_closed_accounts": closed_accounts,
            "equifax_closed_pct": _safe_pct(closed_accounts, total_accounts),
            "equifax_settled_accounts": settled_accounts,
            "equifax_writtenoff_or_settled_pct": _safe_pct(writtenoff_or_settled, total_accounts),
        }

        # -- account ages: measured from the most recent dateReported across
        # this person's OWN accounts, not from today, so the feature is stable
        # regardless of when the pipeline runs.
        reported_dates = [d for d in (_parse_date(a.get("dateReported")) for a in accounts) if d is not None]
        reference_date = max(reported_dates) if reported_dates else None
        ages = []
        if reference_date is not None:
            for a in accounts:
                opened = _parse_date(a.get("dateOpened"))
                if opened is not None:
                    ages.append(max(0, _month_diff(reference_date, opened)))
        features["equifax_oldest_account_age_months"] = max(ages) if ages else None
        features["equifax_newest_account_age_months"] = min(ages) if ages else None
        features["equifax_average_account_age_months"] = round(sum(ages) / len(ages), 4) if ages else None

        # -- totals
        credit_limits = [_to_number(a.get("creditLimit")) for a in accounts]
        sanction_amounts = [_to_number(a.get("sanctionAmount")) for a in accounts]
        balances = [_to_number(a.get("balance")) for a in accounts]
        installments = [_to_number(a.get("installmentAmount")) for a in accounts]

        total_sanction_amount = sum(sanction_amounts) if accounts else None
        total_active_sanction_amount = sum(v for v, a in zip(sanction_amounts, active_flags) if a) if accounts else None
        total_active_balance_amount = sum(v for v, a in zip(balances, active_flags) if a) if accounts else 0.0
        total_active_installment_amount = sum(v for v, a in zip(installments, active_flags) if a) if accounts else 0.0

        features["equifax_total_credit_limit"] = sum(credit_limits) if accounts else None
        features["equifax_total_active_credit_limit"] = (
            sum(v for v, a in zip(credit_limits, active_flags) if a) if accounts else None
        )
        features["equifax_total_sanction_amount"] = total_sanction_amount
        features["equifax_total_active_sanction_amount"] = total_active_sanction_amount
        features["equifax_total_balance_amount"] = sum(balances) if accounts else None
        features["equifax_total_monthly_payment_amount"] = sum(installments) if accounts else None

        # -- per-bucket counts (12 x 2)
        for bucket_name in bc.ACCOUNT_TYPE_BUCKETS:
            features[f"equifax_num_{bucket_name}"] = sum(1 for b in buckets if b == bucket_name)
        for bucket_name in bc.ACCOUNT_TYPE_BUCKETS:
            features[f"equifax_active_count_{bucket_name}"] = sum(
                1 for b, a in zip(buckets, active_flags) if b == bucket_name and a
            )

        # -- institution segments (7 x 2) + psu_bank category total
        active_counts_by_segment = {
            seg: sum(1 for s, a in zip(segments, active_flags) if s == seg and a) for seg in bc.INSTITUTION_SEGMENTS
        }
        for seg in bc.INSTITUTION_SEGMENTS:
            features[f"equifax_total_count_institution_segment_{_SEGMENT_SHORT_NAME[seg]}"] = sum(
                1 for s in segments if s == seg
            )
        for seg in bc.INSTITUTION_SEGMENTS:
            features[f"equifax_active_count_institution_segment_{_SEGMENT_SHORT_NAME[seg]}"] = active_counts_by_segment[seg]
        features["equifax_total_count_institution_category_psu_bank"] = sum(1 for c in categories if c == "PSU Bank")

        # -- is_premium / credit diversity index
        features["equifax_is_premium"] = bc.is_premium(True, active_counts_by_segment)
        if total_accounts == 0:
            features["equifax_credit_diversity_index"] = None
        else:
            bucket_counts = {b: buckets.count(b) for b in bc.ACCOUNT_TYPE_BUCKETS}
            features["equifax_credit_diversity_index"] = round(bc.credit_diversity_index(bucket_counts), 4)

        # -- percentages
        for bucket_name in _SANCTIONED_PCT_BUCKETS:
            short = _BUCKET_SHORT_NAME[bucket_name]
            bucket_total = sum(v for v, b in zip(sanction_amounts, buckets) if b == bucket_name)
            features[f"equifax_pct_of_total_sanctioned_amount_{short}"] = _safe_pct(bucket_total, total_sanction_amount)
        for bucket_name in _SANCTIONED_PCT_BUCKETS:
            short = _BUCKET_SHORT_NAME[bucket_name]
            bucket_active_total = sum(v for v, b, a in zip(sanction_amounts, buckets, active_flags) if b == bucket_name and a)
            features[f"equifax_pct_of_total_sanctioned_amount_active_{short}"] = _safe_pct(
                bucket_active_total, total_active_sanction_amount
            )
        for bucket_name in _BALANCE_ACTIVE_PCT_BUCKETS:
            short = _BUCKET_SHORT_NAME[bucket_name]
            bucket_active_total = sum(v for v, b, a in zip(balances, buckets, active_flags) if b == bucket_name and a)
            features[f"equifax_pct_of_total_balance_amount_active_{short}"] = _safe_pct(
                bucket_active_total, total_active_balance_amount
            )
        for bucket_name in _INSTALLMENT_ACTIVE_PCT_BUCKETS:
            short = _BUCKET_SHORT_NAME[bucket_name]
            bucket_active_total = sum(v for v, b, a in zip(installments, buckets, active_flags) if b == bucket_name and a)
            features[f"equifax_pct_of_total_installment_amount_active_{short}"] = _safe_pct(
                bucket_active_total, total_active_installment_amount
            )

        # -- suit-filed status values: unique non-"*" values across all
        # accounts' 48-month history, pipe-joined. None when there are no
        # accounts to check at all; "" is a real (checked, empty) result.
        if not accounts:
            features["equifax_suit_filed_status_values"] = None
        else:
            values: set[str] = set()
            for a in accounts:
                history = a.get("history48Months")
                if not isinstance(history, list):
                    continue
                for entry in history:
                    if not isinstance(entry, dict):
                        continue
                    v = entry.get("suitFiledStatus")
                    if v is None:
                        continue
                    v_str = str(v).strip()
                    if v_str in ("", "*"):
                        continue
                    values.add(v_str)
            features["equifax_suit_filed_status_values"] = "|".join(sorted(values))

        # -- summary passthrough (verbatim)
        features["equifax_summary_most_severe_status_with_in24_months"] = summary.get("mostSevereStatusWithIn24Months")
        features["equifax_summary_no_of_past_due_accounts"] = summary.get("noOfPastDueAccounts")

        # -- reason buckets, from scoringElements[].description
        reason_counts = {b: 0 for b in _REASON_BUCKETS}
        for element in scoring_elements:
            if not isinstance(element, dict):
                continue
            desc = element.get("description")
            if desc is None:
                continue
            bucket = _REASON_DESCRIPTION_TO_BUCKET.get(str(desc).strip().lower())
            if bucket is None:
                logger.warning("equifax: unmapped scoringElements description %r", desc)
                continue
            reason_counts[bucket] += 1
        for bucket in _REASON_BUCKETS:
            features[f"equifax_reason_count_{bucket}"] = reason_counts[bucket]

        # -- identity / demographics
        personal_info = bc.safe_get(root, "iDAndContactInfo", "personalInfo", default={}) or {}
        identity_info = bc.safe_get(root, "iDAndContactInfo", "identityInfo", default={}) or {}
        features["equifax_gender"] = personal_info.get("gender")
        dob = _parse_date(personal_info.get("dateOfBirth"))
        if dob is not None and reference_date is not None:
            features["equifax_age_years"] = max(0, _month_diff(reference_date, dob) // 12)
        else:
            features["equifax_age_years"] = None
        features["equifax_has_pan"] = bool(identity_info.get("pANId"))
        features["equifax_has_passport"] = bool(identity_info.get("nationalIDCard"))
        features["equifax_has_voter_id"] = bool(identity_info.get("otherId"))

        return features

    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)
