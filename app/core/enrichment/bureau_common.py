"""Shared credit-bureau classification rules.

These are about credit domain semantics (what counts as an active account,
which institution a lender name belongs to, how to bucket an account type),
not about any one bureau's payload format — Equifax uses them today; CRIF
will reuse them unchanged once its own transform() lands, per its own
payload shape.

Every rule here is verified against real production payloads (see the
account-type bucketing note on "credit_cards" vs "kisan credit card", and
the active-status note below) — do not "simplify" these without re-checking
against real data first.

No Streamlit import here or anywhere else under app/core/.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# safe_get — walk nested dicts/lists without raising on a missing/malformed path
# ---------------------------------------------------------------------------


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Walks `d[keys[0]][keys[1]]...`, where an integer key indexes into a
    list/tuple and anything else looks up a dict key. Returns `default` the
    moment the path doesn't exist or the container is the wrong shape —
    never raises. A malformed bureau payload (missing the nested report
    path entirely) is a normal, expected input here, not an error case.
    """
    current = d
    for key in keys:
        if isinstance(key, int) and not isinstance(key, bool):
            if isinstance(current, (list, tuple)) and -len(current) <= key < len(current):
                current = current[key]
            else:
                return default
        else:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
    return current


# ---------------------------------------------------------------------------
# Account-type buckets — 12, first match wins
# ---------------------------------------------------------------------------

OTHER_LOANS = "other_loans"

# (bucket_name, include_substrings, exclude_substrings). Checked in this exact
# order; the first bucket whose include list matches (and exclude list does
# not) wins. credit_cards is checked before business_loans specifically so
# its exclusions apply before "kisan credit card" would otherwise also match
# business_loans' own "kisan credit card" inclusion further down the list.
_ACCOUNT_TYPE_BUCKETS: list[tuple[str, list[str], list[str]]] = [
    ("microfinance_loans", ["microfinance"], []),
    ("credit_cards", ["credit card"], ["loan on credit card", "kisan credit card"]),
    ("gold_loans", ["gold loan"], []),
    ("home_loans", ["housing loan", "awas yojana", "home loan"], []),
    ("education_loans", ["education loan"], []),
    ("consumer_loans", ["consumer loan"], []),
    ("personal_loans", ["personal loan"], []),
    ("property_loans", ["property loan", "loan against property"], []),
    ("overdraft_accounts", ["overdraft"], []),
    ("business_loans", ["business", "mudra", "gecl", "loan to professional", "credit facility", "kisan credit card"], []),
    ("vehicle_loans", ["auto loan", "vehicle loan", "car loan", "tractor loan", "construction equipment", "two wheeler"], []),
]

# All 12 bucket names, in the order listed above plus the fallback.
ACCOUNT_TYPE_BUCKETS: list[str] = [name for name, _, _ in _ACCOUNT_TYPE_BUCKETS] + [OTHER_LOANS]


def normalize_account_type(value: Any) -> str:
    """Lowercase, non-alphanumerics collapsed to spaces, whitespace-trimmed."""
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def classify_account_type(account_type: Any) -> str:
    normalized = normalize_account_type(account_type)
    for bucket, includes, excludes in _ACCOUNT_TYPE_BUCKETS:
        if any(inc in normalized for inc in includes) and not any(exc in normalized for exc in excludes):
            return bucket
    return OTHER_LOANS


# ---------------------------------------------------------------------------
# Active-status rule
# ---------------------------------------------------------------------------

# Deliberately does NOT include Delinquent, Suit Filed or Restructured — those
# are still live obligations. Verified: this reproduces production active
# counts exactly; a stricter `status == "active"` check does not.
CLOSED_STATUSES: frozenset[str] = frozenset({
    "closed", "closed account", "charge off/written off", "written off", "settled",
    "post write off settled", "restructured & closed", "restructured and closed",
    "post write off closed", "account is inactive", "repossessed & settled",
    "sold/purchased",
})

SETTLED_STATUSES: frozenset[str] = frozenset({"settled", "post write off settled"})

WRITTENOFF_OR_SETTLED_STATUSES: frozenset[str] = frozenset({
    "charge off/written off", "written off", "post write off settled", "post write off closed",
} | SETTLED_STATUSES)


def normalize_status(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def is_active_account(status: Any) -> bool:
    return normalize_status(status) not in CLOSED_STATUSES


# ---------------------------------------------------------------------------
# Institution category — first match wins, order as listed
# ---------------------------------------------------------------------------

_PSU_BANKS = [
    "state bank of india", "punjab national bank", "canara bank", "bank of baroda",
    "union bank of india", "bank of india", "indian bank", "indian overseas bank",
    "uco bank", "central bank of india", "idbi bank", "punjab & sind bank",
    "uttar pradesh gramin bank",
]

_FOREIGN_BANKS = [
    "standard chartered bank", "citibank n.a.",
    "hongkong and shanghai banking corporation limited",
    "american express banking corp", "dbs bank ltd india",
    "natwest markets plc", "sbm bank (india) limited",
    "deutsche bank", "barclays bank",
]

_PRIVATE_BANKS = [
    "hdfc", "icici", "axis bank", "kotak mahindra bank", "indusind", "yes bank",
    "rbl bank", "idfc first", "karur vysya", "federal bank", "south indian bank",
    "jammu & kashmir bank", "development credit bank", "dcb bank",
    "au small finance bank", "slice small finance bank", "bandhan bank", "csb bank",
    "karnataka bank", "dhanlaxmi bank", "tamilnad mercantile",
]

_ARC_PATTERNS = [
    "asset reconstruction", "arcil", " arc ", "arc private", "arc limited",
    "assets care", "care & reconstruction",
]

_FINTECH_PATTERNS = [
    "mpokket", "krazybee", "kreditbee", "kisetsu saison", "cashe", "si creva",
    "branch international", "capfloat", "oxyzo", "earlysalary", "fibe",
    "true credits", "payu finance", "dhani", "snapmint", "ndx p2p", "whizdm",
    "innofin", "apollo finvest", "transactree", "fdpl finance", "upmove capital",
    "fincfriends", "moneytap", "zestmoney", "navi finserv", "kissht", "indialends",
    "simpl", "lazypay", "uni cards", "rupeek", "fintech", "trillionloans",
    "wheelsemi", "fairassets", "epimoney",
]

_NBFC_PATTERNS = [
    "finance", "financial", "fincorp", "capital", "credit", "investments",
    "housing", "leasing", "leyland", "securities", "loan", "muthoot",
    "sbi cards", "bob cards", "kotak mahindra prime", "hire purchase",
    "indel money", "lease", "grih rin",
]

INSTITUTION_CATEGORIES = (
    "PSU Bank", "Foreign Bank", "Private Bank", "Asset Reconstruction Company",
    "Fintech / Digital Lender", "NBFC", "Other / Unclassified",
)


def _normalize_institution(value: Any) -> str:
    return str(value or "").strip().lower()


def classify_institution(name: Any) -> str:
    normalized = _normalize_institution(name)
    if not normalized:
        return "Other / Unclassified"
    if any(p in normalized for p in _PSU_BANKS):
        return "PSU Bank"
    if any(p in normalized for p in _FOREIGN_BANKS):
        return "Foreign Bank"
    if any(p in normalized for p in _PRIVATE_BANKS):
        return "Private Bank"
    if any(p in normalized for p in _ARC_PATTERNS):
        return "Asset Reconstruction Company"
    if any(p in normalized for p in _FINTECH_PATTERNS):
        return "Fintech / Digital Lender"
    if any(p in normalized for p in _NBFC_PATTERNS):
        return "NBFC"
    if "bank" in normalized:
        return "Private Bank"
    if "fin" in normalized:
        return "NBFC"
    return "Other / Unclassified"


# ---------------------------------------------------------------------------
# Institution segment — derived from category + name
# ---------------------------------------------------------------------------

_DIGITAL_NBFC_PATTERNS = [
    "home credit", "bajaj finance", "krazybee", "kreditbee", "mpokket",
    "earlysalary", "fibe", "cashe", "dhani", "snapmint", "true credits",
]

_SECURED_NBFC_PATTERNS = [
    "muthoot", "manappuram", "indel money", "housing", "home finance", "leyland",
    "mahindra financial", "tvs credit", "l&t finance", "cholamandalam",
    "hero fincorp", "shriram", "hinduja", "grih rin", "wheelsemi",
]

INSTITUTION_SEGMENTS = (
    "Premium", "Mainstream", "Distressed", "Mass Market - Digital",
    "Mass Market - Secured", "Mass Market - General NBFC", "Unclassified",
)


def classify_institution_segment(category: str, name: Any) -> str:
    normalized = _normalize_institution(name)
    if category == "Foreign Bank" or "american express" in normalized:
        return "Premium"
    if "sbi cards" in normalized or "bob cards" in normalized or category in ("PSU Bank", "Private Bank"):
        return "Mainstream"
    if category == "Asset Reconstruction Company":
        return "Distressed"
    if category == "Fintech / Digital Lender" or (category == "NBFC" and any(p in normalized for p in _DIGITAL_NBFC_PATTERNS)):
        return "Mass Market - Digital"
    if category == "NBFC" and any(p in normalized for p in _SECURED_NBFC_PATTERNS):
        return "Mass Market - Secured"
    if category == "NBFC":
        return "Mass Market - General NBFC"
    return "Unclassified"


# ---------------------------------------------------------------------------
# Credit Diversity Index — Herfindahl-based
# ---------------------------------------------------------------------------


def credit_diversity_index(bucket_counts: dict[str, int]) -> float:
    """0 = all accounts one type. Higher = spread across many types."""
    total = sum(bucket_counts.values())
    if total == 0:
        return 0.0
    hhi = sum((count / total) ** 2 for count in bucket_counts.values())
    return (1 - hhi) * 100


# ---------------------------------------------------------------------------
# is_premium — person level, ACTIVE accounts only, Premium beats Mainstream
# ---------------------------------------------------------------------------


def is_premium(enriched: bool, active_counts_by_segment: dict[str, int]) -> Optional[str]:
    if not enriched:
        return None
    if active_counts_by_segment.get("Premium", 0) > 0:
        return "Premium"
    if active_counts_by_segment.get("Mainstream", 0) > 0:
        return "Mainstream"
    return None
