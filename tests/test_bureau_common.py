"""Tests for app/core/enrichment/bureau_common.py — shared credit-bureau
classification rules (account-type bucketing, active-status, institution
classification, credit diversity index, is_premium). These rules are shared
between Equifax and CRIF; nothing here is bureau-specific.
"""

from __future__ import annotations

from app.core.enrichment import bureau_common as bc


# ---------------------------------------------------------------------------
# safe_get
# ---------------------------------------------------------------------------


def test_safe_get_walks_nested_dicts_and_lists():
    d = {"a": {"b": [1, 2, {"c": 3}]}}
    assert bc.safe_get(d, "a", "b", 2, "c") == 3


def test_safe_get_returns_default_on_missing_key():
    assert bc.safe_get({"a": 1}, "x", "y", default="MISSING") == "MISSING"


def test_safe_get_returns_default_on_out_of_range_index():
    assert bc.safe_get({"a": [1, 2]}, "a", 5, default="MISSING") == "MISSING"


def test_safe_get_returns_default_when_container_shape_is_wrong():
    # "a" resolves to an int, not a dict/list -- indexing further must not raise
    assert bc.safe_get({"a": 1}, "a", "b", default="MISSING") == "MISSING"


def test_safe_get_never_raises_on_completely_malformed_input():
    assert bc.safe_get(None, "a", "b", 0, default=None) is None
    assert bc.safe_get({}, "data", "cCRResponse", "cIRReportDataLst", 0, "cIRReportData") is None


# ---------------------------------------------------------------------------
# Account-type bucketing — 12 buckets, first match wins
# ---------------------------------------------------------------------------


def test_kisan_credit_card_lands_in_business_loans_not_credit_cards():
    # The exclusion list is the whole point: "credit card" alone would match
    # credit_cards first since it's checked earlier in the list.
    assert bc.classify_account_type("Kisan Credit Card") == "business_loans"


def test_loan_on_credit_card_is_excluded_from_credit_cards_and_falls_to_other():
    assert bc.classify_account_type("Loan on Credit Card") == "other_loans"


def test_plain_credit_card_is_credit_cards():
    assert bc.classify_account_type("Credit Card") == "credit_cards"


def test_home_loan_synonyms_all_map_to_home_loans():
    assert bc.classify_account_type("Housing Loan") == "home_loans"
    assert bc.classify_account_type("Pradhan Mantri Awas Yojana") == "home_loans"
    assert bc.classify_account_type("Home Loan") == "home_loans"


def test_unrecognized_type_falls_back_to_other_loans():
    assert bc.classify_account_type("Some Exotic Facility") == "other_loans"


def test_account_type_normalization_is_case_and_punctuation_insensitive():
    assert bc.classify_account_type("  GOLD-LOAN!!  ") == "gold_loans"


def test_account_type_none_falls_back_to_other_loans():
    assert bc.classify_account_type(None) == "other_loans"


def test_all_twelve_buckets_are_reachable():
    samples = {
        "microfinance_loans": "Microfinance Loan", "credit_cards": "Credit Card",
        "gold_loans": "Gold Loan", "home_loans": "Home Loan", "education_loans": "Education Loan",
        "consumer_loans": "Consumer Loan", "personal_loans": "Personal Loan",
        "property_loans": "Loan Against Property", "overdraft_accounts": "Overdraft Facility",
        "business_loans": "Business Loan", "vehicle_loans": "Car Loan", "other_loans": "Unknown Type",
    }
    assert set(samples.keys()) == set(bc.ACCOUNT_TYPE_BUCKETS)
    for expected_bucket, sample in samples.items():
        assert bc.classify_account_type(sample) == expected_bucket


# ---------------------------------------------------------------------------
# Active-status rule
# ---------------------------------------------------------------------------


def test_delinquent_suit_filed_and_restructured_count_as_active():
    # Deliberately still live obligations, per CLAUDE.md-verified production behaviour.
    assert bc.is_active_account("Delinquent") is True
    assert bc.is_active_account("Suit Filed") is True
    assert bc.is_active_account("Restructured") is True


def test_closed_like_statuses_count_as_inactive():
    for status in [
        "Closed", "Closed Account", "Charge Off/Written Off", "Written Off", "Settled",
        "Post Write Off Settled", "Restructured & Closed", "Restructured and Closed",
        "Post Write Off Closed", "Account is Inactive", "Repossessed & Settled", "Sold/Purchased",
    ]:
        assert bc.is_active_account(status) is False, status


def test_active_status_matching_is_whitespace_and_case_insensitive():
    assert bc.is_active_account("  CLOSED  ") is False
    assert bc.is_active_account("closed") is False


def test_active_status_none_is_active():
    assert bc.is_active_account(None) is True


# ---------------------------------------------------------------------------
# Institution category / segment
# ---------------------------------------------------------------------------


def test_psu_banks_classified_correctly():
    assert bc.classify_institution("State Bank Of India") == "PSU Bank"
    assert bc.classify_institution("Punjab National Bank") == "PSU Bank"


def test_foreign_banks_classified_correctly():
    assert bc.classify_institution("Standard Chartered Bank") == "Foreign Bank"


def test_private_banks_classified_by_substring():
    assert bc.classify_institution("HDFC Bank") == "Private Bank"
    assert bc.classify_institution("ICICI Bank Ltd") == "Private Bank"


def test_fintech_lenders_classified_correctly():
    assert bc.classify_institution("MPOKKET Financial Services") == "Fintech / Digital Lender"


def test_nbfc_fallback_substring_match():
    assert bc.classify_institution("Muthoot Finance") == "NBFC"


def test_generic_bank_name_falls_back_to_private_bank():
    assert bc.classify_institution("Some New Cooperative Bank") == "Private Bank"


def test_generic_fin_name_falls_back_to_nbfc():
    assert bc.classify_institution("Random Fin Corp Ltd") == "NBFC"


def test_unrecognized_institution_is_unclassified():
    assert bc.classify_institution("Totally Unknown Entity") == "Other / Unclassified"


def test_blank_institution_is_unclassified():
    assert bc.classify_institution(None) == "Other / Unclassified"
    assert bc.classify_institution("") == "Other / Unclassified"


def test_institution_segment_premium_beats_everything():
    assert bc.classify_institution_segment("Foreign Bank", "Standard Chartered Bank") == "Premium"
    assert bc.classify_institution_segment("NBFC", "American Express Something") == "Premium"


def test_institution_segment_mainstream_for_psu_and_private_banks():
    assert bc.classify_institution_segment("PSU Bank", "State Bank Of India") == "Mainstream"
    assert bc.classify_institution_segment("Private Bank", "HDFC Bank") == "Mainstream"


def test_institution_segment_distressed_for_arc():
    assert bc.classify_institution_segment("Asset Reconstruction Company", "ARCIL") == "Distressed"


def test_institution_segment_digital_nbfc():
    assert bc.classify_institution_segment("NBFC", "Bajaj Finance Limited") == "Mass Market - Digital"


def test_institution_segment_secured_nbfc():
    assert bc.classify_institution_segment("NBFC", "Muthoot Finance") == "Mass Market - Secured"


def test_institution_segment_general_nbfc_fallback():
    assert bc.classify_institution_segment("NBFC", "Some Random Capital Ltd") == "Mass Market - General NBFC"


def test_institution_segment_unclassified_fallback():
    assert bc.classify_institution_segment("Other / Unclassified", "Nothing Matches") == "Unclassified"


# ---------------------------------------------------------------------------
# Credit Diversity Index
# ---------------------------------------------------------------------------


def test_cdi_zero_when_all_accounts_one_type():
    assert bc.credit_diversity_index({"gold_loans": 5}) == 0.0


def test_cdi_higher_when_spread_across_types():
    assert bc.credit_diversity_index({"a": 1, "b": 1, "c": 1, "d": 1}) == 75.0


def test_cdi_zero_accounts_returns_zero_not_nan():
    assert bc.credit_diversity_index({"a": 0, "b": 0}) == 0.0


# ---------------------------------------------------------------------------
# is_premium
# ---------------------------------------------------------------------------


def test_is_premium_none_when_not_enriched():
    assert bc.is_premium(False, {"Premium": 3}) is None


def test_is_premium_premium_beats_mainstream():
    assert bc.is_premium(True, {"Premium": 1, "Mainstream": 5}) == "Premium"


def test_is_premium_mainstream_when_no_premium_active():
    assert bc.is_premium(True, {"Premium": 0, "Mainstream": 2}) == "Mainstream"


def test_is_premium_none_when_no_active_premium_or_mainstream():
    assert bc.is_premium(True, {"Distressed": 4}) is None
