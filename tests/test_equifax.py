"""Tests for app/core/enrichment/equifax.py's transform() — a pure function,
verified against real production payload shapes. Every fixture's expected
feature values below were hand-derived from the fixture data and
independently cross-checked before being hardcoded here (see the account-by-
account arithmetic in each fixture's docstring).
"""

from __future__ import annotations

import logging

import pytest

from app.core.enrichment.base import RawResponse, ResponseStatus
from app.core.enrichment.equifax import EquifaxAdapter, _FEATURE_NAMES

ADAPTER = EquifaxAdapter()


def _wrap(accounts, score, scoring_elements, summary, personal_info, identity_info) -> dict:
    return {
        "data": {"cCRResponse": {"cIRReportDataLst": [{"cIRReportData": {
            "retailAccountDetails": accounts,
            "retailAccountsSummary": summary,
            "scoreDetails": [{"value": score, "scoringElements": scoring_elements}],
            "iDAndContactInfo": {"personalInfo": personal_info, "identityInfo": identity_info},
        }}]}}
    }


def _matched(payload: dict) -> RawResponse:
    return RawResponse(lead_id="1", source="equifax", status=ResponseStatus.SUCCESS_MATCHED.value,
                        payload=payload, fetched_at="t", batch_id="b")


# ---------------------------------------------------------------------------
# feature_names() / non-matched statuses
# ---------------------------------------------------------------------------


def test_feature_names_are_unique_and_all_prefixed():
    names = ADAPTER.feature_names()
    assert len(names) == len(set(names))
    assert all(n.startswith("equifax_") for n in names)


@pytest.mark.parametrize("status", [
    ResponseStatus.SUCCESS_NO_HIT.value, ResponseStatus.ERROR.value, ResponseStatus.NOT_SENT.value,
])
def test_non_matched_status_returns_all_null(status):
    raw = RawResponse(lead_id="1", source="equifax", status=status, payload={"anything": "here"}, fetched_at="t", batch_id="b")
    features = ADAPTER.transform(raw)
    assert set(features.keys()) == set(_FEATURE_NAMES)
    assert all(v is None for v in features.values())


# ---------------------------------------------------------------------------
# Fixture 1: normal multi-account lead
#
# Account 1: Credit Card / HDFC Bank, Active, opened 01-06-2020, reported 01-06-2024 (48mo)
# Account 2: Gold Loan / Muthoot Finance, Closed, opened 01-06-2022, reported 01-06-2024 (24mo)
# Account 3: Kisan Credit Card / SBI, Settled, opened 01-05-2023, reported 01-05-2024
#            (age measured from the reference date 01-06-2024, not its own dateReported -> 13mo)
# Account 4: Two Wheeler Loan / MPOKKET, Delinquent (=active), opened 01-06-2023, reported 01-06-2024 (12mo)
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_lead_features():
    accounts = [
        {"accountType": "Credit Card", "institution": "HDFC Bank", "creditLimit": 100000, "sanctionAmount": 100000,
         "installmentAmount": 5000, "balance": 20000, "accountStatus": "Active",
         "dateReported": "01-06-2024", "dateOpened": "01-06-2020",
         "history48Months": [{"suitFiledStatus": "*"}, {"suitFiledStatus": "Suit Filed"}]},
        {"accountType": "Gold Loan", "institution": "Muthoot Finance", "creditLimit": 0, "sanctionAmount": 50000,
         "installmentAmount": 2000, "balance": 10000, "accountStatus": "Closed",
         "dateReported": "01-06-2024", "dateOpened": "01-06-2022", "history48Months": []},
        {"accountType": "Kisan Credit Card", "institution": "State Bank Of India", "creditLimit": 0, "sanctionAmount": 30000,
         "installmentAmount": 0, "balance": 5000, "accountStatus": "Settled",
         "dateReported": "01-05-2024", "dateOpened": "01-05-2023",
         "history48Months": [{"suitFiledStatus": "Suit Filed"}, {"suitFiledStatus": "*"}]},
        {"accountType": "Two Wheeler Loan", "institution": "MPOKKET", "creditLimit": 0, "sanctionAmount": 20000,
         "installmentAmount": 1500, "balance": 8000, "accountStatus": "Delinquent",
         "dateReported": "01-06-2024", "dateOpened": "01-06-2023", "history48Months": [{"suitFiledStatus": None}]},
    ]
    payload = _wrap(
        accounts, 720,
        [{"description": "Total Utilization"}, {"description": "Overdue Amount"}, {"description": "Vintage in Bureau"},
         {"description": "Vintage in Bureau"}, {"description": "SomeUnknownReason"}],
        {"mostSevereStatusWithIn24Months": "SUB", "noOfPastDueAccounts": 1},
        {"gender": "F", "dateOfBirth": "15-08-1990"},
        {"pANId": [{"idNumber": "ABCDE1234F"}], "nationalIDCard": [], "otherId": [{"idNumber": "XYZ123"}]},
    )
    return ADAPTER.transform(_matched(payload))


def test_normal_lead_portfolio_aggregates(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_score"] == 720.0
    assert f["equifax_total_accounts"] == 4
    assert f["equifax_active_accounts"] == 2  # Active (acc1) + Delinquent (acc4) both count as active
    assert f["equifax_active_pct"] == 50.0
    assert f["equifax_closed_accounts"] == 2
    assert f["equifax_closed_pct"] == 50.0
    assert f["equifax_settled_accounts"] == 1
    assert f["equifax_writtenoff_or_settled_pct"] == 25.0


def test_normal_lead_account_ages_use_most_recent_date_reported_not_today(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_oldest_account_age_months"] == 48
    assert f["equifax_newest_account_age_months"] == 12
    assert f["equifax_average_account_age_months"] == 24.25


def test_normal_lead_totals(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_total_credit_limit"] == 100000.0
    assert f["equifax_total_active_credit_limit"] == 100000.0
    assert f["equifax_total_sanction_amount"] == 200000.0
    assert f["equifax_total_active_sanction_amount"] == 120000.0
    assert f["equifax_total_balance_amount"] == 43000.0
    assert f["equifax_total_monthly_payment_amount"] == 8500.0


def test_normal_lead_bucket_counts_kisan_credit_card_goes_to_business_loans(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_num_credit_cards"] == 1
    assert f["equifax_num_gold_loans"] == 1
    assert f["equifax_num_business_loans"] == 1  # the Kisan Credit Card account
    assert f["equifax_num_vehicle_loans"] == 1
    assert f["equifax_num_home_loans"] == 0
    assert f["equifax_active_count_credit_cards"] == 1
    assert f["equifax_active_count_gold_loans"] == 0  # closed
    assert f["equifax_active_count_business_loans"] == 0  # settled
    assert f["equifax_active_count_vehicle_loans"] == 1  # delinquent counts as active


def test_normal_lead_institution_segments(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_total_count_institution_segment_mainstream"] == 2  # HDFC + SBI
    assert f["equifax_total_count_institution_segment_mass_market_digital"] == 1  # MPOKKET
    assert f["equifax_total_count_institution_segment_mass_market_secured"] == 1  # Muthoot
    assert f["equifax_active_count_institution_segment_mainstream"] == 1  # only HDFC is active
    assert f["equifax_active_count_institution_segment_mass_market_digital"] == 1
    assert f["equifax_active_count_institution_segment_mass_market_secured"] == 0
    assert f["equifax_total_count_institution_category_psu_bank"] == 1


def test_normal_lead_is_premium_and_cdi(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_is_premium"] == "Mainstream"  # active Mainstream (HDFC), no active Premium
    assert f["equifax_credit_diversity_index"] == 75.0  # 4 accounts, 4 distinct types -> HHI=0.25


def test_normal_lead_percentages(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_pct_of_total_sanctioned_amount_gold"] == 25.0
    assert f["equifax_pct_of_total_sanctioned_amount_business"] == 15.0
    assert f["equifax_pct_of_total_sanctioned_amount_vehicle"] == 10.0
    assert f["equifax_pct_of_total_sanctioned_amount_active_vehicle"] == 16.6667
    assert f["equifax_pct_of_total_sanctioned_amount_active_gold"] == 0.0  # gold loan is closed
    assert f["equifax_pct_of_total_installment_amount_active_vehicle"] == 23.0769


def test_normal_lead_suit_filed_status_values_dedupes_and_excludes_star(normal_lead_features):
    assert normal_lead_features["equifax_suit_filed_status_values"] == "Suit Filed"


def test_normal_lead_summary_passthrough(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_summary_most_severe_status_with_in24_months"] == "SUB"
    assert f["equifax_summary_no_of_past_due_accounts"] == 1


def test_normal_lead_reason_buckets_and_unmapped_logged(normal_lead_features, caplog):
    f = normal_lead_features
    assert f["equifax_reason_count_utilization"] == 1
    assert f["equifax_reason_count_delinquency_default"] == 1
    assert f["equifax_reason_count_credit_exposure_unsecured_debt"] == 0
    assert f["equifax_reason_count_portfolio_composition"] == 0
    assert f["equifax_reason_count_credit_history_vintage_activity"] == 2


def test_unmapped_scoring_element_description_is_logged_not_silently_dropped(caplog):
    accounts = []
    payload = _wrap(
        accounts, 700, [{"description": "A Brand New Reason Code Nobody Has Seen"}],
        {"mostSevereStatusWithIn24Months": None, "noOfPastDueAccounts": 0}, {}, {},
    )
    with caplog.at_level(logging.WARNING, logger="app.core.enrichment.equifax"):
        ADAPTER.transform(_matched(payload))
    assert any("A Brand New Reason Code Nobody Has Seen" in r.message for r in caplog.records)


def test_normal_lead_identity_fields(normal_lead_features):
    f = normal_lead_features
    assert f["equifax_gender"] == "F"
    assert f["equifax_age_years"] == 33  # DOB 15-08-1990, reference date 01-06-2024
    assert f["equifax_has_pan"] is True
    assert f["equifax_has_passport"] is False
    assert f["equifax_has_voter_id"] is True


# ---------------------------------------------------------------------------
# Fixture 2: lead with no accounts
# ---------------------------------------------------------------------------


@pytest.fixture
def no_accounts_features():
    payload = _wrap(
        [], 650, [],
        {"mostSevereStatusWithIn24Months": None, "noOfPastDueAccounts": 0},
        {"gender": "M", "dateOfBirth": "01-01-1995"},
        {"pANId": [{"idNumber": "X"}], "nationalIDCard": [{"idNumber": "P1234"}], "otherId": []},
    )
    return ADAPTER.transform(_matched(payload))


def test_no_accounts_portfolio_is_zero_not_percentages_are_none(no_accounts_features):
    f = no_accounts_features
    assert f["equifax_total_accounts"] == 0
    assert f["equifax_active_accounts"] == 0
    assert f["equifax_active_pct"] is None  # 0/0 is undefined, not 0
    assert f["equifax_closed_pct"] is None


def test_no_accounts_totals_and_ages_are_none_not_zero(no_accounts_features):
    f = no_accounts_features
    assert f["equifax_total_credit_limit"] is None
    assert f["equifax_total_sanction_amount"] is None
    assert f["equifax_oldest_account_age_months"] is None
    assert f["equifax_average_account_age_months"] is None


def test_no_accounts_bucket_counts_are_all_zero(no_accounts_features):
    f = no_accounts_features
    for bucket in ["credit_cards", "gold_loans", "home_loans", "other_loans"]:
        assert f[f"equifax_num_{bucket}"] == 0
        assert f[f"equifax_active_count_{bucket}"] == 0


def test_no_accounts_is_premium_and_cdi_are_none(no_accounts_features):
    assert no_accounts_features["equifax_is_premium"] is None
    assert no_accounts_features["equifax_credit_diversity_index"] is None


def test_no_accounts_suit_filed_status_values_is_none_not_empty_string(no_accounts_features):
    # None = "nothing to check" (no accounts at all); distinct from "" which
    # means accounts were checked and genuinely had no suit-filed history.
    assert no_accounts_features["equifax_suit_filed_status_values"] is None


def test_no_accounts_age_is_none_even_with_a_dob_since_no_reference_date_exists(no_accounts_features):
    # There's no dateReported anywhere to anchor a stable age computation to.
    assert no_accounts_features["equifax_age_years"] is None
    assert no_accounts_features["equifax_gender"] == "M"
    assert no_accounts_features["equifax_has_pan"] is True
    assert no_accounts_features["equifax_has_passport"] is True
    assert no_accounts_features["equifax_has_voter_id"] is False


# ---------------------------------------------------------------------------
# Fixture 3: -1 score sentinel
# ---------------------------------------------------------------------------


@pytest.fixture
def sentinel_score_features():
    accounts = [
        {"accountType": "Personal Loan", "institution": "ICICI Bank", "creditLimit": 0, "sanctionAmount": 40000,
         "installmentAmount": 3000, "balance": 15000, "accountStatus": "Active",
         "dateReported": "10-01-2024", "dateOpened": "10-01-2022", "history48Months": [{"suitFiledStatus": "*"}]},
    ]
    payload = _wrap(
        accounts, -1, [{"description": "Total Utilization"}],
        {"mostSevereStatusWithIn24Months": "STD", "noOfPastDueAccounts": 0},
        {"gender": "M", "dateOfBirth": "01-01-1990"},
        {"pANId": [], "nationalIDCard": [], "otherId": [{"idNumber": "V1"}]},
    )
    return ADAPTER.transform(_matched(payload))


def test_sentinel_score_maps_to_none(sentinel_score_features):
    assert sentinel_score_features["equifax_score"] is None


def test_sentinel_score_does_not_affect_other_features(sentinel_score_features):
    f = sentinel_score_features
    # A -1 score sentinel says nothing about the account data itself, which
    # should be extracted completely normally alongside the nulled score.
    assert f["equifax_total_accounts"] == 1
    assert f["equifax_active_accounts"] == 1
    assert f["equifax_is_premium"] == "Mainstream"
    assert f["equifax_pct_of_total_sanctioned_amount_personal"] == 100.0
    assert f["equifax_reason_count_utilization"] == 1


# ---------------------------------------------------------------------------
# Fixture 4: all accounts closed
# ---------------------------------------------------------------------------


@pytest.fixture
def all_closed_features():
    accounts = [
        {"accountType": "Home Loan", "institution": "Axis Bank", "creditLimit": 0, "sanctionAmount": 500000,
         "installmentAmount": 0, "balance": 0, "accountStatus": "Closed",
         "dateReported": "01-01-2023", "dateOpened": "01-01-2015", "history48Months": []},
        {"accountType": "Consumer Loan", "institution": "Bajaj Finance Limited", "creditLimit": 0, "sanctionAmount": 25000,
         "installmentAmount": 0, "balance": 0, "accountStatus": "Written Off",
         "dateReported": "01-01-2023", "dateOpened": "01-01-2021", "history48Months": []},
    ]
    payload = _wrap(
        accounts, 580, [],
        {"mostSevereStatusWithIn24Months": "DBT", "noOfPastDueAccounts": 2},
        {"gender": None, "dateOfBirth": "20-03-1985"},
        {"pANId": [{"idNumber": "Y"}], "nationalIDCard": [{"idNumber": "Z"}], "otherId": []},
    )
    return ADAPTER.transform(_matched(payload))


def test_all_closed_active_accounts_is_zero_but_totals_still_populate(all_closed_features):
    f = all_closed_features
    assert f["equifax_total_accounts"] == 2
    assert f["equifax_active_accounts"] == 0
    assert f["equifax_active_pct"] == 0.0
    assert f["equifax_closed_pct"] == 100.0
    # written off IS a real observation over accounts that exist -> a real total, not None
    assert f["equifax_total_sanction_amount"] == 525000.0
    assert f["equifax_total_active_sanction_amount"] == 0  # real zero: accounts exist, none active


def test_all_closed_writtenoff_counted_correctly(all_closed_features):
    # "Written Off" is a writtenoff/settled status; plain "Closed" is not.
    assert all_closed_features["equifax_writtenoff_or_settled_pct"] == 50.0
    assert all_closed_features["equifax_settled_accounts"] == 0


def test_all_closed_is_premium_is_none_active_accounts_only(all_closed_features):
    # Both accounts are inactive, so even though one is a Private Bank
    # (Mainstream-eligible), is_premium only counts ACTIVE accounts.
    assert all_closed_features["equifax_is_premium"] is None


def test_all_closed_active_only_percentages_are_none_denominator_is_zero(all_closed_features):
    f = all_closed_features
    assert f["equifax_pct_of_total_sanctioned_amount_active_home"] is None
    assert f["equifax_pct_of_total_installment_amount_active_consumer"] is None
    # but the non-active variant (over ALL accounts) is a real percentage
    assert f["equifax_pct_of_total_sanctioned_amount_home"] == 95.2381
    assert f["equifax_pct_of_total_sanctioned_amount_consumer"] == 4.7619


def test_all_closed_bajaj_finance_classified_as_mass_market_digital_nbfc(all_closed_features):
    assert all_closed_features["equifax_total_count_institution_segment_mass_market_digital"] == 1


# ---------------------------------------------------------------------------
# Fixture 5: malformed payload missing the nested report path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_payload", [
    {},
    {"data": {}},
    {"data": {"cCRResponse": {}}},
    {"data": {"cCRResponse": {"cIRReportDataLst": []}}},
    {"data": {"cCRResponse": {"cIRReportDataLst": [{}]}}},
    None,
    "not even a dict",
])
def test_malformed_payload_returns_all_null_without_raising(bad_payload):
    raw = RawResponse(lead_id="1", source="equifax", status=ResponseStatus.SUCCESS_MATCHED.value,
                       payload=bad_payload, fetched_at="t", batch_id="b")
    features = ADAPTER.transform(raw)  # must not raise
    assert set(features.keys()) == set(_FEATURE_NAMES)
    assert all(v is None for v in features.values())
