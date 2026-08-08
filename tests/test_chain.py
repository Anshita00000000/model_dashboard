from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import chain


def _evoke_like_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    levels = ["No Appointment", "Appointment Booked", "Consulted", "Converted"]
    weights = [0.5, 0.25, 0.2, 0.05]
    disposition = rng.choice(levels, size=n, p=weights)
    return pd.DataFrame({"disposition": disposition})


def test_build_funnel_config_k_levels_needs_k_minus_1_stages():
    df = _evoke_like_df()
    funnel = chain.build_funnel_config(
        df, target_column="disposition",
        level_order=["No Appointment", "Appointment Booked", "Consulted", "Converted"],
    )
    assert funnel.n_stages == 3
    assert len(funnel.stage_names) == 3
    assert "4 outcome levels" in chain.describe_funnel_config(funnel)
    assert "3 conditional stage" in chain.describe_funnel_config(funnel)


def test_build_funnel_config_default_stage_names():
    df = _evoke_like_df()
    funnel = chain.build_funnel_config(
        df, target_column="disposition",
        level_order=["No Appointment", "Appointment Booked", "Consulted", "Converted"],
    )
    assert funnel.stage_names == ["reached_appointment_booked", "reached_consulted", "reached_converted"]


def test_build_funnel_config_custom_stage_names():
    df = _evoke_like_df()
    funnel = chain.build_funnel_config(
        df, target_column="disposition",
        level_order=["No Appointment", "Appointment Booked", "Consulted", "Converted"],
        stage_names=["reached_appointment", "reached_consulted", "converted"],
    )
    assert funnel.stage_names == ["reached_appointment", "reached_consulted", "converted"]


def test_build_funnel_config_validates_against_actual_data_values():
    df = _evoke_like_df()
    with pytest.raises(ValueError):
        chain.build_funnel_config(
            df, target_column="disposition",
            level_order=["No Appointment", "Appointment Booked", "Consulted"],  # missing "Converted"
        )


def test_build_funnel_config_rejects_duplicate_levels():
    df = _evoke_like_df()
    with pytest.raises(ValueError):
        chain.build_funnel_config(df, target_column="disposition", level_order=["No Appointment", "No Appointment"])


def test_build_funnel_config_rejects_too_few_levels():
    df = pd.DataFrame({"disposition": ["Converted"] * 10})
    with pytest.raises(ValueError):
        chain.build_funnel_config(df, target_column="disposition", level_order=["Converted"])


def test_build_stage_data_stage_1_trains_on_all_rows():
    df = _evoke_like_df()
    funnel = chain.build_funnel_config(
        df, target_column="disposition",
        level_order=["No Appointment", "Appointment Booked", "Consulted", "Converted"],
    )
    stages = chain.build_stage_data(df, funnel)
    assert stages[0].n_rows == len(df)


def test_build_stage_data_stage_n_trains_only_on_rows_where_stage_n_minus_1_occurred():
    df = _evoke_like_df()
    funnel = chain.build_funnel_config(
        df, target_column="disposition",
        level_order=["No Appointment", "Appointment Booked", "Consulted", "Converted"],
    )
    stages = chain.build_stage_data(df, funnel)

    reached_booked = (df["disposition"] != "No Appointment").sum()
    assert stages[1].n_rows == reached_booked

    reached_consulted = df["disposition"].isin(["Consulted", "Converted"]).sum()
    assert stages[2].n_rows == reached_consulted

    # positives at each stage: progressed past that level
    assert stages[0].n_positive == df["disposition"].isin(["Appointment Booked", "Consulted", "Converted"]).sum()
    assert stages[1].n_positive == df["disposition"].isin(["Consulted", "Converted"]).sum()
    assert stages[2].n_positive == (df["disposition"] == "Converted").sum()


def test_undertrained_warnings():
    stages = [
        chain.StageData(name="stage_a", level_from="x", level_to="y", positions=np.arange(10), y=np.zeros(10, dtype=int), n_rows=10, n_positive=2),
        chain.StageData(name="stage_b", level_from="y", level_to="z", positions=np.arange(200), y=np.zeros(200, dtype=int), n_rows=200, n_positive=100),
    ]
    warnings = chain.undertrained_warnings(stages, min_positive=50)
    assert len(warnings) == 1
    assert "stage_a" in warnings[0]


def test_assert_no_ordinal_violations_passes_for_valid_cumulative_probs():
    stage_1 = np.array([0.9, 0.5, 0.1])
    stage_2 = stage_1 * np.array([0.8, 0.5, 0.9])  # valid cumulative product, non-increasing
    chain.assert_no_ordinal_violations({"stage_1": stage_1, "stage_2": stage_2}, ["stage_1", "stage_2"])


def test_assert_no_ordinal_violations_catches_increase():
    stage_1 = np.array([0.5, 0.5])
    stage_2 = np.array([0.6, 0.4])  # first row INCREASED — a bug (e.g. raw margin instead of probability)
    with pytest.raises(AssertionError):
        chain.assert_no_ordinal_violations({"stage_1": stage_1, "stage_2": stage_2}, ["stage_1", "stage_2"])


def test_assert_no_ordinal_violations_catches_out_of_range():
    stage_1 = np.array([0.5, 1.4])  # invalid probability
    with pytest.raises(AssertionError):
        chain.assert_no_ordinal_violations({"stage_1": stage_1}, ["stage_1"])
