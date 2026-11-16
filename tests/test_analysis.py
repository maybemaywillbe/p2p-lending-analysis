"""Tests use synthetic observations and do not require source workbooks."""

import numpy as np
import pandas as pd
import pytest

from src.borrower_analysis import fit_models
from src.platform_survival import prepare_survival_data, run_analysis


def test_borrower_models_use_model_specific_complete_cases():
    rng = np.random.default_rng(17)
    n = 300
    credit = rng.integers(1, 8, n)
    house = rng.integers(0, 2, n)
    car = rng.integers(0, 2, n)
    education = rng.integers(1, 5, n)
    worktime = rng.integers(1, 5, n)
    probability = 1 / (1 + np.exp(-(-1 + 0.2 * credit - 0.3 * car)))
    default = rng.binomial(1, probability)
    bids = 10 + 2 * credit + 3 * car + rng.normal(0, 2, n)
    frame = pd.DataFrame({
        "DEFAULT": default, "BIDS": bids, "CREDIT": credit,
        "HOUSE": house, "CAR": car, "EDUCATION": education,
        "WORKTIME": worktime,
    })
    frame.loc[0, "BIDS"] = np.nan
    frame.loc[1, "DEFAULT"] = np.nan

    logit, ols, n_logit, n_ols = fit_models(frame)

    assert n_logit == n - 1
    assert n_ols == n - 1
    assert np.isfinite(logit.prsquared)
    assert np.isfinite(ols.rsquared)


def _platform_frame():
    return pd.DataFrame({
        "OnlineTime_YMD": [20140101, 20140101, 20140102, 20140103],
        "Bankrupt_WDZJ": [20140101, 20140105, 20140104, np.nan],
        "Collapse": [1, 1, 1, 0],
        "RegCapital": [100, 200, 300, 400],
        "Joinasso": [1, 0, 1, 0],
    })


def test_platform_duration_preserves_same_day_events_with_common_time_shift():
    result = prepare_survival_data(_platform_frame())

    assert len(result) == 4
    assert result.loc[0, "duration"] == 1
    assert result.loc[0, "event"] == 1
    assert result.loc[1, "duration"] == 5
    assert result.loc[3, "event"] == 0
    assert result.loc[3, "duration"] == 3


def test_platform_data_rejects_negative_duration():
    frame = pd.DataFrame({
        "OnlineTime_YMD": [20140106, 20140101],
        "Bankrupt_WDZJ": [20140105, 20140105],
        "Collapse": [1, 0],
        "RegCapital": [100, 200],
        "Joinasso": [1, 0],
    })

    with pytest.raises(ValueError, match="Negative survival durations"):
        prepare_survival_data(frame)


def test_platform_data_rejects_nonbinary_event_values():
    frame = _platform_frame()
    frame.loc[0, "Collapse"] = 2

    with pytest.raises(ValueError, match="binary 0/1"):
        prepare_survival_data(frame)


def test_survival_analysis_returns_cox_km_logrank_and_ph_diagnostics():
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(2026)
    n = 180
    membership = rng.binomial(1, 0.3, n)
    capital = rng.lognormal(mean=5.0, sigma=0.8, size=n)
    durations = rng.integers(30, 1500, size=n).astype(float)
    events = rng.binomial(1, 0.72, n)
    model_data = pd.DataFrame({
        "duration": durations,
        "event": events,
        "RegCapital": capital,
        "Joinasso": membership,
    })
    assert model_data["event"].nunique() == 2
    assert model_data["Joinasso"].nunique() == 2

    results = run_analysis(model_data)

    assert np.isfinite(results["cox_model"].concordance_index_)
    assert set(results["km_fits"]) == {"Non-member", "Member"}
    assert results["group_summary"]["Platforms"].sum() == n
    assert np.isfinite(results["logrank"].p_value)
    assert {"Joinasso", "RegCapital"}.issubset(
        set(results["ph_test"].summary.index)
    )
