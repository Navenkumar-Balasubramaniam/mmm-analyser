"""
tests/test_model.py
====================
Goal-framed tests for the Bayesian MMM model.

These tests verify:
  - The model builds and fits without crashing
  - Posteriors have the right shape and no NaN values
  - All spend channels appear in the fitted parameters
  - Works with and without optional control columns

PERFORMANCE NOTE: Tests use 30 rows, 2 channels, 50 draws, 2 chains.
This runs in ~30 seconds. It's enough to verify structure — not
statistical quality. Full-quality runs use config.yaml settings.
"""

import pytest
import pandas as pd
import numpy as np
from src.model import build_and_fit_mmm


# ────────────────────────────────────────────────────────────────
# Fixtures — small fast datasets
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """30-week dataset with 2 channels — fast enough for test fits."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    spend_s = np.random.randint(1000, 5000, n).astype(float)
    spend_t = np.random.randint(2000, 7000, n).astype(float)
    revenue = (50000 + spend_s * 3.0 + spend_t * 1.5
               + np.random.normal(0, 2000, n)).round(0)
    return pd.DataFrame({
        "date_week": dates,
        "revenue": revenue,
        "spend_search": spend_s,
        "spend_tv": spend_t,
    })


@pytest.fixture
def small_df_with_controls():
    """30-week dataset with 2 channels + promo and holiday flags."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    spend_s = np.random.randint(1000, 5000, n).astype(float)
    spend_t = np.random.randint(2000, 7000, n).astype(float)
    promo = np.random.choice([0, 1], n, p=[0.8, 0.2])
    holiday = np.random.choice([0, 1], n, p=[0.9, 0.1])
    revenue = (50000 + spend_s * 3.0 + spend_t * 1.5
               + promo * 10000 + holiday * 8000
               + np.random.normal(0, 2000, n)).round(0)
    return pd.DataFrame({
        "date_week": dates,
        "revenue": revenue,
        "spend_search": spend_s,
        "spend_tv": spend_t,
        "promo_flag": promo,
        "holiday_flag": holiday,
    })


# Minimal model config — fast fits for testing
FAST_CONFIG = {
    "draws": 50,
    "tune": 50,
    "chains": 2,
    "target_accept": 0.85,
    "adstock_max_lag": 4,
}


# ────────────────────────────────────────────────────────────────
# GOAL: Model builds and fits without error
# ────────────────────────────────────────────────────────────────

def test_model_fits_without_error(small_df):
    """The model must build and complete MCMC sampling without crashing."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    assert result is not None
    assert "model" in result
    assert "trace" in result


def test_model_fits_with_control_columns(small_df_with_controls):
    """Model must accept and include promo_flag and holiday_flag."""
    result = build_and_fit_mmm(small_df_with_controls, model_config=FAST_CONFIG)
    assert result is not None
    assert "trace" in result


# ────────────────────────────────────────────────────────────────
# GOAL: Trace contains expected parameters
# ────────────────────────────────────────────────────────────────

def test_trace_contains_channel_params(small_df):
    """Posterior must contain key model parameters and the channel coordinate."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    trace = result["trace"]
    # In pymc-marketing 0.19.x, fit_result is an xarray Dataset (the posterior)
    # It must contain the learned adstock and saturation parameters
    assert "adstock_alpha" in trace.data_vars, "Missing adstock_alpha in posterior"
    assert "saturation_lam" in trace.data_vars, "Missing saturation_lam in posterior"
    assert "intercept_contribution" in trace.data_vars, "Missing intercept_contribution in posterior"
    assert "channel_contribution" in trace.data_vars, "Missing channel_contribution"
    # The 'channel' coordinate must list our spend columns
    assert "channel" in trace.coords
    channel_names = list(trace.coords["channel"].values)
    assert "spend_search" in channel_names
    assert "spend_tv" in channel_names


def test_trace_has_no_nan_values(small_df):
    """No NaN values should exist in the posterior samples."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    trace = result["trace"]
    # trace is the posterior Dataset directly — iterate its variables
    for var_name in trace.data_vars:
        values = trace[var_name].values
        assert not np.any(np.isnan(values)), (
            f"NaN found in posterior variable '{var_name}'"
        )


# ────────────────────────────────────────────────────────────────
# GOAL: Result dict has correct structure for downstream modules
# ────────────────────────────────────────────────────────────────

def test_result_contains_spend_columns_list(small_df):
    """Result must list which spend columns were used in the model."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    assert "spend_columns" in result
    assert "spend_search" in result["spend_columns"]
    assert "spend_tv" in result["spend_columns"]


def test_result_contains_control_columns_list(small_df_with_controls):
    """Result must list which control columns were included."""
    result = build_and_fit_mmm(small_df_with_controls, model_config=FAST_CONFIG)
    assert "control_columns" in result
    assert "promo_flag" in result["control_columns"]
    assert "holiday_flag" in result["control_columns"]


def test_result_with_no_controls_has_empty_control_list(small_df):
    """If no control columns present, control_columns should be empty."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    assert "control_columns" in result
    assert len(result["control_columns"]) == 0


def test_result_contains_input_data(small_df):
    """Result must carry the original dataframe for post-model use."""
    result = build_and_fit_mmm(small_df, model_config=FAST_CONFIG)
    assert "data" in result
    assert len(result["data"]) == len(small_df)
