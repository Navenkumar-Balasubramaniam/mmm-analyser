"""
tests/test_eda.py
==================
Goal-framed tests for the exploratory data analysis module.

These tests verify:
  - EDA returns correct structure for UI rendering
  - EDA produces OBSERVATIONAL outputs only (no ROAS, no attribution)
  - EDA handles edge cases gracefully
"""

import pytest
import pandas as pd
import numpy as np
from src.eda import compute_revenue_trend, compute_spend_share, compute_correlations


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """104-week dataset with 3 channels, promo, and holiday flags."""
    np.random.seed(42)
    n = 104
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    t = np.arange(n)
    return pd.DataFrame({
        "date_week": dates,
        "revenue": (100000 + t * 100 + 10000 * np.sin(2 * np.pi * t / 52)
                    + np.random.normal(0, 3000, n)).round(0),
        "spend_search": np.random.randint(2000, 6000, n),
        "spend_social": np.random.randint(500, 3000, n),
        "spend_tv": np.random.randint(3000, 9000, n),
        "promo_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "holiday_flag": np.random.choice([0, 1], n, p=[0.88, 0.12]),
    })


@pytest.fixture
def minimal_df():
    """52-week dataset with only 2 channels, no optional columns."""
    np.random.seed(99)
    n = 52
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    return pd.DataFrame({
        "date_week": dates,
        "revenue": np.random.randint(80000, 140000, n),
        "spend_search": np.random.randint(1000, 5000, n),
        "spend_social": np.random.randint(500, 2500, n),
    })


# ────────────────────────────────────────────────────────────────
# GOAL: Revenue trend returns time series data for charting
# ────────────────────────────────────────────────────────────────

def test_revenue_trend_returns_dates_and_values(sample_df):
    """Must return date and revenue lists of equal length for plotting."""
    result = compute_revenue_trend(sample_df)
    assert "dates" in result
    assert "revenue" in result
    assert len(result["dates"]) == len(result["revenue"])
    assert len(result["dates"]) == len(sample_df)


def test_revenue_trend_includes_quarterly_averages(sample_df):
    """Must return average revenue per quarter for a bar chart."""
    result = compute_revenue_trend(sample_df)
    assert "quarterly_avg" in result
    # Should have entries for each quarter present in data
    assert len(result["quarterly_avg"]) > 0
    # Each entry must have quarter label and value
    for q in result["quarterly_avg"]:
        assert "quarter" in q
        assert "avg_revenue" in q
        assert q["avg_revenue"] > 0


def test_revenue_trend_does_not_contain_regression_coefficients(sample_df):
    """EDA must NOT output trend slopes or R² — that belongs to the model."""
    result = compute_revenue_trend(sample_df)
    forbidden_keys = ["slope", "r_squared", "r2", "p_value", "intercept",
                      "trend_coefficient", "regression"]
    for key in forbidden_keys:
        assert key not in result, f"EDA must not contain '{key}' — model territory"


def test_revenue_trend_does_not_contain_lift_estimates(sample_df):
    """Promo/holiday lift numbers must NOT come from EDA — model only."""
    result = compute_revenue_trend(sample_df)
    forbidden_keys = ["promo_lift", "holiday_lift", "lift"]
    for key in forbidden_keys:
        assert key not in result, f"EDA must not contain '{key}' — model territory"


# ────────────────────────────────────────────────────────────────
# GOAL: Spend share returns correct arithmetic breakdown
# ────────────────────────────────────────────────────────────────

def test_spend_share_returns_all_channels(sample_df):
    """Must return an entry for every spend_ column."""
    result = compute_spend_share(sample_df)
    assert "channels" in result
    channel_names = [c["channel"] for c in result["channels"]]
    assert "spend_search" in channel_names
    assert "spend_social" in channel_names
    assert "spend_tv" in channel_names


def test_spend_share_percentages_sum_to_100(sample_df):
    """Channel shares must sum to 100%."""
    result = compute_spend_share(sample_df)
    total_pct = sum(c["share_pct"] for c in result["channels"])
    assert abs(total_pct - 100.0) < 0.1, f"Shares sum to {total_pct}, expected 100"


def test_spend_share_totals_are_positive(sample_df):
    """Every channel must have a positive total spend."""
    result = compute_spend_share(sample_df)
    for ch in result["channels"]:
        assert ch["total_spend"] > 0
        assert ch["avg_weekly"] > 0


def test_spend_share_includes_grand_total(sample_df):
    """Must return the total spend across all channels."""
    result = compute_spend_share(sample_df)
    assert "total_spend" in result
    assert result["total_spend"] > 0
    # Grand total must equal sum of channel totals
    channel_sum = sum(c["total_spend"] for c in result["channels"])
    assert abs(result["total_spend"] - channel_sum) < 1


# ────────────────────────────────────────────────────────────────
# GOAL: Correlation returns raw signals with proper disclaimers
# ────────────────────────────────────────────────────────────────

def test_correlations_returns_all_channels(sample_df):
    """Must return correlation for every spend column vs revenue."""
    result = compute_correlations(sample_df)
    assert "correlations" in result
    channel_names = [c["channel"] for c in result["correlations"]]
    assert "spend_search" in channel_names
    assert "spend_social" in channel_names
    assert "spend_tv" in channel_names


def test_correlations_have_r_and_pvalue(sample_df):
    """Each correlation must include r value and p-value."""
    result = compute_correlations(sample_df)
    for c in result["correlations"]:
        assert "r" in c
        assert "p_value" in c
        assert -1.0 <= c["r"] <= 1.0
        assert 0.0 <= c["p_value"] <= 1.0


def test_correlations_include_disclaimer(sample_df):
    """Output must include a disclaimer that raw correlation ≠ model ROAS."""
    result = compute_correlations(sample_df)
    assert "disclaimer" in result
    assert len(result["disclaimer"]) > 0


def test_correlations_do_not_contain_roas(sample_df):
    """Correlation output must NOT contain ROAS — that is model territory."""
    result = compute_correlations(sample_df)
    forbidden_keys = ["roas", "contribution", "attribution", "coefficient"]
    for key in forbidden_keys:
        assert key not in result, f"Correlation must not contain '{key}'"
        for c in result["correlations"]:
            assert key not in c, f"Channel correlation must not contain '{key}'"


# ────────────────────────────────────────────────────────────────
# GOAL: EDA works with minimal data (no optional columns)
# ────────────────────────────────────────────────────────────────

def test_revenue_trend_works_without_optional_columns(minimal_df):
    """Must work even without promo_flag and holiday_flag."""
    result = compute_revenue_trend(minimal_df)
    assert len(result["dates"]) == 52


def test_spend_share_works_with_two_channels(minimal_df):
    """Must work with exactly 2 channels."""
    result = compute_spend_share(minimal_df)
    assert len(result["channels"]) == 2


def test_correlations_work_with_two_channels(minimal_df):
    """Must work with exactly 2 channels."""
    result = compute_correlations(minimal_df)
    assert len(result["correlations"]) == 2
