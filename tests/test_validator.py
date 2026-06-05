"""
tests/test_validator.py
========================
Goal-framed tests for the data validator module.

These tests define what "valid CSV" means for our MMM pipeline.
Each test answers a business question:
  - Can we catch a bad file before it wastes 5 mins of model compute?
  - Do we give the user a clear, actionable error message?
  - Do we let clean data through without false alarms?
"""

import pytest
import pandas as pd
import numpy as np
from src.validator import validate_csv, compute_descriptive_stats


# ────────────────────────────────────────────────────────────────
# Fixtures — reusable test data
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """A minimal valid CSV with 52 weeks, 2 channels, and no issues."""
    dates = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    np.random.seed(42)
    return pd.DataFrame({
        "date_week": dates,
        "revenue": np.random.randint(80000, 150000, 52),
        "spend_search": np.random.randint(1000, 5000, 52),
        "spend_social": np.random.randint(500, 3000, 52),
    })


@pytest.fixture
def full_df():
    """A valid CSV with all optional columns included."""
    dates = pd.date_range("2023-01-02", periods=104, freq="W-MON")
    np.random.seed(42)
    n = 104
    return pd.DataFrame({
        "date_week": dates,
        "revenue": np.random.randint(80000, 150000, n),
        "spend_search": np.random.randint(1000, 5000, n),
        "spend_social": np.random.randint(500, 3000, n),
        "spend_tv": np.random.randint(3000, 8000, n),
        "imp_search": np.random.randint(100000, 500000, n),
        "promo_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "holiday_flag": np.random.choice([0, 1], n, p=[0.88, 0.12]),
    })


# ────────────────────────────────────────────────────────────────
# GOAL: Clean data must pass validation without errors
# ────────────────────────────────────────────────────────────────

def test_clean_csv_passes(clean_df):
    """A well-formed CSV with no issues should pass with zero errors."""
    result = validate_csv(clean_df)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0


def test_full_csv_with_optional_columns_passes(full_df):
    """A CSV with all optional columns (imp_*, promo_flag, holiday_flag) should pass."""
    result = validate_csv(full_df)
    assert result["is_valid"] is True


# ────────────────────────────────────────────────────────────────
# GOAL: Missing required columns must be caught
# ────────────────────────────────────────────────────────────────

def test_missing_date_column_fails(clean_df):
    """If date_week is missing, validation must fail with a clear message."""
    df = clean_df.drop(columns=["date_week"])
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("date_week" in e for e in result["errors"])


def test_missing_revenue_column_fails(clean_df):
    """If revenue is missing, validation must fail."""
    df = clean_df.drop(columns=["revenue"])
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("revenue" in e for e in result["errors"])


def test_fewer_than_two_spend_columns_fails(clean_df):
    """MMM needs at least 2 channels. One spend column must fail."""
    df = clean_df.drop(columns=["spend_social"])
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("spend" in e.lower() for e in result["errors"])


# ────────────────────────────────────────────────────────────────
# GOAL: Bad data values must be caught
# ────────────────────────────────────────────────────────────────

def test_negative_spend_detected(clean_df):
    """Negative spend values are invalid — must be caught."""
    df = clean_df.copy()
    df.loc[5, "spend_search"] = -500
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("negative" in e.lower() for e in result["errors"])


def test_null_values_in_revenue_detected(clean_df):
    """Missing revenue values must be caught."""
    df = clean_df.copy()
    df.loc[10, "revenue"] = np.nan
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("missing" in e.lower() or "null" in e.lower() for e in result["errors"])


def test_null_values_in_spend_detected(clean_df):
    """Missing spend values must be caught."""
    df = clean_df.copy()
    df.loc[3, "spend_search"] = np.nan
    result = validate_csv(df)
    assert result["is_valid"] is False


# ────────────────────────────────────────────────────────────────
# GOAL: Date issues must be caught
# ────────────────────────────────────────────────────────────────

def test_non_weekly_frequency_detected(clean_df):
    """If dates are not 7 days apart, validation must flag it."""
    df = clean_df.copy()
    # Create a 14-day gap by removing a row
    df = df.drop(index=5).reset_index(drop=True)
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("gap" in e.lower() or "frequency" in e.lower() for e in result["errors"])


def test_duplicate_dates_detected(clean_df):
    """Duplicate date entries must be caught."""
    df = clean_df.copy()
    df.loc[10, "date_week"] = df.loc[9, "date_week"]
    result = validate_csv(df)
    assert result["is_valid"] is False
    assert any("duplicate" in e.lower() for e in result["errors"])


# ────────────────────────────────────────────────────────────────
# GOAL: Insufficient data must be caught
# ────────────────────────────────────────────────────────────────

def test_too_few_rows_fails():
    """Fewer than the minimum required rows must fail."""
    dates = pd.date_range("2023-01-02", periods=20, freq="W-MON")
    df = pd.DataFrame({
        "date_week": dates,
        "revenue": np.random.randint(80000, 150000, 20),
        "spend_search": np.random.randint(1000, 5000, 20),
        "spend_social": np.random.randint(500, 3000, 20),
    })
    result = validate_csv(df, min_rows=52)
    assert result["is_valid"] is False
    assert any("row" in e.lower() or "minimum" in e.lower() for e in result["errors"])


# ────────────────────────────────────────────────────────────────
# GOAL: Warnings should flag soft issues without blocking
# ────────────────────────────────────────────────────────────────

def test_high_cv_channel_generates_warning(clean_df):
    """A channel with very volatile spend should produce a warning, not an error."""
    df = clean_df.copy()
    # Make spend_social extremely volatile
    df["spend_social"] = np.where(
        np.arange(52) % 2 == 0, 10, 15000
    )
    result = validate_csv(df)
    # Should still be valid — but with a warning
    assert result["is_valid"] is True
    assert len(result["warnings"]) > 0
    assert any("volatile" in w.lower() or "cv" in w.lower() for w in result["warnings"])


def test_high_zero_spend_weeks_generates_warning(clean_df):
    """If >30% of weeks have zero spend for a channel, warn the user."""
    df = clean_df.copy()
    df.loc[:20, "spend_social"] = 0  # 21 out of 52 weeks = 40%
    result = validate_csv(df)
    assert result["is_valid"] is True
    assert len(result["warnings"]) > 0
    assert any("zero" in w.lower() for w in result["warnings"])


# ────────────────────────────────────────────────────────────────
# GOAL: Descriptive stats must return correct structure
# ────────────────────────────────────────────────────────────────

def test_descriptive_stats_returns_all_columns(clean_df):
    """Stats should be computed for revenue and all spend columns."""
    stats = compute_descriptive_stats(clean_df)
    assert "revenue" in stats
    assert "spend_search" in stats
    assert "spend_social" in stats


def test_descriptive_stats_contains_required_metrics(clean_df):
    """Each column's stats must include mean, std, min, max, and CV."""
    stats = compute_descriptive_stats(clean_df)
    for col in ["revenue", "spend_search"]:
        assert "mean" in stats[col]
        assert "std" in stats[col]
        assert "min" in stats[col]
        assert "max" in stats[col]
        assert "cv" in stats[col]


def test_descriptive_stats_values_are_reasonable(clean_df):
    """Mean must be between min and max. CV must be non-negative."""
    stats = compute_descriptive_stats(clean_df)
    for col in ["revenue", "spend_search"]:
        s = stats[col]
        assert s["min"] <= s["mean"] <= s["max"]
        assert s["cv"] >= 0
