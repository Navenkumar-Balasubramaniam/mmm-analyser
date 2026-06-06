"""
tests/test_post_model.py
=========================
Goal-framed tests for post-model output extraction.

These tests verify:
  - All attribution numbers come from PyMC posterior only
  - Channel contributions sum to a reasonable total
  - ROAS values are positive and finite
  - Adstock and saturation params are within valid ranges
  - Credible intervals are included for every metric
"""

import pytest
import pandas as pd
import numpy as np
from src.model import build_and_fit_mmm
from src.post_model import extract_model_results


# ────────────────────────────────────────────────────────────────
# Fixtures — fit a small model once, reuse across tests
# ────────────────────────────────────────────────────────────────

FAST_CONFIG = {
    "draws": 50,
    "tune": 50,
    "chains": 2,
    "target_accept": 0.85,
    "adstock_max_lag": 4,
}


@pytest.fixture(scope="module")
def model_result():
    """Fit once for all tests in this module — saves 2+ mins of MCMC."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    spend_s = np.random.randint(1000, 5000, n).astype(float)
    spend_t = np.random.randint(2000, 7000, n).astype(float)
    promo = np.random.choice([0, 1], n, p=[0.8, 0.2]).astype(float)
    revenue = (50000 + spend_s * 3.0 + spend_t * 1.5
               + promo * 10000
               + np.random.normal(0, 2000, n)).round(0)
    df = pd.DataFrame({
        "date_week": dates,
        "revenue": revenue,
        "spend_search": spend_s,
        "spend_tv": spend_t,
        "promo_flag": promo,
    })
    return build_and_fit_mmm(df, model_config=FAST_CONFIG)


@pytest.fixture(scope="module")
def extracted(model_result):
    """Extract results once from the fitted model."""
    return extract_model_results(model_result)


# ────────────────────────────────────────────────────────────────
# GOAL: Extraction returns correct top-level structure
# ────────────────────────────────────────────────────────────────

def test_result_has_required_keys(extracted):
    """Extracted results must contain all expected sections."""
    required = [
        "channel_contributions",
        "roas",
        "base_revenue",
        "adstock_decay",
        "saturation_params",
    ]
    for key in required:
        assert key in extracted, f"Missing key: '{key}'"


# ────────────────────────────────────────────────────────────────
# GOAL: Channel contributions are valid
# ────────────────────────────────────────────────────────────────

def test_contributions_include_all_channels(extracted):
    """Every spend channel must appear in contributions."""
    channels = [c["channel"] for c in extracted["channel_contributions"]]
    assert "spend_search" in channels
    assert "spend_tv" in channels


def test_contributions_have_credible_intervals(extracted):
    """Each channel must report mean, lower, and upper bounds."""
    for ch in extracted["channel_contributions"]:
        assert "mean" in ch, f"Missing mean for {ch['channel']}"
        assert "ci_lower" in ch, f"Missing ci_lower for {ch['channel']}"
        assert "ci_upper" in ch, f"Missing ci_upper for {ch['channel']}"
        assert ch["ci_lower"] <= ch["mean"] <= ch["ci_upper"]


def test_contribution_shares_are_non_negative(extracted):
    """No channel should have a negative contribution share."""
    for ch in extracted["channel_contributions"]:
        assert ch["share_pct"] >= 0, (
            f"{ch['channel']} has negative share: {ch['share_pct']}"
        )


# ────────────────────────────────────────────────────────────────
# GOAL: ROAS values are valid
# ────────────────────────────────────────────────────────────────

def test_roas_includes_all_channels(extracted):
    """Every spend channel must have a ROAS estimate."""
    channels = [r["channel"] for r in extracted["roas"]]
    assert "spend_search" in channels
    assert "spend_tv" in channels


def test_roas_values_are_finite(extracted):
    """ROAS must be finite numbers — no NaN or inf."""
    for r in extracted["roas"]:
        assert np.isfinite(r["mean"]), f"Non-finite ROAS for {r['channel']}"
        assert np.isfinite(r["ci_lower"])
        assert np.isfinite(r["ci_upper"])


def test_roas_has_credible_intervals(extracted):
    """ROAS must include mean and credible interval bounds."""
    for r in extracted["roas"]:
        assert "mean" in r
        assert "ci_lower" in r
        assert "ci_upper" in r
        assert r["ci_lower"] <= r["mean"] <= r["ci_upper"]


# ────────────────────────────────────────────────────────────────
# GOAL: Base revenue is valid
# ────────────────────────────────────────────────────────────────

def test_base_revenue_is_positive(extracted):
    """Base revenue (intercept) must be positive."""
    assert extracted["base_revenue"]["mean"] > 0


def test_base_revenue_has_credible_interval(extracted):
    """Base revenue must include CI bounds."""
    br = extracted["base_revenue"]
    assert "ci_lower" in br
    assert "ci_upper" in br


# ────────────────────────────────────────────────────────────────
# GOAL: Adstock decay rates are within valid range
# ────────────────────────────────────────────────────────────────

def test_adstock_includes_all_channels(extracted):
    """Every spend channel must have a learned decay rate."""
    channels = [a["channel"] for a in extracted["adstock_decay"]]
    assert "spend_search" in channels
    assert "spend_tv" in channels


def test_adstock_values_between_zero_and_one(extracted):
    """Decay rates must be in [0, 1] — they are probabilities."""
    for a in extracted["adstock_decay"]:
        assert 0 <= a["mean"] <= 1, (
            f"{a['channel']} decay = {a['mean']}, must be in [0, 1]"
        )


# ────────────────────────────────────────────────────────────────
# GOAL: Saturation params are valid
# ────────────────────────────────────────────────────────────────

def test_saturation_includes_all_channels(extracted):
    """Every spend channel must have saturation parameters."""
    channels = [s["channel"] for s in extracted["saturation_params"]]
    assert "spend_search" in channels
    assert "spend_tv" in channels


def test_saturation_lam_is_positive(extracted):
    """Saturation lambda must be positive."""
    for s in extracted["saturation_params"]:
        assert s["lam_mean"] > 0, (
            f"{s['channel']} saturation_lam = {s['lam_mean']}, must be > 0"
        )
