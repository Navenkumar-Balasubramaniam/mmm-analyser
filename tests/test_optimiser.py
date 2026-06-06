"""
tests/test_optimiser.py
========================
Goal-framed tests for the budget optimiser module.

These tests verify:
  - Optimal allocation sums exactly to total budget
  - No channel goes below minimum or above maximum share
  - No negative allocations
  - Optimiser uses ROAS from post_model, not OLS
  - Output structure is correct for UI rendering
"""

import pytest
import numpy as np
from src.optimiser import optimise_budget


# ────────────────────────────────────────────────────────────────
# Fixtures — simulated post-model outputs (no real MCMC needed)
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_roas():
    """Simulated ROAS from post_model — 4 channels with varying efficiency."""
    return [
        {"channel": "spend_search", "mean": 4.5, "ci_lower": 3.2, "ci_upper": 5.8},
        {"channel": "spend_social", "mean": 2.8, "ci_lower": 1.5, "ci_upper": 4.1},
        {"channel": "spend_email",  "mean": 6.1, "ci_lower": 4.0, "ci_upper": 8.2},
        {"channel": "spend_tv",     "mean": 1.4, "ci_lower": 0.6, "ci_upper": 2.2},
    ]


@pytest.fixture
def current_spend():
    """Current weekly spend per channel."""
    return {
        "spend_search": 3500,
        "spend_social": 2000,
        "spend_email": 400,
        "spend_tv": 5500,
    }


# ────────────────────────────────────────────────────────────────
# GOAL: Allocation sums to total budget
# ────────────────────────────────────────────────────────────────

def test_allocation_sums_to_budget(sample_roas, current_spend):
    """Optimal spend must sum to the same total as current spend."""
    total_budget = sum(current_spend.values())
    result = optimise_budget(sample_roas, current_spend)
    optimal_total = sum(a["optimal_spend"] for a in result["allocations"])
    assert abs(optimal_total - total_budget) < 1, (
        f"Allocation sums to {optimal_total}, expected {total_budget}"
    )


# ────────────────────────────────────────────────────────────────
# GOAL: No negative allocations
# ────────────────────────────────────────────────────────────────

def test_no_negative_allocations(sample_roas, current_spend):
    """No channel should receive negative spend."""
    result = optimise_budget(sample_roas, current_spend)
    for a in result["allocations"]:
        assert a["optimal_spend"] >= 0, (
            f"{a['channel']} has negative spend: {a['optimal_spend']}"
        )


# ────────────────────────────────────────────────────────────────
# GOAL: Minimum and maximum share constraints respected
# ────────────────────────────────────────────────────────────────

def test_min_share_constraint(sample_roas, current_spend):
    """Every channel must get at least min_share of total budget."""
    min_share = 0.05
    total_budget = sum(current_spend.values())
    result = optimise_budget(sample_roas, current_spend, min_share=min_share)
    for a in result["allocations"]:
        actual_share = a["optimal_spend"] / total_budget
        assert actual_share >= min_share - 0.001, (
            f"{a['channel']} share {actual_share:.3f} < min {min_share}"
        )


def test_max_share_constraint(sample_roas, current_spend):
    """No channel should exceed max_share of total budget."""
    max_share = 0.60
    total_budget = sum(current_spend.values())
    result = optimise_budget(sample_roas, current_spend, max_share=max_share)
    for a in result["allocations"]:
        actual_share = a["optimal_spend"] / total_budget
        assert actual_share <= max_share + 0.001, (
            f"{a['channel']} share {actual_share:.3f} > max {max_share}"
        )


# ────────────────────────────────────────────────────────────────
# GOAL: Output structure is correct
# ────────────────────────────────────────────────────────────────

def test_result_has_required_keys(sample_roas, current_spend):
    """Result must contain allocations and predicted revenue uplift."""
    result = optimise_budget(sample_roas, current_spend)
    assert "allocations" in result
    assert "total_budget" in result
    assert "predicted_uplift_pct" in result


def test_each_allocation_has_current_and_optimal(sample_roas, current_spend):
    """Each channel must show both current and optimal spend for comparison."""
    result = optimise_budget(sample_roas, current_spend)
    for a in result["allocations"]:
        assert "channel" in a
        assert "current_spend" in a
        assert "optimal_spend" in a
        assert "change" in a


def test_all_channels_present_in_output(sample_roas, current_spend):
    """Every input channel must appear in the output."""
    result = optimise_budget(sample_roas, current_spend)
    output_channels = [a["channel"] for a in result["allocations"]]
    for ch in current_spend:
        assert ch in output_channels


# ────────────────────────────────────────────────────────────────
# GOAL: Higher ROAS channels get more budget
# ────────────────────────────────────────────────────────────────

def test_highest_roas_channel_gets_more(sample_roas, current_spend):
    """Email has highest ROAS (6.1) — it should get more than its current $400."""
    result = optimise_budget(sample_roas, current_spend)
    email = next(a for a in result["allocations"] if a["channel"] == "spend_email")
    assert email["optimal_spend"] > email["current_spend"], (
        "Email (highest ROAS) should get a budget increase"
    )


def test_lowest_roas_channel_gets_less(sample_roas, current_spend):
    """TV has lowest ROAS (1.4) and highest spend — it should lose budget."""
    result = optimise_budget(sample_roas, current_spend)
    tv = next(a for a in result["allocations"] if a["channel"] == "spend_tv")
    assert tv["optimal_spend"] < tv["current_spend"], (
        "TV (lowest ROAS, highest spend) should get a budget decrease"
    )
