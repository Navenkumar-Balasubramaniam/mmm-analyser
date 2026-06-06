"""
src/optimiser.py
=================
Budget reallocation optimiser using scipy.

Reads ROAS from post_model.py (PyMC posterior) ONLY.
Finds the spend allocation that maximises predicted revenue
under a fixed total budget constraint.

Uses a simplified objective: maximise Σ(ROAS_i × spend_i)
subject to:
  - Σ spend_i = total_budget         (budget-neutral)
  - spend_i >= min_share × total     (minimum per channel)
  - spend_i <= max_share × total     (maximum per channel)

This is a linear program — SLSQP solves it reliably.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Any


def optimise_budget(
    roas: list[dict],
    current_spend: dict[str, float],
    min_share: float = 0.05,
    max_share: float = 0.60,
) -> dict[str, Any]:
    """
    Find the optimal budget allocation that maximises predicted revenue.

    Parameters
    ----------
    roas : list[dict]
        ROAS per channel from extract_model_results(). Each dict has:
        channel, mean, ci_lower, ci_upper.
    current_spend : dict[str, float]
        Current weekly spend per channel (e.g. {"spend_search": 3500}).
    min_share : float
        Minimum fraction of total budget per channel (default 0.05 = 5%).
    max_share : float
        Maximum fraction of total budget per channel (default 0.60 = 60%).

    Returns
    -------
    dict with:
        - allocations        : list[dict] — per channel: current, optimal, change
        - total_budget       : float — total budget (unchanged)
        - predicted_uplift_pct : float — expected revenue increase from reallocation
    """
    # ── Align channels ───────────────────────────────────────────
    channels = [r["channel"] for r in roas]
    roas_values = np.array([r["mean"] for r in roas])
    current_values = np.array([current_spend[ch] for ch in channels])
    total_budget = float(current_values.sum())

    # ── Current predicted revenue (baseline) ─────────────────────
    current_revenue = float(np.sum(roas_values * current_values))

    # ── Objective: maximise revenue = Σ(ROAS_i × spend_i) ───────
    # scipy minimises, so negate the objective
    def neg_revenue(x):
        return -float(np.sum(roas_values * x))

    # ── Constraints ──────────────────────────────────────────────
    # Total spend must equal current total (budget-neutral)
    constraints = [{
        "type": "eq",
        "fun": lambda x: float(np.sum(x) - total_budget),
    }]

    # ── Bounds: each channel between min and max share ───────────
    bounds = [
        (min_share * total_budget, max_share * total_budget)
        for _ in channels
    ]

    # ── Initial guess: current allocation ────────────────────────
    x0 = current_values.copy()

    # ── Solve ────────────────────────────────────────────────────
    result = minimize(
        neg_revenue,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_values = result.x

    # ── Predicted revenue at optimal allocation ──────────────────
    optimal_revenue = float(np.sum(roas_values * optimal_values))
    uplift_pct = ((optimal_revenue - current_revenue) / current_revenue * 100
                  if current_revenue > 0 else 0.0)

    # ── Build output ─────────────────────────────────────────────
    allocations = []
    for i, ch in enumerate(channels):
        allocations.append({
            "channel": ch,
            "current_spend": round(float(current_values[i]), 2),
            "optimal_spend": round(float(optimal_values[i]), 2),
            "change": round(float(optimal_values[i] - current_values[i]), 2),
        })

    return {
        "allocations": allocations,
        "total_budget": round(total_budget, 2),
        "predicted_uplift_pct": round(uplift_pct, 2),
    }
