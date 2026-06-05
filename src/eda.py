"""
src/eda.py
===========
Exploratory data analysis — observational summaries only.

This module runs AFTER validation, BEFORE the model.
It produces three outputs:
  1. Revenue trend — time series + quarterly averages (visual only)
  2. Spend share — arithmetic breakdown per channel
  3. Correlations — raw Pearson r per channel vs revenue

DESIGN RULE: This module must NEVER output ROAS, attribution,
regression coefficients, or lift estimates. Those belong to the
PyMC model (src/model.py) and post-model extraction (src/post_model.py).
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Any


def compute_revenue_trend(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract revenue time series and quarterly averages for charting.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe with 'date_week' and 'revenue'.

    Returns
    -------
    dict with:
        - dates           : list[str] — date strings for x-axis
        - revenue         : list[float] — revenue values for y-axis
        - quarterly_avg   : list[dict] — {quarter: str, avg_revenue: float}
    """
    df = df.sort_values("date_week").reset_index(drop=True)

    # ── Time series for line chart ───────────────────────────────
    dates = [str(d.date()) for d in df["date_week"]]
    revenue = df["revenue"].tolist()

    # ── Quarterly averages for bar chart ─────────────────────────
    df_q = df.copy()
    df_q["year"] = df_q["date_week"].dt.year
    df_q["quarter"] = df_q["date_week"].dt.quarter

    quarterly = (
        df_q.groupby(["year", "quarter"])["revenue"]
        .mean()
        .reset_index()
    )
    quarterly_avg = [
        {
            "quarter": f"{int(row['year'])} Q{int(row['quarter'])}",
            "avg_revenue": round(float(row["revenue"]), 2),
        }
        for _, row in quarterly.iterrows()
    ]

    return {
        "dates": dates,
        "revenue": revenue,
        "quarterly_avg": quarterly_avg,
    }


def compute_spend_share(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate total and percentage spend per channel.

    This is pure arithmetic — no model, no attribution.
    Answers: "How is the budget currently allocated?"

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe with spend_* columns.

    Returns
    -------
    dict with:
        - channels    : list[dict] — per channel: channel, total_spend, avg_weekly, share_pct
        - total_spend : float — grand total across all channels
    """
    spend_cols = sorted([c for c in df.columns if c.startswith("spend_")])

    # ── Per-channel totals ───────────────────────────────────────
    channel_totals = []
    for col in spend_cols:
        total = float(df[col].sum())
        avg = float(df[col].mean())
        channel_totals.append({
            "channel": col,
            "total_spend": round(total, 2),
            "avg_weekly": round(avg, 2),
        })

    # ── Grand total and shares ───────────────────────────────────
    grand_total = sum(c["total_spend"] for c in channel_totals)

    for ch in channel_totals:
        ch["share_pct"] = round(
            (ch["total_spend"] / grand_total * 100) if grand_total > 0 else 0,
            2,
        )

    return {
        "channels": channel_totals,
        "total_spend": round(grand_total, 2),
    }


def compute_correlations(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute Pearson correlation between each spend column and revenue.

    These are RAW correlations — they do NOT account for adstock or
    saturation. A low correlation does NOT mean the channel is ineffective.
    The disclaimer is included in the output for the UI to display.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe with 'revenue' and spend_* columns.

    Returns
    -------
    dict with:
        - correlations : list[dict] — per channel: channel, r, p_value, significant
        - disclaimer   : str — warning that raw correlation ≠ model ROAS
    """
    spend_cols = sorted([c for c in df.columns if c.startswith("spend_")])
    correlations = []

    for col in spend_cols:
        r_val, p_val = stats.pearsonr(df[col], df["revenue"])
        correlations.append({
            "channel": col,
            "r": round(float(r_val), 4),
            "p_value": round(float(p_val), 4),
            "significant": p_val < 0.05,
        })

    disclaimer = (
        "These correlations show the raw linear relationship between each "
        "channel's spend and revenue. They do NOT account for carry-over "
        "(adstock) or diminishing returns (saturation). A low or non-significant "
        "correlation does not mean the channel is ineffective — the Bayesian "
        "model corrects for these effects. See model results for true "
        "channel contributions."
    )

    return {
        "correlations": correlations,
        "disclaimer": disclaimer,
    }
