"""
src/post_model.py
==================
Extract all model outputs from the PyMC posterior.

DESIGN RULE: Every number in this module comes from the trace
(xarray Dataset returned by mmm.fit_result). No OLS, no
hardcoded values, no heuristics. This is what makes the pipeline
consistent end to end.

Posterior variables used (from multidimensional MMM 0.19.x):
  - channel_contribution          (chain, draw, date, channel) — normalized
  - intercept_contribution        (chain, draw) — base revenue, normalized
  - adstock_alpha                 (chain, draw, channel) — learned decay [0,1]
  - saturation_lam                (chain, draw, channel) — saturation lambda
  - saturation_beta               (chain, draw, channel) — saturation beta
  - total_media_contribution_original_scale (chain, draw) — total media $ effect
"""

import numpy as np
from typing import Any

# Credible interval bounds (94% CI — Bayesian standard)
CI_LOWER = 3
CI_UPPER = 97


def extract_model_results(model_result: dict) -> dict[str, Any]:
    """
    Extract all attribution outputs from the fitted PyMC model.

    Parameters
    ----------
    model_result : dict
        Output from build_and_fit_mmm(), containing:
        - trace          : xarray.Dataset (posterior samples)
        - spend_columns  : list[str]
        - data           : pd.DataFrame (original input data)
        - model          : MMM object

    Returns
    -------
    dict with:
        - channel_contributions : list[dict] — per channel: mean, CI, share_pct
        - roas                  : list[dict] — per channel: mean, CI
        - base_revenue          : dict — mean, CI for intercept
        - adstock_decay         : list[dict] — per channel: learned decay rate
        - saturation_params     : list[dict] — per channel: lam and beta
    """
    trace = model_result["trace"]
    spend_columns = model_result["spend_columns"]
    df = model_result["data"]

    return {
        "channel_contributions": _extract_contributions(trace, spend_columns, df),
        "roas": _extract_roas(trace, spend_columns, df),
        "base_revenue": _extract_base_revenue(trace, df),
        "adstock_decay": _extract_adstock(trace, spend_columns),
        "saturation_params": _extract_saturation(trace, spend_columns),
    }


def _extract_contributions(
    trace, spend_columns: list[str], df
) -> list[dict[str, Any]]:
    """
    Compute channel contribution as % of total modelled revenue.

    Method:
      1. Sum channel_contribution across dates → per-channel total (normalized)
      2. Compute each channel's share of (intercept + all channels)
      3. Convert to $ using actual total revenue
      4. Report mean and 94% credible interval
    """
    # channel_contribution: (chain, draw, date, channel) — sum over dates
    channel_total = trace["channel_contribution"].sum(dim="date")  # (chain, draw, channel)
    intercept = trace["intercept_contribution"]  # (chain, draw)

    # Total modelled output per sample = intercept + sum of all channels
    total_modelled = channel_total.sum(dim="channel") + intercept  # (chain, draw)

    actual_total_revenue = float(df["revenue"].sum())
    contributions = []

    for col in spend_columns:
        # Per-sample share for this channel
        ch_samples = channel_total.sel(channel=col)  # (chain, draw)
        share_samples = ch_samples / total_modelled  # (chain, draw)
        share_flat = share_samples.values.flatten()

        # Convert share to $ revenue
        revenue_flat = share_flat * actual_total_revenue

        contributions.append({
            "channel": col,
            "mean": round(float(np.mean(revenue_flat)), 2),
            "ci_lower": round(float(np.percentile(revenue_flat, CI_LOWER)), 2),
            "ci_upper": round(float(np.percentile(revenue_flat, CI_UPPER)), 2),
            "share_pct": round(float(np.mean(share_flat) * 100), 2),
        })

    return contributions


def _extract_roas(
    trace, spend_columns: list[str], df
) -> list[dict[str, Any]]:
    """
    Compute ROAS (Return on Ad Spend) per channel from posterior.

    Method:
      ROAS = channel_revenue_attributed / total_channel_spend
      Both computed from posterior samples → gives a distribution of ROAS.
    """
    channel_total = trace["channel_contribution"].sum(dim="date")
    intercept = trace["intercept_contribution"]
    total_modelled = channel_total.sum(dim="channel") + intercept

    actual_total_revenue = float(df["revenue"].sum())
    roas_list = []

    for col in spend_columns:
        total_spend = float(df[col].sum())
        if total_spend == 0:
            roas_list.append({
                "channel": col,
                "mean": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            })
            continue

        # Revenue attributed to this channel per posterior sample
        ch_samples = channel_total.sel(channel=col)
        share_samples = ch_samples / total_modelled
        revenue_samples = share_samples.values.flatten() * actual_total_revenue

        # ROAS = attributed revenue / spend
        roas_samples = revenue_samples / total_spend

        roas_list.append({
            "channel": col,
            "mean": round(float(np.mean(roas_samples)), 4),
            "ci_lower": round(float(np.percentile(roas_samples, CI_LOWER)), 4),
            "ci_upper": round(float(np.percentile(roas_samples, CI_UPPER)), 4),
        })

    return roas_list


def _extract_base_revenue(trace, df) -> dict[str, Any]:
    """
    Extract base revenue (intercept contribution) from posterior.

    Base revenue = revenue the business earns with zero media spend.
    Represents brand strength, organic traffic, repeat customers.
    """
    channel_total = trace["channel_contribution"].sum(dim="date")
    intercept = trace["intercept_contribution"]
    total_modelled = channel_total.sum(dim="channel") + intercept

    actual_total_revenue = float(df["revenue"].sum())

    # Intercept share per sample
    base_share = intercept / total_modelled
    base_revenue = base_share.values.flatten() * actual_total_revenue

    return {
        "mean": round(float(np.mean(base_revenue)), 2),
        "ci_lower": round(float(np.percentile(base_revenue, CI_LOWER)), 2),
        "ci_upper": round(float(np.percentile(base_revenue, CI_UPPER)), 2),
        "share_pct": round(float(np.mean(base_share.values.flatten()) * 100), 2),
    }


def _extract_adstock(
    trace, spend_columns: list[str]
) -> list[dict[str, Any]]:
    """
    Extract learned adstock decay rates from posterior.

    Decay rate (alpha) is in [0, 1]:
      - High (e.g. 0.7) = long memory (TV, OOH)
      - Low  (e.g. 0.2) = near-instant (search, email)
    """
    adstock_list = []

    for col in spend_columns:
        samples = trace["adstock_alpha"].sel(channel=col).values.flatten()
        adstock_list.append({
            "channel": col,
            "mean": round(float(np.mean(samples)), 4),
            "ci_lower": round(float(np.percentile(samples, CI_LOWER)), 4),
            "ci_upper": round(float(np.percentile(samples, CI_UPPER)), 4),
        })

    return adstock_list


def _extract_saturation(
    trace, spend_columns: list[str]
) -> list[dict[str, Any]]:
    """
    Extract learned saturation curve parameters from posterior.

    Two parameters per channel:
      - lam (lambda): controls the steepness of the saturation curve
      - beta: scales the saturated output
    """
    sat_list = []

    for col in spend_columns:
        lam_samples = trace["saturation_lam"].sel(channel=col).values.flatten()
        beta_samples = trace["saturation_beta"].sel(channel=col).values.flatten()

        sat_list.append({
            "channel": col,
            "lam_mean": round(float(np.mean(lam_samples)), 4),
            "lam_ci_lower": round(float(np.percentile(lam_samples, CI_LOWER)), 4),
            "lam_ci_upper": round(float(np.percentile(lam_samples, CI_UPPER)), 4),
            "beta_mean": round(float(np.mean(beta_samples)), 4),
            "beta_ci_lower": round(float(np.percentile(beta_samples, CI_LOWER)), 4),
            "beta_ci_upper": round(float(np.percentile(beta_samples, CI_UPPER)), 4),
        })

    return sat_list
