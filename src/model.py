"""
src/model.py
=============
Bayesian Marketing Mix Model using PyMC-Marketing.

This is the SINGLE SOURCE OF TRUTH for all attribution numbers.
Every downstream output (ROAS, contribution, adstock curves,
saturation curves, promo/holiday lift) is extracted from the
posterior of this model — nothing else.

The model specification:
    Revenue(t) = intercept
               + Σ [ saturation( adstock( spend_channel(t) ) ) × β_channel ]
               + β_promo × promo_flag(t)        (if present)
               + β_holiday × holiday_flag(t)     (if present)
               + yearly_seasonality(t)
               + ε(t)

Adstock and saturation parameters are LEARNED from data via
Bayesian inference (NUTS sampler), not hardcoded.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Any, Optional

from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
)


def build_and_fit_mmm(
    df: pd.DataFrame,
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Build and fit a Bayesian MMM on the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe with date_week, revenue, spend_* columns,
        and optionally promo_flag and holiday_flag.
    model_config : dict, optional
        Override default model settings. Keys:
            - draws         : int (default 2000)
            - tune          : int (default 1000)
            - chains        : int (default 4)
            - target_accept : float (default 0.9)
            - adstock_max_lag : int (default 8)

    Returns
    -------
    dict with:
        - model          : MMM — the fitted PyMC-Marketing model object
        - trace          : xarray.Dataset — posterior samples (chain × draw × params)
        - spend_columns  : list[str] — channel columns used
        - control_columns: list[str] — control columns used
        - data           : pd.DataFrame — the input data (for post-model use)
    """
    # ── Merge config with defaults ───────────────────────────────
    defaults = {
        "draws": 2000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
        "adstock_max_lag": 8,
    }
    cfg = {**defaults, **(model_config or {})}

    # ── Identify columns ─────────────────────────────────────────
    spend_columns = sorted([c for c in df.columns if c.startswith("spend_")])
    control_columns = [
        c for c in ["promo_flag", "holiday_flag"]
        if c in df.columns
    ]

    # ── Prepare data ─────────────────────────────────────────────
    # Ensure date column is datetime and data is sorted
    df = df.copy()
    df["date_week"] = pd.to_datetime(df["date_week"])
    df = df.sort_values("date_week").reset_index(drop=True)

    # Ensure spend columns are float (PyMC requirement)
    for col in spend_columns:
        df[col] = df[col].astype(float)

    # Ensure control columns are float
    for col in control_columns:
        df[col] = df[col].astype(float)

    # Revenue as float
    df["revenue"] = df["revenue"].astype(float)

    # ── Build the model ──────────────────────────────────────────
    # GeometricAdstock: learns the decay rate per channel from data
    #   l_max = maximum lag (weeks of carry-over to consider)
    #
    # LogisticSaturation: learns the saturation curve per channel
    #   Models diminishing returns — first dollar returns more than last
    #
    # Both transforms' parameters are inferred via MCMC, not hardcoded.

    mmm = MMM(
        date_column="date_week",
        channel_columns=spend_columns,
        control_columns=control_columns if control_columns else None,
        adstock=GeometricAdstock(l_max=cfg["adstock_max_lag"]),
        saturation=LogisticSaturation(),
    )

    # ── Prepare X and y ──────────────────────────────────────────
    # X contains date + channels + controls
    x_cols = ["date_week"] + spend_columns + control_columns
    X = df[x_cols]
    # The multidimensional MMM expects y to be named 'y' internally —
    # it converts the Series to a DataFrame and looks up column 'y'
    y = df["revenue"].rename("y")

    # ── Fit the model ────────────────────────────────────────────
    # Suppress PyMC sampling progress bars in non-interactive contexts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm.fit(
            X=X,
            y=y,
            draws=cfg["draws"],
            tune=cfg["tune"],
            chains=cfg["chains"],
            target_accept=cfg["target_accept"],
            random_seed=42,
        )

    # ── Extract trace ────────────────────────────────────────────
    # In pymc-marketing 0.19.x, fit_result returns an xarray Dataset
    # containing the posterior samples directly (not an ArviZ InferenceData).
    # Variables include: adstock_alpha, saturation_lam, saturation_beta,
    # intercept, channel_contribution, channel_contribution_original_scale.
    trace = mmm.fit_result

    return {
        "model": mmm,
        "trace": trace,
        "spend_columns": spend_columns,
        "control_columns": control_columns,
        "data": df,
    }
