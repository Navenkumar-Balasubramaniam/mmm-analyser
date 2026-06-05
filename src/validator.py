"""
src/validator.py
=================
Data quality validation and descriptive statistics.

This module runs BEFORE any model. It answers:
  - Is this CSV safe to feed into PyMC-Marketing?
  - What does the data look like at a glance?

Returns structured dicts — never prints, never raises (returns errors in dict).
"""

import pandas as pd
import numpy as np
from typing import Any


def validate_csv(
    df: pd.DataFrame,
    min_rows: int = 52,
    max_cv_warning: float = 1.0,
    max_zero_spend_pct: float = 0.30,
) -> dict[str, Any]:
    """
    Run all data quality checks on an uploaded CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The uploaded data (already read from CSV).
    min_rows : int
        Minimum number of rows required (default: 52 = 1 year weekly).
    max_cv_warning : float
        Coefficient of variation above which a channel triggers a warning.
    max_zero_spend_pct : float
        Fraction of zero-spend weeks above which a channel triggers a warning.

    Returns
    -------
    dict with keys:
        - is_valid : bool — True if no critical errors found
        - errors   : list[str] — critical issues that block the model
        - warnings : list[str] — soft issues the user should know about
        - summary  : dict — row count, date range, column list, spend columns found
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Required columns ─────────────────────────────────────────
    if "date_week" not in df.columns:
        errors.append("Missing required column: 'date_week'.")

    if "revenue" not in df.columns:
        errors.append("Missing required column: 'revenue'.")

    spend_cols = [c for c in df.columns if c.startswith("spend_")]
    if len(spend_cols) < 2:
        errors.append(
            f"Need at least 2 spend columns (found {len(spend_cols)}). "
            f"Columns must start with 'spend_' (e.g. spend_tv, spend_search)."
        )

    # If critical columns are missing, return early — can't check further
    if errors:
        return _build_result(False, errors, warnings, df, spend_cols)

    # ── Parse dates ──────────────────────────────────────────────
    try:
        df["date_week"] = pd.to_datetime(df["date_week"])
    except Exception:
        errors.append("Column 'date_week' could not be parsed as dates. Expected format: YYYY-MM-DD.")
        return _build_result(False, errors, warnings, df, spend_cols)

    # ── Row count ────────────────────────────────────────────────
    if len(df) < min_rows:
        errors.append(
            f"Not enough data: found {len(df)} rows, minimum required is {min_rows}. "
            f"MMM needs at least 1 year of weekly data."
        )

    # ── Missing values ───────────────────────────────────────────
    required_numeric = ["revenue"] + spend_cols
    for col in required_numeric:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            errors.append(
                f"Missing values in '{col}': {n_null} null(s) found. "
                f"Fill or remove these rows before uploading."
            )

    # ── Negative spend ───────────────────────────────────────────
    for col in spend_cols:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            errors.append(
                f"Negative spend in '{col}': {n_neg} value(s). "
                f"Spend must be >= 0."
            )

    # ── Date continuity ──────────────────────────────────────────
    df_sorted = df.sort_values("date_week").reset_index(drop=True)
    date_diffs = df_sorted["date_week"].diff().dropna()

    # Check for duplicates
    n_dup_dates = (date_diffs == pd.Timedelta(0)).sum()
    if n_dup_dates > 0:
        errors.append(
            f"Duplicate dates found: {n_dup_dates} row(s) share the same date_week."
        )

    # Check for non-7-day gaps
    non_weekly = date_diffs[(date_diffs != pd.Timedelta(days=7)) & (date_diffs != pd.Timedelta(0))]
    if len(non_weekly) > 0:
        errors.append(
            f"Date frequency gap detected: {len(non_weekly)} gap(s) are not 7 days apart. "
            f"Data must be weekly."
        )

    # ── Warnings (soft issues) ───────────────────────────────────
    for col in spend_cols:
        if df[col].isnull().any():
            continue  # already caught as error above

        col_mean = df[col].mean()
        col_std = df[col].std()

        # High coefficient of variation
        if col_mean > 0:
            cv = col_std / col_mean
            if cv > max_cv_warning:
                warnings.append(
                    f"High volatility in '{col}': CV = {cv:.2f}. "
                    f"Spend is very inconsistent week-to-week, which may reduce model accuracy."
                )

        # High zero-spend weeks
        zero_pct = (df[col] == 0).sum() / len(df)
        if zero_pct > max_zero_spend_pct:
            warnings.append(
                f"'{col}' has zero spend in {zero_pct:.0%} of weeks. "
                f"Sparse channels produce wider confidence intervals in the model."
            )

    is_valid = len(errors) == 0
    return _build_result(is_valid, errors, warnings, df, spend_cols)


def _build_result(
    is_valid: bool,
    errors: list[str],
    warnings: list[str],
    df: pd.DataFrame,
    spend_cols: list[str],
) -> dict[str, Any]:
    """Assemble the validation result dict."""
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "spend_columns": spend_cols,
        "n_spend_channels": len(spend_cols),
    }

    # Add date range if parseable
    if "date_week" in df.columns:
        try:
            dates = pd.to_datetime(df["date_week"])
            summary["date_from"] = str(dates.min().date())
            summary["date_to"] = str(dates.max().date())
        except Exception:
            summary["date_from"] = "unparseable"
            summary["date_to"] = "unparseable"

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def compute_descriptive_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute descriptive statistics for revenue and all spend columns.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe (must have 'revenue' and 'spend_*' columns).

    Returns
    -------
    dict mapping column name to:
        - mean : float
        - std  : float
        - min  : float
        - max  : float
        - cv   : float — coefficient of variation (std / mean)
    """
    numeric_cols = ["revenue"] + [c for c in df.columns if c.startswith("spend_")]
    stats = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        col_mean = float(series.mean())
        col_std = float(series.std())

        stats[col] = {
            "mean": round(col_mean, 2),
            "std": round(col_std, 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "cv": round(col_std / col_mean, 4) if col_mean > 0 else 0.0,
        }

    return stats
