"""
src/charts.py
==============
Plotly chart builders for the MMM Analyser UI.

Every function returns a plotly.graph_objects.Figure.
Charts are rendered in Streamlit via st.plotly_chart(fig).

Design rule: charts display data from the module that produced it.
  - EDA charts use EDA outputs (no model numbers)
  - Model charts use post_model outputs (PyMC only)
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Any


# ── Shared styling ───────────────────────────────────────────
COLORS = [
    "#534AB7", "#0F6E56", "#854F0B", "#993C1D",
    "#185FA5", "#993556", "#5F5E5A",
]
LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    height=400,
)


def _apply_layout(fig, title: str = ""):
    """Apply consistent layout to any figure."""
    fig.update_layout(**LAYOUT_DEFAULTS, title=title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


# ────────────────────────────────────────────────────────────────
# EDA CHARTS (Phase 2 — observational only)
# ────────────────────────────────────────────────────────────────

def chart_revenue_trend(trend_data: dict) -> go.Figure:
    """Line chart of revenue over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data["dates"],
        y=trend_data["revenue"],
        mode="lines",
        line=dict(color=COLORS[0], width=2),
        name="Weekly revenue",
    ))
    return _apply_layout(fig, "Revenue over time")


def chart_quarterly_avg(trend_data: dict) -> go.Figure:
    """Bar chart of average revenue per quarter."""
    quarters = [q["quarter"] for q in trend_data["quarterly_avg"]]
    values = [q["avg_revenue"] for q in trend_data["quarterly_avg"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quarters, y=values,
        marker_color=COLORS[0], opacity=0.8,
    ))
    return _apply_layout(fig, "Average revenue by quarter")


def chart_spend_share(spend_data: dict) -> go.Figure:
    """Donut chart of spend share per channel."""
    labels = [ch["channel"].replace("spend_", "").title()
              for ch in spend_data["channels"]]
    values = [ch["total_spend"] for ch in spend_data["channels"]]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=COLORS[:len(labels)]),
        textinfo="label+percent",
        textposition="outside",
    ))
    return _apply_layout(fig, "Spend share by channel")


def chart_correlations(corr_data: dict) -> go.Figure:
    """Horizontal bar chart of Pearson r per channel vs revenue."""
    channels = [c["channel"].replace("spend_", "").title()
                for c in corr_data["correlations"]]
    r_values = [c["r"] for c in corr_data["correlations"]]
    colors = [COLORS[0] if c["significant"] else "#B4B2A9"
              for c in corr_data["correlations"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=r_values, y=channels,
        orientation="h",
        marker_color=colors,
        text=[f"r={r:.3f}" for r in r_values],
        textposition="outside",
    ))
    fig.update_xaxes(range=[-0.3, 0.7], title="Pearson r")
    return _apply_layout(fig, "Raw correlation with revenue (pre-model)")


# ────────────────────────────────────────────────────────────────
# MODEL CHARTS (Phase 4 — from PyMC posterior only)
# ────────────────────────────────────────────────────────────────

def chart_contribution_waterfall(contributions: list, base_revenue: dict) -> go.Figure:
    """Waterfall chart showing revenue decomposition."""
    labels = ["Base / organic"]
    values = [base_revenue["mean"]]
    labels += [ch["channel"].replace("spend_", "").title()
               for ch in contributions]
    values += [ch["mean"] for ch in contributions]
    labels.append("Total")

    measures = ["absolute"] + ["relative"] * len(contributions) + ["total"]

    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        x=labels, y=values,
        measure=measures,
        connector=dict(line=dict(color="rgba(0,0,0,0.1)")),
        increasing=dict(marker=dict(color=COLORS[1])),
        decreasing=dict(marker=dict(color=COLORS[3])),
        totals=dict(marker=dict(color=COLORS[0])),
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
    ))
    return _apply_layout(fig, "Revenue decomposition (from PyMC posterior)")


def chart_roas_bars(roas: list) -> go.Figure:
    """Bar chart of ROAS per channel with credible intervals."""
    channels = [r["channel"].replace("spend_", "").title() for r in roas]
    means = [r["mean"] for r in roas]
    ci_lower = [r["mean"] - r["ci_lower"] for r in roas]
    ci_upper = [r["ci_upper"] - r["mean"] for r in roas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=means,
        marker_color=COLORS[:len(channels)],
        error_y=dict(
            type="data",
            symmetric=False,
            array=ci_upper,
            arrayminus=ci_lower,
            color="rgba(0,0,0,0.3)",
        ),
        text=[f"{m:.2f}x" for m in means],
        textposition="outside",
    ))
    fig.update_yaxes(title="ROAS ($ revenue per $1 spent)")
    return _apply_layout(fig, "ROAS by channel (with 94% credible interval)")


def chart_adstock_bars(adstock: list) -> go.Figure:
    """Bar chart of learned adstock decay rates."""
    channels = [a["channel"].replace("spend_", "").title() for a in adstock]
    means = [a["mean"] for a in adstock]
    ci_lower = [a["mean"] - a["ci_lower"] for a in adstock]
    ci_upper = [a["ci_upper"] - a["mean"] for a in adstock]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=means,
        marker_color=COLORS[:len(channels)],
        error_y=dict(
            type="data", symmetric=False,
            array=ci_upper, arrayminus=ci_lower,
            color="rgba(0,0,0,0.3)",
        ),
        text=[f"{m:.2f}" for m in means],
        textposition="outside",
    ))
    fig.update_yaxes(title="Decay rate (0 = no memory, 1 = full memory)", range=[0, 1])
    return _apply_layout(fig, "Adstock decay rates (learned from data)")


def chart_budget_comparison(allocations: list) -> go.Figure:
    """Grouped bar chart — current vs optimal spend per channel."""
    channels = [a["channel"].replace("spend_", "").title() for a in allocations]
    current = [a["current_spend"] for a in allocations]
    optimal = [a["optimal_spend"] for a in allocations]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current", x=channels, y=current,
        marker_color="#B4B2A9", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="Optimal", x=channels, y=optimal,
        marker_color=COLORS[0],
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Weekly spend ($)")
    return _apply_layout(fig, "Current vs optimal budget allocation")
