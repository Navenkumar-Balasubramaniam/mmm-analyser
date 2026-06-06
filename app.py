"""
app.py
=======
MMM Analyser — Streamlit entry point.

Orchestrates the full pipeline:
  1. CSV upload + validation
  2. Exploratory data analysis
  3. PyMC Bayesian MMM fit
  4. Post-model extraction
  5. Budget optimisation
  6. AI-powered Q&A chat

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path

from src.validator import validate_csv, compute_descriptive_stats
from src.eda import compute_revenue_trend, compute_spend_share, compute_correlations
from src.model import build_and_fit_mmm
from src.post_model import extract_model_results
from src.optimiser import optimise_budget
from src.agent import ask_agent
from src.charts import (
    chart_revenue_trend, chart_quarterly_avg, chart_spend_share,
    chart_correlations, chart_contribution_waterfall, chart_roas_bars,
    chart_adstock_bars, chart_budget_comparison,
)


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MMM Analyser",
    page_icon="📊",
    layout="wide",
)


# ── Load config ──────────────────────────────────────────────
@st.cache_data
def load_config():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


config = load_config()


# ── Session state defaults ───────────────────────────────────
defaults = {
    "df": None,
    "validation": None,
    "eda_done": False,
    "model_result": None,
    "extracted": None,
    "optimiser_result": None,
    "chat_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 MMM Analyser")
    st.caption("Marketing Mix Modelling powered by PyMC + AI")
    st.divider()

    # CSV template info
    with st.expander("📋 CSV template", expanded=False):
        st.markdown("""
        **Required columns:**
        - `date_week` — Monday of each week (YYYY-MM-DD)
        - `revenue` — Weekly revenue
        - `spend_*` — At least 2 spend columns (e.g. `spend_tv`, `spend_search`)

        **Optional columns:**
        - `imp_*` — Impressions per channel
        - `promo_flag` — 0/1 promotion flag
        - `holiday_flag` — 0/1 holiday flag

        **Minimum:** 52 weeks (1 year) of data
        """)

    # Model settings
    with st.expander("⚙️ Model settings", expanded=False):
        model_cfg = config.get("model", {})
        draws = st.number_input("MCMC draws", value=model_cfg.get("draws", 2000),
                                min_value=100, step=100)
        tune = st.number_input("Tune steps", value=model_cfg.get("tune", 1000),
                               min_value=100, step=100)
        chains = st.number_input("Chains", value=model_cfg.get("chains", 4),
                                 min_value=1, max_value=8)

    st.divider()

    # Reset button
    if st.button("🔄 Reset analysis", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()


# ════════════════════════════════════════════════════════════════
# STEP 1 — CSV UPLOAD + VALIDATION
# ════════════════════════════════════════════════════════════════

st.header("1 · Upload your data")

uploaded = st.file_uploader(
    "Upload weekly marketing data (CSV)",
    type=["csv"],
    help="See the CSV template in the sidebar for required columns.",
)

if uploaded is not None and st.session_state.df is None:
    df = pd.read_csv(uploaded, parse_dates=["date_week"])
    st.session_state.df = df

if st.session_state.df is not None:
    df = st.session_state.df

    # ── Validate ─────────────────────────────────────────────
    if st.session_state.validation is None:
        min_rows = config.get("csv_template", {}).get("min_rows", 52)
        validation = validate_csv(df, min_rows=min_rows)
        st.session_state.validation = validation

    validation = st.session_state.validation

    if not validation["is_valid"]:
        st.error("❌ Validation failed")
        for err in validation["errors"]:
            st.error(f"• {err}")
        st.stop()

    # Show warnings if any
    if validation["warnings"]:
        for warn in validation["warnings"]:
            st.warning(f"⚠️ {warn}")

    # Summary
    summary = validation["summary"]
    cols = st.columns(4)
    cols[0].metric("Rows", summary["n_rows"])
    cols[1].metric("Channels", summary["n_spend_channels"])
    cols[2].metric("From", summary.get("date_from", "—"))
    cols[3].metric("To", summary.get("date_to", "—"))

    # Descriptive stats
    with st.expander("📊 Descriptive statistics", expanded=False):
        stats = compute_descriptive_stats(df)
        stats_df = pd.DataFrame(stats).T
        stats_df.columns = ["Mean", "Std", "Min", "Max", "CV"]
        st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)

    st.success("✅ Data validated — ready to analyse")
    st.divider()

    # ════════════════════════════════════════════════════════════
    # STEP 2 — EXPLORATORY DATA ANALYSIS
    # ════════════════════════════════════════════════════════════

    st.header("2 · Exploratory analysis")
    st.caption("Observational summaries — no model outputs shown here")

    # Revenue trend
    trend = compute_revenue_trend(df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_revenue_trend(trend), use_container_width=True)
    with col2:
        st.plotly_chart(chart_quarterly_avg(trend), use_container_width=True)

    # Spend share + correlation
    spend = compute_spend_share(df)
    corr = compute_correlations(df)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_spend_share(spend), use_container_width=True)
    with col4:
        st.plotly_chart(chart_correlations(corr), use_container_width=True)
        st.caption(corr["disclaimer"])

    st.session_state.eda_done = True
    st.divider()

    # ════════════════════════════════════════════════════════════
    # STEP 3 — BAYESIAN MMM
    # ════════════════════════════════════════════════════════════

    st.header("3 · Bayesian Marketing Mix Model")

    if st.session_state.model_result is None:
        st.info(
            "This runs a Bayesian MMM using PyMC-Marketing. "
            "MCMC sampling takes 3–8 minutes depending on data size and settings."
        )
        if st.button("🚀 Run model", type="primary", use_container_width=True):
            model_config = {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "target_accept": config.get("model", {}).get("target_accept", 0.9),
                "adstock_max_lag": 8,
            }

            with st.spinner("Running MCMC sampling... this takes a few minutes"):
                progress = st.progress(0, text="Building model...")
                progress.progress(10, text="Building model...")

                result = build_and_fit_mmm(df, model_config=model_config)
                progress.progress(70, text="Extracting results...")

                extracted = extract_model_results(result)
                progress.progress(85, text="Optimising budget...")

                # Build current spend dict for optimiser
                spend_cols = result["spend_columns"]
                current_spend = {col: float(df[col].mean()) for col in spend_cols}
                opt_config = config.get("optimiser", {})
                opt_result = optimise_budget(
                    extracted["roas"],
                    current_spend,
                    min_share=opt_config.get("min_channel_share", 0.05),
                    max_share=opt_config.get("max_channel_share", 0.60),
                )

                progress.progress(100, text="Done!")

                st.session_state.model_result = result
                st.session_state.extracted = extracted
                st.session_state.optimiser_result = opt_result
                st.rerun()
    else:
        extracted = st.session_state.extracted
        opt_result = st.session_state.optimiser_result

        st.success("✅ Model fitted successfully")

        # ── Results dashboard ────────────────────────────────
        st.subheader("Channel contributions")
        st.plotly_chart(
            chart_contribution_waterfall(
                extracted["channel_contributions"],
                extracted["base_revenue"],
            ),
            use_container_width=True,
        )

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(chart_roas_bars(extracted["roas"]),
                           use_container_width=True)
        with col6:
            st.plotly_chart(chart_adstock_bars(extracted["adstock_decay"]),
                           use_container_width=True)

        # ── Contribution table ───────────────────────────────
        with st.expander("📋 Detailed contribution table", expanded=False):
            contrib_rows = []
            # Base revenue row
            br = extracted["base_revenue"]
            contrib_rows.append({
                "Channel": "Base / organic",
                "Share (%)": br["share_pct"],
                "Revenue ($)": f"${br['mean']:,.0f}",
                "94% CI": f"${br['ci_lower']:,.0f} – ${br['ci_upper']:,.0f}",
            })
            for ch in extracted["channel_contributions"]:
                contrib_rows.append({
                    "Channel": ch["channel"].replace("spend_", "").title(),
                    "Share (%)": ch["share_pct"],
                    "Revenue ($)": f"${ch['mean']:,.0f}",
                    "94% CI": f"${ch['ci_lower']:,.0f} – ${ch['ci_upper']:,.0f}",
                })
            st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True,
                         hide_index=True)

        # ── Budget optimisation ──────────────────────────────
        st.divider()
        st.header("4 · Budget optimisation")

        uplift = opt_result["predicted_uplift_pct"]
        cols_opt = st.columns(3)
        cols_opt[0].metric("Total weekly budget",
                           f"${opt_result['total_budget']:,.0f}")
        cols_opt[1].metric("Predicted uplift",
                           f"+{uplift:.1f}%")
        cols_opt[2].metric("Reallocation",
                           f"{sum(1 for a in opt_result['allocations'] if a['change'] > 0)} ↑ / "
                           f"{sum(1 for a in opt_result['allocations'] if a['change'] < 0)} ↓")

        st.plotly_chart(
            chart_budget_comparison(opt_result["allocations"]),
            use_container_width=True,
        )

        # Reallocation table
        with st.expander("📋 Detailed reallocation table", expanded=True):
            alloc_rows = []
            for a in opt_result["allocations"]:
                direction = "↑" if a["change"] > 0 else "↓" if a["change"] < 0 else "—"
                alloc_rows.append({
                    "Channel": a["channel"].replace("spend_", "").title(),
                    "Current ($)": f"${a['current_spend']:,.0f}",
                    "Optimal ($)": f"${a['optimal_spend']:,.0f}",
                    "Change ($)": f"{direction} ${abs(a['change']):,.0f}",
                })
            st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True,
                         hide_index=True)

        # ════════════════════════════════════════════════════════
        # STEP 5 — AI CHAT
        # ════════════════════════════════════════════════════════

        st.divider()
        st.header("5 · Ask the AI analyst")
        st.caption(
            "Ask questions about your results. The agent answers only from "
            "the model outputs and methodology — it will tell you if something "
            "is outside its context."
        )

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if question := st.chat_input("Ask about your MMM results..."):
            # Show user message
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_agent(
                        question=question,
                        model_results=extracted,
                        optimiser_results=opt_result,
                        history=st.session_state.chat_history[:-1],
                    )
                st.markdown(response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

else:
    # No file uploaded yet — show instructions
    st.info("👆 Upload a CSV file to get started. Check the sidebar for the template spec.")

    # Show sample data preview
    with st.expander("Preview sample data format"):
        sample_path = Path("data/sample.csv")
        if sample_path.exists():
            sample = pd.read_csv(sample_path, nrows=5)
            st.dataframe(sample, use_container_width=True, hide_index=True)
        else:
            st.code(
                "date_week, revenue, spend_search, spend_social, spend_tv, promo_flag\n"
                "2023-01-02, 95000, 3200, 1800, 5400, 0\n"
                "2023-01-09, 102000, 3600, 2100, 5100, 1",
            )
