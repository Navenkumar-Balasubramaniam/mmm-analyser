# MMM Analyser

A Streamlit-based Marketing Mix Modelling tool that runs a Bayesian MMM (PyMC-Marketing) on weekly spend/revenue data and generates AI-powered strategic recommendations via OpenRouter.

## Quick start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd mmm-analyser

# 2. Create conda environment
conda env create -f environment.yaml
conda activate mmm-analyser

# 3. Set up your API key
cp .env.example .env
# Edit .env and paste your OpenRouter key

# 4. Run the app
streamlit run app.py

# 5. Run tests
pytest tests/ -v
```

## CSV template

Your CSV must have these columns (weekly frequency):

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `date_week` | date (YYYY-MM-DD) | Yes | Monday of each week |
| `revenue` | numeric | Yes | Weekly revenue |
| `spend_*` | numeric | Yes (at least 2) | Weekly spend per channel (e.g. `spend_tv`, `spend_search`) |
| `imp_*` | numeric | No | Impressions per channel (optional enrichment) |
| `promo_flag` | 0/1 | No | Promotion active that week |
| `holiday_flag` | 0/1 | No | Holiday period flag |

A sample file is provided at `data/sample.csv`.

## Architecture

```
app.py                  Streamlit UI — orchestrates the pipeline
src/validator.py        Data quality checks and descriptive stats
src/eda.py              Exploratory charts (trend, spend share, correlation)
src/model.py            PyMC-Marketing Bayesian MMM (single source of truth)
src/post_model.py       Extracts all outputs from PyMC posteriors only
src/optimiser.py        Budget reallocation via scipy on PyMC outputs
src/agent.py            LLM agent (OpenRouter) — context-aware Q&A
src/charts.py           Plotly chart builders
docs/methodology.md     Agent knowledge base — what it can answer from
```

## Design principles

1. **Single source of truth** — all attribution numbers come from PyMC posteriors. No OLS or heuristic outputs shown to the user.
2. **Tests define goals** — pytest cases describe what the module should achieve, not how it's implemented internally.
3. **Agent stays in context** — the LLM only answers from `docs/methodology.md` and the model results. Unknown questions get a clear "not in context" response.
