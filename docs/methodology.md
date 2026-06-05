# MMM Analyser — Methodology Reference

This document is the agent's knowledge base. The LLM agent can ONLY answer
questions using information from this document and the model results. If a
question falls outside this scope, the agent must say so.

---

## What is Marketing Mix Modelling (MMM)?

Marketing Mix Modelling is a statistical technique that decomposes revenue (or
any business KPI) into contributions from each marketing channel — such as TV,
paid search, social media, email, and out-of-home advertising. It uses
aggregated weekly time-series data, not user-level tracking, so it does not
depend on cookies or consent.

MMM answers: "Which channels are driving revenue, by how much, and where
should we shift budget?"

---

## Data requirements

The model expects weekly time-series data with:
- `date_week`: Monday of each week (YYYY-MM-DD format)
- `revenue`: Weekly revenue (the dependent variable)
- `spend_*`: Weekly spend per marketing channel (at least 2 channels)
- `promo_flag` (optional): 1 if a promotion ran that week, 0 otherwise
- `holiday_flag` (optional): 1 if the week falls in a holiday period, 0 otherwise

Minimum recommended data: 52 weeks (1 year). Ideal: 104–208 weeks (2–4 years).

---

## Analysis pipeline

### Phase 1 — Data validation (pre-flight)

Checks applied before any analysis runs:
- No missing values in required columns
- No negative spend values
- No duplicate rows
- Date column has consistent 7-day gaps (weekly frequency)
- At least 2 spend columns present
- Minimum row count met (configurable, default 52)

Descriptive statistics (mean, std, min, max, coefficient of variation) are
computed for every numeric column. These are observational summaries — they
do not involve any model.

### Phase 2 — Exploratory data analysis (EDA)

Three analyses run before the model. These are clearly labelled as
pre-model observations and do not produce any attribution or ROAS numbers.

**Revenue trend**: A line chart of revenue over time. Visual only — no
regression slope or lift numbers. Trend and seasonality are estimated
by the Bayesian model in Phase 3.

**Spend share**: Total and percentage spend per channel. Pure arithmetic.
Answers "how is budget currently allocated?" — not "how effective is it?"

**Correlation heatmap**: Pearson correlation between each spend column and
revenue. Labelled as "raw signal before adstock correction." A low
correlation does NOT mean the channel is ineffective — carry-over effects
and saturation distort raw correlation. The model accounts for this.

### Phase 3 — Bayesian MMM (PyMC-Marketing)

This is the single source of truth for all attribution numbers.

**Model specification**:
```
Revenue(t) = intercept
           + Σ [ saturation( adstock( spend_channel(t) ) ) × β_channel ]
           + β_promo × promo_flag(t)
           + β_holiday × holiday_flag(t)
           + yearly_seasonality(t)
           + ε(t)
```

**Adstock transformation**: Models the carry-over effect of advertising.
A TV ad seen in week 1 still influences purchases in weeks 2, 3, etc.
The geometric adstock formula is:

```
Adstocked(t) = Spend(t) + decay × Adstocked(t-1)
```

The decay rate is NOT hardcoded — PyMC-Marketing learns it from data
using Bayesian inference. A high decay (e.g. 0.7) means the channel has
long memory (TV, OOH). A low decay (e.g. 0.2) means near-instant effect
(email, search).

**Saturation transformation**: Models diminishing returns. The first dollar
spent on search returns more than the thousandth. The logistic saturation
curve compresses spend at high levels:

```
Saturated(x) = capacity / (1 + exp(-slope × (x - inflection)))
```

Parameters (capacity, slope, inflection) are learned from data per channel.

**Bayesian inference**: Unlike OLS which gives one coefficient estimate,
PyMC fits a posterior distribution over every parameter using MCMC (Markov
Chain Monte Carlo) with the NUTS sampler. This means every output has a
credible interval — e.g. "Search ROAS = 4.2x ± 0.8x (94% CI)."

**Sampler settings** (configurable in config.yaml):
- Draws: 2000 samples per chain
- Tune: 1000 warm-up samples (discarded)
- Chains: 4 parallel chains
- Target accept: 0.9

**Convergence diagnostics**: The model checks:
- R-hat < 1.05 for all parameters (chains mixed well)
- Effective sample size > 400
- No divergences during sampling

### Phase 4 — Post-model outputs

All outputs in this phase are extracted exclusively from the PyMC posterior.
No OLS, no heuristics, no hardcoded values.

**Channel contribution**: Revenue attributed to each channel.
```
Channel_contribution(t) = β_channel × saturated_adstocked_spend(t)
Channel_% = sum(Channel_contribution) / sum(Revenue) × 100
```
Reported with posterior mean and 94% credible interval.

**ROAS (Return on Ad Spend)**: Marginal revenue per dollar spent per channel.
This is the posterior mean of β_channel, interpreted as: "if this channel's
spend increases by $1, revenue increases by $ROAS — holding all other
channels constant."

**Base revenue**: The intercept — revenue the business would earn with zero
media spend. Represents brand strength, organic traffic, repeat customers.

**Fitted adstock decay rates**: The decay parameter PyMC learned from data
per channel. Shown as a curve of how the effect decays over time.

**Fitted saturation curves**: The logistic curve PyMC learned per channel.
Shows where each channel hits diminishing returns — the inflection point
indicates the optimal spend level.

**Promo and holiday lift**: Posterior coefficients for promo_flag and
holiday_flag. Reported with credible intervals.

### Phase 5 — Budget optimisation

Uses scipy.optimize (SLSQP method) to find the spend allocation that
maximises predicted revenue under the total budget constraint.

Inputs: PyMC posterior ROAS and saturation curves ONLY.
Constraints:
- Total spend = current total (budget-neutral reallocation)
- Each channel gets at least 5% of total (configurable)
- Each channel gets at most 60% of total (configurable)

Output: Optimal spend per channel, predicted revenue uplift, and
specific reallocation moves (e.g. "shift $2,000/week from TV to email").

---

## Known limitations

1. **Correlation, not causation**: MMM infers channel effects from
   co-movement of spend and revenue. If TV always runs during Christmas,
   the model may partially attribute seasonal lift to TV despite control
   variables.

2. **Multicollinearity**: If two channels always move together (same
   campaign periods), the model cannot cleanly separate their effects.
   Bayesian priors stabilise estimates but cannot fully resolve this.

3. **No cross-channel interactions**: The model assumes channels act
   independently. In reality, TV may boost search effectiveness (halo
   effect). This basic model does not capture interaction terms.

4. **Data-dependent decay rates**: PyMC learns adstock decay from data,
   but needs sufficient spend variation. Channels with sparse or
   infrequent spend (e.g. OOH bought quarterly) produce wider credible
   intervals.

5. **Saturation curve extrapolation**: The model is reliable within the
   observed spend range. Predictions for spend levels far above or below
   historical data are uncertain.

---

## What this tool does NOT do

- User-level attribution (no individual tracking)
- Real-time optimisation (model runs on historical weekly data)
- Cross-device or cross-session tracking
- Creative effectiveness measurement (which ad, not which channel)
- Incrementality testing (requires controlled experiments)
