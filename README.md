# Macro Policy Simulator

**Evaluating zero-shot time series foundation models for macroeconomic forecasting across developing economies**

[![AI Forecasting](https://img.shields.io/badge/🤖_AI_Forecasting-Live_Tool-7C3AED?style=for-the-badge)](http://3.229.14.75)

[![MOIRAI Macro Forecasting](https://img.shields.io/badge/🚀_Live_Demo-MOIRAI_Engine-0066FF?style=for-the-badge)](http://3.229.14.75)

The link may not be working all the time as it requires allocating compute (kindly email info.swarajsingh@gmail.com).
---



## Background

This is an ongoing research and development project exploring whether large-scale pre-trained AI models — specifically Universal Time Series Foundation Models (TSFMs) — can be meaningfully applied to macroeconomic forecasting in a development finance context, and how they compare against conventional approaches like VAR, DSGE, and IMF projection baselines.

The core question is straightforward: these models were pre-trained on billions of observations across diverse domains. Does that generalise to sovereign macroeconomic data? Where does it hold, and where does it break down?

The work is grounded in practical questions that matter for development institutions and member states — fiscal space assessment, external vulnerability, debt sustainability, and cross-country shock transmission. The simulator is a working testbed for those questions, not a finished product.

I work at the Organisation of Southern Cooperation in Addis Ababa, and the analytical gaps this tool is trying to address are real ones: development institutions often lack the in-house infrastructure to run rapid, multi-country macroeconomic scenarios at the pace that policy dialogue demands.

---

## What the Tool Does

The simulator runs AI-generated macroeconomic baselines for 35 economies (2026–2030) using IMF WEO data as context, and places them alongside IMF reference projections so the two can be directly compared.

### AI Engine

The forecasting core uses **Moirai** (Salesforce AI Research, 2024) — a deep transformer pre-trained on 27 billion time series observations across 9 domains (LOTSA corpus):

- Zero-shot deployment — no fine-tuning on country data
- Native any-variate architecture: all 5 macroeconomic variables modelled jointly as a single multivariate tensor
- Cross-variate relationship learning captured through attention over the joint historical context
- Exogenous variable support (used for oil scenario channel)
- 500-sample Monte Carlo inference yielding full [p10–p90] uncertainty bands (Can run higher simulations but need greater compute but the performance is better than the slight randomness at this stage)
- 35-year historical context window per country, winsorized [5th–95th pctile] per series before model input

**Variables modelled jointly:**

| Variable | IMF Code |
|---|---|
| Real GDP Growth | `NGDP_RPCH` |
| CPI Inflation | `PCPIPCH` |
| Fiscal Balance (% GDP) | `GGXCNL_NGDP` |
| Current Account (% GDP) | `BCA_NGDPD` |
| Public Debt (% GDP) | `GGXWDG_NGDP` |

### Data

| Source | Coverage |
|---|---|
| IMF WEO April 2026 | 35 countries · 1980–2025 actuals + 2026–2030 projections |
| Brent Crude (EIA) | 1991–2025 · used for oil scenario elasticity translation |

All data is cached as Parquet at startup. No live API calls during inference.

---

## Development Status

### Layer 1 — AI Baseline Forecasting
This layer is reasonably mature. The forecasting pipeline is stable, the data ingestion is clean, and the comparison against IMF projections works as intended across all 35 economies. The 2025 nowcast override mechanic — which allows pre-filling confirmed WEO actuals or substituting analyst conviction estimates before the model sees the context — adds meaningful flexibility for countries where 2025 data is still IMF-projected rather than confirmed.

This is the layer where the core research question is most directly testable: does Moirai's zero-shot output track IMF projections, and where and why does it diverge?

### Layer 2 — Shock Transmission Engine
This layer is still being developed and should be treated as directional rather than analytically rigorous.

The current implementation translates oil price shocks to domestic macroeconomic impacts via country-specific elasticity coefficients, then injects them into the model context before inference. Domestic variable shocks (up to 3 simultaneously) use a persistence-decay injection across the last two context years — a deliberate design choice to signal structural regime shifts to the model rather than transient deviations, which the transformer would otherwise mean-revert.

**Current limitations:**
- Shock elasticities are empirical estimates, not structurally identified — there is no DSGE or VAR backbone calibrating them
- Oil translation is simplified; country heterogeneity is partially captured but not fully
- Cross-variable propagation is entirely model-learned — it reflects patterns in Moirai's pre-training data, not domain-calibrated structural relationships
- The accounting identity problem: Moirai treats fiscal balance and public debt as independent statistical series and does not enforce the identity linking them, which can produce scenarios that are directionally plausible but not jointly coherent over longer horizons

Planned work includes VAR-anchored elasticity estimation and better uncertainty propagation through the oil channel.

### Layer 3 and Beyond
Further analytical layers — debt sustainability analysis, external vulnerability scoring, fiscal space assessment, and early warning signal generation — are in scope but not yet implemented.

---

## Stack

**Backend**: Python · Flask · Gunicorn · Moirai (uni2ts) · Pandas · PyArrow · NumPy

**Frontend**: React 19 · Vite · Recharts

**Deployment**: EC2 (c7i-flex.large) · Nginx

---

## Running Locally

Requires Python 3.10+, Node 18+, ~2GB disk for model download.

```bash
git clone https://github.com/Swaraj1313/macro-ai-engine.git
cd macro-ai-engine

# Backend
python3 -m venv venv && source venv/bin/activate
# Install deps in order — see EC2_DEPLOY_STEPS.md for pinned versions
cd backend && python app.py
# First run downloads Moirai (~1.24GB) from HuggingFace — takes 2-3 min

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

---

## References

- Woo, G. et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers*. Salesforce AI Research. [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)
- IMF World Economic Outlook, April 2026
- LOTSA: Large-Scale Open Time Series Archive

---

