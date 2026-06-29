# Battery Intelligence Platform

A battery health monitoring and prediction platform built with Python, scikit-learn, and Streamlit.

## What it does

- **State of Health (SOH)** tracking across the full cycle life of a lithium-ion cell
- **Remaining Useful Life (RUL)** prediction using a Gradient Boosting model
- **Explainable AI** — feature importance breakdown showing *why* the model made each prediction
- **Calibrating** confidence tag when the model has insufficient history to be reliable

## Pages

| Page | Status |
|---|---|
| Overview | Live |
| Health | Live |
| Insights | Live |
| Recommendations | Phase 2 |
| Economics | Phase 2 |
| Fleet | Phase 2 |
| Sustainability | Phase 2 |
| Reports | Phase 2 |
| Settings | Phase 2 |

## Tech stack

- **Data**: Physics-informed synthetic data with injected cell-to-cell stress variation (temperature, C-rate, depth of discharge). 8 cells, each with a distinct operating stress profile derived from published LiCoO₂ degradation models (Arrhenius SEI growth, empirical C-rate power law, Rainflow DoD scaling). **Not real measured data** — loading real Oxford/NASA data is a hard gate before Phase 2 (Fleet) launches.
- **Model**: Gradient Boosting Regressor (scikit-learn) — interpretable, no deep learning
- **Dashboard**: Streamlit + Plotly
- **Data pipeline**: Pandas, NumPy

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/main.py
```
