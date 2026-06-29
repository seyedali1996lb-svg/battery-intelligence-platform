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

- **Data**: Synthetic data calibrated to the Oxford Battery Degradation Dataset (Birkl et al., 2017)
- **Model**: Gradient Boosting Regressor (scikit-learn) — interpretable, no deep learning
- **Dashboard**: Streamlit + Plotly
- **Data pipeline**: Pandas, NumPy

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/main.py
```
