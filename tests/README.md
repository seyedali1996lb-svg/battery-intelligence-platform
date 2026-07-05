# Tests

```bash
python -m pytest tests/ -v
```

Covers the pure-logic `src/*.py` modules (feature engineering, models, LCO validation, knee detection, trajectory memory, persistence, passport/export, BMS connector guard clauses, webhook notifications, bundle cache, import adapter, dQ/dV).

**Not covered**: `app/main.py` and `app/_pages/*.py` (Streamlit UI code). Testing those needs Streamlit's own `AppTest` harness, which is a different kind of test (simulated widget interaction, not pure function calls) — out of scope for this pass, left as a follow-up.
