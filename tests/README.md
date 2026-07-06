# Tests

```bash
python -m pytest tests/ -v
```

Covers the pure-logic `src/*.py` modules (feature engineering, models, LCO validation, knee detection, trajectory memory, persistence, passport/export, BMS connector guard clauses, webhook notifications, bundle cache, import adapter, dQ/dV).

**`test_app_state_combinations.py`** runs `app/main.py` end-to-end via Streamlit's `AppTest` harness — slower (each test executes the full script, ~3-10s) but this is the only layer that catches state-COMBINATION bugs (role × data source × page), where every individual function is correct in isolation but the wrong one gets called for a specific combination. This test file already found and drove the fix for 4 real bugs this way, including two that were completely masked (unreachable) until an earlier bug in the same code path was fixed first — pure-logic unit tests could not have caught any of them, since the bug was in *which* arguments got passed, not in any single function's logic.
