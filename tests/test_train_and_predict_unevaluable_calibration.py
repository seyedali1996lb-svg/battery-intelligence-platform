"""train_and_predict() on a fleet whose RUL interval is NOT calibratable.

A fleet where no cell reaches end-of-life in the data (Severson is the
real one: every RUL label is formula-extrapolated) makes
run_lco_quantiles() return global_e_star=None and NaN coverage — "not
evaluable", by design. Before 2026-09-13 the note that describes the
calibration to the experiment registry tested the coverage with
`is not None`; NaN passes that test, so the note then formatted the
None E* with `:.2f` and raised TypeError inside the Severson training
thread, and load_everything() re-raised it from the future — a crash on
every genuinely cold boot with Severson present (i.e. every Streamlit
Cloud redeploy). It had been hidden until the same day's _score_count()
fix let calibration succeed at all. tests/test_cold_start_smoke.py
deliberately excludes Severson for cost reasons, so this pins the
shape directly: a slow-fading synthetic fleet with zero observed-EOL rows.
"""

import sys as _sys
import os as _os

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401

import math

import pytest

from conftest import make_cycles_df


@pytest.fixture
def never_reaches_eol_fleet():
    # 0.02%/cycle over 220 cycles: SOH ends near 96%, far above the 80%
    # EOL threshold, so every RUL label is extrapolated — the Severson shape.
    return {
        f"slow_{i}": {"cycles": make_cycles_df(
            n_cycles=220, fade_per_cycle=0.0004 + 0.00002 * i, temperature_c=25.0 + i,
        )}
        for i in range(3)
    }


def test_train_and_predict_survives_unevaluable_calibration(never_reaches_eol_fleet):
    import _data

    raw_fdfs, model_inputs = _data.compute_features_only(never_reaches_eol_fleet)
    # dataset=None/org_id=None keeps this off the registry: the crash was in
    # the note text built just before log_run(), which is reached only when
    # both are given — so exercise that branch through a stubbed log_run.
    logged = {}

    import experiment_registry as reg

    def fake_log_run(**kw):
        logged.update(kw)
        return "fake_run_id"

    original = reg.log_run
    reg.log_run = fake_log_run
    try:
        bundle, featured_dfs, split_cycles = _data.train_and_predict(
            never_reaches_eol_fleet, raw_fdfs, model_inputs, dataset="slowfleet", org_id=1,
        )
    finally:
        reg.log_run = original

    m = bundle["metrics"]
    assert m["n_rul_observed_rows"] == 0, "fixture must have zero observed-EOL rows to reproduce"
    # Not evaluable is reported as None end to end — never NaN leaking into
    # `is not None` checks, never a formatted None.
    assert m["rul_interval_coverage_calibrated"] is None
    assert bundle["interval_e_star"] is None
    assert "NOT available" in logged["notes"]
    cal = logged["lco_metrics"]["calibration_meta"]
    assert cal["rul_interval_coverage_calibrated"] is None
    assert cal["interval_e_star"] is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in cal.values())
