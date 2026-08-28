"""
Performance regression guards (perf-regression batch, docs/history.md).

The point of these tests is to stop silent performance regressions — the
kind that never fail a correctness test but make the demo feel slow (or
blow past a documented ceiling) one commit at a time. The LCO-on-NASA
pipeline and the Parquet cell store are the two hot paths most likely to
regress, so they get explicit timing budgets.

Timing assertions are inherently machine-dependent, so the budgets are
deliberately generous (5-10x above what this codebase measured on its own
dev machine) and the whole timing class is opt-in via ``RUN_PERF_TESTS=1``
— CI (and any slow shared runner) stays deterministic without them, and
anyone tuning performance locally can turn them on:

    RUN_PERF_TESTS=1 python -m pytest tests/test_perf_regressions.py -q

What always runs, with no env var: a correctness smoke of both hot paths
(build LCO metrics on the real committed NASA data, read a cell from the
Parquet store) — so the perf tests still prove the paths work even when
their timing guards are skipped.
"""

import os
import time

import numpy as np
import pandas as pd
import pytest

import cell_store as cs

RUN_PERF = os.environ.get("RUN_PERF_TESTS", "0") == "1"

# Generous budgets (seconds) — see module docstring. Measured on the dev
# machine this project runs on; 5-10x headroom for slower shared runners.
LCO_NASA_BUDGET_S = 20.0
PARQUET_READ_BUDGET_S = 2.0

_skip_unless_perf = pytest.mark.skipif(
    not RUN_PERF, reason="set RUN_PERF_TESTS=1 to run timing guards"
)


# ── Always-run correctness smoke (no timing) ────────────────────────────────


def test_lco_on_nasa_correctness_smoke():
    """The hot path works: real committed NASA cells run through LCO."""
    from batlab.validation.lco import run_lco

    from batlab.datasets import nasa

    cells = nasa.load_nasa_cells()  # served from committed data/raw CSVs
    assert len(cells) == 4
    result = run_lco(cells)
    assert result["soh_r2"] > 0.0  # a real fit, not an error path
    assert set(result["per_cell"].keys()) == set(cells.keys())


def test_cell_store_read_write_smoke(tmp_path, monkeypatch):
    """Hot path works: save → memory-mapped read round-trips exactly."""
    monkeypatch.setattr(cs, "CELL_STORE_DIR", tmp_path / "cell_store")
    cs.clear_lru()
    try:
        df = _big_df("perf_smoke", n_cycles=2000)
        cs.save_cell_df("perf_smoke", df)
        loaded = cs.get_cell_df("perf_smoke")
        pd.testing.assert_frame_equal(loaded, df)
        subset = cs.get_cell_df("perf_smoke", columns=["cycle_number", "soh_pct"])
        assert list(subset.columns) == ["cycle_number", "soh_pct"]
        assert len(subset) == len(df)
    finally:
        cs.clear_lru()


# ── Timing guards (opt-in) ──────────────────────────────────────────────────


@_skip_unless_perf
def test_lco_nasa_budget():
    """LCO on the full NASA fleet stays under budget (opt-in)."""
    from batlab.validation.lco import run_lco

    from batlab.datasets import nasa

    cells = nasa.load_nasa_cells()
    t0 = time.perf_counter()
    run_lco(cells)
    elapsed = time.perf_counter() - t0
    assert elapsed < LCO_NASA_BUDGET_S, f"LCO on NASA took {elapsed:.1f}s (budget {LCO_NASA_BUDGET_S}s)"


@_skip_unless_perf
def test_cell_store_parquet_read_budget(tmp_path, monkeypatch):
    """A full-cell Parquet read stays under budget (opt-in)."""
    monkeypatch.setattr(cs, "CELL_STORE_DIR", tmp_path / "cell_store")
    cs.clear_lru()
    try:
        df = _big_df("perf_budget", n_cycles=5000)
        cs.save_cell_df("perf_budget", df)
        cs.clear_lru()  # force a real disk read
        t0 = time.perf_counter()
        cs.get_cell_df("perf_budget")
        elapsed = time.perf_counter() - t0
        assert elapsed < PARQUET_READ_BUDGET_S, f"cell_store read took {elapsed:.2f}s (budget {PARQUET_READ_BUDGET_S}s)"
    finally:
        cs.clear_lru()


def _big_df(cell_id: str, n_cycles: int) -> pd.DataFrame:
    """A wide-ish per-cycle DataFrame (more columns than real cells carry,
    to make the read a slightly pessimistic bound)."""
    rng = np.random.default_rng(abs(hash(cell_id)) % (2**32))
    cycle = np.arange(1, n_cycles + 1)
    soh = np.clip(100.0 - 0.02 * cycle + rng.normal(0, 0.05, n_cycles), 0, 100)
    return pd.DataFrame({
        "cycle_number": cycle,
        "soh_pct": soh,
        "capacity_ah": 2.0 * soh / 100.0,
        "resistance_ohm": 0.05 + 1e-5 * cycle,
        "fade_rate_30cy": rng.normal(0.02, 0.01, n_cycles),
        "fade_rate_50cy": rng.normal(0.02, 0.01, n_cycles),
        "temperature_c": rng.normal(25, 2, n_cycles),
        "c_rate": np.full(n_cycles, 0.8),
        "is_eol": soh < 80.0,
        "rul_pred": np.maximum(0, n_cycles - cycle),
    })
