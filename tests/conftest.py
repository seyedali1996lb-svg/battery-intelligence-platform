"""Shared fixtures for the pure-logic module test suite."""

import sys as _sys
import os as _os

# Bootstrap the repo root so _paths is importable, then delegate the rest.
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import _paths as _bp  # noqa: F401  (side-effect: adds src/, app/, scripts/ to sys.path)

import numpy as np
import pandas as pd
import pytest

# The leave-cell-out fold cache (batlab.validation.fold_cache) is a real,
# developer-warm byproduct of running the app: a fold result from a previous
# local run of the same fixture would be REPLAYED here. That is correct in
# production and wrong in a test suite — a test that counts model fits, or
# times one, must fail on the code's behaviour, not pass because someone ran
# the app an hour ago. So the suite pins the cache off, and the cache's own
# tests (tests/test_lco_fold_cache.py) opt back in per test, each with its own
# BATLAB_LCO_CACHE_DIR under tmp_path.
_os.environ["BATLAB_LCO_CACHE"] = "off"


@pytest.fixture(autouse=True)
def _warm_cache_hits_allowed(monkeypatch):
    """Let AppTests take warm bundle-cache hits.

    app/_data.py verifies every cache hit against the experiment registry
    (a bundle whose run id has no row in THIS DB is discarded and
    retrained — the 2026-09-13 registry-integrity fix). The suite's
    isolated_db fixtures give each test a fresh, empty registry, so that
    check would reject every warm hit and retrain every fleet in every
    AppTest (and each retrain stamps the cache with a run id that exists
    only in that test's temp DB, so the next test rejects it again). The
    check is exercised directly in tests/test_cache_registry_consistency.py;
    tests that want the app-level behaviour set
    _data.VERIFY_CACHED_BUNDLES back to True themselves.
    """
    try:
        import _data
    except Exception:  # a test collecting without the app importable
        yield
        return
    monkeypatch.setattr(_data, "VERIFY_CACHED_BUNDLES", False)
    yield


def make_cycles_df(
    n_cycles: int = 200,
    initial_capacity_ah: float = 2.0,
    fade_per_cycle: float = 0.0006,
    initial_resistance_ohm: float = 0.05,
    resistance_rise_per_cycle: float = 0.00005,
    temperature_c: float = 25.0,
    first_resistance_is_zero: bool = False,
) -> pd.DataFrame:
    """A synthetic, monotonically-fading cycles DataFrame with soh_pct already
    computed (matching enrich_cycles()'s output shape), reused across tests
    for batlab.features.engineering, batlab.models.gbrt, batlab.validation.lco,
    batlab.features.knee_detection, src/trajectory_memory.py.
    """
    cycles = np.arange(1, n_cycles + 1)
    capacity = initial_capacity_ah - fade_per_cycle * cycles
    resistance = initial_resistance_ohm + resistance_rise_per_cycle * cycles
    if first_resistance_is_zero:
        resistance[0] = 0.0  # Severson-style missing-first-cycle marker
    df = pd.DataFrame({
        "cycle_number": cycles,
        "capacity_ah": capacity,
        "resistance_ohm": resistance,
        "temperature_c": np.full(n_cycles, temperature_c),
    })
    df["soh_pct"] = (df["capacity_ah"] / initial_capacity_ah) * 100.0
    return df


@pytest.fixture
def cycles_df():
    return make_cycles_df()
