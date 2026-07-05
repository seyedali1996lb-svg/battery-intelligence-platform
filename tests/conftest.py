"""Shared fixtures for the pure-logic module test suite."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest


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
    for src/features.py, src/model.py, src/lco_eval.py, src/knee_detection.py,
    src/trajectory_memory.py.
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
