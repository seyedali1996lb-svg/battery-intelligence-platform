"""Unit tests for src/oxford_loader.py — pure-logic pieces only.

No network calls or real .mat parsing here (that needs the ~822 MB source
zip, downloaded once locally per the module docstring) — these test the
two pieces that don't require it: the max-Amphr-delta-across-steps
capacity extraction rule, and the SOH-relative-to-checkpoint-0 computation.
"""

import numpy as np
import pandas as pd
from oxford_loader import _capacity_ah_from_table, _add_soh_column


def _make_maccor_table(step_specs: list[tuple]) -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking the real Maccor column shape
    (Cyc, Step, Amphr, Amps, ...). step_specs: list of (step_number,
    n_rows, amphr_start, amphr_end) — Amphr ramps linearly within a step,
    matching how the real per-timestamp Amphr column behaves."""
    rows = []
    for step, n, amphr_start, amphr_end in step_specs:
        amphr = np.linspace(amphr_start, amphr_end, n)
        for a in amphr:
            rows.append({"Cyc": 0.0, "Step": float(step), "Amphr": a, "Amps": -1.5, "Volts": 3.7})
    return pd.DataFrame(rows)


def test_capacity_extraction_picks_largest_amphr_swing_step():
    # Step 1: rest (no swing). Step 2: small partial discharge (0.5Ah swing).
    # Step 3: full reference discharge (3.05Ah swing) — this is the one that
    # should be picked, matching the real dataset's pattern where the full
    # discharge step has by far the largest Amphr delta.
    df = _make_maccor_table([
        (1, 50, 0.0, 0.0),
        (2, 50, 0.0, 0.5),
        (3, 200, 0.0, 3.05),
    ])
    cap = _capacity_ah_from_table(df)
    assert abs(cap - 3.05) < 1e-9


def test_capacity_extraction_robust_to_step_numbering_shift():
    """Real dataset quirk: step numbers shifted between files for the same
    cell (BoL had steps up to 41, RPT1 up to 36) — the extraction rule must
    not depend on a hardcoded step number, only on which step has the
    largest swing."""
    df_a = _make_maccor_table([(8, 100, 0.0, 3.1), (1, 20, 0.0, 0.0)])
    df_b = _make_maccor_table([(19, 100, 0.0, 2.9), (1, 20, 0.0, 0.0), (2, 20, 0.0, 0.1)])
    assert abs(_capacity_ah_from_table(df_a) - 3.1) < 1e-9
    assert abs(_capacity_ah_from_table(df_b) - 2.9) < 1e-9


def test_capacity_extraction_all_rest_returns_zero():
    df = _make_maccor_table([(1, 10, 0.0, 0.0), (2, 10, 0.0, 0.0)])
    assert _capacity_ah_from_table(df) == 0.0


def test_soh_relative_to_first_checkpoint():
    df = pd.DataFrame({
        "checkpoint_index": [0, 1, 2, 3],
        "checkpoint_label": ["BoL", "RPT1", "RPT2", "RPT3"],
        "capacity_ah": [3.0, 2.85, 2.7, 2.55],
    })
    out = _add_soh_column(df)
    assert out["soh_pct"].iloc[0] == 100.0
    assert abs(out["soh_pct"].iloc[1] - 95.0) < 1e-9
    assert abs(out["soh_pct"].iloc[3] - 85.0) < 1e-9


def test_soh_column_monotonic_for_monotonic_capacity():
    df = pd.DataFrame({
        "checkpoint_index": [0, 1, 2],
        "checkpoint_label": ["BoL", "RPT1", "RPT2"],
        "capacity_ah": [3.0, 2.9, 2.8],
    })
    out = _add_soh_column(df)
    assert out["soh_pct"].is_monotonic_decreasing
