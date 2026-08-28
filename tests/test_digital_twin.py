"""Tests for src/digital_twin.py — the Phase 3 CellTwin architecture."""

import numpy as np
import pandas as pd

from conftest import make_cycles_df
from digital_twin import CellTwin, twin_from_cell


def _df(n_cycles=200, fade=0.0006, cap0=2.0):
    return make_cycles_df(n_cycles=n_cycles, fade_per_cycle=fade, initial_capacity_ah=cap0)


def test_twin_snapshot_structure():
    # A strongly-fading cell so the 500-cycle projection window reaches EOL
    twin = twin_from_cell("CellA", _df(n_cycles=200, fade=0.0018), data_mode="nasa", anchor_spm=False)
    snap = twin.snapshot()

    assert snap["cell_id"] == "CellA"
    assert snap["data_mode"] == "nasa"
    assert snap["updated_at"] is not None
    assert snap["history"]["n_cycles"] == 200
    assert snap["history"]["last_cycle"] == 200
    assert snap["indicators"]["soh_pct"] is not None
    assert snap["indicators"]["is_eol"] is False
    assert snap["projection"]["beta"] is not None
    assert snap["projection"]["rul_cycles_to_eol"] is not None
    # Projection sampled every 10 cycles over 500 → ~50 points
    assert len(snap["projection"]["proj_cycles"]) == 50
    # Honest labels always present
    assert any("not a live-synced digital twin" in l for l in snap["labels"])


def test_faster_fading_cell_gets_shorter_projection():
    slow = twin_from_cell("Slow", _df(n_cycles=200, fade=0.0003), data_mode="nasa", anchor_spm=False)
    fast = twin_from_cell("Fast", _df(n_cycles=200, fade=0.0015), data_mode="nasa", anchor_spm=False)
    fast_rul = fast.snapshot()["projection"]["rul_cycles_to_eol"]
    slow_rul = slow.snapshot()["projection"]["rul_cycles_to_eol"]
    # The fast cell reaches EOL inside the window; the slow one may not —
    # either way the fast cell's projection must be strictly shorter.
    assert fast_rul is not None
    assert slow_rul is None or fast_rul < slow_rul


def test_update_is_idempotent_per_cycle_and_extendable():
    twin = CellTwin("CellA", "nasa", anchor_spm=False)
    twin.update(_df(n_cycles=100))
    n1 = twin.snapshot()["history"]["n_cycles"]
    # Re-feeding the same 100 cycles must be a no-op merge
    twin.update(_df(n_cycles=100))
    assert twin.snapshot()["history"]["n_cycles"] == n1
    # Then extend to 150
    twin.update(_df(n_cycles=150))
    assert twin.snapshot()["history"]["n_cycles"] == 150
    assert twin.snapshot()["history"]["last_cycle"] == 150


def test_knee_and_fade_indicators_present():
    snap = twin_from_cell("CellA", _df(n_cycles=250, fade=0.0012), data_mode="synthetic", anchor_spm=False).snapshot()
    assert "knee" in snap["indicators"]
    assert snap["indicators"]["fade_rate_30cy"] is not None


def test_update_rejects_empty_frame():
    twin = CellTwin("CellA", "nasa", anchor_spm=False)
    try:
        twin.update(pd.DataFrame())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_insufficient_cycles_yields_no_projection():
    twin = CellTwin("CellA", "nasa", anchor_spm=False)
    twin.update(_df(n_cycles=3))
    snap = twin.snapshot()
    assert snap["projection"] is None  # needs >=5 for a fade fit


def test_twin_survives_anomalous_data_with_error_label():
    """A bad row must degrade to an honest error label, never crash the caller."""
    twin = CellTwin("CellA", "nasa", anchor_spm=False)
    df = _df(n_cycles=50)
    df.loc[0, "capacity_ah"] = np.nan
    df.loc[0, "cycle_number"] = 0  # non-positive start confuses the fit
    twin.update(df)
    assert "last_error" in twin.snapshot()
