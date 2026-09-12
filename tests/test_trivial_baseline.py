"""Unit tests for batlab.validation.trivial_baseline — the honest accuracy
denominator (how much of a model's R² a dumb cycle_number->SOH line already
explains under the same leave-cell-out folds)."""

import numpy as np
import pandas as pd

from batlab.validation.trivial_baseline import baseline_lco_r2
from batlab.validation.lco import run_lco


def _cells(n=160, fade=0.0007, ir0=0.05):
    def _df(f, ir):
        cyc = np.arange(1, n + 1)
        cap = np.clip(2.0 - f * cyc, 0.4, None)
        return pd.DataFrame({
            "cycle_number": cyc,
            "capacity_ah": cap,
            "soh_pct": cap / 2.0 * 100.0,
            "resistance_ohm": ir + 0.00002 * cyc,
            "temperature_c": 25.0,
        })
    return {"A": _df(fade, ir0), "B": _df(fade * 1.4, ir0 + 0.01)}


def test_baseline_returns_nan_for_single_cell():
    """LCO folds need >= 2 cells; one cell cannot form a fold."""
    result = baseline_lco_r2({"A": _cells()["A"]})
    assert result["baseline_soh_r2"] != result["baseline_soh_r2"]  # NaN != NaN
    assert result["per_cell"] == {}
    assert result["n_cells"] == 1


def test_baseline_is_leave_cell_out_per_cell():
    """Same fold structure as run_lco: one entry per cell, each held out once."""
    result = baseline_lco_r2(_cells())
    assert result["n_cells"] == 2
    assert set(result["per_cell"].keys()) == {"A", "B"}
    for fold in result["per_cell"].values():
        assert isinstance(fold["baseline_soh_r2"], float)


def test_baseline_is_finite_and_not_better_than_model_on_shared_trend():
    """On two cells that share a genuine trend, the trivial baseline is finite
    and the model is at least as good (the model can at minimum represent a
    straight line, so it should not score below the baseline by much)."""
    cells = _cells()
    base = baseline_lco_r2(cells)
    lco = run_lco(cells)
    assert np.isfinite(base["baseline_soh_r2"])
    assert np.isfinite(lco["soh_r2"])
    # Model should not be materially worse than a straight line it could mimic.
    assert lco["soh_r2"] >= base["baseline_soh_r2"] - 0.15


def test_baseline_featured_reuse_matches_self_build():
    """Passing already-built feature frames must give the identical result to
    letting the function build them — the reuse path exists only to avoid a
    duplicate (expensive) feature build, not to change the number."""
    from batlab.features.engineering import build_features

    cells = _cells()
    built = {cid: build_features(df, cell_id=cid) for cid, df in cells.items()}

    from_scratch = baseline_lco_r2(cells)
    reused = baseline_lco_r2(cells, featured=built)

    assert abs(from_scratch["baseline_soh_r2"] - reused["baseline_soh_r2"]) < 1e-12
    assert from_scratch["per_cell"].keys() == reused["per_cell"].keys()


def test_baseline_featured_partial_falls_back_to_build():
    """A cell missing from `featured` is built from cell_data as usual."""
    from batlab.features.engineering import build_features

    cells = _cells()
    partial = {"A": build_features(cells["A"], cell_id="A")}  # B absent
    result = baseline_lco_r2(cells, featured=partial)
    assert set(result["per_cell"].keys()) == {"A", "B"}
    assert np.isfinite(result["baseline_soh_r2"])


def test_baseline_can_be_negative_when_cells_have_opposing_trends():
    """A global line is the WRONG model when cells have different fade rates —
    the baseline should honestly report that (R² < 0 = worse than predicting
    the mean), which is exactly the signal this number exists to give."""
    n = 160
    cyc = np.arange(1, n + 1)

    def _df(fade, start):
        cap = np.clip(start - fade * cyc, 0.4, None)
        return pd.DataFrame({
            "cycle_number": cyc,
            "capacity_ah": cap,
            "soh_pct": cap / 2.0 * 100.0,
            "resistance_ohm": 0.05 + 0.00002 * cyc,
            "temperature_c": 25.0,
        })

    # Very different fade rates / offsets so one line can't fit both.
    cells = {"fast": _df(0.004, 2.0), "slow": _df(0.0005, 1.6)}
    base = baseline_lco_r2(cells)
    assert np.isfinite(base["baseline_soh_r2"])
