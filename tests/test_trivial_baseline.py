"""Unit tests for batlab.validation.trivial_baseline — the honest accuracy
denominator (how much of a model's R² a dumb cycle_number->SOH line already
explains under the same leave-cell-out folds).

A second group covers the blank-measurement class: a cell summary can carry a
row with no capacity in it (Severson's S-b1c0 at cycle 11 and S-b1c18 at cycle
39 do), and sklearn's LinearRegression RAISES on a NaN target while r2_score
returns NaN for one. Both used to reach the mean, which is how a real fleet's
baseline became a silent None in the app and could have become a NaN headline
anywhere that checked `is not None`.
"""

import numpy as np
import pandas as pd

from batlab.features.engineering import build_features
from batlab.validation.trivial_baseline import (
    _safe_r2,
    baseline_lco_r2,
    rul_formula_baseline_lco,
)
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


# ── Blank measurements (NaN targets) ───────────────────────────────────────


def _blank_row(cells, cell_id, row=5):
    """A copy of `cells` with one row's measurement blanked, the way Severson's
    S-b1c0 (cycle 11) and S-b1c18 (cycle 39) carry an empty capacity column."""
    out = {cid: df.copy() for cid, df in cells.items()}
    df = out[cell_id]
    df.loc[df.index[row], ["capacity_ah", "soh_pct"]] = float("nan")
    return out


def test_baseline_tolerates_a_blank_target_row():
    """A blank target used to raise (LinearRegression: "Input y contains NaN"),
    which the app's bare except turned into a silent None baseline. The row is
    now set aside, counted, and the fold it belonged to is still scored."""
    cells = _cells()
    result = baseline_lco_r2(_blank_row(cells, "A"))

    assert np.isfinite(result["baseline_soh_r2"])
    assert result["n_nonfinite_target_rows"] == 1
    assert result["n_folds_scored"] == 2
    assert result["n_folds_skipped"] == 0
    assert result["per_cell"]["A"]["n_rows"] == len(cells["A"]) - 1


def test_blank_target_row_is_excluded_not_imputed():
    """Setting the row aside must equal deleting it: the trivial line neither
    fills a value in nor lets a NaN into the mean."""
    cells = _cells()
    blanked = _blank_row(cells, "A")
    deleted = {
        cid: (df.drop(df.index[5]) if cid == "A" else df)
        for cid, df in cells.items()
    }

    assert baseline_lco_r2(blanked)["baseline_soh_r2"] == baseline_lco_r2(deleted)["baseline_soh_r2"]


def test_baseline_reports_an_unscorable_fold_instead_of_averaging_nan():
    """When NOTHING can be scored the fold is reported with a reason; NaN comes
    back only as 'no fold scored at all', never as a fold that entered the mean."""
    cells = _cells()
    cells["B"] = cells["B"].copy()
    cells["B"]["soh_pct"] = float("nan")

    result = baseline_lco_r2(cells)

    assert result["n_folds_scored"] == 0
    assert result["n_folds_skipped"] == 2
    assert result["per_cell"]["A"]["baseline_soh_r2"] is None
    assert "not scored" in result["per_cell"]["A"]["note"]
    assert result["n_nonfinite_target_rows"] == len(cells["B"])
    # NaN means "no fold could be scored" — the app converts that to None.
    assert result["baseline_soh_r2"] != result["baseline_soh_r2"]


def test_baseline_accepts_the_wrapper_cell_shape():
    """build_battery()/the dataset loaders hand out {"cell_id": {"cycles": df}}.
    run_lco() unwraps that shape, and the metric gate passes the SAME dict to
    run_lco and to this baseline — which used to raise AttributeError ('dict'
    object has no attribute 'sort_values') the moment anyone wired the two
    together."""
    cells = _cells()
    wrapped = {cid: {"cell_id": cid, "cycles": df} for cid, df in cells.items()}

    assert (
        baseline_lco_r2(wrapped)["baseline_soh_r2"]
        == baseline_lco_r2(cells)["baseline_soh_r2"]
    )


def test_rul_formula_baseline_never_returns_a_nan_headline():
    """A blank fade rate, a blank capacity, or a blank opening capacity (which
    used to make eol_capacity NaN and with it every prediction) must not reach
    the headline: the result is a number or an explicit 'none' pool."""
    cells = _cells()
    frames = {cid: build_features(df, cell_id=cid) for cid, df in cells.items()}
    frames["A"] = frames["A"].copy()
    frames["A"].loc[frames["A"].index[5], "fade_rate_50cy"] = float("nan")

    poisoned_fade = rul_formula_baseline_lco(cells, featured=frames)
    assert poisoned_fade["n_nonfinite_rows_excluded"] == 1
    assert poisoned_fade["rul_formula_baseline_r2"] is None or np.isfinite(
        poisoned_fade["rul_formula_baseline_r2"]
    )

    blank_open = _cells()
    blank_open["A"] = blank_open["A"].copy()
    blank_open["A"].loc[blank_open["A"].index[0], "capacity_ah"] = float("nan")

    result = rul_formula_baseline_lco(blank_open)
    assert result["rul_formula_baseline_r2"] is None or np.isfinite(
        result["rul_formula_baseline_r2"]
    )
    assert result["rul_formula_baseline_mae"] is None or np.isfinite(
        result["rul_formula_baseline_mae"]
    )


def test_rul_formula_baseline_accepts_the_wrapper_cell_shape():
    cells = _cells()
    wrapped = {cid: {"cycles": df} for cid, df in cells.items()}

    assert (
        rul_formula_baseline_lco(wrapped)["rul_formula_baseline_r2"]
        == rul_formula_baseline_lco(cells)["rul_formula_baseline_r2"]
    )


def test_safe_r2_rejects_non_finite_inputs():
    """r2_score returns NaN for a blank input without raising, so the guard is
    what keeps one out of a published headline."""
    assert _safe_r2(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])) is None
    assert _safe_r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0])) is None
    assert _safe_r2(np.array([1.0, np.inf]), np.array([1.0, 2.0])) is None
    assert _safe_r2([1.0, 2.0], [1.0, 2.0]) == 1.0
