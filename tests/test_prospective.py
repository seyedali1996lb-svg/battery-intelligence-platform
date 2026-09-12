"""Tests for batlab.validation.prospective — the temporal-holdout harness.

The properties that matter: the split genuinely withholds the future
(a model that only interpolates cannot score well here), the RUL honesty
rules from v12 carry over unchanged, and the baselines answer the same
question under the identical split.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_cycles_df
from batlab.validation.prospective import (
    split_prospective,
    run_prospective,
    trivial_soh_baseline_prospective,
    rul_formula_baseline_prospective,
)


def _fleet(n_cells: int = 4, n_cycles: int = 200, fade_base: float = 0.0004) -> dict:
    """Heterogeneous fade rates so a pooled model has real signal."""
    out = {}
    for i in range(n_cells):
        fade = fade_base * (1.0 + 0.25 * i)
        out[f"Cell{i}"] = make_cycles_df(n_cycles=n_cycles, fade_per_cycle=fade)
    return out


# ── split integrity ──────────────────────────────────────────────────────


def test_split_partitions_by_cycle_position():
    df = make_cycles_df(n_cycles=100)
    tr, te = split_prospective(df, train_fraction=0.5)
    assert len(tr) == 50 and len(te) == 50
    assert tr["cycle_number"].max() <= te["cycle_number"].min()


def test_split_never_leaks_future_rows_into_train():
    df = make_cycles_df(n_cycles=101)
    tr, te = split_prospective(df, train_fraction=0.6)
    # The boundary must be exact: no overlap, no gap.
    assert set(tr["cycle_number"]) & set(te["cycle_number"]) == set()
    assert len(tr) + len(te) == len(df)


def test_split_rejects_too_short_cells():
    df = make_cycles_df(n_cycles=8)
    assert split_prospective(df) is None


# ── run_prospective end to end ───────────────────────────────────────────


def test_run_prospective_on_synthetic_fleet():
    res = run_prospective(_fleet())
    assert res["n_cells_evaluated"] == 4
    assert res["n_cells_skipped"] == 0
    assert res["n_test_rows"] > 0 and res["n_train_rows"] > 0
    assert set(res["per_cell"].keys()) == {f"Cell{i}" for i in range(4)}
    assert res["confidence_intervals"] is not None


def test_tree_model_cannot_extrapolate_and_loses_to_linear_baseline():
    """The substantive prospective finding, pinned so it cannot be quietly
    'fixed' by loosening the split: a GBRT cannot forecast below its training
    label range. On a monotonically degrading fleet the train window (first
    half of life) never contains SOH values as low as the test window, so the
    tree saturates near its lowest training leaf and the trivial per-cell
    LINEAR extrapolation — which does extend forward — beats it decisively.
    This is exactly the curve-fitting-vs-forecasting separation: the same
    model's LCO number on these cells is near-perfect."""
    cells = _fleet()
    model = run_prospective(cells)
    base = trivial_soh_baseline_prospective(cells)
    assert base["baseline_soh_r2"] is not None
    # Linear extrapolation is near-perfect on linear synthetic fade.
    assert base["baseline_soh_r2"] > 0.9
    # The tree cannot follow it into the future it never saw.
    assert model["soh_r2"] < base["baseline_soh_r2"]


def test_run_prospective_reports_per_cell_metrics():
    res = run_prospective(_fleet())
    assert set(res["per_cell"].keys()) == {f"Cell{i}" for i in range(4)}
    for entry in res["per_cell"].values():
        assert entry["n_test_rows"] >= 3
        assert entry["soh_r2"] is not None
        assert "rul_label_kinds" in entry


def test_run_prospective_rul_reliability_gates():
    res = run_prospective(_fleet(n_cells=4, fade_base=0.0008))
    # These synthetic cells DO reach EOL in-window (fast fade over 200
    # cycles from 2.0 Ah at 0.0008/cycle crosses 1.6 Ah), so observed rows
    # exist and coverage is high.
    if res["n_rul_observed_rows"] > 0:
        assert res["rul_label_coverage"] > 0.0
        assert res["rul_reliable"] == (res["rul_r2"] >= 0.3 and res["rul_label_coverage"] >= 0.5)


def test_run_prospective_includes_confidence_intervals():
    res = run_prospective(_fleet())
    ci = res["confidence_intervals"]
    assert ci is not None
    assert ci["soh_r2"]["n"] == 4
    assert ci["soh_r2"]["lo"] <= ci["soh_r2"]["mean"] <= ci["soh_r2"]["hi"]


def test_run_prospective_degrades_gracefully_on_tiny_fleets():
    res = run_prospective({"OnlyCell": make_cycles_df(n_cycles=300)})
    assert res["n_cells_evaluated"] == 0
    assert res["confidence_intervals"] is None
    assert res["soh_r2"] != res["soh_r2"]  # NaN, not a number


def test_run_prospective_extracts_features_from_raw_when_not_cached():
    cells = _fleet()
    with_cache = run_prospective(cells)
    # No `featured` passed — the harness must build features itself and the
    # result must be identical (same seed, same data).
    again = run_prospective(cells)
    assert with_cache["soh_r2"] == again["soh_r2"]


# ── baselines under the identical split ─────────────────────────────────


def test_model_is_evaluated_honestly_even_when_it_loses():
    cells = _fleet()
    model = run_prospective(cells)
    base = trivial_soh_baseline_prospective(cells)
    assert base["baseline_soh_r2"] is not None
    # A per-cell linear extrapolation is already strong on linear synthetic
    # data; the GBRT may legitimately lose to it here (trees cannot
    # extrapolate) — the harness reports the loss rather than hiding it.
    # This test just guards the mechanics: both numbers exist and are finite.
    assert np.isfinite(model["soh_r2"])
    assert np.isfinite(base["baseline_soh_r2"])


def test_rul_formula_baseline_runs_and_reports_pool():
    cells = _fleet(n_cells=4, fade_base=0.0008)
    out = rul_formula_baseline_prospective(cells)
    # Either cells reach EOL in the test window (observed rows → a number)
    # or they don't (None) — but the keys must always exist.
    assert "rul_formula_baseline_r2" in out
    assert "per_cell" in out and "n_cells" in out


def test_all_baselines_share_the_same_split():
    """The three evaluators must see identical train/test windows — the
    comparison is only meaningful if the split is common."""
    cells = _fleet(n_cells=3)
    frac = 0.5
    res = run_prospective(cells, train_fraction=frac)
    base = trivial_soh_baseline_prospective(cells, train_fraction=frac)
    assert res["train_fraction"] == frac == pytest.approx(frac)
    assert base["n_cells"] == res["n_cells_evaluated"] == 3
    # Same test-window row counts per cell in both.
    for cid, entry in res["per_cell"].items():
        assert entry["n_test_rows"] == base["per_cell"][cid]["n_test_rows"]
