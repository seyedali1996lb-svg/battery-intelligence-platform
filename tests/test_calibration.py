"""Unit tests for batlab.validation.calibration — quantile-interval calibration."""

import numpy as np
import pytest

from conftest import make_cycles_df
from batlab.validation.calibration import (
    NOMINAL_INTERVAL_COVERAGE,
    empirical_coverage,
    interval_width_mean,
    recalibrate_lco_intervals,
    run_lco_quantiles,
)


def test_empirical_coverage_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert empirical_coverage(y, np.zeros(4), np.full(4, 5.0)) == 1.0
    # only 3.0 of [1,2,3,4] falls inside [2.5, 3.5]
    assert empirical_coverage(y, np.full(4, 2.5), np.full(4, 3.5)) == 0.25
    assert np.isnan(empirical_coverage([], [], []))


def test_interval_width_mean():
    assert interval_width_mean(np.array([0, 10]), np.array([10, 20])) == 10.0
    # degenerate/negative widths are clamped to >= 0
    assert interval_width_mean(np.array([5, 5]), np.array([2, 2])) == 0.0


def test_run_lco_quantiles_structure():
    # Fast fade so cells cross EOL in-window → observed RUL labels exist.
    cell_data = {
        "CellA": make_cycles_df(n_cycles=300, fade_per_cycle=0.003),
        "CellB": make_cycles_df(n_cycles=300, fade_per_cycle=0.0035, initial_resistance_ohm=0.06),
    }
    result = run_lco_quantiles(cell_data)
    for key in ("rul_interval_coverage", "rul_interval_width_mean", "rul_r2",
                "rul_mae", "rul_reliable", "per_cell", "rul_label_coverage"):
        assert key in result
    assert set(result["per_cell"]) == {"CellA", "CellB"}
    for fold in result["per_cell"].values():
        assert fold["rul_interval_coverage"] is not None
        assert 0.0 <= fold["rul_interval_coverage"] <= 1.0
        assert fold["rul_true"].shape == fold["rul_q10"].shape == fold["rul_q90"].shape
        assert fold["rul_interval_width_mean"] >= 0.0
    assert result["rul_label_coverage"] == 1.0


def test_run_lco_quantiles_returns_global_conformal_correction():
    """Tier 3: run_lco_quantiles must also return the pooled cross-cell
    conformity machinery — global_e_star (the widening applied by the serve
    path) and recalibrated_coverage (the MEASURED coverage the widened
    interval achieves) — so the deployment can serve calibrated intervals
    and disclose their measured coverage next to every display."""
    cell_data = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.003 + i * 1e-4)
        for i in range(4)
    }
    result = run_lco_quantiles(cell_data)

    for key in ("global_e_star", "recalibrated_coverage",
                "recalibrated_width_mean", "pooled_conformity_scores"):
        assert key in result, f"missing {key}"

    e_star = result["global_e_star"]
    assert e_star is not None and np.isfinite(e_star) and e_star >= 0.0
    # Signed conformity scores E_i = max(q10 - y, y - q90): negative means
    # the true value sat INSIDE the interval. The pooled set is returned
    # unclamped for transparency; the derived E* is clamped at 0 because the
    # serve path only ever widens.
    scores = result["pooled_conformity_scores"]
    assert len(scores) > 0

    # Widening an interval can only maintain or improve coverage.
    raw = result["rul_interval_coverage"]
    cal = result["recalibrated_coverage"]
    assert cal >= raw - 1e-9
    # And the widened interval is at least as wide as the raw one.
    assert result["recalibrated_width_mean"] >= result["rul_interval_width_mean"] - 1e-9


def test_run_lco_quantiles_no_observed_rows_yields_none_e_star():
    """With zero observed-EOL rows there is no honest calibration set: the
    correction must be None (serve path stays un-widened), never 0 or a
    fabricated number."""
    cell_data = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.0002 + i * 1e-5)
        for i in range(3)
    }
    result = run_lco_quantiles(cell_data)
    assert result["global_e_star"] is None
    # NaN is this module's established not-evaluable convention for coverage
    # metrics; the app boundary (app/_data.py) sanitizes NaN -> None before
    # the values reach JSON/UI surfaces.
    assert result["recalibrated_coverage"] != result["recalibrated_coverage"]
    assert result["recalibrated_width_mean"] != result["recalibrated_width_mean"]
    assert result["pooled_conformity_scores"].size == 0


def test_run_lco_quantiles_without_observed_rows_is_not_reliable():
    """A fleet whose cells never reach EOL has 100% formula-extrapolated RUL
    labels: interval quality on those rows is formula recovery, so coverage
    must come back not-evaluable and reliability False (Tier-0 regression)."""
    cell_data = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.0002 + i * 1e-5)
        for i in range(3)
    }
    result = run_lco_quantiles(cell_data)
    assert result["rul_interval_coverage"] != result["rul_interval_coverage"]  # NaN
    assert result["rul_reliable"] is False
    assert result["rul_label_coverage"] == 0.0


def test_recalibration_widens_overconfident_intervals_toward_nominal():
    """Fabricate an LCO result whose Q10/Q90 intervals are far too narrow
    (model overconfident): raw coverage ~0.12 instead of the nominal 0.80.
    Isotonic recalibration must widen the interval so the corrected
    coverage lands near nominal."""
    rng = np.random.default_rng(7)
    folds = {}
    for cell in ["c1", "c2", "c3", "c4"]:
        y = rng.normal(10.0, 2.0, size=200)
        folds[cell] = {
            "rul_true": y,
            "rul_q10": np.full(200, 9.7),
            "rul_q90": np.full(200, 10.3),
        }
    out = recalibrate_lco_intervals({"per_cell": folds})

    raw_cov = out["raw"]["rul_interval_coverage"]
    new_cov = out["recalibrated"]["rul_interval_coverage"]
    assert raw_cov < 0.30                       # demonstrably overconfident
    assert new_cov > raw_cov                    # recalibration corrects the width
    assert abs(new_cov - NOMINAL_INTERVAL_COVERAGE) < 0.12
    assert out["recalibrated"]["rul_interval_width_mean"] > out["raw"]["rul_interval_width_mean"]


def test_recalibration_does_not_leak_the_held_out_fold():
    """The recalibrator for a fold must be built only from the OTHER folds.
    c1's truths live around 20 with a tight raw interval; c2's truths live
    around 5 with an over-wide raw interval (every c2 point inside it), so
    c2's conformity scores are negative and the conformal E* is negative.
    If c1's own (under-covered, positive-score) data had leaked into its
    recalibration, c1's interval would have been WIDENED and would cover
    ~80%. It must instead be narrowed — coverage ~0% — because the
    recalibrator only ever saw c2's regime."""
    rng = np.random.default_rng(3)
    folds = {}
    for cell, mu, sigma in [("c1", 20.0, 1.0), ("c2", 5.0, 0.01)]:
        y = rng.normal(mu, sigma, size=150)
        folds[cell] = {
            "rul_true": y,
            "rul_q10": np.full(150, mu - 0.2),
            "rul_q90": np.full(150, mu + 0.2),
        }
    out = recalibrate_lco_intervals({"per_cell": folds})
    assert out["recalibrated"]["per_cell"]["c1"] < 0.10


def test_recalibration_result_structure():
    rng = np.random.default_rng(1)
    folds = {
        f"c{i}": {
            "rul_true": rng.normal(5.0, 1.0, size=100),
            "rul_q10": np.full(100, 4.5),
            "rul_q90": np.full(100, 5.5),
        }
        for i in range(3)
    }
    out = recalibrate_lco_intervals({"per_cell": folds})
    assert out["nominal"] == NOMINAL_INTERVAL_COVERAGE
    assert set(out["skipped"]) == set(folds)
    for part in ("raw", "recalibrated"):
        assert "rul_interval_coverage" in out[part]
        assert "rul_interval_width_mean" in out[part]
        assert set(out[part]["per_cell"]) == set(folds)
