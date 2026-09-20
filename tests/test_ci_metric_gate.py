"""
CI regression gate: the fixture fleet's headline numbers must stay inside
their declared floors/baselines (tests/metric_gate_expectations.json).

Run directly:
    python -m pytest tests/test_ci_metric_gate.py -q -m metric_gate
(or via the CI workflow step, which runs the same module).

The gate is deliberately ALSO a standalone script (scripts/check_metric_gate.py)
so the same check runs against the registry's real logged runs, not just the
fixture fleet — a regression that only shows on real data still fails a
release check. This test pins the fleet path end to end: real run_lco(),
real expectations file, real verdict.
"""

import sys as _sys
import os as _os

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
# tests/ itself (no __init__.py): makes `import metric_gate_fleet` work when
# this file is run outside pytest's own path insertion.
_tests = _os.path.dirname(_os.path.abspath(__file__))
if _tests not in _sys.path:
    _sys.path.insert(0, _tests)

import pytest


def _run_gate():
    import metric_gate_fleet as _mgf

    return _mgf.run_metric_gate()


def test_metric_gate_fleet_is_deterministic():
    """The fingerprints must be byte-stable across a rebuild: the gate's
    meaning depends on 'same fleet' being checkable, not assumed."""
    import metric_gate_fleet as _mgf
    from batlab.validation.fingerprints import dataset_fingerprint

    cells_a = _mgf.build_gate_fleet()
    fp_a = dataset_fingerprint(cells_a)
    # Rebuild from cache and re-fingerprint.
    cells_b = _mgf.build_gate_fleet()
    fp_b = dataset_fingerprint(cells_b)
    assert fp_a["dataset_sha256"] == fp_b["dataset_sha256"]
    assert fp_a["cell_digests"] == fp_b["cell_digests"]


def test_run_lco_result_carries_fingerprint():
    """Every LCO result now pins its data and environment."""
    import metric_gate_fleet as _mgf

    lco = _mgf.run_gate_fleet_lco(_mgf.build_gate_fleet())
    fp = lco["fingerprint"]
    assert set(fp) == {"dataset", "environment"}
    assert fp["dataset"]["n_cells"] == len(_mgf.GATE_FLEET_CELL_IDS)
    assert len(fp["dataset"]["cell_digests"]) == fp["dataset"]["n_cells"]
    for pkg in ("python", "numpy", "pandas", "scikit-learn"):
        assert pkg in fp["environment"]


def test_gate_fleet_baseline_is_tracked_and_whole():
    """baseline_soh_r2 is part of the promise now (see GATE_TRACKED_METRICS):
    every fold scored, no blank target row set aside, and the number the gate
    pins is the one the model's advantage is measured against."""
    import math

    import metric_gate_fleet as _mgf

    baseline = _mgf.compute_fleet_baseline()
    assert baseline["n_cells"] == len(_mgf.GATE_FLEET_CELL_IDS)
    assert baseline["n_folds_scored"] == len(_mgf.GATE_FLEET_CELL_IDS)
    assert baseline["n_folds_skipped"] == 0
    assert baseline["n_nonfinite_target_rows"] == 0
    assert math.isfinite(baseline["baseline_soh_r2"])


def test_blank_target_row_does_not_delete_the_baseline():
    """The regression this tracking exists to catch.

    One blank capacity row used to make the trivial LinearRegression raise
    ("Input y contains NaN"), which app/_data.py's bare `except` recorded as
    None — a headline that stopped being computable, silently. On Severson it
    happened twice (S-b1c0 cycle 11, S-b1c18 cycle 39). Here the row is set
    aside, counted, every fold is still scored, and the value stays inside the
    declared tolerance — while the old outcomes (None, or a NaN handed to the
    gate) both FAIL."""
    import math

    import numpy as np

    import metric_gate_fleet as _mgf
    from batlab.features.engineering import build_features, get_model_matrix
    from batlab.validation.lco import unwrap_cell_data
    from batlab.validation.metric_gate import check_metric

    cells = _mgf.build_gate_fleet()
    raw = unwrap_cell_data(cells)
    victim = sorted(raw)[0]
    df = raw[victim].copy()
    blank_row = df.index[len(df) // 2]
    df.loc[blank_row, ["capacity_ah", "soh_pct"]] = float("nan")
    poisoned = dict(raw)
    poisoned[victim] = df

    clean = _mgf.compute_fleet_baseline(cells)
    dirty = _mgf.compute_fleet_baseline(poisoned)

    assert dirty["n_nonfinite_target_rows"] == 1
    assert dirty["n_folds_scored"] == clean["n_folds_scored"]
    assert math.isfinite(dirty["baseline_soh_r2"])
    # Measured |delta| is ~8e-4 against a 5% tolerance (0.088) — the row is
    # dropped, not imputed, so the number barely moves.
    assert abs(dirty["baseline_soh_r2"] - clean["baseline_soh_r2"]) < 0.01

    expectation = _mgf.load_expectations_json()["metrics"]["baseline_soh_r2"]
    assert check_metric("baseline_soh_r2", clean["baseline_soh_r2"], expectation)["verdict"] == "pass"
    assert check_metric("baseline_soh_r2", dirty["baseline_soh_r2"], expectation)["verdict"] == "pass"
    # The two ways this used to disappear, both caught.
    assert check_metric("baseline_soh_r2", None, expectation)["verdict"] == "fail"
    assert check_metric("baseline_soh_r2", float("nan"), expectation)["verdict"] == "fail"

    # Why blanking the row cannot move the model's own number either: the same
    # row is dropped from run_lco's model matrix (dod_proxy = capacity/initial
    # is a feature), so the baseline and the model are on the same population.
    _, y_soh, _ = get_model_matrix(build_features(df, cell_id=victim))
    assert not np.isnan(np.asarray(y_soh, dtype=float)).any()


@pytest.mark.metric_gate
def test_ci_metric_gate_passes():
    """THE gate: fixture-fleet headline numbers within declared floors and
    baselines. An unexplained move fails CI; the fix is deliberate
    (scripts/update_metric_baselines.py), never silent."""
    import io
    import contextlib

    from batlab.validation.metric_gate import format_gate_report

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = _run_gate()
    assert res["gate"]["verdict"] == "pass", format_gate_report(res["gate"])

