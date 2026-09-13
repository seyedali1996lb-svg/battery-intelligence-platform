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

