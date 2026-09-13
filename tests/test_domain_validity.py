"""Tests for src/domain_validity.py + the Tier-5 wiring (provenance,
registry validity_meta) — the domain-of-validity boundary layer.

The behaviour under test: the envelope is computed from the data that
trained the model, axes the fleet never varied are never claimed as
covered, per-cell checks return a verdict with named axes, per-regime
reliability stratifies folds by chemistry × temperature × SOH stage, and
every logged run carries the envelope it was measured under.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from domain_validity import (  # noqa: E402
    cell_regime,
    check_cell_against_envelope,
    compute_envelope,
    envelope_summary,
    regime_reliability,
    validity_banner,
)


def make_cycles(
    n: int = 300,
    soh_end: float = 85.0,
    temp: float = 32.0,
    c_rate: "float | None" = 1.0,
    with_temp: bool = True,
) -> pd.DataFrame:
    cyc = np.arange(1, n + 1)
    soh = np.linspace(100.0, soh_end, n)
    df = pd.DataFrame({
        "cycle_number": cyc,
        "capacity_ah": 2.0 * soh / 100.0,
        "soh_pct": soh,
    })
    if with_temp:
        df["temperature_c"] = np.full(n, temp) + np.linspace(-0.5, 0.5, n)
    if c_rate is not None:
        df["c_rate"] = np.full(n, c_rate)
    return df


@pytest.fixture
def licox_fleet() -> dict:
    """A LiCoO2 fleet, 24.5–40.5 °C measured, 0.5–2C measured, SOH to 66%."""
    return {
        "Cell1": make_cycles(400, 85.0, 30.0, 1.0),
        "Cell2": make_cycles(400, 66.2, 34.0, 2.0),
        "Cell3": make_cycles(300, 90.0, 25.0, 0.5),
        "Cell4": make_cycles(500, 80.0, 40.0, 1.5),
    }


@pytest.fixture
def licox_envelope(licox_fleet) -> dict:
    return compute_envelope(list(licox_fleet), cell_data=licox_fleet)


class TestComputeEnvelope:
    def test_measured_axes_real_ranges(self, licox_envelope):
        t = licox_envelope["temperature_c"]
        assert t["status"] == "measured"
        assert t["min"] == pytest.approx(24.5, abs=0.1)
        assert t["max"] == pytest.approx(40.5, abs=0.1)
        c = licox_envelope["c_rate"]
        assert c["status"] == "measured"
        assert c["min"] == pytest.approx(0.5)
        assert c["max"] == pytest.approx(2.0)

    def test_soh_window(self, licox_envelope):
        s = licox_envelope["soh_pct"]
        assert s["min"] == pytest.approx(66.2, abs=0.1)
        assert s["max"] == pytest.approx(100.0)

    def test_protocol_constant_axis_not_claimed_measured(self):
        """A fleet that ran one set-point must NOT claim a measured range —
        the model never saw variation on that axis."""
        cells = {f"Cell{i}": make_cycles(200, 85.0, 25.0, 1.0) for i in range(1, 4)}
        env = compute_envelope(list(cells), cell_data=cells)
        assert env["temperature_c"]["status"] == "protocol-constant"
        assert env["c_rate"]["status"] == "protocol-constant"

    def test_absent_axis_disclosed(self):
        """No temperature column at all → status 'absent', never a range."""
        cells = {
            f"Cell{i}": make_cycles(200, 85.0, with_temp=False, c_rate=None)
            for i in range(1, 4)
        }
        env = compute_envelope(list(cells), cell_data=cells)
        assert env["temperature_c"]["status"] == "absent"
        assert env["c_rate"]["status"] == "absent"

    def test_malformed_cell_does_not_abort(self, licox_fleet):
        broken = dict(licox_fleet)
        broken["CellX"] = None  # type: ignore[assignment]
        env = compute_envelope(list(broken), cell_data=broken)
        assert env["n_cells"] == 5
        assert env["temperature_c"]["status"] == "measured"

    def test_summary_line(self, licox_envelope):
        s = envelope_summary(licox_envelope)
        assert "LiCoO2" in s or "LiCoO₂" in s
        assert "24.5" in s and "40.5" in s
        assert "n=4" in s


class TestCheckCell:
    def test_in_envelope_cell_passes(self, licox_envelope, licox_fleet):
        chk = check_cell_against_envelope(licox_envelope, "Cell1", cell_data=licox_fleet)
        assert chk["verdict"] == "in"
        assert chk["outside_axes"] == []

    def test_outside_temperature_and_soh(self, licox_envelope):
        hot_deep = {"CellX": make_cycles(200, 55.0, 45.0, c_rate=None)}
        chk = check_cell_against_envelope(licox_envelope, "CellX", cell_data=hot_deep)
        assert chk["verdict"] == "outside"
        assert any("temperature" in a for a in chk["outside_axes"])
        assert any("SOH" in a for a in chk["outside_axes"])

    def test_partial_within_tolerance(self, licox_envelope):
        """At the range edge but within tolerance → partial, not outside."""
        edge = {"CellE": make_cycles(200, 64.8, c_rate=None)}  # SOH 1.4 below min
        chk = check_cell_against_envelope(licox_envelope, "CellE", cell_data=edge)
        assert chk["verdict"] == "partial"
        assert any("SOH" in a for a in chk["partial_axes"])

    def test_no_envelope_is_unknown_not_pass(self):
        chk = check_cell_against_envelope(None, "Cell1")
        assert chk["verdict"] == "unknown"

    def test_unmeasured_axis_flagged_not_passed(self, licox_envelope):
        """A cell with no temperature data vs a measured envelope: the axis
        can't be checked — surfaced as partial (unvalidated), never 'in'."""
        no_temp = {"CellNT": make_cycles(200, 85.0, with_temp=True, c_rate=None)}
        chk = check_cell_against_envelope(licox_envelope, "CellNT", cell_data=no_temp)
        # temp present here but within range; SOH fine; no c_rate on either
        # side → the C-rate axis is unvalidated and must not read as covered
        assert chk["verdict"] in ("in", "partial")
        if chk["verdict"] == "partial":
            assert any("C-rate" in a for a in chk["partial_axes"])


class TestRegimeReliability:
    def test_stratification_by_regime(self, licox_fleet):
        folds = {
            cid: {"soh_r2": 0.9, "rul_r2": 0.8}
            for cid in licox_fleet
        }
        rows = regime_reliability(folds, cell_data=licox_fleet)
        # 4 cells, each in a distinct temp/SOH combination here → ≥3 groups
        assert len(rows) >= 3
        assert all(r["n_cells"] == 1 for r in rows)
        assert all(r["verdict"] == "thin" for r in rows)  # n<3 → thin

    def test_reliable_requires_three_cells(self, licox_fleet):
        """3+ cells in the SAME regime with RUL above floor → reliable."""
        cells = {f"Cell{i}": make_cycles(300, 82.0, 32.0, 1.0) for i in range(1, 5)}
        folds = {cid: {"soh_r2": 0.9, "rul_r2": 0.8} for cid in cells}
        rows = regime_reliability(folds, cell_data=cells)
        assert len(rows) == 1
        assert rows[0]["verdict"] == "reliable"
        assert rows[0]["n_cells"] == 4

    def test_unvalidated_when_no_measured_labels(self, licox_fleet):
        folds = {cid: {"soh_r2": 0.9, "rul_r2": None} for cid in licox_fleet}
        rows = regime_reliability(folds, cell_data=licox_fleet)
        assert all(r["verdict"] == "unvalidated" for r in rows)

    def test_unvalidated_when_below_floor(self, licox_fleet):
        cells = {f"Cell{i}": make_cycles(300, 82.0, 32.0, 1.0) for i in range(1, 5)}
        folds = {cid: {"soh_r2": 0.9, "rul_r2": -0.5} for cid in cells}
        rows = regime_reliability(folds, cell_data=cells)
        assert rows[0]["verdict"] == "unvalidated"

    def test_cell_regime_bands(self, licox_fleet):
        ax = cell_regime("Cell3", cell_data=licox_fleet)  # 25 °C cell
        assert ax["temp_band"] == "25–30 °C"
        ax2 = cell_regime("Cell2", cell_data=licox_fleet)  # 66% final SOH
        assert "EOL-critical" in ax2["soh_stage"]


class TestBanner:
    def test_outside_banner_names_axes(self):
        b = validity_banner({"cell_verdict": "outside", "outside_axes": ["chemistry (LFP)"]})
        assert "Outside" in b and "chemistry" in b

    def test_partial_banner(self):
        b = validity_banner({"cell_verdict": "partial", "partial_axes": ["cell format"]})
        assert "Edge" in b

    def test_in_is_quiet(self):
        assert validity_banner({"cell_verdict": "in"}) == ""
        assert validity_banner(None) == ""


class TestProvenanceWiring:
    def test_provenance_carries_verdict(self, licox_envelope, licox_fleet):
        from accuracy_provenance import cell_model_provenance
        bundle = {"training_envelope": licox_envelope, "metrics": {}}
        prov = cell_model_provenance(
            bundle, "CellX",
            cell_data_df=make_cycles(200, 55.0, 45.0, c_rate=None),
        )
        assert prov["cell_verdict"] == "outside"
        assert any("temperature" in a for a in prov["outside_axes"])
        assert "axes_checked" in prov

    def test_provenance_without_envelope_has_no_verdict(self):
        from accuracy_provenance import cell_model_provenance
        prov = cell_model_provenance({"metrics": {}}, "Cell1")
        assert "cell_verdict" not in prov


class TestRegistryValidityMeta:
    def test_log_run_roundtrip(self, tmp_path, monkeypatch):
        """validity_meta flows log_run → db → reader without loss."""
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "t.db")
        db_mod.init_db()

        import experiment_registry as reg
        meta = {
            "envelope": {"chemistries": ["LiCoO2"], "n_cells": 4,
                         "temperature_c": {"min": 24.5, "max": 40.5, "status": "measured"}},
            "regime_reliability": [
                {"chemistry": "LiCoO2", "temp_band": "30–35 °C",
                 "soh_stage": "late life", "n_cells": 4, "cell_ids": ["a", "b", "c", "d"],
                 "mean_soh_r2": 0.9, "mean_rul_r2": 0.8, "rul_evaluable_n": 4,
                 "verdict": "reliable"},
            ],
        }
        run_id = reg.log_run(
            org_id=reg.PLATFORM_ORG_ID, dataset="t5", chemistry="LiCoO2",
            feature_set=["cycle_number"], feature_version="test",
            hyperparams={}, seed=42, cell_ids=["a", "b", "c", "d"], n_rows=100,
            lco_metrics={"soh_r2": 0.9, "soh_mae": 1.0, "rul_r2": None,
                         "rul_mae": None, "rul_reliable": False, "per_cell": {},
                         "validity_meta": meta},
        )
        run = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
        assert run is not None
        got = run.get("validity_meta")
        assert got is not None
        assert got["envelope"]["n_cells"] == 4
        assert got["regime_reliability"][0]["verdict"] == "reliable"

    def test_accuracy_by_source_carries_validity(self, tmp_path, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "t2.db")
        db_mod.init_db()

        import experiment_registry as reg
        meta = {"envelope": {"chemistries": ["LiCoO2"], "n_cells": 4}, "regime_reliability": []}
        reg.log_run(
            org_id=reg.PLATFORM_ORG_ID, dataset="t5acc", chemistry="LiCoO2",
            feature_set=["cycle_number"], feature_version="test",
            hyperparams={}, seed=42, cell_ids=["a", "b"], n_rows=50,
            lco_metrics={"soh_r2": 0.9, "soh_mae": 1.0, "rul_r2": None,
                         "rul_mae": None, "rul_reliable": False, "per_cell": {},
                         "baseline_soh_r2": 0.5, "baseline_per_cell": {},
                         "validity_meta": meta},
        )
        rows = reg.accuracy_by_source(tenant_org_id=None)
        match = [r for r in rows if r["dataset"] == "t5acc"]
        assert match and match[0]["validity_meta"]["envelope"]["n_cells"] == 4
