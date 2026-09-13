"""Tests for Tier-4 modeling additions.

Covers: hierarchical partial-pooling LCO (incl. chemistry-conditional
prior), the GBRT+PINN ensemble with inner-CV weight selection, the
degraded-input robustness harness, the condition-axis audit, SoP proxy
validation, physics-parameter provenance, and the registry study runner
including its exclusion from the LCO accuracy tables.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_cycles_df


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Throwaway SQLite for the registry tests (same pattern as
    tests/test_experiment_registry.py)."""
    import db as db_module
    test_db_path = tmp_path / "test_tier4.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _lin(n_cycles=300, fade=0.0008, cap0=2.0, noise=0.0, seed=0, r0=0.05, rg=5e-5):
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    capacity = cap0 - fade * cycles + (rng.normal(0, noise, n_cycles) if noise else 0.0)
    df = pd.DataFrame({
        "cycle_number": cycles,
        "capacity_ah": capacity,
        "resistance_ohm": r0 + rg * cycles,
        "temperature_c": np.full(n_cycles, 25.0),
    })
    df["soh_pct"] = df["capacity_ah"] / cap0 * 100.0
    return df


# ---------------------------------------------------------------------------
# Hierarchical partial pooling
# ---------------------------------------------------------------------------

def test_hierarchical_recovers_linear_fleet():
    """On a noiseless linear fleet the model should be near-exact."""
    from batlab.validation.hierarchical_lco import run_hierarchical_lco
    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, seed=i) for i in range(5)}
    res = run_hierarchical_lco(cells)
    assert res["soh_r2"] > 0.99
    assert res["rul_r2"] > 0.99
    assert res["prior_scope"] == "fleet-local"
    assert set(res["per_cell"]) == set(cells)


def test_hierarchical_shrinks_noisy_cells_toward_prior():
    """With noise, the shrinkage machinery engages: the shrunk log rate lies
    between the local estimate and the prior mean."""
    from batlab.validation.hierarchical_lco import run_hierarchical_lco
    cells = {f"N{i}": _lin(fade=0.0005 + 2e-5 * i, n_cycles=400, noise=0.004, seed=i) for i in range(6)}
    res = run_hierarchical_lco(cells)
    assert np.isfinite(res["soh_r2"]) and res["soh_r2"] > 0.5
    for theta in res["per_cell_theta"].values():
        lo = min(theta["local_log_fade"], theta["prior_mu"])
        hi = max(theta["local_log_fade"], theta["prior_mu"])
        assert lo - 1e-9 <= theta["shrunk_log_fade"] <= hi + 1e-9
        assert 0.0 <= theta["shrinkage_weight_prior"] <= 1.0


def test_hierarchical_chemistry_conditional_prior():
    """With chemistry metadata the prior pools per chemistry: each fold's
    prior is estimated ONLY from same-chemistry training cells, so A-folds
    and B-folds see different fleet priors (per-fold by design — the prior
    excludes the held-out cell)."""
    from batlab.validation.hierarchical_lco import run_hierarchical_lco
    cells = {
        **{f"A{i}": _lin(fade=0.0008 + i * 0.0002, seed=i) for i in range(4)},
        **{f"B{i}": _lin(fade=0.0020 + i * 0.0003, seed=10 + i) for i in range(4)},
    }
    chem = {**{f"A{i}": "LiCoO2" for i in range(4)}, **{f"B{i}": "LFP" for i in range(4)}}
    res = run_hierarchical_lco(cells, chemistry_by_cell=chem)
    assert res["prior_scope"] == "per-chemistry"
    # Every A-fold prior sits at the A fleet's log-fade scale (~log 0.001),
    # every B-fold prior at B's (~log 0.0025): the two groups are fully
    # separated — no A-fold prior exceeds any B-fold prior.
    a_mus = [t["prior_mu"] for c, t in res["per_cell_theta"].items() if c.startswith("A")]
    b_mus = [t["prior_mu"] for c, t in res["per_cell_theta"].items() if c.startswith("B")]
    assert max(a_mus) < min(b_mus)


def test_hierarchical_single_cell_fleet_is_empty_not_crash():
    from batlab.validation.hierarchical_lco import run_hierarchical_lco
    res = run_hierarchical_lco({"Only": _lin()})
    assert res["per_cell"] == {}
    assert np.isnan(res["soh_r2"])


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def test_ensemble_runs_and_returns_weights():
    from batlab.validation.ensemble_lco import run_ensemble_lco
    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, noise=0.003, seed=i) for i in range(4)}
    chem = {f"Cell{i}": "LiCoO2" for i in range(4)}
    res = run_ensemble_lco(cells, chemistry_by_cell=chem, seed=42)
    assert np.isfinite(res["soh_r2"])
    assert set(res["per_cell_weights"]) == set(cells)
    # Weights come from the fixed grid.
    for w in res["per_cell_weights"].values():
        assert abs(w * 10 - round(w * 10)) < 1e-9
    assert res["weights_by_chemistry"]["LiCoO2"]["n_folds"] == 4


def test_ensemble_collapses_to_pure_member_when_dominant():
    """On near-linear noiseless data the GBRT dominates: the inner CV should
    pick w=1.0 (pure GBRT), which is the honest outcome, not a forced blend."""
    from batlab.validation.ensemble_lco import run_ensemble_lco
    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, noise=0.001, seed=i) for i in range(4)}
    res = run_ensemble_lco(cells, seed=42)
    assert max(res["per_cell_weights"].values()) == 1.0


def test_ensemble_needs_two_cells():
    from batlab.validation.ensemble_lco import run_ensemble_lco
    res = run_ensemble_lco({"Only": _lin()})
    assert res["per_cell"] == {}


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_degrade_modes_shape_and_determinism():
    from batlab.validation.robustness import degrade_cycles
    df = make_cycles_df(n_cycles=300)
    d1 = degrade_cycles(df, "missing_cycles", 0.2, seed=7)
    d2 = degrade_cycles(df, "missing_cycles", 0.2, seed=7)
    assert len(d1) == 240
    assert d1.equals(d2)  # deterministic
    assert len(degrade_cycles(df, "bms_reset", 3, seed=7)) == 300 - 3 * 15
    spiked = degrade_cycles(df, "current_spike", 5, seed=7)
    assert (spiked["resistance_ohm"] > df["resistance_ohm"]).sum() >= 5
    with pytest.raises(ValueError):
        degrade_cycles(df, "unknown_mode", 1)


def test_robustness_lco_reports_baseline_and_deltas():
    from batlab.validation.robustness import run_robustness_lco
    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, n_cycles=250, noise=0.004, seed=i) for i in range(5)}
    res = run_robustness_lco(cells)
    assert np.isfinite(res["baseline"]["soh_r2"])
    assert len(res["scenarios"]) == 9
    for s in res["scenarios"]:
        assert "soh_r2_delta" in s
    assert set(res["failure_threshold"]) == {"missing_cycles", "bms_reset", "current_spike"}


# ---------------------------------------------------------------------------
# Condition axes
# ---------------------------------------------------------------------------

def test_condition_axis_audit_detects_measured_axis():
    from batlab.features.condition_axes import condition_axis_audit
    cells = {
        "A": _lin(seed=0).assign(temperature_c=25.0),
        "B": _lin(seed=1).assign(temperature_c=35.0),
        "C": _lin(seed=2).assign(temperature_c=45.0),
    }
    rows = {r["axis"]: r for r in condition_axis_audit(cells)}
    assert rows["temperature_c"]["status"] == "measured"
    assert rows["temperature_c"]["usable_for_modeling"] is True
    # C-rate absent entirely
    assert rows["c_rate"]["status"] == "absent"


def test_condition_axis_audit_detects_protocol_constant():
    from batlab.features.condition_axes import condition_axis_audit
    cells = {f"C{i}": _lin(seed=i) for i in range(4)}  # all 25.0 °C
    rows = {r["axis"]: r for r in condition_axis_audit(cells)}
    assert rows["temperature_c"]["status"] == "protocol-constant"
    assert rows["temperature_c"]["usable_for_modeling"] is False


def test_condition_axis_audit_detects_zero_sentinels():
    from batlab.features.condition_axes import condition_axis_audit
    good = _lin(seed=0).assign(temperature_c=25.0)
    poisoned = _lin(seed=1).assign(temperature_c=35.0)  # cross-cell spread is real
    poisoned.loc[poisoned.index[:50], "temperature_c"] = 0.0  # the Severson pattern
    rows = {r["axis"]: r for r in condition_axis_audit({"G": good, "P": poisoned})}
    assert rows["temperature_c"]["sentinel_cells"] == 1
    # Sentinel rows are excluded, not treated as 0 °C measurements: the
    # per-cell means stay separated, so the axis remains usable.
    assert rows["temperature_c"]["status"] == "measured"
    assert rows["temperature_c"]["usable_for_modeling"] is True


# ---------------------------------------------------------------------------
# SoP proxy validation
# ---------------------------------------------------------------------------

def test_sop_validation_flags_consistent_fleet():
    from batlab.validation.sop_validation import sop_proxy_validation
    cells = {}
    for i in range(3):
        df = _lin(seed=i)
        r0 = float(df["resistance_ohm"].iloc[0])
        df["sop_pct"] = r0 / df["resistance_ohm"] * 100.0
        cells[f"C{i}"] = df
    res = sop_proxy_validation(cells)
    assert res["fleet_consistent_fraction"] == 1.0
    assert res["uncertainty_band_pct"] == 25.0
    assert "PROXY" in res["scoping_label"]


def test_sop_validation_flags_drift():
    from batlab.validation.sop_validation import sop_proxy_validation
    df = _lin(seed=0)
    r0 = float(df["resistance_ohm"].iloc[0])
    df["sop_pct"] = r0 / df["resistance_ohm"] * 100.0
    # Corrupt the served SoP away from its definition.
    df.loc[df.index[100:], "sop_pct"] *= 0.7
    res = sop_proxy_validation({"C0": df})
    assert res["fleet_consistent_fraction"] == 0.0
    assert res["per_cell"]["C0"]["consistent"] is False


def test_sop_validation_no_resistance():
    from batlab.validation.sop_validation import sop_proxy_validation
    df = make_cycles_df(n_cycles=100).drop(columns=["resistance_ohm"], errors="ignore")
    res = sop_proxy_validation({"C0": df})
    assert res["per_cell"] == {}
    assert "not computed" in res["verdict"]


# ---------------------------------------------------------------------------
# Physics provenance
# ---------------------------------------------------------------------------

def test_physics_provenance_tiers():
    from batlab.validation.physics_scoping import physics_parameter_provenance
    strong = _lin(seed=0).assign(physics_fit_r2=0.95)
    weak = _lin(seed=1).assign(physics_fit_r2=0.4)
    plain = _lin(seed=2)
    res = physics_parameter_provenance({"S": strong, "W": weak, "P": plain}, source_kind="nasa")
    assert res["features_available"] is True
    assert res["per_cell"]["S"]["tier"] == "strong"
    assert res["per_cell"]["W"]["tier"] == "weak"
    assert res["per_cell"]["P"]["tier"] == "absent"
    assert res["tier_counts"] == {"strong": 1, "usable": 0, "weak": 1, "absent": 1}
    assert "EIS" in res["validation_gap"]
    assert "NOT independent measurements" in res["provenance_statement"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_modeling_study_logs_and_excludes(db):
    """The Tier-4 study logs hierarchical/ensemble rows keyed by model_kind,
    and accuracy_by_source never mixes them (or robustness rows) into the
    per-chemistry GBRT accuracy table."""
    import experiment_registry as reg

    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, n_cycles=250, noise=0.003, seed=i) for i in range(4)}
    chem = {f"Cell{i}": "LiCoO2" for i in range(4)}
    # Seed the GBRT row the study joins to.
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v12-rul-label-provenance",
        hyperparams={}, seed=42, cell_ids=list(cells), n_rows=1000,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.95, "rul_mae": 10.0, "rul_r2": 0.7,
                     "rul_reliable": True, "per_cell": {}, "baseline_soh_r2": 0.60},
    )
    out = reg.run_modeling_benchmark_study(
        {"nasa": cells}, refresh=True,
    )
    kinds = {r["model_kind"] for r in out}
    assert kinds == {"hierarchical", "ensemble"}

    rows = reg.modeling_benchmark()
    assert {(r["dataset"], r["model_kind"]) for r in rows} == {("nasa", "hierarchical"), ("nasa", "ensemble")}
    for r in rows:
        assert r["gbrt_soh_r2"] == 0.95

    # accuracy_by_source must show exactly one GBRT row for nasa.
    acc = reg.accuracy_by_source()
    nasa_rows = [a for a in acc if a["dataset"] == "nasa"]
    assert len(nasa_rows) == 1

    # A robustness row must be excluded as well.
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa_robustness", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v12-rul-label-provenance",
        hyperparams={}, seed=42, cell_ids=list(cells), n_rows=1000,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.1, "rul_mae": 10.0, "rul_r2": 0.1,
                     "rul_reliable": False, "per_cell": {}, "baseline_soh_r2": 0.05},
    )
    acc2 = reg.accuracy_by_source()
    assert all(a["dataset"] != "nasa_robustness" for a in acc2)


def test_registry_robustness_study_round_trip(db):
    import experiment_registry as reg

    cells = {f"Cell{i}": _lin(fade=0.0008 + i * 0.0002, n_cycles=220, noise=0.003, seed=i) for i in range(4)}
    out = reg.run_robustness_study({"nasa": cells}, refresh=True)
    assert len(out) == 1

    rows = reg.robustness_study()
    assert len(rows) == 1
    rr = rows[0]
    assert rr["dataset"] == "nasa"
    assert len(rr["scenarios"]) == 9
    modes = {s["mode"] for s in rr["scenarios"]}
    assert modes == {"missing_cycles", "bms_reset", "current_spike"}
    # Idempotent: second call logs nothing new.
    out2 = reg.run_robustness_study({"nasa": cells})
    assert out2 == []
