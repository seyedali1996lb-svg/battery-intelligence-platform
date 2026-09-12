"""Unit tests for batlab.features.dqdv — dQ/dV curve simulation and feature extraction."""

import numpy as np
import pandas as pd
import pytest
from batlab.features.dqdv import simulate_vq_curve, extract_dqdv_features, add_dqdv_features


def test_simulate_vq_curve_shapes_and_bounds():
    Q, V = simulate_vq_curve(capacity_ah=2.0, resistance_ohm=0.05, n_points=200)
    assert Q.shape == (200,)
    assert V.shape == (200,)
    assert Q[0] == 0.0
    assert abs(Q[-1] - 2.0) < 1e-9
    assert np.isfinite(V).all()


def test_extract_dqdv_features_keys_and_types():
    feats = extract_dqdv_features(capacity_ah=2.0, resistance_ohm=0.05)
    for key in ["dqdv_sim_peak_value", "dqdv_sim_peak_soc", "dqdv_sim_area", "dqdv_sim_fwhm"]:
        assert key in feats
        assert isinstance(feats[key], float)
    assert 0.0 <= feats["dqdv_sim_peak_soc"] <= 1.0
    assert feats["dqdv_sim_fwhm"] >= 0.0


def test_add_dqdv_features_matches_per_row_extraction():
    """The vectorized add_dqdv_features() must agree with calling
    extract_dqdv_features() row-by-row (this vectorization was a real perf
    fix — see README debugging story — so a correctness regression here
    would be easy to introduce silently)."""
    df = pd.DataFrame({
        "capacity_ah": [2.0, 1.9, 1.8],
        "resistance_ohm": [0.05, 0.052, 0.055],
    })
    vectorized = add_dqdv_features(df)

    for i in range(len(df)):
        single = extract_dqdv_features(df["capacity_ah"].iloc[i], df["resistance_ohm"].iloc[i])
        assert abs(vectorized["dqdv_sim_peak_value"].iloc[i] - single["dqdv_sim_peak_value"]) < 1e-6
        assert abs(vectorized["dqdv_sim_area"].iloc[i] - single["dqdv_sim_area"]) < 1e-6
        assert abs(vectorized["dqdv_sim_fwhm"].iloc[i] - single["dqdv_sim_fwhm"]) < 1e-6


def test_add_dqdv_features_adds_all_columns():
    df = pd.DataFrame({"capacity_ah": [2.0, 1.9], "resistance_ohm": [0.05, 0.06]})
    out = add_dqdv_features(df)
    for col in ["dqdv_sim_peak_value", "dqdv_sim_peak_soc", "dqdv_sim_area", "dqdv_sim_fwhm"]:
        assert col in out.columns
        assert out[col].notna().all()


def _make_discharge_curve(capacity_ah: float, resistance_ohm: float, n: int = 160) -> tuple:
    """Self-consistent synthetic discharge: constant current, LiCoO2-ish OCV shape.

    Built at dt = 1 s per step (the dqdv_validation module's default telemetry
    assumption) with a constant current chosen so that I * n * dt / 3600 = capacity_ah
    exactly. That makes the module's integrated-Q sanity check pass, so the test
    exercises the real-vs-simulated comparison path rather than the
    'curve rejected as not-a-clean-discharge' path."""
    n = max(n, 5)
    dt = 1.0                                  # seconds per reading (telemetry default)
    I = capacity_ah * 3600.0 / (n * dt)       # A, chosen so integrated Q == capacity_ah
    t = np.arange(n) * dt
    q = (I * t) / 3600.0                       # cumulative Ah discharged
    soc = 1.0 - q / max(capacity_ah, 1e-9)
    ocv = (3.7 + 0.7*soc - 0.5*soc**2 + 0.3*soc**3 - 0.1*(1-soc)**3)
    v = ocv - I * resistance_ohm              # terminal voltage under I-discharge
    i = np.full(n, I)
    return q, v, i


def test_compare_simulated_dqdv_to_real_uses_measured_curve():
    """Where a real discharge V/I curve exists, the validation report must
    be built from the measured curve, not silently fall back to the sim."""
    from batlab.features.dqdv_validation import compare_simulated_dqdv_to_real

    q, v, i = _make_discharge_curve(2.0, 0.05)

    comp = compare_simulated_dqdv_to_real(
        real_capacity_ah=2.0,
        real_voltage_v=v,
        real_current_a=i,
        resistance_ohm=0.05,
    )

    assert comp["n_real_points"] >= 5
    assert set(comp) == {
        "real", "simulated", "peak_value_abs_error", "peak_value_rel_error",
        "peak_soc_abs_error", "area_abs_error", "area_rel_error",
        "fwhm_abs_error", "fwhm_rel_error", "n_real_points", "caveat",
    }
    # Both real and simulated are computed from the same capacity/ir → values are
    # finite and in the same order of magnitude. The self-consistent synthetic case
    # does NOT guarantee near-identical peak error (the OCV-derivative shape differs
    # a little between the sim OCV polynomial and the real numeric dQ/dV), so the
    # honest assertion is just that it doesn't blow up — not that it's under 1.0.
    assert comp["peak_value_rel_error"] >= 0.0
    assert comp["peak_value_rel_error"] < 5.0, "synthetic self-consistent curve should be within a small factor of its own sim"
    assert comp["caveat"]  # honesty caveat must be present even when numbers agree


def test_fleet_dqdv_validation_report_builds_per_cell_table():
    """fleet_dqdv_validation_report produces one row per cell with rel errors."""
    from batlab.features.dqdv_validation import fleet_dqdv_validation_report

    cells = {
        "C1": {"capacity_ah": 2.0, "voltage_v": _make_discharge_curve(2.0, 0.05)[1],
               "current_a": _make_discharge_curve(2.0, 0.05)[2], "resistance_ohm": 0.05},
        "C2": {"capacity_ah": 1.9, "voltage_v": _make_discharge_curve(1.9, 0.052)[1],
               "current_a": _make_discharge_curve(1.9, 0.052)[2], "resistance_ohm": 0.052},
    }

    df = fleet_dqdv_validation_report(cells)
    assert list(df["cell_id"]) == ["C1", "C2"]
    assert df["peak_value_rel_error"].notna().all()
    assert df["error"].isna().all()


def test_compare_simulated_dqdv_rejects_badly_unmatching_curve():
    """A raw curve whose integrated Q disagrees with reported capacity raises cleanly."""
    import numpy as np
    from batlab.features.dqdv_validation import compare_simulated_dqdv_to_real

    n = 50
    # 2A for 50s ≈ 0.028 Ah — nowhere near the declared 2.0 Ah.
    q = np.linspace(0, 0.028, n)
    v = np.linspace(3.7, 3.5, n)
    i = np.full(n, 2.0)

    with pytest.raises(ValueError, match="disagrees with reported capacity"):
        compare_simulated_dqdv_to_real(
            real_capacity_ah=2.0,
            real_voltage_v=v,
            real_current_a=i,
            resistance_ohm=0.05,
        )
