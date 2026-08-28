"""
Tests for src.ml_anomaly — per-cell IsolationForest novelty detection over
cycle-level features, with honest warmup handling and caveats.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml_anomaly import (
    MLAnomalyDetector,
    detect_anomalous_cycles,
    detect_fleet_anomalies,
    MIN_CYCLES_FOR_FIT,
)


def _healthy_cell(n=200, seed=7):
    rng = np.random.default_rng(seed)
    cap = 2.0 - 0.0005 * np.arange(1, n + 1) + rng.normal(0, 0.002, n)
    res = 0.05 + 0.00002 * np.arange(1, n + 1) + rng.normal(0, 0.0005, n)
    temp = np.full(n, 25.0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "cycle_number": np.arange(1, n + 1),
        "capacity_ah": cap,
        "resistance_ohm": res,
        "temperature_c": temp,
    })


def _with_injections(df, cap_dip_pos=None, temp_spike_pos=None):
    out = df.copy()
    if cap_dip_pos is not None:
        out.loc[out.index[cap_dip_pos], "capacity_ah"] -= 0.5
    if temp_spike_pos is not None:
        out.loc[out.index[temp_spike_pos], "temperature_c"] += 12.0
    return out


# ---------------------------------------------------------------------------
# detect_anomalous_cycles
# ---------------------------------------------------------------------------

def test_flags_injected_anomalies():
    df = _with_injections(_healthy_cell(), cap_dip_pos=100, temp_spike_pos=150)
    report = detect_anomalous_cycles(df)
    # Position 100 -> cycle 101; position 150 -> cycle 151.
    assert 101 in report["flagged_cycles"]
    assert 151 in report["flagged_cycles"]
    assert report["n_flagged"] > 0


def test_warmup_cycles_unscored():
    df = _with_injections(_healthy_cell(), cap_dip_pos=100, temp_spike_pos=150)
    report = detect_anomalous_cycles(df)
    assert report["n_warmup_unscored"] == 30
    assert report["n_scored"] == report["n_cycles"] - 30
    # Warmup rows: no score, honest note, never flagged.
    for row in report["per_cycle"][:30]:
        assert row["anomaly_score"] is None
        assert row["is_anomaly"] is False
        assert "warmup" in row["note"]
    assert all(c >= 31 for c in report["flagged_cycles"])


def test_healthy_cell_flags_low_count():
    # Without injected anomalies, at most ~contamination fraction of the
    # SCORED cycles (170, after 30 warmup) flagged — 5% -> ~8.5 -> <= 10.
    report = detect_anomalous_cycles(_healthy_cell())
    assert report["n_flagged"] <= 10


def test_report_shape_and_caveats():
    report = detect_anomalous_cycles(_healthy_cell())
    assert set(report) >= {"n_cycles", "n_scored", "n_warmup_unscored", "n_flagged",
                           "contamination_assumed", "flagged_cycles", "per_cycle", "caveats"}
    assert report["contamination_assumed"] == 0.05
    assert any("not fault classification" in c for c in report["caveats"])
    assert any("contamination-based" in c for c in report["caveats"])


def test_report_json_serializable_with_nulls():
    report = detect_anomalous_cycles(_healthy_cell())
    payload = json.dumps(report)  # must not produce NaN
    assert "NaN" not in payload


def test_too_few_cycles_raises():
    small = _healthy_cell(n=MIN_CYCLES_FOR_FIT - 1)
    with pytest.raises(ValueError, match="Need at least"):
        detect_anomalous_cycles(small)


def test_exactly_thirty_cycles_raises():
    # 30 total cycles -> 0 with full rolling history (30 warmup) -> refuse.
    with pytest.raises(ValueError):
        detect_anomalous_cycles(_healthy_cell(n=MIN_CYCLES_FOR_FIT))


def test_missing_required_column_raises():
    df = _healthy_cell().drop(columns=["capacity_ah"])
    with pytest.raises(ValueError, match="missing required columns"):
        detect_anomalous_cycles(df)


def test_capacity_only_fit_allowed():
    df = _healthy_cell()[["cycle_number", "capacity_ah"]]
    report = detect_anomalous_cycles(df)
    assert report["n_scored"] == 170
    assert "capacity_ah" in report["feature_columns"]
    assert "fade_rate_30cy" in report["feature_columns"]


def test_deterministic_same_seed():
    a = detect_anomalous_cycles(_healthy_cell())
    b = detect_anomalous_cycles(_healthy_cell())
    assert a["flagged_cycles"] == b["flagged_cycles"]


def test_invalid_contamination_raises():
    with pytest.raises(ValueError):
        MLAnomalyDetector(contamination=0.0)


# ---------------------------------------------------------------------------
# detect_fleet_anomalies
# ---------------------------------------------------------------------------

def test_fleet_aggregates_and_skips_small_cells():
    df = _with_injections(_healthy_cell(), cap_dip_pos=100)
    small = _healthy_cell(n=MIN_CYCLES_FOR_FIT - 1)
    report = detect_fleet_anomalies({"c1": df, "c2": small})
    assert report["summary"]["n_cells"] == 2
    assert report["summary"]["n_flagged_cells"] == 1
    assert report["summary"]["n_flagged_cycles"] == report["per_cell"]["c1"]["n_flagged"]
    assert "c2" in report["summary"]["skipped"]
    assert "c1" in report["per_cell"]
