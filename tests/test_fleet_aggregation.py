"""
Tests for src.fleet_aggregation — per-cell health/SoC headroom aggregated
into a VPP-style dispatchable-capacity offer.
"""

import pytest

from fleet_aggregation import cell_dispatchable_capacity, fleet_dispatchable_offer


def _cells():
    return [
        {"cell_id": "c1", "nominal_kwh": 10.0, "soh_pct": 95.0, "soc_pct": 30.0, "sop_pct": 90.0},
        {"cell_id": "c2", "nominal_kwh": 10.0, "soh_pct": 74.0, "soc_pct": 50.0,
         "sop_pct": 55.0, "rul_cycles": 90, "rul_reliable": True},
        {"cell_id": "c3", "nominal_kwh": 10.0, "soh_pct": 98.0, "soc_pct": 97.0},
    ]


# ---------------------------------------------------------------------------
# cell_dispatchable_capacity
# ---------------------------------------------------------------------------

def test_cell_energy_is_soh_limited():
    cap = cell_dispatchable_capacity(nominal_kwh=10.0, soh_pct=80.0, soc_pct=30.0)
    assert cap["current_kwh"] == pytest.approx(8.0)
    # band [10, 95] -> 85pp of current capacity is dispatchable from 30% SOC.
    assert cap["energy_kwh"] == pytest.approx(8.0 * (95 - 30) / 100.0, abs=1e-6)


def test_cell_power_scaled_by_sop():
    full = cell_dispatchable_capacity(10.0, 100.0, 30.0, c_rate=0.5, sop_pct=100.0)
    limited = cell_dispatchable_capacity(10.0, 100.0, 30.0, c_rate=0.5, sop_pct=50.0)
    assert full["power_kw"] == pytest.approx(5.0)
    assert limited["power_kw"] == pytest.approx(2.5)


def test_cell_caution_narrows_band():
    healthy = cell_dispatchable_capacity(10.0, 95.0, 30.0)
    caution = cell_dispatchable_capacity(10.0, 95.0, 30.0, rul_cycles=90, rul_reliable=True)
    assert caution["band"]["caution"] is True
    assert caution["band"]["max_soc_pct"] < healthy["band"]["max_soc_pct"]
    assert caution["band"]["min_soc_pct"] > healthy["band"]["min_soc_pct"]


def test_cell_excluded_when_soc_at_band_top():
    cap = cell_dispatchable_capacity(10.0, 95.0, 97.0)
    assert cap["excluded"] is True
    assert cap["energy_kwh"] == 0.0
    assert cap["exclude_reason"]


def test_cell_invalid_inputs_raise():
    with pytest.raises(ValueError):
        cell_dispatchable_capacity(0.0, 90.0, 30.0)
    with pytest.raises(ValueError):
        cell_dispatchable_capacity(10.0, 0.0, 30.0)


# ---------------------------------------------------------------------------
# fleet_dispatchable_offer
# ---------------------------------------------------------------------------

def test_offer_aggregates_and_excludes():
    offer = fleet_dispatchable_offer(_cells())
    assert offer["offer"]["service"] == "dispatchable_capacity"
    assert offer["offer"]["window_hours"] == 2
    assert offer["offer"]["energy_kwh"] > 0
    assert offer["offer"]["power_kw"] > 0
    assert [e["cell_id"] for e in offer["excluded_cells"]] == ["c3"]  # SOC at band top
    assert offer["fleet_context"]["n_cells"] == 3
    assert offer["fleet_context"]["n_included"] == 2
    assert offer["fleet_context"]["n_excluded"] == 1
    assert offer["fleet_context"]["n_health_caution"] == 1  # c2
    assert offer["caveats"]


def test_offer_sum_equals_components():
    offer = fleet_dispatchable_offer(_cells())
    energy_sum = sum(c["energy_kwh"] for c in offer["cells"])
    power_sum = sum(c["power_kw"] for c in offer["cells"])
    assert offer["offer"]["energy_kwh"] == pytest.approx(energy_sum, abs=0.01)
    assert offer["offer"]["power_kw"] == pytest.approx(power_sum, abs=0.01)


def test_offer_sorted_by_energy_desc():
    offer = fleet_dispatchable_offer(_cells())
    energies = [c["energy_kwh"] for c in offer["cells"]]
    assert energies == sorted(energies, reverse=True)


def test_offer_empty_cells_raises():
    with pytest.raises(ValueError):
        fleet_dispatchable_offer([])
