"""Unit tests for src/passport.py — build_passport()."""

import pandas as pd
from passport import build_passport


def _sample_df():
    return pd.DataFrame({
        "soh_pct": [100, 99, 98, 97],
        "cycle_number": [1, 2, 3, 4],
        "resistance_ohm": [0.05, 0.051, 0.052, 0.053],
        "fade_rate_30cy": [0.001] * 4,
        "capacity_ah": [2.0, 1.98, 1.96, 1.94],
        "rul_pred": [None, None, None, 600.0],
    })


def test_build_passport_field_counts_are_consistent():
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    p = build_passport("B0005", _sample_df(), bundle, rul_reliable=True)

    all_fields = p["identity"] + p["soh"] + p["lifecycle"] + p["carbon"]
    summ = p["summary"]
    assert summ["n_total"] == len(all_fields)
    assert summ["n_available"] == sum(1 for f in all_fields if f["state"] == "available")
    assert summ["n_estimated"] == sum(1 for f in all_fields if f["state"] == "estimated")
    assert summ["n_unavailable"] == sum(1 for f in all_fields if f["state"] == "unavailable")


def test_build_passport_nasa_vs_synthetic_chemistry_label():
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    nasa_p = build_passport("B0005", _sample_df(), bundle, rul_reliable=True)
    synth_p = build_passport("Cell1", _sample_df(), bundle, rul_reliable=True)

    nasa_chem = next(f["value"] for f in nasa_p["identity"] if f["label"] == "Chemistry type")
    synth_chem = next(f["value"] for f in synth_p["identity"] if f["label"] == "Chemistry type")
    assert "NASA" in nasa_chem or "LiCoO" in nasa_chem
    assert "Synthetic" in synth_chem


def test_build_passport_severson_cell_is_not_mislabeled_synthetic():
    """Regression guard: build_passport() used to have only an is_nasa/else
    branch, so every non-NASA source (Severson, Oxford, uploads) was
    unconditionally labeled "Synthetic Li-ion model" — including real
    measured LFP cells. Chemistry, capacity, and data source must now be
    resolved per-source via ChemistryProfile.for_cell(), not a boolean."""
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    p = build_passport("S-b1c2", _sample_df(), bundle, rul_reliable=True)

    chem = next(f["value"] for f in p["identity"] if f["label"] == "Chemistry type")
    capacity = next(f for f in p["identity"] if f["label"] == "Nominal capacity")
    data_src = next(f["value"] for f in p["identity"] if f["label"] == "Data source")

    assert "LFP" in chem
    assert "Synthetic" not in chem
    assert "Severson" in data_src
    assert "Synthetic" not in data_src
    assert "Oxford-style" not in capacity["note"]


def test_build_passport_unknown_chemistry_falls_back_to_measured_capacity():
    """A cell ID that matches no registered source (e.g. a bare user upload)
    must not silently borrow the synthetic nominal-capacity constant — it
    should report its own measured cycle-1 capacity instead, marked
    "estimated" rather than "available"."""
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    p = build_passport("MyUploadedCell42", _sample_df(), bundle, rul_reliable=True)

    chem_field = next(f for f in p["identity"] if f["label"] == "Chemistry type")
    capacity_field = next(f for f in p["identity"] if f["label"] == "Nominal capacity")
    assert chem_field["state"] == "unavailable"
    assert "2.000 Ah" in capacity_field["value"]  # this cell's own cycle-1 capacity_ah
    assert capacity_field["state"] == "estimated"


def test_build_passport_rul_withheld_when_not_reliable():
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    p = build_passport("TestCell", _sample_df(), bundle, rul_reliable=False)
    rul_field = next(f for f in p["soh"] if "Remaining Useful Life" in f["label"])
    assert "Not calibrated" in rul_field["value"]
    assert "600" not in rul_field["value"]
