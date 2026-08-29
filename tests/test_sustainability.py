"""Unit tests for src/sustainability.py's critical_materials_for_chemistry().

Circular Economy Coverage finding: the Sustainability tab's Critical
Materials Tracker used to show LiCoO2-specific (Harper et al. 2019) gram
figures for every chemistry, with only a disclosure banner for LFP/NCA
cells. These tests confirm each chemistry now gets its own real
(LiCoO2) or derived (LFP/NCA) figures instead of silently reusing
LiCoO2's numbers.
"""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from sustainability import (
    critical_materials_for_chemistry, material_content_for_cell,
    cradle_to_grave_footprint, MANUFACTURING_CO2_PER_KWH, RECYCLING_AVOIDED_CO2_PER_KWH,
)

_NAMES = ("Cobalt (Co)", "Lithium (Li)", "Graphite (C)", "Nickel (Ni)")


def _by_name(materials):
    return {m["name"]: m for m in materials}


def test_licoo2_matches_harper_reference_figures():
    mats = _by_name(critical_materials_for_chemistry("LiCoO2"))
    assert mats["Cobalt (Co)"]["g_per_2ah"] == 6.5
    assert mats["Lithium (Li)"]["g_per_2ah"] == 1.8
    assert mats["Nickel (Ni)"]["g_per_2ah"] == 0.1
    assert "Harper" in mats["Cobalt (Co)"]["source"]


def test_lfp_has_zero_cobalt_and_nickel():
    """LiFePO4 is cobalt-free and nickel-free -- a chemistry fact, not an
    estimate, so these must be exactly 0, not a small "trace" figure."""
    mats = _by_name(critical_materials_for_chemistry("LFP"))
    assert mats["Cobalt (Co)"]["g_per_2ah"] == 0.0
    assert mats["Nickel (Ni)"]["g_per_2ah"] == 0.0
    assert mats["Cobalt (Co)"]["label"] == "Verified"
    assert mats["Nickel (Ni)"]["label"] == "Verified"


def test_lfp_lithium_is_real_figure_not_licoo2s():
    mats = _by_name(critical_materials_for_chemistry("LFP"))
    licoo2_mats = _by_name(critical_materials_for_chemistry("LiCoO2"))
    assert mats["Lithium (Li)"]["g_per_2ah"] != licoo2_mats["Lithium (Li)"]["g_per_2ah"]
    assert mats["Lithium (Li)"]["g_per_2ah"] > 0
    assert "stoichiometric" in mats["Lithium (Li)"]["label"].lower()


def test_nca_nickel_is_primary_not_trace():
    """NCA is nickel-dominant -- its Nickel figure must be far larger than
    LiCoO2's trace (0.1 g) figure, unlike the old shared-table behaviour
    that showed every chemistry the same "trace only in LiCoO2" number."""
    mats = _by_name(critical_materials_for_chemistry("NCA"))
    licoo2_mats = _by_name(critical_materials_for_chemistry("LiCoO2"))
    assert mats["Nickel (Ni)"]["g_per_2ah"] > licoo2_mats["Nickel (Ni)"]["g_per_2ah"] * 10
    assert mats["Nickel (Ni)"]["g_per_2ah"] > mats["Cobalt (Co)"]["g_per_2ah"]


def test_nca_cobalt_lower_than_licoo2():
    mats = _by_name(critical_materials_for_chemistry("NCA"))
    licoo2_mats = _by_name(critical_materials_for_chemistry("LiCoO2"))
    assert 0 < mats["Cobalt (Co)"]["g_per_2ah"] < licoo2_mats["Cobalt (Co)"]["g_per_2ah"]


def test_graphite_shared_across_chemistries():
    """Anode graphite content isn't cathode-chemistry-specific -- same
    figure should appear for all three tracked chemistries."""
    licoo2_g = _by_name(critical_materials_for_chemistry("LiCoO2"))["Graphite (C)"]["g_per_2ah"]
    lfp_g    = _by_name(critical_materials_for_chemistry("LFP"))["Graphite (C)"]["g_per_2ah"]
    nca_g    = _by_name(critical_materials_for_chemistry("NCA"))["Graphite (C)"]["g_per_2ah"]
    assert licoo2_g == lfp_g == nca_g


def test_unknown_chemistry_falls_back_to_licoo2():
    unknown_mats = critical_materials_for_chemistry("Custom")
    licoo2_mats  = critical_materials_for_chemistry("LiCoO2")
    assert unknown_mats == licoo2_mats


def test_every_chemistry_returns_all_four_materials_in_order():
    for chem in ("LiCoO2", "LFP", "NCA"):
        names = tuple(m["name"] for m in critical_materials_for_chemistry(chem))
        assert names == _NAMES


def test_material_content_for_cell_scales_by_capacity():
    """A 4 Ah cell should show double the material content of an
    equivalent 2 Ah reference figure."""
    cell_kwh_2ah = 2.0 * 3.6 / 1000
    cell_kwh_4ah = 4.0 * 3.6 / 1000
    assert material_content_for_cell(6.5, cell_kwh_4ah) == \
        material_content_for_cell(6.5, cell_kwh_2ah) * 2


# ---------------------------------------------------------------------------
# cradle_to_grave_footprint() — chemistry-specific manufacturing + real
# per-cell use-phase (cumulative_kwh_delivered) + optional EOL credit.
# ---------------------------------------------------------------------------

def test_manufacturing_kg_uses_chemistry_specific_figure():
    lfp = cradle_to_grave_footprint("LFP", nominal_kwh=1.0, cumulative_kwh_delivered=0.0,
                                     grid_carbon_intensity_kg_per_kwh=0.25)
    nca = cradle_to_grave_footprint("NCA", nominal_kwh=1.0, cumulative_kwh_delivered=0.0,
                                     grid_carbon_intensity_kg_per_kwh=0.25)
    assert lfp["manufacturing_kg"] == MANUFACTURING_CO2_PER_KWH["LFP"]["value"]
    assert nca["manufacturing_kg"] == MANUFACTURING_CO2_PER_KWH["NCA"]["value"]
    assert lfp["manufacturing_kg"] != nca["manufacturing_kg"]


def test_unknown_chemistry_falls_back_to_licoo2_manufacturing():
    result = cradle_to_grave_footprint("Custom", nominal_kwh=1.0, cumulative_kwh_delivered=0.0,
                                        grid_carbon_intensity_kg_per_kwh=0.25)
    assert result["manufacturing_kg"] == MANUFACTURING_CO2_PER_KWH["LiCoO2"]["value"]


def test_use_phase_scales_with_real_cumulative_kwh_not_nominal():
    """Two cells with the same nominal capacity but different real
    cumulative throughput (one heavily cycled, one lightly) must get
    different use-phase figures -- this is the whole point of using
    cumulative_kwh_delivered instead of a nominal-capacity approximation."""
    light = cradle_to_grave_footprint("LiCoO2", nominal_kwh=0.007, cumulative_kwh_delivered=1.0,
                                       grid_carbon_intensity_kg_per_kwh=0.25)
    heavy = cradle_to_grave_footprint("LiCoO2", nominal_kwh=0.007, cumulative_kwh_delivered=10.0,
                                       grid_carbon_intensity_kg_per_kwh=0.25)
    assert heavy["use_phase_kg"] == light["use_phase_kg"] * 10
    assert light["manufacturing_kg"] == heavy["manufacturing_kg"]  # unaffected by use-phase


def test_recycle_pathway_applies_negative_credit():
    result = cradle_to_grave_footprint("LFP", nominal_kwh=1.0, cumulative_kwh_delivered=5.0,
                                        grid_carbon_intensity_kg_per_kwh=0.25, end_of_life_pathway="recycle")
    assert result["end_of_life_kg"] == -RECYCLING_AVOIDED_CO2_PER_KWH["value"]
    assert result["end_of_life_kg"] < 0


def test_non_recycle_pathways_apply_no_credit():
    for pathway in ("undetermined", "landfill"):
        result = cradle_to_grave_footprint("LFP", nominal_kwh=1.0, cumulative_kwh_delivered=5.0,
                                            grid_carbon_intensity_kg_per_kwh=0.25, end_of_life_pathway=pathway)
        assert result["end_of_life_kg"] == 0.0


def test_total_kg_is_sum_of_three_stages():
    result = cradle_to_grave_footprint("LFP", nominal_kwh=1.0, cumulative_kwh_delivered=5.0,
                                        grid_carbon_intensity_kg_per_kwh=0.25, end_of_life_pathway="recycle")
    assert abs(result["total_kg"] - (result["manufacturing_kg"] + result["use_phase_kg"] + result["end_of_life_kg"])) < 1e-9


def test_manufacturing_source_citation_present_for_display():
    result = cradle_to_grave_footprint("NCA", nominal_kwh=1.0, cumulative_kwh_delivered=0.0,
                                        grid_carbon_intensity_kg_per_kwh=0.25)
    assert "source" in result["manufacturing_source"]
    assert "label" in result["manufacturing_source"]
