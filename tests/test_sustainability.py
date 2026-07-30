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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from sustainability import critical_materials_for_chemistry, material_content_for_cell

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
