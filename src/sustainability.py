"""
Phase 7 — Sustainability module constants.

Material content and recovery figures for LiCoO₂, LFP, and NCA 18650 cells,
and EU Battery Regulation (EU) 2023/1542 recycled-content targets. All figures
carry explicit provenance labels — no aggregated index, no invented
circularity scores.

Synthetic cells model electrochemical behavior only — material content figures
apply to the equivalent real LiCoO₂ 18650 chemistry, not the simulation.
"""

# ---------------------------------------------------------------------------
# Critical material content per 18650 cell (~2 Ah reference), by chemistry
# ---------------------------------------------------------------------------
# LiCoO2 figures are directly measured teardown data (Harper et al. 2019).
# LFP and NCA figures are derived, not measured: real stoichiometry for each
# chemistry's cathode formula (LiFePO4; LiNi0.8Co0.15Al0.05O2 — the standard
# formula for both the Severson A123 APR18650M1A cell and the Oxford
# NCR18650BD-family cell) applied to that cell's real manufacturer-datasheet
# mass, using the ~23.5% cathode-active-material mass fraction implied by
# Harper et al.'s own LiCoO2 figures (6.5 g Co / 60.2% Co-in-LiCoO2 = 10.8 g
# CAM; 10.8 g / 46 g cell = 23.5%). This is a transparent derivation, not a
# teardown measurement — see each entry's "source" for the exact chain, and
# its "label" ("Estimated — stoichiometric" vs "Cited estimate") for how much
# to trust it. It excludes electrolyte lithium (unlike the LiCoO2 Li figure,
# which Harper et al. report as cathode + electrolyte combined), so LFP/NCA
# lithium figures are a lower bound, not a full-cell total.
#
# All chemistries scale proportionally by Ah capacity via
# material_content_for_cell() below.

_GRAPHITE = {
    "name":     "Graphite (C)",
    "formula":  "anode active material",
    "g_per_2ah": 7.0,           # mid-point of 6–8 g range
    "g_range":  "6–8 g",
    "recovery_pct": 40,
    "recovery_note": "not commercially prioritised; lower recovery than metals",
    "label":    "Illustrative — not sourced",
    "source":   "No per-cell figure for 18650 specifically; estimate from anode mass fraction. "
                "Not chemistry-specific — the same graphite anode family is used across "
                "LiCoO₂/LFP/NCA/NMC 18650 cells, so this figure is shared across chemistries.",
    "eu_critical": False,
}

CRITICAL_MATERIALS_LICOO2 = [
    {
        "name":     "Cobalt (Co)",
        "formula":  "LiCoO₂ cathode",
        "g_per_2ah": 6.5,           # mid-point of 5–8 g range
        "g_range":  "5–8 g",
        "recovery_pct": 95,
        "recovery_note": "hydrometallurgical process",
        "label":    "Cited estimate",
        "source":   "Harper et al. (2019) Nature Reviews Materials; Sommerville et al. (2020)",
        "eu_critical": True,
    },
    {
        "name":     "Lithium (Li)",
        "formula":  "cathode + electrolyte (LiPF₆)",
        "g_per_2ah": 1.8,           # mid-point of 1.5–2 g range (cathode + electrolyte)
        "g_range":  "1.5–2 g",
        "recovery_pct": 80,
        "recovery_note": "hydromet; lower than Co due to electrolyte loss",
        "label":    "Cited estimate",
        "source":   "Harper et al. (2019) Nature Reviews Materials",
        "eu_critical": True,
    },
    _GRAPHITE,
    {
        "name":     "Nickel (Ni)",
        "formula":  "trace only in LiCoO₂",
        "g_per_2ah": 0.1,
        "g_range":  "< 0.5 g",
        "recovery_pct": None,
        "recovery_note": "negligible in LiCoO₂; significant in NMC/NCA chemistries",
        "label":    "Illustrative — not sourced",
        "source":   "LiCoO₂ is a pure cobalt oxide cathode — nickel is not a primary material",
        "eu_critical": True,
    },
]

CRITICAL_MATERIALS_LFP = [
    {
        "name":     "Cobalt (Co)",
        "formula":  "not present — LiFePO₄ (olivine) cathode",
        "g_per_2ah": 0.0,
        "g_range":  "0 g",
        "recovery_pct": None,
        "recovery_note": "not present in this chemistry",
        "label":    "Verified",
        "source":   "LiFePO₄'s olivine cathode structure contains no cobalt — a well-established "
                    "chemistry fact, not an estimate.",
        "eu_critical": True,
    },
    {
        "name":     "Lithium (Li)",
        "formula":  "LiFePO₄ cathode (excludes electrolyte)",
        "g_per_2ah": 0.73,
        "g_range":  "~0.6–0.8 g (derived)",
        "recovery_pct": None,
        "recovery_note": "recovery economics for cobalt-free cathodes are less mature — no Co/Ni "
                          "credit to offset hydrometallurgical processing cost",
        "label":    "Estimated — stoichiometric",
        "source":   "LiFePO₄ stoichiometry (Li = 4.4% of cathode mass by molar mass) applied to the "
                    "A123 APR18650M1A datasheet cell mass (39 g, 1.1 Ah — the cell used in the "
                    "Severson et al. 2019 dataset) using the ~23.5% cathode-mass-fraction derived "
                    "above from Harper et al. (2019). Cathode-only — excludes electrolyte lithium, "
                    "unlike the LiCoO₂ figure, which includes it.",
        "eu_critical": True,
    },
    _GRAPHITE,
    {
        "name":     "Nickel (Ni)",
        "formula":  "not present — LiFePO₄ (olivine) cathode",
        "g_per_2ah": 0.0,
        "g_range":  "0 g",
        "recovery_pct": None,
        "recovery_note": "not present in this chemistry",
        "label":    "Verified",
        "source":   "LiFePO₄'s olivine cathode structure contains no nickel — a well-established "
                    "chemistry fact, not an estimate.",
        "eu_critical": True,
    },
]

CRITICAL_MATERIALS_NCA = [
    {
        "name":     "Cobalt (Co)",
        "formula":  "LiNi₀.₈Co₀.₁₅Al₀.₀₅O₂ cathode",
        "g_per_2ah": 0.62,
        "g_range":  "~0.5–0.7 g (derived)",
        "recovery_pct": 95,
        "recovery_note": "hydrometallurgical process",
        "label":    "Estimated — stoichiometric",
        "source":   "LiNi0.8Co0.15Al0.05O2 stoichiometry (Co = 9.2% of cathode mass by molar mass) "
                    "applied to the Panasonic NCR18650B datasheet cell mass (48.5 g, 3.4 Ah — same "
                    "NCA family as the NCR18650BD cell used in the Oxford Path-Dependent dataset) "
                    "using the ~23.5% cathode-mass-fraction derived above from Harper et al. (2019). "
                    "Lower Co share than LiCoO2 by design.",
        "eu_critical": True,
    },
    {
        "name":     "Lithium (Li)",
        "formula":  "LiNi₀.₈Co₀.₁₅Al₀.₀₅O₂ cathode (excludes electrolyte)",
        "g_per_2ah": 0.48,
        "g_range":  "~0.4–0.6 g (derived)",
        "recovery_pct": 80,
        "recovery_note": "hydromet; lower than Co/Ni due to electrolyte loss",
        "label":    "Estimated — stoichiometric",
        "source":   "LiNi0.8Co0.15Al0.05O2 stoichiometry (Li = 7.2% of cathode mass by molar mass) "
                    "applied to the Panasonic NCR18650B datasheet cell mass (48.5 g, 3.4 Ah) using "
                    "the ~23.5% cathode-mass-fraction derived above from Harper et al. (2019). "
                    "Cathode-only — excludes electrolyte lithium, unlike the LiCoO₂ figure, which "
                    "includes it.",
        "eu_critical": True,
    },
    _GRAPHITE,
    {
        "name":     "Nickel (Ni)",
        "formula":  "LiNi₀.₈Co₀.₁₅Al₀.₀₅O₂ cathode — primary material",
        "g_per_2ah": 3.28,
        "g_range":  "~2.8–3.8 g (derived)",
        "recovery_pct": 90,
        "recovery_note": "hydrometallurgical process",
        "label":    "Estimated — stoichiometric",
        "source":   "LiNi0.8Co0.15Al0.05O2 stoichiometry (Ni = 48.9% of cathode mass by molar mass) "
                    "applied to the Panasonic NCR18650B datasheet cell mass (48.5 g, 3.4 Ah) using "
                    "the ~23.5% cathode-mass-fraction derived above from Harper et al. (2019). "
                    "NCA's primary cathode material by mass, unlike LiCoO2 where nickel is "
                    "trace-only.",
        "eu_critical": True,
    },
]


def critical_materials_for_chemistry(short_name: str) -> list:
    """Return the 4 tracked critical-material entries for a chemistry.

    LiCoO2, LFP, and NCA have chemistry-specific figures (see each entry's
    "source" for provenance — LiCoO2 is measured teardown data, LFP/NCA are
    stoichiometric derivations). Any other chemistry (e.g. an unspecified
    user upload) falls back to the LiCoO2 reference figures, since no
    chemistry-specific data exists for it.
    """
    return {
        "LiCoO2": CRITICAL_MATERIALS_LICOO2,
        "LFP":    CRITICAL_MATERIALS_LFP,
        "NCA":    CRITICAL_MATERIALS_NCA,
    }.get(short_name, CRITICAL_MATERIALS_LICOO2)


# Backward-compatible alias — the original LiCoO2-only list.
CRITICAL_MATERIALS = CRITICAL_MATERIALS_LICOO2


# ---------------------------------------------------------------------------
# EU Battery Regulation (EU) 2023/1542 recycled content targets
# ---------------------------------------------------------------------------
# Source: EU 2023/1542, Annex XII — minimum recycled content requirements
# for industrial batteries and EV batteries by mass of active materials.
# Targets apply from the specified year.

EU_RECYCLED_TARGETS = [
    {
        "material":  "Cobalt",
        "target_2031_pct": 16,
        "target_2036_pct": 26,
        "source":    "EU 2023/1542 Annex XII",
        "current_industry_range": "~5–10%",
        "current_note": "Industry-wide estimate; Cobalt recycling is most mature (hydromet), but certified recycled-content batteries remain rare. Figures from Sommerville et al. (2020) and BloombergNEF 2023.",
    },
    {
        "material":  "Lithium",
        "target_2031_pct": 6,
        "target_2036_pct": 12,
        "source":    "EU 2023/1542 Annex XII",
        "note":      "Applying from 2031; 2036 target reflects increased recycling infrastructure",
        "current_industry_range": "~1–3%",
        "current_note": "Lithium recycling infrastructure is nascent; hydromet recovery rates are improving but certified content is very low. Estimate from IEA Critical Minerals Report 2023.",
    },
    {
        "material":  "Nickel",
        "target_2031_pct": 6,
        "target_2036_pct": 15,
        "source":    "EU 2023/1542 Annex XII",
        "current_industry_range": "N/A for LiCoO₂",
        "current_note": "Nickel is not a primary material in LiCoO₂ — the EU recycled-content target applies to NMC/NCA chemistries where nickel is a major cathode component.",
    },
]


# ---------------------------------------------------------------------------
# EU Green Deal alignment fields (three-state: available / estimated / unavailable)
# ---------------------------------------------------------------------------

EU_GREEN_DEAL_FIELDS = [
    {
        "label":  "Carbon footprint estimate (manufacturing phase)",
        "state":  "estimated",
        "note":   "Phase 4 Consequences module — IVL 2019, cited estimate",
    },
    {
        "label":  "Second-life application scoring",
        "state":  "available",
        "note":   "Phase 4 application_fit() — validated against NREL / IEEE 1881-2019 / IRENA thresholds",
    },
    {
        "label":  "Material recovery value",
        "state":  "estimated",
        "note":   "Sommerville et al. (2020) cobalt/lithium spot price estimate",
    },
    {
        "label":  "Recycled content certification",
        "state":  "unavailable",
        "note":   "Requires manufacturer supply chain records — not available in this demonstration",
    },
    {
        "label":  "Full lifecycle carbon audit (Art. 7 scope)",
        "state":  "unavailable",
        "note":   "Requires third-party accredited audit — a computed (not certified) cradle-to-grave "
                  "estimate is shown in the Cradle-to-Grave Footprint section below, using this cell's "
                  "own real cumulative energy throughput for the use phase",
    },
    {
        "label":  "Critical material sourcing declaration",
        "state":  "unavailable",
        "note":   "No supply chain provenance data in this demonstration",
    },
]


def material_content_for_cell(g_per_2ah: float, cell_kwh: float) -> float:
    """
    Scale a material's content from the 2 Ah reference cell to this cell's
    capacity. Linear scaling by Ah (cell_kwh / 3.6V / 1000 → Ah).
    """
    cell_ah = cell_kwh / (3.6 / 1000)   # kWh → Ah at 3.6V nominal
    return g_per_2ah * (cell_ah / 2.0)


# ---------------------------------------------------------------------------
# Cradle-to-grave carbon footprint
# ---------------------------------------------------------------------------
# Closes the gap EU_GREEN_DEAL_FIELDS above has always disclosed honestly
# ("Full lifecycle carbon audit (Art. 7 scope)" -- unavailable, "use-phase
# CO2 shown here is illustrative only"): a genuine per-cell cradle-to-grave
# NUMBER, not just three separate scenario bars on the Sustainability page's
# existing Lifecycle Carbon Chart. This is still NOT a certified Art. 7
# audit -- that requires third-party accreditation no software can provide
# -- so the EU_GREEN_DEAL_FIELDS entry above stays "unavailable" even after
# this exists; what changes is that the "illustrative only" caveat now
# points at a real computed estimate instead of nothing.
#
# Distinct from consequences.ASSUMPTIONS['co2_manufacture'] (0.55 kg/cell,
# IVL 2019, chemistry-agnostic, used by the existing Lifecycle Carbon Chart
# and left untouched for backward compatibility): these are chemistry-
# SPECIFIC, per-kWh manufacturing figures from more recent, chemistry-
# resolved literature, so LFP and NCA cells get their own real numbers
# instead of sharing one LiCoO2-derived estimate.
MANUFACTURING_CO2_PER_KWH = {
    "LiCoO2": {
        "value": 75.0,
        "label": "Cited estimate",
        "source": "IVL (2019) range midpoint, 50-100 kg CO2e/kWh -- the same source "
                   "consequences.ASSUMPTIONS['co2_manufacture'] uses, expressed per-kWh "
                   "instead of per-cell so it's directly comparable across chemistries.",
    },
    "LFP": {
        "value": 62.0,
        "label": "Cited estimate",
        "source": "Nature Communications (2024), 'Carbon footprint distributions of "
                   "lithium-ion batteries and their materials' -- LFP cathode range "
                   "midpoint, 54-69 kg CO2e/kWh.",
    },
    "NCA": {
        "value": 74.0,
        "label": "Estimated — chemistry proxy",
        "source": "Same 2024 Nature Communications study's NMC811 range midpoint "
                   "(59-115, mid 74 kg CO2e/kWh) -- used as the nearest well-documented "
                   "high-nickel layered-oxide proxy, since dedicated NCA cradle-to-grave "
                   "manufacturing figures are less available in the literature than LFP/NMC.",
    },
}

# Avoided-emissions credit from hydrometallurgical recycling displacing
# virgin material production, kg CO2e per kWh of RECOVERED (not delivered)
# capacity. A literature-review range (25.5-30.9 kg CO2e/kWh) rather than a
# single-study figure -- midpoint used here, distinct from
# consequences.sustainability_snapshot()'s existing co2_recycling_credit
# (Dunn et al. 2015, 15% cathode-material recovery credit, per-cell) which
# the Sustainability page's Lifecycle Carbon Chart already uses; this is a
# second, independently-sourced EOL figure for the cradle-to-grave total
# specifically, not a replacement for that one.
RECYCLING_AVOIDED_CO2_PER_KWH = {
    "value": 28.0,
    "label": "Cited estimate",
    "source": "Literature review range 25.5-30.9 kg CO2e/kWh avoided via hydrometallurgical "
              "recycling vs. virgin material production; midpoint used here.",
}


def cradle_to_grave_footprint(
    chemistry_short_name: str,
    nominal_kwh: float,
    cumulative_kwh_delivered: float,
    grid_carbon_intensity_kg_per_kwh: float,
    end_of_life_pathway: str = "undetermined",
) -> dict:
    """
    A genuine per-cell cradle-to-grave CO2e total, three separately-cited
    stages so a reader can see exactly which numbers are solid and which
    are assumptions -- never presented as one opaque aggregate score (this
    platform's standing "no aggregated sustainability index" position).

    cumulative_kwh_delivered should be this cell's OWN measured/computed
    cumulative_kwh (batlab.features.engineering's real running total from
    actual per-cycle capacity) -- not a nominal-capacity x cycle-count
    approximation. This is the one stage genuinely specific to this cell's
    real cycling history rather than a chemistry-wide literature average.

    end_of_life_pathway: "recycle" applies the avoided-emissions credit;
    "landfill" or "undetermined" (default -- most cells in this platform
    are still in active service) apply no credit, since it hasn't happened
    yet and applying it early would overstate today's footprint as better
    than it currently is.

    Returns a dict with manufacturing_kg/use_phase_kg/end_of_life_kg/total_kg
    plus the manufacturing figure's own citation object (manufacturing_source)
    for direct display -- callers should show all three stages, not just
    total_kg, so nothing reads as more certain than it is.
    """
    mfg = MANUFACTURING_CO2_PER_KWH.get(chemistry_short_name, MANUFACTURING_CO2_PER_KWH["LiCoO2"])
    manufacturing_kg = mfg["value"] * nominal_kwh
    use_phase_kg = grid_carbon_intensity_kg_per_kwh * cumulative_kwh_delivered
    end_of_life_kg = (
        -RECYCLING_AVOIDED_CO2_PER_KWH["value"] * nominal_kwh
        if end_of_life_pathway == "recycle" else 0.0
    )
    return {
        "manufacturing_kg": manufacturing_kg,
        "manufacturing_source": mfg,
        "use_phase_kg": use_phase_kg,
        "end_of_life_kg": end_of_life_kg,
        "end_of_life_pathway": end_of_life_pathway,
        "total_kg": manufacturing_kg + use_phase_kg + end_of_life_kg,
    }
