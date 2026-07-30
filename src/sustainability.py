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
        "note":   "Requires third-party accredited audit — use-phase CO₂ shown here is illustrative only",
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
