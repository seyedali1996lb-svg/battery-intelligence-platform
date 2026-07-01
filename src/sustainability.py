"""
Phase 7 — Sustainability module constants.

Material content and recovery figures for LiCoO₂ 18650 cells, and EU Battery
Regulation (EU) 2023/1542 recycled-content targets. All figures carry explicit
provenance labels — no aggregated index, no invented circularity scores.

Synthetic cells model electrochemical behavior only — material content figures
apply to the equivalent real LiCoO₂ 18650 chemistry, not the simulation.
"""

# ---------------------------------------------------------------------------
# Critical material content per 18650 LiCoO₂ cell (~2 Ah, ~46 g total)
# ---------------------------------------------------------------------------
# Source: Harper et al. (2019) Nature Reviews Materials — LiCoO₂ cell teardown
# and composition analysis. Figures are per-cell for a nominal 2 Ah 18650.
# Synthetic cells (0.74 Ah) scale proportionally by Ah capacity.

CRITICAL_MATERIALS = [
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
    {
        "name":     "Graphite (C)",
        "formula":  "anode active material",
        "g_per_2ah": 7.0,           # mid-point of 6–8 g range
        "g_range":  "6–8 g",
        "recovery_pct": 40,
        "recovery_note": "not commercially prioritised; lower recovery than metals",
        "label":    "Illustrative — not sourced",
        "source":   "No per-cell figure for 18650 specifically; estimate from anode mass fraction",
        "eu_critical": False,
    },
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
