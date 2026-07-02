"""
Chemistry profiles for registered cell chemistries (#18 Expandability).

Adding a new chemistry = add one subclass here, not editing any page file.
Import ChemistryProfile.for_cell(cell_id) anywhere in the app to get
chemistry-specific behaviour (CRM fields, dQ/dV applicability, labels).

CRM fields returned by get_crm_fields() must be compatible with
_passport_field_row() in main.py:
    {"label": str, "value": str, "state": "available"|"estimated"|"unavailable", "note": str}
"""
from __future__ import annotations

_NASA_CELL_IDS = frozenset({"B0005", "B0006", "B0007", "B0018"})

_CRM_DEFAULTS = {
    "lico2_co_pct": 14.0,
    "lico2_ni_pct":  0.0,
    "lico2_li_pct":  7.0,
    "lfp_li_pct":    4.4,
}


class ChemistryProfile:
    """Base class for a registered cell chemistry.

    Subclass for each chemistry; override get_crm_fields() and
    get_health_sections(). All other attributes are class-level.
    """

    display_name:     str  = "Unknown"
    short_name:       str  = "?"
    dqdv_applicable:  bool = False
    provenance:       str  = "synthetic"
    dataset_citation: str  = ""

    @classmethod
    def for_cell(cls, cell_id: str) -> "ChemistryProfile":
        """Factory: return the right profile for a given cell ID."""
        if cell_id.startswith("S-"):
            return LFPSeversonProfile()
        if cell_id in _NASA_CELL_IDS:
            return LiCoO2NASAProfile()
        if cell_id.startswith(("Cell", "OxBat")):
            return LiCoO2SyntheticProfile()
        return UserDefinedProfile()

    def get_crm_fields(self, settings: dict | None = None) -> list[dict]:
        """Return CRM field dicts compatible with _passport_field_row()."""
        return []

    def get_health_sections(self) -> list[str]:
        """Names of analysis sections active on the Health page."""
        return ["capacity_fade", "resistance", "coulombic_efficiency",
                "rate_capability", "resistance_proxy"]


class LFPSeversonProfile(ChemistryProfile):
    display_name    = "LFP (Severson 2019)"
    short_name      = "LFP"
    dqdv_applicable = False
    provenance      = "measured"
    dataset_citation = "Severson et al., Nature Energy 2019"

    def get_crm_fields(self, settings: dict | None = None) -> list[dict]:
        s  = settings or {}
        li = s.get("crm_lfp_li_pct", _CRM_DEFAULTS["lfp_li_pct"])
        return [
            {"label": "Cobalt (Co) content",
             "value": "~0 wt% — cobalt-free chemistry", "state": "available",
             "note": "LFP (LiFePO4) uses iron-phosphate cathode; no cobalt required. "
                     "EU Art. 13 recycled-Co threshold does not apply."},
            {"label": "Nickel (Ni) content",
             "value": "~0 wt% — nickel-free chemistry", "state": "available",
             "note": "LFP does not use nickel. Lower critical-material dependency vs NMC/NCA."},
            {"label": "Lithium (Li) content",
             "value": f"~{li:.1f} wt% (est., configurable in Settings -> CRM)",
             "state": "estimated",
             "note": "Lithium in LiFePO4 cathode + graphite anode. "
                     "Recycled Li target: 4% by 2027, 10% by 2031 (EU Annex X)."},
            {"label": "Recycled Co content", "value": "N/A -- cobalt-free", "state": "unavailable",
             "note": "EU 2027 recycled-Co threshold does not apply to LFP."},
            {"label": "Recycled Ni content", "value": "N/A -- nickel-free", "state": "unavailable",
             "note": "EU 2027 recycled-Ni threshold does not apply to LFP."},
            {"label": "Article 52 due diligence", "value": "Required for Li supply chain",
             "state": "unavailable",
             "note": "Third-party audit of Li supply chain required for EU market access."},
        ]

    def get_health_sections(self) -> list[str]:
        return ["capacity_fade", "resistance", "coulombic_efficiency",
                "rate_capability", "resistance_proxy"]


class LiCoO2NASAProfile(ChemistryProfile):
    display_name    = "LiCoO2 (NASA PCoE)"
    short_name      = "LiCoO2"
    dqdv_applicable = True
    provenance      = "measured"
    dataset_citation = "NASA PCoE Battery Aging Dataset, Saha & Goebel 2007"

    def get_crm_fields(self, settings: dict | None = None) -> list[dict]:
        s      = settings or {}
        co     = s.get("crm_nca_co_pct",           _CRM_DEFAULTS["lico2_co_pct"])
        ni     = s.get("crm_nca_ni_pct",           _CRM_DEFAULTS["lico2_ni_pct"])
        li     = s.get("crm_nca_li_pct",           _CRM_DEFAULTS["lico2_li_pct"])
        rec_co = s.get("crm_nca_recycled_co_pct")
        rec_ni = s.get("crm_nca_recycled_ni_pct")
        return [
            {"label": "Cobalt (Co) content",
             "value": f"~{co:.1f} wt% (LiCoO2 cathode, est.)", "state": "estimated",
             "note": "Configurable in Settings -> CRM. "
                     "EU Art. 13 supply-chain due diligence required from 2026."},
            {"label": "Nickel (Ni) content",
             "value": f"~{ni:.1f} wt% (pure LiCoO2 baseline)", "state": "estimated",
             "note": "NMC variants show 15-33 wt%. Configurable in Settings -> CRM."},
            {"label": "Lithium (Li) content",
             "value": f"~{li:.1f} wt% (cathode + anode, est.)", "state": "estimated",
             "note": "Configurable in Settings -> CRM. "
                     "Recycled Li target: 4% by 2027, 10% by 2031 (EU Annex X)."},
            {"label": "Recycled Co content",
             "value": f"{rec_co:.1f}% recycled" if rec_co is not None
                      else "Not specified -- enter in Settings -> CRM",
             "state": "estimated" if rec_co is not None else "unavailable",
             "note": "EU 2030 target: 12% recycled Co (Annex X). Enter in Settings -> CRM."},
            {"label": "Recycled Ni content",
             "value": f"{rec_ni:.1f}% recycled" if rec_ni is not None
                      else "Not specified -- enter in Settings -> CRM",
             "state": "estimated" if rec_ni is not None else "unavailable",
             "note": "EU 2030 target: 4% recycled Ni (Annex X). Enter in Settings -> CRM."},
            {"label": "Article 52 due diligence", "value": "Not assessed", "state": "unavailable",
             "note": "Third-party audit of Co/Ni/Li supply chain required for EU market access."},
        ]

    def get_health_sections(self) -> list[str]:
        return ["capacity_fade", "resistance", "coulombic_efficiency",
                "rate_capability", "resistance_proxy", "dqdv"]


class LiCoO2SyntheticProfile(ChemistryProfile):
    display_name    = "LiCoO2 (synthetic)"
    short_name      = "LiCoO2"
    dqdv_applicable = True
    provenance      = "synthetic"
    dataset_citation = "Physics-informed synthetic model"

    def get_crm_fields(self, settings: dict | None = None) -> list[dict]:
        s  = settings or {}
        co = s.get("crm_synth_co_pct", _CRM_DEFAULTS["lico2_co_pct"])
        ni = s.get("crm_synth_ni_pct", _CRM_DEFAULTS["lico2_ni_pct"])
        li = s.get("crm_synth_li_pct", _CRM_DEFAULTS["lico2_li_pct"])
        return [
            {"label": "Cobalt (Co) content",
             "value": f"~{co:.1f} wt% (LiCoO2, est.)", "state": "estimated",
             "note": "Synthetic cell -- no manufacturing record. Configurable in Settings -> CRM."},
            {"label": "Nickel (Ni) content",
             "value": f"~{ni:.1f} wt% (pure LiCoO2 baseline)", "state": "estimated",
             "note": "Configurable in Settings -> CRM."},
            {"label": "Lithium (Li) content",
             "value": f"~{li:.1f} wt% (cathode + anode, est.)", "state": "estimated",
             "note": "Configurable in Settings -> CRM."},
            {"label": "Recycled Co content", "value": "Not applicable -- synthetic cell",
             "state": "unavailable", "note": "No manufacturing record for synthetic cells."},
            {"label": "Recycled Ni content", "value": "Not applicable -- synthetic cell",
             "state": "unavailable", "note": "No manufacturing record for synthetic cells."},
            {"label": "Article 52 due diligence", "value": "Not assessed", "state": "unavailable",
             "note": "Synthetic cells have no real supply chain."},
        ]

    def get_health_sections(self) -> list[str]:
        return ["capacity_fade", "resistance", "coulombic_efficiency",
                "rate_capability", "resistance_proxy", "dqdv"]


class UserDefinedProfile(ChemistryProfile):
    display_name    = "User-defined"
    short_name      = "Custom"
    dqdv_applicable = True
    provenance      = "measured"
    dataset_citation = "User upload"

    def get_crm_fields(self, settings: dict | None = None) -> list[dict]:
        s      = settings or {}
        co     = s.get("crm_user_co_pct")
        ni     = s.get("crm_user_ni_pct")
        li     = s.get("crm_user_li_pct")
        rec_co = s.get("crm_user_recycled_co_pct")
        rec_ni = s.get("crm_user_recycled_ni_pct")

        def _v(val: float | None, name: str) -> str:
            return f"~{val:.1f} wt%" if val is not None else f"Not specified -- enter {name} in Settings -> CRM"

        def _s(val: float | None) -> str:
            return "estimated" if val is not None else "unavailable"

        return [
            {"label": "Cobalt (Co) content",   "value": _v(co, "Co wt%"),   "state": _s(co),
             "note": "Configure in Settings -> CRM Configuration."},
            {"label": "Nickel (Ni) content",   "value": _v(ni, "Ni wt%"),   "state": _s(ni),
             "note": "Configure in Settings -> CRM Configuration."},
            {"label": "Lithium (Li) content",  "value": _v(li, "Li wt%"),   "state": _s(li),
             "note": "Configure in Settings -> CRM Configuration."},
            {"label": "Recycled Co content",
             "value": f"{rec_co:.1f}% recycled" if rec_co is not None else "Not specified",
             "state": "estimated" if rec_co is not None else "unavailable",
             "note": "Enter supply-chain audit data in Settings -> CRM Configuration."},
            {"label": "Recycled Ni content",
             "value": f"{rec_ni:.1f}% recycled" if rec_ni is not None else "Not specified",
             "state": "estimated" if rec_ni is not None else "unavailable",
             "note": "Enter supply-chain audit data in Settings -> CRM Configuration."},
            {"label": "Article 52 due diligence", "value": "Not assessed", "state": "unavailable",
             "note": "Third-party audit of Co/Ni/Li supply chain required for EU market access."},
        ]
