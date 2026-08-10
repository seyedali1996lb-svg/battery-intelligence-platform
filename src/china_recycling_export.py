"""
China NEV traction battery recycling — traceability-shaped export.

Same honest pattern as src/passport.py's EU Battery Passport and
src/us_ira_export.py: a field-structure demonstration, not a compliance
claim or a real registration with China's national platform.

Source (verified via live web research, 2026-08 — not recalled from
training data, since this regulation is new): six Chinese government
agencies jointly issued the "Interim Measures for the Management of
Recycling and Comprehensive Utilization of Waste New Energy Vehicle
Traction Batteries" on 2026-01-16, effective 2026-04-01, replacing prior
policy documents with a legally binding framework. Key provisions:
  - Extended Producer Responsibility (EPR): vehicle/battery manufacturers
    must build collection outlets, publish collection information, and
    may not refuse retired batteries.
  - A national NEV Traction Battery Traceability Information Platform
    plus a "Digital ID" system for full lifecycle battery traceability.
  - Battery producers code batteries per GB/T 34014-2017.
  - A "whole channel, whole chain, whole lifecycle" oversight approach.

This platform has no API access to China's national traceability
platform (it doesn't publish one for third-party research tools), and no
manufacturer-identity/EPR records — those fields are marked unavailable,
same three-state discipline as the EU Passport. What this platform CAN
genuinely offer: a recycling-channel recommendation using its own
eol_r_code_recommendation() and src/recycler_directory.py (which
includes an Asia/China-region recycler, GEM Co., with real chemistry-
specific process lines) — computed, not registered.
"""

import datetime

PREFIX = "cn_nev_battery_"


def build_china_recycling_entry(
    cell_id: str,
    chemistry: str,
    soh: float,
    r_code: str,
    recommended_recycler_name: "str | None",
) -> dict:
    """
    Build one field-structure-demonstration entry for China's 2026
    Interim Measures traceability concepts.

    r_code: this cell's eol_r_code_recommendation() result — reused
    directly, not re-derived, same "single source of truth" discipline
    consequences.py's own docstrings establish elsewhere in this codebase.
    recommended_recycler_name: the top China/Asia-region match from
    src/recycler_directory.py's recommend_recyclers(), or None if this
    cell's R-code doesn't call for recycling.
    """
    return {
        "type": "china_nev_battery_recycling_traceability",
        "id": cell_id,
        "attributes": {
            f"{PREFIX}chemistry": chemistry,
            f"{PREFIX}soh_pct": round(float(soh), 2),
            f"{PREFIX}regulation_reference": {
                "value": "Interim Measures for the Management of Recycling and Comprehensive "
                         "Utilization of Waste New Energy Vehicle Traction Batteries (effective 2026-04-01)",
                "state": "available",
                "note": "A regulatory citation, not a per-cell measurement.",
            },
            f"{PREFIX}gbt_34014_2017_coding_registered": {
                "value": None, "state": "unavailable",
                "note": "A real GB/T 34014-2017 code requires registration with the manufacturer's own coding system — this platform cannot mint or verify one.",
            },
            f"{PREFIX}digital_id_registered": {
                "value": None, "state": "unavailable",
                "note": "Requires API access to China's national NEV Traction Battery Traceability Information Platform, which this platform does not have.",
            },
            f"{PREFIX}epr_responsible_party": {
                "value": None, "state": "unavailable",
                "note": "Requires manufacturer identity/EPR registration records — the same gap the EU Passport already discloses for its own identity fields.",
            },
            f"{PREFIX}recommended_recycling_pathway": {
                "value": r_code, "state": "estimated",
                "note": "This platform's own eol_r_code_recommendation() output (IEC 62902 taxonomy), reused directly — not a China-specific determination, but the same underlying SOH/fade/second-life-fit analysis.",
            },
            f"{PREFIX}recommended_recycler": {
                "value": recommended_recycler_name, "state": "estimated" if recommended_recycler_name else "unavailable",
                "note": "Computed from this platform's own recycler directory (src/recycler_directory.py) — a point-in-time research snapshot, not a registered collection-outlet assignment under the EPR framework." if recommended_recycler_name
                        else "No recycling recommendation applies — this cell's R-code doesn't call for recycling.",
            },
        },
    }


def to_china_recycling_document(entry: dict, cell_id: str) -> dict:
    """Wrap build_china_recycling_entry()'s output with a meta/disclaimer
    block — same pattern as optimade_export.to_optimade_document() /
    us_ira_export.to_ira_30d_document()."""
    return {
        "meta": {
            "cell_id": cell_id,
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "format": "china-nev-battery-recycling-traceability-structure",
            "format_reference": "Interim Measures for the Management of Recycling and Comprehensive "
                                 "Utilization of Waste New Energy Vehicle Traction Batteries (effective 2026-04-01)",
            "disclaimer": (
                "Field-structure demonstration of China's 2026 Interim Measures "
                "traceability concepts, NOT a real registration with China's national "
                "NEV Traction Battery Traceability Information Platform or a "
                "compliance claim. This platform has no API access to that platform "
                "and no manufacturer EPR records — every field requiring them is "
                "marked unavailable."
            ),
        },
        "data": entry,
    }
