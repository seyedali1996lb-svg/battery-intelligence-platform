"""
US Inflation Reduction Act (IRA) Section 30D critical-minerals/battery-
components traceability-shaped export.

Same honest pattern as src/passport.py's EU Battery Passport: a field-
structure demonstration, not a compliance claim. Section 30D's actual
requirement (confirmed via live research at the time this module was
written, 2026-08 — not recalled from training data, since these
percentages step up annually) is that a real manufacturer must document,
with auditable supply-chain-of-custody records, what fraction of a
battery's critical minerals were extracted/processed in the US or a US
free-trade-agreement country (or recycled in North America), and what
fraction of battery components were manufactured/assembled in North
America — plus confirm no Foreign Entity of Concern (FEOC) involvement.

This platform has NO supply-chain-of-custody data at all — it works from
public research datasets (NASA PCoE, Severson et al., Oxford) and
synthetic/uploaded cycling data, none of which records where a cell's
raw materials came from. Every field below that would require that data
is explicitly marked unavailable, exactly the same three-state
available/estimated/unavailable discipline the EU Passport already uses
— never a fabricated percentage.

Sources (verified via live web research, 2026-08):
  - 2026 critical minerals applicable percentage: 70%
  - 2026 battery components applicable percentage: 70%
  - FEOC restriction: critical minerals/components extracted, processed,
    or recycled by a Foreign Entity of Concern disqualify a vehicle,
    for vehicles acquired after 2024.
  - "Recycled in North America" pathway: critical minerals recycled in
    North America count toward the critical-minerals percentage
    regardless of where they were originally extracted — this is the
    ONE field this platform can genuinely help with, by checking
    src/recycler_directory.py for a North-America-region recycler
    compatible with this cell's chemistry.
  (26 U.S.C. § 30D; Treasury/IRS final rule, 89 FR 37706, May 6, 2024)
"""

import datetime

PREFIX = "ira30d_"


def build_ira_30d_entry(
    cell_id: str,
    chemistry: str,
    soh: float,
    recycled_in_north_america_pathway_available: bool,
) -> dict:
    """
    Build one field-structure-demonstration entry for Section 30D's
    critical-minerals/battery-components reporting concepts.

    recycled_in_north_america_pathway_available: caller-computed via
    src/recycler_directory.py — True if a North-America-region recycler
    in this platform's directory is compatible with this cell's
    chemistry, meaning IF this specific cell were actually routed there
    for recycling, the recovered critical minerals WOULD be eligible
    under the "recycled in North America" pathway. This is a genuine,
    computed fact about this platform's own recycler directory — not a
    claim that this cell (built from public research data, not a real
    manufactured vehicle battery) is itself IRA-eligible.
    """
    return {
        "type": "us_ira_section_30d_traceability",
        "id": cell_id,
        "attributes": {
            f"{PREFIX}chemistry": chemistry,
            f"{PREFIX}soh_pct": round(float(soh), 2),
            f"{PREFIX}critical_minerals_percentage_required_2026": {
                "value": 70, "unit": "%", "state": "available",
                "note": "Statutory threshold for vehicles placed in service in calendar year 2026 — a regulatory constant, not a per-cell measurement.",
            },
            f"{PREFIX}battery_components_percentage_required_2026": {
                "value": 70, "unit": "%", "state": "available",
                "note": "Statutory threshold for vehicles placed in service in calendar year 2026 — a regulatory constant, not a per-cell measurement.",
            },
            f"{PREFIX}critical_minerals_percentage_actual": {
                "value": None, "state": "unavailable",
                "note": "Requires auditable supply-chain-of-custody records (extraction/processing location per mineral) this platform does not have — it works from public research datasets and synthetic/uploaded cycling data, not manufacturer sourcing records.",
            },
            f"{PREFIX}battery_components_percentage_actual": {
                "value": None, "state": "unavailable",
                "note": "Requires manufacturer component-sourcing records this platform does not have.",
            },
            f"{PREFIX}feoc_compliant": {
                "value": None, "state": "unavailable",
                "note": "Requires supply-chain provenance data (which entities extracted/processed/recycled each mineral) this platform does not have.",
            },
            f"{PREFIX}recycled_in_north_america_pathway_available": {
                "value": bool(recycled_in_north_america_pathway_available), "state": "estimated",
                "note": "Computed from this platform's own recycler directory (src/recycler_directory.py) — True means a North-America-region recycler compatible with this cell's chemistry exists in that directory, so critical minerals recovered THERE would qualify under the recycled-in-North-America pathway. Not a claim that this specific cell is itself an IRA-eligible vehicle battery.",
            },
        },
    }


def to_ira_30d_document(entry: dict, cell_id: str) -> dict:
    """Wrap build_ira_30d_entry()'s output with a meta/disclaimer block —
    same "wrap the core dict, don't reinvent it" pattern as
    optimade_export.to_optimade_document()."""
    return {
        "meta": {
            "cell_id": cell_id,
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "format": "us-ira-section-30d-traceability-structure",
            "format_reference": "26 U.S.C. § 30D; Treasury/IRS final rule, 89 FR 37706 (2024-05-06)",
            "disclaimer": (
                "Field-structure demonstration of Section 30D's critical-minerals/"
                "battery-components reporting concepts, NOT a compliance claim or "
                "tax-credit eligibility determination. This platform has no "
                "supply-chain-of-custody data — every field requiring it is marked "
                "unavailable. A real Section 30D filing requires a qualified "
                "manufacturer's own auditable sourcing records."
            ),
        },
        "data": entry,
    }
