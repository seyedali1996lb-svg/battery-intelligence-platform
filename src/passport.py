"""
Phase 5 — Battery Passport data model.

Builds the EU Battery Regulation (EU) 2023/1542-structured field set for a
single cell, with every field tagged as one of three states:

  "available"   — real pipeline output (SOH, RUL, chemistry, cycle count...)
  "estimated"   — Phase 4 cited/illustrative assumption (CO2, second-life value)
  "unavailable" — field the regulation requires that this demo genuinely
                  does not have (manufacturer record, supply chain, audit)

This is a data-structure demonstration, not a regulatory submission. No
field here should be read as a compliance claim.

One function (`build_passport`) is the single source of truth for field
values, consumed by both the Streamlit Passport page and the PDF report,
so the two surfaces can never drift out of sync.
"""

from consequences import CELL_NOMINAL_KWH, ASSUMPTIONS, sustainability_snapshot


def build_passport(
    cell_id: str,
    df,
    bundle: dict,
    rul_reliable: bool,
    is_nasa: bool,
) -> dict:
    """
    Returns a dict keyed by group name, each a list of field dicts:
      {"label": str, "value": str, "state": "available"|"estimated"|"unavailable",
       "note": str (optional source/citation)}
    """
    source = "nasa" if is_nasa else "synth"
    latest = df.iloc[-1]

    soh         = float(latest["soh_pct"])
    cycle_count = int(latest["cycle_number"])
    resistance  = float(latest.get("resistance_ohm", float("nan")))
    fade_30     = float(latest.get("fade_rate_30cy", float("nan")))
    rul_pred    = latest.get("rul_pred", None)
    lco_soh_r2  = bundle["metrics"].get("lco_soh_r2", float("nan"))

    cell_kwh = CELL_NOMINAL_KWH[source]

    if is_nasa:
        chemistry = "LiCoO₂ (lithium cobalt oxide), 18650 cylindrical"
        data_src  = "NASA PCoE Battery Aging Dataset — real measured data"
        usage_src = "Test conditions: 24°C, 2A discharge, 100% DoD (Saha & Goebel, 2007)"
    else:
        chemistry = "Synthetic Li-ion model (physics-informed, not a real cell)"
        data_src  = "Synthetic generator — Arrhenius SEI growth + C-rate + Rainflow DoD"
        usage_src = "Injected stress profile — see sidebar for this cell's T / C-rate / DoD"

    identity = [
        {"label": "Cell ID", "value": cell_id, "state": "available"},
        {"label": "Chemistry type", "value": chemistry, "state": "available"},
        {
            "label": "Nominal capacity",
            "value": f"{cell_kwh*1000:.2f} Wh ({cell_kwh/3.6*1000:.2f} Ah @ 3.6V nominal)",
            "state": "available",
            "note": "NASA PCoE datasheet spec" if is_nasa else "Oxford-style 18650 dataset spec",
        },
        {"label": "Data source", "value": data_src, "state": "available"},
        {"label": "Manufacturer", "value": "Not available in this demonstration", "state": "unavailable"},
        {"label": "Serial number / production batch", "value": "Not available in this demonstration", "state": "unavailable"},
    ]

    rul_value = f"{float(rul_pred):.0f} cycles" if (rul_reliable and rul_pred is not None) else "Not calibrated (fold R² below 0.30 reliability floor)"

    soh_group = [
        {"label": "State of Health", "value": f"{soh:.1f}%", "state": "available", "note": "Validated model output — leave-cell-out tested"},
        {"label": "Cycle count", "value": f"{cycle_count:,}", "state": "available"},
        {"label": "Internal resistance", "value": f"{resistance:.4f} Ω" if resistance == resistance else "n/a", "state": "available"},
        {"label": "Fade rate (30-cycle window)", "value": f"{fade_30*1000:.2f} mAh/cy" if fade_30 == fade_30 else "n/a", "state": "available", "note": "Validated model output"},
        {"label": "Remaining Useful Life (RUL)", "value": rul_value, "state": "available"},
        {"label": "SOH model accuracy (leave-cell-out R²)", "value": f"{lco_soh_r2:.3f}" if lco_soh_r2 == lco_soh_r2 else "n/a", "state": "available"},
    ]

    lifecycle = [
        {"label": "Cycle history (SOH vs. cycle)", "value": "Available — see Health page chart", "state": "available"},
        {"label": "Usage profile / test conditions", "value": usage_src, "state": "available"},
        {"label": "Repair / refurbishment history", "value": "Not available in this demonstration", "state": "unavailable", "note": "Neither dataset models repair events"},
        {"label": "Prior-owner / installation history", "value": "Not available in this demonstration", "state": "unavailable"},
    ]

    # Carbon — use ASSUMPTIONS mid-point defaults (no sliders on the Passport page;
    # adjust scenario values on the Consequences page).
    co2_default = float(ASSUMPTIONS["co2_manufacture"]["value"])
    mat_default = float(ASSUMPTIONS["material_recovery"]["value"])
    sus = sustainability_snapshot(source=source, co2_per_cell=co2_default, material_recovery=mat_default)

    carbon = [
        {
            "label": "CO₂ avoided by reuse (vs. new cell manufacture)",
            "value": f"~{sus['co2_avoided_by_reuse']:.2f} kg CO₂e",
            "state": "estimated",
            "note": ASSUMPTIONS["co2_manufacture"]["source"] + " (default mid-point shown; adjust on Consequences page)",
        },
        {
            "label": "CO₂ recycling credit",
            "value": f"~{sus['co2_recycling_credit']:.2f} kg CO₂e",
            "state": "estimated",
            "note": "~15% cathode-material credit, Dunn et al. (2015) — hardcoded factor, no slider",
        },
        {"label": "Full lifecycle carbon audit (mining → manufacture → transport → use → end-of-life)", "value": "Not available in this demonstration", "state": "unavailable", "note": "Requires real supply-chain data this platform does not have"},
        {"label": "Verified carbon footprint declaration (Art. 7)", "value": "Not available in this demonstration", "state": "unavailable", "note": "Requires third-party accredited audit"},
    ]

    all_fields = identity + soh_group + lifecycle + carbon
    n_available   = sum(1 for f in all_fields if f["state"] == "available")
    n_estimated   = sum(1 for f in all_fields if f["state"] == "estimated")
    n_unavailable = sum(1 for f in all_fields if f["state"] == "unavailable")

    return {
        "cell_id":   cell_id,
        "identity":  identity,
        "soh":       soh_group,
        "lifecycle": lifecycle,
        "carbon":    carbon,
        "summary": {
            "n_available": n_available,
            "n_estimated": n_estimated,
            "n_unavailable": n_unavailable,
            "n_total": len(all_fields),
        },
    }
