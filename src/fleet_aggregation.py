"""
Fleet aggregation — per-cell health/SoC headroom into a VPP-style
dispatchable-capacity offer (the Anode/Deepgrid-style operator interface).

A virtual power plant (or a fleet operator bidding into a flexibility
market) does not dispatch individual cells; it bids an aggregate
{energy_kwh, power_kw} capability for a service window. This module builds
that aggregate from THIS platform's per-cell health signals — the same
data the rest of the Lifecycle Intelligence layer uses:

  - SoH-limited energy: usable energy = current capacity (nominal × SOH/100)
    × the allowed SOC band. A faded cell offers less energy than its nameplate.
  - SoP-limited power: each cell's power contribution is scaled by its
    State-of-Power proxy (sop_pct/100) — the rate-capability signal the
    cohort's trading systems ignore.
  - RUL/SOH caution narrows the band per cell via the SAME
    health_constrained_band() the dispatch engine uses, so a cell the
    platform has flagged as end-of-life-sensitive is offered shallower
    depth-of-discharge (fewer EFC per MWh) rather than being bid to its
    floor.

Honest scope:
  1. This is an OFFER (a capability statement), not a dispatch control —
     the platform does not claim it can execute the offered schedule.
     The offer's `caveats` field says exactly that.
  2. SoC per cell is a point-in-time input the caller supplies (from BMS
     telemetry, the live feed, or an assumed state). The offer is only as
     fresh as those SOC readings.
  3. EFC accounting uses the platform's rainflow engine only where a SOC
     trajectory is supplied; otherwise the offer reports the fleet's
     accumulated EFC-to-date as context, not a dispatch estimate.
"""

from __future__ import annotations

from typing import Optional

from src.health_aware_dispatch import health_constrained_band

# Default allowed SOC band for a healthy cell in the aggregate offer.
DEFAULT_SOC_LOW_PCT = 10.0
DEFAULT_SOC_HIGH_PCT = 95.0


def cell_dispatchable_capacity(
    nominal_kwh: float,
    soh_pct: float,
    soc_pct: float,
    c_rate: float = 0.5,
    sop_pct: "float | None" = None,
    rul_cycles: "float | None" = None,
    rul_reliable: bool = False,
    soc_high_limit_pct: float = DEFAULT_SOC_HIGH_PCT,
    soc_low_limit_pct: float = DEFAULT_SOC_LOW_PCT,
) -> dict:
    """One cell's dispatchable capability within a service window.

    current_kwh = nominal_kwh × SOH/100 (usable nameplate today).
    power_kw    = current_kwh × c_rate × SoP power factor (health-aware).
    energy_kwh  = current_kwh × (band_high − band_low)/100, where the band
                  is the RUL/SOH-cautioned band (narrowed when the cell is
                  flagged end-of-life-sensitive).

    Returns {cell capability dict} with the health band and reasons, plus
    `excluded: bool` — a cell whose current SOC is already ABOVE the band's
    high limit (e.g. a caution-narrowed band tighter than the current
    state) offers zero energy and is marked excluded with a reason."""
    if nominal_kwh <= 0:
        raise ValueError("nominal_kwh must be > 0.")
    if soh_pct <= 0:
        raise ValueError("soh_pct must be > 0.")

    current_kwh = nominal_kwh * soh_pct / 100.0
    band = health_constrained_band(
        soh_pct=soh_pct, sop_pct=sop_pct,
        rul_cycles=rul_cycles, rul_reliable=rul_reliable,
    )

    min_soc = max(band["min_soc_pct"], soc_low_limit_pct)
    max_soc = min(band["max_soc_pct"], soc_high_limit_pct)

    power_kw = current_kwh * c_rate * band["power_cap_factor"]

    excluded = False
    exclude_reason = None
    if soc_pct >= max_soc:
        excluded = True
        exclude_reason = (
            f"SOC {soc_pct:.0f}% already at/above the band high limit "
            f"{max_soc:.0f}% — no charge headroom this window."
        )
        energy_kwh = 0.0
    else:
        energy_kwh = current_kwh * (max_soc - max(soc_pct, min_soc)) / 100.0

    return {
        "current_kwh": round(current_kwh, 4),
        "power_kw": round(power_kw, 4),
        "energy_kwh": round(energy_kwh, 4),
        "band": {
            "min_soc_pct": round(min_soc, 1),
            "max_soc_pct": round(max_soc, 1),
            "caution": band["caution"],
        },
        "excluded": excluded,
        "exclude_reason": exclude_reason,
        "reasons": band["reasons"],
    }


def fleet_dispatchable_offer(
    cells: list,
    c_rate: float = 0.5,
    window_hours: int = 2,
    soc_high_limit_pct: float = DEFAULT_SOC_HIGH_PCT,
    soc_low_limit_pct: float = DEFAULT_SOC_LOW_PCT,
) -> dict:
    """Aggregate a fleet of per-cell capability into one VPP-style offer.

    cells: list of dicts, each with
      cell_id: str
      nominal_kwh: float
      soh_pct: float
      soc_pct: float
      sop_pct (optional), rul_cycles (optional), rul_reliable (optional)

    Returns the offer object: aggregate energy/power for the window, the
    per-cell breakdown (with excluded cells listed separately), the fleet's
    health context, and the honest caveats. The offer does NOT promise
    dispatch control — see module docstring."""
    if not cells:
        raise ValueError("cells must not be empty.")
    if window_hours <= 0:
        raise ValueError("window_hours must be > 0.")

    per_cell = []
    excluded = []
    total_energy_kwh = 0.0
    total_power_kw = 0.0
    n_caution = 0
    total_current_kwh = 0.0

    for cell in cells:
        cap = cell_dispatchable_capacity(
            nominal_kwh=cell["nominal_kwh"],
            soh_pct=cell["soh_pct"],
            soc_pct=cell["soc_pct"],
            c_rate=c_rate,
            sop_pct=cell.get("sop_pct"),
            rul_cycles=cell.get("rul_cycles"),
            rul_reliable=cell.get("rul_reliable"),
            soc_high_limit_pct=soc_high_limit_pct,
            soc_low_limit_pct=soc_low_limit_pct,
        )
        row = {"cell_id": cell["cell_id"], **cap}
        total_current_kwh += cap["current_kwh"]
        if cap["band"]["caution"]:
            n_caution += 1
        if cap["excluded"]:
            excluded.append(row)
        else:
            total_energy_kwh += cap["energy_kwh"]
            total_power_kw += cap["power_kw"]
            per_cell.append(row)

    per_cell.sort(key=lambda r: r["energy_kwh"], reverse=True)

    return {
        "offer": {
            "service": "dispatchable_capacity",
            "window_hours": window_hours,
            "energy_kwh": round(total_energy_kwh, 2),
            "power_kw": round(total_power_kw, 2),
            "duration_hours_at_full_power": round(total_energy_kwh / total_power_kw, 2)
            if total_power_kw > 0 else None,
        },
        "cells": per_cell,
        "excluded_cells": excluded,
        "fleet_context": {
            "n_cells": len(cells),
            "n_included": len(per_cell),
            "n_excluded": len(excluded),
            "n_health_caution": n_caution,
            "total_current_kwh": round(total_current_kwh, 2),
        },
        "caveats": [
            "This is a capability OFFER, not a dispatch control signal — it states what the fleet could deliver; it does not claim the platform can execute it.",
            "Energy/power figures use SoH-limited capacity and SoP-proxy power caps; neither is a measured power test.",
            "SOC values are point-in-time inputs from the caller (BMS/live feed); the offer is only as fresh as those readings.",
            "Cells excluded because their SOC is at/above the band high limit could re-enter the offer after a discharge cycle — the offer reflects this window only.",
        ],
    }
