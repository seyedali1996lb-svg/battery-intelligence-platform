"""
Tariff-aware managed EV charging — the Optiwatt-adjacent module of the
Lifecycle Intelligence layer.

An EV plugged into an OCPP charge point (see src/bms_connectors.py's
fetch_ocpp_sessions()/OCPPAdapter for how this platform already reads
completed charging sessions) can be charged at any hour within its plug-in
window. Managed charging picks the cheapest hours, shifting the session's
energy into off-peak/low-price hours — the same demand-balancing use case
Optiwatt-style platforms sell, but built on this platform's own market-data
adapter and measured with its own partial-cycle/EFC engine.

Honest scope:
  1. The plan is computed from a supplied price series (any MarketDataAdapter
     feed via to_eur_per_kwh()); it is a recommendation, not a control
     signal — this module does NOT push commands to a charger. The OCPP
     adapter in this repo reads sessions from a Central System's REST
     reporting API and is explicit about not speaking the live OCPP
     protocol (see that module's docstring); pushing a computed schedule to
     a charge point is the same out-of-scope, untested-against-hardware
     territory and is deliberately not attempted.
  2. The "flexibility delivered" by a managed schedule is measured as the
     shift in cost vs. unmanaged charging AND the Equivalent Full Cycles
     (EFC) the session's SOC trajectory represents, via the platform's own
     ASTM E1049-85 rainflow engine — a real stress ledger, not just a
     cost delta.
  3. Charging efficiency is an assumption (default 0.94, cited below), not
     a measured value for any specific EV.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from batlab.features.partial_cycles import rainflow_counting

# Assumed AC charging efficiency (wall-to-battery). Cited estimate: AC
# level-2 charging losses are commonly reported in the 5-10% range
# (US DOE / EVSE surveys); 0.94 is a mid-range default.
CHARGING_EFFICIENCY = 0.94


def _unmanaged_cost(prices: list, charge_kw_hours: list, efficiency: float) -> float:
    return sum(p * e / efficiency for p, e in zip(prices, charge_kw_hours))


def managed_charge_plan(
    prices_eur_per_kwh: list,
    battery_kwh: float,
    initial_soc_pct: float,
    target_soc_pct: float,
    max_charge_kw: float,
    efficiency: float = CHARGING_EFFICIENCY,
    unmanaged_start_hour: "int | None" = None,
) -> dict:
    """Cheapest-hour charging plan over a horizon of len(prices) hours.

    The EV is assumed plugged in for the whole horizon (a single session
    window). Greedy fill: the energy needed to reach target_soc_pct is
    scheduled into the cheapest hours, at up to max_charge_kw per hour,
    never exceeding the battery's remaining headroom (efficiency applied so
    wall energy >= battery energy). Ties (equal prices) fill earlier hours
    first — deterministic and reproducible.

    The unmanaged baseline (what a dumb charger does) charges at
    max_charge_kw starting at `unmanaged_start_hour` (default: hour 0, the
    moment the car plugs in) — the standard "plug in and charge now"
    behavior managed charging improves on.

    Returns:
      plan: [{hour, charge_kw, price_eur_per_kwh, soc_pct}]
      cost_eur, unmanaged_cost_eur, savings_eur, savings_pct,
      delivered_kwh (battery-side), wall_energy_kwh,
      flexibility_efc (rainflow EFC of the SOC trajectory),
      limitations: [str]
    """
    prices = [float(p) for p in prices_eur_per_kwh]
    n = len(prices)
    if n == 0:
        raise ValueError("prices_eur_per_kwh must not be empty.")
    if battery_kwh <= 0:
        raise ValueError("battery_kwh must be > 0.")
    if max_charge_kw <= 0:
        raise ValueError("max_charge_kw must be > 0.")
    if not (0 <= initial_soc_pct <= 100 and 0 <= target_soc_pct <= 100):
        raise ValueError("initial/target SOC must be in [0, 100].")
    if target_soc_pct < initial_soc_pct:
        raise ValueError("target_soc_pct must be >= initial_soc_pct.")

    battery_energy_needed_kwh = battery_kwh * (target_soc_pct - initial_soc_pct) / 100.0
    if battery_energy_needed_kwh <= 1e-9:
        # Nothing to charge — early return with a zero-cost plan.
        return {
            "plan": [{"hour": h, "charge_kw": 0.0, "price_eur_per_kwh": round(p, 5),
                      "soc_pct": initial_soc_pct} for h, p in enumerate(prices)],
            "cost_eur": 0.0,
            "unmanaged_cost_eur": 0.0,
            "savings_eur": 0.0,
            "savings_pct": 0.0,
            "delivered_kwh": 0.0,
            "wall_energy_kwh": 0.0,
            "flexibility_efc": 0.0,
            "limitations": _LIMITATIONS,
        }

    wall_energy_needed_kwh = battery_energy_needed_kwh / efficiency

    # Greedy cheapest-hour fill. (hour, price) pairs sorted by price then
    # hour (stable, deterministic). Respect max_charge_kw per hour.
    charge_kw_hours = [0.0] * n
    remaining_wall_kwh = wall_energy_needed_kwh
    for hour, price in sorted(enumerate(prices), key=lambda hp: (hp[1], hp[0])):
        if remaining_wall_kwh <= 0:
            break
        take = min(max_charge_kw, remaining_wall_kwh)
        charge_kw_hours[hour] = round(take, 4)
        remaining_wall_kwh -= take

    if remaining_wall_kwh > 1e-6:
        # The horizon is too short to reach target at max_charge_kw —
        # report honestly rather than pretending the target was reached.
        reached_soc_pct = initial_soc_pct + (wall_energy_needed_kwh - remaining_wall_kwh) * efficiency / battery_kwh * 100.0
    else:
        reached_soc_pct = target_soc_pct

    # SOC trajectory (battery-side energy), for the EFC ledger.
    soc = float(initial_soc_pct)
    plan = []
    for h in range(n):
        charge_battery_kwh = charge_kw_hours[h] * efficiency
        soc = min(100.0, soc + charge_battery_kwh / battery_kwh * 100.0)
        plan.append({
            "hour": h,
            "charge_kw": charge_kw_hours[h],
            "price_eur_per_kwh": round(prices[h], 5),
            "soc_pct": round(soc, 2),
        })

    cost = _unmanaged_cost(prices, charge_kw_hours, efficiency)

    # Unmanaged baseline: charge at max_charge_kw from unmanaged_start_hour.
    start_hour = unmanaged_start_hour if unmanaged_start_hour is not None else 0
    unmanaged_charge = [0.0] * n
    remaining = wall_energy_needed_kwh
    for h in range(start_hour, n):
        if remaining <= 0:
            break
        take = min(max_charge_kw, remaining)
        unmanaged_charge[h] = take
        remaining -= take
    unmanaged_cost = _unmanaged_cost(prices, unmanaged_charge, efficiency)

    # Turning-point preprocessing before rainflow: the engine's extrema
    # detector misses plateau-heavy profiles (see
    # src/health_aware_dispatch._turning_points for the full explanation).
    soc_series = [s["soc_pct"] for s in plan]
    turning = [soc_series[0]]
    for v in soc_series[1:]:
        if abs(v - turning[-1]) > 1e-9:
            turning.append(v)
    rainflow = rainflow_counting(np.array(turning, dtype=float))

    savings = max(0.0, unmanaged_cost - cost)
    savings_pct = (savings / unmanaged_cost * 100.0) if unmanaged_cost > 0 else 0.0

    return {
        "plan": plan,
        "cost_eur": round(cost, 2),
        "unmanaged_cost_eur": round(unmanaged_cost, 2),
        "savings_eur": round(savings, 2),
        "savings_pct": round(savings_pct, 1),
        "delivered_kwh": round(battery_energy_needed_kwh, 2),
        "wall_energy_kwh": round(wall_energy_needed_kwh, 2),
        "reached_soc_pct": round(reached_soc_pct, 2),
        "flexibility_efc": rainflow["equivalent_full_cycles"],
        "limitations": _LIMITATIONS,
    }


_LIMITATIONS = [
    "This is a recommended schedule, not a control signal — this module does not push commands to a charger (the OCPP adapter only reads completed sessions; pushing a schedule to a live charge point is untested-against-hardware and out of scope).",
    "The car is assumed plugged in for the whole horizon at max_charge_kw capability — real vehicles negotiate lower power, which only reduces the achievable savings.",
    "Charging efficiency is an assumption (default 0.94), not a measured value for any specific EV.",
    "Prices are assumed known in advance (day-ahead-style); real-time charging decisions would need a forecast.",
]
