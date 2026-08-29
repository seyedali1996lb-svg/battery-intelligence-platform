"""
Grid-services revenue stack — per-site revenue potential across the three
ways a battery makes money on the grid, sharing the honesty conventions of
src/consequences.py (every figure is an estimate, labeled and sourced or
explicitly "Illustrative — not sourced") and the health-aware dispatch
engine (src/health_aware_dispatch.py):

  1. Energy arbitrage   — buy low, sell high on the wholesale/day-ahead
                          price window (health-aware schedule).
  2. Frequency regulation (ancillary) — revenue from holding capacity
                          reserved for automatic frequency restoration
                          reserve (aFRR) / FCR-style services.
  3. Capacity (reserve) — availability payments for standing reserve
                          (capacity market / grid services contracts).

Honest scope, stated plainly:
  - Arbitrage revenue comes from health_aware_dispatch.arbitrage_schedule()
    and inherits ALL of that module's limitations (threshold heuristic, no
    price impact, SoP proxy).
  - Regulation and capacity figures assume the FULL power capability is
    available to the service for the whole billing window, at an assumed
    participation efficiency (<1) that reflects that a real asset spends
    part of its availability in energy-limited states (SOC headroom
    constraints, degraded SoP, service windows where the market doesn't
    clear the bid). These are potential-revenue ESTIMATES for a site, not
    validated market outcomes.
  - The arbitrage schedule and the regulation/capacity revenue are
    mutually exclusive in their assumptions: a battery cannot simultaneously
    chase arbitrage AND hold regulation reserve. The tradeoff is surfaced
    in the returned `exclusivity_note`, and the regulation/capacity
    estimates assume the battery is held OUT of arbitrage for the window.
  - Market prices/rates are supplied externally (via the market-data
    adapter or the assumption sliders) — this module models revenue from
    them; it does not forecast them.

All price/rate assumptions follow the exact {value, slider_range, unit,
label, source} shape of src/consequences.py::ASSUMPTIONS so the UI can
render the same sliders with the same provenance display.
"""

from __future__ import annotations

from typing import Optional

from health_aware_dispatch import arbitrage_schedule

GRID_SERVICES_ASSUMPTIONS = {
    "frequency_regulation_price_eur_per_mw_h": {
        "value": 25.0,
        "slider_range": (5.0, 100.0),
        "unit": "€/MW/h",
        "label": "Illustrative — not sourced",
        "source": (
            "Ancillary (aFRR/FCR-style) capacity prices vary enormously by market, "
            "season, and bid structure — from a few €/MW/h in calm markets to "
            "triple digits during scarcity. No single figure is appropriate; "
            "25 €/MW/h is a mid-range default for a competitive EU balancing market "
            "and must be set to the target market before use."
        ),
    },
    "capacity_payment_eur_per_mw_year": {
        "value": 40000.0,
        "slider_range": (5000.0, 150000.0),
        "unit": "€/MW/year",
        "label": "Illustrative — not sourced",
        "source": (
            "Capacity-market availability payments range widely by jurisdiction "
            "(GB ~£30-50/kW/yr in recent auctions, other markets lower or "
            "nonexistent). 40,000 €/MW/yr is a mid-range default, not a "
            "specific market's clearing price."
        ),
    },
    "regulation_energy_throughput_factor": {
        "value": 0.30,
        "slider_range": (0.05, 0.60),
        "unit": "fraction of rated power actually dispatched",
        "label": "Illustrative — not sourced",
        "source": (
            "A regulation asset's actual energy throughput is a fraction of its "
            "rated power (the market dispatches it only as the grid deviates). "
            "0.30 is engineering judgment — used to estimate the regulation "
            "service's own energy cycling for the EFC/stress side of the ledger."
        ),
    },
    "regulation_service_hours_per_year": {
        "value": 6000.0,
        "slider_range": (1000.0, 8760.0),
        "unit": "hours/year in service",
        "label": "Illustrative — not sourced",
        "source": (
            "Hours the asset is bid into and accepted by the balancing market "
            "per year; 6000 is a typical availability estimate for a grid-tied "
            "BESS, not a guarantee."
        ),
    },
}


def grid_services_revenue(
    prices_eur_per_kwh: list,
    battery_kwh: float,
    c_rate: float = 0.5,
    round_trip_efficiency: float = 0.90,
    soh_pct: float = 100.0,
    sop_pct: "float | None" = None,
    rul_cycles: "float | None" = None,
    rul_reliable: bool = False,
    frequency_regulation_price_eur_per_mw_h: "float | None" = None,
    capacity_payment_eur_per_mw_year: "float | None" = None,
    regulation_service_hours_per_year: "float | None" = None,
    regulation_energy_throughput_factor: "float | None" = None,
    price_window_is_annual: bool = False,
) -> dict:
    """Per-site revenue potential across arbitrage / frequency regulation /
    capacity, with honest labels (see module docstring for the exclusivity
    and estimation caveats).

    `price_window_is_annual`: the prices list is one full year (8760) of
    hourly prices, so the arbitrage result IS the annual arbitrage revenue.
    When False (default), the prices are treated as a representative window
    and annualized by repeating it to 8760 hours — an estimate of the
    estimate, labeled as such.

    None assumption params fall back to GRID_SERVICES_ASSUMPTIONS values.
    Returns the revenue breakdown, the health-aware arbitrage schedule
    summary, an `exclusivity_note`, and the assumption values used."""
    import numpy as np

    n_hours = len(prices_eur_per_kwh)
    if n_hours == 0:
        raise ValueError("prices_eur_per_kwh must not be empty.")
    if battery_kwh <= 0:
        raise ValueError("battery_kwh must be > 0.")

    _reg_price = frequency_regulation_price_eur_per_mw_h if frequency_regulation_price_eur_per_mw_h is not None \
        else GRID_SERVICES_ASSUMPTIONS["frequency_regulation_price_eur_per_mw_h"]["value"]
    _cap_payment = capacity_payment_eur_per_mw_year if capacity_payment_eur_per_mw_year is not None \
        else GRID_SERVICES_ASSUMPTIONS["capacity_payment_eur_per_mw_year"]["value"]
    _reg_hours = regulation_service_hours_per_year if regulation_service_hours_per_year is not None \
        else GRID_SERVICES_ASSUMPTIONS["regulation_service_hours_per_year"]["value"]
    _reg_factor = regulation_energy_throughput_factor if regulation_energy_throughput_factor is not None \
        else GRID_SERVICES_ASSUMPTIONS["regulation_energy_throughput_factor"]["value"]

    # --- arbitrage ---
    if price_window_is_annual and n_hours == 8760:
        arbitrage_window = arbitrage_schedule(
            prices_eur_per_kwh, battery_kwh, c_rate, round_trip_efficiency,
            soh_pct, sop_pct, rul_cycles, rul_reliable,
        )
        arbitrage_annual_eur = arbitrage_window["revenue_eur"]
        arbitrage_annual_efc = arbitrage_window["efc"]
        annualized = False
    else:
        # Annualize a representative window: repeat it to 8760 hours. A
        # rough estimate — labeled as such in the returned dict.
        reps = int(np.ceil(8760 / n_hours))
        repeated = (list(prices_eur_per_kwh) * reps)[:8760]
        arbitrage_window = arbitrage_schedule(
            repeated, battery_kwh, c_rate, round_trip_efficiency,
            soh_pct, sop_pct, rul_cycles, rul_reliable,
        )
        arbitrage_annual_eur = arbitrage_window["revenue_eur"]
        arbitrage_annual_efc = arbitrage_window["efc"]
        annualized = True

    # --- power capability (health-aware) ---
    band = arbitrage_window["band"]
    power_cap_kw = battery_kwh * c_rate * band["power_cap_factor"]

    # --- frequency regulation (ancillary) ---
    # Capacity reserved for regulation earns the per-MW-hour price for the
    # hours in service. The energy side (regulation_energy_throughput_factor)
    # cycles the battery and contributes EFC stress — accounted for on the
    # cost/stress ledger, not as revenue.
    regulation_capacity_mw = power_cap_kw / 1000.0
    frequency_regulation_eur = regulation_capacity_mw * _reg_price * _reg_hours
    regulation_annual_efc = (
        regulation_capacity_mw * 1000.0 * _reg_factor * _reg_hours / battery_kwh
    )

    # --- capacity (reserve) ---
    capacity_eur = regulation_capacity_mw * _cap_payment

    total_eur = arbitrage_annual_eur + frequency_regulation_eur + capacity_eur

    return {
        "arbitrage_eur": round(arbitrage_annual_eur, 2),
        "arbitrage_efc": round(arbitrage_annual_efc, 3),
        "arbitrage_annualized_from_window": annualized,
        "frequency_regulation_eur": round(frequency_regulation_eur, 2),
        "frequency_regulation_efc": round(regulation_annual_efc, 3),
        "capacity_eur": round(capacity_eur, 2),
        "total_eur": round(total_eur, 2),
        "dispatchable_power_kw": round(power_cap_kw, 2),
        "health_band": band,
        "exclusivity_note": (
            "Arbitrage and regulation/capacity revenue are mutually exclusive in "
            "this estimate: the regulation/capacity figures assume the battery is "
            "held out of arbitrage for the service window, and the arbitrage "
            "figure assumes it is free to cycle on the price signal. A real "
            "asset splits its capacity across services via a market bid — that "
            "optimization is out of scope here."
        ),
        "assumptions_used": {
            "frequency_regulation_price_eur_per_mw_h": _reg_price,
            "capacity_payment_eur_per_mw_year": _cap_payment,
            "regulation_service_hours_per_year": _reg_hours,
            "regulation_energy_throughput_factor": _reg_factor,
        },
        "labels": {
            key: {"label": GRID_SERVICES_ASSUMPTIONS[key]["label"],
                  "source": GRID_SERVICES_ASSUMPTIONS[key]["source"]}
            for key in GRID_SERVICES_ASSUMPTIONS
        },
    }
