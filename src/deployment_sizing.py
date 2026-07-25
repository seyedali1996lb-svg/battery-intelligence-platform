"""
Solar + Storage Sizing — deployment sizing and payback/NPV for a candidate
second-life battery paired with a new PV array (Consequences page).

IMPORTANT: estimate_annual_savings() is a MONTHLY ENERGY-BALANCE
APPROXIMATION, not an hourly dispatch simulation. A naive month-level
match between PV generation and consumption systematically OVERSTATES
real self-consumption, because it ignores day/night mismatch within the
month (well documented in the PV self-consumption literature, e.g.
Luthander et al. 2015, "Photovoltaic self-consumption in buildings: A
review", Applied Energy 142). SIZING_ASSUMPTIONS["self_consumption_derating"]
exists specifically to correct for this — do not remove or set it to 1.0
to make results look better; that would make the tool systematically
oversell paybacks, which is the opposite of this app's citation/honesty
standard (see src/consequences.py's ASSUMPTIONS for the same convention).

This module has no Streamlit dependency (mirrors src/consequences.py /
src/pack_builder.py). PV yield data is supplied externally (normally via
src/pvgis_client.py) rather than modelled locally — see size_deployment()'s
pv_yield_fn parameter.
"""

from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------
# Same {value, slider_range, unit, label, source} shape as
# src/consequences.py::ASSUMPTIONS. label is one of:
# "Cited estimate", "Illustrative — not sourced".
#
# These are a SEPARATE cost basis from consequences.py's repack_cost /
# second_life_value_per_kwh, which price a cell's reuse/recycle decision,
# not an installed grid-tied BESS+PV system (inverter, BMS, enclosure,
# wiring, PV mounting/labour) — do not conflate the two.

SIZING_ASSUMPTIONS = {
    "round_trip_efficiency": {
        "value": 0.90,
        "slider_range": (0.80, 0.98),
        "unit": "fraction",
        "label": "Cited estimate",
        "source": (
            "Modern lithium-ion BESS AC-AC round-trip efficiency is typically "
            "reported in the 85-95% range by storage cost/performance surveys "
            "(e.g. NREL/IRENA battery storage reports); 0.90 is a mid-range default."
        ),
    },
    "self_consumption_derating": {
        "value": 0.70,
        "slider_range": (0.40, 0.95),
        "unit": "fraction",
        "label": "Illustrative — not sourced",
        "source": (
            "Corrects for the known overestimation bias of a monthly (rather than "
            "hourly) PV/load balance — see module docstring and Luthander et al. "
            "(2015, Applied Energy 142) for the qualitative finding. The specific "
            "0.70 default is engineering judgment, not taken directly from that paper."
        ),
    },
    "battery_cycles_per_day": {
        "value": 1.0,
        "slider_range": (0.5, 2.0),
        "unit": "cycles/day",
        "label": "Illustrative — not sourced",
        "source": (
            "Typical single daily charge/discharge duty cycle assumed for "
            "behind-the-meter residential/commercial self-consumption + "
            "tariff-arbitrage use — most such systems don't cycle more than once a day."
        ),
    },
    "pv_install_cost_eur_per_kwp": {
        "value": 1200.0,
        "slider_range": (700.0, 2000.0),
        "unit": "€/kWp",
        "label": "Illustrative — not sourced",
        "source": (
            "Broadly consistent with EU residential/small-commercial installed PV "
            "system cost ranges reported by IRENA's Renewable Power Generation Costs "
            "series and SolarPower Europe market reports — varies significantly by "
            "country and system size, so treat this as an order-of-magnitude default."
        ),
    },
    "bess_install_cost_eur_per_kwh": {
        "value": 600.0,
        "slider_range": (300.0, 1000.0),
        "unit": "€/kWh",
        "label": "Illustrative — not sourced",
        "source": (
            "Installed behind-the-meter BESS cost (inverter, BMS, enclosure, "
            "wiring, labour) is materially higher than raw cell/pack cost — "
            "broadly consistent with BloombergNEF Battery Price Survey and NREL "
            "ATB storage cost ranges for small-scale systems. Distinct from "
            "consequences.py's repack_cost, which prices cell-level reuse labour only."
        ),
    },
    "panel_density_kwp_per_m2": {
        "value": 0.18,
        "slider_range": (0.15, 0.22),
        "unit": "kWp/m²",
        "label": "Cited estimate",
        "source": (
            "Typical commercial crystalline-silicon PV module power density is "
            "roughly 180-220 Wp/m² per current module datasheets (IEA PVPS)."
        ),
    },
    "discount_rate": {
        "value": 0.08,
        "slider_range": (0.03, 0.20),
        "unit": "fraction/yr",
        "label": "Illustrative — not sourced",
        "source": (
            "Same 8% WACC default already used as the discount-rate slider default "
            "in this page's NPV Scenario Planner — kept consistent across both tools."
        ),
    },
    "feed_in_tariff_eur": {
        "value": 0.0,
        "slider_range": (0.0, 0.20),
        "unit": "€/kWh",
        "label": "Illustrative — not sourced",
        "source": (
            "PV export remuneration is jurisdiction- and utility-specific and varies "
            "enormously (from 0 to several times the import tariff) — conservatively "
            "defaulted to 0 (self-consumption only) rather than assuming a rate."
        ),
    },
}

_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def estimate_annual_savings(
    pv_monthly_kwh: list,
    monthly_consumption_kwh: list,
    battery_kwh: float,
    tariff_high_eur: float,
    tariff_low_eur: float,
    round_trip_efficiency: float = 0.90,
    battery_cycles_per_day: float = 1.0,
    self_consumption_derating: float = 0.70,
    feed_in_tariff_eur: float = 0.0,
) -> dict:
    """
    Monthly energy-balance approximation of annual bill savings from PV
    self-consumption + battery tariff arbitrage. NOT an hourly dispatch
    simulation — see module docstring for the overestimation caveat.

    Per month: PV first offsets load directly (derated for hourly mismatch);
    any PV surplus charges the battery to further offset load; any remaining
    battery budget is used for grid tariff arbitrage (charge at
    tariff_low_eur, discharge to offset load billed at tariff_high_eur);
    any PV surplus left over after charging the battery is exported at
    feed_in_tariff_eur.

    Returns {"annual_savings_eur": float, "monthly": list[12 dicts]}.
    """
    months = []
    annual_savings = 0.0

    for i in range(12):
        pv_gen = float(pv_monthly_kwh[i])
        load = float(monthly_consumption_kwh[i])
        days = _DAYS_IN_MONTH[i]

        direct_pv = min(pv_gen, load) * self_consumption_derating
        pv_surplus = pv_gen - (direct_pv / self_consumption_derating if self_consumption_derating > 0 else 0.0)
        pv_surplus = max(pv_surplus, 0.0)
        residual_load = max(load - direct_pv, 0.0)

        battery_month_kwh = max(battery_kwh, 0.0) * days * battery_cycles_per_day

        pv_to_batt = min(pv_surplus, battery_month_kwh)
        batt_output_from_pv = pv_to_batt * round_trip_efficiency
        pv_batt_offset = min(batt_output_from_pv, residual_load)

        remaining_batt_budget = battery_month_kwh - pv_to_batt
        remaining_load = residual_load - pv_batt_offset

        arb_charge = min(
            remaining_batt_budget,
            remaining_load / round_trip_efficiency if round_trip_efficiency > 0 else 0.0,
        )
        arb_discharge = arb_charge * round_trip_efficiency

        grid_import = max(remaining_load - arb_discharge, 0.0)
        export_kwh = max(pv_surplus - pv_to_batt, 0.0)
        export_value = export_kwh * feed_in_tariff_eur

        savings = (
            (direct_pv + pv_batt_offset) * tariff_high_eur
            + (arb_discharge * tariff_high_eur - arb_charge * tariff_low_eur)
            + export_value
        )
        annual_savings += savings

        months.append({
            "month": i + 1,
            "direct_pv_kwh": direct_pv,
            "pv_via_battery_kwh": pv_batt_offset,
            "arbitrage_kwh": arb_discharge,
            "grid_import_kwh": grid_import,
            "export_kwh": export_kwh,
            "savings_eur": savings,
        })

    return {"annual_savings_eur": annual_savings, "monthly": months}


def payback_years(investment_eur: float, annual_savings_eur: float) -> "float | None":
    """None (not ZeroDivisionError) when savings are zero or negative."""
    if annual_savings_eur <= 0:
        return None
    return investment_eur / annual_savings_eur


def npv_eur(
    investment_eur: float,
    annual_savings_eur: float,
    discount_rate: float,
    horizon_years: int = 15,
) -> float:
    """Discounted-cash-flow NPV of a constant annual savings stream against
    an upfront investment. Same discounting idea as the existing (separate,
    intentionally untouched) NPV Scenario Planner in app/_pages/consequences.py,
    but as a proper named/tested function rather than an inline closure."""
    pv_of_savings = sum(
        annual_savings_eur / ((1.0 + discount_rate) ** t)
        for t in range(1, horizon_years + 1)
    )
    return pv_of_savings - investment_eur


# ---------------------------------------------------------------------------
# Sizing search
# ---------------------------------------------------------------------------

def size_deployment(
    lat: float,
    lon: float,
    tilt_deg: float,
    azimuth_compass_deg: float,
    available_area_m2: float,
    cell_kwh_per_cell: float,
    monthly_consumption_kwh: list,
    tariff_high_eur: float,
    tariff_low_eur: float,
    max_payoff_years: float,
    max_investment_eur: float,
    n_cells_range: range = range(1, 21),
    pv_yield_fn: Optional[Callable] = None,
    assumptions: Optional[dict] = None,
) -> dict:
    """
    Grid search over PV size (kWp, derived from available_area_m2 x
    panel_density_kwp_per_m2, coarse-stepped) x battery size (n_cells x
    cell_kwh_per_cell, capped by max_investment_eur so the search never
    explores obviously-infeasible counts).

    PV yield depends only on pv_kwp, not on n_cells — pv_yield_fn is called
    AT MOST ONCE PER UNIQUE pv_kwp STEP (cached locally), never once per
    (pv_kwp, n_cells) pair.

    pv_yield_fn defaults to pvgis_client.fetch_pv_yield, resolved inside the
    function body (not as a default-arg reference) so tests can inject a
    fake without needing to patch a module attribute.

    Winner = highest npv_eur() among candidates satisfying
    payback <= max_payoff_years AND investment <= max_investment_eur. If no
    candidate is feasible, the max-NPV candidate is still returned (an honest
    near-miss) with feasible=False and a constraint_note explaining why.

    Returns {"feasible": bool, "winner": dict | None, "candidates": list[dict],
    "constraint_note": str | None, "pv_errors": list[str]}.
    """
    if pv_yield_fn is None:
        from pvgis_client import fetch_pv_yield
        pv_yield_fn = fetch_pv_yield
    from pvgis_client import compass_to_pvgis_azimuth

    a = {k: v["value"] for k, v in SIZING_ASSUMPTIONS.items()}
    if assumptions:
        a.update(assumptions)

    azimuth_pvgis = compass_to_pvgis_azimuth(azimuth_compass_deg)

    max_pv_kwp = max(available_area_m2 * a["panel_density_kwp_per_m2"], 0.0)
    n_pv_steps = 6
    if max_pv_kwp <= 0:
        pv_kwp_steps = [0.0]
    else:
        pv_kwp_steps = [max_pv_kwp * i / (n_pv_steps - 1) for i in range(n_pv_steps)]

    bess_cost_per_kwh = a["bess_install_cost_eur_per_kwh"]
    if cell_kwh_per_cell > 0 and bess_cost_per_kwh > 0:
        max_affordable_cells = int(max_investment_eur / (cell_kwh_per_cell * bess_cost_per_kwh))
        capped_max = max(1, min(max(n_cells_range), max_affordable_cells + 1))
    else:
        capped_max = max(n_cells_range)
    n_cells_list = [n for n in n_cells_range if n <= capped_max] or [min(n_cells_range)]

    pv_yield_cache = {}
    pv_errors = []
    candidates = []

    for pv_kwp in pv_kwp_steps:
        key = round(pv_kwp, 3)
        if key not in pv_yield_cache:
            if pv_kwp <= 0:
                pv_yield_cache[key] = {"annual_kwh": 0.0, "monthly_kwh": [0.0] * 12}
            else:
                pv_yield_cache[key] = pv_yield_fn(
                    lat=lat, lon=lon, peakpower_kwp=pv_kwp,
                    tilt_deg=tilt_deg, azimuth_deg=azimuth_pvgis,
                )
        yield_result = pv_yield_cache[key]
        if "error" in yield_result:
            pv_errors.append(f"{pv_kwp:.2f} kWp: {yield_result['error']}")
            continue

        pv_monthly = yield_result["monthly_kwh"]

        for n_cells in n_cells_list:
            battery_kwh = n_cells * cell_kwh_per_cell
            savings = estimate_annual_savings(
                pv_monthly_kwh=pv_monthly,
                monthly_consumption_kwh=monthly_consumption_kwh,
                battery_kwh=battery_kwh,
                tariff_high_eur=tariff_high_eur,
                tariff_low_eur=tariff_low_eur,
                round_trip_efficiency=a["round_trip_efficiency"],
                battery_cycles_per_day=a["battery_cycles_per_day"],
                self_consumption_derating=a["self_consumption_derating"],
                feed_in_tariff_eur=a["feed_in_tariff_eur"],
            )
            annual_savings_eur = savings["annual_savings_eur"]
            investment_eur = (
                pv_kwp * a["pv_install_cost_eur_per_kwp"]
                + battery_kwh * a["bess_install_cost_eur_per_kwh"]
            )
            payback = payback_years(investment_eur, annual_savings_eur)
            npv = npv_eur(investment_eur, annual_savings_eur, a["discount_rate"])

            feasible = (
                payback is not None
                and payback <= max_payoff_years
                and investment_eur <= max_investment_eur
            )
            candidates.append({
                "pv_kwp": pv_kwp,
                "n_cells": n_cells,
                "battery_kwh": battery_kwh,
                "investment_eur": investment_eur,
                "annual_savings_eur": annual_savings_eur,
                "payback_years": payback,
                "npv_eur": npv,
                "feasible": feasible,
                "monthly": savings["monthly"],
            })

    if not candidates:
        return {
            "feasible": False,
            "winner": None,
            "candidates": [],
            "constraint_note": "No PV/battery size produced a usable result — PVGIS was unavailable at every size explored.",
            "pv_errors": pv_errors,
        }

    feasible_candidates = [c for c in candidates if c["feasible"]]
    if feasible_candidates:
        winner = max(feasible_candidates, key=lambda c: c["npv_eur"])
        return {
            "feasible": True,
            "winner": winner,
            "candidates": candidates,
            "constraint_note": None,
            "pv_errors": pv_errors,
        }

    winner = max(candidates, key=lambda c: c["npv_eur"])
    note = (
        f"No PV+battery size pays back within {max_payoff_years:.1f} years and "
        f"€{max_investment_eur:,.0f} — showing the closest (highest-NPV) option instead."
    )
    return {
        "feasible": False,
        "winner": winner,
        "candidates": candidates,
        "constraint_note": note,
        "pv_errors": pv_errors,
    }
