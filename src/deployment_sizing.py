"""
Solar + Storage Sizing — deployment sizing and payback/NPV for a candidate
second-life battery paired with a new PV array (Consequences page).

size_deployment() now runs on a real HOURLY dispatch simulation
(simulate_hourly_dispatch(), fed by pvgis_client.fetch_pv_yield_hourly())
rather than a monthly approximation — see simulate_hourly_dispatch()'s
docstring for the real limitations of that simulation (it's a documented
threshold heuristic, not an LP/MILP optimizer, and uses one fixed historical
reference year of weather).

estimate_annual_savings() (below) is the ORIGINAL monthly energy-balance
approximation from before the hourly engine existed. It is no longer used
by size_deployment() or the UI's primary path, but is kept — still tested,
still correct for what it is — as a simpler/faster reference implementation.
Its self_consumption_derating exists specifically to correct for the known
overestimation bias of monthly-level PV/load matching (see Luthander et al.
2015, "Photovoltaic self-consumption in buildings: A review", Applied
Energy 142) — the hourly engine doesn't need that fudge factor because it
matches PV and load hour-by-hour for real, but do not remove or repurpose
this function's derating logic; it's a legitimate approximation on its own
terms, just not the one driving the UI anymore.

This module has no Streamlit dependency (mirrors src/consequences.py /
src/pack_builder.py). PV yield data is supplied externally (normally via
src/pvgis_client.py) rather than modelled locally — see size_deployment()'s
pv_yield_fn parameter.

Since the hourly engine shipped, size_deployment() also: (1) scales each
candidate's single-reference-year hourly PV shape to match PVGIS's
multi-year climate average annual total (scale_hourly_to_multiyear_average()) —
so results reflect long-term-typical generation, not one arbitrary year's
weather, while keeping that year's real hour-to-hour shape; (2) accepts an
explicit utc_offset_override to replace the longitude-based timezone
approximation when a caller/user knows the site's real civil offset; (3)
supports an optional weekend_daily_shape distinct from the weekday shape
(build_hourly_consumption()); (4) derates the battery's hourly power cap by
ambient temperature (temperature_derate_factor(), fed by PVGIS's T2m —
already present in the same hourly response, no extra API call). A true
LP/MILP optimizer to replace the threshold-heuristic dispatch itself was
considered and explicitly declined — a different order of engineering
investment (new solver dependency, full re-architecture) for a tool whose
whole premise is an honest, cited ESTIMATE, not an optimal-control
benchmark. See docs/history.md for that scope decision.
"""

from datetime import date, timedelta
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
            "0.70 default is engineering judgment, not taken directly from that paper. "
            "Only backs the legacy estimate_annual_savings() path — the hourly engine "
            "(simulate_hourly_dispatch()) doesn't need this fudge factor at all."
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
            "tariff-arbitrage use — most such systems don't cycle more than once a day. "
            "Only backs the legacy estimate_annual_savings() path — the hourly engine "
            "uses battery_c_rate (a real power limit) instead of a monthly throughput budget."
        ),
    },
    "battery_c_rate": {
        "value": 0.5,
        "slider_range": (0.2, 1.5),
        "unit": "C (fraction of kWh/hour)",
        "label": "Cited estimate",
        "source": (
            "Typical continuous charge/discharge power rating for stationary Li-ion "
            "BESS (residential/C&I second-life systems) commonly falls in the 0.3C-1C "
            "range per NREL/IRENA storage technology characterizations; 0.5C (full "
            "charge or discharge in ~2h) is a mid-range default. Used by "
            "simulate_hourly_dispatch() as the hourly power cap on charge/discharge."
        ),
    },
    "pv_install_cost_eur_per_kwp": {
        "value": 1450.0,
        "slider_range": (900.0, 1800.0),
        "unit": "€/kWp",
        "label": "Cited estimate",
        "source": (
            "IRENA Renewable Power Generation Costs in 2024 (Jul 2025 summary) + "
            "Fraunhofer ISE Photovoltaics Report: EU residential PV installed costs "
            "roughly €1,100-1,800/kWp (e.g. Germany ~€1,400-1,600/kWp, France "
            "~€1,300-1,700/kWp), commercial-scale systems 15-25% lower per kWp."
        ),
    },
    "bess_install_cost_eur_per_kwh": {
        "value": 700.0,
        "slider_range": (400.0, 1000.0),
        "unit": "€/kWh",
        "label": "Cited estimate",
        "source": (
            "European residential battery storage averaged ~€711/kWh in H2 2025 per "
            "market reporting (a ~47% drop from H1 2023); Germany examples (5kWh "
            "add-on €2,000-5,000, 10kWh system €7,000-9,000) are consistent with this "
            "range. Distinct from BloombergNEF's ~$117-177/kWh 2025 turnkey figures, "
            "which are utility/commercial-scale averages, not small residential "
            "behind-the-meter systems — and distinct from consequences.py's "
            "repack_cost, which prices cell-level reuse labour only, not a full "
            "installed system (inverter, BMS, enclosure, wiring)."
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
    "temp_derate_floor_factor": {
        "value": 0.3,
        "slider_range": (0.1, 1.0),
        "unit": "fraction",
        "label": "Cited estimate",
        "source": (
            "Controls only the DISCHARGE floor at -20°C in "
            "temperature_derate_factor() — the -20°C/60°C discharge range and "
            "5-45°C charge fast-charge window / 0°C no-charge floor / 50°C "
            "charge ceiling are grounded in Battery University / Cadex's "
            "'BU-410: Charging at High and Low Temperatures' (a widely-cited "
            "industry reference), corroborated by a 280Ah LFP cell's "
            "derating documentation surveyed via EVreporter (2023); charging "
            "below 0°C risking lithium plating is the same mechanism cited in "
            "battery_knowledge.py's 'iec62619-undertemperature' entry, per "
            "IEC 62619:2022. The exact ramp SHAPE between the full-power band "
            "and each floor (not just the boundary temperatures) remains "
            "engineering judgment, not a datasheet curve for any one cell/BMS "
            "— and the charge-side floor (0.05) is fixed, not user-adjustable, "
            "reflecting that 'cannot charge below 0°C' is a much harder "
            "real-world constraint than a soft derate."
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
# Hourly dispatch simulation
# ---------------------------------------------------------------------------
# Precompute which of 8760 hours (a fixed non-leap reference year, see
# pvgis_client.HOURLY_REFERENCE_YEAR) falls in which calendar month, once at
# import time — every hourly function below shares this.
_HOUR_TO_MONTH = []
for _month_idx, _days in enumerate(_DAYS_IN_MONTH):
    _HOUR_TO_MONTH.extend([_month_idx] * (_days * 24))
assert len(_HOUR_TO_MONTH) == 8760


def utc_offset_hours(lon: float) -> int:
    """
    Longitude-based solar-time approximation of a site's UTC offset
    (15 degrees of longitude per hour), clamped to [-12, 14].

    This is NOT the site's actual civil timezone/DST — real timezone
    boundaries and daylight-saving rules don't follow longitude exactly,
    so this can be off by roughly an hour near timezone boundaries or
    during DST periods. It exists because PVGIS's seriescalc hourly
    timestamps are verified UTC (see pvgis_client's module docstring),
    while consumption/tariff hour-of-day patterns are inherently local
    wall-clock — some correction is far better than none (leaving PV
    generation curves several hours out of alignment with load curves in
    extreme-longitude cases), even though it isn't a real timezone lookup.
    """
    offset = round(lon / 15.0)
    return max(-12, min(14, offset))


def shift_to_local_hours(hourly_values: list, utc_offset: int) -> list:
    """
    Rolls an 8760-length UTC-indexed hourly array to approximate local-time
    indexing (index h of the result ~= local hour h). Wraps around the
    reference year's boundary (a negligible, few-hour edge effect out of
    8760 hours) rather than needing real adjacent-year data.
    """
    n = len(hourly_values)
    if n == 0:
        return []
    offset = utc_offset % n
    return hourly_values[-offset:] + hourly_values[:-offset] if offset else list(hourly_values)


def scale_hourly_to_multiyear_average(
    hourly_kwh: list,
    single_year_annual_kwh: float,
    multi_year_annual_kwh: float,
) -> list:
    """
    Scales an hourly PV series (fetch_pv_yield_hourly()'s one fixed
    historical reference year) so its annual total matches PVGIS's
    multi-year climate-average total (fetch_pv_yield()'s PVcalc E_y) —
    while preserving that reference year's real hour-to-hour/seasonal
    shape. Results then reflect long-term-typical generation rather than
    one arbitrary year's weather (see module docstring). Guards against a
    zero/near-zero single_year_annual_kwh (degenerate case, e.g.
    peakpower=0) by returning the series unscaled rather than dividing by zero.
    """
    if single_year_annual_kwh <= 0:
        return list(hourly_kwh)
    scale = multi_year_annual_kwh / single_year_annual_kwh
    return [v * scale for v in hourly_kwh]


def temperature_derate_factor(
    temp_c: float,
    mode: str = "discharge",
    min_full_power_c: "float | None" = None,
    max_full_power_c: float = 45.0,
    floor_temp_c: "float | None" = None,
    high_floor_temp_c: "float | None" = None,
    floor_factor: "float | None" = None,
) -> float:
    """
    Piecewise-linear derating of battery charge/discharge power vs ambient
    temperature. mode="discharge" (default) or mode="charge" select real,
    cited reference bounds for any of min_full_power_c/floor_temp_c/
    high_floor_temp_c/floor_factor left as None (explicit values always
    override the mode default):

    - "discharge": full power down to 0°C, linearly reduced to floor_factor
      (default 0.3, SIZING_ASSUMPTIONS["temp_derate_floor_factor"]) at
      -20°C, and toward the same floor above 60°C. Li-ion cells' permissible
      discharge range is commonly cited as -20°C to 60°C.
    - "charge": full power in a 5-45°C fast-charge window, dropping steeply
      to a near-zero floor (0.05, not user-adjustable) at 0°C ("no charge
      permitted below freezing" — most Li-ion chemistries risk lithium
      plating charging below 0°C, see IEC 62619:2022 and this app's own
      battery_knowledge.py "iec62619-undertemperature" entry for the
      mechanism), and toward the same floor above 50°C (charging above
      50°C is often chargers'-prohibited). Charge gets a much steeper/lower
      floor than discharge, a real qualitative difference, not the same
      curve applied symmetrically to both directions.

    Source for the specific boundary temperatures (5°C/45°C fast-charge
    window, 0°C no-charge floor, 50°C charge ceiling, -20°C/60°C discharge
    range): Battery University / Cadex, "BU-410: Charging at High and Low
    Temperatures" — a widely-cited industry reference — corroborated by a
    280Ah LFP cell's derating documentation (surveyed via EVreporter, 2023).
    The exact SHAPE of each ramp BETWEEN the full-power band and each floor
    (not just the boundary temperatures themselves) is still engineering
    judgment, not a specific datasheet curve for any one cell/BMS.

    Full power (1.0) within [min_full_power_c, max_full_power_c], linearly
    reduced outside that band down to floor_factor at floor_temp_c (cold) /
    high_floor_temp_c (hot), clamped at floor_factor beyond either.
    """
    if mode == "charge":
        if min_full_power_c is None:
            min_full_power_c = 5.0
        if floor_temp_c is None:
            floor_temp_c = 0.0
        if high_floor_temp_c is None:
            high_floor_temp_c = 50.0
        if floor_factor is None:
            floor_factor = 0.05
    else:
        if min_full_power_c is None:
            min_full_power_c = 0.0
        if floor_temp_c is None:
            floor_temp_c = -20.0
        if high_floor_temp_c is None:
            high_floor_temp_c = 60.0
        if floor_factor is None:
            floor_factor = 0.3

    if min_full_power_c <= temp_c <= max_full_power_c:
        return 1.0
    if temp_c < min_full_power_c:
        if temp_c <= floor_temp_c:
            return floor_factor
        span = min_full_power_c - floor_temp_c
        frac = (temp_c - floor_temp_c) / span if span > 0 else 1.0
        return floor_factor + frac * (1.0 - floor_factor)
    if temp_c >= high_floor_temp_c:
        return floor_factor
    span = high_floor_temp_c - max_full_power_c
    frac = (high_floor_temp_c - temp_c) / span if span > 0 else 1.0
    return floor_factor + frac * (1.0 - floor_factor)


def night_window_hours(start_hour: int, end_hour: int) -> set:
    """
    Wrap-around-aware set of hour-of-day integers (0-23) for a "night"
    window, e.g. night_window_hours(23, 7) -> {23, 0, 1, 2, 3, 4, 5, 6}
    (from 23:00 up to, but not including, 07:00).
    """
    start_hour = int(start_hour) % 24
    end_hour = int(end_hour) % 24
    if start_hour == end_hour:
        return set()
    if start_hour < end_hour:
        return set(range(start_hour, end_hour))
    return set(range(start_hour, 24)) | set(range(0, end_hour))


def build_typical_year_pv_shape(ghi_wm2_hourly: list, monthly_pv_kwh: list, n_hours: int = 8760) -> list:
    """
    Redistributes each month's REAL PVcalc-derived kWh total
    (monthly_pv_kwh, from pvgis_client.fetch_pv_yield()) across that
    month's hours proportional to TMY global horizontal irradiance
    (ghi_wm2_hourly, from pvgis_client.fetch_tmy_ghi()) — a shape-only
    proxy for "how much sun fell this specific hour relative to other
    hours in the same month", without this app reimplementing PV
    performance modelling (irradiance transposition to the panel's actual
    tilt/azimuth, temperature coefficients, inverter efficiency curves —
    all things PVGIS's own PVcalc/seriescalc already do server-side for
    the MAGNITUDE this function distributes). Result sums to exactly
    monthly_pv_kwh's total within each month.

    If a month's total GHI is zero (a degenerate edge case), that month's
    hours split evenly instead of dividing by zero.
    """
    hourly = [0.0] * n_hours
    hour_idx = 0
    for month_idx, days in enumerate(_DAYS_IN_MONTH):
        month_hours = days * 24
        month_ghi = ghi_wm2_hourly[hour_idx:hour_idx + month_hours]
        month_total_ghi = sum(month_ghi)
        month_kwh = float(monthly_pv_kwh[month_idx])
        if month_total_ghi > 0:
            for i, g in enumerate(month_ghi):
                hourly[hour_idx + i] = month_kwh * (g / month_total_ghi)
        else:
            even_share = month_kwh / month_hours if month_hours else 0.0
            for i in range(month_hours):
                hourly[hour_idx + i] = even_share
        hour_idx += month_hours
    if hour_idx != n_hours:
        raise ValueError(f"built {hour_idx} hourly values, expected {n_hours}")
    return hourly


def build_tariff_hour_arrays(
    tariff_model: str,
    tariff_high_eur: float,
    tariff_low_eur: float,
    low_tariff_hours: Optional[set] = None,
    n_hours: int = 8760,
) -> tuple:
    """
    Build per-hour (price, is_low_tariff) arrays for the hourly dispatch sim.

    tariff_model: "single_rate" | "day_night" | "custom".
    - "single_rate": every hour is_low=False, priced tariff_high_eur — no
      time-of-use distinction at all. This is exactly why
      simulate_hourly_dispatch()'s discharge heuristic ("only discharge
      when NOT a low-tariff hour") degrades cleanly to plain
      self-consumption for a flat tariff: every hour qualifies.
    - "day_night" / "custom": hours in low_tariff_hours (a set of ints
      0-23) are priced tariff_low_eur, all others tariff_high_eur. The
      SAME 24-hour pattern repeats on every calendar day of the year — no
      weekday/weekend or seasonal daily-pattern variation. That's a real
      simplification, stated here rather than left implicit.

    Returns (price_eur_per_kwh: list[n_hours], is_low_tariff: list[n_hours bool]).
    """
    if tariff_model == "single_rate":
        return [tariff_high_eur] * n_hours, [False] * n_hours

    low_hours = low_tariff_hours or set()
    price = []
    is_low = []
    for h in range(n_hours):
        hour_of_day = h % 24
        low = hour_of_day in low_hours
        price.append(tariff_low_eur if low else tariff_high_eur)
        is_low.append(low)
    return price, is_low


def build_hourly_consumption(
    monthly_consumption_kwh: list,
    daily_shape: list,
    weekend_daily_shape: Optional[list] = None,
    reference_year: Optional[int] = None,
    n_hours: int = 8760,
) -> list:
    """
    Spreads each month's total consumption evenly across its calendar days,
    then each day's total across 24 hours via daily_shape (a length-24
    weight list summing to 1.0).

    If weekend_daily_shape is given, real Saturdays/Sundays of
    reference_year (default pvgis_client.HOURLY_REFERENCE_YEAR — the same
    calendar the hourly PV series uses, so weekday/weekend line up hour for
    hour with size_deployment()'s PV data) use it instead of daily_shape.
    Each day's TOTAL is always that day's even share of its month
    (monthly_kwh/days) regardless of which shape is used — only the
    within-day hourly distribution differs — so switching shapes never
    changes the monthly or annual total.

    Without weekend_daily_shape, the same daily_shape applies to every day
    of every month — no weekday/weekend variation — still a real
    improvement on treating every hour of a month as identical (the
    monthly estimate_annual_savings()'s implicit assumption).
    """
    if reference_year is None:
        from pvgis_client import HOURLY_REFERENCE_YEAR
        reference_year = HOURLY_REFERENCE_YEAR

    hourly = []
    current_date = date(reference_year, 1, 1)
    for month_idx, days in enumerate(_DAYS_IN_MONTH):
        monthly_kwh = float(monthly_consumption_kwh[month_idx])
        daily_kwh = monthly_kwh / days if days else 0.0
        for _day in range(days):
            is_weekend = current_date.weekday() >= 5
            shape = weekend_daily_shape if (is_weekend and weekend_daily_shape) else daily_shape
            for hour_weight in shape:
                hourly.append(daily_kwh * hour_weight)
            current_date += timedelta(days=1)
    if len(hourly) != n_hours:
        raise ValueError(f"built {len(hourly)} hourly values, expected {n_hours}")
    return hourly


def simulate_hourly_dispatch(
    pv_hourly_kwh: list,
    load_hourly_kwh: list,
    tariff_hourly_eur: list,
    is_low_tariff_hourly: list,
    battery_kwh: float,
    battery_c_rate: float = 0.5,
    round_trip_efficiency: float = 0.90,
    feed_in_tariff_eur: float = 0.0,
    temp_hourly_c: Optional[list] = None,
    temp_derate_floor_factor: float = 0.3,
) -> dict:
    """
    Real hour-by-hour dispatch simulation over one calendar year (8760
    hours) — replaces estimate_annual_savings()'s monthly approximation as
    size_deployment()'s primary engine.

    Real limitations, stated plainly (matching this app's citation/honesty
    convention — see SIZING_ASSUMPTIONS and src/consequences.py's
    ASSUMPTIONS for the same standard applied elsewhere):

    1. This is a threshold HEURISTIC, not an LP/MILP optimizer. The battery
       discharges to cover load in ANY hour that isn't flagged low-tariff,
       with no forecasting — it can fully deplete early in a long
       non-low-tariff stretch and then sit empty for the rest of it,
       understating value relative to a forecast-aware dispatcher. (A true
       LP/MILP replacement was considered and explicitly declined — see
       module docstring.)
    2. pv_hourly_kwh is normally magnitude-scaled to a multi-year climate
       average by the caller (size_deployment() via
       scale_hourly_to_multiyear_average()) — but its intra-year hour-to-hour
       SHAPE (which specific days are sunny/cloudy) is still just one fixed
       historical reference year's real weather pattern, not averaged.
    3. The tariff pattern (tariff_hourly_eur/is_low_tariff_hourly) is
       whatever build_tariff_hour_arrays() built — typically the identical
       24-hour pattern on every calendar day, no weekday/weekend/holiday
       tariff variation (unlike load_hourly_kwh, which CAN carry a real
       weekday/weekend distinction via build_hourly_consumption()'s
       weekend_daily_shape).
    4. Power cap is battery_c_rate x battery_kwh, optionally derated per
       hour by ambient temperature if temp_hourly_c is supplied — via
       temperature_derate_factor(), called separately for charge (PV-charge
       + arbitrage-charge, a steep near-zero floor below 0°C) and discharge
       (a gentler floor down to -20°C), since real Li-ion charge/discharge
       temperature limits genuinely differ, not the same curve both ways.
       Pass temp_hourly_c=None (default) to keep a flat cap. No
       SOC-dependent derating either way.
    5. Battery starts empty (SOC=0) at hour 0 of the reference year — a
       cold-start bias confined to roughly the first day out of 365,
       negligible in the annual total.
    6. pv_hourly_kwh must already be shifted to local time (see
       shift_to_local_hours()) — this function has no timezone awareness
       of its own.

    Per-hour accounting: PV first offsets load directly; any PV surplus
    charges the battery (bounded by remaining capacity AND the power cap);
    any surplus left after that is exported at feed_in_tariff_eur; if load
    still exceeds PV and this hour is not low-tariff, the battery
    discharges to cover it (round_trip_efficiency applied at discharge
    only — energy is stored 1:1 on charge); if this hour IS low-tariff and
    the battery has headroom, it charges from the grid for arbitrage,
    SHARING the same hourly power cap with any PV-charging that already
    happened this hour (charge and arbitrage-charge never compete with
    discharge in the same hour, since PV-surplus-charging only occurs when
    residual_load is 0, which is exactly when discharge cannot fire).

    savings_h derivation: counterfactual (no PV/battery) cost this hour =
    load_h * price_h. Actual cost = (grid_import_h + arb_charge_h) * price_h
    - export_h * feed_in_tariff. Since load_h - grid_import_h =
    direct_pv_h + battery_output_h, savings collapses to:
        savings_h = (direct_pv_h + battery_output_h) * price_h
                    - arb_charge_h * price_h + export_h * feed_in_tariff_eur

    Returns {"annual_savings_eur": float, "monthly": list[12 dicts]} — same
    dict shape as estimate_annual_savings()'s "monthly" output, except
    "pv_via_battery_kwh"/"arbitrage_kwh" (which required knowing WHICH
    charge source a discharged kWh came from) collapse into one
    "battery_output_kwh" (undifferentiated discharge) — tracking that
    split would require a second SOC ledger inventing a provenance
    distinction the physics doesn't actually support. "arb_charge_kwh" is
    kept separately because it's a real, directly-measurable quantity on
    the CHARGE side (how much was bought from the grid at the low tariff
    to charge the battery), not an invented one.
    """
    n_hours = len(pv_hourly_kwh)
    if not (len(load_hourly_kwh) == len(tariff_hourly_eur) == len(is_low_tariff_hourly) == n_hours):
        raise ValueError("pv_hourly_kwh/load_hourly_kwh/tariff_hourly_eur/is_low_tariff_hourly must be equal length")

    power_cap = max(battery_kwh, 0.0) * max(battery_c_rate, 0.0)
    soc = 0.0

    monthly = [
        {"month": m + 1, "direct_pv_kwh": 0.0, "battery_output_kwh": 0.0,
         "arb_charge_kwh": 0.0, "grid_import_kwh": 0.0, "export_kwh": 0.0, "savings_eur": 0.0}
        for m in range(12)
    ]
    annual_savings = 0.0

    for h in range(n_hours):
        pv = pv_hourly_kwh[h]
        load = load_hourly_kwh[h]
        price = tariff_hourly_eur[h]
        is_low = is_low_tariff_hourly[h]
        charge_used = 0.0

        if temp_hourly_c is not None:
            t = temp_hourly_c[h]
            hour_charge_cap = power_cap * temperature_derate_factor(t, mode="charge", floor_factor=temp_derate_floor_factor)
            hour_discharge_cap = power_cap * temperature_derate_factor(t, mode="discharge", floor_factor=temp_derate_floor_factor)
        else:
            hour_charge_cap = hour_discharge_cap = power_cap

        direct_pv = min(pv, load)
        residual_load = load - direct_pv
        pv_surplus = pv - direct_pv

        room = battery_kwh - soc
        to_charge = max(min(pv_surplus, room, hour_charge_cap - charge_used), 0.0)
        soc += to_charge
        charge_used += to_charge
        export_kwh = max(pv_surplus - to_charge, 0.0)

        battery_output = 0.0
        if residual_load > 0 and not is_low:
            draw = max(min(soc, hour_discharge_cap, residual_load / round_trip_efficiency if round_trip_efficiency > 0 else 0.0), 0.0)
            soc -= draw
            battery_output = draw * round_trip_efficiency
            residual_load -= battery_output

        arb_charge = 0.0
        if is_low:
            room = battery_kwh - soc
            arb_charge = max(min(room, hour_charge_cap - charge_used), 0.0)
            soc += arb_charge

        grid_import = max(residual_load, 0.0)
        savings_h = (
            (direct_pv + battery_output) * price
            - arb_charge * price
            + export_kwh * feed_in_tariff_eur
        )
        annual_savings += savings_h

        soc = min(max(soc, 0.0), battery_kwh)

        m = monthly[_HOUR_TO_MONTH[h]]
        m["direct_pv_kwh"] += direct_pv
        m["battery_output_kwh"] += battery_output
        m["arb_charge_kwh"] += arb_charge
        m["grid_import_kwh"] += grid_import
        m["export_kwh"] += export_kwh
        m["savings_eur"] += savings_h

    return {"annual_savings_eur": annual_savings, "monthly": monthly}


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
    daily_load_shape: list,
    tariff_model: str,
    tariff_high_eur: float,
    tariff_low_eur: float,
    max_payoff_years: float,
    max_investment_eur: float,
    low_tariff_hours: Optional[set] = None,
    weekend_daily_shape: Optional[list] = None,
    utc_offset_override: Optional[int] = None,
    n_cells_range: range = range(1, 21),
    n_pv_steps: int = 6,
    n_cells_coarse_steps: int = 6,
    pv_yield_fn: Optional[Callable] = None,
    pv_yield_annual_fn: Optional[Callable] = None,
    assumptions: Optional[dict] = None,
    reference_year: Optional[int] = None,
    load_hourly_kwh_override: Optional[list] = None,
    pv_weather_source: str = "typical_year",
    tmy_ghi_fn: Optional[Callable] = None,
) -> dict:
    """
    Grid search over PV size (kWp, derived from available_area_m2 x
    panel_density_kwp_per_m2) x battery size (n_cells x cell_kwh_per_cell,
    capped by max_investment_eur), using the real hourly dispatch engine
    (simulate_hourly_dispatch()) for every candidate.

    tariff_model / tariff_high_eur / tariff_low_eur / low_tariff_hours feed
    build_tariff_hour_arrays(); monthly_consumption_kwh / daily_load_shape /
    weekend_daily_shape feed build_hourly_consumption() — both built once,
    up front, reused across every candidate.

    load_hourly_kwh_override, if given (a real length-8760 hourly/smart-meter
    consumption series), is used AS-IS instead of build_hourly_consumption() —
    monthly_consumption_kwh/daily_load_shape/weekend_daily_shape are then
    ignored (still required arguments for a stable call signature, but their
    values don't matter in this path). This is the most accurate consumption
    input this tool supports — real measured data instead of any preset shape.

    utc_offset_override, if given, replaces utc_offset_hours(lon)'s
    longitude-based approximation — pass a real known civil UTC offset for
    the site when available.

    pv_weather_source="typical_year" (the DEFAULT — chosen because it's both
    more accurate and lighter per unique pv_kwp step than single_year, see
    below) uses TMY global-horizontal-irradiance data (tmy_ghi_fn, defaults
    to pvgis_client.fetch_tmy_ghi — fetched ONCE for the whole site, not per
    pv_kwp, since TMY is a weather dataset independent of PV system
    geometry) to redistribute each pv_kwp's real PVcalc monthly kWh totals
    (pv_yield_annual_fn, defaults to pvgis_client.fetch_pv_yield) across
    hours via build_typical_year_pv_shape() — a more representative shape
    AND correct magnitude in one step. If TMY or the monthly PVcalc fetch
    fails, this mode falls back to pv_weather_source="single_year" below
    (logged in "scaling_notes", never fatal).

    pv_weather_source="single_year" (the pre-TMY approach, still fully
    supported and used as the above mode's fallback): pv_yield_fn (defaults
    to pvgis_client.fetch_pv_yield_hourly, resolved inside the function body
    so tests can inject a fake without patching a module attribute) is
    called AT MOST ONCE PER UNIQUE pv_kwp step. Its single-reference-year
    result is then scaled to PVGIS's multi-year climate average via
    pv_yield_annual_fn and scale_hourly_to_multiyear_average() — if that
    second call fails, the candidate still proceeds on the unscaled
    single-year data (logged in "scaling_notes", never fatal). The result
    (PV + ambient temperature, both shifted to local time via
    shift_to_local_hours()) is cached and reused across every n_cells
    candidate at that pv_kwp.

    Either way, PV yield depends only on pv_kwp, not on n_cells.

    Sizing precision: a coarse pass (n_pv_steps x n_cells_coarse_steps,
    <=36 candidates) finds an approximate winner, then a refine pass fixes
    pv_kwp at the coarse winner's value (zero additional PVGIS calls — reuses
    the cached hourly array) and sweeps every integer n_cells within one
    coarse-step's width of the coarse winner (~9 more candidates) — restores
    integer-level battery-count precision without a full fine-grained sweep
    over every (pv_kwp, n_cells) pair, which would be far more expensive
    now that each candidate runs a real 8760-hour simulation. Total
    candidates actually evaluated is hard-capped (MAX_CANDIDATES) so a
    future parameter tweak can't silently blow up runtime.

    Winner = highest npv_eur() among candidates satisfying
    payback <= max_payoff_years AND investment <= max_investment_eur. If no
    candidate is feasible, the max-NPV candidate is still returned (an honest
    near-miss) with feasible=False and a constraint_note explaining why.

    Returns {"feasible": bool, "winner": dict | None, "candidates": list[dict],
    "constraint_note": str | None, "pv_errors": list[str], "scaling_notes": list[str]}.
    """
    if pv_yield_fn is None:
        from pvgis_client import fetch_pv_yield_hourly
        pv_yield_fn = fetch_pv_yield_hourly
    if pv_yield_annual_fn is None:
        from pvgis_client import fetch_pv_yield
        pv_yield_annual_fn = fetch_pv_yield
    if tmy_ghi_fn is None:
        from pvgis_client import fetch_tmy_ghi
        tmy_ghi_fn = fetch_tmy_ghi
    from pvgis_client import compass_to_pvgis_azimuth, HOURLY_REFERENCE_YEAR
    if reference_year is None:
        reference_year = HOURLY_REFERENCE_YEAR

    a = {k: v["value"] for k, v in SIZING_ASSUMPTIONS.items()}
    if assumptions:
        a.update(assumptions)

    azimuth_pvgis = compass_to_pvgis_azimuth(azimuth_compass_deg)
    offset = utc_offset_override if utc_offset_override is not None else utc_offset_hours(lon)

    tariff_hourly, is_low_hourly = build_tariff_hour_arrays(
        tariff_model, tariff_high_eur, tariff_low_eur, low_tariff_hours,
    )
    if load_hourly_kwh_override is not None:
        load_hourly = list(load_hourly_kwh_override)
        if len(load_hourly) != 8760:
            raise ValueError(f"load_hourly_kwh_override must have 8760 values, got {len(load_hourly)}")
    else:
        load_hourly = build_hourly_consumption(
            monthly_consumption_kwh, daily_load_shape, weekend_daily_shape, reference_year,
        )

    max_pv_kwp = max(available_area_m2 * a["panel_density_kwp_per_m2"], 0.0)
    if max_pv_kwp <= 0:
        pv_kwp_steps = [0.0]
    else:
        pv_kwp_steps = [max_pv_kwp * i / (n_pv_steps - 1) for i in range(n_pv_steps)]

    min_n = min(n_cells_range)
    bess_cost_per_kwh = a["bess_install_cost_eur_per_kwh"]
    if cell_kwh_per_cell > 0 and bess_cost_per_kwh > 0:
        max_affordable_cells = int(max_investment_eur / (cell_kwh_per_cell * bess_cost_per_kwh))
        capped_max = max(1, min(max(n_cells_range), max_affordable_cells + 1))
    else:
        capped_max = max(n_cells_range)

    if capped_max <= min_n:
        n_cells_coarse = [min_n]
    else:
        n_cells_coarse = sorted(set(
            round(min_n + (capped_max - min_n) * i / (n_cells_coarse_steps - 1))
            for i in range(n_cells_coarse_steps)
        ))

    pv_hourly_cache = {}
    pv_errors = []
    scaling_notes = []
    _tmy_cache = {}

    def _get_tmy():
        if "result" not in _tmy_cache:
            _tmy_cache["result"] = tmy_ghi_fn(lat=lat, lon=lon)
        return _tmy_cache["result"]

    def _pv_hourly_for(pv_kwp):
        key = round(pv_kwp, 3)
        if key in pv_hourly_cache:
            return pv_hourly_cache[key]
        if pv_kwp <= 0:
            entry = {"pv": [0.0] * 8760, "temp": None}
            pv_hourly_cache[key] = entry
            return entry

        if pv_weather_source == "typical_year":
            tmy = _get_tmy()
            annual_result = pv_yield_annual_fn(
                lat=lat, lon=lon, peakpower_kwp=pv_kwp, tilt_deg=tilt_deg, azimuth_deg=azimuth_pvgis,
            )
            if tmy and "error" not in tmy and annual_result and "error" not in annual_result:
                shape = build_typical_year_pv_shape(tmy["ghi_wm2"], annual_result["monthly_kwh"])
                entry = {
                    "pv": shift_to_local_hours(shape, offset),
                    "temp": shift_to_local_hours(tmy["temp_c"], offset),
                }
                pv_hourly_cache[key] = entry
                return entry
            reason = (tmy or {}).get("error") or (annual_result or {}).get("error") or "no data"
            scaling_notes.append(
                f"{pv_kwp:.2f} kWp: typical-year (TMY) data unavailable ({reason}) — "
                "falling back to single reference year."
            )
            # falls through to the single-year path below

        result = pv_yield_fn(
            lat=lat, lon=lon, peakpower_kwp=pv_kwp,
            tilt_deg=tilt_deg, azimuth_deg=azimuth_pvgis,
            year=reference_year,
        )
        if "error" in result:
            pv_errors.append(f"{pv_kwp:.2f} kWp: {result['error']}")
            pv_hourly_cache[key] = None
            return None

        pv_kwh = result["pv_kwh"]
        temp_c = result.get("temp_c")
        single_year_annual = sum(pv_kwh)

        annual_result = pv_yield_annual_fn(
            lat=lat, lon=lon, peakpower_kwp=pv_kwp, tilt_deg=tilt_deg, azimuth_deg=azimuth_pvgis,
        )
        if annual_result and "error" not in annual_result:
            pv_kwh = scale_hourly_to_multiyear_average(pv_kwh, single_year_annual, annual_result["annual_kwh"])
        else:
            scaling_notes.append(
                f"{pv_kwp:.2f} kWp: multi-year averaging unavailable "
                f"({(annual_result or {}).get('error', 'no data')}) — using single reference year unscaled."
            )

        entry = {
            "pv": shift_to_local_hours(pv_kwh, offset),
            "temp": shift_to_local_hours(temp_c, offset) if temp_c is not None else None,
        }
        pv_hourly_cache[key] = entry
        return entry

    def _evaluate(pv_kwp, n_cells):
        pv_entry = _pv_hourly_for(pv_kwp)
        if pv_entry is None:
            return None
        battery_kwh = n_cells * cell_kwh_per_cell
        sim = simulate_hourly_dispatch(
            pv_hourly_kwh=pv_entry["pv"],
            load_hourly_kwh=load_hourly,
            tariff_hourly_eur=tariff_hourly,
            is_low_tariff_hourly=is_low_hourly,
            battery_kwh=battery_kwh,
            battery_c_rate=a["battery_c_rate"],
            round_trip_efficiency=a["round_trip_efficiency"],
            feed_in_tariff_eur=a["feed_in_tariff_eur"],
            temp_hourly_c=pv_entry["temp"],
            temp_derate_floor_factor=a["temp_derate_floor_factor"],
        )
        annual_savings_eur = sim["annual_savings_eur"]
        investment_eur = pv_kwp * a["pv_install_cost_eur_per_kwp"] + battery_kwh * a["bess_install_cost_eur_per_kwh"]
        payback = payback_years(investment_eur, annual_savings_eur)
        npv = npv_eur(investment_eur, annual_savings_eur, a["discount_rate"])
        feasible = payback is not None and payback <= max_payoff_years and investment_eur <= max_investment_eur
        return {
            "pv_kwp": pv_kwp, "n_cells": n_cells, "battery_kwh": battery_kwh,
            "investment_eur": investment_eur, "annual_savings_eur": annual_savings_eur,
            "payback_years": payback, "npv_eur": npv, "feasible": feasible,
            "monthly": sim["monthly"],
        }

    MAX_CANDIDATES = 60
    seen = set()
    candidates = []

    def _add_candidate(pv_kwp, n_cells):
        key = (round(pv_kwp, 3), n_cells)
        if key in seen or len(seen) >= MAX_CANDIDATES:
            return
        seen.add(key)
        c = _evaluate(pv_kwp, n_cells)
        if c is not None:
            candidates.append(c)

    # Coarse pass
    for pv_kwp in pv_kwp_steps:
        for n_cells in n_cells_coarse:
            _add_candidate(pv_kwp, n_cells)

    # Refine pass: fix pv_kwp at the coarse winner (reuses the cached hourly
    # array, zero new PVGIS calls), sweep integer n_cells near it.
    if candidates:
        coarse_winner = max(candidates, key=lambda c: c["npv_eur"])
        step_width = max(1, (capped_max - min_n) // max(1, n_cells_coarse_steps - 1))
        refine_lo = max(min_n, coarse_winner["n_cells"] - step_width)
        refine_hi = min(capped_max, coarse_winner["n_cells"] + step_width)
        for n_cells in range(refine_lo, refine_hi + 1):
            _add_candidate(coarse_winner["pv_kwp"], n_cells)

    if not candidates:
        return {
            "feasible": False,
            "winner": None,
            "candidates": [],
            "constraint_note": "No PV/battery size produced a usable result — PVGIS was unavailable at every size explored.",
            "pv_errors": pv_errors,
            "scaling_notes": scaling_notes,
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
            "scaling_notes": scaling_notes,
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
        "scaling_notes": scaling_notes,
    }
