"""
Health-aware arbitrage dispatch — the Lifecycle Intelligence wedge.

The battery-storage software cohort this platform compares against
(Capture Energy, Solship, Deepgrid — see the competitive comparison in
README.md's "Lifecycle Intelligence layer" section) optimizes dispatch
against market prices while ASSUMING the battery is healthy and fully
dispatchable. This module is the counterpoint: it prices
the SAME arbitrage opportunity but constrains the schedule with this
platform's own leave-cell-out-validated health signals, so a battery that
is power-limited (low sop_pct), near end-of-life (reliable low RUL), or
deeply faded (low SOH) is not bid as if it were new.

Health constraints implemented (all additive, all optional):
  - SoP-limited power cap: the hourly charge/discharge power cap is scaled
    by sop_pct/100 (State-of-Power, the resistance-derived rate-capability
    proxy from batlab.features.engineering). A cell at 50% SoP can only
    deliver half its nominal C-rate power — the exact failure mode
    Solship-style "2× revenue" claims ignore.
  - RUL/SOH-aware operating band: cells with a RELIABLE RUL below
    RUL_CAUTION_CYCLES, or SOH below SOH_CAUTION_PCT, get a narrowed
    [min_soc, max_soc] window (reduced depth-of-discharge per cycle).
    The mechanism is the same one second-life literature cites for
    stretching a fading battery's remaining life: shallower DoD per cycle
    reduces stress — see src/consequences.py's SECOND_LIFE_APPS rationale
    and the partial-cycle/EFC engine (batlab.features.partial_cycles).

Honest scope, matching this project's explicit dispatch-scope decision
(see src/deployment_sizing.py's module docstring):
  1. This is a threshold HEURISTIC, not an LP/MILP optimizer — the same
     explicit scope call the solar+storage hourly engine already made.
     Charge when price is below the window's low percentile, discharge when
     above the high percentile; no forecast horizon, no lookahead.
  2. RUL is cycles-to-80%-SOH (the model's validated eol threshold); using
     it to narrow the operating band is an engineering rule, NOT a
     validated "optimal degradation-aware control" result.
  3. SoP is a rate-capability PROXY (∝ 1/R), not a measured power test.
  4. Revenue assumes the market clears at the input prices (no price impact
     from this battery's own dispatch).
  5. The schedule assumes the battery is free to cycle whenever the tariff
     says so — no owner load, no PV, no minimum-reserve obligation
     (managed_charging.py handles the EV session case; grid_services.py
     accounts for the arbitrage-vs-ancillary tradeoff).

EFC accounting: the resulting SOC trajectory is run through the platform's
own rainflow engine (batlab.features.partial_cycles.rainflow_counting) so
every schedule reports the Equivalent Full Cycles it actually delivered —
the same stress metric the fleet aggregation layer prices.
"""

from __future__ import annotations

from typing import Optional

from batlab.features.partial_cycles import rainflow_counting

# A reliable RUL below this (cycles to 80% SOH) marks a cell as
# end-of-life-sensitive: dispatch narrows its operating band to reduce
# per-cycle stress. Engineering rule, not a validated optimum (see docstring).
RUL_CAUTION_CYCLES = 200
# SOH below this triggers the same narrowed band even without a reliable RUL.
SOH_CAUTION_PCT = 80.0
# Operating-band defaults for a healthy cell.
DEFAULT_MIN_SOC_PCT = 10.0
DEFAULT_MAX_SOC_PCT = 95.0
# Band used when the RUL/SOH caution triggers (reduced depth-of-discharge).
CAUTION_MIN_SOC_PCT = 40.0
CAUTION_MAX_SOC_PCT = 85.0
# Floor on the SoP power factor: even a severely power-limited cell keeps a
# small dispatch capability (0.3 = 30% of nominal C-rate power).
SOP_POWER_FACTOR_FLOOR = 0.3


def health_constrained_band(
    soh_pct: float,
    sop_pct: "float | None" = None,
    rul_cycles: "float | None" = None,
    rul_reliable: bool = False,
) -> dict:
    """Operating constraints a dispatcher must respect for one cell/battery:

      power_cap_factor: fraction of the nominal C-rate power the cell can
                        actually deliver (sop_pct/100, floored at
                        SOP_POWER_FACTOR_FLOOR; 1.0 when sop_pct unknown).
      min_soc_pct / max_soc_pct: allowed SOC operating band.
      caution: bool — whether the RUL/SOH caution triggered.
      reasons: human-readable list of which constraint fired.

    All inputs optional/additive: a healthy cell with no health signals
    supplied gets the default band and full power — so existing callers
    that pass nothing keep today's behavior, exactly like
    consequences.application_fit()'s optional sop_pct pattern."""
    reasons = []
    power_cap_factor = 1.0
    if sop_pct is not None and sop_pct > 0:
        power_cap_factor = max(SOP_POWER_FACTOR_FLOOR, min(1.0, sop_pct / 100.0))
        if power_cap_factor < 1.0:
            reasons.append(
                f"SoP {sop_pct:.0f}% caps dispatch power at {power_cap_factor * 100:.0f}% "
                f"of nominal C-rate (SoP is a rate-capability proxy, not a measured power test)."
            )

    caution = False
    if rul_reliable and rul_cycles is not None and rul_cycles < RUL_CAUTION_CYCLES:
        caution = True
        reasons.append(
            f"Reliable RUL {rul_cycles:.0f} cycles < {RUL_CAUTION_CYCLES} — "
            f"narrowing the SOC band to reduce per-cycle depth-of-discharge."
        )
    elif not rul_reliable and soh_pct < SOH_CAUTION_PCT:
        caution = True
        reasons.append(
            f"SOH {soh_pct:.1f}% < {SOH_CAUTION_PCT:.0f}% (no reliable RUL) — "
            f"narrowing the SOC band to reduce per-cycle depth-of-discharge."
        )

    min_soc = CAUTION_MIN_SOC_PCT if caution else DEFAULT_MIN_SOC_PCT
    max_soc = CAUTION_MAX_SOC_PCT if caution else DEFAULT_MAX_SOC_PCT

    return {
        "power_cap_factor": round(power_cap_factor, 4),
        "min_soc_pct": min_soc,
        "max_soc_pct": max_soc,
        "caution": caution,
        "reasons": reasons,
    }


def _turning_points(values: list) -> list:
    """Drop consecutive duplicates from a SOC trajectory so the rainflow
    engine sees real turning points. The platform's ASTM E1049-85
    implementation detects extrema via sign changes in adjacent deltas,
    which MISSES plateau-heavy square-wave profiles (a battery that holds
    a SOC level for hours then swings) and reports zero cycles for them;
    deduplicating to turning points is the standard preprocessing for
    rainflow counting and makes EFC/DoD accounting work for real
    dispatch-shaped trajectories."""
    out = []
    for v in values:
        if not out or abs(float(v) - out[-1]) > 1e-9:
            out.append(float(v))
    return out


def _charge_discharge_thresholds(prices: list) -> tuple:
    """Low/high price thresholds from the window's own percentiles — a
    battery charges below the 35th percentile and discharges above the 65th,
    so the heuristic adapts to the actual price level of the window rather
    than a hardcoded absolute. Degenerate (all-equal-price) windows degrade
    to the mean (no arbitrage opportunity)."""
    import numpy as np

    arr = np.asarray(prices, dtype=float)
    lo = float(np.percentile(arr, 35))
    hi = float(np.percentile(arr, 65))
    if hi <= lo:
        hi = lo = float(np.mean(arr))
    return lo, hi


def arbitrage_schedule(
    prices_eur_per_kwh: list,
    battery_kwh: float,
    c_rate: float = 0.5,
    round_trip_efficiency: float = 0.90,
    soh_pct: float = 100.0,
    sop_pct: "float | None" = None,
    rul_cycles: "float | None" = None,
    rul_reliable: bool = False,
    initial_soc_pct: float = 50.0,
) -> dict:
    """Hour-by-hour price-arbitrage schedule respecting the cell's health
    constraints (see health_constrained_band()). Threshold heuristic, not
    an LP/MILP optimizer — see module docstring for the honest scope.

    Returns:
      schedule: [{hour, price_eur_per_kwh, charge_kw, discharge_kw,
                  soc_pct}]
      band:     health_constrained_band() output
      revenue_eur, throughput_kwh, efc, dod_mean_pct,
      limitations: [str]
    """
    import numpy as np

    prices = [float(p) for p in prices_eur_per_kwh]
    if not prices:
        raise ValueError("prices_eur_per_kwh must not be empty.")
    if battery_kwh <= 0:
        raise ValueError("battery_kwh must be > 0.")

    band = health_constrained_band(
        soh_pct=soh_pct, sop_pct=sop_pct,
        rul_cycles=rul_cycles, rul_reliable=rul_reliable,
    )

    nominal_power_kw = battery_kwh * c_rate
    power_cap_kw = nominal_power_kw * band["power_cap_factor"]
    min_soc = band["min_soc_pct"]
    max_soc = band["max_soc_pct"]

    lo_thresh, hi_thresh = _charge_discharge_thresholds(prices)

    soc = float(initial_soc_pct)
    soc = min(max(soc, min_soc), max_soc)
    schedule = []
    revenue = 0.0
    throughput_kwh = 0.0

    for hour, price in enumerate(prices):
        charge_kw = 0.0
        discharge_kw = 0.0

        if price < lo_thresh and soc < max_soc:
            # Charge: energy stored 1:1 on charge (efficiency applied at
            # discharge, same accounting as deployment_sizing).
            headroom_kwh = battery_kwh * (max_soc - soc) / 100.0
            charge_kw = min(power_cap_kw, headroom_kwh)
            soc += charge_kw / battery_kwh * 100.0
            revenue -= charge_kw * price
            throughput_kwh += charge_kw
        elif price > hi_thresh and soc > min_soc:
            available_kwh = battery_kwh * (soc - min_soc) / 100.0
            discharge_kw = min(power_cap_kw, available_kwh)
            delivered_kwh = discharge_kw * round_trip_efficiency
            soc -= discharge_kw / battery_kwh * 100.0
            revenue += delivered_kwh * price
            throughput_kwh += discharge_kw

        schedule.append({
            "hour": hour,
            "price_eur_per_kwh": round(price, 5),
            "charge_kw": round(charge_kw, 4),
            "discharge_kw": round(discharge_kw, 4),
            "soc_pct": round(soc, 2),
        })

    # EFC delivered, via the platform's own ASTM E1049-85 rainflow engine on
    # the resulting SOC trajectory (turning-point preprocessed — see
    # _turning_points() for why).
    soc_series = [s["soc_pct"] for s in schedule]
    rainflow = rainflow_counting(np.array(_turning_points(soc_series), dtype=float))

    # Mean cycle DoD, weighted by rainflow count — the per-cycle stress
    # measure that actually reflects a narrowed operating band (a plain
    # mean hourly SOC swing can be LOWER for a wide band that sits idle,
    # which would misreport the stress the band protects against).
    weighted_dod_sum = sum(c["range_dod"] * c["count"] for c in rainflow["cycles"])
    weighted_dod_count = sum(c["count"] for c in rainflow["cycles"])
    mean_cycle_dod = (weighted_dod_sum / weighted_dod_count) if weighted_dod_count > 0 else 0.0

    return {
        "schedule": schedule,
        "band": band,
        "revenue_eur": round(revenue, 2),
        "throughput_kwh": round(throughput_kwh, 2),
        "efc": rainflow["equivalent_full_cycles"],
        "mean_cycle_dod_pct": round(float(mean_cycle_dod), 2),
        "thresholds_eur_per_kwh": {"charge_below": round(lo_thresh, 5), "discharge_above": round(hi_thresh, 5)},
        "limitations": [
            "Threshold heuristic, not an LP/MILP optimizer — no forecast horizon or lookahead.",
            "Revenue assumes the market clears at the input prices (no price impact from this battery).",
            "SoP is a rate-capability proxy (1/R), not a measured power test.",
            "RUL-based band narrowing is an engineering rule, not a validated optimal-control result.",
            "Schedule ignores owner load/PV/minimum-reserve obligations; arbitrage and ancillary services are mutually exclusive in any hour.",
        ],
    }


def schedule_comparison(
    prices_eur_per_kwh: list,
    battery_kwh: float,
    c_rate: float = 0.5,
    round_trip_efficiency: float = 0.90,
    soh_pct: float = 100.0,
    sop_pct: "float | None" = None,
    rul_cycles: "float | None" = None,
    rul_reliable: bool = False,
    initial_soc_pct: float = 50.0,
) -> dict:
    """Dispatch the same price window twice — once under the cohort's own
    implicit assumption (the battery is healthy: soh_pct=100, no SoP limit,
    no RUL caution — exactly what Capture/Solship/Deepgrid-style optimizers
    assume when they bid dispatchable capacity), once health-constrained
    with the cell's REAL health signals (this module) — and report the
    honest tradeoff: revenue given up vs. stress (EFC / cycle DoD) avoided.
    This is the platform's differentiator made directly measurable.

    Returns {"healthy_assumption": {...schedule result...},
             "health_constrained": {...}, "delta": {...}} where delta keys
    are signed (positive = the healthy-assumption schedule earned more /
    cycled harder; with the threshold heuristic the sign can flip on
    degenerate price shapes, so present them as a signed comparison, not
    as guaranteed losses)."""
    # Deliberately soh_pct=100 on this leg: modeling the cohort's "assume
    # healthy" behavior, not accidentally passing the real SOH through
    # (which would re-trigger the caution band and defeat the comparison).
    unconstrained = arbitrage_schedule(
        prices_eur_per_kwh=prices_eur_per_kwh, battery_kwh=battery_kwh,
        c_rate=c_rate, round_trip_efficiency=round_trip_efficiency,
        soh_pct=100.0, sop_pct=None, rul_cycles=None, rul_reliable=False,
        initial_soc_pct=initial_soc_pct,
    )
    constrained = arbitrage_schedule(
        prices_eur_per_kwh=prices_eur_per_kwh, battery_kwh=battery_kwh,
        c_rate=c_rate, round_trip_efficiency=round_trip_efficiency,
        soh_pct=soh_pct, sop_pct=sop_pct,
        rul_cycles=rul_cycles, rul_reliable=rul_reliable,
        initial_soc_pct=initial_soc_pct,
    )
    return {
        "healthy_assumption": unconstrained,
        "health_constrained": constrained,
        "delta": {
            "revenue_eur_delta": round(unconstrained["revenue_eur"] - constrained["revenue_eur"], 2),
            "efc_delta": round(unconstrained["efc"] - constrained["efc"], 3),
            "mean_cycle_dod_pct_delta": round(
                unconstrained["mean_cycle_dod_pct"] - constrained["mean_cycle_dod_pct"], 2
            ),
        },
    }
