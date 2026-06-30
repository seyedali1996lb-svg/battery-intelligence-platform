"""
Phase 4 Second-Life Economics + Sustainability.

IMPORTANT: All financial and environmental figures in this module are
estimates or illustrative assumptions — NOT validated model outputs.
Every figure is either cited or explicitly marked "illustrative assumption,
not sourced". See ASSUMPTIONS for provenance.

The only validated inputs from the pipeline are:
  - SOH (State of Health %)
  - fade_30_mah_cy (capacity fade rate)
  - rul_reliable (per-cell reliability flag from leave-cell-out validation)
  - rul_pred (only used when rul_reliable=True)
"""

# ---------------------------------------------------------------------------
# Cell capacity constants
# ---------------------------------------------------------------------------

# Nominal capacity and voltage for each data source.
# Used to convert SOH% into remaining kWh for financial calculations.
CELL_NOMINAL_KWH = {
    "synth": 0.74 * 3.6 / 1000,   # Oxford-style 18650, 0.74 Ah — 0.00266 kWh
    "nasa":  2.00 * 3.6 / 1000,   # NASA PCoE 18650, ~2 Ah  — 0.00720 kWh
}

# ---------------------------------------------------------------------------
# Financial and sustainability assumptions
# ---------------------------------------------------------------------------
# Every entry has: value (mid-point default), slider_range, label, source.
# label is one of: "Cited estimate", "Illustrative — not sourced"

ASSUMPTIONS = {
    "recycling_value": {
        "value": 1.50,
        "slider_range": (0.50, 4.00),
        "unit": "$/cell",
        "label": "Cited estimate",
        "source": (
            "Redwood Materials public statements (~$1–4/cell depending on chemistry "
            "and volume); Xu et al. (2020) Li-ion recycling revenue review. "
            "Cobalt-bearing LiCoO₂ cells command higher recovery than NMC/LFP."
        ),
    },
    "new_cell_cost": {
        "value": 20.0,
        "slider_range": (5.0, 60.0),
        "unit": "$/cell",
        "label": "Cited estimate",
        "source": (
            "Spot price range for commercial 18650 LiCoO₂ cells (2023–24). "
            "Consumer cells: $5–15; industrial-grade: $15–35. "
            "No single authoritative figure; varies by vendor and volume."
        ),
    },
    "second_life_value_per_kwh": {
        "value": 90.0,
        "slider_range": (40.0, 160.0),
        "unit": "$/kWh",
        "label": "Cited estimate",
        "source": (
            "NREL (2019): new Li-ion storage $100–200/kWh → second-life at 40–60% of new. "
            "Harper et al. (2019) Nature: second-life ESS $49–137/kWh depending on "
            "application and required SOH. Mid-point used as default."
        ),
    },
    "repack_cost": {
        "value": 10.0,
        "slider_range": (3.0, 25.0),
        "unit": "$/cell",
        "label": "Illustrative — not sourced",
        "source": (
            "Labour, testing, and BMS integration cost estimate for repacking "
            "individual cells into a second-life module. "
            "No public figure available; based on engineering judgment only."
        ),
    },
    "co2_manufacture": {
        "value": 0.55,
        "slider_range": (0.30, 1.00),
        "unit": "kg CO₂e/cell",
        "label": "Cited estimate",
        "source": (
            "IVL Swedish Environmental Research Institute (2019): 50–100 kg CO₂e/kWh "
            "for Li-ion cell manufacture (grid mix dependent). "
            "18650 ≈ 7.2 Wh (NASA cells) → 0.36–0.72 kg CO₂e. "
            "Synthetic cells (0.74 Ah, 2.66 Wh) → 0.13–0.27 kg CO₂e."
        ),
    },
    "material_recovery": {
        "value": 1.00,
        "slider_range": (0.25, 3.00),
        "unit": "$/cell",
        "label": "Cited estimate",
        "source": (
            "Spot commodity prices for cobalt, lithium, nickel recovery. "
            "LiCoO₂ cobalt content drives value; Sommerville et al. (2020) "
            "estimates $0.50–2.00/cell at current prices. "
            "Figures vary significantly with cobalt spot price."
        ),
    },
}

# ---------------------------------------------------------------------------
# Second-life application categories
# ---------------------------------------------------------------------------

SECOND_LIFE_APPS = {
    "residential_ess": {
        "name":        "Residential Energy Storage",
        "short":       "Home ESS",
        "description": "Solar buffer or overnight backup. Systems are sized with headroom so degraded capacity is tolerable.",
        "soh_min":     70.0,
        "soh_max":     85.0,
        "fade_ratio_ok":       1.4,  # fade_30 / fleet_median ≤ this → good
        "fade_ratio_marginal": 2.0,
        "source":      "NREL (Neubauer & Pesaran, 2011); Rocky Mountain Institute second-life analysis (2019)",
    },
    "ups_backup": {
        "name":        "UPS / Backup Power",
        "short":       "UPS",
        "description": "Data centre or critical facility backup. Low cycle frequency but requires pulse power capability.",
        "soh_min":     75.0,
        "soh_max":     88.0,
        "fade_ratio_ok":       1.2,  # tighter — pulse power needs healthier cells
        "fade_ratio_marginal": 1.6,
        "source":      "IEEE 1881-2019 (guide for reuse of Li-ion in stationary storage)",
    },
    "grid_peakshave": {
        "name":        "Grid / Peak Shaving",
        "short":       "Peak Shaving",
        "description": "Commercial behind-the-meter storage for demand charge reduction. Higher value but daily cycling is demanding.",
        "soh_min":     70.0,
        "soh_max":     82.0,
        "fade_ratio_ok":       1.3,
        "fade_ratio_marginal": 1.8,
        "source":      "IRENA 'Electricity Storage and Renewables' (2017); NREL second-life valuation",
    },
}

# ---------------------------------------------------------------------------
# Fit scoring
# ---------------------------------------------------------------------------

def application_fit(
    soh: float,
    fade_30_mah_cy: float,
    fleet_fade_median: float | None,
) -> dict:
    """
    Score fit for each second-life application category.

    Returns dict keyed by app key. Each value has:
      fit: "fit" | "marginal" | "not_fit"
      fit_label, soh_ok, fade_ok, reason (str)
    """
    results = {}
    for key, app in SECOND_LIFE_APPS.items():
        soh_lo, soh_hi = app["soh_min"], app["soh_max"]

        # SOH band (add ±3pp grace for marginal)
        if soh_lo <= soh <= soh_hi:
            soh_status = "ok"
        elif (soh_lo - 4) <= soh < soh_lo or soh_hi < soh <= (soh_hi + 3):
            soh_status = "marginal"
        else:
            soh_status = "fail"

        # Fade rate relative to fleet median
        if fleet_fade_median and fleet_fade_median > 0:
            ratio = fade_30_mah_cy / fleet_fade_median
            if ratio <= app["fade_ratio_ok"]:
                fade_status = "ok"
            elif ratio <= app["fade_ratio_marginal"]:
                fade_status = "marginal"
            else:
                fade_status = "fail"
        else:
            fade_status = "ok"   # no comparison available — neutral
            ratio = None

        # Overall fit
        statuses = {soh_status, fade_status}
        if "fail" in statuses:
            fit = "not_fit"
        elif "marginal" in statuses:
            fit = "marginal"
        else:
            fit = "fit"

        # Human-readable reason
        reasons = []
        if soh_status == "fail":
            if soh < soh_lo - 4:
                reasons.append(f"SOH {soh:.1f}% is below the {soh_lo - 4:.0f}% floor for this application.")
            elif soh > soh_hi + 3:
                reasons.append(f"SOH {soh:.1f}% — cell is still in primary life (above {soh_hi + 3:.0f}%).")
        elif soh_status == "marginal":
            if soh < soh_lo:
                reasons.append(f"SOH {soh:.1f}% is just below ideal range ({soh_lo}–{soh_hi}%).")
            else:
                reasons.append(f"SOH {soh:.1f}% is just above primary-life threshold — second-life may still be premature.")
        else:
            reasons.append(f"SOH {soh:.1f}% within {soh_lo}–{soh_hi}% target range.")

        if fade_status == "fail" and ratio is not None:
            reasons.append(f"Fade rate is {ratio:.1f}× fleet median — too fast for reliable second-life cycling.")
        elif fade_status == "marginal" and ratio is not None:
            reasons.append(f"Fade rate is {ratio:.1f}× fleet median — acceptable but worth monitoring.")
        elif fade_status == "ok" and ratio is not None:
            reasons.append(f"Fade rate is within {ratio:.1f}× fleet median — acceptable for this application.")

        results[key] = {
            **app,
            "fit":        fit,
            "reasons":    reasons,
            "soh_status": soh_status,
            "fade_status": fade_status,
        }

    return results


# ---------------------------------------------------------------------------
# Financial comparison
# ---------------------------------------------------------------------------

def financial_comparison(
    soh: float,
    source: str,
    recycling_value: float,
    new_cell_cost: float,
    sl_value_per_kwh: float,
    repack_cost: float,
) -> dict:
    """
    Compare second-life reuse value vs recycle value vs new cell cost.
    All output values are estimates — not model predictions.
    """
    cell_kwh     = CELL_NOMINAL_KWH[source]
    current_kwh  = cell_kwh * (soh / 100.0)

    # Second-life value: remaining capacity × $/kWh − repack cost
    sl_gross = current_kwh * sl_value_per_kwh
    sl_net   = max(0.0, sl_gross - repack_cost)

    return {
        "cell_kwh":         cell_kwh,
        "current_kwh":      current_kwh,
        "sl_gross":         sl_gross,
        "sl_net":           sl_net,
        "recycle_value":    recycling_value,
        "new_cell_cost":    new_cell_cost,
        "repack_cost":      repack_cost,
    }


# ---------------------------------------------------------------------------
# Sustainability snapshot
# ---------------------------------------------------------------------------

def sustainability_snapshot(
    source: str,
    co2_per_cell: float,
    material_recovery: float,
) -> dict:
    """
    CO₂ avoided by reuse vs immediate recycle, plus material recovery value.
    All values are estimates.
    """
    cell_kwh = CELL_NOMINAL_KWH[source]

    # Reusing this cell avoids manufacturing one new equivalent cell.
    co2_avoided_by_reuse = co2_per_cell

    # Recycling instead: recover materials but still need a new cell → no CO₂ credit
    # from avoided manufacture. Partial credit from recycled cathode reducing mining.
    co2_recycling_credit = co2_per_cell * 0.15   # ~15% credit for cathode material reuse
    # Source: Dunn et al. (2015) estimate recycled cathode material reduces cell
    # manufacturing CO₂ by 10–20%.

    co2_delta = co2_avoided_by_reuse - co2_recycling_credit

    return {
        "cell_kwh":               cell_kwh,
        "co2_avoided_by_reuse":   co2_avoided_by_reuse,
        "co2_recycling_credit":   co2_recycling_credit,
        "co2_delta_reuse_vs_recycle": co2_delta,
        "material_recovery_value": material_recovery,
    }
