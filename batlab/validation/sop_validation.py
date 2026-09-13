"""
SoP proxy validation — what the 1/R State-of-Power number is, what it
assumes, and how wrong it could plausibly be.

Why this exists
---------------
`sop_pct` in batlab.features.engineering is a PROXY: peak power is
estimated as proportional to 1/R under a constant-voltage approximation.
Every downstream consumer (health-aware dispatch power caps, application
fit scoring, bankability, fleet aggregation) scales power by this
number, and the Health page shows it as "Peak power: N% of initial".
None of the platform's datasets contain a measured power test (a pulse
discharge with simultaneous V/I logging would be needed to measure peak
power directly), so the proxy cannot be validated against ground truth
here. What CAN be done honestly:

  1. STATE the physical assumption and its failure modes. P_peak =
     V_cutoff² / (R_total + R_pulse_path) under a constant-V cutoff; the
     proxy drops the (R_pulse_path) term and the voltage-dependence of
     R. For Li-ion with realistic ohmic-dominated R at room temperature
     the 1/R form is the standard first-order rate-capability estimate
     — but the absolute number can be off by tens of percent when
     charge-transfer polarization dominates (low temperature, high
     C-rate).
  2. CHECK internal consistency, which the data DOES support. If the
     cell's fade is dominated by uniform resistance growth (no
     structural damage shifting the R↔power relationship), then
     sop_pct ≈ R0/R(t) and should track resistance_normalized⁻¹
     exactly; deviations flag model-structure drift, and the check is
     mechanical.
  3. PUBLISH the uncertainty band with the number. The proxy's plausible
     error is not symmetric — we carry a ±25% band derived from the
     spread of published R_pulse/R_ohmic ratios for Li-ion 18650s at
     20-40 °C — and every consumer decision threshold sits far inside
     the band's implications, so the band's honest message is "SoP
     ordering between cells is far more trustworthy than any absolute
     SoP value".

`sop_proxy_validation()` computes the consistency metrics for one fleet
and returns the standard scoping label every display can attach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plausible relative error of the 1/R constant-V proxy for the absolute
# peak-power level. Derived from the spread of published charge-transfer
# vs ohmic resistance ratios for Li-ion 18650 cells at 20-40 °C: when
# R_ct is 10-50% of R_total and V-sag shifts the effective cutoff, the
# true P_peak deviates from V²/R by roughly this band. NOT a measured
# property of any platform fleet — it is a literature-scale prior, which
# is exactly why the scoping label says what it says.
SOP_PROXY_UNCERTAINTY_PCT = 25.0

# Deviation between sop_pct and 1/resistance_normalized beyond which the
# fleet's R↔power relationship has drifted from the uniform-growth
# assumption (the proxy's core assumption) — a structural flag, not noise.
CONSISTENCY_DEVIATION_FLOOR_PCT = 5.0

SCOPING_LABEL = (
    "SoP is a 1/R rate-capability PROXY under a constant-voltage approximation, "
    "not a measured power test — no pulse-power data exists in any reference "
    "fleet. Absolute values carry an uncertainty on the order of ±25% "
    "(literature-scale prior, not measured here); cross-cell and cross-time "
    "ORDERING is substantially more trustworthy than any absolute number."
)


def sop_proxy_validation(cell_data: dict) -> dict:
    """Characterize the SoP proxy's behavior on one fleet.

    Computes, per cell, the consistency between the served sop_pct and
    its definition (100 × R0/R), then fleet-level aggregates:

    Returns:
        {
          "scoping_label": str,        # the standard disclosure
          "uncertainty_band_pct": 25.0,
          "per_cell": {cell_id: {
              "sop_latest": float,          # served value
              "sop_from_definition": float, # 100 * R0/R(latest)
              "max_abs_deviation_pct": float, # worst row deviation over life
              "consistent": bool,
          }},
          "fleet_consistent_fraction": float,  # share of cells passing
          "verdict": str,
        }
    """
    per_cell = {}
    for cell_id, df in (cell_data or {}).items():
        if df is None or "sop_pct" not in df.columns or "resistance_ohm" not in df.columns:
            continue
        r = pd.to_numeric(df["resistance_ohm"], errors="coerce").to_numpy(dtype=float)
        sop = pd.to_numeric(df["sop_pct"], errors="coerce").to_numpy(dtype=float)
        r_pos = r[np.isfinite(r) & (r > 0)]
        if len(r_pos) == 0:
            continue
        r0 = float(r_pos[0])
        valid = np.isfinite(sop) & np.isfinite(r) & (r > 0)
        if valid.sum() < 2:
            continue
        sop_def = 100.0 * r0 / r[valid]
        dev = np.abs(sop[valid] - sop_def) / np.maximum(sop_def, 1e-9) * 100.0
        max_dev = float(np.max(dev))
        per_cell[cell_id] = {
            "sop_latest": float(sop[valid][-1]),
            "sop_from_definition": float(sop_def[-1]),
            "max_abs_deviation_pct": round(max_dev, 3),
            "consistent": bool(max_dev <= CONSISTENCY_DEVIATION_FLOOR_PCT),
        }

    n = len(per_cell)
    consistent_n = sum(1 for v in per_cell.values() if v["consistent"])
    frac = (consistent_n / n) if n else 0.0
    if n == 0:
        verdict = (
            "no resistance data — SoP not computed on this fleet; "
            "power-capable consumers must treat the cell as unrated"
        )
    elif frac >= 0.99:
        verdict = (
            "sop_pct is internally consistent with its 1/R definition on every "
            "cell — the proxy's behavior is understood; the ±25% absolute "
            "uncertainty stands, the ordering is trustworthy"
        )
    else:
        verdict = (
            f"{n - consistent_n}/{n} cell(s) deviate > "
            f"{CONSISTENCY_DEVIATION_FLOOR_PCT:.0f}% from the 1/R definition "
            "over life — the uniform-resistance-growth assumption behind the "
            "proxy is drifting on those cells; treat their absolute SoP with "
            "correspondingly less trust"
        )

    return {
        "scoping_label": SCOPING_LABEL,
        "uncertainty_band_pct": SOP_PROXY_UNCERTAINTY_PCT,
        "consistency_deviation_floor_pct": CONSISTENCY_DEVIATION_FLOOR_PCT,
        "per_cell": per_cell,
        "fleet_consistent_fraction": round(frac, 3),
        "verdict": verdict,
    }
