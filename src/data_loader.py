"""
Battery data loader — Phase 0 Foundation.

Data model:
  Battery  → Cell  → Cycle (one row per charge/discharge cycle)

Data source: physics-informed synthetic data with injected cell-to-cell
stress variation (temperature, C-rate, depth of discharge).

Each cell is assigned an operating stress profile. Degradation curves are
derived from those profiles using physically grounded formulae — not tuned
to produce a desired feature importance result.

Real CSV files (Oxford or NASA) can be dropped into data/raw/ to replace
synthetic generation without any other code changes.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NOMINAL_CAPACITY_AH = 0.74   # Oxford 18650 cell nominal capacity


# ---------------------------------------------------------------------------
# Per-cell stress profiles
#
# Physical basis for stress parameter ranges:
#   Temperature: Oxford cells characterised at 25°C ambient. Real-world
#     Li-ion cells operate 15–45°C. Each °C above 25°C exponentially
#     accelerates SEI layer growth (Arrhenius).
#   C-rate: Oxford tests at 1C. Higher C-rates cause lithium plating and
#     mechanical fatigue. 0.5C (slow/overnight charge) to 2C (fast charge).
#   DoD: Oxford tests at 100% DoD. Partial cycling (70–90%) significantly
#     reduces per-cycle stress. Standard EV battery management targets 80%.
#
# n_cycles is set so most cells reach or approach EOL (80% SOH) within
# the simulated window — necessary for the RUL model to have real targets.
# ---------------------------------------------------------------------------

CELL_STRESS_PROFILES = {
    "Cell1": dict(temp_mean=25.0, c_rate=1.0, dod=1.00, n_cycles=1200),  # baseline
    "Cell2": dict(temp_mean=30.0, c_rate=1.0, dod=1.00, n_cycles=1000),  # mild heat
    "Cell3": dict(temp_mean=35.0, c_rate=1.0, dod=1.00, n_cycles=800),   # moderate heat
    "Cell4": dict(temp_mean=25.0, c_rate=1.5, dod=1.00, n_cycles=1000),  # fast charge
    "Cell5": dict(temp_mean=25.0, c_rate=0.5, dod=0.80, n_cycles=1500),  # gentle use
    "Cell6": dict(temp_mean=35.0, c_rate=1.5, dod=0.90, n_cycles=650),   # combined stress
    "Cell7": dict(temp_mean=20.0, c_rate=0.5, dod=0.70, n_cycles=1500),  # optimal conditions
    "Cell8": dict(temp_mean=40.0, c_rate=2.0, dod=1.00, n_cycles=500),   # worst case
}


# ---------------------------------------------------------------------------
# Stress physics
# ---------------------------------------------------------------------------

def _stress_factor(temp_mean: float, c_rate: float, dod: float) -> float:
    """
    Compute a dimensionless stress multiplier relative to the baseline cell
    (25°C, 1C, 100% DoD → stress_factor = 1.0).

    Formulae:
      Temperature: Arrhenius SEI growth, activation energy ~0.5 eV for LiCoO2.
        Simplified to exp(0.05 × ΔT). At 35°C: ×1.65. At 40°C: ×2.12.
        Source: Birkl et al. (2017), Waldmann et al. (2014).

      C-rate: Empirical power law from cycle-life studies.
        C^0.7 — sub-linear because lithium plating mechanism saturates.
        At 2C: ×1.62. At 0.5C: ×0.62.
        Source: Schmalstieg et al. (2014), Omar et al. (2014).

      DoD: Rainflow counting / Wöhler curve analogy.
        DoD^1.5 — super-linear because deeper discharge stresses electrodes
        more than proportionally (mechanical + chemical).
        At 80% DoD: ×0.72. At 70% DoD: ×0.59.
        Source: Xu et al. (2016), Abdel-Monem et al. (2017).
    """
    temp_factor  = np.exp(0.05 * (temp_mean - 25.0))
    crate_factor = (c_rate / 1.0) ** 0.7
    dod_factor   = (dod   / 1.0) ** 1.5
    return temp_factor * crate_factor * dod_factor


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_cell_data(
    cell_id: str,
    n_cycles: int,
    temp_mean: float,
    c_rate: float,
    dod: float,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic cycle-summary DataFrame for one cell.

    Degradation model:
      capacity(n) = Q0 × (1 − k_fade × n^1.1) + noise
      resistance(n) = R0 + k_res × n + noise

    where k_fade and k_res are both scaled by the stress factor.
    The 1.1 power-law exponent reflects the accelerating nature of SEI
    growth observed in Oxford and NASA datasets.

    Per-cycle temperature is drawn from Normal(temp_mean, 2.5°C) to
    simulate realistic measurement noise around the cell's operating point.
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)

    sf = _stress_factor(temp_mean, c_rate, dod)

    # ── Capacity fade ──
    base_fade_rate = 0.00008 * sf
    cap_noise = rng.normal(0, 0.002, n_cycles)
    capacity_ah = (
        nominal_capacity * (1.0 - base_fade_rate * cycles ** 1.1)
        + cap_noise
    )
    capacity_ah = np.clip(capacity_ah, 0.01, nominal_capacity)

    # ── Internal resistance growth ──
    r0 = 0.150
    r_growth = 0.00007 * sf
    r_noise = rng.normal(0, 0.002, n_cycles)
    resistance_ohm = r0 + r_growth * cycles + r_noise
    resistance_ohm = np.clip(resistance_ohm, 0.10, 0.90)

    # ── Per-cycle temperature ──
    # Varies around the cell's mean operating temperature with ±2.5°C noise.
    # This is the only stress parameter carried into the feature matrix —
    # C-rate and DoD are not measured per-cycle in Oxford-style test data.
    temperature_c = rng.normal(temp_mean, 2.5, n_cycles)

    return pd.DataFrame({
        "cycle_number":  cycles,
        "capacity_ah":   capacity_ah,
        "resistance_ohm": resistance_ohm,
        "temperature_c": temperature_c,
    })


# ---------------------------------------------------------------------------
# Load or generate one cell
# ---------------------------------------------------------------------------

def load_or_generate_cell(cell_id: str) -> pd.DataFrame:
    """
    Load from local CSV if present and valid, otherwise generate and cache.

    A cached file is considered stale if it's missing the 'temperature_c'
    column (which was added in the stress-variation rewrite). Stale files
    are deleted and regenerated automatically.
    """
    local_path = os.path.join(DATA_DIR, f"{cell_id}_summary.csv")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # Stale-cache check: pre-stress-rewrite files lack temperature_c.
        if "temperature_c" in df.columns and "capacity_ah" in df.columns:
            print(f"  [cache] {cell_id} loaded from local CSV ({len(df)} cycles)")
            return df
        else:
            print(f"  [stale] {cell_id} cache missing expected columns — regenerating")
            os.remove(local_path)

    # Use profile if defined, else fall back to baseline.
    profile = CELL_STRESS_PROFILES.get(cell_id, CELL_STRESS_PROFILES["Cell1"])
    seed = int(cell_id.replace("Cell", "")) * 42

    sf = _stress_factor(profile["temp_mean"], profile["c_rate"], profile["dod"])
    print(
        f"  [generate] {cell_id}: T={profile['temp_mean']}°C  "
        f"C-rate={profile['c_rate']}C  DoD={profile['dod']*100:.0f}%  "
        f"stress={sf:.2f}×  n={profile['n_cycles']} cycles"
    )

    df = generate_cell_data(cell_id=cell_id, seed=seed, **profile)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"  [cache] Saved to {local_path}")
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from a real CSV to the internal schema."""
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[\s/()+]+", "_", regex=True)
    )
    renames = {}
    for candidate in ["cycle_index", "cycle", "cycle_number"]:
        if candidate in df.columns:
            renames[candidate] = "cycle_number"; break
    for candidate in ["discharge_capacity_ah_", "discharge_capacity", "capacity_ah", "qd"]:
        if candidate in df.columns:
            renames[candidate] = "capacity_ah"; break
    for candidate in ["internal_resistance_ohm_", "internal_resistance", "resistance_ohm"]:
        if candidate in df.columns:
            renames[candidate] = "resistance_ohm"; break
    for candidate in ["temperature_c_", "temperature", "temperature_c"]:
        if candidate in df.columns:
            renames[candidate] = "temperature_c"; break
    return df.rename(columns=renames)


# ---------------------------------------------------------------------------
# Enrich cycles with derived metrics
# ---------------------------------------------------------------------------

def enrich_cycles(df: pd.DataFrame, eol_threshold_pct: float = 80.0) -> pd.DataFrame:
    """Add computed columns: SOH, capacity fade, EOL flag, rolling averages."""
    df = df.copy().sort_values("cycle_number").reset_index(drop=True)

    initial_capacity = df["capacity_ah"].iloc[0]
    df["soh_pct"]          = (df["capacity_ah"] / initial_capacity) * 100.0
    df["capacity_fade_ah"] = initial_capacity - df["capacity_ah"]
    df["is_eol"]           = df["soh_pct"] < eol_threshold_pct
    df["soh_rolling_avg"]  = df["soh_pct"].rolling(10, min_periods=1).mean()
    df["capacity_fade_rate"] = df["capacity_ah"].diff().abs().rolling(5, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# Build the Battery → Cells → Cycles hierarchy
# ---------------------------------------------------------------------------

def build_battery(battery_id: str, cell_ids: list[str]) -> dict:
    """
    Load and structure data for one battery containing one or more cells.

    Returns:
    {
        "battery_id": str,
        "cells": {
            "Cell1": {"cell_id": str, "cycles": DataFrame},
            ...
        }
    }
    """
    battery = {"battery_id": battery_id, "cells": {}}

    for cell_id in cell_ids:
        print(f"\nLoading {cell_id}...")
        raw_df   = load_or_generate_cell(cell_id)
        clean_df = _normalise_columns(raw_df) if "capacity_ah" not in raw_df.columns else raw_df
        enriched = enrich_cycles(clean_df)

        n       = len(enriched)
        soh_end = enriched["soh_pct"].iloc[-1]
        eol_cy  = (
            enriched.loc[enriched["is_eol"], "cycle_number"].iloc[0]
            if enriched["is_eol"].any() else "not reached"
        )
        print(f"  [ok] {n} cycles | Final SOH: {soh_end:.1f}% | EOL at cycle: {eol_cy}")

        battery["cells"][cell_id] = {"cell_id": cell_id, "cycles": enriched}

    return battery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cell_df(battery: dict, cell_id: str) -> pd.DataFrame:
    return battery["cells"][cell_id]["cycles"]


def list_cells(battery: dict) -> list[str]:
    return list(battery["cells"].keys())


def quality_report(battery: dict) -> pd.DataFrame:
    rows = []
    for cell_id, cell in battery["cells"].items():
        df = cell["cycles"]
        rows.append({
            "cell_id":          cell_id,
            "n_cycles":         len(df),
            "missing_capacity": df["capacity_ah"].isna().sum(),
            "cycles_monotonic": df["cycle_number"].is_monotonic_increasing,
            "final_soh_pct":    round(df["soh_pct"].iloc[-1], 1),
            "eol_reached":      df["is_eol"].any(),
            "stress_factor":    round(
                _stress_factor(
                    CELL_STRESS_PROFILES.get(cell_id, {}).get("temp_mean", 25),
                    CELL_STRESS_PROFILES.get(cell_id, {}).get("c_rate",    1.0),
                    CELL_STRESS_PROFILES.get(cell_id, {}).get("dod",       1.0),
                ), 2
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Battery Intelligence Platform — Phase 0 Data Load ===\n")
    print("Stress factors per cell:")
    for cid, p in CELL_STRESS_PROFILES.items():
        sf = _stress_factor(p["temp_mean"], p["c_rate"], p["dod"])
        print(f"  {cid}: T={p['temp_mean']}°C  C={p['c_rate']}C  "
              f"DoD={p['dod']*100:.0f}%  → stress={sf:.2f}×")

    print()
    battery = build_battery(
        battery_id="Oxford_B1",
        cell_ids=list(CELL_STRESS_PROFILES.keys()),
    )

    print("\n--- Quality Report ---")
    print(quality_report(battery).to_string(index=False))
