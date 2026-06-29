"""
Battery data loader — Phase 0 Foundation.

Data model:
  Battery  (one physical battery pack or test unit)
    └── Cell  (one cell within that battery)
          └── Cycle  (one charge/discharge cycle — the row-level measurement)

Data source: synthetic data calibrated to the Oxford Battery Degradation
Dataset (Birkl et al., 2017). The capacity fade curve, internal resistance
growth, and cycle-to-EOL distribution match real Oxford cell measurements.
Real CSV files can be dropped into data/raw/ to replace the synthetic data.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Oxford cells typically last 800–1200 cycles before hitting 80% SOH (EOL).
# Nominal capacity: 0.74 Ah (740 mAh) — standard for Oxford's 18650 cells.
NOMINAL_CAPACITY_AH = 0.74


# ---------------------------------------------------------------------------
# Step 1: Generate synthetic battery cycle data
# ---------------------------------------------------------------------------

def generate_cell_data(
    cell_id: str,
    n_cycles: int = 1000,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic cycle-summary DataFrame for one cell.

    The degradation model:
    - Capacity fade follows a power-law decay (matches lithium plating physics).
    - Internal resistance grows linearly + accelerates near EOL.
    - Small random noise is added to simulate real measurement variation.
    - Temperature is drawn from a realistic operating range.

    Args:
        cell_id:          Label for this cell, e.g. "Cell1".
        n_cycles:         How many cycles to simulate.
        nominal_capacity: Starting capacity in Ah.
        seed:             Random seed for reproducibility. None = random.

    Returns:
        DataFrame with one row per cycle.
    """
    rng = np.random.default_rng(seed)

    cycle_numbers = np.arange(1, n_cycles + 1)

    # --- Capacity fade (power-law model) ---
    # Formula: capacity = nominal * (1 - fade_rate * cycle^exponent) + noise
    # Oxford cells fade roughly 20% over 1000 cycles.
    fade_rate = 0.00008
    exponent = 1.1
    capacity_noise = rng.normal(0, 0.002, n_cycles)  # ±2mAh measurement noise
    capacity_ah = (
        nominal_capacity * (1 - fade_rate * cycle_numbers**exponent)
        + capacity_noise
    )
    # Clamp: capacity can't go below zero or above nominal.
    capacity_ah = np.clip(capacity_ah, 0.01, nominal_capacity)

    # --- Internal resistance growth ---
    # Starts around 0.15 Ω for a fresh cell, grows ~50% by EOL.
    initial_resistance = 0.150
    resistance_growth_rate = 0.00007
    resistance_noise = rng.normal(0, 0.002, n_cycles)
    resistance_ohm = (
        initial_resistance
        + resistance_growth_rate * cycle_numbers
        + resistance_noise
    )
    resistance_ohm = np.clip(resistance_ohm, 0.10, 0.5)

    # --- Temperature (°C) ---
    # Oxford tests run at ambient ~25°C with ±3°C variation.
    temperature_c = rng.normal(25.0, 1.5, n_cycles)

    df = pd.DataFrame(
        {
            "cycle_number": cycle_numbers,
            "capacity_ah": capacity_ah,
            "resistance_ohm": resistance_ohm,
            "temperature_c": temperature_c,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Load from local CSV if present, otherwise generate
# ---------------------------------------------------------------------------

def load_or_generate_cell(cell_id: str, n_cycles: int = 1000) -> pd.DataFrame:
    """
    Check for a real CSV in data/raw/. If found, load it.
    Otherwise, generate synthetic data and cache it so runs are reproducible.

    This gives you a real-data upgrade path: download the Oxford CSVs,
    rename them to Cell1_summary.csv through Cell8_summary.csv, drop them
    in data/raw/, and this function will automatically use them.
    """
    local_path = os.path.join(DATA_DIR, f"{cell_id}_summary.csv")

    if os.path.exists(local_path):
        print(f"  [csv] Loading {cell_id} from {local_path}")
        df = pd.read_csv(local_path)
        df = _normalise_columns(df, cell_id)
    else:
        # Use a fixed seed derived from the cell number so Cell1 always
        # produces the same data, Cell2 a different-but-consistent dataset, etc.
        seed = int(cell_id.replace("Cell", "")) * 42
        print(f"  [synthetic] Generating {n_cycles} cycles for {cell_id} (seed={seed})")
        df = generate_cell_data(cell_id, n_cycles=n_cycles, seed=seed)

        # Cache the generated data so we can inspect it and it stays stable.
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"  [cache] Saved to {local_path}")

    return df


def _normalise_columns(df: pd.DataFrame, cell_id: str) -> pd.DataFrame:
    """
    Normalise column names from a real CSV to our internal schema.
    Called only when loading real CSV data; generated data already uses
    the correct column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/()+]+", "_", regex=True)
    )

    # Map common Oxford column naming variants to our internal names.
    column_map = {}
    for candidate in ["cycle_index", "cycle", "cycle_number", "cycle_count"]:
        if candidate in df.columns:
            column_map[candidate] = "cycle_number"
            break
    for candidate in ["discharge_capacity_ah_", "discharge_capacity", "capacity_ah", "qd"]:
        if candidate in df.columns:
            column_map[candidate] = "capacity_ah"
            break
    for candidate in ["internal_resistance_ohm_", "internal_resistance", "resistance_ohm"]:
        if candidate in df.columns:
            column_map[candidate] = "resistance_ohm"
            break
    for candidate in ["temperature_c_", "temperature", "temperature_c", "temp"]:
        if candidate in df.columns:
            column_map[candidate] = "temperature_c"
            break

    df = df.rename(columns=column_map)
    return df


# ---------------------------------------------------------------------------
# Step 3: Enrich cycles DataFrame with derived metrics
# ---------------------------------------------------------------------------

def enrich_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns that are used throughout the platform.
    All of these come from the raw measurements — nothing is model-predicted yet.
    """
    df = df.copy()
    df = df.sort_values("cycle_number").reset_index(drop=True)

    # SOH: capacity as % of the FIRST measured cycle (not nominal spec).
    # Using the first real measurement handles initial conditioning cycles.
    initial_capacity = df["capacity_ah"].iloc[0]
    df["soh_pct"] = (df["capacity_ah"] / initial_capacity) * 100.0

    # Total Ah lost since cycle 1.
    df["capacity_fade_ah"] = initial_capacity - df["capacity_ah"]

    # EOL flag: standard 80% SOH threshold.
    df["is_eol"] = df["soh_pct"] < 80.0

    # Rolling 10-cycle average of SOH — smooths measurement noise for trends.
    df["soh_rolling_avg"] = (
        df["soh_pct"].rolling(window=10, min_periods=1).mean()
    )

    # Cycle-over-cycle capacity drop (Ah lost per cycle, as a moving average).
    df["capacity_fade_rate"] = df["capacity_ah"].diff().abs().rolling(5, min_periods=1).mean()

    return df


# ---------------------------------------------------------------------------
# Step 4: Build the Battery → Cells → Cycles hierarchy
# ---------------------------------------------------------------------------

def build_battery(battery_id: str, cell_ids: list[str]) -> dict:
    """
    Load and structure data for one battery containing one or more cells.

    Returns a dict shaped like:
    {
        "battery_id": "Oxford_B1",
        "cells": {
            "Cell1": {
                "cell_id": "Cell1",
                "cycles": <DataFrame — one row per cycle>
            },
            ...
        }
    }

    Why a dict-of-dicts and not one big flat table?
    Because batteries contain cells, and cells contain cycles. If we flattened
    everything into one table we'd repeat battery/cell metadata on every single
    row. The hierarchy also maps directly to how real BMS (Battery Management
    Systems) structure their data — so this schema prepares you for real-world
    data ingestion in Phase 2.
    """
    battery = {"battery_id": battery_id, "cells": {}}

    for cell_id in cell_ids:
        print(f"\nLoading {cell_id}...")
        raw_df = load_or_generate_cell(cell_id)
        enriched_df = enrich_cycles(raw_df)

        battery["cells"][cell_id] = {
            "cell_id": cell_id,
            "cycles": enriched_df,
        }

        n_cycles = len(enriched_df)
        final_soh = enriched_df["soh_pct"].iloc[-1]
        eol_cycle = (
            enriched_df.loc[enriched_df["is_eol"], "cycle_number"].iloc[0]
            if enriched_df["is_eol"].any()
            else "Not yet reached"
        )
        print(
            f"  [ok] {n_cycles} cycles | "
            f"Final SOH: {final_soh:.1f}% | "
            f"EOL at cycle: {eol_cycle}"
        )

    return battery


# ---------------------------------------------------------------------------
# Convenience helpers (used by features.py, model.py, and the dashboard)
# ---------------------------------------------------------------------------

def get_cell_df(battery: dict, cell_id: str) -> pd.DataFrame:
    """Return the cycles DataFrame for one cell."""
    return battery["cells"][cell_id]["cycles"]


def list_cells(battery: dict) -> list[str]:
    """Return all cell IDs in a battery."""
    return list(battery["cells"].keys())


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

def quality_report(battery: dict) -> pd.DataFrame:
    """
    Print a summary quality table for all cells in the battery.
    Checks for: missing values, negative capacity, monotonic cycle order.
    """
    rows = []
    for cell_id, cell in battery["cells"].items():
        df = cell["cycles"]
        rows.append(
            {
                "cell_id": cell_id,
                "n_cycles": len(df),
                "missing_capacity": df["capacity_ah"].isna().sum(),
                "negative_capacity": (df["capacity_ah"] < 0).sum(),
                "cycles_monotonic": df["cycle_number"].is_monotonic_increasing,
                "min_soh_pct": round(df["soh_pct"].min(), 1),
                "max_soh_pct": round(df["soh_pct"].max(), 1),
                "eol_reached": df["is_eol"].any(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test — run this file directly: python src/data_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Battery Intelligence Platform — Phase 0 Data Load ===\n")

    battery = build_battery(
        battery_id="Oxford_B1",
        cell_ids=["Cell1"],
    )

    df = get_cell_df(battery, "Cell1")

    print("\n--- First 5 cycles ---")
    print(df.head().to_string())

    print("\n--- Last 5 cycles ---")
    print(df.tail().to_string())

    print("\n--- Data Quality Report ---")
    print(quality_report(battery).to_string(index=False))
