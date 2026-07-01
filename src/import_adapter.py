"""
Piece 3 — Upload adapter: transforms a validated user DataFrame into the
Battery → Cells → Cycles schema that the existing pipeline expects.

CONSTRAINT: this module calls enrich_cycles() from data_loader — the same
function used for NASA and synthetic cells. It does NOT re-implement SOH
calculation or any other pipeline logic. The pipeline is the source of truth.

Output schema matches data_loader.build_battery() exactly:
{
    "battery_id": str,
    "cells": {
        cell_id: {
            "cell_id":              str,
            "cycles":               pd.DataFrame,   # enriched — same columns as pipeline
            "temperature_assumed":  bool,           # True when temp_c was missing/blank
        }
    },
    "temperature_assumed_cells": [cell_id, ...],   # convenience list
    "source": "uploaded",                          # distinguishes from "synth"/"nasa"
}
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import enrich_cycles


def adapt_upload_to_pipeline(validated_df: pd.DataFrame) -> dict:
    """
    Transform a validated upload DataFrame into a battery dict that the
    existing training pipeline (_train_on_cells, run_lco, etc.) can consume.

    validated_df: output of validate_upload() — already confirmed to have
    required columns, correct types, ≥2 unique cells, no duplicates.
    Temperature may be missing; handled by filling 25.0 and flagging.

    Returns the battery dict described in the module docstring.
    """
    df = validated_df.copy()

    # Ensure correct numeric types (validate_upload already confirmed castable)
    df["cycle_number"]   = pd.to_numeric(df["cycle_number"]).astype(int)
    df["capacity_ah"]    = pd.to_numeric(df["capacity_ah"]).astype(float)
    df["resistance_ohm"] = pd.to_numeric(df["resistance_ohm"]).astype(float)

    has_temp_col = "temperature_c" in df.columns
    if has_temp_col:
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")

    cells: dict = {}
    temperature_assumed_cells: list = []

    for cell_id in df["cell_id"].unique():
        cell_df = (
            df[df["cell_id"] == cell_id]
            .copy()
            .sort_values("cycle_number")
            .reset_index(drop=True)
        )

        # ── Temperature handling ──────────────────────────────────────────
        if not has_temp_col:
            # Column entirely absent
            cell_df["temperature_c"] = 25.0
            temp_assumed = True
        else:
            n_missing_temp = cell_df["temperature_c"].isna().sum()
            cell_df["temperature_c"] = cell_df["temperature_c"].fillna(25.0)
            # Flag as assumed if ALL values were missing for this cell
            # (partial fill is still flagged as partial — keep honest)
            temp_assumed = n_missing_temp > 0

        if temp_assumed:
            temperature_assumed_cells.append(cell_id)

        # ── Build cycle DataFrame — pipeline-expected columns only ────────
        cycle_df = cell_df[["cycle_number", "capacity_ah",
                             "resistance_ohm", "temperature_c"]].copy()

        # ── Enrich: SOH, fade, EOL — identical call as data_loader ───────
        # capacity_ah / first_cycle_capacity × 100 — same normalisation.
        # Do NOT reimplement: call enrich_cycles() directly.
        enriched = enrich_cycles(cycle_df)

        cells[cell_id] = {
            "cell_id":             cell_id,
            "cycles":              enriched,
            "temperature_assumed": temp_assumed,
        }

    return {
        "battery_id":                "Upload_B1",
        "cells":                     cells,
        "temperature_assumed_cells": temperature_assumed_cells,
        "source":                    "uploaded",
    }


def summarise_adapted(battery: dict) -> str:
    """Human-readable summary of an adapted battery dict for verification."""
    lines = [f"battery_id: {battery['battery_id']}",
             f"source: {battery['source']}",
             f"n_cells: {len(battery['cells'])}",
             f"temperature_assumed_cells: {battery['temperature_assumed_cells']}",
             ""]
    for cid, cell in battery["cells"].items():
        df = cell["cycles"]
        lines.append(f"  {cid}:")
        lines.append(f"    cycles: {len(df)}")
        lines.append(f"    columns: {list(df.columns)}")
        lines.append(f"    capacity_ah: {df['capacity_ah'].min():.3f} – {df['capacity_ah'].max():.3f}")
        lines.append(f"    soh_pct:     {df['soh_pct'].min():.1f} – {df['soh_pct'].max():.1f}")
        lines.append(f"    resistance:  {df['resistance_ohm'].min():.4f} – {df['resistance_ohm'].max():.4f}")
        lines.append(f"    temperature: {df['temperature_c'].min():.1f} – {df['temperature_c'].max():.1f}°C")
        lines.append(f"    temp_assumed: {cell['temperature_assumed']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick verification against the import template
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "import_template.csv"
    )
    df = pd.read_csv(template_path)
    print("Raw template CSV:")
    print(df.to_string(index=False))
    print()

    battery = adapt_upload_to_pipeline(df)
    print("Adapted battery dict:")
    print(summarise_adapted(battery))
    print()

    for cid, cell in battery["cells"].items():
        print(f"Enriched cycles for {cid}:")
        print(cell["cycles"].to_string(index=False))
        print()
