"""
Severson 2019 (Nature Energy) dataset loader.

Download from: https://data.matr.io/1/ (Severson et al., Nature Energy 2019).
Place batch1.pkl, batch2.pkl, batch3.pkl in data/severson/

Dataset structure (Python pickle, list of cell dicts):
    {
      'summary': {
        'QD':    np.array,   # discharge capacity per cycle (Ah)
        'IR':    np.array,   # internal resistance (Ohm)
        'Tavg':  np.array,   # average temperature (°C)
        'cycle': np.array,   # cycle numbers
      },
      'cycles': {
        '2': {'Qd': array, 'V': array, 't': array, ...},
        ...
      },
      'barcode': str,
    }
"""

import os
import pickle
import numpy as np
import pandas as pd


def load_severson_batch(pkl_path: str) -> dict[str, pd.DataFrame]:
    """
    Load one Severson batch pickle file.

    Parameters
    ----------
    pkl_path : str
        Absolute or relative path to a batch*.pkl file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of cell_id -> DataFrame with columns:
        cycle_number, capacity_ah, resistance_ohm, temperature_c.
        Cells with fewer than 100 cycles are skipped.
    """
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    cells: dict[str, pd.DataFrame] = {}

    for i, cell in enumerate(batch):
        # Determine cell ID
        barcode = cell.get("barcode", None)
        if barcode:
            cell_id = f"SEV_{barcode}"
        else:
            cell_id = f"SEV_{i:03d}"

        summary = cell.get("summary", {})

        # Extract cycle numbers
        cycle_arr = summary.get("cycle", None)
        if cycle_arr is None or len(cycle_arr) < 100:
            continue

        n = len(cycle_arr)

        # Capacity (QD)
        qd = summary.get("QD", None)
        capacity_ah = np.array(qd, dtype=float) if qd is not None else np.full(n, np.nan)

        # Internal resistance (IR)
        ir = summary.get("IR", None)
        resistance_ohm = np.array(ir, dtype=float) if ir is not None else np.full(n, np.nan)

        # Temperature (Tavg)
        tavg = summary.get("Tavg", None)
        temperature_c = np.array(tavg, dtype=float) if tavg is not None else np.full(n, np.nan)

        # Align lengths (some arrays may differ by ±1 row)
        min_len = min(n, len(capacity_ah), len(resistance_ohm), len(temperature_c))
        df = pd.DataFrame({
            "cycle_number":  np.array(cycle_arr[:min_len], dtype=int),
            "capacity_ah":   capacity_ah[:min_len],
            "resistance_ohm": resistance_ohm[:min_len],
            "temperature_c": temperature_c[:min_len],
        })

        if len(df) < 100:
            continue

        cells[cell_id] = df

    return cells


def load_severson_directory(directory: str) -> dict[str, pd.DataFrame]:
    """
    Load all batch*.pkl files found in `directory`.

    Parameters
    ----------
    directory : str
        Path to a directory containing one or more .pkl files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Merged mapping of all cell_id -> DataFrame across all batches.
    """
    all_cells: dict[str, pd.DataFrame] = {}

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            batch_cells = load_severson_batch(fpath)
            all_cells.update(batch_cells)
            print(f"  [severson] Loaded {len(batch_cells)} cells from {fname}")
        except Exception as exc:
            print(f"  [severson] Warning: could not load {fname}: {exc}")

    return all_cells
