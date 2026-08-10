"""
Severson 2019 LFP dataset loader.

Reference: Severson et al., "Data-driven prediction of battery cycle life
before capacity degradation", Nature Energy 2019.
Data: https://data.matr.io/1/  (research use)
Citation/license: see batlab.cite.cite(dataset="severson2019")

Downloads Batch 1 of the MATLAB file (~2.9 GB) on first call, extracts
per-cycle discharge capacity, resistance, and temperature for 12
representative cells, and caches them as CSVs in data/raw/severson/. Returns
the standardized batlab cycle schema (see batlab.datasets.schema).

Cell selection spans 4 cycle-life bands:
  Short  (<500 cy):   b1c2, b1c3, b1c4
  Medium (500-900):   b1c7, b1c8, b1c9
  Long   (900-1200):  b1c13, b1c14, b1c15
  Extra  (>1200):     b1c26, b1c27, b1c28
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import requests

from batlab.datasets._integrity import verify_sha256
from batlab.datasets.schema import compute_soh_pct

_BATCH1_URL = "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download"

# SHA-256 of the file _BATCH1_URL served when last verified against a real
# download (2026-07 — see batlab/datasets/_integrity.py for why this is
# checked). Batch 1's full HDF5 .mat is ~2.9 GB (all cells in the batch,
# not just the 12 this loader extracts), not the "~115 MB" this module's
# docstring used to claim.
_EXPECTED_SHA256 = "9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba"

_CELL_KEYS = [
    "b1c2",  "b1c3",  "b1c4",
    "b1c7",  "b1c8",  "b1c9",
    "b1c13", "b1c14", "b1c15",
    "b1c26", "b1c27", "b1c28",
]
# 0-based index of each cell in the batch HDF5 struct array (b1cN → index N-1)
_CELL_INDICES = {k: int(k[3:]) - 1 for k in _CELL_KEYS}

# batlab/datasets/severson.py -> repo root is three levels up.
_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "severson"
SEVERSON_CELL_IDS: list[str] = [f"S-{k}" for k in _CELL_KEYS]

CHEMISTRY = "LFP"


def _csv_path(k: str) -> pathlib.Path:
    return _RAW_DIR / f"{k}_summary.csv"


def _all_cached() -> bool:
    return all(_csv_path(k).exists() for k in _CELL_KEYS)


def any_cached() -> bool:
    """Return True if at least one cell CSV exists locally."""
    return any(_csv_path(k).exists() for k in _CELL_KEYS)


def _download_and_cache(status_fn=None) -> None:
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py required to parse the Severson dataset: pip install 'batlab[severson]'"
        )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    mat_path = _RAW_DIR / "batch1.mat"

    if not mat_path.exists():
        if status_fn:
            status_fn("Downloading Severson 2019 Batch 1 (~2.9 GB, one-time)…")
        resp = requests.get(_BATCH1_URL, stream=True, timeout=300)
        resp.raise_for_status()
        with open(mat_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if status_fn:
        status_fn("Verifying download integrity…")
    verify_sha256(mat_path, _EXPECTED_SHA256, "Severson 2019 Batch 1 .mat")

    if status_fn:
        status_fn("Parsing Severson cell summaries…")

    with h5py.File(mat_path, "r") as f:
        batch   = f["batch"]
        summ_ds = batch["summary"]   # (N_cells, 1) array of HDF5 object refs

        for key, idx in _CELL_INDICES.items():
            try:
                summ = f[summ_ds[idx, 0]]   # per-cell summary Group

                qd = np.array(summ["QDischarge"]).flatten().astype(float)
                # Remove cycle-0 pre-charge row if present
                if qd[0] == 0:
                    qd = qd[1:]
                n = len(qd)
                if n == 0:
                    continue

                cyc_raw = np.array(summ["cycle"]).flatten().astype(float)
                cycles  = cyc_raw[cyc_raw > 0] if cyc_raw[0] == 0 else cyc_raw
                cycles  = cycles[:n]
                if len(cycles) < n:
                    cycles = np.arange(1, n + 1, dtype=float)

                ir   = np.array(summ["IR"]).flatten().astype(float)[:n]   if "IR"   in summ else np.full(n, np.nan)
                tavg = np.array(summ["Tavg"]).flatten().astype(float)[:n] if "Tavg" in summ else np.full(n, 30.0)
                q0   = float(qd[0]) if float(qd[0]) > 0 else 1.0

                pd.DataFrame({
                    "cycle_number":   cycles.astype(int),
                    "capacity_ah":    qd,
                    "soh_pct":        qd / q0 * 100.0,
                    "resistance_ohm": ir,
                    "temperature_c":  tavg,
                }).to_csv(_csv_path(key), index=False)
            except (KeyError, TypeError, IndexError, OSError) as e:
                print(f"  [severson] skipping {key}: {e}")
                continue


def _load_cached(key: str) -> pd.DataFrame | None:
    path = _csv_path(key)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 5:
        return None
    cell_id = f"S-{key}"
    df.attrs["cell_id"] = cell_id
    df.attrs["source"] = "severson2019"
    df.attrs["chemistry"] = CHEMISTRY
    df.attrs["citation"] = "severson2019"
    df.attrs["license"] = "Research use per data.matr.io terms"
    # Protocol-known test conditions, verified against Severson et al. 2019
    # (Nature Energy 4, 383-391): discharge cutoff 2.0V (uniform across all
    # 124 cells) and 30C constant chamber temperature. voltage_charge_cutoff_v
    # is deliberately NOT set — cells were charged 0-80% SOC under one of 72
    # distinct multi-step fast-charging policies, not a single CC/CV voltage
    # cutoff, so no one number would honestly represent the protocol (see
    # condition_completeness()'s severson2019 caveat in batlab.datasets.schema).
    df.attrs["voltage_discharge_cutoff_v"] = 2.0
    df.attrs["test_temperature_c"] = 30.0
    return df


def download_and_prepare(status_fn=None) -> bool:
    """Download Batch 1, extract CSVs, and return True on success.
    Call this once locally, then commit data/raw/severson/ to the repo.
    """
    try:
        _download_and_cache(status_fn=status_fn or print)
        return _all_cached()
    except Exception as exc:
        print(f"[severson] Failed: {exc}")
        return False


def load_severson_cells(status_fn=None) -> dict[str, pd.DataFrame]:
    """
    Download-and-cache on first call, then load from CSV.

    Returns {cell_id: DataFrame} satisfying batlab.datasets.schema's
    kind="cycle" contract, with df.attrs set. Returns {} on failure.
    """
    if not _all_cached():
        try:
            _download_and_cache(status_fn=status_fn)
        except Exception as exc:
            print(f"[severson] Download failed — skipping real data: {exc}")
            return {}
    cells = {}
    for key in _CELL_KEYS:
        df = _load_cached(key)
        if df is not None:
            cells[df.attrs["cell_id"]] = df
    return cells


if __name__ == "__main__":
    print("Downloading Severson 2019 Batch 1 and extracting cell CSVs…")
    ok = download_and_prepare(status_fn=print)
    if ok:
        print(f"Done — CSVs written to {_RAW_DIR}")
        print("Commit data/raw/severson/ to the repo to enable Severson mode on Streamlit Cloud.")
    else:
        print("Download failed — see errors above.")
