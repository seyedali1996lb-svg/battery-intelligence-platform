"""
Severson 2019 LFP dataset loader.

Reference: Severson et al., "Data-driven prediction of battery cycle life
before capacity degradation", Nature Energy 2019.
Data: https://data.matr.io/1/  (research use)

Downloads Batch 1 of the MATLAB file on first call, extracts per-cycle
discharge capacity, resistance, and temperature for 12 representative cells,
and caches them as CSVs in data/raw/severson/.

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

_BATCH1_URL = "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download"

_CELL_KEYS = [
    "b1c2",  "b1c3",  "b1c4",
    "b1c7",  "b1c8",  "b1c9",
    "b1c13", "b1c14", "b1c15",
    "b1c26", "b1c27", "b1c28",
]
# 0-based index of each cell in the batch HDF5 struct array (b1cN → index N-1)
_CELL_INDICES = {k: int(k[3:]) - 1 for k in _CELL_KEYS}

_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "severson"
SEVERSON_CELL_IDS: list[str] = [f"S-{k}" for k in _CELL_KEYS]


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
        raise ImportError("h5py required to parse Severson dataset: pip install h5py")

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    mat_path = _RAW_DIR / "batch1.mat"

    if not mat_path.exists():
        if status_fn:
            status_fn("Downloading Severson 2019 Batch 1 (~115 MB, one-time)…")
        resp = requests.get(_BATCH1_URL, stream=True, timeout=300)
        resp.raise_for_status()
        with open(mat_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if status_fn:
        status_fn("Parsing Severson cell summaries…")

    with h5py.File(mat_path, "r") as f:
        batch   = f["batch"]
        summ_ds = batch["summary"]   # (N_cells, 1) array of HDF5 object refs

        for key, idx in _CELL_INDICES.items():
            try:
                summ = f[summ_ds[idx, 0]]   # per-cell summary Group

                def _col(name, fallback=None, _s=summ):
                    if name in _s:
                        arr = np.array(_s[name]).flatten().astype(float)
                        # skip cycle 0 (pre-charge artefact)
                        return arr[arr > 0] if name == "QDischarge" else arr
                    return fallback

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


def _load_cached(key: str) -> dict | None:
    path = _csv_path(key)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 5:
        return None
    return {"cell_id": f"S-{key}", "source": "severson2019", "chemistry": "LFP", "cycles": df}


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


def load_severson_cells(status_fn=None) -> dict[str, dict]:
    """Download-and-cache on first call, then load from CSV. Returns {} on failure."""
    if not _all_cached():
        try:
            _download_and_cache(status_fn=status_fn)
        except Exception as exc:
            print(f"[severson] Download failed — skipping real data: {exc}")
            return {}
    cells = {}
    for key in _CELL_KEYS:
        c = _load_cached(key)
        if c:
            cells[c["cell_id"]] = c
    return cells


if __name__ == "__main__":
    print("Downloading Severson 2019 Batch 1 and extracting cell CSVs…")
    ok = download_and_prepare(status_fn=print)
    if ok:
        print(f"Done — CSVs written to {_RAW_DIR}")
        print("Commit data/raw/severson/ to the repo to enable Severson mode on Streamlit Cloud.")
    else:
        print("Download failed — see errors above.")
