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
import io, pathlib
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

_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "severson"
SEVERSON_CELL_IDS: list[str] = [f"S-{k}" for k in _CELL_KEYS]


def _csv_path(k: str) -> pathlib.Path:
    return _RAW_DIR / f"{k}_summary.csv"


def _all_cached() -> bool:
    return all(_csv_path(k).exists() for k in _CELL_KEYS)


def _download_and_cache(status_fn=None) -> None:
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy required to parse Severson dataset.")

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    if status_fn:
        status_fn("Downloading Severson 2019 Batch 1 (~115 MB, one-time)…")

    resp = requests.get(_BATCH1_URL, stream=True, timeout=300)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(chunk_size=1 << 20))

    if status_fn:
        status_fn("Parsing Severson cell summaries…")

    mat = scipy.io.loadmat(
        io.BytesIO(raw),
        squeeze_me=True, struct_as_record=False, simplify_cells=True,
    )
    batch = mat.get("batch", mat)

    for key in _CELL_KEYS:
        try:
            s = batch[key]["summary"]
            cycles = np.atleast_1d(s["cycle"]).astype(int)
            qd     = np.atleast_1d(s["QDischarge"]).astype(float)
            ir     = np.atleast_1d(s.get("IR",      np.full(len(cycles), np.nan))).astype(float)
            tavg   = np.atleast_1d(s.get("Tavg",    np.full(len(cycles), 30.0))).astype(float)
            q0     = float(qd[0]) if float(qd[0]) > 0 else 1.0
            pd.DataFrame({
                "cycle_number":   cycles,
                "capacity_ah":    qd,
                "soh_pct":        qd / q0 * 100.0,
                "resistance_ohm": ir,
                "temperature_c":  tavg,
            }).to_csv(_csv_path(key), index=False)
        except (KeyError, TypeError, IndexError):
            continue


def _load_cached(key: str) -> dict | None:
    path = _csv_path(key)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 5:
        return None
    return {"cell_id": f"S-{key}", "source": "severson2019", "chemistry": "LFP", "cycles": df}


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
