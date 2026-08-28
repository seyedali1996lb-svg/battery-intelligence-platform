"""
Zhu et al. 2022 voltage-relaxation dataset loader (NCM+NCA family).

Reference: Zhu, J. et al., "Data-driven capacity estimation of commercial
lithium-ion batteries from voltage relaxation", Nature Communications 13,
2261 (2022). DOI: 10.1038/s41467-022-29837-w
Data: https://zenodo.org/records/6405084 (CC BY 4.0)
Citation/license: see batlab.cite.cite(dataset="zhu2022")

This loader covers Dataset_3 of the Zenodo record: 9 commercial 18650 cells
with a blended NCM+NCA cathode, cycled at 25 °C at a 0.5C charge / ~1C
discharge protocol (the cell-name convention "CY25-05_1-#1" decodes as
CY{temperature}-{rate}_{condition}-#{replicate}, per the dataset's own
description). Each cell's raw file is a dense voltage/current time series
with cumulative per-cycle charge/discharge throughput columns.

Per-cycle discharge capacity is derived from the source's `Q discharge/mA.h`
column, which resets at each cycle boundary and accumulates only while the
cell is discharging (current < 0). A handful of special cycles in the raw
files (reference/characterization tests) contain more than one discharge
segment, so a cycle's capacity is taken as the charge transferred in the
cycle's LARGEST single contiguous discharge run (`max over runs of [Q at
run end - Q at run start]`, mAh → Ah) — the naive per-cycle maximum would
inflate those special cycles (e.g. 3.58 Ah vs. the real ~2.42 Ah for one
cell's characterization cycle). The loader caches the derived per-cycle
summaries as small CSVs (one row per cycle) in data/raw/zhu2022/ — the raw
~356 MB zip is auto-downloaded once and never committed (see
batlab.datasets._integrity for the checksum policy).

Verified against a real download (2026-08): all 9 cells yield ~910-1030
cycles with first-cycle capacity ~2.47-2.50 Ah fading to ~62-67% SOH —
dense enough for the GBRT + leave-cell-out pipeline, and a genuinely new
chemistry for this library (blended NCM+NCA, vs. NASA/CALCE LiCoO2 and
Severson LFP).
"""

from __future__ import annotations
import pathlib
import zipfile

import numpy as np
import pandas as pd
import requests

from batlab.datasets._integrity import verify_sha256
from batlab.datasets.schema import compute_soh_pct

# Zenodo record 6405084, Dataset_3 (NCM+NCA blend) — the smallest of the
# record's three chemistry zips. Stable, ungated, CC BY 4.0.
_ZIP_URL = "https://zenodo.org/records/6405084/files/Dataset_3_NCM_NCA_battery.zip?download=1"

# SHA-256 of the exact bytes _ZIP_URL served when last verified (2026-08,
# during this loader's build — the file was downloaded in full and hashed).
# The archive's own published md5 (d11e68e410a638058906af5e2f5f60f3, from
# the Zenodo API) matched the download as well, so both are recorded here.
_EXPECTED_SHA256 = "a82e33ada100f91ab6cf99a654bb9f0572ce48ee7edbc5dad9d0665e7eccfe3a"
_ZENODO_MD5 = "d11e68e410a638058906af5e2f5f60f3"

# The 9 cell files inside the zip (3 conditions × 3 replicates, all 25 °C,
# 0.5C per the dataset's own naming description).
_CELL_FILES = [
    "CY25-05_1-#1.csv", "CY25-05_1-#2.csv", "CY25-05_1-#3.csv",
    "CY25-05_2-#1.csv", "CY25-05_2-#2.csv", "CY25-05_2-#3.csv",
    "CY25-05_4-#1.csv", "CY25-05_4-#2.csv", "CY25-05_4-#3.csv",
]
_ZIP_MEMBER_PREFIX = "Dataset_3_NCM_NCA_battery/"

# batlab/datasets/zhu2022.py -> repo root is three levels up.
_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "zhu2022"
_EXTRACT_DIR = _RAW_DIR / "Dataset_3_NCM_NCA_battery"

CHEMISTRY = "NCM+NCA"

# Physical plausibility gate for derived capacity: an 18650 NCM+NCA cell in
# this dataset is ~2.5 Ah nominal. A derived first-cycle capacity outside a
# broad 1.5-4.0 Ah band means the raw file was misparsed, and that cell is
# dropped rather than returned with garbage numbers.
_PLAUSIBLE_FIRST_CAPACITY_AH = (1.5, 4.0)

# A cycle with a derived discharge capacity at or below this (Ah) is treated
# as a partial/zero-discharge cycle (test interruptions, final partial
# cycle) and dropped — the source files end with such cycles.
_MIN_CYCLE_CAPACITY_AH = 0.5


def _zip_path() -> pathlib.Path:
    return _RAW_DIR / "Dataset_3_NCM_NCA_battery.zip"


def _summary_path(stem: str, raw_dir: pathlib.Path | None = None) -> pathlib.Path:
    return (raw_dir or _RAW_DIR) / f"{stem}_summary.csv"


def _all_summaries_present(raw_dir: pathlib.Path | None = None) -> bool:
    return all(_summary_path(f.removesuffix(".csv"), raw_dir).exists() for f in _CELL_FILES)


def any_cached(raw_dir: pathlib.Path | None = None) -> bool:
    """Return True if at least one per-cell summary CSV exists locally."""
    return any(_summary_path(f.removesuffix(".csv"), raw_dir).exists() for f in _CELL_FILES)


def derive_cell_summary(csv_path: str | pathlib.Path) -> pd.DataFrame | None:
    """
    Derive the standardized per-cycle summary from one raw Zhu 2022 CSV.

    Reads only the three needed columns ('Q discharge/mA.h', '<I>/mA',
    'cycle number'), takes each cycle's largest single discharge-run charge
    transfer (see the module docstring for why not the per-cycle maximum),
    drops partial/zero-discharge cycles, and computes `soh_pct` via the
    shared batlab.datasets.schema.compute_soh_pct() helper — never a
    re-implemented copy.

    Returns None if the file has too few valid cycles or a physically
    implausible first-cycle capacity (a misparse guard, not a filter on
    legitimate data).
    """
    df = pd.read_csv(csv_path, usecols=["Q discharge/mA.h", "<I>/mA", "cycle number"])
    cyc = df["cycle number"].to_numpy()
    qd = df["Q discharge/mA.h"].to_numpy()
    current = df["<I>/mA"].to_numpy()

    capacity_mah: dict[float, float] = {}
    for c in np.unique(cyc):
        idx = np.flatnonzero((cyc == c) & (current < 0))
        if len(idx) == 0:
            continue
        # Contiguous discharge rows (gap > 1 row = a new run). Rest/charge
        # rows are excluded by the current < 0 mask, so a rest between two
        # discharge segments correctly splits them into two runs.
        run_breaks = np.flatnonzero(np.diff(idx) > 1) + 1
        runs = np.split(idx, run_breaks)
        deltas = [qd[r[-1]] - qd[r[0]] for r in runs]
        capacity_mah[c] = float(max(deltas))

    capacity_ah = (pd.Series(capacity_mah) / 1000.0).astype(float)
    capacity_ah = capacity_ah[capacity_ah > _MIN_CYCLE_CAPACITY_AH]

    if len(capacity_ah) < 10:
        return None
    first = float(capacity_ah.iloc[0])
    if not (_PLAUSIBLE_FIRST_CAPACITY_AH[0] <= first <= _PLAUSIBLE_FIRST_CAPACITY_AH[1]):
        return None

    out = pd.DataFrame({
        "cycle_number": capacity_ah.index.astype(int),
        "capacity_ah": capacity_ah.values,
    })
    out["soh_pct"] = compute_soh_pct(out["capacity_ah"])
    return out.reset_index(drop=True)


def _finalize_cell(summary: pd.DataFrame, cell_stem: str) -> pd.DataFrame:
    """Attach the REQUIRED_ATTRS + protocol metadata every batlab loader sets."""
    df = summary.copy()
    df.attrs["cell_id"] = cell_stem
    df.attrs["source"] = "zhu2022"
    df.attrs["chemistry"] = CHEMISTRY
    df.attrs["citation"] = "zhu2022"
    df.attrs["license"] = "Creative Commons Attribution 4.0 (CC BY 4.0) — Zenodo record 6405084"
    # Protocol-known test condition, from the dataset's own description of
    # its cell-naming convention: CY25 = 25 °C chamber temperature. Voltage
    # cutoffs are NOT set — the published description documents temperature
    # and rate but no charge/discharge cutoffs (see condition_completeness()
    # in batlab/datasets/schema.py for the disclosed caveat).
    df.attrs["test_temperature_c"] = 25.0
    return df


def _download_and_extract(status_fn=None) -> None:
    """Download the ~356 MB zip (once), verify it, extract the 9 cell CSVs."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _zip_path()

    if not zip_path.exists():
        if status_fn:
            status_fn("Downloading Zhu 2022 Dataset_3 (NCM+NCA, ~356 MB, one-time)…")
        resp = requests.get(_ZIP_URL, stream=True, timeout=600)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if status_fn:
        status_fn("Verifying download integrity…")
    verify_sha256(zip_path, _EXPECTED_SHA256, "Zhu 2022 Dataset_3 (NCM+NCA) zip")

    _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for fname in _CELL_FILES:
            member = f"{_ZIP_MEMBER_PREFIX}{fname}"
            if member not in names:
                raise RuntimeError(f"Zhu 2022 zip missing expected member {member!r} — "
                                   "the upstream archive may have changed.")
            target = _EXTRACT_DIR / fname
            if not target.exists():
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def _cache_summaries(status_fn=None, raw_dir: pathlib.Path | None = None) -> None:
    """Derive + write the per-cell summary CSVs from the extracted raw files."""
    raw_dir = raw_dir or _RAW_DIR
    if status_fn:
        status_fn("Deriving per-cycle capacity summaries…")
    for fname in _CELL_FILES:
        stem = fname.removesuffix(".csv")
        summary = derive_cell_summary(_EXTRACT_DIR / fname)
        if summary is None:
            if status_fn:
                status_fn(f"  [zhu2022] skipping {stem}: no plausible cycle summary derivable")
            continue
        summary.to_csv(_summary_path(stem, raw_dir), index=False)


def download_and_prepare(status_fn=None, raw_dir: pathlib.Path | None = None) -> bool:
    """Download, extract, and cache all 9 per-cell summaries. Returns True on success.

    Call this once locally, then commit data/raw/zhu2022/*_summary.csv so the
    dataset loads instantly without a 356 MB download (the zip itself is
    gitignored).
    """
    try:
        _download_and_extract(status_fn=status_fn or print)
        _cache_summaries(status_fn=status_fn or print, raw_dir=raw_dir)
        return _all_summaries_present(raw_dir)
    except Exception as exc:
        print(f"[zhu2022] Failed: {exc}")
        return False


def _load_cached(stem: str, raw_dir: pathlib.Path | None = None) -> pd.DataFrame | None:
    path = _summary_path(stem, raw_dir)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 10:
        return None
    return _finalize_cell(df, stem)


def load_zhu2022_cells(status_fn=None, raw_dir: pathlib.Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Download-and-cache on first call, then load the 9 per-cell summaries.

    Returns {cell_id: DataFrame} satisfying batlab.datasets.schema's
    kind="cycle" contract, with df.attrs set (cell ids are the raw source
    stems, e.g. "CY25-05_1-#1"). Returns {} on failure.

    `raw_dir` overrides the cache directory. When a non-default `raw_dir`
    is passed, no network download is attempted — missing summaries simply
    return {} — so tests can point at fixture directories without touching
    the network.
    """
    if not _all_summaries_present(raw_dir):
        if raw_dir is not None:
            return {}
        try:
            _download_and_extract(status_fn=status_fn)
            _cache_summaries(status_fn=status_fn)
        except Exception as exc:
            print(f"[zhu2022] Download failed — skipping real data: {exc}")
            return {}
    cells = {}
    for fname in _CELL_FILES:
        stem = fname.removesuffix(".csv")
        df = _load_cached(stem, raw_dir)
        if df is not None:
            cells[df.attrs["cell_id"]] = df
    return cells


if __name__ == "__main__":
    print("Downloading Zhu 2022 Dataset_3 and deriving per-cell summaries…")
    ok = download_and_prepare(status_fn=print)
    if ok:
        print(f"Done — summaries written to {_RAW_DIR}")
        print("Commit data/raw/zhu2022/*_summary.csv to the repo to enable Zhu 2022 mode on Streamlit Cloud.")
    else:
        print("Download failed — see errors above.")
