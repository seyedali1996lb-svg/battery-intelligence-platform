"""
Oxford "Path Dependent Battery Degradation Dataset" loader (2020).

Reference: Raj, T. et al., "Investigation of Path Dependent Degradation in
Lithium-Ion Batteries", Batteries & Supercaps (Wiley, 2020).
Data: https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e
License: ODC Open Database License (ODC-ODbL).

Cells are 3 Ah NCR18650BD cylindrical cells with NCA (nickel cobalt
aluminium oxide) positive electrodes — genuinely new chemistry for this
app versus NASA's LiCoO2 and Severson's LFP. All 4 groups are used (12
cells total, ~3.1 GB compressed): Group 1 (cells 9/15/20, 1-day cycling
@ C/2), Group 2 (cells 3/4/8, 1-day cycling @ C/4), Group 3 (cells
10/11/14, 2-day cycling @ C/2), Group 4 (cells 12/18/19, 2-day cycling
@ C/4) — all 5 days calendar rest @ 90% SoC (10 days for Groups 3/4).
Each group is its own zip, downloaded and cached independently.

Unlike NASA/Severson, this dataset is NOT dense per-cycle data. Each cell
only has ~14 Reference Performance Test (RPT) checkpoints across its life
(2 "beginning of life" runs plus RPT1-RPT12), interspersed with daily
ARTEMIS drive-cycle "Profile" stress segments that this loader does not
parse. Because of that, this loader returns a sparse per-cell CHECKPOINT
curve (real measured capacity fade, ~14 points/cell) — not a per-cycle
DataFrame — and deliberately does NOT feed a GBRT+LCO model the way NASA/
Severson/synthetic data does: 14 points across 3 cells is too little to
honestly leave-cell-out-validate a predictive model. See app/_pages/
explore.py's "Reference Datasets" view, which shows this as a real
measured reference curve rather than a prediction.

Parsing note: the source .mat files are MATLAB `table` objects (MCOS
serialization), which scipy.io.loadmat cannot read at all. The `mat-io`
package's high-level table converter also fails on these specific files
(a version-metadata mismatch: "no field of name versionSavedFrom") and
falls back to a raw MatlabOpaque object — but that object's
`.properties["varnames"]`/`.properties["data"]` still expose the table's
column names and per-column arrays directly, which this module
reconstructs into a DataFrame manually. Verified against real downloaded
files from Group 1.
"""

from __future__ import annotations

import io
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd
import requests

_GROUP_URLS: dict[int, str] = {
    1: "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/dwh246s15b",
    2: "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/dt148fh185",
    3: "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/dff365528c",
    4: "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/d41687h50f",
}

_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "oxford"


def _zip_path(group: int) -> pathlib.Path:
    return _RAW_DIR / f"Group_{group}.zip"


# Which group each cell belongs to — determines both which zip to open and
# the "TPG{group}" filename prefix used inside that zip.
_CELL_GROUP: dict[int, int] = {
    9: 1, 15: 1, 20: 1,
    3: 2, 4: 2, 8: 2,
    10: 3, 11: 3, 14: 3,
    12: 4, 18: 4, 19: 4,
}

OXFORD_CELL_IDS: list[str] = [f"OX-{c}" for c in sorted(_CELL_GROUP)]

# Per-cell TPG-file-index -> procedure label, transcribed from the dataset's
# own Guide_to_Datafiles.pdf. Only "BoL RPT"/"RPTn" indices are reference
# performance tests (real full discharges); "Profile"/"CCCV"/"Tester
# Malfunction" indices are daily drive-cycle stress segments, out of scope.
_PROCEDURE_LABELS: dict[int, dict[int, str]] = {
    # ── Group 1 — 1-day cycling @ C/2, 5-day calendar @ 90% SoC ──
    9: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "Profile",
        7: "Profile", 8: "RPT2", 9: "RPT3", 10: "Profile", 11: "RPT4", 12: "Profile",
        13: "RPT5", 14: "Profile", 15: "RPT6", 16: "RPT7", 17: "Profile", 18: "Profile",
        19: "Profile", 20: "RPT8", 21: "RPT9", 22: "RPT10", 23: "RPT11", 24: "Profile",
        25: "RPT12",
    },
    15: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "RPT2",
        7: "RPT3", 8: "RPT4", 9: "Profile", 10: "RPT5", 11: "RPT6", 12: "RPT7",
        13: "Profile", 14: "RPT8", 15: "Profile", 16: "Profile", 17: "RPT9", 18: "Profile",
        19: "RPT10", 20: "RPT11", 21: "RPT12",
    },
    20: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "RPT2",
        7: "RPT3", 8: "RPT4", 9: "Profile", 10: "RPT5", 11: "RPT6", 12: "RPT7",
        13: "RPT8", 14: "Profile", 15: "Profile", 16: "RPT9", 17: "RPT10", 18: "RPT11",
        19: "RPT12",
    },
    # ── Group 2 — 1-day cycling @ C/4, 5-day calendar @ 90% SoC ──
    3: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "Profile", 8: "RPT2", 9: "Profile", 10: "Profile", 11: "RPT3", 12: "Profile",
        13: "RPT4", 14: "Profile", 15: "Profile", 16: "Profile", 17: "RPT5", 18: "Profile",
        19: "Profile", 20: "RPT6",
    },
    4: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "Profile", 8: "Profile", 9: "Profile", 10: "RPT2", 11: "Profile", 12: "RPT3",
        13: "Profile", 14: "Profile", 15: "Tester Malfunction", 16: "Tester Malfunction",
        17: "Profile", 18: "RPT4", 19: "Profile", 20: "Profile", 21: "Profile",
        22: "RPT5", 23: "Profile", 24: "Profile", 25: "RPT6",
    },
    8: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "Profile", 8: "RPT2", 9: "Profile", 10: "Profile", 11: "Profile", 12: "RPT3",
        13: "Profile", 14: "RPT4", 15: "Profile", 16: "Profile", 17: "Profile",
        18: "RPT5", 19: "Profile", 20: "Profile", 21: "RPT6",
    },
    # ── Group 3 — 2-day cycling @ C/2, 10-day calendar @ 90% SoC ──
    10: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "RPT2",
        7: "RPT3", 8: "RPT4", 9: "Profile", 10: "RPT5", 11: "RPT6", 12: "RPT7",
        13: "RPT8", 14: "Profile", 15: "RPT9", 16: "Profile", 17: "RPT10", 18: "RPT11",
        19: "RPT12",
    },
    11: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "RPT2",
        7: "RPT3", 8: "RPT4", 9: "Profile", 10: "RPT5", 11: "RPT6", 12: "RPT7",
        13: "RPT8", 14: "Profile", 15: "RPT9", 16: "Profile", 17: "RPT10", 18: "RPT11",
        19: "RPT12",
    },
    14: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "RPT1", 5: "Profile", 6: "RPT2",
        7: "RPT3", 8: "RPT4", 9: "Profile", 10: "RPT5", 11: "RPT6", 12: "RPT7",
        13: "RPT8", 14: "RPT9", 15: "Profile", 16: "RPT10", 17: "RPT11", 18: "RPT12",
    },
    # ── Group 4 — 2-day cycling @ C/4, 10-day calendar @ 90% SoC ──
    12: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "RPT2", 8: "Profile", 9: "Profile", 10: "RPT3", 11: "Profile", 12: "RPT4",
        13: "Profile", 14: "Profile", 15: "RPT5", 16: "Profile", 17: "Profile", 18: "RPT6",
    },
    18: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "RPT2", 8: "Profile", 9: "Profile", 10: "RPT3", 11: "Profile", 12: "RPT4",
        13: "Profile", 14: "Profile", 15: "RPT5", 16: "Profile", 17: "Profile", 18: "RPT6",
    },
    19: {
        1: "BoL", 2: "BoL", 3: "Profile", 4: "Profile", 5: "RPT1", 6: "Profile",
        7: "Profile", 8: "RPT2", 9: "Profile", 10: "Profile", 11: "Profile", 12: "Profile",
        13: "Profile", 14: "Profile", 15: "RPT3", 16: "Profile", 17: "Profile", 18: "RPT4",
        19: "Profile", 20: "Profile", 21: "Profile", 22: "Profile", 23: "Profile",
        24: "RPT5", 25: "Profile", 26: "Profile", 27: "RPT6",
    },
}


def _reference_test_indices(cell: int) -> list[int]:
    """File indices for this cell whose procedure is a real reference discharge
    (BoL or RPTn), in chronological order — the checkpoints this loader extracts."""
    labels = _PROCEDURE_LABELS[cell]
    return [i for i in sorted(labels) if labels[i].startswith(("BoL", "RPT"))]


def _csv_path(cell: int) -> pathlib.Path:
    return _RAW_DIR / f"cell{cell}_checkpoints.csv"


def any_cached() -> bool:
    """Return True if at least one cell's checkpoint CSV exists locally."""
    return any(_csv_path(c).exists() for c in _PROCEDURE_LABELS)


def _all_cached() -> bool:
    return all(_csv_path(c).exists() for c in _PROCEDURE_LABELS)


def _capacity_ah_from_table(df: pd.DataFrame) -> float:
    """
    Extract the reference discharge capacity from one RPT/BoL test's raw
    Maccor time series. The full-capacity reference discharge is whichever
    Step has the largest cumulative Amphr swing in this file — robust to
    step-numbering drift between files (verified: Step numbering shifted
    between BoL and RPT1 files for the same cell, but the max-delta step
    was still the correct physical discharge in both).
    """
    best_delta = 0.0
    for _step, group in df.groupby("Step"):
        delta = float(group["Amphr"].max() - group["Amphr"].min())
        if delta > best_delta:
            best_delta = delta
    return best_delta


def _add_soh_column(df: pd.DataFrame) -> pd.DataFrame:
    """SOH% relative to this cell's own first checkpoint (checkpoint_index 0,
    its beginning-of-life reference discharge) — each cell is its own baseline,
    same convention as every other loader in this app."""
    q0 = float(df["capacity_ah"].iloc[0]) or 1.0
    df["soh_pct"] = df["capacity_ah"] / q0 * 100.0
    return df


def _table_from_mat_bytes(mat_bytes: bytes) -> pd.DataFrame:
    """
    Reconstruct a MATLAB `table` object's DataFrame from a .mat file's bytes.
    mat-io's high-level converter succeeds on some of these files (returning
    a DataFrame directly) but fails on others with a version-metadata
    mismatch, falling back to a raw MatlabOpaque object instead — both cases
    are handled here (see module docstring for why the fallback is needed).
    """
    from matio import load_from_mat

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(mat_bytes)
        tmp_path = tmp.name
    try:
        data = load_from_mat(tmp_path, raw_data=False)
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    obj = next(iter(data.values()))
    if isinstance(obj, pd.DataFrame):
        return obj

    props = obj.properties
    varnames = [str(v[0]) for v in props["varnames"][0]]
    cols_raw = props["data"][0]
    return pd.DataFrame({name: col.flatten() for name, col in zip(varnames, cols_raw)})


def _find_zip_entry(names: list[str], group: int, cell: int, tpg_index: int) -> "str | None":
    """
    Filenames inside the zip are inconsistently spaced/hyphenated (e.g.
    "TPG1 - Cell 9.mat" for index 1, "TPG1.4-Cell 9.mat" for index 4,
    "TPG1.10-Cell9.mat" for index 10) — match by parsing the numeric
    suffix rather than exact string matching. The "TPG{group}" prefix
    matches the zip a cell's own group was downloaded into.
    """
    pattern = re.compile(rf"TPG{group}(?:\.(\d+))?\s*-?\s*Cell\s*{cell}(?!\d)\.mat$")
    for name in names:
        m = pattern.search(name)
        if m:
            idx = int(m.group(1)) if m.group(1) else 1
            if idx == tpg_index:
                return name
    return None


def _download_group_zip(group: int, status_fn=None) -> None:
    zip_path = _zip_path(group)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        return
    if status_fn:
        status_fn(f"Downloading Oxford Path-Dependent Group {group} (one-time)…")
    resp = requests.get(_GROUP_URLS[group], stream=True, timeout=600)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _extract_group(group: int, status_fn=None) -> None:
    cells_in_group = [c for c, g in _CELL_GROUP.items() if g == group]
    with zipfile.ZipFile(_zip_path(group)) as z:
        names = z.namelist()
        for cell in cells_in_group:
            labels = _PROCEDURE_LABELS[cell]
            indices = _reference_test_indices(cell)
            rows = []
            for checkpoint_index, tpg_index in enumerate(indices):
                entry = _find_zip_entry(names, group, cell, tpg_index)
                if entry is None:
                    continue
                if status_fn:
                    status_fn(f"  Cell {cell}: {labels[tpg_index]} ({entry})")
                try:
                    mat_bytes = z.read(entry)
                    table = _table_from_mat_bytes(mat_bytes)
                    capacity_ah = _capacity_ah_from_table(table)
                except Exception as e:
                    print(f"  [oxford] skipping cell {cell} index {tpg_index}: {e}")
                    continue
                rows.append({
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_label": labels[tpg_index],
                    "capacity_ah": capacity_ah,
                })
            if not rows:
                continue
            df = _add_soh_column(pd.DataFrame(rows))
            df.to_csv(_csv_path(cell), index=False)


def _download_and_cache(status_fn=None, groups: "list[int] | None" = None) -> None:
    """Download and extract each requested group's zip in turn (default: all 4).
    Each group's zip is only downloaded once (skipped if already on disk) and
    can be deleted again after extraction — only the small per-cell CSVs are
    committed to the repo."""
    for group in (groups or sorted(_GROUP_URLS)):
        _download_group_zip(group, status_fn=status_fn)
        if status_fn:
            status_fn(f"Parsing Oxford NCA reference-test checkpoints — Group {group}…")
        _extract_group(group, status_fn=status_fn)


def _load_cached(cell: int) -> "dict | None":
    path = _csv_path(cell)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 3:
        return None
    return {
        "cell_id": f"OX-{cell}",
        "source": "oxford_pathdep_2020",
        "chemistry": "NCA",
        "checkpoints": df,
    }


def download_and_prepare(status_fn=None, groups: "list[int] | None" = None) -> bool:
    """Download the requested group(s) (default: all 4), extract checkpoint
    CSVs, and return True on success. Call this once locally per group, then
    commit data/raw/oxford/*.csv to the repo — the zips themselves are never
    committed (gitignored)."""
    try:
        _download_and_cache(status_fn=status_fn or print, groups=groups)
        return _all_cached()
    except Exception as exc:
        print(f"[oxford] Failed: {exc}")
        return False


def load_oxford_cells() -> dict[str, dict]:
    """
    Load from cached CSVs only — deliberately does NOT auto-download the
    822 MB source zip (unlike severson_loader.load_severson_cells(), which
    auto-downloads its much smaller 115 MB file). Returns {} if nothing is
    cached yet; run `python src/oxford_loader.py` locally once to populate
    data/raw/oxford/.
    """
    cells = {}
    for cell in _PROCEDURE_LABELS:
        c = _load_cached(cell)
        if c:
            cells[c["cell_id"]] = c
    return cells


if __name__ == "__main__":
    import sys as _sys
    _groups = [int(g) for g in _sys.argv[1:]] or None
    print(f"Downloading Oxford Path-Dependent group(s) {_groups or 'all'} and extracting checkpoint CSVs…")
    ok = download_and_prepare(status_fn=print, groups=_groups)
    if ok:
        print(f"Done — CSVs written to {_RAW_DIR}")
        print("Commit data/raw/oxford/*.csv to the repo to enable the Reference Datasets view.")
    else:
        print("Extraction failed — see errors above.")
