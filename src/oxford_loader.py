"""
Oxford "Path Dependent Battery Degradation Dataset" loader (2020).

Reference: Raj, T. et al., "Investigation of Path Dependent Degradation in
Lithium-Ion Batteries", Batteries & Supercaps (Wiley, 2020).
Data: https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e
License: ODC Open Database License (ODC-ODbL).

Cells are 3 Ah NCR18650BD cylindrical cells with NCA (nickel cobalt
aluminium oxide) positive electrodes — genuinely new chemistry for this
app versus NASA's LiCoO2 and Severson's LFP. Only Group 1 (3 cells: 9,
15, 20; ~822 MB compressed) is used here, not all 4 groups (~3.1 GB) —
deliberately scoped down to keep the one-time local extraction tractable.

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

_GROUP1_URL = "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/dwh246s15b"

_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "oxford"
_ZIP_PATH = _RAW_DIR / "Group_1.zip"

OXFORD_CELL_IDS: list[str] = ["OX-9", "OX-15", "OX-20"]

# Per-cell TPG-file-index -> procedure label, transcribed from the dataset's
# own Guide_to_Datafiles.pdf. Only "BoL RPT"/"RPTn" indices are reference
# performance tests (real full discharges); "Profile"/"CCCV"/"Tester
# Malfunction" indices are daily drive-cycle stress segments, out of scope.
_PROCEDURE_LABELS: dict[int, dict[int, str]] = {
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


def _find_zip_entry(names: list[str], cell: int, tpg_index: int) -> "str | None":
    """
    Filenames inside the zip are inconsistently spaced/hyphenated (e.g.
    "TPG1 - Cell 9.mat" for index 1, "TPG1.4-Cell 9.mat" for index 4,
    "TPG1.10-Cell9.mat" for index 10) — match by parsing the numeric
    suffix rather than exact string matching.
    """
    pattern = re.compile(rf"TPG1(?:\.(\d+))?\s*-?\s*Cell\s*{cell}(?!\d)\.mat$")
    for name in names:
        m = pattern.search(name)
        if m:
            idx = int(m.group(1)) if m.group(1) else 1
            if idx == tpg_index:
                return name
    return None


def _download_and_cache(status_fn=None) -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not _ZIP_PATH.exists():
        if status_fn:
            status_fn("Downloading Oxford Path-Dependent Group 1 (~822 MB, one-time)…")
        resp = requests.get(_GROUP1_URL, stream=True, timeout=600)
        resp.raise_for_status()
        with open(_ZIP_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if status_fn:
        status_fn("Parsing Oxford NCA reference-test checkpoints…")

    with zipfile.ZipFile(_ZIP_PATH) as z:
        names = z.namelist()
        for cell in _PROCEDURE_LABELS:
            labels = _PROCEDURE_LABELS[cell]
            indices = _reference_test_indices(cell)
            rows = []
            for checkpoint_index, tpg_index in enumerate(indices):
                entry = _find_zip_entry(names, cell, tpg_index)
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


def download_and_prepare(status_fn=None) -> bool:
    """Download Group 1, extract checkpoint CSVs, and return True on success.
    Call this once locally, then commit data/raw/oxford/*.csv to the repo —
    the ~822 MB zip itself is never committed (gitignored)."""
    try:
        _download_and_cache(status_fn=status_fn or print)
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
    print("Downloading Oxford Path-Dependent Group 1 and extracting checkpoint CSVs…")
    ok = download_and_prepare(status_fn=print)
    if ok:
        print(f"Done — CSVs written to {_RAW_DIR}")
        print("Commit data/raw/oxford/*.csv to the repo to enable the Reference Datasets view.")
    else:
        print("Extraction failed — see errors above.")
