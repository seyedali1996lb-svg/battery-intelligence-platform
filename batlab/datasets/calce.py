"""
CALCE CS2 battery dataset loader (University of Maryland, Center for
Advanced Life Cycle Engineering).

CS2 cells are 1.1 Ah prismatic LiCoO2 cells, cycled to failure under
constant-current protocols, periodically at varying rates and temperatures.
This is one of the most-cited public li-ion cycling datasets in the
battery-ML literature, and the one this library did not already have a
loader for (a `calce_loader.py` was referenced in this repo's phase
history and later removed as dead code — this is a fresh implementation,
not a revival of that one).

Citation/license: see batlab.cite.cite(dataset="calce").

Manual download required
-------------------------
Unlike the NASA/Severson/Oxford loaders, this loader does NOT auto-download.
CALCE's battery-data page (https://calce.umd.edu/battery-data) serves the
CS2 files as per-cell zip archives (CS2_33.zip, CS2_34.zip, ...) without an
account/registration gate, but does not publish a stable, individually-
addressable per-file URL this library could script against without
breaking on any site restructure — and, same as every other batlab loader,
this project does not commit third-party raw data to the repo. Call
load_calce_cells() with cells already downloaded and placed locally (see
its docstring); if nothing is found, it raises CalceDataNotFoundError with
exact instructions on where to get the files and where to place them.

File format
-----------
CALCE cells were cycled on Arbin BT2000 testers; each test session is
published as one Excel workbook per cell (filename dated, e.g.
"CS2_35_1_9_11.xlsx"), containing a per-datapoint sheet ("Channel_*") and
often a per-cycle summary sheet ("Statistics_*"). This loader reads
whichever sheet is present and reduces it to one row per Cycle_Index via
groupby (taking the max of the cumulative-within-cycle Discharge_Capacity/
Charge_Capacity columns — the same convention every other loader in this
project uses for cycle-level capacity). Column names follow the standard
Arbin BT2000 export schema shared across the wider public battery-ML
literature (CALCE, HNEI, and other CALCE-adjacent datasets distribute
Arbin-derived files with this same schema) — this loader's exact column
names were NOT verified against a live downloaded CALCE file in the
environment that wrote it (no network access to CALCE's per-file URLs was
available); if a cell's real column names differ from what
_cycle_summary_from_raw() below looks for, please open an issue or send
a fixture file.

Cycle_Index resets to 1 in every new test-session file for the same cell
(a documented Arbin/CALCE quirk) — load_calce_cells() renumbers cycles
cumulatively across a cell's files, sorted by filename, so cycle_number
is monotonically increasing across the cell's full life, matching every
other batlab loader's convention.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from batlab.datasets.schema import compute_soh_pct

CALCE_INFO_URL = "https://calce.umd.edu/battery-data"

# batlab/datasets/calce.py -> repo root is three levels up.
_RAW_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "calce"

CHEMISTRY = "LiCoO2"

# Standard Arbin BT2000 export column names this loader looks for. See the
# module docstring's honesty note on why these are the *typical* Arbin
# names rather than independently verified against a live CALCE download.
# Matched by substring (case-insensitive) since Arbin's auxiliary-channel
# suffixes vary by test rig (e.g. "Aux_Temperature(C)_1", "Temperature (C)_1").
_TEMPERATURE_COL_HINTS = ("temperature",)
_RESISTANCE_COL_HINTS = ("internal_resistance",)


class CalceDataNotFoundError(RuntimeError):
    """Raised when no local CALCE cell data is found — always includes
    exact instructions on where to download it and where to place it."""


def any_cached() -> bool:
    """Return True if at least one cell's raw workbook directory exists locally."""
    return _RAW_DIR.exists() and any(_RAW_DIR.iterdir())


def _find_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for col in columns:
        low = col.lower()
        if any(h in low for h in hints):
            return col
    return None


def _cycle_summary_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure-logic core: reduce one workbook sheet's rows (whether already
    one-row-per-cycle or raw per-datapoint) to a per-cycle summary.

    Independently testable with a small synthetic DataFrame — no real
    Excel file required — matching this project's established pattern
    for pure-logic dataset-parsing pieces (see batlab.datasets.oxford's
    _capacity_ah_from_table for the same convention).
    """
    if "Cycle_Index" not in df.columns:
        raise CalceDataNotFoundError(
            f"Expected a 'Cycle_Index' column in a CALCE raw sheet, got: {list(df.columns)}. "
            "See batlab.datasets.calce's module docstring — column names were not verified "
            "against a live CALCE download."
        )
    if "Discharge_Capacity(Ah)" not in df.columns:
        raise CalceDataNotFoundError(
            f"Expected a 'Discharge_Capacity(Ah)' column in a CALCE raw sheet, got: {list(df.columns)}."
        )

    grouped = df.groupby("Cycle_Index", as_index=False)
    out = grouped["Discharge_Capacity(Ah)"].max().rename(
        columns={"Cycle_Index": "cycle_number", "Discharge_Capacity(Ah)": "capacity_ah"}
    )

    if "Charge_Capacity(Ah)" in df.columns:
        charge = grouped["Charge_Capacity(Ah)"].max()["Charge_Capacity(Ah)"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ce = out["capacity_ah"].to_numpy() / charge.replace(0, np.nan).to_numpy()
        out["coulombic_efficiency"] = ce

    temp_col = _find_column(list(df.columns), _TEMPERATURE_COL_HINTS)
    if temp_col:
        out["temperature_c"] = grouped[temp_col].mean()[temp_col].to_numpy()

    resistance_col = _find_column(list(df.columns), _RESISTANCE_COL_HINTS)
    if resistance_col:
        out["resistance_ohm"] = grouped[resistance_col].mean()[resistance_col].to_numpy()

    return out.sort_values("cycle_number").reset_index(drop=True)


def _read_workbook(path: pathlib.Path) -> pd.DataFrame:
    """Read one CALCE Arbin workbook and return its per-cycle summary
    (cycle_number starting at 1, NOT yet offset across files)."""
    sheets = pd.read_excel(path, sheet_name=None)  # requires openpyxl

    stats_sheet = next((n for n in sheets if "statistics" in n.lower()), None)
    channel_sheet = next((n for n in sheets if "channel" in n.lower()), None)
    sheet_name = stats_sheet or channel_sheet or next(iter(sheets))

    return _cycle_summary_from_raw(sheets[sheet_name])


def _sorted_cell_files(cell_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(cell_dir.glob("*.xlsx")) + sorted(cell_dir.glob("*.xls"))


def load_calce_cells(cell_ids: list[str] | None = None, data_dir: str | pathlib.Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load CALCE CS2 cells from locally placed workbook files.

    Directory layout expected (mirrors how CALCE's own per-cell zips
    unpack): data/raw/calce/{cell_id}/*.xlsx — one workbook per test
    session, any filenames, read in sorted (filename) order.

    Parameters
    ----------
    cell_ids : which cell subdirectories to load (default: every
        subdirectory found under the CALCE raw data dir).
    data_dir : override the default data/raw/calce location (mainly for
        tests — points fixture directories at this function without
        touching the real data dir).

    Returns
    -------
    {cell_id: DataFrame} satisfying batlab.datasets.schema's kind="cycle"
    contract, with df.attrs set.

    Raises
    ------
    CalceDataNotFoundError if no matching local cell data is found —
    includes exactly where to download it and where to place it.
    """
    raw_dir = pathlib.Path(data_dir) if data_dir is not None else _RAW_DIR

    if cell_ids is None:
        cell_ids = sorted(p.name for p in raw_dir.iterdir()) if raw_dir.exists() else []

    if not cell_ids:
        raise CalceDataNotFoundError(
            "No local CALCE CS2 cell data found.\n\n"
            f"Download cell .zip files (e.g. CS2_35.zip) from {CALCE_INFO_URL}, "
            "extract them, and place each cell's Excel workbook(s) at:\n"
            f"    {raw_dir / '<cell_id>'}/*.xlsx\n"
            "e.g. data/raw/calce/CS2_35/CS2_35_1_9_11.xlsx\n\n"
            "This loader does not auto-download — see batlab.datasets.calce's "
            "module docstring for why."
        )

    results: dict[str, pd.DataFrame] = {}
    for cell_id in cell_ids:
        files = _sorted_cell_files(raw_dir / cell_id)
        if not files:
            continue

        parts = []
        cycle_offset = 0
        for f in files:
            summary = _read_workbook(f)
            summary["cycle_number"] = summary["cycle_number"] + cycle_offset
            parts.append(summary)
            cycle_offset = int(summary["cycle_number"].max())

        df = pd.concat(parts, ignore_index=True).sort_values("cycle_number").reset_index(drop=True)
        if df.empty:
            continue

        df["soh_pct"] = compute_soh_pct(df["capacity_ah"])
        df.attrs["cell_id"] = cell_id
        df.attrs["source"] = "calce"
        df.attrs["chemistry"] = CHEMISTRY
        df.attrs["citation"] = "calce"
        df.attrs["license"] = (
            "Open access per CALCE's battery-data page; cite the requested CALCE "
            "reference for any publication use — see batlab.cite.cite(dataset='calce')."
        )
        results[cell_id] = df

    if not results:
        raise CalceDataNotFoundError(
            f"Found cell director{'y' if len(cell_ids) == 1 else 'ies'} for "
            f"{cell_ids} under {raw_dir}, but no readable .xlsx/.xls workbooks inside. "
            f"Re-download from {CALCE_INFO_URL} and place the workbook files directly "
            "in each cell's subdirectory."
        )

    return results


if __name__ == "__main__":
    print(f"CALCE CS2 has no scripted downloader — see {CALCE_INFO_URL}")
    print(f"Place downloaded workbooks under {_RAW_DIR}/<cell_id>/*.xlsx, then re-run.")
    try:
        cells = load_calce_cells()
        print(f"Loaded {len(cells)} cell(s): {sorted(cells)}")
    except CalceDataNotFoundError as exc:
        print(f"\n{exc}")
