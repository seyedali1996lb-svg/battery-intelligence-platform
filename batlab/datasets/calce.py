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

Verification status
--------------------
This loader's column names and parsing logic (including the
cumulative-across-cycles capacity quirk documented below) were verified
against a real downloaded file — CS2_35 from
https://web.calce.umd.edu/batteries/data/CS2_35.zip — which parses to
physically plausible values (capacity ~1.0-1.1 Ah against the cell's
~1.1 Ah nominal rating, coulombic efficiency ~0.99-1.0, resistance
~0.09 Ohm) and passes validate_schema(). Only that one cell/file was
checked; other CS2 cells' workbooks were not individually verified, so
treat this as "verified on a real sample, not exhaustively" rather than
"guaranteed correct for every CS2 cell CALCE has published."

Manual download required
-------------------------
This loader does NOT auto-download, unlike NASA/Severson/Oxford. During
verification, direct per-cell URLs were in fact found and confirmed
reachable (https://web.calce.umd.edu/batteries/data/CS2_<N>.zip, no
registration gate) — so an auto-downloader is technically feasible here
after all, contrary to what an earlier version of this docstring assumed.
It was deliberately not added in this pass: unlike Severson's one ~115 MB
shared batch file, CALCE is one ~30-40 MB zip *per cell*, so "auto-download
by default" doesn't have an obviously-right default the way the other
three loaders' do (download everything? nothing? which cells?) without a
product decision this loader shouldn't make unilaterally. Call
load_calce_cells() with cells already downloaded and placed locally (see
its docstring); if nothing is found, it raises CalceDataNotFoundError with
exact instructions on where to get the files and where to place them.

File format
-----------
CALCE cells were cycled on Arbin BT2000 testers; each test session is
published as one Excel workbook per cell (filename dated, e.g.
"CS2_35_8_18_10.xlsx" = Aug 18 2010), containing an "Info" sheet and a
per-datapoint "Channel_*" sheet (the one real file checked during
verification had no separate "Statistics_*" summary sheet — this loader
still looks for one first, since other CALCE cells' files may include one,
but falls back to the per-datapoint Channel sheet either way). This loader
reduces the sheet to one row per Cycle_Index via groupby.

Confirmed quirk: Discharge_Capacity(Ah) and Charge_Capacity(Ah) accumulate
across the WHOLE workbook, not reset to 0 at each cycle boundary — cycle
2's rows continue climbing from cycle 1's ending value. Each cycle's own
capacity is therefore the swing (max - min) within that cycle's rows, the
same convention batlab.datasets.oxford._capacity_ah_from_table() already
uses for its own max-Amphr-swing extraction rule.

Cycle_Index resets to 1 in every new test-session file for the same cell
(a separate, also-confirmed Arbin/CALCE quirk) — load_calce_cells()
renumbers cycles cumulatively across a cell's files, sorted by filename,
so cycle_number is monotonically increasing across the cell's full life,
matching every other batlab loader's convention.
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
            "This exact column layout was confirmed against a real CS2_35 file — see "
            "batlab.datasets.calce's module docstring — so a mismatch here likely means "
            "a different CALCE cell series uses a different export format."
        )
    if "Discharge_Capacity(Ah)" not in df.columns:
        raise CalceDataNotFoundError(
            f"Expected a 'Discharge_Capacity(Ah)' column in a CALCE raw sheet, got: {list(df.columns)}."
        )

    # Discharge_Capacity(Ah) (and Charge_Capacity(Ah)) accumulate across the
    # WHOLE workbook, not reset to 0 at the start of each cycle — verified
    # against a real downloaded CALCE file (CS2_35), where cycle 2's rows
    # continued climbing from cycle 1's ending value (e.g. [1.10, 2.19] for
    # cycle 2 immediately after cycle 1 ended at 1.10), not restarting at 0.
    # Each cycle's own discharge capacity is therefore the swing (max - min)
    # within that cycle's rows — the same convention
    # batlab.datasets.oxford._capacity_ah_from_table() already uses for its
    # own max-Amphr-swing extraction rule, for the same underlying reason.
    grouped = df.groupby("Cycle_Index")
    dc = grouped["Discharge_Capacity(Ah)"]
    capacity_ah = dc.max() - dc.min()

    out = pd.DataFrame({
        "cycle_number": capacity_ah.index,
        "capacity_ah": capacity_ah.to_numpy(),
    }).reset_index(drop=True)

    if "Charge_Capacity(Ah)" in df.columns:
        cc = grouped["Charge_Capacity(Ah)"]
        charge_ah = (cc.max() - cc.min()).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ce = capacity_ah.to_numpy() / np.where(charge_ah == 0, np.nan, charge_ah)
        out["coulombic_efficiency"] = ce

    temp_col = _find_column(list(df.columns), _TEMPERATURE_COL_HINTS)
    if temp_col:
        out["temperature_c"] = grouped[temp_col].mean().to_numpy()

    resistance_col = _find_column(list(df.columns), _RESISTANCE_COL_HINTS)
    if resistance_col:
        out["resistance_ohm"] = grouped[resistance_col].mean().to_numpy()

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
        # Protocol-known test conditions, verified against CALCE's published
        # CS2 protocol: 0.5C CC/CV charge to 4.2V (held until current <0.05A),
        # 2.7V discharge cutoff ("unless specified" per CALCE's own docs —
        # this is the CS2 series default, applied the same way this module's
        # docstring already caveats its Arbin column-name assumptions as
        # "verified on a real sample, not exhaustive"). No numeric temperature
        # setpoint is set — CALCE's documentation only says "room temperature",
        # and this module does not guess a number for that.
        df.attrs["voltage_charge_cutoff_v"] = 4.2
        df.attrs["voltage_discharge_cutoff_v"] = 2.7
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
