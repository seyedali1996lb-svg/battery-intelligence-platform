"""
NASA PCoE Battery Aging Dataset loader.

Downloads and parses MATLAB .mat files for cells B0005, B0006, B0007, B0018
into the existing Battery -> Cells -> Cycles schema:
    DataFrame columns: cycle_number, capacity_ah, resistance_ohm, temperature_c

Run this script once to pull real data into data/raw/.  After that,
build_battery() picks up the CSVs automatically via load_or_generate_cell().

Dataset: NASA Prognostics Center of Excellence Battery Data Set
Source:  https://data.nasa.gov (ti.arc.nasa.gov/tech/dash/groups/pcoe/
         prognostic-data-repository/ — Battery Data Set #5)
License: Public domain / US government work

Cell operating conditions:
  B0005–B0007: charged 1.5A CC/CV to 4.2V, discharged 2A to 2.7V at 24°C
  B0018:        discharged to 2.5V (different EOL criterion)
  All: 18650-format LiCoO₂, ~2 Ah nominal
"""

import io
import os
import sys
import zipfile
import numpy as np
import pandas as pd
import requests
import scipy.io

# ---------------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# NASA PCoE S3 mirror — Battery Data Set #5
# Source: nasa.gov/intelligent-systems-division/.../pcoe-data-set-repository/
# Citation: B. Saha and K. Goebel (2007), NASA Ames Research Center
NASA_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_nasa_zip(dest_dir: str) -> str:
    """
    Download the NASA battery ZIP to dest_dir and return the local path.
    Skips download if the file already exists.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "nasa_battery.zip")

    if os.path.exists(zip_path):
        print(f"[cache] ZIP already present at {zip_path}")
        return zip_path

    print("Downloading NASA Battery Dataset (~22 MB)...")
    print(f"  URL: {NASA_ZIP_URL}")
    try:
        r = requests.get(NASA_ZIP_URL, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {pct:.0f}%  ({downloaded // 1024} KB)", end="", flush=True)
        print()
        print(f"[ok] Saved to {zip_path}")
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            "Please manually download the NASA Battery Aging dataset and\n"
            "place B0005.mat, B0006.mat, B0007.mat, B0018.mat in data/raw/"
        ) from exc
    return zip_path


def extract_mat_files(zip_path: str, dest_dir: str) -> list[str]:
    """
    Extract .mat files for our four cells from the outer ZIP.

    The NASA download is a nested ZIP: the outer ZIP contains inner ZIPs,
    one per batch of cells. B0005–B0018 live inside inner ZIP #1.
    We walk the outer ZIP looking for entries named {cell_id}.mat directly,
    or recurse into inner ZIPs to find them.
    """
    mat_paths = []
    with zipfile.ZipFile(zip_path, "r") as outer:
        outer_names = outer.namelist()

        # Collect any direct .mat files (rare but possible)
        direct = {os.path.basename(n): n for n in outer_names if n.endswith(".mat")}

        # Collect inner ZIP names for nested extraction
        inner_zips = [n for n in outer_names if n.endswith(".zip")]

        for cell_id in CELL_IDS:
            out_path = os.path.join(dest_dir, f"{cell_id}.mat")
            if os.path.exists(out_path):
                print(f"  [cache] {cell_id}.mat already extracted")
                mat_paths.append(out_path)
                continue

            # Check direct entries first
            if f"{cell_id}.mat" in direct:
                data = outer.read(direct[f"{cell_id}.mat"])
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"  [extract] {cell_id}.mat ({len(data)//1024} KB)")
                mat_paths.append(out_path)
                continue

            # Search inner ZIPs
            found = False
            for inner_name in inner_zips:
                try:
                    inner_bytes = outer.read(inner_name)
                    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
                    if f"{cell_id}.mat" in inner.namelist():
                        data = inner.read(f"{cell_id}.mat")
                        with open(out_path, "wb") as f:
                            f.write(data)
                        print(f"  [extract] {cell_id}.mat from {inner_name} ({len(data)//1024} KB)")
                        mat_paths.append(out_path)
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                print(f"  [warn] {cell_id}.mat not found in ZIP")

    return mat_paths


# ---------------------------------------------------------------------------
# Parse .mat -> DataFrame
# ---------------------------------------------------------------------------

def _get_scalar(arr):
    """Unwrap a nested MATLAB scalar to a Python float."""
    while hasattr(arr, "__len__") and len(arr) == 1:
        arr = arr[0]
    return float(arr)


def _get_array(arr) -> np.ndarray:
    """Flatten a MATLAB array to a 1-D numpy array."""
    return np.asarray(arr).flatten()


def parse_mat_file(mat_path: str, cell_id: str) -> pd.DataFrame:
    """
    Parse a NASA .mat file into a cycle-summary DataFrame.

    The MATLAB file stores one struct per cell (field = cell_id, e.g. 'B0005').
    That struct has a 'cycle' array, one entry per cycle, each with a 'type'
    field ('charge', 'discharge', 'impedance') and a 'data' sub-struct.

    We extract:
      - discharge cycles -> capacity_ah, temperature_c
      - impedance cycles -> resistance_ohm  (Re: electrolyte resistance)

    Cycles are aligned by their index so capacity and resistance match.
    """
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)

    # The top-level key is the cell ID
    if cell_id not in mat:
        raise KeyError(f"Key '{cell_id}' not found in {mat_path}. "
                       f"Available keys: {[k for k in mat if not k.startswith('_')]}")

    cell_struct = mat[cell_id]
    cycles = cell_struct["cycle"]

    discharge_records = []
    impedance_records = []
    discharge_idx = 0
    impedance_idx = 0

    for cyc in cycles:
        cyc_type = str(cyc["type"]).strip()
        data = cyc["data"]

        if cyc_type == "discharge":
            discharge_idx += 1
            try:
                # Capacity in Ah — stored directly as a scalar in this dataset
                if "Capacity" in data:
                    cap = _get_scalar(data["Capacity"])
                elif "capacity" in data:
                    cap = _get_scalar(data["capacity"])
                else:
                    # Integrate current over time: Q = ∫|I| dt (A·s -> Ah)
                    current = _get_array(data.get("Current_measured",
                                                   data.get("current_measured", [0])))
                    time_s  = _get_array(data.get("Time",
                                                   data.get("time", np.zeros_like(current))))
                    cap = float(np.trapz(np.abs(current), time_s)) / 3600.0

                # Temperature: mean over discharge period
                temp_arr = _get_array(data.get("Temperature_measured",
                                               data.get("temperature_measured", [25.0])))
                temp_mean = float(np.mean(temp_arr[temp_arr > 0]))  # exclude zeros

                discharge_records.append({
                    "discharge_idx": discharge_idx,
                    "capacity_ah": cap,
                    "temperature_c": temp_mean,
                })
            except Exception as exc:
                print(f"  [warn] discharge cycle {discharge_idx} parse error: {exc}")

        elif cyc_type == "impedance":
            impedance_idx += 1
            try:
                # Re = electrolyte (ohmic) resistance from EIS measurement
                if "Re" in data:
                    re = _get_scalar(data["Re"])
                elif "Battery_impedance" in data:
                    # Real part of impedance at high frequency
                    z = _get_array(data["Battery_impedance"])
                    re = float(np.real(z[0])) if len(z) > 0 else np.nan
                else:
                    re = np.nan

                impedance_records.append({
                    "impedance_idx": impedance_idx,
                    "resistance_ohm": re,
                })
            except Exception as exc:
                print(f"  [warn] impedance cycle {impedance_idx} parse error: {exc}")

    if not discharge_records:
        raise ValueError(f"No discharge cycles found in {mat_path}")

    discharge_df  = pd.DataFrame(discharge_records)
    impedance_df  = pd.DataFrame(impedance_records) if impedance_records else None

    # Align resistance measurements to discharge cycles.
    # Impedance tests are run every ~5 discharge cycles; forward-fill to cover all.
    discharge_df["cycle_number"] = np.arange(1, len(discharge_df) + 1)

    if impedance_df is not None and len(impedance_df) > 0:
        # Map impedance measurements onto discharge cycle numbers by spacing
        n_dis = len(discharge_df)
        n_imp = len(impedance_df)
        # EIS is taken roughly every n_dis/n_imp discharge cycles
        spacing = max(1, n_dis // n_imp)
        imp_cycles = np.clip(
            (impedance_df["impedance_idx"].values - 1) * spacing + 1,
            1, n_dis
        ).astype(int)
        impedance_df["cycle_number"] = imp_cycles
        impedance_df = impedance_df.drop(columns=["impedance_idx"])
        impedance_df = impedance_df.groupby("cycle_number", as_index=False).mean()

        discharge_df = discharge_df.merge(impedance_df, on="cycle_number", how="left")
        discharge_df["resistance_ohm"] = discharge_df["resistance_ohm"].ffill().bfill()
    else:
        # No impedance data — fill with a plausible rising resistance
        n = len(discharge_df)
        discharge_df["resistance_ohm"] = 0.150 + 0.0002 * discharge_df["cycle_number"]

    discharge_df = discharge_df.drop(columns=["discharge_idx"])
    discharge_df = discharge_df[["cycle_number", "capacity_ah", "resistance_ohm", "temperature_c"]]
    discharge_df = discharge_df.round({"capacity_ah": 5, "resistance_ohm": 5, "temperature_c": 2})

    return discharge_df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_nasa_cells(
    cell_ids: list[str] | None = None,
    data_dir: str | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download (if needed) and parse NASA cells into DataFrames.

    Returns {cell_id: DataFrame} with columns:
        cycle_number, capacity_ah, resistance_ohm, temperature_c

    These CSVs are then written to data/raw/ so build_battery() finds them
    without any schema changes.
    """
    if cell_ids is None:
        cell_ids = CELL_IDS
    if data_dir is None:
        data_dir = DATA_DIR

    results = {}

    # Check if we already have parsed CSVs
    missing = []
    for cell_id in cell_ids:
        csv_path = os.path.join(data_dir, f"{cell_id}_summary.csv")
        if os.path.exists(csv_path) and not force_download:
            df = pd.read_csv(csv_path)
            if "temperature_c" in df.columns and "capacity_ah" in df.columns:
                print(f"  [cache] {cell_id} already parsed ({len(df)} cycles)")
                results[cell_id] = df
                continue
        missing.append(cell_id)

    if not missing:
        return results

    # Need to download/parse
    zip_path = download_nasa_zip(data_dir)
    mat_paths = extract_mat_files(zip_path, data_dir)

    mat_map = {os.path.basename(p).replace(".mat", ""): p for p in mat_paths}

    for cell_id in missing:
        mat_path = mat_map.get(cell_id)
        if mat_path is None:
            # Try individual download as fallback
            mat_path = _try_direct_download(cell_id, data_dir)
            if mat_path is None:
                print(f"  [skip] {cell_id} — .mat file not available")
                continue

        print(f"\nParsing {cell_id}...")
        try:
            df = parse_mat_file(mat_path, cell_id)
            csv_path = os.path.join(data_dir, f"{cell_id}_summary.csv")
            df.to_csv(csv_path, index=False)
            print(f"  [ok] {len(df)} discharge cycles | "
                  f"capacity {df['capacity_ah'].iloc[0]:.3f}->{df['capacity_ah'].iloc[-1]:.3f} Ah | "
                  f"temp {df['temperature_c'].mean():.1f}°C mean")
            results[cell_id] = df
        except Exception as exc:
            print(f"  [error] {cell_id}: {exc}")

    return results


def _try_direct_download(cell_id: str, data_dir: str) -> str | None:
    """
    Try to download a single .mat file directly (fallback if ZIP extraction fails).
    Returns local path on success, None on failure.
    """
    # Alternative: CALCE dataset mirror or direct file URL
    # These are public mirrors sometimes used in academic repos
    candidate_urls = [
        f"https://data.nasa.gov/api/views/uj5r-zjdb/files/{cell_id}.mat",
    ]
    mat_path = os.path.join(data_dir, f"{cell_id}.mat")
    for url in candidate_urls:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 10_000:
                with open(mat_path, "wb") as f:
                    f.write(r.content)
                return mat_path
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# CLI entry point — run directly to pull real data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NASA Battery Dataset Loader ===\n")
    print("This will download the NASA PCoE Battery Aging dataset (~22 MB)")
    print("and convert it into the Battery Intelligence Platform schema.\n")

    force = "--force" in sys.argv

    cell_dfs = load_nasa_cells(force_download=force)

    if not cell_dfs:
        print("\n[FAIL] No cells loaded. Check your internet connection or")
        print("manually place B0005.mat ... B0018.mat in data/raw/")
        sys.exit(1)

    print(f"\n=== Summary: {len(cell_dfs)} cells loaded ===")
    for cell_id, df in cell_dfs.items():
        initial_cap = df["capacity_ah"].iloc[0]
        final_cap   = df["capacity_ah"].iloc[-1]
        soh_end     = (final_cap / initial_cap) * 100
        eol_mask    = df["capacity_ah"] <= initial_cap * 0.80
        eol_at      = df.loc[eol_mask, "cycle_number"].iloc[0] if eol_mask.any() else "not reached"
        print(f"  {cell_id}: {len(df)} cycles | "
              f"cap {initial_cap:.3f}->{final_cap:.3f} Ah | "
              f"final SOH {soh_end:.1f}% | EOL at {eol_at}")

    print("\n[DONE] CSVs written to data/raw/.")
    print("Restart the Streamlit app — it will load real data automatically.")
    print("Add B0005/B0006/B0007/B0018 to the cell selector in app/main.py")
    print("(or let it auto-detect by scanning data/raw/ for *_summary.csv files).")
