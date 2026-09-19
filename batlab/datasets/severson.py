"""
Severson 2019 LFP dataset loader.

Reference: Severson et al., "Data-driven prediction of battery cycle life
before capacity degradation", Nature Energy 2019.
Data: https://data.matr.io/1/  (research use)
Citation/license: see batlab.cite.cite(dataset="severson2019")

Downloads Batch 1 of the MATLAB file (~2.9 GB) on first call, extracts
per-cycle discharge capacity, resistance, and temperature for ALL 46 cell
records the batch contains, and caches them as CSVs in data/raw/severson/.
Returns the standardized batlab cycle schema (see batlab.datasets.schema).

Why all 46 (was 12)
-------------------
The loader originally shipped with 12 representative cells spanning 4
cycle-life bands, and every accuracy number in the Validation section
carried an n<=12 disclaimer with wide brackets. Batch 1 actually contains
46 cell records, ALL of which carry usable summary channels — the 5 cells
the original authors exclude from their 124-cell release (b1c8, b1c10,
b1c12, b1c13, b1c22, per their own Load Data notebook) are excluded there
because they never reach 80% capacity, not because their data is corrupt
(content check against the .mat: every record has 500+ finite QDischarge
rows). Excluding never-EOL cells is a MODELLING choice, and this platform
now has the machinery to keep them honestly: the v12 RUL rules mark them
"extrapolated / not evaluable" and the survival readout
(batlab.validation.survival) treats them as right-censored observations.
Dropping them here would discard exactly the data the censored-RUL item
exists to use, so all 46 are extracted and served.

Naming: cell ids are S-b1cN where N is the RECORD's position in the batch
struct array (0-based). This is a deliberate, disclosed one-position shift
from the legacy loader, which named records with 1-based indices: the legacy
S-b1c2 (1177 cycles) is record position 1 and is now S-b1c1, legacy S-b1c3
is now S-b1c2, and so on — content-verified against the previous 12 CSVs'
cycle counts and capacity traces. Every registry row, fingerprint and cache
key written before this change carries the legacy names, so the shift is a
fact to state, not hide: cached bundles keyed on cell content miss and
retrain once, and historical runs keep their original ids unchanged.
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import requests

from batlab.datasets._integrity import verify_sha256
from batlab.datasets._paths import raw_data_dir
from batlab.datasets.schema import compute_soh_pct

_BATCH1_URL = "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download"

# SHA-256 of the file _BATCH1_URL served when last verified against a real
# download (2026-07 — see batlab/datasets/_integrity.py for why this is
# checked). Batch 1's full HDF5 .mat is ~2.9 GB (all cells in the batch,
# not just the 12 this loader extracts), not the "~115 MB" this module's
# docstring used to claim.
_EXPECTED_SHA256 = "9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba"

# Every record position of Batch 1 (0-based). Content check against the
# .mat (2026-09): all 46 records carry 500+ finite QDischarge rows — there
# are no corrupt/empty records to skip (the original authors' 5 exclusions,
# b1c8/10/12/13/22, are cells that never reach 80% capacity in-window, a
# modelling exclusion their pipeline made and this platform's survival
# machinery now handles honestly instead). Keys are consecutive so the
# S-b1cN name matches the record's batch-array position exactly.
_CELL_KEYS = [f"b1c{i}" for i in range(46)]
# 0-based index of each record in the batch HDF5 struct array (b1cN → N).
_CELL_INDICES = {k: int(k[3:]) for k in _CELL_KEYS}

# A123 APR18650M1A nominal capacity (Severson et al. 2019): the reference
# for the physically-impossible-reading guard below.
_NOMINAL_CAPACITY_AH = 1.1

# Resolved through batlab.datasets._paths: an explicit BATLAB_DATA_DIR wins,
# then a source checkout's data/raw, then a per-user cache directory — never
# site-packages, which is where the old repo-relative path pointed in a wheel.
_RAW_DIR = raw_data_dir("severson")
SEVERSON_CELL_IDS: list[str] = [f"S-{k}" for k in _CELL_KEYS]

CHEMISTRY = "LFP"


def _csv_path(k: str) -> pathlib.Path:
    return _RAW_DIR / f"{k}_summary.csv"


def _all_cached() -> bool:
    return all(_csv_path(k).exists() for k in _CELL_KEYS)


def any_cached() -> bool:
    """Return True if at least one cell CSV exists locally."""
    return any(_csv_path(k).exists() for k in _CELL_KEYS)


def fully_cached_cell_ids() -> list[str]:
    """Every cell ID, but ONLY when the whole fleet is cached; [] otherwise.

    Deliberately not the same question as any_cached(): a PARTIAL cache still
    makes load_severson_cells() try to download the remaining ~2.9 GB batch,
    which is a reasonable decision for a training script to make and a bad one
    to trigger on a UI click. A caller that wants to offer Severson as an
    option, or to promise an offline run, needs this stronger guarantee —
    any_cached() cannot provide it.
    """
    return list(SEVERSON_CELL_IDS) if _all_cached() else []


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
        batch = f["batch"]
        # h5py's stubs type these containers as `Datatype`/`Any` unions and
        # flag every subscription/indexing below; at runtime they are the
        # standard Group/Dataset objects. Silence the class of false
        # positive once here rather than scattering ignores per line.
        batch_group: "h5py.Group" = batch  # type: ignore[assignment]
        summ_ds: "h5py.Dataset" = batch_group["summary"]  # type: ignore[assignment]   # (N_cells, 1) array of HDF5 object refs

        if summ_ds.shape[0] < len(_CELL_KEYS):
            # The .mat carries fewer usable records than the cell list claims
            # (batches 2/3 are separate files) — extract what exists, name
            # what didn't, never fabricate.
            print(f"  [severson] batch file holds {summ_ds.shape[0]} records; "
                  f"{len(_CELL_KEYS)} keys requested — extracting what exists")
        for key, idx in _CELL_INDICES.items():
            if idx >= summ_ds.shape[0]:
                print(f"  [severson] skipping {key}: no record at batch index {idx}")
                continue
            try:
                summ: "h5py.Group" = f[summ_ds[idx, 0]]  # type: ignore[assignment]  # per-cell summary Group (refs dereference to Group at runtime)
                qd = np.array(summ["QDischarge"]).flatten().astype(float)  # type: ignore[index]
                # Remove cycle-0 pre-charge row if present
                if qd[0] == 0:
                    qd = qd[1:]
                n = len(qd)
                if n == 0 or not np.isfinite(qd).any():
                    # Defensive, not currently exercised (content check
                    # 2026-09: all 46 records carry 500+ finite rows): if a
                    # re-download ever serves a file version with an empty
                    # QDischarge channel, that record is skipped and NAMED,
                    # never folded into the fleet as NaNs.
                    print(f"  [severson] skipping {key}: empty/corrupt QDischarge channel")
                    continue

                cyc_raw = np.array(summ["cycle"]).flatten().astype(float)  # type: ignore[index]
                cycles  = cyc_raw[cyc_raw > 0] if cyc_raw[0] == 0 else cyc_raw
                cycles  = cycles[:n]
                if len(cycles) < n:
                    cycles = np.arange(1, n + 1, dtype=float)

                ir   = np.array(summ["IR"]).flatten().astype(float)[:n]   if "IR"   in summ else np.full(n, np.nan)   # type: ignore[index,operator]
                tavg = np.array(summ["Tavg"]).flatten().astype(float)[:n] if "Tavg" in summ else np.full(n, 30.0)  # type: ignore[index,operator]
                # Tavg absent in the source is stored by the raw export as
                # 0.0 — a missing-value sentinel, not a measurement (no
                # cycle runs at absolute zero). Left in place it silently
                # dragged every rolling-temperature feature toward zero;
                # convert to NaN so build_features()'s min_periods handling
                # skips them honestly.
                tavg = np.where(tavg <= 0.0, np.nan, tavg)
                # Physically-impossible capacity readings are cycler
                # glitches, not states of charge: a 1.1 Ah APR18650M1A
                # cannot deliver >1.2× nominal. Two batch-1 records (the
                # ones behind the legacy loader's 12, never) carry such
                # spikes (up to 2.7× nominal); NaN them so rolling windows
                # with min_periods skip them and get_model_matrix drops the
                # rows — the exact convention the Tavg 0.0-sentinel fix
                # above established. Rows are KEPT (cycle axis unchanged);
                # only the impossible measurement is voided.
                _glitch = qd > _NOMINAL_CAPACITY_AH * 1.2
                if _glitch.any():
                    print(f"  [severson] {key}: {int(_glitch.sum())} capacity "
                          "reading(s) above 1.2x nominal — NaN'd as cycler glitches")
                    qd = qd.copy()
                    qd[_glitch] = np.nan
                # Reference capacity: the first FINITE reading — identical
                # to the legacy convention for every clean cell (so the
                # served soh_pct series is byte-comparable with pre-existing
                # fingerprints), and immune only where the legacy convention
                # was actually broken (a glitch AT cycle 1).
                _finite = qd[np.isfinite(qd)]
                q0 = float(_finite[0]) if len(_finite) else 1.0
                if q0 <= 0:
                    q0 = 1.0

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
    # Sentinel cleanup at LOAD time, not just download time: the Tier-4
    # fix in _download_and_cache only touches fresh exports, but CSVs
    # cached before that fix keep their 0.0 sentinels on disk. No cycle
    # runs at absolute zero — a 0.0 °C Tavg is a missing value, and left
    # in place it poisons temp_rolling_30cy and (as the Tier-5 envelope
    # showed) reads as a 0 °C fleet minimum. NaN lets min_periods skip it.
    if "temperature_c" in df.columns:
        t = pd.to_numeric(df["temperature_c"], errors="coerce")
        df["temperature_c"] = t.mask(t.to_numpy(dtype="float64") <= 0.0)
    # Capacity-glitch guard at LOAD time too (same pattern as the
    # temperature sentinel above): CSVs cached before the glitch fix keep
    # their impossible readings on disk. A 1.1 Ah cell cannot deliver
    # >1.2x nominal — void the reading, keep the row.
    if "capacity_ah" in df.columns:
        c = pd.to_numeric(df["capacity_ah"], errors="coerce")
        _c_arr = c.to_numpy(dtype="float64")
        n_glitch = int(np.count_nonzero(_c_arr > _NOMINAL_CAPACITY_AH * 1.2))
        if n_glitch:
            df["capacity_ah"] = c.mask(_c_arr > _NOMINAL_CAPACITY_AH * 1.2)
            # soh_pct is capacity/q0*100 by construction — recompute so the
            # two columns can never disagree about which rows are voided.
            # q0 = first finite reading (the loader's download-time rule).
            _q0_series = df["capacity_ah"].dropna()
            _q0 = float(_q0_series.iloc[0]) if len(_q0_series) else 1.0
            df["soh_pct"] = df["capacity_ah"] / _q0 * 100.0 if _q0 > 0 else np.nan
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
