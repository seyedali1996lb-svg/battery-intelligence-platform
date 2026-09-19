"""Where a reference fleet's boot training time actually goes.

Why this exists: the app's cold boot is dominated by model training, and until
2026-09-19 nobody could say WHICH part of it. The answer (see app/_data.py's
"Boot layers" section) is that it is essentially all leave-cell-out refitting —
run_lco() plus run_lco_quantiles() — while the hierarchical / censored-survival
/ routing work that looks expensive is 0.1-0.2 s per fleet. This script measures
the same split on YOUR machine, using the real functions the app calls (so it
cannot drift from the code it profiles):

    python scripts/profile_boot.py nasa zhu2022
    python scripts/profile_boot.py severson --eager

Each fleet is reported as:

    core        train_models + predictions (what load_everything() now waits for)
    validation  _layer_validation  — leave-cell-out GBRT + baselines + envelope
    forecast    _layer_forecast    — hierarchical pooling, survival, routing
    calibration _layer_calibration — quantile LCO + the conformal widening

Read it next to BATLAB_BOOT_LAYERS (app/_data.py): "core" is what a boot costs
with the default background mode, and core+layers is what `eager` costs.

Nothing is written: the profiler trains in memory only (no bundle cache, no
cell store, no registry row). A fleet whose features are not cached pays for
feature engineering once — that cost is printed separately.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _paths import setup  # noqa: E402

setup()

import _data  # noqa: E402
from bundle_cache import load_features_cached  # noqa: E402


def _cells(key: str) -> dict:
    """The same cell dicts load_everything() builds, per fleet."""
    if key in ("nasa", "synth"):
        from data_loader import CELL_STRESS_PROFILES, build_battery

        if key == "nasa":
            ids = _data.nasa_cells_available()
            if not ids:
                raise SystemExit("no NASA CSVs in data/raw — run python -m batlab.datasets.nasa")
            return build_battery(battery_id="NASA_B1", cell_ids=ids)["cells"]
        return build_battery(
            battery_id="Oxford_B1", cell_ids=list(CELL_STRESS_PROFILES.keys())
        )["cells"]
    if key == "severson":
        from batlab.datasets.severson import load_severson_cells

        return {
            cid: {"cycles": df}
            for cid, df in load_severson_cells(status_fn=lambda _m: None).items()
        }
    if key == "zhu2022":
        from batlab.datasets.zhu2022 import load_zhu2022_cells

        return {
            cid: {"cycles": df}
            for cid, df in load_zhu2022_cells(status_fn=lambda _m: None).items()
        }
    if key == "calce":
        from batlab.datasets.calce import load_calce_cells

        return {
            cid: {"cycles": df}
            for cid, df in load_calce_cells().items()
        }
    raise SystemExit(f"unknown fleet '{key}' (nasa, synth, severson, zhu2022, calce)")


def profile(key: str, *, eager: bool) -> None:
    t0 = time.perf_counter()
    cells = _cells(key)
    load_s = time.perf_counter() - t0
    n_rows = sum(len(c["cycles"]) for c in cells.values())
    print(f"\n=== {key}: {len(cells)} cells, {n_rows} cycles (loaded in {load_s:.1f}s) ===", flush=True)

    t0 = time.perf_counter()
    cached = load_features_cached(key, cells)
    if cached is None:
        raw_fdfs, model_inputs = _data.compute_features_only(cells)
        print(f"    features (cold)          {time.perf_counter() - t0:8.1f}s", flush=True)
    else:
        raw_fdfs, model_inputs = cached
        print(f"    features (cache hit)     {time.perf_counter() - t0:8.1f}s", flush=True)

    cell_cycles = {cid: c["cycles"] for cid, c in cells.items()}

    if eager:
        t0 = time.perf_counter()
        bndl, _frames, _splits = _data.train_and_predict(
            cells, raw_fdfs, model_inputs, defer_layers=False
        )
        print(
            f"    core+layers (eager)      {time.perf_counter() - t0:8.1f}s"
            f"  (boot_layers={bndl['metrics'].get('boot_layers')})",
            flush=True,
        )
        return

    t0 = time.perf_counter()
    bndl, _frames, _splits = _data.train_and_predict(
        cells, raw_fdfs, model_inputs, defer_layers=True
    )
    print(
        f"    core (what a boot waits) {time.perf_counter() - t0:8.1f}s"
        "  <- the default background mode",
        flush=True,
    )

    # The layers this script is here to measure (guarded: a layer that fails is
    # reported, not fatal — the bundle keeps its placeholders).
    t0 = time.perf_counter()
    _data.run_boot_layers(bndl, cell_cycles, raw_fdfs, key=key, guarded=True)
    print(f"    layers (background)      {time.perf_counter() - t0:8.1f}s", flush=True)

    seconds = (_data.boot_layers_status()["seconds"] or {}).get(key) or {}
    for layer in _data.BOOT_LAYER_NAMES:
        if layer in seconds:
            print(f"      {layer:<22} {seconds[layer]:8.1f}s", flush=True)
    failed = _data.boot_layers_status()["failed"]
    if failed:
        print(f"      failed layers: {failed}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Profile where a reference fleet's boot training time goes."
    )
    ap.add_argument("fleets", nargs="+", help="nasa, synth, severson, zhu2022, calce")
    ap.add_argument(
        "--eager", action="store_true",
        help="run the validation/forecast/calibration layers inline (BATLAB_BOOT_LAYERS=eager behaviour)",
    )
    args = ap.parse_args()
    for key in args.fleets:
        profile(key, eager=args.eager)


if __name__ == "__main__":
    main()
