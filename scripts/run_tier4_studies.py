"""One-off driver: run the Tier-4 modeling + robustness studies on the real
reference fleets and log them to the experiment registry (data/app.db).

Run detached so multi-minute fleets (Zhu ~970 cycles/cell, Severson 12 cells)
survive terminal timeouts:

    python scripts/run_tier4_studies.py            # all fleets
    python scripts/run_tier4_studies.py zhu2022    # one fleet

Idempotent: studies already logged for (dataset, model_kind/FEATURE_VERSION)
are skipped, so re-running after a partial completion only fills the gaps.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import _paths  # noqa: F401  (side effect: src/, app/, scripts/ on sys.path)

import pandas as pd  # noqa: F401  (lco harness contract reference)

import experiment_registry as reg
from data_loader import build_battery, CELL_STRESS_PROFILES
from utils import NASA_CELL_IDS


def nasa_cell_dicts() -> dict:
    """{cell_id: raw cycles df} for the NASA fleet — loaded through the
    SAME path the app uses (build_battery → load_or_generate_cell), not
    batlab.datasets.nasa directly: the two differ in preprocessing (580
    vs 636 rows for the same cells), and a study on different raw data
    than the app's GBRT row would make every "vs GBRT" delta dishonest."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    ids = [
        cid for cid in NASA_CELL_IDS
        if os.path.exists(os.path.join(data_dir, f"{cid}_summary.csv"))
    ]
    if not ids:
        return {}
    battery = build_battery(battery_id="NASA_B1", cell_ids=ids)
    return {cid: cell["cycles"] for cid, cell in battery["cells"].items()}


def main() -> None:
    wanted = set(sys.argv[1:]) or None

    fleets: dict[str, dict] = {}

    synth = build_battery(battery_id="Oxford_B1", cell_ids=list(CELL_STRESS_PROFILES.keys()))
    fleets["synth"] = {cid: c["cycles"] for cid, c in synth["cells"].items()}

    nasa = nasa_cell_dicts()
    if nasa:
        fleets["nasa"] = nasa

    try:
        from batlab.datasets.severson import load_severson_cells
        sev = load_severson_cells(status_fn=lambda msg: None)
        if sev:
            fleets["severson"] = {cid: df for cid, df in sev.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"severson load failed: {exc!r}")

    try:
        from batlab.datasets.zhu2022 import load_zhu2022_cells
        zhu = load_zhu2022_cells(status_fn=lambda msg: None)
        if zhu:
            fleets["zhu2022"] = {cid: df for cid, df in zhu.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"zhu2022 load failed: {exc!r}")

    if wanted:
        fleets = {k: v for k, v in fleets.items() if k in wanted}
        # keep any requested key that failed to load so the report is explicit
        missing = wanted - set(fleets)
        if missing:
            print(f"requested but unavailable: {sorted(missing)}")

    print(f"fleets: { {k: len(v) for k, v in fleets.items()} }", flush=True)

    t0 = time.time()
    try:
        out = reg.run_modeling_benchmark_study(fleets)
        print(f"modeling study: {len(out)} runs in {time.time() - t0:.0f}s", flush=True)
        for r in out:
            print(
                f"  {r['dataset']:9s} {r['model_kind']:13s}"
                f" soh_r2={r['soh_r2']} rul_r2={r['rul_r2']}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"modeling study FAILED: {exc!r}", flush=True)

    t1 = time.time()
    try:
        out = reg.run_robustness_study(fleets)
        print(f"robustness study: {len(out)} rows in {time.time() - t1:.0f}s", flush=True)
        for r in out:
            thr = r.get("failure_threshold") or {}
            print(
                f"  {r['dataset']:20s} clean soh_r2={r['baseline'].get('soh_r2')}"
                f" thresholds={ {k: v for k, v in thr.items() if v} }",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"robustness study FAILED: {exc!r}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
