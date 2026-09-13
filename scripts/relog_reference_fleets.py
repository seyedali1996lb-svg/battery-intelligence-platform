"""Re-log the reference fleets' plain-GBRT registry rows, headless.

Why this exists
---------------
The 2026-09-13 incident: app training sessions on 2026-09-12 wrote their
log_run() rows to an ephemeral DB, but the bundle caches survived. Because
cache hits never re-log, zhu2022 had NO plain-GBRT row in the registry at
all, and nasa/severson/synth headlines fell back to their pre-v12 rows
(Severson's stayed the v11 formula-recovery RUL R²=0.9994 that v12
redefined as not-evaluable). The durable fix is the registry-verified
cache-hit guard in app/_data.py + the MODEL_VERSION v5 bust; this script
is the recovery path that populates the rows NOW, headless, through the
EXACT functions the app runs (compute_features_only → train_and_predict →
persist → save_cached), so the logged rows are identical to what the next
cold app load would produce — not a parallel implementation that could
drift.

Usage:
    python scripts/relog_reference_fleets.py            # all four fleets
    python scripts/relog_reference_fleets.py zhu2022    # one fleet
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

import db  # noqa: E402  (app/_data.py's log_run path needs the engine bound first)
import experiment_registry as reg  # noqa: E402


def _build_fleets() -> dict[str, dict]:
    """The same cell-dict construction app/_data.load_everything() uses."""
    import os
    from data_loader import build_battery, CELL_STRESS_PROFILES
    from utils import NASA_CELL_IDS
    from batlab.datasets.severson import load_severson_cells
    from batlab.datasets.zhu2022 import load_zhu2022_cells
    from batlab.datasets.calce import load_calce_cells, CalceDataNotFoundError

    fleets: dict[str, dict] = {
        "synth": build_battery(battery_id="Oxford_B1", cell_ids=list(CELL_STRESS_PROFILES.keys()))["cells"],
    }

    nasa_ids = [cid for cid in NASA_CELL_IDS if os.path.exists(f"data/raw/{cid}_summary.csv")]
    if nasa_ids:
        fleets["nasa"] = build_battery(battery_id="NASA_B1", cell_ids=nasa_ids)["cells"]

    try:
        if load_severson_cells.__module__:  # pragma: no cover - always true
            sev = load_severson_cells(status_fn=lambda msg: None)
            if sev:
                fleets["severson"] = {cid: {"cycles": df} for cid, df in sev.items()}
    except Exception:
        pass

    try:
        zhu = load_zhu2022_cells(status_fn=lambda msg: None)
        if zhu:
            fleets["zhu2022"] = {cid: {"cycles": df} for cid, df in zhu.items()}
    except Exception:
        pass

    try:
        calce = load_calce_cells()
        if calce:
            fleets["calce"] = {cid: {"cycles": df} for cid, df in calce.items()}
    except CalceDataNotFoundError:
        pass  # manual download not performed on this deployment
    except Exception:
        pass

    return fleets


def _persist_cell_data(featured_dfs: dict) -> None:
    """Same persistence load_everything() performs after a training run."""
    import cell_store

    for cell_id, df in featured_dfs.items():
        cell_store.save_cell_df(cell_id, df)
        db.upsert_cell_summary(reg.PLATFORM_ORG_ID, cell_id, cell_store.build_summary(cell_id, df))


def main() -> int:
    import app._data as app_data
    from bundle_cache import load_cached, save_cached, load_features_cached, save_features_cached, clear_cache

    requested = set(sys.argv[1:]) or None
    fleets = _build_fleets()
    if requested:
        missing = requested - set(fleets)
        if missing:
            print(f"unavailable fleets (no data on this deployment): {sorted(missing)}")
        fleets = {k: v for k, v in fleets.items() if k in requested}

    exit_code = 0
    for key, cell_dict in fleets.items():
        if not cell_dict:
            continue
        cached = load_cached(key, cell_dict)
        if cached is not None and not app_data.cached_bundle_run_missing(cached):
            print(f"[skip] {key}: cache hit verified against registry (row exists)")
            continue
        if cached is not None:
            print(f"[stale] {key}: cached bundle's experiment_run_id has no registry row — rebuilding")

        feat_cached = load_features_cached(key, cell_dict)
        if feat_cached is not None:
            raw_fdfs, model_inputs = feat_cached
            print(f"[feat] {key}: features cache hit")
        else:
            raw_fdfs, model_inputs = app_data.compute_features_only(cell_dict)
            save_features_cached(key, cell_dict, raw_fdfs, model_inputs)
            print(f"[feat] {key}: features rebuilt")

        bundle, featured_dfs, split_cycles = app_data.train_and_predict(
            cell_dict, raw_fdfs, model_inputs, dataset=key, org_id=reg.PLATFORM_ORG_ID,
        )
        _persist_cell_data(featured_dfs)
        save_cached(key, cell_dict, (bundle, split_cycles))
        m = bundle.get("metrics", {})
        print(
            f"[logged] {key}: run_id={m.get('experiment_run_id')} "
            f"soh_r2={m.get('lco_soh_r2') or m.get('soh_r2')} "
            f"rul_r2={m.get('lco_rul_r2') or m.get('rul_r2')} "
            f"rul_reliable={m.get('rul_reliable')}"
        )

    # Final verification readout straight from the DB
    print("\n=== registry state after run ===")
    con = db.engine.raw_connection()
    try:
        rows = con.execute(
            "SELECT dataset, model_kind, feature_version, timestamp, soh_r2, rul_r2 "
            "FROM experiment_runs WHERE dataset IN ('synth','nasa','severson','zhu2022','calce') "
            "ORDER BY timestamp DESC"
        ).fetchall()
        seen: set = set()
        for ds, mk, fv, ts, soh, rul in rows:
            marker = (mk or "gbrt")
            if (ds, marker, fv) in seen:
                continue
            seen.add((ds, marker, fv))
            print(f"{ds:<10} {str(marker):<12} fv={str(fv):<28} {str(ts)[:19]} soh={soh if soh is None else round(soh, 4)} rul={rul if rul is None else round(rul, 4)}")
    finally:
        con.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
