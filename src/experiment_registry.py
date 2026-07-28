"""
Experiment registry — logs every GBRT training run for later comparison,
filtering, and exact-pipeline replay.

Same split as trajectory_memory.py / src/db.py's FailureSignature table:
this module owns the domain dataclass and orchestration logic (building a
run_id, resolving the git commit, querying/filtering/sorting, replaying a
recorded run); src/db.py owns the ExperimentRun table and raw CRUD.

Why every run is logged automatically, not opt-in
--------------------------------------------------
A benchmark leaderboard is only trustworthy if it can't silently miss a
run — an engineer who forgot to click "log this" produces a leaderboard
that quietly under-represents whichever runs were inconvenient to log by
hand. log_run() is called directly from the two real GBRT training call
sites (app/main.py's _train_and_predict, used by every load_everything()
pipeline; app/_pages/import_page.py's upload-analysis pipeline) — there is
no manual logging step for a caller to forget.

PLATFORM_ORG_ID (org scoping)
------------------------------
Every other org-scoped table in db.py requires a real tenant org_id. The
three built-in reference-dataset training runs (NASA / synthetic /
Severson, trained once per process inside load_everything()'s
@st.cache_resource) belong to no single tenant — they are the platform's
own shared benchmark fleet, the same "shared reference cells" concept
src/api.py already uses when merging an org's uploaded fleet on top of
them. PLATFORM_ORG_ID (0) is a reserved sentinel for these; a real
tenant's own uploaded-data training runs are logged under that tenant's
actual org_id. leaderboard() takes both so a tenant's view combines its
own runs with the shared platform benchmark, mirroring src/api.py's
_get_featured_dfs()/_get_bundles() merge pattern.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import uuid
from dataclasses import dataclass

PLATFORM_ORG_ID = 0  # shared reference-dataset runs (NASA/synthetic/Severson)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class RunRecord:
    run_id: str
    org_id: int
    dataset: str
    chemistry: str
    feature_set: list[str]
    feature_version: str
    hyperparams: dict
    seed: int
    cell_ids: list[str]
    n_cells: int
    n_rows: int
    soh_mae: float
    soh_r2: float
    rul_mae: float
    rul_r2: float
    rul_reliable: bool
    fold_metrics: dict
    git_commit: str
    timestamp: str
    notes: "str | None" = None


_git_commit_cache: "str | None" = None


def _git_commit_hash() -> str:
    """Short git commit hash of the running code, or 'unknown' if this
    isn't a git checkout (e.g. some deployment environments) or git isn't
    installed. Cached per-process — the commit can't change mid-run."""
    global _git_commit_cache
    if _git_commit_cache is not None:
        return _git_commit_cache
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_REPO_ROOT,
        )
        _git_commit_cache = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        _git_commit_cache = "unknown"
    return _git_commit_cache


def log_run(
    org_id: int,
    dataset: str,
    chemistry: str,
    feature_set: list[str],
    feature_version: str,
    hyperparams: dict,
    seed: int,
    cell_ids: list[str],
    n_rows: int,
    lco_metrics: dict,
    notes: "str | None" = None,
) -> str:
    """
    Log one completed GBRT fit. `lco_metrics` is the dict returned by
    batlab.validation.lco.run_lco() (soh_mae/soh_r2/rul_mae/rul_r2/
    rul_reliable/per_cell). Returns the generated run_id.
    """
    import db

    timestamp = datetime.datetime.now().isoformat()
    run_id = f"{dataset}_{timestamp.replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:6]}"

    record = RunRecord(
        run_id=run_id,
        org_id=org_id,
        dataset=dataset,
        chemistry=chemistry,
        feature_set=list(feature_set),
        feature_version=feature_version,
        hyperparams=hyperparams,
        seed=seed,
        cell_ids=list(cell_ids),
        n_cells=len(cell_ids),
        n_rows=n_rows,
        soh_mae=lco_metrics.get("soh_mae"),
        soh_r2=lco_metrics.get("soh_r2"),
        rul_mae=lco_metrics.get("rul_mae"),
        rul_r2=lco_metrics.get("rul_r2"),
        rul_reliable=bool(lco_metrics.get("rul_reliable", False)),
        fold_metrics=lco_metrics.get("per_cell", {}),
        git_commit=_git_commit_hash(),
        timestamp=timestamp,
        notes=notes,
    )
    db.save_experiment_run(org_id, {
        "run_id":          record.run_id,
        "dataset":         record.dataset,
        "chemistry":       record.chemistry,
        "feature_set":     json.dumps(record.feature_set),
        "feature_version": record.feature_version,
        "hyperparams":     json.dumps(record.hyperparams),
        "seed":            record.seed,
        "cell_ids":        json.dumps(record.cell_ids),
        "n_cells":         record.n_cells,
        "n_rows":          record.n_rows,
        "soh_mae":         record.soh_mae,
        "soh_r2":          record.soh_r2,
        "rul_mae":         record.rul_mae,
        "rul_r2":          record.rul_r2,
        "rul_reliable":    int(record.rul_reliable),
        "fold_metrics":    json.dumps(record.fold_metrics),
        "git_commit":      record.git_commit,
        "timestamp":       record.timestamp,
        "notes":           record.notes,
    })
    return run_id


def get_run(org_id: int, run_id: str) -> "dict | None":
    """Look up one logged run by id, scoped to org_id (or PLATFORM_ORG_ID
    for a shared reference-dataset run)."""
    import db
    return db.get_experiment_run(org_id, run_id)


def leaderboard(
    tenant_org_id: "int | None",
    dataset: "str | None" = None,
    chemistry: "str | None" = None,
    sort_by: str = "rul_mae",
    ascending: bool = True,
) -> list[dict]:
    """
    Combined leaderboard: the shared platform benchmark runs (PLATFORM_ORG_ID)
    plus tenant_org_id's own uploaded-data runs (omit tenant_org_id to see
    only the shared platform runs). Optionally filtered by dataset/chemistry,
    sorted by any numeric column (default: rul_mae ascending — lowest error
    first). Rows with a None value in the sort column sort last regardless
    of `ascending`, so a run missing that metric never masquerades as the
    best (or worst) result.
    """
    import db

    org_ids = [PLATFORM_ORG_ID] if tenant_org_id is None else [PLATFORM_ORG_ID, tenant_org_id]
    runs = db.load_experiment_runs(org_ids)

    if dataset is not None:
        runs = [r for r in runs if r["dataset"] == dataset]
    if chemistry is not None:
        runs = [r for r in runs if r["chemistry"] == chemistry]

    def _sort_key(r: dict):
        val = r.get(sort_by)
        if val is None:
            return (True, 0.0)
        return (False, val if ascending else -val)

    runs.sort(key=_sort_key)
    return runs


def replay_run(org_id: int, run_id: str, cell_data: dict) -> dict:
    """
    Re-run the exact recorded leave-cell-out pipeline for a logged run
    against freshly supplied raw cycle data, reusing
    batlab.validation.manifest.evaluate_from_manifest() (the same
    seed/feature_version reproducibility guard used for split manifests)
    rather than re-implementing it.

    Parameters
    ----------
    org_id, run_id : identify the logged run (see get_run()).
    cell_data : {cell_id: DataFrame} — must contain every cell_id the run
        was originally trained on (run["cell_ids"]).

    Returns
    -------
    A dict with the reproduced metrics (from evaluate_from_manifest) plus
    "recorded" (the original logged soh_mae/soh_r2/rul_mae/rul_r2) so a
    caller can show reproduced-vs-recorded side by side, and "run" (the
    full logged run dict) for context.

    Raises
    ------
    ValueError if no run with this id exists, or if cell_data is missing
    a required cell (propagated from evaluate_from_manifest).
    """
    from batlab.validation.manifest import evaluate_from_manifest

    run = get_run(org_id, run_id)
    if run is None:
        raise ValueError(f"No logged run found for run_id={run_id!r} (org_id={org_id})")

    manifest = {
        "cell_ids":        run["cell_ids"],
        "feature_version": run["feature_version"],
        "seed":            run["seed"],
        "environment":     {},  # unknown for historical runs — evaluate_from_manifest treats absent as no-diff
    }
    result = evaluate_from_manifest(manifest, cell_data)
    result["recorded"] = {
        "soh_mae": run["soh_mae"], "soh_r2": run["soh_r2"],
        "rul_mae": run["rul_mae"], "rul_r2": run["rul_r2"],
    }
    result["run"] = run
    return result


# ---------------------------------------------------------------------------
# Reference-dataset reload — for replay_run() and the cross-chemistry study
# ---------------------------------------------------------------------------

REFERENCE_DATASETS = ("nasa", "synth", "severson")


def reload_reference_cell_data(dataset: str, cell_ids: "list[str] | None" = None) -> dict:
    """
    Reconstruct {cell_id: raw_cycles_DataFrame} for one of the platform's
    built-in reference datasets by calling the same loaders
    app/main.py's load_everything() uses. These three sources are always
    reproducibly reloadable (checked-in/cached CSVs, or a deterministic
    physics generator), unlike a tenant's uploaded data — see
    PLATFORM_ORG_ID's docstring above — so replay_run() never needs the
    raw input to have been separately persisted for these three.

    cell_ids restricts the reload to a specific run's population (e.g.
    run["cell_ids"]); omit to reload every cell currently available for
    that dataset.

    Raises ValueError for "uploaded" or any unrecognized dataset key —
    there is no reference loader for tenant-uploaded data (the raw
    uploaded cycles are never persisted, only the trained result — see
    src/bundle_cache.py's save_tenant_bundle()) — and for "severson" when
    the raw dataset isn't cached locally on this deployment.
    """
    if dataset == "synth":
        from data_loader import build_battery, CELL_STRESS_PROFILES
        ids = cell_ids or list(CELL_STRESS_PROFILES.keys())
        battery = build_battery(battery_id="Oxford_B1", cell_ids=ids)
        return {cid: cell["cycles"] for cid, cell in battery["cells"].items()}

    if dataset == "nasa":
        from data_loader import build_battery
        from batlab.datasets.nasa import CELL_IDS as _NASA_IDS
        ids = cell_ids or list(_NASA_IDS)
        battery = build_battery(battery_id="NASA_B1", cell_ids=ids)
        return {cid: cell["cycles"] for cid, cell in battery["cells"].items()}

    if dataset == "severson":
        from batlab.datasets.severson import load_severson_cells, any_cached
        if not any_cached():
            raise ValueError(
                "Severson raw data is not cached locally on this deployment — cannot reload for replay."
            )
        all_cells = load_severson_cells(status_fn=lambda msg: None)
        if cell_ids is None:
            return all_cells
        missing = [c for c in cell_ids if c not in all_cells]
        if missing:
            raise ValueError(f"Severson cache is missing {len(missing)} requested cell(s): {missing}")
        return {cid: all_cells[cid] for cid in cell_ids}

    raise ValueError(
        f"No reference reload available for dataset={dataset!r} — only "
        f"{REFERENCE_DATASETS} are reloadable; a tenant's uploaded-data raw "
        "cycles are never persisted, so an 'uploaded' run cannot be replayed."
    )
