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


# ---------------------------------------------------------------------------
# Cross-chemistry generalization study — honest zero-shot transfer error
# ---------------------------------------------------------------------------
#
# "Cross-dataset transfer" here means: fit one GBRT model on ALL of the
# training domain's cells (not leave-cell-out — there is no held-out cell
# from the *same* domain), then evaluate it, unmodified, on a completely
# different chemistry's cells it has never seen a single row of. This is a
# much harder ask than LCO (same chemistry, held-out cell) and is expected
# to produce a materially worse number — see
# src/battery_knowledge.py's "why-resistance-scales-differ" entry, which
# already documents that a combined NASA+synthetic model produced a
# negative R2 for exactly this reason (incompatible resistance scales
# across sources). The whole point of this study is to report that real
# number, however bad, not to engineer a flattering one.
#
# A model can only be evaluated on a feature vector shaped like the one it
# was trained on, so the model here is trained on the INTERSECTION of the
# two domains' available feature columns (get_model_matrix()'s own
# per-source availability filtering already makes NASA/Severson/synthetic
# each expose a different column set — see FEATURE_COLUMNS usage in
# batlab.features.engineering). If that intersection is too small to be a
# meaningful model, run_cross_chemistry_transfer() raises rather than
# silently reporting a number from 1-2 weak features.

MIN_COMMON_FEATURES_FOR_TRANSFER = 3


def run_cross_chemistry_transfer(
    train_dataset: str,
    train_cell_data: dict,
    eval_dataset: str,
    eval_cell_data: dict,
    org_id: int = PLATFORM_ORG_ID,
) -> dict:
    """
    Fit a GBRT model on train_cell_data (all cells, single fit — not LCO),
    zero-shot evaluate it on eval_cell_data, log the result to the
    registry under dataset=f"{train_dataset}_to_{eval_dataset}", and
    return {"run_id", "soh_mae", "soh_r2", "rul_mae", "rul_r2",
    "n_common_features"}.

    Raises ValueError if the two domains share fewer than
    MIN_COMMON_FEATURES_FOR_TRANSFER usable feature columns — a genuine
    schema incompatibility (e.g. Oxford's checkpoint-indexed data has no
    cycle_number/resistance_ohm/temperature_c at all) makes any number
    this function could produce scientifically meaningless, not just
    pessimistic; the honest response is to refuse, not to report a
    number built on 1-2 coincidentally-shared columns.
    """
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score
    from batlab.features.engineering import build_features, get_model_matrix, FEATURE_VERSION, FEATURE_COLUMNS
    from batlab.models.gbrt import train_models, predict, GBRT_PARAMS
    from chemistry_profiles import ChemistryProfile

    def _featured(cell_data: dict) -> dict:
        out = {}
        for cid, df in cell_data.items():
            feat = build_features(df)
            X, y_soh, y_rul = get_model_matrix(feat)
            out[cid] = (X, y_soh, y_rul)
        return out

    train_inputs = _featured(train_cell_data)
    eval_inputs  = _featured(eval_cell_data)

    train_cols = set(next(iter(train_inputs.values()))[0].columns)
    eval_cols  = set(next(iter(eval_inputs.values()))[0].columns)
    common = [c for c in FEATURE_COLUMNS if c in train_cols and c in eval_cols]

    if len(common) < MIN_COMMON_FEATURES_FOR_TRANSFER:
        raise ValueError(
            f"{train_dataset} and {eval_dataset} share only {len(common)} "
            f"usable feature column(s) ({common}) — fewer than the "
            f"{MIN_COMMON_FEATURES_FOR_TRANSFER} required for a meaningful "
            "cross-chemistry transfer evaluation. This is a real schema "
            "incompatibility (missing raw quantities like resistance or "
            "temperature in one domain), not something a larger dataset "
            "would fix."
        )

    X_train = pd.concat([X[common] for X, _, _ in train_inputs.values()])
    y_soh_train = pd.concat([y for _, y, _ in train_inputs.values()])
    y_rul_train = pd.concat([y for _, _, y in train_inputs.values()])
    bndl = train_models(X_train, y_soh_train, y_rul_train)

    X_eval = pd.concat([X[common] for X, _, _ in eval_inputs.values()])
    y_soh_eval = pd.concat([y for _, y, _ in eval_inputs.values()])
    y_rul_eval = pd.concat([y for _, _, y in eval_inputs.values()])
    preds = predict(bndl, X_eval)

    soh_mae = float(mean_absolute_error(y_soh_eval, preds["soh_pred"]))
    soh_r2  = float(r2_score(y_soh_eval, preds["soh_pred"]))
    rul_mae = float(mean_absolute_error(y_rul_eval, preds["rul_pred"]))
    rul_r2  = float(r2_score(y_rul_eval, preds["rul_pred"]))

    train_sample = next(iter(train_cell_data))
    eval_sample  = next(iter(eval_cell_data))
    dataset_key  = f"{train_dataset}_to_{eval_dataset}"

    run_id = log_run(
        org_id=org_id,
        dataset=dataset_key,
        chemistry=f"{ChemistryProfile.for_cell(train_sample).short_name} -> "
                  f"{ChemistryProfile.for_cell(eval_sample).short_name}",
        feature_set=common,
        feature_version=FEATURE_VERSION,
        hyperparams=dict(GBRT_PARAMS),
        seed=GBRT_PARAMS["random_state"],
        cell_ids=list(train_cell_data.keys()) + list(eval_cell_data.keys()),
        n_rows=len(X_train) + len(X_eval),
        lco_metrics={
            "soh_mae": soh_mae, "soh_r2": soh_r2,
            "rul_mae": rul_mae, "rul_r2": rul_r2,
            "rul_reliable": False,  # never claim reliable on an out-of-domain zero-shot transfer
            "per_cell": {},         # not leave-cell-out — a single train-on-all/eval-on-all split
        },
        notes=(
            f"Cross-chemistry generalization study: trained on {train_dataset} "
            f"({len(train_cell_data)} cells), zero-shot evaluated on "
            f"{eval_dataset} ({len(eval_cell_data)} cells), using the "
            f"{len(common)} feature column(s) both domains have available "
            f"({', '.join(common)}). Not leave-cell-out — a single "
            "train-on-all-of-one-domain / eval-on-all-of-the-other split. "
            "Report the number honestly, including a poor one — see "
            "src/battery_knowledge.py's why-resistance-scales-differ entry "
            "for why cross-chemistry transfer is expected to be weak."
        ),
    )

    return {
        "run_id": run_id, "soh_mae": soh_mae, "soh_r2": soh_r2,
        "rul_mae": rul_mae, "rul_r2": rul_r2, "n_common_features": len(common),
    }


def cross_chemistry_runs_for_train_dataset(train_dataset: str, tenant_org_id: "int | None" = None) -> list[dict]:
    """
    Every logged cross-chemistry transfer run (evaluated or "not
    evaluated") whose training side was train_dataset — e.g.
    cross_chemistry_runs_for_train_dataset("nasa") returns the
    nasa_to_severson / nasa_to_oxford rows. Used by the Overview page's
    cross-dataset transfer-error honesty badge to show, next to a NASA
    cell's RUL sample-size badge, how well the NASA-trained model
    generalizes to chemistries it wasn't trained on.
    """
    all_runs = leaderboard(tenant_org_id=tenant_org_id)
    prefix = f"{train_dataset}_to_"
    return [r for r in all_runs if r["dataset"].startswith(prefix)]


def log_cross_chemistry_unavailable(
    train_dataset: str,
    eval_dataset: str,
    reason: str,
    org_id: int = PLATFORM_ORG_ID,
) -> str:
    """
    Log an honest "attempted but not possible" record for a cross-chemistry
    pairing that can't be evaluated at all (e.g. Oxford's checkpoint-
    indexed schema has no cycle_number/resistance_ohm/temperature_c, so
    run_cross_chemistry_transfer() would have to refuse). Every metric is
    None (not a fabricated number, not silently skipped) — the Benchmark
    leaderboard shows this row with "—" for every metric and `reason` in
    its notes, same transparency the rest of this registry gives every
    other run.
    """
    return log_run(
        org_id=org_id,
        dataset=f"{train_dataset}_to_{eval_dataset}",
        chemistry="not evaluated",
        feature_set=[],
        feature_version="n/a",
        hyperparams={},
        seed=0,
        cell_ids=[],
        n_rows=0,
        lco_metrics={
            "soh_mae": None, "soh_r2": None, "rul_mae": None, "rul_r2": None,
            "rul_reliable": False, "per_cell": {},
        },
        notes=f"Not evaluated: {reason}",
    )
