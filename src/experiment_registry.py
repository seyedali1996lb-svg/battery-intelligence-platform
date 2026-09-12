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

The cross-chemistry benchmark is permanent, not a one-off study
----------------------------------------------------------------
run_cross_chemistry_study() fits and logs the zero-shot transfer for every
ordered pair of available datasets whose chemistries differ, and
app/_data.py's load_everything() calls it so the counterexample is a
standing, visible part of the Benchmark (see cross_chemistry_benchmark())
rather than a number someone has to re-derive from a script. It is
idempotent per FEATURE_VERSION, so a warm deployment pays a registry read,
and it re-runs exactly when the feature set changes — the one thing that
could legitimately move the number. Pairings that cannot be evaluated at
all are logged as honest "not evaluated" rows instead of being omitted.

The replay contract — which RunRecord fields replay_run() actually uses
------------------------------------------------------------------------
RunRecord stores 15 fields, but replay_run() (via
batlab.validation.manifest.evaluate_from_manifest() -> batlab.validation.
lco.run_lco()) only ever feeds THREE of them back into the actual
re-computation:

  - cell_ids         (which cells' raw data to restrict to)
  - feature_version  (raises if it doesn't match the installed batlab's
                       FEATURE_VERSION — a stale-feature-code guard)
  - seed             (passed through to run_lco()'s GradientBoostingRegressor)

Every other field — dataset, chemistry, feature_set, hyperparams, n_cells,
n_rows, fold_metrics, git_commit, timestamp, notes — is descriptive/audit
metadata: shown on the Benchmark leaderboard and the "Regenerate this
report" caption, logged for provenance, but NOT threaded into the replay
computation itself. In particular, **hyperparams is not actually
replayed**, despite reading as if it might be: run_lco() always trains
with the current module-level GBRT_PARAMS (batlab.models.gbrt.GBRT_PARAMS,
re-exported as batlab.validation.lco.GBRT_PARAMS — one shared object, not
two independently-tunable copies) — it has no hyperparams argument at all.
A run's own logged "hyperparams" field is a snapshot dict taken at
training time by the live call sites in app/main.py / app/_pages/
import_page.py, so it reflects whatever GBRT_PARAMS held back then, not
necessarily what it holds now.

This is a real, structurally-live fragility even with the constant
unified: if GBRT_PARAMS is ever tuned going forward, replay_run() would
silently train an old run's replay with different hyperparameters than
that run actually used, and the "reproduced" vs "recorded" numbers on the
Regenerate-report UI could diverge for a reason that has nothing to do
with code/environment/data drift — the three things this contract's
guards actually check. replay_run() below adds a hyperparams_match/
hyperparams_diff check (same non-raising style as evaluate_from_manifest()'s
environment_match/environment_diff) so that divergence is visible instead
of silently misattributed.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import uuid
import numpy as np
from dataclasses import dataclass

PLATFORM_ORG_ID = 0  # shared reference-dataset runs (NASA/synthetic/Severson)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class RunRecord:
    run_id: str
    org_id: int
    dataset: str
    chemistry: str
    model_kind: str
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
    baseline_soh_r2: float | None
    rul_reliable: bool
    fold_metrics: dict
    baseline_per_cell: dict | None
    git_commit: str
    timestamp: str
    notes: "str | None" = None
    # Tier-0 RUL-label-provenance fields (FEATURE_VERSION v12+). rul_r2 above
    # is observed-EOL rows only from v12 on; these carry the honest denominator
    # and the formula baseline next to it.
    rul_formula_baseline_r2: "float | None" = None
    rul_baseline_pool: "str | None" = None
    rul_label_coverage: "float | None" = None
    n_rul_observed_rows: "int | None" = None
    n_rul_extrapolated_rows: "int | None" = None
    # Tier-1: fold-level bootstrap CIs (batlab.validation.bootstrap) — the
    # spread of the headline mean across leave-cell-out folds, stored JSON.
    ci_intervals: "dict | None" = None
    calibration_meta: "dict | None" = None


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
    model_kind: str = "gbrt",
) -> str:
    """
    Log one completed GBRT fit. `lco_metrics` is the dict returned by
    batlab.validation.lco.run_lco() (soh_mae/soh_r2/rul_mae/rul_r2/
    rul_reliable/per_cell). `baseline_r2` is the trivial linear baseline
    (cycle_number -> SOH) R² under the same leave-cell-out folds, from
    batlab.validation.trivial_baseline.baseline_lco_r2() — the honest
    "how much of this R² is smooth curves vs. the model" denominator.
    Returns the generated run_id.
    """
    import db

    timestamp = datetime.datetime.now().isoformat()
    run_id = f"{dataset}_{timestamp.replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:6]}"

    record = RunRecord(
        run_id=run_id,
        org_id=org_id,
        dataset=dataset,
        chemistry=chemistry,
        model_kind=model_kind,
        feature_set=list(feature_set),
        feature_version=feature_version,
        hyperparams=hyperparams,
        seed=seed,
        cell_ids=list(cell_ids),
        n_cells=len(cell_ids),
        n_rows=n_rows,
        soh_mae=lco_metrics.get("soh_mae"),  # pyright: ignore[reportArgumentType]
        soh_r2=lco_metrics.get("soh_r2"),  # pyright: ignore[reportArgumentType]
        rul_mae=lco_metrics.get("rul_mae"),  # pyright: ignore[reportArgumentType]
        rul_r2=lco_metrics.get("rul_r2"),  # pyright: ignore[reportArgumentType]
        baseline_soh_r2=(lco_metrics.get("baseline_soh_r2") if isinstance(lco_metrics, dict) else None),
        rul_reliable=bool(lco_metrics.get("rul_reliable", False)),
        fold_metrics=lco_metrics.get("per_cell", {}),
        baseline_per_cell=(lco_metrics.get("baseline_per_cell") if isinstance(lco_metrics, dict) else None),
        rul_formula_baseline_r2=(lco_metrics.get("rul_formula_baseline_r2") if isinstance(lco_metrics, dict) else None),
        rul_baseline_pool=(lco_metrics.get("rul_baseline_pool") if isinstance(lco_metrics, dict) else None),
        rul_label_coverage=(lco_metrics.get("rul_label_coverage") if isinstance(lco_metrics, dict) else None),
        n_rul_observed_rows=(lco_metrics.get("n_rul_observed_rows") if isinstance(lco_metrics, dict) else None),
        n_rul_extrapolated_rows=(lco_metrics.get("n_rul_extrapolated_rows") if isinstance(lco_metrics, dict) else None),
        ci_intervals=(lco_metrics.get("confidence_intervals") if isinstance(lco_metrics, dict) else None),
        calibration_meta=(lco_metrics.get("calibration_meta") if isinstance(lco_metrics, dict) else None),
        git_commit=_git_commit_hash(),
        timestamp=timestamp,
        notes=notes,
    )
    db.save_experiment_run(org_id, {
        "run_id":          record.run_id,
        "dataset":         record.dataset,
        "chemistry":       record.chemistry,
        "model_kind":      record.model_kind,
        "feature_set":     json.dumps(record.feature_set),
        "feature_version": record.feature_version,
        "hyperparams":     json.dumps(record.hyperparams),
        "seed":            record.seed,
        "cell_ids":        json.dumps(record.cell_ids),
        "n_cells":         record.n_cells,
        "n_rows":          record.n_rows,
        "soh_mae":         record.soh_mae,
        "soh_r2":          record.soh_r2,
        "baseline_soh_r2": record.baseline_soh_r2,
        "rul_mae":         record.rul_mae,
        "rul_r2":          record.rul_r2,
        "rul_reliable":    int(record.rul_reliable),
        "fold_metrics":    json.dumps(record.fold_metrics),
        "baseline_per_cell": (json.dumps(record.baseline_per_cell)
                              if record.baseline_per_cell is not None else None),
        "rul_formula_baseline_r2": record.rul_formula_baseline_r2,
        "rul_baseline_pool": record.rul_baseline_pool,
        "rul_label_coverage": record.rul_label_coverage,
        "n_rul_observed_rows": record.n_rul_observed_rows,
        "n_rul_extrapolated_rows": record.n_rul_extrapolated_rows,
        "ci_intervals": (json.dumps(record.ci_intervals) if record.ci_intervals is not None else None),
        "calibration_meta": (json.dumps(record.calibration_meta) if record.calibration_meta is not None else None),
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


def hyperparams_divergence(run: "dict | None") -> dict:
    """
    Compare a logged run's recorded GBRT hyperparameters against the ones
    the installed code would actually train with today — the single source
    of truth for this comparison, shared by replay_run(), the model card,
    and the "Regenerate this report" UI so those three can never disagree
    about whether a run is still faithfully reproducible.

    Returns {param: (recorded, current)} for every differing key — an empty
    dict means the run would be reproduced with the same hyperparameters it
    was trained with. See this module's "The replay contract" docstring for
    why this check exists at all: run_lco() has no hyperparams argument, so
    replay always uses the current module-level GBRT_PARAMS regardless of
    what the run recorded.
    """
    from batlab.validation.lco import GBRT_PARAMS as _current
    recorded = (run or {}).get("hyperparams") or {}
    if not isinstance(recorded, dict):
        recorded = {}
    return {
        k: (recorded.get(k), v)
        for k, v in _current.items()
        if recorded.get(k) != v
    }


def format_hyperparams_diff(diff: dict) -> str:
    """Human-readable one-liner for a hyperparams_divergence() dict, e.g.
    "n_estimators: 100 → 200; learning_rate: 0.05 → 0.1". Returns "" when
    there is no divergence."""
    return "; ".join(f"{k}: {rec} → {cur}" for k, (rec, cur) in (diff or {}).items())


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


def accuracy_by_source(tenant_org_id: "int | None" = None) -> list[dict]:
    """
    Per-source accuracy summary for the Benchmark page: one row per
    (dataset, chemistry) LCO run, showing the model R², the trivial
    baseline R², and the model's real advantage over it.

    Grouped by (dataset, chemistry), not chemistry alone — the synthetic
    reference fleet and the NASA PCoE cells are BOTH chemistry "LiCoO2", so
    grouping by chemistry would silently conflate a simulated fleet with real
    measured cells. Keeping the dataset in the group key keeps a real-vs-
    simulated comparison honest while still making the NASA-vs-Severson LFP
    contrast directly visible.

    Only runs that actually carry a trivial baseline are included. A
    cross-chemistry transfer run has no baseline by construction (a single
    train-on-one-domain / eval-on-the-other split, not leave-cell-out), so it
    is excluded rather than shown with a fabricated "—"-derived advantage.
    Where several runs exist for the same group (repeated processes, or a
    retrain), the NEWEST is reported — the older duplicates are superseded by
    definition.

    Returns a list sorted by real advantage descending (largest genuine model
    skill first), each entry:
        {
          "dataset", "chemistry", "n_cells", "feature_version",
          "timestamp", "soh_r2", "baseline_soh_r2", "advantage",
          "rul_r2", "rul_reliable",
        }
    """
    runs = leaderboard(tenant_org_id=tenant_org_id)

    latest: dict = {}
    for r in runs:
        if r.get("soh_r2") is None or r.get("baseline_soh_r2") is None:
            continue  # not an LCO run with a baseline — nothing comparable to report
        # Prospective-split runs evaluate a DIFFERENT population (each cell's
        # test window under a temporal holdout, not leave-cell-out folds) —
        # they are the per-chemistry table's complement, reported in their
        # own section, and must never be averaged into these rows.
        if (r.get("dataset") or "").endswith("_prospective"):
            continue
        # GBRT only: this table is the per-chemistry accuracy of the
        # production model. PINN runs are reported through
        # model_kind_comparison() instead, so a physics-estimator number can
        # never be averaged into — or mistaken for — the GBRT's accuracy.
        if (r.get("model_kind") or "gbrt") != "gbrt":
            continue
        key = (r.get("dataset"), r.get("chemistry"))
        prev = latest.get(key)
        if prev is None or (r.get("timestamp") or "") > (prev.get("timestamp") or ""):
            latest[key] = r

    rows = []
    for r in latest.values():
        soh_r2 = float(r["soh_r2"])
        baseline = float(r["baseline_soh_r2"])
        # ── Honest RUL gate (Tier-0) ──
        # rul_reliable from v12 on already encodes "observed-row R² above the
        # floor AND observed coverage above the floor". For older runs (and
        # as defense in depth) a stored reliable flag must also survive two
        # checks we can always apply here: the run's RUL R² must beat the
        # formula baseline (when one was computed) and the run must have
        # declared an observed-coverage fraction at all.
        rul_r2 = r.get("rul_r2")
        formula_baseline = r.get("rul_formula_baseline_r2")
        beats_formula = (
            rul_r2 is None or formula_baseline is None
            or (not (isinstance(rul_r2, float) and rul_r2 != rul_r2) and float(rul_r2) > float(formula_baseline))
        )
        coverage = r.get("rul_label_coverage")
        rul_ok = bool(r.get("rul_reliable")) and beats_formula and (coverage is None or float(coverage) >= 0.5)
        rows.append({
            "dataset":          r.get("dataset"),
            "chemistry":        r.get("chemistry") or "—",
            "n_cells":          r.get("n_cells"),
            "feature_version":  r.get("feature_version"),
            "timestamp":        r.get("timestamp"),
            "soh_r2":           soh_r2,
            "baseline_soh_r2":  baseline,
            "advantage":        soh_r2 - baseline,
            "rul_r2":           rul_r2,
            "rul_formula_baseline_r2": formula_baseline,
            "rul_label_coverage": coverage,
            "n_rul_observed_rows": r.get("n_rul_observed_rows"),
            "ci_intervals":     r.get("ci_intervals"),
            "rul_reliable":     rul_ok,
        })
    rows.sort(key=lambda d: d["advantage"], reverse=True)
    return rows


MODEL_KIND_LABELS = {
    "gbrt": "GBRT (production)",
    "pinn": "PINN (physics-regularized)",
}


def model_kind_comparison(tenant_org_id: "int | None" = None) -> list[dict]:
    """
    GBRT vs PINN, side by side, on the SAME leave-cell-out population.

    Both model kinds are fitted and evaluated by the same harness shape
    (leave-one-cell-out, identical folds, identical trivial baseline), so the
    only difference between the two numbers is the model itself. This is the
    honest answer to "is the physics-regularized estimator actually better?"
    — including when it is not.

    One row per (dataset, chemistry, model_kind), newest run of each kind,
    plus a `delta` field on the GBRT row of each group expressing the PINN's
    SOH R² minus the GBRT's. A negative delta is reported as-is (the PINN
    lost), never hidden. Returns [] when nothing is logged.
    """
    runs = leaderboard(tenant_org_id=tenant_org_id)

    latest: dict = {}
    for r in runs:
        if r.get("soh_r2") is None:
            continue
        # Prospective-split rows compare against themselves (their own
        # section) — a different population than LCO folds.
        if (r.get("dataset") or "").endswith("_prospective"):
            continue
        key = (r.get("dataset"), r.get("chemistry"), r.get("model_kind") or "gbrt")
        prev = latest.get(key)
        if prev is None or (r.get("timestamp") or "") > (prev.get("timestamp") or ""):
            latest[key] = r

    # Group by (dataset, chemistry) so the two model kinds sit together.
    groups: dict = {}
    for (dataset, chemistry, kind), r in latest.items():
        groups.setdefault((dataset, chemistry), {})[kind] = r

    rows = []
    for (dataset, chemistry), kinds in groups.items():
        gbrt = kinds.get("gbrt")
        pinn = kinds.get("pinn")
        delta = None
        if gbrt is not None and pinn is not None:
            try:
                delta = float(pinn["soh_r2"]) - float(gbrt["soh_r2"])
            except (TypeError, ValueError):
                delta = None
        for kind, r in (("gbrt", gbrt), ("pinn", pinn)):
            if r is None:
                continue
            baseline = r.get("baseline_soh_r2")
            rows.append({
                "dataset":       dataset,
                "chemistry":     chemistry or "—",
                "model_kind":    kind,
                "model_label":   MODEL_KIND_LABELS.get(kind, kind),
                "n_cells":       r.get("n_cells"),
                "n_rows":        r.get("n_rows"),
                "soh_r2":        r.get("soh_r2"),
                "soh_mae":       r.get("soh_mae"),
                "baseline_soh_r2": baseline,
                "advantage": (None if baseline is None
                              else float(r["soh_r2"]) - float(baseline)),
                "rul_r2":        r.get("rul_r2"),
                "rul_mae":       r.get("rul_mae"),
                "rul_reliable":  bool(r.get("rul_reliable")),
                "hyperparams":   r.get("hyperparams") or {},
                "feature_version": r.get("feature_version"),
                "timestamp":     r.get("timestamp"),
                "notes":         r.get("notes"),
                # Carried on the PINN row, where it reads naturally: PINN R²
                # minus GBRT R². The GBRT row shows no delta (it is the
                # reference). Negative means the physics estimator lost.
                "pinn_minus_gbrt_soh_r2": (delta if kind == "pinn" else None),
                "has_counterpart": (gbrt is not None if kind == "pinn" else pinn is not None),
            })
    rows.sort(key=lambda d: (d["dataset"] or "", d["model_kind"] == "pinn"))
    return rows


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
    caller can show reproduced-vs-recorded side by side, "run" (the full
    logged run dict) for context, and "hyperparams_match"/"hyperparams_diff"
    — see this module's docstring ("The replay contract") for why this
    check exists: the actual replay always trains with the CURRENT
    batlab.models.gbrt.GBRT_PARAMS, not the run's own logged "hyperparams"
    field (a snapshot taken at training time), so this surfaces any
    divergence between the two instead of leaving it invisible.

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

    hyperparams_diff = hyperparams_divergence(run)
    result["hyperparams_match"] = not hyperparams_diff
    result["hyperparams_diff"] = hyperparams_diff
    return result


# ---------------------------------------------------------------------------
# Reference-dataset reload — for replay_run() and the cross-chemistry study
# ---------------------------------------------------------------------------

REFERENCE_DATASETS = ("nasa", "synth", "severson", "zhu2022", "calce")


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

    if dataset == "zhu2022":
        from batlab.datasets.zhu2022 import load_zhu2022_cells
        all_cells = load_zhu2022_cells(status_fn=lambda msg: None)
        if not all_cells:
            raise ValueError(
                "Zhu 2022 data is not cached locally on this deployment — cannot reload for replay."
            )
        if cell_ids is None:
            return all_cells
        missing = [c for c in cell_ids if c not in all_cells]
        if missing:
            raise ValueError(f"Zhu 2022 cache is missing {len(missing)} requested cell(s): {missing}")
        return {cid: all_cells[cid] for cid in cell_ids}

    if dataset == "calce":
        from batlab.datasets.calce import load_calce_cells, CalceDataNotFoundError
        try:
            all_cells = load_calce_cells()
        except CalceDataNotFoundError as exc:
            raise ValueError(
                "CALCE data requires a manual download on this deployment — "
                "cannot reload for replay."
            ) from exc
        if cell_ids is None:
            return all_cells
        missing = [c for c in cell_ids if c not in all_cells]
        if missing:
            raise ValueError(f"CALCE cache is missing {len(missing)} requested cell(s): {missing}")
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
    train_featured: "dict | None" = None,
    eval_featured: "dict | None" = None,
) -> dict:
    """
    Fit a GBRT model on train_cell_data (all cells, single fit — not LCO),
    zero-shot evaluate it on eval_cell_data, log the result to the
    registry under dataset=f"{train_dataset}_to_{eval_dataset}", and
    return {"run_id", "soh_mae", "soh_r2", "rul_mae", "rul_r2",
    "n_common_features"}.

    train_featured / eval_featured are optional {cell_id: (X, y_soh, y_rul)}
    dicts (the shape bundle_cache.load_features_cached() returns as its
    second element) so a caller that has already run build_features() —
    load_everything()'s reference pipelines — can reuse those frames
    instead of paying for the full feature pipeline (including the
    PyBaMM-backed physics calibration) a second time. Omit either and the
    corresponding domain is featured here.

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
            feat = build_features(df, cell_id=cid)
            X, y_soh, y_rul = get_model_matrix(feat)
            out[cid] = (X, y_soh, y_rul)
        return out

    train_inputs = train_featured if train_featured else _featured(train_cell_data)
    eval_inputs  = eval_featured if eval_featured else _featured(eval_cell_data)

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
    bndl = train_models(X_train, y_soh_train, y_rul_train)  # pyright: ignore[reportArgumentType]

    X_eval = pd.concat([X[common] for X, _, _ in eval_inputs.values()])
    y_soh_eval = pd.concat([y for _, y, _ in eval_inputs.values()])
    y_rul_eval = pd.concat([y for _, _, y in eval_inputs.values()])
    preds = predict(bndl, X_eval)

    soh_mae = float(mean_absolute_error(y_soh_eval, preds["soh_pred"]))
    soh_r2  = float(r2_score(y_soh_eval, preds["soh_pred"]))

    # ── RUL split by label provenance (same rule as run_lco, FEATURE_VERSION >= v12) ──
    # rul_label_kind is constant WITHIN a cell — a cell either reached EOL in-window
    # (all rows observed) or it did not (all rows formula-extrapolated) — so the eval
    # pool's kinds can be derived from the raw cycle tables without re-featuring.
    # Reporting the extrapolated pool's R² as transfer skill would grade formula
    # recovery (Severson eval rows were 100% extrapolated: the old -1.11 was scored
    # against labels GENERATED by fade_rate_50cy, itself a model feature).
    def _cell_label_kind(raw_df) -> str:
        eol_capacity = float(raw_df["capacity_ah"].iloc[0]) * 0.80
        return "observed" if bool((raw_df["capacity_ah"] <= eol_capacity).any()) else "extrapolated"

    obs_mask_parts = []
    for cid, (X_c, _ys, _yr) in eval_inputs.items():
        kind = _cell_label_kind(eval_cell_data[cid])
        obs_mask_parts.append(np.full(len(X_c), kind == "observed", dtype=bool))
    obs_mask = np.concatenate(obs_mask_parts) if obs_mask_parts else np.array([], dtype=bool)

    rul_true = np.asarray(y_rul_eval, dtype=float)
    rul_pred = np.asarray(preds["rul_pred"], dtype=float)
    n_obs = int(obs_mask.sum())
    n_ext = int(len(obs_mask) - n_obs)

    def _pooled_r2(mask) -> float:
        if int(mask.sum()) < 2 or float(np.var(rul_true[mask])) <= 0.0:
            return float("nan")
        return float(r2_score(rul_true[mask], rul_pred[mask]))

    rul_r2 = _pooled_r2(obs_mask)              # headline: measured labels only
    rul_r2_extrapolated = _pooled_r2(~obs_mask)  # formula-recovery diagnostic
    rul_mae = (
        float(mean_absolute_error(rul_true[obs_mask], rul_pred[obs_mask]))
        if n_obs >= 2 else float("nan")
    )

    train_sample = next(iter(train_cell_data))
    eval_sample  = next(iter(eval_cell_data))
    dataset_key  = f"{train_dataset}_to_{eval_dataset}"
    train_chem   = ChemistryProfile.for_cell(train_sample).short_name
    eval_chem    = ChemistryProfile.for_cell(eval_sample).short_name
    same_chem    = train_chem == eval_chem
    study_kind   = (
        "Same-chemistry cross-source generalization study"
        if same_chem else "Cross-chemistry generalization study"
    )

    run_id = log_run(
        org_id=org_id,
        dataset=dataset_key,
        chemistry=f"{train_chem} -> {eval_chem}",
        feature_set=common,
        feature_version=FEATURE_VERSION,
        hyperparams=dict(GBRT_PARAMS),
        seed=GBRT_PARAMS["random_state"],  # pyright: ignore[reportArgumentType]
        cell_ids=list(train_cell_data.keys()) + list(eval_cell_data.keys()),
        n_rows=len(X_train) + len(X_eval),
        lco_metrics={
            "soh_mae": soh_mae, "soh_r2": soh_r2,
            "baseline_soh_r2": None,  # not leave-cell-out — baseline_lco_r2 requires LCO folds
            "rul_mae": rul_mae, "rul_r2": rul_r2,
            "rul_reliable": False,  # never claim reliable on an out-of-domain zero-shot transfer
            "per_cell": {},         # not leave-cell-out — a single train-on-all/eval-on-all split
            "baseline_per_cell": None,
        },
        notes=(
            f"{study_kind}: trained on {train_dataset} "
            f"({len(train_cell_data)} cells), zero-shot evaluated on "
            f"{eval_dataset} ({len(eval_cell_data)} cells), using the "
            f"{len(common)} feature column(s) both domains have available "
            f"({', '.join(common)}). Not leave-cell-out — a single "
            "train-on-all-of-one-domain / eval-on-all-of-the-other split. "
            f"RUL R² is computed on observed-EOL rows only ({n_obs} of "
            f"{n_obs + n_ext} eval rows; nan means the eval domain has no "
            "measured-EOL rows, so its RUL labels are formula extrapolations "
            "and no RUL transfer claim is evaluable there). "
            "Report the number honestly, including a poor one — see "
            "src/battery_knowledge.py's why-resistance-scales-differ entry "
            "for why cross-chemistry transfer is expected to be weak."
        ),
    )

    return {
        "run_id": run_id, "soh_mae": soh_mae, "soh_r2": soh_r2,
        "rul_mae": rul_mae, "rul_r2": rul_r2, "n_common_features": len(common),
        "n_rul_observed_rows": n_obs,
        "n_rul_extrapolated_rows": n_ext,
        "rul_r2_extrapolated": rul_r2_extrapolated,
    }


def dataset_chemistry(cell_data: dict) -> str:
    """The chemistry short-name of a dataset, taken from its first cell —
    used to decide which dataset pairings are genuinely cross-chemistry.
    Raises if cell_data is empty."""
    from chemistry_profiles import ChemistryProfile
    return ChemistryProfile.for_cell(next(iter(cell_data))).short_name


def run_cross_chemistry_study(
    datasets: dict,
    featured: "dict | None" = None,
    org_id: int = PLATFORM_ORG_ID,
    refresh: bool = False,
    unavailable: "list[tuple[str, str, str]] | None" = None,
) -> dict:
    """
    Run and log the zero-shot transfer study for EVERY ordered pair of the
    supplied datasets, so the resulting map is a permanent, visible part of
    the platform's benchmark rather than a one-off script output.

    Two kinds of pair, both informative:
    - cross-chemistry (NASA LiCoO2 -> Severson LFP): tests whether a model
      transfers across cathode chemistries. The historic result is a
      catastrophic failure, and that failure is the point.
    - same-chemistry cross-source (NASA -> CALCE, both LiCoO2 but different
      form factor, cycler and aging protocol): tests whether a model
      generalizes beyond its own dataset's cells even when the chemistry
      matches. This is the honest "will it work on a cell I haven't seen?"
      question WITHIN one chemistry.

    Parameters
    ----------
    datasets : {dataset_key: {cell_id: raw_cycles_df}} — the reference
        datasets available on this deployment (e.g. nasa/severson/synth).
    featured : optional {dataset_key: {cell_id: (X, y_soh, y_rul)}} so an
        already-featured domain isn't re-featured (see
        run_cross_chemistry_transfer()).
    refresh : re-run a pairing that is already logged for the current
        FEATURE_VERSION. False (the default) makes this safe to call on
        every process start — it becomes a cheap registry read.
    unavailable : optional [(train_dataset, eval_dataset, reason)] pairings
        that cannot be evaluated at all (schema incompatibility). These are
        logged as honest "not evaluated" rows instead of being silently
        omitted.

    Idempotence is keyed two different ways, deliberately: an *evaluated*
    pairing is re-run when FEATURE_VERSION changes (the number itself
    depends on the feature set), while an *unavailable* pairing is logged
    once ever — its reason is a schema fact about the two domains, not a
    function of the feature code.

    Returns {"evaluated": [...], "unavailable": [...], "skipped": [...]}.
    """
    from batlab.features.engineering import FEATURE_VERSION

    all_runs = leaderboard(tenant_org_id=None)
    logged_current = {(r["dataset"], r["feature_version"]) for r in all_runs}
    logged_any_pair = {r["dataset"] for r in all_runs}

    evaluated: list[dict] = []
    unavailable_logged: list[dict] = []
    skipped: list[str] = []

    def _log_unavailable(train_ds: str, eval_ds: str, reason: str) -> None:
        pair = f"{train_ds}_to_{eval_ds}"
        log_cross_chemistry_unavailable(train_ds, eval_ds, reason, org_id=org_id)
        logged_any_pair.add(pair)
        unavailable_logged.append({"pair": pair, "reason": reason})

    for train_ds, train_cells in datasets.items():
        for eval_ds, eval_cells in datasets.items():
            if train_ds == eval_ds or not train_cells or not eval_cells:
                continue
            try:
                # Chemistry is still computed per pair — the transfer function
                # uses it to label cross-chemistry vs same-chemistry runs —
                # but same-chemistry pairs are no longer skipped: with more
                # than one real source per chemistry (NASA + CALCE for
                # LiCoO2), cross-SOURCE transfer within a chemistry is its
                # own generalization test.
                dataset_chemistry(train_cells)
                dataset_chemistry(eval_cells)
            except Exception:
                continue

            pair = f"{train_ds}_to_{eval_ds}"
            if not refresh and (pair, FEATURE_VERSION) in logged_current:
                skipped.append(pair)
                continue
            try:
                result = run_cross_chemistry_transfer(
                    train_ds, train_cells, eval_ds, eval_cells, org_id=org_id,
                    train_featured=(featured or {}).get(train_ds),
                    eval_featured=(featured or {}).get(eval_ds),
                )
            except ValueError as exc:
                # A genuine schema incompatibility (too few shared feature
                # columns) — record it honestly rather than dropping the pair.
                if not refresh and pair in logged_any_pair:
                    skipped.append(pair)
                else:
                    _log_unavailable(train_ds, eval_ds, str(exc))
                continue
            evaluated.append(result)
            logged_current.add((pair, FEATURE_VERSION))

    for train_ds, eval_ds, reason in (unavailable or []):
        pair = f"{train_ds}_to_{eval_ds}"
        if not refresh and pair in logged_any_pair:
            skipped.append(pair)
            continue
        _log_unavailable(train_ds, eval_ds, reason)

    return {"evaluated": evaluated, "unavailable": unavailable_logged, "skipped": skipped}


def run_pinn_benchmark_study(
    datasets: dict,
    featured: "dict | None" = None,
    baselines: "dict | None" = None,
    org_id: int = PLATFORM_ORG_ID,
    refresh: bool = False,
) -> list[dict]:
    """
    Run and log the PINN physics estimator through the SAME leave-cell-out
    harness as the GBRT, for every supplied reference dataset, so the two
    model kinds sit side by side in the leaderboard with a comparable number
    — including when the PINN loses, which is a result too.

    Parameters
    ----------
    datasets : {dataset_key: {cell_id: raw_cycles_df}}.
    featured : optional {dataset_key: {cell_id: df_feat}} so the (expensive)
        feature build is not repeated.
    baselines : optional {dataset_key: trivial_baseline_soh_r2}. The baseline
        is model-independent (cycle_number -> SOH under identical folds), so
        the GBRT's already-computed denominator applies to the PINN row
        unchanged — the comparison is then apples-to-apples.
    refresh : re-run a dataset already logged for the current FEATURE_VERSION.
        False (the default) makes this a cheap registry read on a warm start.

    Returns the list of newly logged PINN results ([] when everything was
    already up to date).
    """
    from batlab.features.engineering import FEATURE_VERSION
    from batlab.validation.pinn_lco import run_pinn_lco, MODEL_KIND, pinn_hyperparams
    from chemistry_profiles import ChemistryProfile

    all_runs = leaderboard(tenant_org_id=None)
    logged = {
        (r["dataset"], r["feature_version"])
        for r in all_runs
        if (r.get("model_kind") or "gbrt") == MODEL_KIND
    }

    out: list[dict] = []
    for key, cell_cycles in (datasets or {}).items():
        if not cell_cycles:
            continue
        if not refresh and (key, FEATURE_VERSION) in logged:
            continue
        try:
            metrics = run_pinn_lco(
                cell_cycles,
                featured=(featured or {}).get(key),
            )
        except Exception:
            # A dataset the physics fit genuinely cannot run on is skipped,
            # not logged with a fabricated number.
            continue

        baseline = (baselines or {}).get(key)
        metrics = {
            **metrics,
            "baseline_soh_r2": baseline,
            "baseline_per_cell": None,
        }
        sample_cell = next(iter(cell_cycles))
        n_rows = sum(len(df) for df in cell_cycles.values())
        run_id = log_run(
            org_id=org_id,
            dataset=key,
            chemistry=ChemistryProfile.for_cell(sample_cell).short_name,
            feature_set=["cycle_number", "soh_pct"],  # the PINN's actual inputs
            feature_version=FEATURE_VERSION,
            hyperparams=pinn_hyperparams(),
            seed=42,
            cell_ids=list(cell_cycles.keys()),
            n_rows=n_rows,
            lco_metrics=metrics,
            model_kind=MODEL_KIND,
            notes=(
                "PINN physics-regularized estimator benchmarked through the "
                "SAME leave-cell-out folds as the GBRT (see "
                "batlab/validation/pinn_lco.py). Fitted on the training cells' "
                "pooled degradation curve; each cell anchored at its own first "
                "observed SOH. Report the number even when it is worse than "
                "the GBRT — that is the useful result."
            ),
        )
        out.append({
            "dataset": key, "run_id": run_id,
            "soh_r2": metrics.get("soh_r2"), "soh_mae": metrics.get("soh_mae"),
            "rul_r2": metrics.get("rul_r2"), "rul_mae": metrics.get("rul_mae"),
            "baseline_soh_r2": baseline,
        })
        logged.add((key, FEATURE_VERSION))
    return out


def run_prospective_benchmark_study(
    datasets: dict,
    featured: "dict | None" = None,
    org_id: int = PLATFORM_ORG_ID,
    refresh: bool = False,
    train_fraction: float = 0.5,
) -> list[dict]:
    """
    Run and log the PROSPECTIVE evaluation — train on the first half of each
    cell's cycles, predict the remainder — for every supplied reference
    dataset, under the identical RUL-honesty rules the LCO harness uses.

    This is the only evaluation here that separates forecasting from
    curve-fitting: leave-cell-out holds out whole CELLS but still lets the
    model see the held-out cell's future; the prospective split withholds
    the future itself. The LCO-vs-prospective gap is the amount of
    interpolation that was riding along in the LCO number. Runs are logged
    under dataset=f"{key}_prospective" so they can never be averaged into
    the LCO accuracy tables (accuracy_by_source / model_kind_comparison
    exclude the suffix).

    Idempotent per (dataset, FEATURE_VERSION, train_fraction): a warm start
    is a cheap registry read, and it re-runs when the feature set or the
    split fraction changes — exactly when the number could legitimately move.
    """
    from batlab.features.engineering import FEATURE_VERSION
    from batlab.models.gbrt import GBRT_PARAMS
    from batlab.validation.prospective import (
        run_prospective,
        trivial_soh_baseline_prospective,
        rul_formula_baseline_prospective,
    )
    from chemistry_profiles import ChemistryProfile

    all_runs = leaderboard(tenant_org_id=None)
    logged = {
        (r.get("dataset") or "", r.get("feature_version"), r.get("seed"))
        for r in all_runs
        if (r.get("dataset") or "").endswith("_prospective")
    }

    out: list[dict] = []
    for key, cell_cycles in (datasets or {}).items():
        if not cell_cycles:
            continue
        ds_key = f"{key}_prospective"
        if not refresh and (ds_key, FEATURE_VERSION, int(train_fraction * 100)) in logged:
            continue
        try:
            metrics = run_prospective(
                cell_cycles,
                featured=(featured or {}).get(key),
                train_fraction=train_fraction,
            )
            soh_base = trivial_soh_baseline_prospective(
                cell_cycles, featured=(featured or {}).get(key),
                train_fraction=train_fraction,
            )
            rul_base = rul_formula_baseline_prospective(
                cell_cycles, featured=(featured or {}).get(key),
                train_fraction=train_fraction,
            )
        except Exception:
            # A dataset the prospective split genuinely cannot run on (e.g.
            # too few cycles per cell) is skipped, not logged with a
            # fabricated number.
            continue

        if metrics.get("n_cells_evaluated", 0) < 2:
            continue

        sample_cell = next(iter(cell_cycles))
        run_id = log_run(
            org_id=org_id,
            dataset=ds_key,
            chemistry=ChemistryProfile.for_cell(sample_cell).short_name,
            feature_set=list(metrics.get("features") or []),
            feature_version=FEATURE_VERSION,
            hyperparams={**GBRT_PARAMS, "train_fraction": train_fraction},
            seed=int(train_fraction * 100),  # split fraction is part of the identity
            cell_ids=list(cell_cycles.keys()),
            n_rows=(metrics.get("n_train_rows") or 0) + (metrics.get("n_test_rows") or 0),
            lco_metrics={
                **metrics,
                "baseline_soh_r2": soh_base.get("baseline_soh_r2"),
                "baseline_per_cell": soh_base.get("per_cell") or None,
                "rul_formula_baseline_r2": rul_base.get("rul_formula_baseline_r2"),
                "rul_baseline_pool": "observed",
            },
            notes=(
                "PROSPECTIVE evaluation — the only test here that separates "
                "forecasting from curve-fitting. Train window = first "
                f"{train_fraction:.0%} of each cell's cycles; the model is "
                "scored only on the remainder and never sees a cycle from "
                "the window it is evaluated on. NOT leave-cell-out: the same "
                "cell supplies its early cycles to training and its late "
                "cycles to testing (the deployment question — what happens "
                "NEXT for a cell we have history for — is the opposite axis "
                "from LCO's new-cell question). The LCO-vs-prospective gap "
                "measures interpolation the LCO number was quietly earning. "
                + (metrics.get("train_fraction_note") or "")
            ),
        )
        out.append({
            "dataset": key, "run_id": run_id,
            "soh_r2": metrics.get("soh_r2"), "soh_mae": metrics.get("soh_mae"),
            "rul_r2": metrics.get("rul_r2"), "rul_mae": metrics.get("rul_mae"),
            "baseline_soh_r2": soh_base.get("baseline_soh_r2"),
            "rul_formula_baseline_r2": rul_base.get("rul_formula_baseline_r2"),
        })
    return out


def prospective_benchmark(tenant_org_id: "int | None" = None) -> list[dict]:
    """
    Every logged prospective-split result, newest per dataset — the
    Benchmark page's forecasting-vs-curve-fitting section.

    Each entry carries the LCO headline next to it when available, because
    the GAP is the finding: soh_r2 vs the same dataset's LCO soh_r2 shows
    how much of the leave-cell-out number survives contact with the future.
    """
    runs = leaderboard(tenant_org_id=tenant_org_id)
    latest: dict = {}
    for r in runs:
        ds = r.get("dataset") or ""
        if not ds.endswith("_prospective"):
            continue
        prev = latest.get(ds)
        if prev is None or (r.get("timestamp") or "") > (prev.get("timestamp") or ""):
            latest[ds] = r

    lco_by_ds: dict = {}
    for r in runs:
        if (r.get("dataset") or "").endswith("_prospective"):
            continue
        if (r.get("model_kind") or "gbrt") != "gbrt" or r.get("soh_r2") is None:
            continue
        ds = r.get("dataset")
        prev = lco_by_ds.get(ds)
        if prev is None or (r.get("timestamp") or "") > (prev.get("timestamp") or ""):
            lco_by_ds[ds] = r

    rows = []
    for ds, r in latest.items():
        base_ds = ds[: -len("_prospective")]
        lco = lco_by_ds.get(base_ds)
        rows.append({
            "dataset":       base_ds,
            "chemistry":     r.get("chemistry") or "—",
            "train_fraction": (
                (r.get("hyperparams") or {}).get("train_fraction")
                if isinstance(r.get("hyperparams"), dict) else None
            ),
            "n_cells":       r.get("n_cells"),
            "soh_r2":        r.get("soh_r2"),
            "lco_soh_r2":    lco.get("soh_r2") if lco else None,
            "forecasting_gap": (
                (float(r["soh_r2"]) - float(lco["soh_r2"]))
                if (r.get("soh_r2") is not None and lco and lco.get("soh_r2") is not None)
                else None
            ),
            "baseline_soh_r2": r.get("baseline_soh_r2"),
            "rul_r2":        r.get("rul_r2"),
            "rul_formula_baseline_r2": r.get("rul_formula_baseline_r2"),
            "rul_label_coverage": r.get("rul_label_coverage"),
            "rul_reliable":  r.get("rul_reliable"),
            "ci_intervals":  r.get("ci_intervals"),
            "feature_version": r.get("feature_version"),
            "timestamp":     r.get("timestamp"),
            "notes":         r.get("notes"),
        })
    return rows


def cross_chemistry_benchmark(tenant_org_id: "int | None" = None) -> list[dict]:
    """
    Every logged cross-chemistry transfer result — the honest answer to
    "will this model work on a cell it has never seen?" — newest record per
    (train, eval) pairing, for the Benchmark page's permanent
    cross-chemistry section.

    Includes BOTH evaluated pairings and the honest "not evaluated" rows
    (e.g. nasa_to_oxford), since the latter is itself a real disclosure and
    belongs next to the numbers rather than being omitted.

    Rows are sorted so the most negative SOH R² comes FIRST: the point of
    this table is the counterexample, and it must be impossible to miss.
    Not-evaluated rows follow the evaluated ones.

    Each entry:
        {
          "train_dataset", "eval_dataset", "chemistry", "evaluated",
          "soh_r2", "soh_mae", "rul_mae", "rul_r2", "n_features",
          "feature_version", "timestamp", "notes",
        }
    """
    latest: dict = {}
    for r in leaderboard(tenant_org_id=tenant_org_id):
        ds = r.get("dataset") or ""
        if "_to_" not in ds:
            continue
        train_ds, eval_ds = ds.split("_to_", 1)
        key = (train_ds, eval_ds)
        prev = latest.get(key)
        if prev is None or (r.get("timestamp") or "") > (prev.get("timestamp") or ""):
            latest[key] = r

    rows = []
    for (train_ds, eval_ds), r in latest.items():
        rows.append({
            "train_dataset":   train_ds,
            "eval_dataset":    eval_ds,
            "chemistry":       r.get("chemistry") or "—",
            "evaluated":       r.get("rul_r2") is not None,
            "soh_r2":          r.get("soh_r2"),
            "soh_mae":         r.get("soh_mae"),
            "rul_mae":         r.get("rul_mae"),
            "rul_r2":          r.get("rul_r2"),
            "n_features":      len(r.get("feature_set") or []),
            "feature_version": r.get("feature_version"),
            "timestamp":       r.get("timestamp"),
            "notes":           r.get("notes"),
        })
    rows.sort(key=lambda d: (
        not d["evaluated"],
        d["soh_r2"] if d["soh_r2"] is not None else 0.0,
    ))
    return rows


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


def log_dataset_unavailable_for_modelling(
    dataset: str,
    reason: str,
    org_id: int = PLATFORM_ORG_ID,
) -> str:
    """
    Log an honest "this dataset cannot be modelled by the default pipeline"
    record — e.g. Oxford's checkpoint-indexed schema has no
    cycle_number/resistance_ohm/temperature_c, so build_features() cannot
    even run on it and no LCO number exists. Every metric is None (not a
    fabricated number, not silently skipped) — the Benchmark leaderboard
    shows this row with "—" for every metric and `reason` in its notes,
    same transparency the rest of this registry gives every other run.
    """
    return log_run(
        org_id=org_id,
        dataset=dataset,
        chemistry="not evaluable",
        feature_set=[],
        feature_version="n/a",
        hyperparams={},
        seed=0,
        cell_ids=[],
        n_rows=0,
        lco_metrics={
            "soh_mae": None, "soh_r2": None, "baseline_soh_r2": None,
            "rul_mae": None, "rul_r2": None,
            "rul_reliable": False, "per_cell": {},
            "baseline_per_cell": None,
        },
        notes=f"Not evaluable by the default pipeline: {reason}",
    )


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
            "soh_mae": None, "soh_r2": None, "baseline_soh_r2": None, "rul_mae": None, "rul_r2": None,
            "rul_reliable": False, "per_cell": {},
            "baseline_per_cell": None,
        },
        notes=f"Not evaluated: {reason}",
    )
