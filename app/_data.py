"""
Data loading, feature engineering, model training, and platform graph.

Extracted from app/main.py to keep the orchestrator thin.  All functions
here are Streamlit-cacheable resource loaders or pure-logic helpers that
don't render any UI.
"""

from __future__ import annotations

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

import os
import threading
import time
from typing import Any

import numpy as np
import streamlit as st
import pandas as pd

from data_loader import build_battery, CELL_STRESS_PROFILES, _stress_factor
from batlab.features.engineering import build_features, get_model_matrix
from batlab.models.gbrt import train_models, predict
from batlab.validation.lco import run_lco, RUL_RELIABLE_FLOOR
from trajectory_memory import TrajectoryMemory
from utils import NASA_CELL_IDS

# Single source of truth for why Oxford cannot be modelled at all — used both
# by the "dataset unavailable" disclosure row and by the cross-chemistry
# study's "not evaluated" pairings.
_OXFORD_NOT_EVALUABLE_REASON = (
    "Oxford's checkpoint-indexed per-RPT schema has no cycle_number, "
    "resistance_ohm, or temperature_c columns — the standard feature "
    "pipeline (build_features / get_model_matrix) has no input it can "
    "consume, so no transferred GBRT model can be evaluated on it."
)
from bundle_cache import (
    load_cached, save_cached,
    load_features_cached, save_features_cached,
    clear_cache,
)
import cell_store
from chemistry_profiles import ChemistryProfile


# ---------------------------------------------------------------------------
# NASA availability
# ---------------------------------------------------------------------------

def nasa_cells_available() -> list[str]:
    """Return which NASA cell CSVs are present in data/raw/."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    return [
        cid for cid in NASA_CELL_IDS
        if os.path.exists(os.path.join(data_dir, f"{cid}_summary.csv"))
    ]


# ---------------------------------------------------------------------------
# Feature engineering (no model training)
# ---------------------------------------------------------------------------

def compute_features_only(battery_dict: dict) -> tuple[dict, dict]:
    """Feature engineering pass — no model training.

    Returns (raw_fdfs, model_inputs) where:
      raw_fdfs:     {cell_id: df_feat}         — build_features() output
      model_inputs: {cell_id: (X, y_soh, y_rul)} — ready for train_models()
    """
    raw_fdfs: dict[str, pd.DataFrame] = {}
    model_inputs: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
    for cell_id, cell in battery_dict.items():
        df_feat = build_features(cell["cycles"], cell_id=cell_id)
        X, y_soh, y_rul = get_model_matrix(df_feat)
        raw_fdfs[cell_id] = df_feat
        model_inputs[cell_id] = (X, y_soh, y_rul)
    return raw_fdfs, model_inputs


# ---------------------------------------------------------------------------
# Model training + prediction
# ---------------------------------------------------------------------------

def _score_count(scores) -> int:
    """Length of the pooled conformity-scores array, or 0 when absent.

    Deliberately NOT `len(x or [])`: the scores are a NUMPY array, and a
    multi-element array in a boolean context raises ValueError — the exact
    bug that silently discarded every calibrated-coverage measurement on
    2026-09-13 (run_lco_quantiles succeeded, the very next line threw, and
    the bare except recorded 'calibration_attempted: true' instead of the
    measured numbers for all four reference fleets).
    """
    if scores is None:
        return 0
    try:
        return int(len(scores))
    except TypeError:
        return 0


def _baseline_absence_reason(result: dict) -> "str | None":
    """WHY a trivial baseline came back without a number, or None when it has one.

    A baseline can now be absent without anything raising: every fold unscorable
    (a frame that carries no `soh_pct` column, a single-cell fleet) returns NaN,
    and NaN→None at the caller's seam is indistinguishable from "not computed
    yet" unless the reason travels with it. `baseline_lco_r2()` puts a reason on
    each unscored fold; this lifts them to one string for the bundle and the
    registry row, so an absent number is never a mystery again — the same rule
    the except branch below applies to a raised failure.
    """
    value = result.get("baseline_soh_r2")
    if value is not None and value == value:
        return None  # a real number: nothing to explain
    notes = [
        f"{cid}: {entry.get('note')}"
        for cid, entry in (result.get("per_cell") or {}).items()
        if isinstance(entry, dict) and entry.get("note")
    ]
    if notes:
        shown = "; ".join(notes[:3])
        return shown + (f" (+{len(notes) - 3} more folds)" if len(notes) > 3 else "")
    if int(result.get("n_cells") or 0) < 2:
        return "fewer than two cells: leave-cell-out cannot form a fold"
    return "no fold produced an R²"


def _finite_or_none(value):
    """`value` when it is a finite number, else None.

    NaN passes every `is not None` check a consumer writes, so a metric that
    can come back NaN has to be converted where it becomes a served number —
    the same rule _layer_calibration applies in-line to the interval coverage
    numbers, which is what took the 2026-09-13 cold boot down (`:.2f` on a
    None that a NaN had sailed past). A NaN baseline stored raw would render
    as "+nan over the trivial baseline" instead of an absent number.
    """
    try:
        return None if (value is None or value != value) else value
    except (TypeError, ValueError):
        return None


# Whether _load_or_train_bg() runs cached_bundle_run_missing() on a cache
# hit. On for the app. tests/conftest.py turns it OFF for the whole suite:
# every AppTest uses an isolated_db fixture whose registry is empty by
# construction, so under the real check no test could ever take a warm
# cache hit — each would retrain every fleet from scratch (minutes per
# test) and then write a bundle stamped with a run id that only exists in
# that test's temp DB, poisoning the cache for the next test too. The
# check itself is covered directly in tests/test_cache_registry_consistency.py.
VERIFY_CACHED_BUNDLES = True


def cached_bundle_run_missing(cached) -> bool:
    """True when a cache-hit bundle's experiment_run_id has no row in THIS
    deployment's registry — the signature of a bundle trained while its
    log_run() writes went to a different/ephemeral DB.

    Such a hit must NOT be served: cache hits never re-log, so the
    plain-GBRT row would stay missing and the Benchmark headline would
    silently fall back to a stale older run (the 2026-09-13 incident:
    zhu2022 had no gbrt row at all; nasa/severson/synth served pre-v12
    rows). Fail-open on registry errors — refusing to start the app over
    a registry blip is worse than serving a bundle whose divergence the
    fingerprint column makes visible anyway.
    """
    if not isinstance(cached, tuple) or not cached:
        return False  # shape mismatch: let the normal path handle it
    bundle = cached[0] if isinstance(cached[0], dict) else None
    rid = (bundle or {}).get("metrics", {}).get("experiment_run_id")
    if not rid:
        return False  # trained outside the registry (e.g. import path)
    try:
        import experiment_registry as _reg
        return not _reg.run_exists_in_db(_reg.PLATFORM_ORG_ID, rid)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Boot layers: validation / forecast / calibration
# ---------------------------------------------------------------------------
# Measured on the dev machine (10-core / 12-thread laptop, warm features
# cache, via scripts/profile_boot.py + scripts/profile_lco.py — the profilers
# that motivated this split):
#
#   fleet (cells/rows)         train_models   run_lco   quantile LCO   hier+surv+routing
#   nasa    (4 /     580)           2.7 s      1.6 s          < 1 s              0.17 s
#   zhu2022 (9 /   8 726)           6.3 s     23.6 s        30.2 s              0.17 s
#   severson(46 / 38 765)          43.5 s      710 s          2.0 s               0.1 s
#
# On a COLD bundle cache a reference fleet's boot is therefore 80-95%
# leave-cell-out refitting (run_lco + run_lco_quantiles), not model training
# and not the hierarchical / survival / routing work — which is 0.1-0.2 s per
# fleet at every fleet size shipped here.
#
# Severson's two numbers are the interesting ones, because they are the two
# halves of the same finding — leave-cell-out work that obtains nothing:
#
#   * quantile LCO 2.0 s (was ~13 min): Severson's 46 cells carry 38 765 RUL
#     rows and NOT ONE observed-EOL label, and every interval metric is
#     computed on observed rows only. So the old path trained 46 folds x 3
#     GBRT fits — measured 132 s per fold — to return nan/None and nothing
#     else. batlab.validation.calibration now reads label provenance first
#     (_observed_label_total) and returns the identical not-evaluable result
#     with no fit at all. Numbers unchanged by construction, asserted by
#     test_not_evaluable_short_circuit_numbers_match_the_fitted_path.
#   * run_lco 710 s: 92 GBRT fits (46 SOH + 46 RUL) on 35 822 rows, ~53 s
#     each, ~7x effective parallelism on this laptop. That is the honest
#     floor for this fleet's LCO numbers and it is NOT removable without
#     changing them: the 46 SOH fits are the real headline (soh_r2 0.992),
#     and the 46 RUL fits are the only source of the extrapolated-label
#     formula-recovery diagnostic (observed-label RUL is not evaluable here,
#     so there is no headline to lose — but the diagnostic is reported
#     per-cell on Model Validation).
#
# So the post-fitting work is split into three named layers which
# load_everything() can either run inline or hand to a background thread that
# finishes the bundle afterwards (re-serve the frames, complete the registry
# row, rewrite the bundle cache):
#
#   validation  — leave-cell-out GBRT, the honest denominators (observed-EOL
#                 RUL pool, formula baselines, label coverage), the validity
#                 envelope and the per-regime reliability table.
#   forecast    — hierarchical partial pooling + its LCO, censored survival,
#                 per-cell forecast routing.
#   calibration — the quantile LCO that MEASURES the served Q10/Q90 interval's
#                 real coverage and derives the conformal widening E*.
#
# Two boot-time consequences worth knowing before reading a registry row:
# the experiment run for a fleet is logged WHEN ITS LAYERS LAND, not at boot
# (a row logged at boot would carry lco_metrics=None until someone patched it,
# and nothing re-logs a cache hit), so a cached core-only bundle keeps its
# experiment_run_id unset until the runner completes it; and the oxford
# "not evaluable for modelling" disclosure — gated on the registry already
# holding a real run — therefore arrives with the first completed fleet
# instead of on the very first cold boot.
#
# What a deferred bundle is missing is disclosed, never guessed:
# metrics["boot_layers"] carries the state and _mark_layers_pending() installs
# every layer metric as an explicit None — never a missing key, so a consumer's
# `is not None` check reads "not computed yet" instead of raising or silently
# reading as "not evaluated".
BOOT_LAYER_NAMES = ("validation", "forecast", "calibration")

BOOT_LAYERS: dict[str, Any] = {
    "state": "idle",     # idle | running | done | off
    "started_at": None,
    "finished_at": None,
    "pending": [],       # fleet keys whose layers are still computing
    "completed": [],     # fleet keys whose layer runner returned
    "failed": [],        # "<fleet>:<layer>" for any layer that raised
    "skipped": {},       # fleet key -> why its layers were NOT (re)launched
    "seconds": {},       # fleet key -> {layer: measured seconds}
}

# Every metric _mark_layers_pending() must pre-create, mapped to the value that
# means "not computed yet" for THAT metric. Scalars are None so an
# `is not None` check reads as unavailable; anything consumers index or call
# .get() on is its empty container instead — a None there does not read as
# "pending", it raises (`per_cell_ok.get(cell_id, ...)` on a None
# per_cell_rul_reliable took the cold-boot graph build down on 2026-09-19).
# A test pins this map against the layer bodies, so a new metric cannot be
# added without a placeholder for the deferred case.
_LAYER_METRIC_PLACEHOLDERS: dict[str, Any] = {
    # validation — scalars
    "lco_soh_r2": None, "lco_rul_r2": None, "rul_reliable": None,
    "baseline_soh_r2": None, "rul_formula_baseline_r2": None,
    # WHY a baseline is absent, when it is: a bare `except` here once recorded
    # None for every fleet whose SOH target had a single blank row, so the
    # absence was indistinguishable from "not computed yet" or from a real
    # zero. None means "computed"; a string means "failed, and here is why".
    "baseline_soh_r2_error": None, "rul_formula_baseline_r2_error": None,
    "rul_label_coverage": None, "n_rul_observed_rows": None,
    "n_rul_extrapolated_rows": None,
    # validation — containers
    "lco_per_cell": {}, "per_cell_rul_reliable": {}, "baseline_lco_per_cell": {},
    "rul_baseline_pool": {}, "rul_formula_baseline_per_cell": {},
    "regime_reliability": {},
    # forecast
    "hierarchical_validation": {}, "hierarchical_available": None,
    "censored_rul": {}, "forecast_routing": [],
    # calibration
    "rul_interval_coverage": None, "rul_interval_coverage_calibrated": None,
    "rul_interval_width_mean": None, "rul_interval_width_calibrated": None,
    "rul_interval_n_calibration_rows": None,
}
_LAYER_METRIC_KEYS = tuple(_LAYER_METRIC_PLACEHOLDERS)
_LAYER_BUNDLE_KEYS = (
    "training_envelope", "hierarchical_fit", "hier_forecasts", "hier_cycles",
    "interval_e_star",
    # Provenance of the validation layer itself: {mode, enabled, key, dir,
    # hits, fitted} from batlab.validation.fold_cache. None until that layer
    # lands, so a deferred bundle cannot read as "no folds were reused"
    # (which is what a confident zero would say) rather than "not computed".
    "lco_fold_cache",
)

_LAYERS_LOCK = threading.Lock()


def boot_layers_status() -> dict:
    """Snapshot of BOOT_LAYERS for the UI (copy, not the live dict)."""
    return {
        **BOOT_LAYERS,
        "pending": list(BOOT_LAYERS["pending"]),
        "completed": list(BOOT_LAYERS["completed"]),
        "failed": list(BOOT_LAYERS["failed"]),
        "skipped": dict(BOOT_LAYERS["skipped"]),
        "seconds": {k: dict(v) for k, v in BOOT_LAYERS["seconds"].items()},
    }


def _boot_layers_mode() -> str:
    """BATLAB_BOOT_LAYERS: background (default) | eager | off.

    background — load_everything() serves the core bundle (features + GBRT +
                 predictions) and finishes the three layers on a daemon
                 thread.
    eager      — inline, the pre-split behaviour; for scripts (and tests)
                 that need a complete bundle, and therefore a complete
                 registry row, before they return.
    off        — compute nothing; the bundle keeps its placeholders and the
                 app says the layers are off.
    """
    mode = (os.environ.get("BATLAB_BOOT_LAYERS") or "background").strip().lower()
    return mode if mode in ("background", "eager", "off") else "background"


def _mark_layers_pending(bndl: dict, state: str = "pending") -> None:
    """Mark a bundle whose three layers have not been computed yet."""
    metrics = bndl.setdefault("metrics", {})
    for _k, _placeholder in _LAYER_METRIC_PLACEHOLDERS.items():
        # A FRESH empty container per bundle, never the shared prototype.
        metrics[_k] = None if _placeholder is None else type(_placeholder)()
    for _k in _LAYER_BUNDLE_KEYS:
        bndl[_k] = None
    metrics["boot_layers"] = state


def _layer_validation(bndl: dict, cell_cycles: dict, raw_fdfs: dict, lco: dict) -> dict:
    """Leave-cell-out GBRT + baselines + validity envelope, in place.

    run_lco() is passed `featured=raw_fdfs`: without it the harness rebuilds
    every cell's features — including the PyBaMM-backed physics calibration —
    a second time per fold, which is pure duplicate work on the boot path
    (the same frames were built moments earlier by compute_features_only()).
    """
    lco = run_lco(cell_cycles, featured=raw_fdfs)
    # How much of this fleet's leave-cell-out validation was replayed from
    # batlab.validation.fold_cache rather than refitted — a repeated or
    # interrupted boot reports 46/46 reused instead of leaving it to be
    # inferred from a layer that finished in 0.4 s instead of 12 min. A
    # bundle key, not a metrics key: it is provenance about the computation,
    # and nothing that iterates metrics should have to step over a dict.
    bndl["lco_fold_cache"] = lco.get("fold_cache")
    bndl["metrics"]["lco_soh_r2"]        = lco["soh_r2"]
    bndl["metrics"]["lco_rul_r2"]        = lco["rul_r2"]
    bndl["metrics"]["rul_reliable"]      = lco["rul_reliable"]
    bndl["metrics"]["lco_per_cell"]      = lco["per_cell"]
    per_cell_rul_ok = {
        cid: (
            fold.get("rul_r2") is not None
            and fold["rul_r2"] >= RUL_RELIABLE_FLOOR
        )
        for cid, fold in lco["per_cell"].items()
    }
    bndl["metrics"]["per_cell_rul_reliable"] = per_cell_rul_ok

    # Honest accuracy denominator: how much of the model R² is smooth aging
    # curves rather than the engineered features / model. Same LCO folds as
    # run_lco() so it is an apples-to-apples comparison (the notebook's
    # +0.203 framing), not a different, easier baseline.
    try:
        from batlab.validation.trivial_baseline import baseline_lco_r2
        # Pass the already-built feature frames (raw_fdfs) so this does not
        # re-run the full feature pipeline — including the PyBaMM-backed
        # physics calibration — a second time per cell.
        _base = baseline_lco_r2(cell_cycles, featured=raw_fdfs)
        # NaN → None at this seam. A non-finite baseline is "not evaluable",
        # and NaN is not valid JSON — it would render as "nan" beside a real R².
        _base_r2 = _finite_or_none(_base["baseline_soh_r2"])
        bndl["metrics"]["baseline_soh_r2"] = _base_r2
        bndl["metrics"]["baseline_lco_per_cell"] = _base["per_cell"]
        # Not an exception, but still an absent number: the per-fold reasons are
        # lifted into the same field so "no baseline" is always accompanied by why.
        bndl["metrics"]["baseline_soh_r2_error"] = _baseline_absence_reason(_base)
        # Merge into the dict handed to log_run() so the baseline reaches the
        # registry/model card too, not just the in-memory bundle — with how many
        # folds were scored and how many blank target rows were set aside, so an
        # n-of-fewer-than-the-fleet baseline is inspectable rather than implied.
        lco = {**lco, "baseline_soh_r2": _base_r2,
               "baseline_per_cell": _base["per_cell"],
               "baseline_n_folds_scored": _base.get("n_folds_scored"),
               "baseline_n_nonfinite_target_rows": _base.get("n_nonfinite_target_rows")}
    except Exception as _exc:
        # DISCLOSED, never silently None. This bare `except` used to record None
        # for every fleet whose SOH target carried a blank row — Severson has two
        # (S-b1c0 cycle 11, S-b1c18 cycle 39: capacity is blank in the summary
        # CSV, so soh_pct = capacity/q0*100 is blank with it, and sklearn's
        # LinearRegression raises on a NaN in y) — which left the "+X over the
        # trivial baseline" claim with no number behind it and no trace of why.
        # The reason now rides the bundle AND the registry row.
        _err = f"{type(_exc).__name__}: {_exc}"
        bndl["metrics"]["baseline_soh_r2"] = None
        bndl["metrics"]["baseline_soh_r2_error"] = _err
        bndl["metrics"]["baseline_lco_per_cell"] = None
        lco = {**lco, "baseline_soh_r2_error": _err}

    # RUL formula baseline (Tier-0 #2): the closed form that GENERATED the
    # extrapolated RUL labels, evaluated under the same LCO folds. A model
    # that cannot beat this on the observed pool learned nothing — it is
    # recovering (or losing to) its own target's generating formula.
    try:
        from batlab.validation.trivial_baseline import rul_formula_baseline_lco
        _fb = rul_formula_baseline_lco(cell_cycles, featured=raw_fdfs)
        _fb_r2 = _finite_or_none(_fb["rul_formula_baseline_r2"])
        bndl["metrics"]["rul_formula_baseline_r2"] = _fb_r2
        bndl["metrics"]["rul_baseline_pool"] = _fb["rul_baseline_pool"]
        bndl["metrics"]["rul_formula_baseline_per_cell"] = _fb["per_cell"]
        bndl["metrics"]["rul_formula_baseline_r2_error"] = None
        lco = {**lco, "rul_formula_baseline_r2": _fb_r2,
               "rul_baseline_pool": _fb["rul_baseline_pool"],
               "rul_formula_n_nonfinite_rows_excluded":
                   _fb.get("n_nonfinite_rows_excluded")}
    except Exception as _exc:
        _err = f"{type(_exc).__name__}: {_exc}"
        bndl["metrics"]["rul_formula_baseline_r2"] = None
        bndl["metrics"]["rul_baseline_pool"] = None
        bndl["metrics"]["rul_formula_baseline_per_cell"] = None
        bndl["metrics"]["rul_formula_baseline_r2_error"] = _err
        lco = {**lco, "rul_formula_baseline_r2_error": _err}

    # Label-population transparency: what fraction of the RUL evaluation pool
    # actually carries measured (observed-EOL) labels. Surfaces the honest
    # denominator next to every RUL number instead of hiding it.
    bndl["metrics"]["rul_label_coverage"] = lco.get("rul_label_coverage")
    bndl["metrics"]["n_rul_observed_rows"] = lco.get("n_rul_observed_rows")
    bndl["metrics"]["n_rul_extrapolated_rows"] = lco.get("n_rul_extrapolated_rows")

    # ── Domain of validity (Tier 5) ─────────────────────────────────
    # The envelope the model's numbers were measured under, computed from
    # the very data that trained and validated it — plus per-regime
    # reliability (chemistry × temperature band × SOH stage), the layer
    # between per-cell gating and fleet averages. Both ride on the bundle
    # (surfaces check every cell against the envelope) and into the
    # registry (validity_meta) so the envelope is inspectable, not folklore.
    try:
        from domain_validity import compute_envelope, regime_reliability
        _envelope = compute_envelope(cell_cycles, cell_data=cell_cycles, featured=raw_fdfs)
        _regimes = regime_reliability(
            lco.get("per_cell") or {}, cell_data=cell_cycles, featured=raw_fdfs,
        )
        bndl["training_envelope"] = _envelope
        bndl["metrics"]["regime_reliability"] = _regimes
        lco = {**lco, "validity_meta": {
            "envelope": _envelope,
            "regime_reliability": _regimes,
        }}
    except Exception:
        # No envelope recorded: surfaces must render "envelope unknown"
        # (the absence is itself disclosed), never assume validity.
        bndl["training_envelope"] = None
        lco = {**lco, "validity_meta": {"envelope": None, "envelope_attempted": True}}

    return lco


def _layer_forecast(bndl: dict, cell_cycles: dict, raw_fdfs: dict, lco: dict) -> dict:
    """Hierarchical partial pooling + censored survival + per-cell routing.

    Measured at 0.1-0.2 s for every fleet shipped here (see the section
    header) — it is deferred for uniformity with the two expensive layers
    rather than for its own cost, and because it writes the served forecast
    columns that only the re-serve path should add.
    """
    # ── Hierarchical partial-pooling: the served forecasting model ─────
    # The GBRT interpolates; the prospective split shows it collapsing below
    # a straight line when denied the future. The hierarchical partial-
    # pooling model (batlab.models.hierarchical — empirical-Bayes prior over
    # log fade rates pooled per chemistry, the held-out/served cell
    # contributing only its early window) is the platform's forecasting
    # answer and is served ALONGSIDE the GBRT: per-row forecast columns
    # (soh_forecast / rul_forecast / posterior interval), a validation run
    # through the SAME leave-cell-out folds as the GBRT (logged so the two
    # models' numbers stay comparable), the per-cell regime routing verdicts
    # (src/forecast_routing.py — which model may answer "what happens
    # next" per cell), and the censored-data survival readout
    # (batlab.validation.survival — makes "still alive at last cycle"
    # informative for fleets whose RUL is not evaluable, e.g. Severson).
    hier_forecasts: dict = {}
    hier_cycles: dict = {}
    try:
        from batlab.models.hierarchical import fit_hierarchical, forecast_soh
        from batlab.validation.hierarchical_lco import run_hierarchical_lco
        from batlab.validation.survival import censored_rul_readout
        from forecast_routing import route_forecast_for_cell

        _chem_by_cell = {
            cid: ChemistryProfile.for_cell(cid).short_name for cid in cell_cycles
        }
        _hfit = fit_hierarchical(cell_cycles, chemistry_by_cell=_chem_by_cell)
        _hier_val = run_hierarchical_lco(cell_cycles, featured=raw_fdfs, chemistry_by_cell=_chem_by_cell)
        bndl["hierarchical_fit"] = _hfit
        bndl["metrics"]["hierarchical_validation"] = {
            "soh_r2": _hier_val.get("soh_r2"),
            "soh_mae": _hier_val.get("soh_mae"),
            "rul_r2": _hier_val.get("rul_r2"),
            "rul_label_coverage": _hier_val.get("rul_label_coverage"),
            "prior_scope": _hier_val.get("prior_scope"),
            "hyperparams": _hier_val.get("hyperparams"),
            "per_cell": _hier_val.get("per_cell"),
        }
        bndl["metrics"]["hierarchical_available"] = bool(_hier_val.get("per_cell"))

        for _hid, _hdf in cell_cycles.items():
            if not isinstance(_hdf, pd.DataFrame) or "cycle_number" not in _hdf.columns:
                continue
            _hs = _hdf.sort_values("cycle_number", kind="stable")
            _hx = _hs["cycle_number"].to_numpy(dtype=np.float64)
            _hy = _hs["capacity_ah"].to_numpy(dtype=float)
            _hfc = forecast_soh(_hfit, _hid, _hx, _hy)
            if _hfc is None:
                continue
            hier_forecasts[_hid] = _hfc
            hier_cycles[_hid] = _hx

        # On the bundle, not in a local: _serve_frames() re-reads these when
        # the deferred runner re-serves the frames, and this layer is the only
        # writer.
        bndl["hier_forecasts"] = hier_forecasts
        bndl["hier_cycles"] = hier_cycles

        _surv = censored_rul_readout(
            cell_cycles, hierarchical_fit=_hfit, featured=raw_fdfs,
            chemistry_by_cell=_chem_by_cell,
        )
        bndl["metrics"]["censored_rul"] = _surv
        # Fleet-level routing verdicts (which model answers per cell) ride on
        # the bundle so surfaces read the routing without recomputing it.
        bndl["metrics"]["forecast_routing"] = [
            {"cell_id": _cid, "served": _rf["served"], "reason": _rf["reason"]}
            for _cid, _rf in (
                (_cid, route_forecast_for_cell(
                    _cid,
                    {(bndl.get("serve_meta") or {}).get("dataset") or "fleet": bndl},
                    featured_dfs=None,
                ))
                for _cid in cell_cycles
            )
        ]
        lco = {
            **lco,
            "hierarchical_meta": {
                "soh_r2": _hier_val.get("soh_r2"),
                "rul_r2": _hier_val.get("rul_r2"),
                "prior_scope": _hier_val.get("prior_scope"),
                "hyperparams": _hier_val.get("hyperparams"),
            },
            "censored_rul_meta": {
                "n_cells": (_surv.get("kaplan_meier") or {}).get("n_cells"),
                "n_events": (_surv.get("kaplan_meier") or {}).get("n_events"),
                "n_censored": (_surv.get("kaplan_meier") or {}).get("n_censored"),
                "upper_bound_eol_prob": _surv.get("upper_bound_eol_prob"),
                "censoring_note": _surv.get("censoring_note"),
            },
        }
        if isinstance(lco.get("validity_meta"), dict):
            lco["validity_meta"] = {
                **lco["validity_meta"],
                "hierarchical": {
                    "available": bndl["metrics"]["hierarchical_available"],
                    "soh_r2": _hier_val.get("soh_r2"),
                },
                # JSON-safe fleet-level routing verdicts: WHICH model answers
                # "what happens next" per cell, with the reason. Rides into
                # the registry with every logged run so a historical run can
                # be audited for which model was serving.
                "forecast_routing": bndl["metrics"].get("forecast_routing") or [],
            }
    except Exception:
        # No hierarchical fit: every consumer renders "hierarchical forecast
        # unavailable" and routing falls back to its refuse/honest path —
        # the absence is disclosed, never silently papered over by the GBRT.
        bndl["metrics"]["hierarchical_available"] = False
    return lco


def _layer_calibration(bndl: dict, cell_cycles: dict, raw_fdfs: dict, lco: dict) -> dict:
    """Measure the served Q10/Q90 interval's real coverage, and widen it.

    The nominal 80% is a CLAIM; run_lco_quantiles() evaluates the quantile
    models under the same leave-cell-out folds as the point models and derives
    global_e_star, the conformal widening whose resulting coverage is reported
    as rul_interval_coverage_calibrated. predict() applies it, which is why the
    caller re-serves the frames after this layer.
    """

    # ── Calibrated uncertainty (Tier 3) ─────────────────────────────────
    # The served Q10/Q90 interval's nominal 80% is a CLAIM; what ships here
    # is a MEASUREMENT. run_lco_quantiles() evaluates the quantile models
    # under the same leave-cell-out folds as the point models, computes the
    # pooled cross-cell conformity scores (each row scored by a model that
    # never trained on its cell — the honest calibration set for a
    # production correction), and derives global_e_star: the conformal
    # widening whose resulting coverage is reported as
    # rul_interval_coverage_calibrated. predict() widens served intervals
    # by this correction, so the "80% interval" a decision surface shows is
    # the one whose real coverage was measured on cells the model never saw.
    try:
        from batlab.validation.calibration import run_lco_quantiles
        _cal = run_lco_quantiles(cell_cycles, featured=raw_fdfs)

        # NaN is sanitized to None on BOTH copies below: the calibration
        # module uses NaN as its in-memory not-evaluable convention, but
        # NaN is not valid JSON (it would render as "nan%" downstream
        # instead of "—"), and — the 2026-09-13 cold-boot crash — NaN
        # passes every `is not None` check, so a fleet with zero
        # observed-EOL rows (Severson) reached the calibrated-note branch
        # below with coverage=NaN and E*=None and formatted None with
        # `:.2f`, taking the whole training thread down. "Not evaluable"
        # is None everywhere, or it is not "not evaluable".
        def _nn(v):
            try:
                return None if (v is None or v != v) else v
            except TypeError:
                return v
        bndl["metrics"]["rul_interval_coverage"] = _nn(_cal["rul_interval_coverage"])
        bndl["metrics"]["rul_interval_coverage_calibrated"] = _nn(_cal["recalibrated_coverage"])
        bndl["metrics"]["rul_interval_width_mean"] = _nn(_cal["rul_interval_width_mean"])
        bndl["metrics"]["rul_interval_width_calibrated"] = _nn(_cal["recalibrated_width_mean"])
        bndl["metrics"]["rul_interval_n_calibration_rows"] = _score_count(_cal.get("pooled_conformity_scores"))
        bndl["interval_e_star"] = _nn(_cal["global_e_star"])
        # Also merge into the dict handed to log_run() so the measured
        # coverage reaches the registry (calibration_meta JSON column) and
        # the Benchmark page's calibration section — not just the live bundle.
        lco["calibration_meta"] = {
            "rul_interval_coverage": _nn(_cal["rul_interval_coverage"]),
            "rul_interval_coverage_calibrated": _nn(_cal["recalibrated_coverage"]),
            "rul_interval_width_mean": _nn(_cal["rul_interval_width_mean"]),
            "rul_interval_width_calibrated": _nn(_cal["recalibrated_width_mean"]),
            "rul_interval_n_calibration_rows": _score_count(_cal.get("pooled_conformity_scores")),
            "interval_e_star": _nn(_cal["global_e_star"]),
            "nominal_coverage": 0.8,
        }
    except Exception:
        # No calibrated measurement available: leave the raw holdout number
        # in place, serve un-widened intervals, and let the UI's
        # "uncalibrated" flag state that plainly rather than pretending.
        bndl["metrics"]["rul_interval_coverage_calibrated"] = None
        bndl["interval_e_star"] = None
        lco = {**lco, "calibration_meta": {
            "rul_interval_coverage_calibrated": None,
            "nominal_coverage": 0.8,
            "calibration_attempted": True,
        }}
    return lco


def train_and_predict(
    battery_dict: dict,
    raw_fdfs: dict[str, pd.DataFrame],
    model_inputs: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
    dataset: str | None = None,
    org_id: int | None = None,
    defer_layers: bool = False,
) -> tuple[dict, dict, dict]:
    """Train SOH+RUL models on pre-computed features and apply predictions.

    Separated from feature engineering so load_everything() can use a
    features cache hit to skip build_features() while still running model
    training.

    `defer_layers=True` hands the three post-fitting layers — validation,
    forecast and calibration (see the section above) — to the background
    runner and returns this bundle with metrics["boot_layers"] == "pending"
    for the caller to complete. The served predictions in that bundle are the
    CORE model's (un-widened Q10/Q90, no hierarchical columns);
    _launch_boot_layers() re-serves them once the layers land. Deferring
    changes WHEN numbers appear, never WHICH: each layer is the same code in
    both modes.
    """
    X_all     = pd.concat([m[0] for m in model_inputs.values()])
    y_soh_all = pd.concat([m[1] for m in model_inputs.values()])
    y_rul_all = pd.concat([m[2] for m in model_inputs.values()])

    bndl = train_models(X_all, y_soh_all, y_rul_all)
    bndl["metrics"]["n_cells"] = len(battery_dict)
    bndl["metrics"]["n_rows"]  = len(X_all)
    # What a later re-serve, and the deferred runner's log_run(), need and
    # cannot derive from the bundle itself.
    bndl["serve_meta"] = {
        "cell_ids": list(battery_dict.keys()),
        "feature_columns": [str(c) for c in X_all.columns],
        # The fleet key the per-cell routing verdicts are recorded under — the
        # forecast layer runs off this bundle, not off this function's args.
        "dataset": dataset,
    }

    cell_cycles = {cid: cell["cycles"] for cid, cell in battery_dict.items()}
    if defer_layers:
        _mark_layers_pending(bndl)
        lco: dict = {}
    else:
        lco = run_boot_layers(bndl, cell_cycles, raw_fdfs)

    if not defer_layers:
        _log_bundle_run(bndl, lco, dataset, org_id)

    featured_dfs, split_cycles = _serve_frames(bndl, raw_fdfs, model_inputs)

    return bndl, featured_dfs, split_cycles


def run_boot_layers(
    bndl: dict,
    cell_cycles: dict,
    raw_fdfs: dict,
    key: str | None = None,
    guarded: bool = False,
) -> dict:
    """Run the three post-fitting layers in place; returns the `lco` dict.

    Order matters (validation → forecast → calibration): calibration's served
    widening is read by predict(), which is why the caller re-serves the frames
    afterwards. `guarded` — used by the background runner — contains a layer
    failure to that layer, leaving its placeholders in place; an eager boot
    stays unguarded, so it fails exactly as it always has.
    """
    lco: dict = {}
    failed = False
    for name, layer in (
        ("validation", _layer_validation),
        ("forecast", _layer_forecast),
        ("calibration", _layer_calibration),
    ):
        t0 = time.perf_counter()
        try:
            lco = layer(bndl, cell_cycles, raw_fdfs, lco)
        except Exception:
            if not guarded:
                raise
            failed = True
            BOOT_LAYERS["failed"].append(f"{key}:{name}")
        finally:
            if key:
                BOOT_LAYERS["seconds"].setdefault(key, {})[name] = round(
                    time.perf_counter() - t0, 2
                )
    bndl["metrics"]["boot_layers"] = "failed" if failed else "done"
    return lco


def _persist_cell_data(featured_dfs: dict) -> None:
    """Write each cell's full per-cycle DataFrame to cell_store's Parquet
    store + a precomputed CellSummary row."""
    import db as _db_persist
    from experiment_registry import PLATFORM_ORG_ID as _PLATFORM_ORG_ID

    for cell_id, df in featured_dfs.items():
        cell_store.save_cell_df(cell_id, df)
        _db_persist.upsert_cell_summary(
            _PLATFORM_ORG_ID, cell_id, cell_store.build_summary(cell_id, df),
        )


def _serve_frames(
    bndl: dict,
    raw_fdfs: dict[str, pd.DataFrame],
    model_inputs: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
) -> tuple[dict, dict]:
    """Apply the bundle's models to every cell's featured frame.

    The hierarchical forecast columns are read off the bundle
    (bndl["hier_forecasts"] / ["hier_cycles"], written by _layer_forecast)
    rather than taken as arguments, so the deferred runner can re-serve the SAME
    frames once the layers land: predict() widens the served Q10/Q90 by
    bndl["interval_e_star"], which only the calibration layer sets, so the first
    serve (core only, un-widened) and the second run the same code path.
    """
    hier_forecasts = bndl.get("hier_forecasts") or {}
    hier_cycles = bndl.get("hier_cycles") or {}
    featured_dfs: dict[str, pd.DataFrame] = {}
    split_cycles: dict[str, int] = {}
    for cell_id, (X, y_soh, y_rul) in model_inputs.items():
        df_feat = raw_fdfs[cell_id]
        preds   = predict(bndl, X)
        df_out  = df_feat.loc[X.index].copy()
        df_out["soh_pred"]       = preds["soh_pred"]
        df_out["rul_pred"]       = preds["rul_pred"]
        df_out["rul_q10"]        = preds.get("rul_q10", preds["rul_pred"])
        df_out["rul_q90"]        = preds.get("rul_q90", preds["rul_pred"])
        df_out["confidence_tag"] = preds["confidence_tag"]
        # Hierarchical forecast columns, joined by cycle_number (the served
        # rows are a subset of the featured rows; the forecast was computed
        # over the cell's full recorded window). dict(zip()) keeps the last
        # value on any duplicated cycle number rather than raising.
        _hfc = hier_forecasts.get(cell_id)
        _hcy = hier_cycles.get(cell_id)
        if _hfc is not None and _hcy is not None:
            df_out["soh_forecast"] = df_out["cycle_number"].map(dict(zip(_hcy, _hfc["soh_forecast"])))
            df_out["forecast_kind"] = df_out["cycle_number"].map(dict(zip(_hcy, _hfc["forecast_kind"])))
            df_out["rul_forecast"] = df_out["cycle_number"].map(dict(zip(_hcy, _hfc["rul_forecast"])))
            df_out["rul_q10_hier"] = df_out["cycle_number"].map(dict(zip(_hcy, _hfc["rul_q10"])))
            df_out["rul_q90_hier"] = df_out["cycle_number"].map(dict(zip(_hcy, _hfc["rul_q90"])))
        featured_dfs[cell_id]  = df_out
        split_idx = int(len(X) * 0.8)
        split_cycles[cell_id]  = int(X["cycle_number"].iloc[split_idx])
    return featured_dfs, split_cycles


def _log_bundle_run(bndl: dict, lco: dict, dataset: str | None, org_id: int | None) -> None:
    """Log one completed reference-fleet fit + its layer evidence.

    Called from the boot path (eager mode) or from the deferred layer runner
    once the layers have landed — one row per fleet per fit either way, and
    never a row carrying only the metrics that happened to be ready.
    """
    if dataset is None or org_id is None:
        return
    import experiment_registry as _reg
    from batlab.features.engineering import FEATURE_VERSION as _FV
    from batlab.models.gbrt import GBRT_PARAMS as _GBRT_PARAMS
    from chemistry_profiles import ChemistryProfile as _CP

    serve_meta = bndl.get("serve_meta") or {}
    _cell_ids = list(serve_meta.get("cell_ids") or [])
    if not _cell_ids:
        return  # nothing identifiable to log against
    _sample_cell = _cell_ids[0]
    _cal_note = ""
    if (
        bndl["metrics"].get("rul_interval_coverage_calibrated") is not None
        and bndl["metrics"].get("rul_interval_coverage") is not None
        and bndl.get("interval_e_star") is not None
    ):
        _cal_note = (
            f" Calibrated interval: served Q10/Q90 widened by a cross-cell "
            f"conformal correction E*={bndl.get('interval_e_star'):.2f} cycles, "
            f"measured coverage "
            f"{bndl['metrics']['rul_interval_coverage_calibrated'] * 100:.0f}% "
            f"(raw {bndl['metrics']['rul_interval_coverage'] * 100:.0f}%) at "
            f"nominal 80%, from "
            f"{bndl['metrics']['rul_interval_n_calibration_rows']} "
            "leave-cell-out calibration rows (observed-EOL only)."
        )
    else:
        _cal_note = " Calibrated interval: NOT available on this fleet — served Q10/Q90 are the raw quantile regressor's nominal 80%, coverage unmeasured on unseen cells."
    bndl["metrics"]["experiment_run_id"] = _reg.log_run(
        org_id=org_id,
        dataset=dataset,
        chemistry=_CP.for_cell(_sample_cell).short_name,
        feature_set=list(serve_meta.get("feature_columns") or []),
        feature_version=_FV,
        hyperparams=dict(_GBRT_PARAMS),
        seed=_GBRT_PARAMS["random_state"],
        cell_ids=_cell_ids,
        n_rows=int(bndl["metrics"].get("n_rows") or 0),
        lco_metrics=lco,
        notes=_cal_note.strip(),
    )


def _launch_boot_layers(
    key: str,
    cell_dict: dict,
    bndl: dict,
    cell_cycles: dict,
    raw_fdfs: dict,
    model_inputs: dict,
) -> None:
    """Finish a deferred bundle on a daemon thread.

    The job runs the three layers, re-serves the frames (so the calibrated
    widening and the hierarchical columns reach the served rows), persists
    them, then logs the run and rewrites the bundle cache — one completion step,
    so a process that dies mid-layer leaves NO half-validated bundle cached and
    NO run row missing its evidence; the next boot simply retries (see
    _resume_deferred_layers). Only when every layer returned does the cache
    written here replace the core-only one saved at boot.
    """
    if _boot_layers_mode() == "off":
        BOOT_LAYERS["state"] = "off"
        return
    with _LAYERS_LOCK:
        if key in BOOT_LAYERS["pending"]:
            return  # this fleet's layers are already in flight
        BOOT_LAYERS["pending"].append(key)
    BOOT_LAYERS.update(state="running", started_at=time.time(), finished_at=None)
    BOOT_LAYERS["skipped"].pop(key, None)

    def _job() -> None:
        try:
            import experiment_registry as _reg

            lco = run_boot_layers(bndl, cell_cycles, raw_fdfs, key=key, guarded=True)
            featured_dfs, split_cycles = _serve_frames(bndl, raw_fdfs, model_inputs)
            _persist_cell_data(featured_dfs)
            if (bndl.get("metrics") or {}).get("boot_layers") == "done":
                _log_bundle_run(bndl, lco, key, _reg.PLATFORM_ORG_ID)
                save_cached(key, cell_dict, (bndl, split_cycles))
                BOOT_LAYERS["completed"].append(key)
            # boot_layers == "failed": the core-only cache entry is left in
            # place deliberately (no registry row, no completed bundle) so the
            # next boot retries the layers instead of serving them as done.
        except Exception:
            BOOT_LAYERS["failed"].append(f"{key}:runner")
            bndl.setdefault("metrics", {})["boot_layers"] = "failed"
        finally:
            with _LAYERS_LOCK:
                if key in BOOT_LAYERS["pending"]:
                    BOOT_LAYERS["pending"].remove(key)
                BOOT_LAYERS["finished_at"] = time.time()
                if not BOOT_LAYERS["pending"] and BOOT_LAYERS["state"] == "running":
                    BOOT_LAYERS["state"] = "done"

    threading.Thread(target=_job, name=f"boot-layers-{key}", daemon=True).start()


def _resume_deferred_layers(key: str, cached: Any, cell_dict: dict) -> None:
    """Re-launch a cached bundle's deferred layers, when they never landed.

    A bundle cached with boot_layers "pending"/"failed" is completable rather
    than broken: the features cache the same signature produced is reloaded and
    the layer runner runs again. Without that cache the bundle is served as-is
    and the pending state is disclosed in BOOT_LAYERS["skipped"] — never
    silently upgraded, and never confused with an eager bundle (which has no
    boot_layers key at all and is therefore left alone).
    """
    bndl = cached[0] if isinstance(cached, tuple) and cached else None
    if not isinstance(bndl, dict):
        return
    state = (bndl.get("metrics") or {}).get("boot_layers")
    if state not in ("pending", "failed") or _boot_layers_mode() != "background":
        return
    feat_cached = load_features_cached(key, cell_dict)
    if feat_cached is None:
        BOOT_LAYERS["skipped"][key] = "features cache absent — layers not retried"
        return
    raw_fdfs, model_inputs = feat_cached
    _launch_boot_layers(
        key, cell_dict, bndl,
        {cid: cell["cycles"] for cid, cell in cell_dict.items()},
        raw_fdfs, model_inputs,
    )


def train_on_cells(battery_dict: dict) -> tuple[dict, dict, dict]:
    """Full pipeline: feature engineering + model training + predictions.

    Called by page_import() for user-uploaded data.  For built-in data,
    load_everything() uses compute_features_only + train_and_predict
    so each stage can be cached independently.
    """
    raw_fdfs, model_inputs = compute_features_only(battery_dict)
    return train_and_predict(battery_dict, raw_fdfs, model_inputs)


# ---------------------------------------------------------------------------
# Main data loader (cached once per process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_everything() -> tuple[Any, dict, dict]:
    """
    Load synthetic, NASA, and Severson cells in parallel (independent pipelines).

    Two separate models are trained — one per data source — because the
    synthetic and NASA resistance measurements are on incompatible scales.
    """
    import concurrent.futures as _cf

    def _backfill_validity(cached: tuple, cell_dict: dict) -> None:
        """Stamp the Tier-5 training envelope onto a cache-hit bundle.

        Bundles cached before validity tracking lack `training_envelope`;
        computing it is pure data inspection (no model fitting), so it is
        done lazily on every load rather than forcing a full retrain. The
        per-regime reliability table is NOT backfilled here (it needs the
        LCO per-cell folds from a fresh run) — the Benchmark section
        renders "envelope without regime table" honestly in that case.
        """
        bundle = cached[0] if isinstance(cached, tuple) else None
        if not isinstance(bundle, dict) or bundle.get("training_envelope") is not None:
            return
        try:
            from domain_validity import compute_envelope
            bundle["training_envelope"] = compute_envelope(
                cell_dict, cell_data=cell_dict,
            )
        except Exception:
            pass

    def _load_or_train_bg(key: str, cell_dict: dict) -> tuple[dict, dict]:
        """2-tier cache: full bundle → features-only → full pipeline.

        Registry-verified cache hits (Tier-6 honesty): a cached bundle
        carries the experiment_run_id of the log_run() call that trained
        it, and the hit is only served if that row EXISTS in this
        deployment's database. A bundle trained while its registry writes
        went to a different/ephemeral DB would otherwise be served on
        every subsequent load — cache hits never re-log — so the
        plain-GBRT row would stay missing and the Benchmark headline
        would silently fall back to a stale older run (the 2026-09-13
        incident: zhu2022 had no gbrt row at all; nasa/severson/synth
        served pre-v12 rows). Detected here, the stale cache is discarded
        and the fleet retrains through the normal path, which re-logs.
        """
        import experiment_registry as _reg

        cached = load_cached(key, cell_dict)
        if cached is not None:
            if VERIFY_CACHED_BUNDLES and cached_bundle_run_missing(cached):
                clear_cache(key)
            else:
                _backfill_validity(cached, cell_dict)
                _resume_deferred_layers(key, cached, cell_dict)
                return cached
        feat_cached = load_features_cached(key, cell_dict)
        if feat_cached is not None:
            raw_fdfs, model_inputs = feat_cached
        else:
            raw_fdfs, model_inputs = compute_features_only(cell_dict)
            save_features_cached(key, cell_dict, raw_fdfs, model_inputs)

        # Boot split (see the "Boot layers" section): the core fit is what the
        # app needs to render a number, so it is the only part load_everything()
        # waits for. The validation / forecast / calibration layers are 80-95%
        # of a cold fleet's cost and go to a background runner that finishes
        # the bundle (re-serve, persist, registry row, cache rewrite).
        # BATLAB_BOOT_LAYERS=eager restores the inline behaviour.
        _mode = _boot_layers_mode()
        bundle, featured_dfs, split_cycles = train_and_predict(
            cell_dict, raw_fdfs, model_inputs, dataset=key, org_id=_reg.PLATFORM_ORG_ID,
            defer_layers=_mode in ("background", "off"),
        )
        _persist_cell_data(featured_dfs)
        if (bundle.get("metrics") or {}).get("boot_layers") in ("pending", "failed"):
            if _mode == "off":
                # "off" means the layers are neither running nor coming; label
                # it so no surface reports a retry that will never happen.
                _mark_layers_pending(bundle, "off")
            _launch_boot_layers(
                key, cell_dict, bundle,
                {cid: cell["cycles"] for cid, cell in cell_dict.items()},
                raw_fdfs, model_inputs,
            )
        result = (bundle, split_cycles)
        save_cached(key, cell_dict, result)
        return result

    with st.status("Initialising platform…", expanded=False) as _status:
        _prog = st.progress(0, text="Loading data sources in parallel…")

        # Build cell dicts (main thread — fast)
        synth_ids     = list(CELL_STRESS_PROFILES.keys())
        battery_synth = build_battery(battery_id="Oxford_B1", cell_ids=synth_ids)

        nasa_ids = nasa_cells_available()
        battery_nasa = build_battery(battery_id="NASA_B1", cell_ids=nasa_ids) if nasa_ids else None

        sev_cell_dicts: dict[str, dict] = {}
        try:
            from batlab.datasets.severson import load_severson_cells, any_cached as _sev_any_cached
            if _sev_any_cached():
                sev_cells = load_severson_cells(status_fn=lambda msg: None)
                if sev_cells:
                    sev_cell_dicts = {cid: {"cycles": df} for cid, df in sev_cells.items()}
        except Exception:
            pass

        # Zhu 2022 (NCM+NCA — third chemistry) and CALCE CS2 (second real
        # LiCoO2 source). Both degrade gracefully: Zhu serves from committed
        # per-cycle summaries (auto-download only if absent), CALCE raises
        # CalceDataNotFoundError without a manual download — an absent source
        # simply yields no bundle, and every consumer already handles that.
        zhu_cell_dicts: dict[str, dict] = {}
        try:
            from batlab.datasets.zhu2022 import load_zhu2022_cells
            zhu_cells = load_zhu2022_cells(status_fn=lambda msg: None)
            if zhu_cells:
                zhu_cell_dicts = {cid: {"cycles": df} for cid, df in zhu_cells.items()}
        except Exception:
            pass

        calce_cell_dicts: dict[str, dict] = {}
        try:
            from batlab.datasets.calce import load_calce_cells, CalceDataNotFoundError
            try:
                calce_cells = load_calce_cells()
                if calce_cells:
                    calce_cell_dicts = {cid: {"cycles": df} for cid, df in calce_cells.items()}
            except CalceDataNotFoundError:
                pass  # manual download not performed on this deployment
        except Exception:
            pass

        # Run the reference pipelines concurrently
        _prog.progress(10, text="Training models…")
        futures: dict[str, Any] = {}
        with _cf.ThreadPoolExecutor(max_workers=5) as _pool:
            futures["synth"] = _pool.submit(_load_or_train_bg, "synth", battery_synth["cells"])
            if battery_nasa:
                futures["nasa"] = _pool.submit(_load_or_train_bg, "nasa", battery_nasa["cells"])
            if sev_cell_dicts:
                futures["severson"] = _pool.submit(_load_or_train_bg, "severson", sev_cell_dicts)
            if zhu_cell_dicts:
                futures["zhu2022"] = _pool.submit(_load_or_train_bg, "zhu2022", zhu_cell_dicts)
            if calce_cell_dicts:
                futures["calce"] = _pool.submit(_load_or_train_bg, "calce", calce_cell_dicts)

        _prog.progress(90, text="Merging results…")

        bundle_synth, sc_synth = futures["synth"].result()
        bundle_nasa,  sc_nasa  = futures["nasa"].result()    if "nasa"     in futures else (None, {})
        bundle_sev,   sc_sev   = futures["severson"].result() if "severson" in futures else (None, {})
        bundle_zhu,   sc_zhu   = futures["zhu2022"].result() if "zhu2022" in futures else (None, {})
        bundle_calce, sc_calce = futures["calce"].result()   if "calce"   in futures else (None, {})

        _prog.progress(100, text="Platform ready ✓")
        if _status is not None:
            _status.update(label="Platform ready ✓", state="complete", expanded=False)

    sev_ids = list(sev_cell_dicts.keys())
    zhu_ids = list(zhu_cell_dicts.keys())
    calce_ids = list(calce_cell_dicts.keys())
    split_cycles = {**sc_synth, **sc_nasa, **sc_sev, **sc_zhu, **sc_calce}
    bundles = {
        "synth": bundle_synth, "nasa": bundle_nasa, "severson": bundle_sev,
        "zhu2022": bundle_zhu, "calce": bundle_calce,
    }
    featured_dfs = cell_store.LazyCellFrameMap(
        synth_ids + nasa_ids + sev_ids + zhu_ids + calce_ids
    )

    # ── Honest disclosures not tied to a training call ──────────────────────
    # Both blocks below log registry rows WITHOUT a bundle-cache miss / retrain
    # to trigger them — unlike the reference-dataset runs above, which only log
    # when they actually fit. They therefore share one guard: run only once the
    # registry already holds at least one real run.
    #
    # Without it, a fresh install (or a warm disk cache with a fresh registry)
    # would show "runs" before any work happened, and the Benchmark page's
    # "no experiment runs logged yet" empty state could never appear — the
    # platform would claim a result it hasn't produced. In production the
    # reference runs log earlier in this same call on a cold load, so the
    # disclosures appear right alongside them.
    _platform_real_run = False
    try:
        import experiment_registry as _reg
        _platform_real_run = any(
            (r.get("model_kind") or "gbrt") != "pinn" and r["dataset"] != "oxford"
            for r in _reg.leaderboard(tenant_org_id=None)
        )
    except Exception:
        _platform_real_run = False

    # Oxford is a real supported dataset loader, but its checkpoint-indexed
    # schema has no cycle_number / resistance_ohm / temperature_c at all, so
    # build_features() (and therefore the GBRT pipeline + LCO) cannot run on
    # it. Rather than silently omitting it from the benchmark, log an honest
    # "not evaluable" row so the platform's accuracy claims don't silently
    # imply Oxford is supported when it isn't yet. Idempotent: the row's
    # content is static by construction, so it is logged once, not once per
    # process start.
    if _platform_real_run:
        try:
            import experiment_registry as _reg
            if not _reg.leaderboard(tenant_org_id=None, dataset="oxford"):
                _reg.log_dataset_unavailable_for_modelling(
                    "oxford",
                    _OXFORD_NOT_EVALUABLE_REASON
                    + " Substituting checkpoint_index for cycle_number would "
                    "misrepresent calendar-time RPT checkpoints as charge/discharge "
                    "cycle counts (a real methodological error, not a data-density "
                    "inconvenience).",
                    org_id=_reg.PLATFORM_ORG_ID,
                )
        except Exception:
            pass

        # ── Permanent cross-chemistry transfer benchmark ────────────────────
        # The most informative real-world accuracy number this project has is
        # the NASA -> Severson zero-shot transfer failure (SOH R² = -34.6,
        # RUL R² = -1.14) — the honest answer to "will this model work on a
        # cell I have never seen?". Until now it existed only as a one-off
        # study output; running it here makes it a permanent, visible part of
        # the benchmark instead of a number someone has to go re-derive.
        #
        # run_cross_chemistry_study() is idempotent per FEATURE_VERSION, so
        # after the first run this is a cheap registry read, not a retrain —
        # and it re-runs automatically when the feature set changes, which is
        # exactly when the number could legitimately have moved. Features come
        # from the features cache so it doesn't rebuild what the reference
        # pipelines above already built (the expensive PyBaMM-backed pass).
        # Built once, outside the try, so both studies below share the same
        # reference-dataset map (and a failure in one can't leave the other
        # with an undefined name).
        _study_datasets: dict = {"synth": battery_synth["cells"]}
        if battery_nasa:
            _study_datasets["nasa"] = battery_nasa["cells"]
        if sev_cell_dicts:
            _study_datasets["severson"] = sev_cell_dicts
        if zhu_cell_dicts:
            _study_datasets["zhu2022"] = zhu_cell_dicts
        if calce_cell_dicts:
            _study_datasets["calce"] = calce_cell_dicts


        # The five benchmark studies below (cross-chemistry transfer, PINN,
        # prospective split, modeling candidates, robustness) are Benchmark-
        # page content, not something any other page needs to render. They
        # are multi-minute jobs on a cold registry — every fleet re-fit under
        # several alternative evaluations — so they run AFTER this function
        # returns, on a background thread, instead of blocking first render
        # (before 2026-09-13 they ran inline here and a cold Streamlit Cloud
        # deploy sat on the spinner for the better part of an hour). See
        # _launch_benchmark_studies() for the eager/off overrides.
        _launch_benchmark_studies(_study_datasets, bundles)

    return featured_dfs, bundles, split_cycles


# ── Background benchmark studies ────────────────────────────────────────────
# State of the post-boot study run, readable by the Benchmark page so it can
# say "still computing" instead of showing an empty section as if the study
# had been skipped. Module-level (not session_state): the thread is
# process-wide, exactly like the @st.cache_resource result it follows.
BENCHMARK_STUDIES = {
    "state": "idle",        # idle | running | done | off
    "started_at": None,     # time.time()
    "finished_at": None,
    "completed": [],        # study names that returned
    "current": None,        # study name in progress
}
_STUDY_NAMES = ("cross_chemistry", "pinn", "prospective", "modeling", "robustness")


def benchmark_studies_status() -> dict:
    """Snapshot of BENCHMARK_STUDIES for the UI (copy, not the live dict)."""
    return {**BENCHMARK_STUDIES, "completed": list(BENCHMARK_STUDIES["completed"])}


def _launch_benchmark_studies(study_datasets: dict, bundles: dict) -> None:
    """Run _run_benchmark_studies() according to BATLAB_BOOT_STUDIES:

      background (default) — daemon thread; load_everything() returns at once
                             and the Benchmark page fills in as studies land.
      eager                — inline, the pre-2026-09-13 behaviour; for scripts
                             that need a complete registry when this returns.
      off                  — skip entirely (scripts/run_tier4_studies.py and
                             friends can still log them on demand).

    Every study is idempotent per FEATURE_VERSION (skips datasets already
    logged), so a restart part-way through only fills the gaps.
    """
    import threading
    import time as _time

    mode = (os.environ.get("BATLAB_BOOT_STUDIES") or "background").strip().lower()
    if mode == "off":
        BENCHMARK_STUDIES["state"] = "off"
        return
    if BENCHMARK_STUDIES["state"] == "running":
        return  # a previous load_everything() already started one

    BENCHMARK_STUDIES.update(
        state="running", started_at=_time.time(), finished_at=None, completed=[], current=None,
    )

    def _run() -> None:
        try:
            _run_benchmark_studies(study_datasets, bundles)
        finally:
            BENCHMARK_STUDIES.update(state="done", finished_at=_time.time(), current=None)

    if mode == "eager":
        _run()
        return
    threading.Thread(target=_run, name="benchmark-studies", daemon=True).start()


def _run_benchmark_studies(study_datasets: dict, bundles: dict) -> None:
    """The five post-boot benchmark studies, in the order they were added.
    Each is wrapped in its own try/except so one failing study never blocks
    the next; BENCHMARK_STUDIES records which ones actually returned.
    Must not touch `st.*` — this runs on a plain thread with no script
    context."""

    def _mark(name: str) -> None:
        BENCHMARK_STUDIES["current"] = name

    def _done(name: str) -> None:
        BENCHMARK_STUDIES["completed"].append(name)

    # df-shaped featured map shared by the PINN and prospective studies
    # (both consume full featured frames, unlike the LCO study's (X, y)
    # tuples). Computed once here, outside every study's try/except, so
    # a failure in one study can never leave a later one with an
    # undefined name — each study independently degrades to a skip.
    _df_featured: dict = {}
    for _pkey in study_datasets:
        try:
            _pfc = load_features_cached(_pkey, study_datasets.get(_pkey, {}))
            if _pfc is not None:
                _df_featured[_pkey] = _pfc[0]  # {cid: df_feat}
        except Exception:
            pass

    try:
        import experiment_registry as _reg_study

        _study_featured: dict = {}
        for _skey, _scells in study_datasets.items():
            _fc = load_features_cached(_skey, _scells)
            if _fc is not None:
                _study_featured[_skey] = _fc[1]  # {cid: (X, y_soh, y_rul)}

        _mark("cross_chemistry")
        _reg_study.run_cross_chemistry_study(
            study_datasets,
            featured=_study_featured,
            org_id=_reg_study.PLATFORM_ORG_ID,
            # Oxford can never be the eval side of a transfer study — its
            # checkpoint-indexed schema has no cycle_number/
            # resistance_ohm/temperature_c, so build_features() produces no
            # vector to evaluate against. Disclosed as honest "not
            # evaluated" rows rather than omitted.
            unavailable=[
                ("nasa", "oxford", _OXFORD_NOT_EVALUABLE_REASON),
                ("severson", "oxford", _OXFORD_NOT_EVALUABLE_REASON),
            ],
        )
        _done("cross_chemistry")
    except Exception:
        pass

    # ── PINN through the same LCO harness as the GBRT ───────────────────
    # The GBRT had an honest published number; the physics-regularized
    # estimator did not, so "which model should we use?" was answered by
    # assertion. Running the PINN through the identical leave-cell-out
    # folds makes the comparison measurable — and reports the result even
    # when the PINN loses, which is itself the useful finding.
    #
    # The trivial-baseline denominator is model-independent (cycle_number
    # -> SOH under the same folds), so the GBRT's already-computed value is
    # reused unchanged: the two rows differ only in the model.
    try:
        import experiment_registry as _reg_pinn

        # The PINN harness takes {cid: raw cycles df}, not the {cid: {"cycles": df}}
        # wrappers the other studies unwrap themselves.
        _pinn_datasets: dict = {
            key: {cid: c["cycles"] for cid, c in cells.items()}
            for key, cells in study_datasets.items()
        }

        _pinn_featured = _df_featured

        _pinn_baselines = {
            k: (b or {}).get("metrics", {}).get("baseline_soh_r2")
            for k, b in bundles.items() if b
        }

        _mark("pinn")
        _reg_pinn.run_pinn_benchmark_study(
            _pinn_datasets,
            featured=_pinn_featured,
            baselines=_pinn_baselines,
            org_id=_reg_pinn.PLATFORM_ORG_ID,
        )
        _done("pinn")
    except Exception:
        pass

    # ── Prospective (temporal-holdout) benchmark ───────────────────────
    # The only evaluation that separates forecasting from curve-fitting:
    # train on the first half of each cell's cycles, score the remainder.
    # Leave-cell-out still lets the model see the held-out cell's future;
    # this split withholds it. Idempotent per (dataset, FEATURE_VERSION,
    # train_fraction) — a warm start is a registry read, not a retrain.
    try:
        import experiment_registry as _reg_prosp

        _mark("prospective")
        _reg_prosp.run_prospective_benchmark_study(
            study_datasets,
            # The df-featured map (same cache the PINN study reuses) —
            # the prospective harness needs full featured frames, not
            # the (X, y) tuples the LCO study consumes.
            featured=_df_featured,
            org_id=_reg_prosp.PLATFORM_ORG_ID,
        )
        _done("prospective")
    except Exception:
        pass

    # ── Tier-4 modeling candidates + robustness benchmark ─────────────
    # Hierarchical partial-pooling and the GBRT+PINN ensemble through
    # the SAME LCO harness (idempotent per dataset/model_kind/feature
    # version), and the degraded-input robustness benchmark for the
    # production model. Each independently guarded: one study's failure
    # can never skip the others.
    try:
        import experiment_registry as _reg_model

        _mark("modeling")
        _reg_model.run_modeling_benchmark_study(
            study_datasets,
            featured=_df_featured,
            baselines={
                k: (b or {}).get("metrics", {}).get("baseline_soh_r2")
                for k, b in bundles.items() if b
            },
            org_id=_reg_model.PLATFORM_ORG_ID,
        )
        _done("modeling")
    except Exception:
        pass
    try:
        import experiment_registry as _reg_rob

        _mark("robustness")
        _reg_rob.run_robustness_study(
            study_datasets,
            org_id=_reg_rob.PLATFORM_ORG_ID,
        )
        _done("robustness")
    except Exception:
        pass



# ---------------------------------------------------------------------------
# Cell summary sync (bridges cache and DB)
# ---------------------------------------------------------------------------

def ensure_cell_summaries_synced(cell_ids: list) -> None:
    """Backfill CellSummary rows for cells whose data exists on disk
    but whose DB rows are missing (e.g. after a fresh test fixture)."""
    if st.session_state.get("_cell_summaries_synced"):
        return
    import db as _db_sync
    from experiment_registry import PLATFORM_ORG_ID as _PID

    existing = {r["cell_id"] for r in _db_sync.get_cell_summaries(_PID, include_platform=False)}
    for cid in cell_ids:
        if cid in existing:
            continue
        df = cell_store.get_cell_df(cid)
        if df is None:
            continue
        _db_sync.upsert_cell_summary(_PID, cid, cell_store.build_summary(cid, df))
    st.session_state["_cell_summaries_synced"] = True


# ---------------------------------------------------------------------------
# Platform knowledge graph (built once per process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_platform_graph(_featured_dfs_all: dict, _bundles: dict) -> Any:
    """Build the Battery Digital Knowledge Graph for the platform's
    shared reference fleets."""
    import knowledge_graph as kg
    import experiment_registry as reg

    graph = kg.build_platform_graph(_featured_dfs_all, _bundles, tenant_org_id=None)
    try:
        kg.save_graph(reg.PLATFORM_ORG_ID, graph)
    except Exception:
        pass
    return graph
