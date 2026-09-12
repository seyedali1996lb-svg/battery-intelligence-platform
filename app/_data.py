"""
Data loading, feature engineering, model training, and platform graph.

Extracted from app/main.py to keep the orchestrator thin.  All functions
here are Streamlit-cacheable resource loaders or pure-logic helpers that
don't render any UI.
"""

from __future__ import annotations

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

import os
from typing import Any

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

def train_and_predict(
    battery_dict: dict,
    raw_fdfs: dict[str, pd.DataFrame],
    model_inputs: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
    dataset: str | None = None,
    org_id: int | None = None,
) -> tuple[dict, dict, dict]:
    """Train SOH+RUL models on pre-computed features and apply predictions.

    Separated from feature engineering so load_everything() can use a
    features cache hit to skip build_features() while still running
    LCO + model training.
    """
    X_all     = pd.concat([m[0] for m in model_inputs.values()])
    y_soh_all = pd.concat([m[1] for m in model_inputs.values()])
    y_rul_all = pd.concat([m[2] for m in model_inputs.values()])

    bndl = train_models(X_all, y_soh_all, y_rul_all)
    bndl["metrics"]["n_cells"] = len(battery_dict)
    bndl["metrics"]["n_rows"]  = len(X_all)

    cell_cycles = {cid: cell["cycles"] for cid, cell in battery_dict.items()}
    lco = run_lco(cell_cycles)
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
        bndl["metrics"]["baseline_soh_r2"] = _base["baseline_soh_r2"]
        bndl["metrics"]["baseline_lco_per_cell"] = _base["per_cell"]
        # Merge into the dict handed to log_run() so the baseline reaches the
        # registry/model card too, not just the in-memory bundle.
        lco = {**lco, "baseline_soh_r2": _base["baseline_soh_r2"],
               "baseline_per_cell": _base["per_cell"]}
    except Exception:
        bndl["metrics"]["baseline_soh_r2"] = None
        bndl["metrics"]["baseline_lco_per_cell"] = None

    # RUL formula baseline (Tier-0 #2): the closed form that GENERATED the
    # extrapolated RUL labels, evaluated under the same LCO folds. A model
    # that cannot beat this on the observed pool learned nothing — it is
    # recovering (or losing to) its own target's generating formula.
    try:
        from batlab.validation.trivial_baseline import rul_formula_baseline_lco
        _fb = rul_formula_baseline_lco(cell_cycles, featured=raw_fdfs)
        bndl["metrics"]["rul_formula_baseline_r2"] = _fb["rul_formula_baseline_r2"]
        bndl["metrics"]["rul_baseline_pool"] = _fb["rul_baseline_pool"]
        bndl["metrics"]["rul_formula_baseline_per_cell"] = _fb["per_cell"]
        lco = {**lco, "rul_formula_baseline_r2": _fb["rul_formula_baseline_r2"],
               "rul_baseline_pool": _fb["rul_baseline_pool"]}
    except Exception:
        bndl["metrics"]["rul_formula_baseline_r2"] = None
        bndl["metrics"]["rul_baseline_pool"] = None
        bndl["metrics"]["rul_formula_baseline_per_cell"] = None

    # Label-population transparency: what fraction of the RUL evaluation pool
    # actually carries measured (observed-EOL) labels. Surfaces the honest
    # denominator next to every RUL number instead of hiding it.
    bndl["metrics"]["rul_label_coverage"] = lco.get("rul_label_coverage")
    bndl["metrics"]["n_rul_observed_rows"] = lco.get("n_rul_observed_rows")
    bndl["metrics"]["n_rul_extrapolated_rows"] = lco.get("n_rul_extrapolated_rows")

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
        bndl["metrics"]["rul_interval_coverage"] = _cal["rul_interval_coverage"]
        bndl["metrics"]["rul_interval_coverage_calibrated"] = _cal["recalibrated_coverage"]
        bndl["metrics"]["rul_interval_width_mean"] = _cal["rul_interval_width_mean"]
        bndl["metrics"]["rul_interval_width_calibrated"] = _cal["recalibrated_width_mean"]
        bndl["metrics"]["rul_interval_n_calibration_rows"] = int(len(_cal.get("pooled_conformity_scores") or []))
        bndl["interval_e_star"] = _cal["global_e_star"]
        # Also merge into the dict handed to log_run() so the measured
        # coverage reaches the registry (calibration_meta JSON column) and
        # the Benchmark page's calibration section — not just the live bundle.
        # NaN is sanitized to None here: the calibration module uses NaN as
        # its in-memory not-evaluable convention, but NaN is not valid JSON
        # and would render as "nan%" downstream instead of "—".
        def _nn(v):
            try:
                return None if (v is None or v != v) else v
            except TypeError:
                return v
        lco = {**lco, "calibration_meta": {
            "rul_interval_coverage": _nn(_cal["rul_interval_coverage"]),
            "rul_interval_coverage_calibrated": _nn(_cal["recalibrated_coverage"]),
            "rul_interval_width_mean": _nn(_cal["rul_interval_width_mean"]),
            "rul_interval_width_calibrated": _nn(_cal["recalibrated_width_mean"]),
            "rul_interval_n_calibration_rows": int(len(_cal.get("pooled_conformity_scores") or [])),
            "interval_e_star": _nn(_cal["global_e_star"]),
            "nominal_coverage": 0.8,
        }}
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

    if dataset is not None and org_id is not None:
        import experiment_registry as _reg
        from batlab.features.engineering import FEATURE_VERSION as _FV
        from batlab.models.gbrt import GBRT_PARAMS as _GBRT_PARAMS
        from chemistry_profiles import ChemistryProfile as _CP

        _sample_cell = next(iter(battery_dict))
        _cal_note = ""
        if bndl["metrics"].get("rul_interval_coverage_calibrated") is not None:
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
            feature_set=list(X_all.columns),
            feature_version=_FV,
            hyperparams=dict(_GBRT_PARAMS),
            seed=_GBRT_PARAMS["random_state"],
            cell_ids=list(battery_dict.keys()),
            n_rows=len(X_all),
            lco_metrics=lco,
            notes=_cal_note.strip(),
        )

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
        featured_dfs[cell_id]  = df_out
        split_idx = int(len(X) * 0.8)
        split_cycles[cell_id]  = int(X["cycle_number"].iloc[split_idx])

    return bndl, featured_dfs, split_cycles


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

    def _persist_cell_data(featured_dfs: dict) -> None:
        """Write each cell's full per-cycle DataFrame to cell_store's
        Parquet store + a precomputed CellSummary row."""
        import db as _db_persist
        from experiment_registry import PLATFORM_ORG_ID as _PLATFORM_ORG_ID

        for cell_id, df in featured_dfs.items():
            cell_store.save_cell_df(cell_id, df)
            _db_persist.upsert_cell_summary(
                _PLATFORM_ORG_ID, cell_id, cell_store.build_summary(cell_id, df),
            )

    def _load_or_train_bg(key: str, cell_dict: dict) -> tuple[dict, dict]:
        """2-tier cache: full bundle → features-only → full pipeline."""
        import experiment_registry as _reg

        cached = load_cached(key, cell_dict)
        if cached is not None:
            return cached
        feat_cached = load_features_cached(key, cell_dict)
        if feat_cached is not None:
            raw_fdfs, model_inputs = feat_cached
        else:
            raw_fdfs, model_inputs = compute_features_only(cell_dict)
            save_features_cached(key, cell_dict, raw_fdfs, model_inputs)

        bundle, featured_dfs, split_cycles = train_and_predict(
            cell_dict, raw_fdfs, model_inputs, dataset=key, org_id=_reg.PLATFORM_ORG_ID,
        )
        _persist_cell_data(featured_dfs)
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

        # df-shaped featured map shared by the PINN and prospective studies
        # (both consume full featured frames, unlike the LCO study's (X, y)
        # tuples). Computed once here, outside every study's try/except, so
        # a failure in one study can never leave a later one with an
        # undefined name — each study independently degrades to a skip.
        _df_featured: dict = {}
        for _pkey in _study_datasets:
            try:
                _pfc = load_features_cached(_pkey, _study_datasets.get(_pkey, {}))
                if _pfc is not None:
                    _df_featured[_pkey] = _pfc[0]  # {cid: df_feat}
            except Exception:
                pass

        try:
            import experiment_registry as _reg_study

            _study_featured: dict = {}
            for _skey, _scells in _study_datasets.items():
                _fc = load_features_cached(_skey, _scells)
                if _fc is not None:
                    _study_featured[_skey] = _fc[1]  # {cid: (X, y_soh, y_rul)}

            _reg_study.run_cross_chemistry_study(
                _study_datasets,
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

            _pinn_datasets: dict = {
                "synth": {cid: c["cycles"] for cid, c in battery_synth["cells"].items()}
            }
            if battery_nasa:
                _pinn_datasets["nasa"] = {cid: c["cycles"] for cid, c in battery_nasa["cells"].items()}
            if sev_cell_dicts:
                _pinn_datasets["severson"] = {
                    cid: c["cycles"] for cid, c in sev_cell_dicts.items()
                }
            if zhu_cell_dicts:
                _pinn_datasets["zhu2022"] = {
                    cid: c["cycles"] for cid, c in zhu_cell_dicts.items()
                }
            if calce_cell_dicts:
                _pinn_datasets["calce"] = {
                    cid: c["cycles"] for cid, c in calce_cell_dicts.items()
                }

            _pinn_featured = _df_featured

            _pinn_baselines = {
                k: (b or {}).get("metrics", {}).get("baseline_soh_r2")
                for k, b in bundles.items() if b
            }

            _reg_pinn.run_pinn_benchmark_study(
                _pinn_datasets,
                featured=_pinn_featured,
                baselines=_pinn_baselines,
                org_id=_reg_pinn.PLATFORM_ORG_ID,
            )
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

            _reg_prosp.run_prospective_benchmark_study(
                _study_datasets,
                # The df-featured map (same cache the PINN study reuses) —
                # the prospective harness needs full featured frames, not
                # the (X, y) tuples the LCO study consumes.
                featured=_df_featured,
                org_id=_reg_prosp.PLATFORM_ORG_ID,
            )
        except Exception:
            pass

    return featured_dfs, bundles, split_cycles


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
