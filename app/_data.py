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
    bndl["metrics"]["lco_soh_r2"]   = lco["soh_r2"]
    bndl["metrics"]["lco_rul_r2"]   = lco["rul_r2"]
    bndl["metrics"]["rul_reliable"] = lco["rul_reliable"]
    bndl["metrics"]["lco_per_cell"] = lco["per_cell"]
    per_cell_rul_ok = {
        cid: (fold["rul_r2"] >= RUL_RELIABLE_FLOOR)
        for cid, fold in lco["per_cell"].items()
    }
    bndl["metrics"]["per_cell_rul_reliable"] = per_cell_rul_ok

    if dataset is not None and org_id is not None:
        import experiment_registry as _reg
        from batlab.features.engineering import FEATURE_VERSION as _FV
        from batlab.models.gbrt import GBRT_PARAMS as _GBRT_PARAMS
        from chemistry_profiles import ChemistryProfile as _CP

        _sample_cell = next(iter(battery_dict))
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

        # Run three pipelines concurrently
        _prog.progress(10, text="Training models…")
        futures: dict[str, Any] = {}
        with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
            futures["synth"] = _pool.submit(_load_or_train_bg, "synth", battery_synth["cells"])
            if battery_nasa:
                futures["nasa"] = _pool.submit(_load_or_train_bg, "nasa", battery_nasa["cells"])
            if sev_cell_dicts:
                futures["severson"] = _pool.submit(_load_or_train_bg, "severson", sev_cell_dicts)

        _prog.progress(90, text="Merging results…")

        bundle_synth, sc_synth = futures["synth"].result()
        bundle_nasa,  sc_nasa  = futures["nasa"].result()  if "nasa"     in futures else (None, {})
        bundle_sev,   sc_sev   = futures["severson"].result() if "severson" in futures else (None, {})

        _prog.progress(100, text="Platform ready ✓")
        if _status is not None:
            _status.update(label="Platform ready ✓", state="complete", expanded=False)

    sev_ids = list(sev_cell_dicts.keys())
    split_cycles = {**sc_synth, **sc_nasa, **sc_sev}
    bundles = {"synth": bundle_synth, "nasa": bundle_nasa, "severson": bundle_sev}
    featured_dfs = cell_store.LazyCellFrameMap(synth_ids + nasa_ids + sev_ids)

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
