"""
Data loading / caching extracted from main.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from data_loader import build_battery, CELL_STRESS_PROFILES
from features import build_features, get_model_matrix
from model import train_models, predict
from lco_eval import run_lco, RUL_RELIABLE_FLOOR
from bundle_cache import load_cached, save_cached

from utils import NASA_CELL_IDS


def _nasa_cells_available() -> list[str]:
    """Return which NASA cell CSVs are present in data/raw/."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    return [
        cid for cid in NASA_CELL_IDS
        if os.path.exists(os.path.join(data_dir, f"{cid}_summary.csv"))
    ]


def _train_on_cells(battery_dict: dict) -> tuple[dict, dict, dict]:
    """Train SOH+RUL models and run LCO on a battery_dict of enriched cells."""
    all_X, all_y_soh, all_y_rul = [], [], []
    cell_featured = {}
    for cell_id, cell in battery_dict.items():
        df_feat = build_features(cell["cycles"])
        X, y_soh, y_rul = get_model_matrix(df_feat)
        all_X.append(X); all_y_soh.append(y_soh); all_y_rul.append(y_rul)
        cell_featured[cell_id] = (df_feat, X)
    X_all = pd.concat(all_X)
    y_soh_all = pd.concat(all_y_soh)
    y_rul_all = pd.concat(all_y_rul)
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

    featured_dfs, split_cycles = {}, {}
    for cell_id, (df_feat, X) in cell_featured.items():
        preds = predict(bndl, X)
        df_out = df_feat.loc[X.index].copy()
        df_out["soh_pred"]       = preds["soh_pred"]
        df_out["rul_pred"]       = preds["rul_pred"]
        df_out["rul_q10"]        = preds.get("rul_q10", preds["rul_pred"])
        df_out["rul_q90"]        = preds.get("rul_q90", preds["rul_pred"])
        df_out["confidence_tag"] = preds["confidence_tag"]
        featured_dfs[cell_id] = df_out
        split_idx = int(len(X) * 0.8)
        split_cycles[cell_id] = int(X["cycle_number"].iloc[split_idx])
    return bndl, featured_dfs, split_cycles


@st.cache_resource(show_spinner="Loading cells and training model...")
def load_everything():
    """
    Load synthetic cells (Cell1-Cell8) and NASA cells (B0005-B0018) separately.
    """
    # ── Synthetic cells (disk-cached after first run) ──
    synth_ids = list(CELL_STRESS_PROFILES.keys())
    battery_synth = build_battery(battery_id="Oxford_B1", cell_ids=synth_ids)
    cached = load_cached("synth", battery_synth["cells"])
    if cached is not None:
        bundle_synth, fdfs_synth, sc_synth = cached
    else:
        bundle_synth, fdfs_synth, sc_synth = _train_on_cells(battery_synth["cells"])
        save_cached("synth", battery_synth["cells"], (bundle_synth, fdfs_synth, sc_synth))

    # ── NASA real cells (disk-cached after first run) ──
    nasa_ids = _nasa_cells_available()
    bundle_nasa, fdfs_nasa, sc_nasa = None, {}, {}
    if nasa_ids:
        battery_nasa = build_battery(battery_id="NASA_B1", cell_ids=nasa_ids)
        cached_nasa = load_cached("nasa", battery_nasa["cells"])
        if cached_nasa is not None:
            bundle_nasa, fdfs_nasa, sc_nasa = cached_nasa
        else:
            bundle_nasa, fdfs_nasa, sc_nasa = _train_on_cells(battery_nasa["cells"])
            save_cached("nasa", battery_nasa["cells"], (bundle_nasa, fdfs_nasa, sc_nasa))

    # Merge cell outputs; keep bundles separate
    featured_dfs = {**fdfs_synth, **fdfs_nasa}
    split_cycles = {**sc_synth, **sc_nasa}
    bundles = {"synth": bundle_synth, "nasa": bundle_nasa}

    return featured_dfs, bundles, split_cycles
