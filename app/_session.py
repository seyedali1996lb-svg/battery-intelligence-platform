"""
Session state initialization, persistence hydration, and data-mode resolution.

Extracted from main.py to separate concerns: this module owns everything
that hydrates Streamlit session state from the database on first render,
and resolves which data mode (nasa/severson/synthetic/uploaded) is active.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import _paths  # noqa: F401

import cell_store
from chemistry_profiles import ChemistryProfile
from utils import load_tenant_bundle_cached


# ---------------------------------------------------------------------------
# Persistence hydration — pull DB-backed settings into session state
# ---------------------------------------------------------------------------

_SETTINGS_KEYS = [
    "pinned_cell", "app_profile", "cost_of_delay_mult",
    "webhook_url", "webhook_secret", "webhook_events", "eol_threshold_pct",
]


def hydrate_persistence(org_id: int) -> None:
    """Load DB-backed decisions and settings into session state (idempotent)."""
    import db as _db
    _db.init_db()

    if "decision_log" not in st.session_state:
        st.session_state["decision_log"] = _db.load_decisions(org_id)

    settings = _db.get_settings(org_id, keys=_SETTINGS_KEYS)

    # Simple string/float settings — hydrate once, never overwrite.
    for key in ("pinned_cell",):
        if key not in st.session_state:
            st.session_state[key] = settings.get(key)

    for key in ("app_profile", "cost_of_delay_mult"):
        if key not in st.session_state:
            val = settings.get(key)
            if val is not None:
                st.session_state[key] = val

    # Webhook settings — hydrate once.
    for wh_key in ("webhook_url", "webhook_secret", "webhook_events"):
        if wh_key not in st.session_state:
            val = settings.get(wh_key)
            if val is not None:
                st.session_state[wh_key] = val

    # EOL threshold — defaults to 80% when not persisted.
    if "eol_threshold_pct" not in st.session_state:
        val = settings.get("eol_threshold_pct")
        st.session_state["eol_threshold_pct"] = val if val is not None else 80.0


# ---------------------------------------------------------------------------
# Cell partitioning — split the full fleet by data-source kind
# ---------------------------------------------------------------------------

def partition_cells(
    featured_dfs_all: dict,
    split_cycles_all: dict,
) -> dict[str, Any]:
    """Return a dict with keys nasa_fdfs, sev_fdfs, synth_fdfs, nasa_sc, sev_sc, synth_sc."""
    _kind = lambda cid: ChemistryProfile.for_cell(cid).source_kind

    nasa_fdfs = cell_store.LazyCellFrameMap(
        [k for k in featured_dfs_all if _kind(k) == "nasa"]
    )
    sev_fdfs = cell_store.LazyCellFrameMap(
        [k for k in featured_dfs_all if _kind(k) == "severson"]
    )
    synth_fdfs = cell_store.LazyCellFrameMap(
        [k for k in featured_dfs_all if _kind(k) == "synth"]
    )

    nasa_sc = {k: v for k, v in split_cycles_all.items() if _kind(k) == "nasa"}
    sev_sc = {k: v for k, v in split_cycles_all.items() if _kind(k) == "severson"}
    synth_sc = {k: v for k, v in split_cycles_all.items() if _kind(k) == "synth"}

    return {
        "nasa_fdfs": nasa_fdfs,
        "sev_fdfs": sev_fdfs,
        "synth_fdfs": synth_fdfs,
        "nasa_sc": nasa_sc,
        "sev_sc": sev_sc,
        "synth_sc": synth_sc,
    }


# ---------------------------------------------------------------------------
# Data-mode resolution — which dataset is currently active?
# ---------------------------------------------------------------------------

def resolve_data_mode(
    *,
    nasa_fdfs,
    sev_fdfs,
    synth_fdfs,
    nasa_sc: dict,
    sev_sc: dict,
    synth_sc: dict,
    bundles: dict,
    up_fdfs: dict,
    up_bundle,
    up_sc: dict,
) -> tuple[str, Any, Any, Any]:
    """Resolve and persist the active data mode; return (mode, active_fdfs, active_sc, active_bundle).

    If the persisted mode is invalid or its backing data is missing, falls
    back to the first available built-in source.
    """
    mode = st.session_state.get("data_mode", "")

    # Default on first run.
    if not mode:
        mode = "severson" if sev_fdfs else ("nasa" if nasa_fdfs else "synthetic")
        st.session_state["data_mode"] = mode
    if "uploaded_mode_meta" not in st.session_state:
        st.session_state["uploaded_mode_meta"] = None

    # Fallback if uploaded mode has no data.
    if mode == "uploaded" and (not up_fdfs or up_bundle is None):
        st.session_state["data_mode"] = "nasa"
        mode = "nasa"

    # Fallback if mode is completely unknown.
    if mode not in ("nasa", "synthetic", "severson", "uploaded"):
        mode = "severson" if sev_fdfs else ("nasa" if nasa_fdfs else "synthetic")
        st.session_state["data_mode"] = mode

    if mode == "severson":
        return mode, sev_fdfs, sev_sc, bundles.get("severson") or bundles.get("nasa")
    elif mode == "nasa":
        return mode, nasa_fdfs, nasa_sc, bundles["nasa"]
    elif mode == "synthetic":
        return mode, synth_fdfs, synth_sc, bundles["synth"]
    else:  # uploaded
        return mode, up_fdfs, up_sc, up_bundle
