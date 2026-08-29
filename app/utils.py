"""
Shared utilities — cache/data helpers and backward-compatible re-exports.

This module re-exports everything from ``_design_tokens`` and ``_ui_helpers``
so that existing ``from utils import X`` call sites continue to work unchanged.
New code should import from the specific source module instead.
"""

from __future__ import annotations

import _paths  # noqa: F401

import streamlit as st
import pandas as pd

from batlab.features.knee_detection import detect_knee as _detect_knee

# ── Re-exports for backward compatibility ──────────────────────────────────
# Every name that was previously in utils.py is re-exported here so that
# existing ``from utils import X`` call sites don't break.  New code should
# import from the specific module (e.g. ``from _design_tokens import LEGEND_H``).

from _design_tokens import (  # noqa: F401
    NASA_CELL_IDS,
    SEVERSON_CELL_PREFIX,
    MEASURED_CELL_IDS,
    LEGEND_H,
    PLOTLY_CONFIG,
    FEATURE_LABELS,
    CARD_BG,
    CARD_BORDER,
    PACK_BUNDLE_KEY,
)
_PACK_BUNDLE_KEY = PACK_BUNDLE_KEY  # backward compat alias

from _ui_helpers import (  # noqa: F401
    _md_html,
    _empty_state,
    _action_bar,
    render_card,
    metric_tile_html,
    base_layout,
    soh_status,
    friendly,
    _soh_sparkline_svg,
    _cell_provenance,
    _analysis_provenance,
    _cell_source,
    render_regenerate_report_button,
    render_pack_builder,
)


# ── Cache / data helpers (unique to this module) ───────────────────────────

@st.cache_data(show_spinner=False)
def cached_detect_knee(soh_series: "pd.Series", cycle_series: "pd.Series") -> dict:
    """Cached wrapper around batlab.features.knee_detection.detect_knee()."""
    return _detect_knee(soh_series, cycle_series)


@st.cache_data(show_spinner=False)
def cached_match_fleet(_trajectory_memory, all_featured_dfs: dict) -> dict:
    """Cached wrapper around TrajectoryMemory.match_fleet()."""
    return _trajectory_memory.match_fleet(all_featured_dfs)


@st.cache_resource(show_spinner=False)
def load_tenant_bundle_cached(org_id: int):
    """Streamlit-resource-cached wrapper around bundle_cache.load_tenant_bundle()."""
    from bundle_cache import load_tenant_bundle
    return load_tenant_bundle(org_id)


def _resample_df(df: "pd.DataFrame", max_points: int = 500) -> "pd.DataFrame":
    """Return df downsampled to at most max_points rows for trend charts."""
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    idx  = list(range(0, len(df), step))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    return df.iloc[idx].copy()
