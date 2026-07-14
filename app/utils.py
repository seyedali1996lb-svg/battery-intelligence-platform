"""
Shared UI helpers, constants, and small pure functions extracted from main.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from design_system import provenance_banner, BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC
from batlab.features.knee_detection import detect_knee as _detect_knee


@st.cache_data(show_spinner=False)
def cached_detect_knee(soh_series: "pd.Series", cycle_series: "pd.Series") -> dict:
    """Cached wrapper around batlab.features.knee_detection.detect_knee().

    Fleet page calls this once per cell in a loop, uncached — any widget
    interaction anywhere on the Fleet page (any single cell's data
    unchanged) recomputed knee detection for the WHOLE fleet on every
    single rerun. st.cache_data has built-in support for hashing pandas
    Series by content, so this is correctly cached per (cell's SOH
    series, cycle series) pair with no extra key-construction needed.
    """
    return _detect_knee(soh_series, cycle_series)


@st.cache_data(show_spinner=False)
def cached_match_fleet(_trajectory_memory, all_featured_dfs: dict) -> dict:
    """Cached wrapper around TrajectoryMemory.match_fleet().

    This was being computed up to 3x per Fleet-page render, and once on
    EVERY page in the app (not just Fleet) — app/main.py calls
    match_fleet() unconditionally before page routing just to size the
    sidebar trajectory-match badge, so every single widget interaction
    anywhere paid this cost, not only on Fleet.

    _trajectory_memory (leading underscore -> excluded from Streamlit's
    own argument hashing, its documented convention) is a TrajectoryMemory
    instance built once per session and never mutated afterward (see
    app/main.py's "built once per session" block) — safe to exclude from
    the cache key since it can't silently go stale mid-session the way a
    per-request object could. all_featured_dfs (the actual cache key) is
    a dict[str, DataFrame] — st.cache_data has built-in support for
    hashing dicts of DataFrames by content, verified before relying on it
    here, so no manual key construction is needed the way the PDF caching
    fix needed one (TrajectoryMemory's signatures aren't reliably
    JSON-serializable the way a passport dict is).
    """
    return _trajectory_memory.match_fleet(all_featured_dfs)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]
SEVERSON_CELL_PREFIX = "S-"   # all Severson cells start with "S-"
MEASURED_CELL_IDS = set(NASA_CELL_IDS)  # extended at load time

LEGEND_H = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#718096"),
)

# Publication-quality export config — Plotly toolbar SVG button.
PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "battery_intel_chart",
        "height": 500,
        "width": 900,
        "scale": 2,
    },
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}

FEATURE_LABELS = {
    "cycle_number":        "Cycle age",
    "fade_rate_10cy":      "Fade rate (10-cy)",
    "fade_rate_30cy":      "Fade rate (30-cy)",
    "fade_rate_50cy":      "Fade rate (50-cy)",
    "fade_acceleration":   "Fade acceleration",
    "soh_velocity_50cy":   "SOH velocity",
    "resistance_ohm":      "Internal resistance",
    "resistance_normalized": "Resistance (norm.)",
    "resistance_trend_30cy": "Resistance trend",
    "temp_rolling_30cy":   "Temperature (30-cy avg)",
    "c_rate_rolling_10cy": "C-rate (10-cy avg)",
    "stress_index":        "Composite stress index",
    "dod_proxy":           "Depth of Discharge (proxy)",
}

# ---------------------------------------------------------------------------
# HTML rendering helper
# ---------------------------------------------------------------------------

def _md_html(html: str) -> None:
    """Render an HTML string via st.markdown with blank-line stripping."""
    cleaned = "\n".join(ln for ln in html.split("\n") if ln.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def _empty_state(
    title: str,
    reason: str,
    action: str = "",
    icon: str = "○",
) -> None:
    """Render a designed empty state instead of a bare st.info()."""
    _md_html(
        f"<div class='empty-state'>"
        f"<div class='empty-state-icon'>{icon}</div>"
        f"<div class='empty-state-title'>{title}</div>"
        f"<div class='empty-state-body'>{reason}</div>"
        + (f"<div class='empty-state-action'>{action}</div>" if action else "")
        + "</div>"
    )


def _action_bar(page: str) -> None:
    """No-op spacer — sidebar covers all navigation needs."""
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared card/tile components (UI Design review finding)
# ---------------------------------------------------------------------------
# The exact bordered-card wrapper string
# ("background:#1e2a38;border:1px solid #2d3748;border-radius:...;padding:...")
# was hand-authored independently in 9+ page files (31+ occurrences) — no
# single enforced rendering path for "a bordered card" or "a label/value
# metric tile" meant every author reinvented it slightly differently, and
# drift (inconsistent colors, missing light-mode overrides, the
# UI_GREEN/UI_YELLOW NameError found earlier this session) was inevitable.
# These two primitives don't try to model every card's unique inner content
# — that would just become an unmaintainable options grab-bag — they
# replace the *wrapper* and the single most-repeated *inner* pattern
# (label/value/sub), which is what was actually duplicated everywhere.

CARD_BG     = "#1e2a38"
CARD_BORDER = "#2d3748"


def render_card(inner_html: str, border_color: str = CARD_BORDER,
                 padding: str = "16px 18px", extra_style: str = "") -> None:
    """Render one bordered card. Callers supply only their unique inner
    HTML — the wrapper (background/border/radius/padding) is centralized
    here instead of being hand-copied into every page file."""
    _md_html(
        f"<div style='background:{CARD_BG};border:1px solid {border_color};"
        f"border-radius:10px;padding:{padding};{extra_style}'>"
        f"{inner_html}"
        f"</div>"
    )


def metric_tile_html(label: str, value: str, sub: str = "",
                      value_color: str = "#e2e8f0", value_size: str = "20px") -> str:
    """Return the HTML for one label/value/sub metric tile — the single
    most-repeated inner pattern across Fleet/Overview/Decide & Ask/
    Compliance's metric rows. Returns a string (not rendered directly) so
    callers can compose several tiles inside one render_card() or a
    st.columns() layout."""
    return (
        f"<div style='font-size:10px;color:#4a5568;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:{value_size};font-weight:800;color:{value_color}'>{value}</div>"
        + (f"<div style='font-size:11px;color:#718096;margin-top:2px'>{sub}</div>" if sub else "")
    )


@st.cache_resource(show_spinner=False)
def load_tenant_bundle_cached(org_id: int):
    """Streamlit-resource-cached wrapper around bundle_cache.load_tenant_bundle().

    An org's uploaded ("My Data") featured_dfs/bundle/split_cycles triple was
    previously kept in st.session_state AND persisted to disk simultaneously
    — meaning every logged-in session held its own duplicate copy of the full
    DataFrames + trained model bundle in server RAM, on top of the disk copy,
    for the lifetime of that session. st.cache_resource is process-wide (one
    shared copy across every session of the same org on this server), not
    per-session, so routing every read through this function instead of raw
    session_state removes that duplication.

    Call load_tenant_bundle_cached.clear() after writing a NEW upload (see
    app/_pages/import_page.py) so the next call re-reads from disk instead of
    serving a stale cached triple — this function has no way to know the
    underlying disk file changed otherwise. Reads that only need to see the
    org's *existing* data (e.g. clearing/reverting to NASA mode) don't need
    to clear this cache — the disk data hasn't changed, so the cached triple
    is still correct.
    """
    from bundle_cache import load_tenant_bundle
    return load_tenant_bundle(org_id)


def _resample_df(df: "pd.DataFrame", max_points: int = 500) -> "pd.DataFrame":
    """Return df downsampled to at most max_points rows for trend charts.

    Uses uniform stride so the time axis stays evenly spaced. Always keeps
    the first and last row so chart endpoints are accurate. Applied to all
    trend-chart DataFrames before plotting — has no effect when len(df) <=
    max_points, so small datasets are returned unchanged.
    """
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    idx  = list(range(0, len(df), step))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    return df.iloc[idx].copy()


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _cell_provenance(cell_id: str) -> str:
    """Return the data-origin token for a given cell."""
    if cell_id in NASA_CELL_IDS or cell_id.startswith(SEVERSON_CELL_PREFIX):
        return "measured"
    return "synthetic"


def _analysis_provenance(cell_id: str, analysis: str = "derived") -> str:
    """Return the provenance token for a specific analysis type."""
    if cell_id in NASA_CELL_IDS or cell_id.startswith(SEVERSON_CELL_PREFIX):
        return "measured" if analysis == "cycle" else "simulated"
    return "synthetic"


def _cell_source(cell_id: str) -> str:
    """Coarse data-source tag — used to block physically-meaningless mixed-source
    packs (capacity/resistance scales aren't comparable across chemistries) and
    to pick the right per-cell bundle for RUL-reliability lookups."""
    from data_loader import CELL_STRESS_PROFILES
    if cell_id in NASA_CELL_IDS:
        return "nasa"
    if cell_id.startswith(SEVERSON_CELL_PREFIX):
        return "severson"
    if cell_id in CELL_STRESS_PROFILES:
        return "synthetic"
    return "uploaded"


# ---------------------------------------------------------------------------
# Chart base layout helper
# ---------------------------------------------------------------------------

def base_layout(**overrides) -> dict:
    _light = st.session_state.get("light_mode", False)
    _font_c = "#4a5568" if _light else "#a0aec0"
    _grid_c = "#e2e8f0" if _light else "#232d3b"
    _line_c = "#cbd5e0" if _light else "#2d3748"
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_font_c, size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
    )
    # Merge (not replace) so a caller-supplied xaxis/yaxis dict (e.g. just a
    # custom title or range) still inherits theme-aware grid/line colors
    # unless it explicitly overrides them itself.
    _default_axis = dict(gridcolor=_grid_c, linecolor=_line_c, zeroline=False)
    for _axis_key in ("xaxis", "yaxis"):
        _caller_axis = overrides.pop(_axis_key, None)
        _merged = dict(_default_axis)
        if _caller_axis:
            _merged.update(_caller_axis)
        layout[_axis_key] = _merged
    layout.update(overrides)
    return layout


def soh_status(soh: float) -> tuple[str, str]:
    if soh >= 90: return "Healthy",    "hero-green"
    if soh >= 80: return "Degrading",  "hero-yellow"
    return "End of Life", "hero-red"


def friendly(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def _soh_sparkline_svg(soh_series: "pd.Series", width: int = 120, height: int = 32) -> str:
    """Inline SVG mini-chart of recent SOH trend (last 50 cycles)."""
    vals = soh_series.dropna().tail(50).tolist()
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    pad = 2
    w, h = width - pad * 2, height - pad * 2
    pts = []
    for i, v in enumerate(vals):
        x = pad + i / (len(vals) - 1) * w
        y = pad + (1 - (v - lo) / span) * h
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    delta = vals[-1] - vals[0]
    stroke = "#48bb78" if delta >= -0.5 else ("#f6ad55" if delta >= -2 else "#fc8181")
    trend_word = "stable" if delta >= -0.5 else ("declining" if delta >= -2 else "fast-declining")
    aria_label = f"SOH sparkline: {trend_word}, {vals[-1]:.1f}% latest"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle" '
        f'role="img" aria-label="{aria_label}">'
        f'<title>{aria_label}</title>'
        f'<polyline points="{polyline}" fill="none" stroke="{stroke}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Virtual Pack Builder — shared by the Fleet page and Explore's Pack Builder tab
# ---------------------------------------------------------------------------

_PACK_BUNDLE_KEY = {"nasa": "nasa", "severson": "severson", "synthetic": "synth", "uploaded": "upload"}


def render_pack_builder(featured_dfs: dict, bundles: dict, key_prefix: str) -> None:
    """
    Shared Virtual Pack Builder — cell selection, series/parallel topology,
    pack metrics, and pairwise cell-matching scores. Used by both the Fleet
    page and the Explore page's Pack Builder tab, which previously had three
    independent, organically-diverged implementations (Fleet alone had two)
    computing slightly different things and silently allowing physically
    meaningless mixed-source packs in two of the three.

    key_prefix namespaces session_state keys (e.g. "fleet"/"explore") so
    both call sites keep independent selections without colliding.
    """
    from pack_builder import compute_pack_metrics, compute_matching_scores

    st.markdown("<div class='section-header'>Virtual Pack Builder</div>", unsafe_allow_html=True)
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Select cells to model as a series or parallel pack. Capacity and resistance scale "
        "differently across chemistries and sources, so selections must come from a single "
        "data source (NASA, Severson, synthetic, or uploaded)."
        "</div>"
    )

    cell_ids  = list(featured_dfs.keys())
    cells_key = f"{key_prefix}_pack_cells"
    topo_key  = f"{key_prefix}_pack_topology"

    # Drop any previously-selected cells that no longer exist in this data
    # source (e.g. after switching modes) — multiselect raises if its stored
    # session_state value contains an option not in the new options list.
    if cells_key in st.session_state:
        st.session_state[cells_key] = [c for c in st.session_state[cells_key] if c in cell_ids]

    selected = st.multiselect(
        "Select cells for virtual pack", options=cell_ids,
        default=cell_ids[:min(4, len(cell_ids))],
        key=cells_key,
    )

    if len(selected) < 2:
        _empty_state(
            "Select at least 2 cells",
            "Choose 2 or more cells from the same data source to build a virtual pack.",
            icon="🔋",
        )
        return

    sources = {_cell_source(c) for c in selected}
    if len(sources) > 1:
        _empty_state(
            "Mixed data sources selected",
            f"Selected cells span {', '.join(sorted(sources))} — capacity and resistance "
            "scales are not comparable across chemistries/sources. Choose cells from a "
            "single source.",
            "→ Narrow your selection to one source and try again.",
            "⚠",
        )
        return

    topology = st.radio("Configuration", ["Series", "Parallel"], horizontal=True, key=topo_key)

    _bundle = bundles.get(_PACK_BUNDLE_KEY.get(next(iter(sources)), "synth"))
    _per_cell_ok = (_bundle or {}).get("metrics", {}).get("per_cell_rul_reliable", {})
    _default_ok  = (_bundle or {}).get("metrics", {}).get("rul_reliable", False)

    cell_stats = []
    for cid in selected:
        df = featured_dfs.get(cid)
        if df is None or len(df) == 0:
            continue
        latest = df.iloc[-1]
        soh = float(latest["soh_pct"]) if "soh_pct" in latest.index else float("nan")
        if soh != soh:  # NaN
            continue
        cell_stats.append({
            "cell_id":        cid,
            "soh_pct":        soh,
            "capacity_ah":    float(latest["capacity_ah"]) if "capacity_ah" in latest.index else float("nan"),
            "resistance_ohm": float(latest["resistance_ohm"]) if "resistance_ohm" in latest.index else float("nan"),
            "rul_pred":       float(latest["rul_pred"]) if "rul_pred" in latest.index else None,
            "rul_reliable":   _per_cell_ok.get(cid, _default_ok),
        })

    if len(cell_stats) < 2:
        _empty_state(
            "Insufficient data",
            "Selected cells are missing capacity or SOH data at the latest cycle.",
            icon="⚠",
        )
        return

    metrics = compute_pack_metrics(cell_stats, topology)

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric(metrics["pack_soh_label"], f"{metrics['pack_soh']:.1f}%")
    _m2.metric("Pack RUL", f"{metrics['pack_rul']:.0f} cy" if metrics["pack_rul"] is not None else "—")
    _m3.metric("Pack Capacity", f"{metrics['pack_capacity_ah'] * 1000:.0f} mAh")
    _pack_res = metrics["pack_resistance_ohm"]
    _m4.metric("Pack Resistance", f"{_pack_res * 1000:.1f} mΩ" if _pack_res == _pack_res else "—")

    if metrics["spread_level"] == "Imbalanced":
        st.error(
            f"⚠️ **{metrics['bottleneck_cell_id']}** is the pack bottleneck "
            f"(SOH spread σ={metrics['soh_stdev']:.1f}%, range {metrics['soh_spread']:.1f}%). "
            f"Consider replacing or rebalancing."
        )
    elif metrics["spread_level"] == "Watch":
        st.warning(
            f"⚡ SOH spread is σ={metrics['soh_stdev']:.1f}% (range {metrics['soh_spread']:.1f}%). "
            f"Monitor {metrics['bottleneck_cell_id']} closely."
        )
    else:
        st.success(
            f"✅ Pack is well-balanced (SOH spread σ={metrics['soh_stdev']:.1f}%, "
            f"range {metrics['soh_spread']:.1f}%)"
        )
    if metrics["n_uncalibrated"]:
        st.caption(f"{metrics['n_uncalibrated']} cell(s) excluded from Pack RUL — not calibrated.")

    st.caption(
        f"Pack SOH is reported as {metrics['pack_soh_label'].lower()} — bottleneck-cell SOH is "
        "meaningful for series packs (usable capacity is gated by the weakest cell); "
        "capacity-weighted average is meaningful for parallel packs (capacity sums across cells)."
    )

    _soh_values = [c["soh_pct"] for c in cell_stats]
    _bar_colors = []
    for _sv in _soh_values:
        _dist = abs(_sv - metrics["pack_soh"])
        _bar_colors.append("#48bb78" if _dist <= 2 else ("#f6ad55" if _dist <= 5 else "#fc8181"))
    _fig_pack = go.Figure(go.Bar(
        x=[c["cell_id"] for c in cell_stats], y=_soh_values,
        marker_color=_bar_colors,
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    _fig_pack.add_hline(
        y=metrics["pack_soh"], line_dash="dash", line_color="#63b3ed", line_width=1,
        annotation_text=f"Pack SOH {metrics['pack_soh']:.1f}%", annotation_font_color="#63b3ed",
    )
    _fig_pack.update_layout(**base_layout(height=250, yaxis=dict(title="SOH %", range=[50, 102])))
    st.plotly_chart(_fig_pack, use_container_width=True)

    with st.expander("Cell matching & per-cell breakdown", expanded=False):
        st.caption(
            "Cells with similar degradation trajectories are better matched for pack "
            "assembly (minimises balancing losses)."
        )
        match_rows = compute_matching_scores(cell_stats)
        if match_rows:
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(cell_stats).set_index("cell_id"), use_container_width=True)
