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
