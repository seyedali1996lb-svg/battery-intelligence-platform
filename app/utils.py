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


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _cell_provenance(cell_id: str) -> str:
    """Return the data-origin token for a given cell."""
    return "measured" if cell_id in NASA_CELL_IDS else "synthetic"


def _analysis_provenance(cell_id: str, analysis: str = "derived") -> str:
    """Return the provenance token for a specific analysis type."""
    if cell_id in NASA_CELL_IDS:
        return "measured" if analysis == "cycle" else "simulated"
    return "synthetic"


# ---------------------------------------------------------------------------
# Chart base layout helper
# ---------------------------------------------------------------------------

def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
    )
    if "xaxis" not in overrides:
        layout["xaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
    if "yaxis" not in overrides:
        layout["yaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
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
