"""
Small UI rendering helpers: HTML primitives, card/tile components, chart layout,
provenance labels, and backward-compatible re-exports of extracted widgets.

Big self-contained widgets (pack builder, report regeneration) now live in
their own modules but are re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import pandas as pd

import _paths  # noqa: F401

from _design_tokens import (
    CARD_BG, CARD_BORDER, FEATURE_LABELS,
)
from design_system import provenance_banner, BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC

# ── Re-exports from extracted widget modules (backward compat) ─────────────
from _pack_builder import render_pack_builder  # noqa: F401
from _report_regen import render_regenerate_report_button  # noqa: F401


# ---------------------------------------------------------------------------
# HTML rendering
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
# Shared card/tile components
# ---------------------------------------------------------------------------

def render_card(inner_html: str, border_color: str = CARD_BORDER,
                 padding: str = "16px 18px", extra_style: str = "") -> None:
    """Render one bordered card. Callers supply only their unique inner HTML."""
    _md_html(
        f"<div style='background:{CARD_BG};border:1px solid {border_color};"
        f"border-radius:10px;padding:{padding};{extra_style}'>"
        f"{inner_html}"
        f"</div>"
    )


def metric_tile_html(label: str, value: str, sub: str = "",
                      value_color: str = "#e2e8f0", value_size: str = "20px") -> str:
    """Return the HTML for one label/value/sub metric tile."""
    return (
        f"<div style='font-size:10px;color:#a0aec0;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:{value_size};font-weight:800;color:{value_color}'>{value}</div>"
        + (f"<div style='font-size:11px;color:#a0aec0;margin-top:2px'>{sub}</div>" if sub else "")
    )


# ---------------------------------------------------------------------------
# Chart layout
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
    _default_axis = dict(gridcolor=_grid_c, linecolor=_line_c, zeroline=False)
    for _axis_key in ("xaxis", "yaxis"):
        _caller_axis = overrides.pop(_axis_key, None)
        _merged = dict(_default_axis)
        if _caller_axis:
            _merged.update(_caller_axis)
        layout[_axis_key] = _merged  # pyright: ignore[reportArgumentType]
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def soh_status(soh: float) -> tuple[str, str]:
    if soh >= 90: return "Healthy",    "hero-green"
    if soh >= 80: return "Degrading",  "hero-yellow"
    return "End of Life", "hero-red"


def friendly(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

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
# Provenance helpers
# ---------------------------------------------------------------------------

def _cell_provenance(cell_id: str) -> str:
    """Return the data-origin token for a given cell."""
    from chemistry_profiles import ChemistryProfile
    return ChemistryProfile.for_cell(cell_id).provenance


def _analysis_provenance(cell_id: str, analysis: str = "derived") -> str:
    """Return the provenance token for a specific analysis type."""
    from chemistry_profiles import ChemistryProfile
    if ChemistryProfile.for_cell(cell_id).provenance == "measured":
        return "measured" if analysis == "cycle" else "simulated"
    return "synthetic"


def _cell_source(cell_id: str) -> str:
    """Coarse data-source tag for pack-builder source validation."""
    from chemistry_profiles import ChemistryProfile
    _kind = ChemistryProfile.for_cell(cell_id).source_kind
    return {"synth": "synthetic", "upload": "uploaded"}.get(_kind, _kind)  # pyright: ignore[reportReturnType]
