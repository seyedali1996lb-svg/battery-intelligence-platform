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
    SOH_EOL_MIN, SOH_HEALTHY_MIN,
)
from design_system import provenance_banner, BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC

# Widget re-exports are installed at the end of this module.  Importing them
# here would create a cycle because the widgets use helpers defined below.


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
    # Thresholds live in _design_tokens, not here: the scene's bands, the pack
    # chart's cylinders and the 3D scatter's ramp all read the same two numbers,
    # so no view can drift into calling a cell healthy while this one calls it
    # degrading. The CSS classes resolve to the same hexes (app/static/theme.css).
    if soh >= SOH_HEALTHY_MIN: return "Healthy",    "hero-green"
    if soh >= SOH_EOL_MIN:     return "Degrading",  "hero-yellow"
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


def _rul_interval_band_svg(
    cycles: pd.Series,
    rul_pred: pd.Series | None = None,
    rul_q10: pd.Series | None = None,
    rul_q90: pd.Series | None = None,
    observed_rul: pd.Series | None = None,
    eol_pct: float = 80.0,
    width: int = 320,
    height: int = 124,
    max_points: int = 220,
) -> str:
    """Inline SVG: the served RUL interval across the cell's life, drawn.

    x is the cycle an estimate was made at, y is cycles remaining to the
    end-of-life threshold, and the shaded band is the served Q10-Q90 interval
    at each of those cycles. Its width at any x is the interval width the model
    actually produced there -- narrowing, or refusing to narrow, in front of
    the reader instead of being asserted in prose. That is the point: an
    interval whose coverage was *measured* (batlab.validation.calibration)
    belongs on the first screen, not in a paragraph below it.

    ``observed_rul`` is the measured remaining life where the label is a
    genuinely observed end-of-life. The caller passes None (or all-NaN) for
    formula extrapolations, and nothing is drawn -- the chart never invents an
    outcome to compare against. Where it is passed, the served interval can be
    read against what actually happened rather than taken on faith.

    With no served interval -- RUL withheld because a fleet's labels are not
    observed end-of-life -- no band is drawn, and with no forecast at all the
    chart renders nothing. An absent band is the honest picture, not a missing
    feature.
    """
    def _series(v) -> pd.Series | None:
        if v is None:
            return None
        try:
            return pd.to_numeric(v, errors="coerce").reset_index(drop=True)
        except Exception:
            return None

    _cyc = _series(cycles)
    if _cyc is None or len(_cyc) < 2:
        return ""

    _frame = pd.DataFrame({"x": _cyc})
    if _frame["x"].isna().all():
        # Uploaded data can arrive without a cycle_number column: positional x
        # is the honest fallback, not a chart that collapses to nothing.
        _frame["x"] = range(len(_frame))
    _frame["x"] = _frame["x"].ffill().bfill()
    for _name, _vals in (
        ("pred", _series(rul_pred)),
        ("q10", _series(rul_q10)),
        ("q90", _series(rul_q90)),
        ("obs", _series(observed_rul)),
    ):
        # float64 throughout (missing -> NaN): a column that was never handed
        # over must behave like one that is entirely missing, not like an
        # object column that clip/compare will refuse to touch.
        _frame[_name] = pd.to_numeric(
            _vals if _vals is not None else pd.Series([pd.NA] * len(_frame)), errors="coerce"
        ).astype("float64")
    if _frame[["pred", "q90", "obs"]].isna().all().all():
        return ""          # nothing was forecast and nothing was observed

    if len(_frame) > max_points:
        _keep = sorted({round(i * (len(_frame) - 1) / (max_points - 1)) for i in range(max_points)})
        _frame = _frame.iloc[_keep].reset_index(drop=True)

    # A negative lower bound is not a remaining life: it is the model saying
    # the cell may already be past the threshold, so it clamps to 0 rather than
    # dropping the row -- the band should keep reading "0 to Q90" at the tail
    # instead of stopping short of the end of life.
    _frame["q10"] = _frame["q10"].clip(lower=0)
    # A row serves an interval only if both edges are present and the upper edge
    # is above the lower one. Q10 == 0 is NOT excluded: it is the served lower
    # bound on every cycle of some fleets, i.e. the interval is open at the low
    # end, and an open end is information -- hiding it would hide exactly the
    # case a user most needs to see, that the model cannot rule out that the
    # cell is already at end of life.
    _served = _frame["q10"].notna() & _frame["q90"].notna() & (_frame["q90"] > _frame["q10"])
    _ceiling = max(
        [float(_frame[c].max()) for c in ("q90", "pred", "obs") if _frame[c].notna().any()] or [0.0]
    )
    if _ceiling <= 0:
        return ""          # no remaining life anywhere in the series

    def _runs(mask) -> list[list[int]]:
        """Index runs of True, long enough to be drawn as a polygon."""
        out: list[list[int]] = []
        cur: list[int] = []
        for i, ok in enumerate(list(mask)):
            if ok:
                cur.append(i)
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return [run for run in out if len(run) >= 2]

    pad, _top = 4, 12          # _top leaves a strip for the legend line
    plot_w, plot_h = width - pad * 2, height - _top - pad
    x0, x1 = float(_frame["x"].iloc[0]), float(_frame["x"].iloc[-1])
    x_span = (x1 - x0) or 1.0

    def _px(c: float) -> float:
        return pad + (float(c) - x0) / x_span * plot_w

    def _py(v: float) -> float:
        return _top + (1 - min(max(float(v), 0.0), _ceiling) / _ceiling) * plot_h

    def _polyline(col: str, stroke: str, dash: str = "", weight: float = 1.5) -> str:
        rows = _frame[_frame[col].notna()]
        if len(rows) < 2:
            return ""
        pts = " ".join(f"{_px(c):.1f},{_py(v):.1f}" for c, v in zip(rows["x"], rows[col]))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{weight}"'
            f'{dash_attr} stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # One polygon per contiguous run of served cycles: a gap in the served
    # interval (or in the data) must not be bridged by a straight edge.
    _bands = ""
    _served_runs = _runs(_served)
    for _run in _served_runs:
        rows = _frame.iloc[_run]
        upper = " ".join(f"{_px(c):.1f},{_py(v):.1f}" for c, v in zip(rows["x"], rows["q90"]))
        lower = " ".join(
            f"{_px(c):.1f},{_py(v):.1f}" for c, v in reversed(list(zip(rows["x"], rows["q10"])))
        )
        _bands += (
            f'<polygon points="{upper} {lower}" fill="#63b3ed" fill-opacity="0.16" '
            f'stroke="#63b3ed" stroke-opacity="0.5" stroke-width="0.8" stroke-dasharray="3,3"/>'
        )

    _y_zero = _py(0.0)
    _legend = []
    if _served_runs:
        _legend.append(f"band = served {eol_pct:.0f}% interval")
    if _frame["obs"].notna().any():
        _legend.append("dashed = observed outcome")
    if not _legend:
        _legend.append("forecast only - no interval served")

    if _served_runs:
        _last = _frame[_served].iloc[-1]
        _aria = (
            f"Remaining-life chart: {len(_frame)} cycles plotted. The served {eol_pct:.0f}% "
            f"interval is drawn for {int(_served.sum())} of them, most recently "
            f"{_last['q10']:.0f} to {_last['q90']:.0f} cycles remaining at cycle {_last['x']:,.0f}"
        )
    else:
        _aria = (
            f"Remaining-life chart: {len(_frame)} cycles plotted, no {eol_pct:.0f}% interval "
            f"served for this cell"
        )
    if _frame["obs"].notna().any():
        _aria += ", drawn against the observed remaining life"
    _aria += "."

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;margin:6px auto 0" '
        f'role="img" aria-label="{_aria}">'
        f'<title>{_aria}</title>'
        f'<text x="{pad}" y="9" fill="#8896a8" font-size="9" font-family="monospace">'
        f'{" · ".join(_legend)}</text>'
        # Zero remaining life is the threshold the interval is defined against.
        f'<line x1="{pad}" y1="{_y_zero:.1f}" x2="{pad + plot_w}" y2="{_y_zero:.1f}" '
        f'stroke="#8896a8" stroke-width="1" stroke-dasharray="4,4" stroke-opacity="0.7"/>'
        f'<text x="{pad + plot_w}" y="{_y_zero - 3:.1f}" text-anchor="end" fill="#8896a8" '
        f'font-size="9" font-family="monospace">{eol_pct:.0f}% EOL = 0 cycles left</text>'
        f'{_bands}'
        f'{_polyline("pred", "#63b3ed")}'
        f'{_polyline("obs", "#48bb78", dash="4,3")}'
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


# Backward-compatible widget re-exports.  They are imported only after all
# helper definitions exist, avoiding the _ui_helpers <-> _pack_builder cycle.
from _pack_builder import render_pack_builder  # noqa: F401, E402
from _report_regen import render_regenerate_report_button  # noqa: F401, E402
