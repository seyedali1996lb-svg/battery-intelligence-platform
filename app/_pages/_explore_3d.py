"""
Explore > 3D Degradation Space — the fleet drawn with all three axes a
cycler record actually has.

Every other fleet chart in this app is a 2D projection of a cell's life: SOH
against cycle number (the fade curve), SOH against fade rate (Cluster's phase
plot). Both are useful and both throw away a third axis that the records
already carry. This view plots each cell's whole trajectory in
(cycle age, SOH, third axis) so aging keeps its shape: a fade curve is a
descent, a knee is a bend in the path, and resistance growth is a wall the
trajectory climbs as it falls. Cells that look identical in 2D separate here.

Everything drawn comes from the cell's own record — no model output, no
projection, no fitted curve. The one piece of reference geometry is the
translucent plane at the platform's 80% end-of-life convention, which is a
threshold applied to the cell's own first *measured* capacity (METHODOLOGY.md),
not a manufacturer's specification.

Data access follows the fleet-scale convention in src/cell_store.py's module
docstring: a cell's series is read as a pruned, memory-mapped Parquet column
read (`get_cell_df(columns=...)`) rather than by iterating every cell's full
DataFrame, and only the cells being plotted are read at all. Column
availability is probed from the file footer (`available_columns()`), because a
source can lack an axis outright — Zhu 2022's records have no resistance
column, and a single-temperature fleet's ``temperature_c`` is a logged
set-point rather than a measured range.

The pure builders at the top of this module (``axis_coverage``,
``build_cell_path``, ``z_at_cycle``, ``build_degradation_figure``) take DataFrames
and return arrays/figures, so they are unit-tested directly in
tests/test_degradation_space_3d.py without a Streamlit runtime.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

from _design_tokens import SOH_DEGRADING_COLOR, SOH_EOL_COLOR, SOH_EOL_MIN, SOH_HEALTHY_COLOR

from _ui_helpers import _empty_state, base_layout
from utils import _resample_df

# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


class ZAxisSpec(TypedDict):
    """One candidate third axis (see Z_AXIS_SPECS)."""

    label: str          # axis title as shown to the reader
    scale: float        # stored unit -> displayed unit
    positive_only: bool  # non-positive readings are artefacts, not measurements
    basis: str          # measured / derived — stated in the view's notes


# Third-axis candidates in preference order. ``scale`` converts the stored unit
# to the unit shown on the axis. ``positive_only`` marks a quantity whose
# non-positive readings are sentinel/parse artefacts rather than measurements —
# a 0 Ω internal resistance is not a physical reading (Severson's raw r1 log
# column parses to 0.0 on some cycles), whereas SOH non-increase is a real
# observation and is kept.
Z_AXIS_SPECS: dict[str, ZAxisSpec] = {
    "resistance_ohm": {
        "label": "Internal resistance (mΩ)",
        "scale": 1000.0,
        "positive_only": True,
        "basis": "derived — start-of-discharge voltage step",
    },
    "fade_rate_30cy": {
        "label": "30-cycle fade rate (mSOH/cy)",
        "scale": 1000.0,
        "positive_only": False,
        "basis": "derived — SOH slope over the trailing 30 cycles",
    },
    "temperature_c": {
        "label": "Temperature (°C)",
        "scale": 1.0,
        "positive_only": False,
        "basis": "measured or logged set-point — see the notes below",
    },
}

# The platform's end-of-life convention — the same number soh_status() and the
# pack chart band on, read from _design_tokens rather than retyped here.
EOL_PCT = SOH_EOL_MIN  # the platform's own end-of-life convention

# SOH-derived colour ramp, matching the app's card palette: red at/below End of
# Life, amber through the degrading band, green when healthy — the three band
# colours from _design_tokens, so this ramp cannot disagree with the bands the
# rest of the app paints a cell with. The range is pinned (not data-fitted) so
# two cells at the same SOH are the same colour.
SOH_COLORSCALE = [
    [0.00, SOH_EOL_COLOR],
    [0.50, SOH_EOL_COLOR],
    [0.65, SOH_DEGRADING_COLOR],
    [0.90, SOH_DEGRADING_COLOR],
    [1.00, SOH_HEALTHY_COLOR],
]
SOH_CBAR_MIN, SOH_CBAR_MAX = 60.0, 100.0

_CELL_PALETTE = [
    "#63b3ed", "#68d391", "#f6ad55", "#fc8181",
    "#9f7aea", "#4fd1c5", "#f687b3", "#ecc94b",
]
# Past the legend limit every path shares one muted thread (hover carries the
# cell id) — #8896a8 is this app's existing muted chart tone and reads on both
# the light and dark themes.
_SHARED_LINE_COLOR = "#8896a8"
LEGEND_CELL_LIMIT = 8      # past this, a per-cell legend is noise — hover carries the name
MAX_PATH_POINTS = 400      # per-cell points drawn (evenly sampled, last cycle always kept)
DEFAULT_PLOT_ALL_LIMIT = 12
DEFAULT_PLOTTED_CELLS = 8


# ---------------------------------------------------------------------------
# Pure builders (unit-tested directly)
# ---------------------------------------------------------------------------

def axis_coverage(column_lists: list[list[str]]) -> dict[str, int]:
    """How many of the given records carry each candidate third axis.

    ``column_lists`` is one column-name list per cell (from
    ``cell_store.available_columns()``). Order follows Z_AXIS_SPECS, i.e. the
    preference order used to pick a default axis.
    """
    return {
        col: sum(1 for cols in column_lists if col in cols)
        for col in Z_AXIS_SPECS
    }


def build_cell_path(
    df: "pd.DataFrame | None",
    z_column: str,
    *,
    scale: float = 1.0,
    positive_only: bool = False,
    max_points: int = MAX_PATH_POINTS,
) -> "dict[str, object] | None":
    """One cell's trajectory through the space, as plottable arrays.

    Rows are dropped when any of the three coordinates is missing or
    non-finite, and — for a ``positive_only`` axis — when the third-axis
    reading is not a physical value. The dropped count is *returned*, not
    hidden, so the caller can disclose it rather than silently plotting a
    tidied path.

    Returns None when there is nothing to plot: an absent column, an empty
    record, or no usable rows.
    """
    if df is None or len(df) == 0:
        return None
    if z_column not in df.columns:
        return None
    for required in ("cycle_number", "soh_pct"):
        if required not in df.columns:
            return None

    work = pd.DataFrame({
        "cycle_number": pd.to_numeric(df["cycle_number"], errors="coerce"),
        "soh_pct": pd.to_numeric(df["soh_pct"], errors="coerce"),
        "z": pd.to_numeric(df[z_column], errors="coerce").astype(float) * scale,
    })
    if positive_only:
        work.loc[work["z"] <= 0, "z"] = np.nan

    n_series = len(work)
    work = work.dropna()
    if work.empty:
        return None

    plotted = _resample_df(work, max_points=max_points)
    return {
        "cycle_number": plotted["cycle_number"].to_numpy(dtype=float),
        "soh_pct": plotted["soh_pct"].to_numpy(dtype=float),
        "z": plotted["z"].to_numpy(dtype=float),
        "n_series": n_series,
        "n_plotted": int(len(work)),
        "n_dropped": n_series - int(len(work)),
        "first_cycle": float(work["cycle_number"].iloc[0]),
        "last_cycle": float(work["cycle_number"].iloc[-1]),
        "first_z": float(work["z"].iloc[0]),
        "last_z": float(work["z"].iloc[-1]),
        "first_soh": float(work["soh_pct"].iloc[0]),
        "latest_soh": float(work["soh_pct"].iloc[-1]),
    }


def z_at_cycle(path: "dict[str, object]", cycle: float) -> "float | None":
    """The third-axis value a plotted path passes through at ``cycle``.

    Linear interpolation between the recorded points. Returns None when the
    cycle falls outside the recorded range — a marker there would be invented
    rather than read, which is the one thing this view must not do.
    """
    xs = np.asarray(path["cycle_number"], dtype=float)
    zs = np.asarray(path["z"], dtype=float)
    if len(xs) < 2 or cycle < float(xs[0]) or cycle > float(xs[-1]):
        return None
    return float(np.interp(float(cycle), xs, zs))


def build_degradation_figure(
    paths: "dict[str, dict]",
    z_column: str,
    *,
    knee_points: "list[tuple[str, float, float, float]] | None" = None,
) -> go.Figure:
    """The 3D figure: one trajectory per cell, the EOL plane, optional knees.

    ``paths`` maps cell_id -> build_cell_path() output, in plot order.
    ``knee_points`` is a list of (cell_id, cycle, soh, z) tuples.
    """
    spec = Z_AXIS_SPECS[z_column]
    z_label = spec["label"]
    base = base_layout()
    _x2d = base.get("xaxis") or {}
    _font = base.get("font") or {}
    grid_c = _x2d.get("gridcolor", "#232d3b")
    line_c = _x2d.get("linecolor", "#2d3748")
    font_c = _font.get("color", "#a0aec0")
    many = len(paths) > LEGEND_CELL_LIMIT

    fig = go.Figure()
    for idx, (cell_id, path) in enumerate(paths.items()):
        soh = np.asarray(path["soh_pct"], dtype=float)
        fig.add_trace(go.Scatter3d(
            x=np.asarray(path["cycle_number"], dtype=float),
            y=soh,
            z=np.asarray(path["z"], dtype=float),
            mode="lines+markers",
            name=str(cell_id),
            showlegend=not many,
            line=dict(
                color=_SHARED_LINE_COLOR if many else _CELL_PALETTE[idx % len(_CELL_PALETTE)],
                width=3,
            ),
            marker=dict(
                size=3,
                color=soh,
                colorscale=SOH_COLORSCALE,
                cmin=SOH_CBAR_MIN,
                cmax=SOH_CBAR_MAX,
                showscale=(idx == 0),
                colorbar=dict(
                    title=dict(text="SOH %", font=dict(color=font_c, size=11)),
                    thickness=12, len=0.55, outlinewidth=0,
                    tickfont=dict(color=font_c, size=10),
                ),
            ),
            hovertemplate=(
                f"<b>{cell_id}</b><br>Cycle %{{x:.0f}}<br>SOH %{{y:.1f}}%"
                f"<br>{z_label}: %{{z:.3f}}<extra></extra>"
            ),
        ))

    x_all = np.concatenate([np.asarray(p["cycle_number"], dtype=float) for p in paths.values()])
    z_all = np.concatenate([np.asarray(p["z"], dtype=float) for p in paths.values()])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    z_min, z_max = float(np.min(z_all)), float(np.max(z_all))
    z_pad = (z_max - z_min) * 0.05 or max(abs(z_max), 1.0) * 0.05

    # End-of-life plane: the platform's 80%-of-own-first-measured-capacity
    # convention, drawn as reference geometry the trajectories can cross.
    fig.add_trace(go.Surface(
        x=[[x_min, x_max], [x_min, x_max]],
        y=[[EOL_PCT, EOL_PCT], [EOL_PCT, EOL_PCT]],
        z=[[z_min - z_pad, z_min - z_pad], [z_max + z_pad, z_max + z_pad]],
        showscale=False,
        opacity=0.12,
        colorscale=[[0.0, "#fc8181"], [1.0, "#fc8181"]],
        hoverinfo="skip",
        showlegend=True,
        name=f"{EOL_PCT:.0f}% EOL threshold",
    ))

    if knee_points:
        fig.add_trace(go.Scatter3d(
            x=[k[1] for k in knee_points],
            y=[k[2] for k in knee_points],
            z=[k[3] for k in knee_points],
            mode="markers",
            name=f"Knee point ({len(knee_points)})",
            text=[k[0] for k in knee_points],
            marker=dict(size=6, color="#f6ad55", symbol="diamond",
                        line=dict(width=1, color="#0e1117")),
            hovertemplate=(
                "<b>%{text}</b><br>Detected knee — cycle %{x:.0f}, "
                f"SOH %{{y:.1f}}%<br>{z_label}: %{{z:.3f}}<extra></extra>"
            ),
        ))

    def _scene_axis(title: str) -> dict:
        return dict(
            title=title, gridcolor=grid_c, linecolor=line_c, zeroline=False,
            showbackground=False, backgroundcolor="rgba(0,0,0,0)",
        )

    layout = {k: v for k, v in base.items() if k not in ("xaxis", "yaxis")}
    layout.update(
        height=620,
        hovermode="closest",
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.98, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=font_c),
        ),
        scene=dict(
            xaxis=_scene_axis("Cycle age"),
            yaxis=_scene_axis("SOH %"),
            zaxis=_scene_axis(z_label),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.85)),
        ),
    )
    fig.update_layout(**layout)  # pyright: ignore[reportArgumentType]
    return fig


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _stored_columns(cell_id: str) -> list[str]:
    """Column names in a cell's stored series (Parquet footer only)."""
    try:
        from cell_store import available_columns
        return list(available_columns(cell_id) or [])
    except Exception:
        return []


def _cell_columns(cell_id: str, active_fdfs: dict) -> list[str]:
    """Column names available for ``cell_id``.

    Prefers the stored series' footer (no data decoded). Only when a cell has
    no stored Parquet file — e.g. a session-only uploaded fleet living in the
    tenant bundle — does this fall back to the loaded frame itself, which for a
    single explicitly-selected cell is the bounded-subset case cell_store.py's
    module docstring allows.
    """
    cols = _stored_columns(cell_id)
    if cols:
        return cols
    try:
        df = active_fdfs.get(cell_id)
    except Exception:
        df = None
    return list(df.columns) if df is not None else []


def _read_series(cell_id: str, columns: tuple, active_fdfs: dict) -> "pd.DataFrame | None":
    """A pruned (column-subset) read of one cell's series, or None."""
    wanted = [c for c in columns]
    try:
        from cell_store import get_cell_df
        df = get_cell_df(cell_id, columns=wanted)
        if df is not None:
            return df
    except Exception:
        pass
    try:
        df = active_fdfs.get(cell_id)
    except Exception:
        df = None
    if df is None:
        return None
    return df[[c for c in wanted if c in df.columns]]


def _knee_rows_by_cell(org_id: "int | None") -> dict:
    """CellSummary rows keyed by cell_id — the precomputed knee estimates.

    Missing rows (a cell whose summary was never persisted) simply carry no
    marker; nothing is recomputed here from the full history.
    """
    if org_id is None:
        return {}
    try:
        import db as _db
        return {r["cell_id"]: r for r in _db.get_cell_summaries(int(org_id))}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_degradation_space_3d(cell_ids: list, active_fdfs: dict) -> None:
    """Explore's 3D view: every plotted cell's path through
    (cycle age, SOH, selected third axis)."""
    st.markdown("<h4 class='section-header'>Fleet degradation space</h4>", unsafe_allow_html=True)
    st.caption(
        "Each line is one cell's own measured record — cycle age, SOH, and a third axis of "
        "your choosing — drawn as a path instead of a flat curve. Drag to rotate; the shape "
        "of a knee, and the resistance growth behind it, are both visible here."
    )

    if not cell_ids:
        _empty_state(
            "No cells in the active fleet",
            "Load a reference fleet or import your own cycling data to plot its degradation space.",
            icon="◎",
        )
        return

    _plot_all = st.checkbox(
        f"Plot every cell in the active fleet ({len(cell_ids)})",
        value=len(cell_ids) <= DEFAULT_PLOT_ALL_LIMIT,
        key="explore3d_plot_all",
        help="Series are read per cell as pruned Parquet column reads, so the view stays "
             "cheap at fleet scale — no full-fleet DataFrame is materialized.",
    )
    if _plot_all:
        selected = list(cell_ids)
        st.caption(f"Plotting all {len(selected)} cells.")
    else:
        _stored = st.session_state.get("explore3d_cells")
        if isinstance(_stored, list):
            _kept = [c for c in _stored if c in cell_ids]
            if len(_kept) != len(_stored):
                st.session_state["explore3d_cells"] = _kept
        selected = st.multiselect(
            "Cells to plot", options=list(cell_ids),
            default=list(cell_ids[:DEFAULT_PLOTTED_CELLS]), key="explore3d_cells",
        )
    if not selected:
        _empty_state(
            "Select at least one cell",
            "Choose one or more cells to draw their trajectories.",
            icon="◎",
        )
        return

    # ── Which third axis the selected records actually carry ────────────────
    coverage = axis_coverage([_cell_columns(cid, active_fdfs) for cid in selected])
    axes = [col for col in Z_AXIS_SPECS if coverage.get(col)]
    if not axes:
        _empty_state(
            "No third axis available for these cells",
            "None of the selected records carry an internal-resistance, fade-rate, or "
            "temperature column — only cycle age and SOH could be plotted.",
            icon="◎",
        )
        return

    if st.session_state.get("explore3d_z_axis") not in axes:
        st.session_state["explore3d_z_axis"] = axes[0]
    z_col = st.radio(
        "Third axis", options=axes,
        format_func=lambda c: str(Z_AXIS_SPECS[c]["label"]),
        horizontal=True, key="explore3d_z_axis",
    )
    spec = Z_AXIS_SPECS[z_col]
    z_label = spec["label"]

    # ── Build one path per cell ─────────────────────────────────────────────
    paths: dict[str, dict] = {}
    lacking_axis: list[str] = []
    no_usable_rows: list[str] = []
    for cid in selected:
        df = _read_series(cid, ("cycle_number", "soh_pct", z_col), active_fdfs)
        path = build_cell_path(
            df, z_col, scale=spec["scale"], positive_only=spec["positive_only"],
        )
        if path is None:
            if df is not None and z_col in df.columns:
                no_usable_rows.append(cid)
            else:
                lacking_axis.append(cid)
            continue
        paths[cid] = path

    if not paths:
        _empty_state(
            f"No usable {z_label} readings",
            "The selected cells have no finite value on this axis — try another third axis.",
            icon="◎",
        )
        return

    # ── Optional knee markers, from the precomputed CellSummary rows ────────
    summaries = _knee_rows_by_cell(st.session_state.get("auth_org_id"))
    has_knees = any(
        (summaries.get(cid) or {}).get("knee_detected") for cid in paths
    )
    show_knees = st.checkbox(
        "Mark detected knee points", value=has_knees, key="explore3d_knees",
        disabled=not has_knees,
        help="Knee = the inflection where linear fade turns into accelerating loss "
             "(batlab.features.knee_detection). A cell with no knee detected in-window "
             "carries no marker — that is the honest output, not a missing one.",
    )
    knee_points: list[tuple[str, float, float, float]] = []
    if show_knees and has_knees:
        for cid, path in paths.items():
            row = summaries.get(cid) or {}
            if not row.get("knee_detected"):
                continue
            knee_cycle, knee_soh = row.get("knee_cycle"), row.get("knee_soh")
            if knee_cycle is None or knee_soh is None:
                continue
            z_knee = z_at_cycle(path, float(knee_cycle))
            if z_knee is None:
                continue
            knee_points.append((cid, float(knee_cycle), float(knee_soh), z_knee))

    # ── Chart ───────────────────────────────────────────────────────────────
    fig = build_degradation_figure(paths, z_col, knee_points=knee_points)
    st.plotly_chart(fig, use_container_width=True)

    # ── What the picture says, in numbers ───────────────────────────────────
    _n = len(paths)
    _cyc_lo = min(float(p["first_cycle"]) for p in paths.values())
    _cyc_hi = max(float(p["last_cycle"]) for p in paths.values())
    _soh_lo = min(float(np.min(np.asarray(p["soh_pct"], dtype=float))) for p in paths.values())
    _soh_hi = max(float(np.max(np.asarray(p["soh_pct"], dtype=float))) for p in paths.values())
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Cells plotted", f"{_n}")
    _m2.metric("Cycle span", f"{_cyc_lo:,.0f}–{_cyc_hi:,.0f}")
    _m3.metric("SOH range", f"{_soh_lo:.1f}–{_soh_hi:.1f}%")
    _m4.metric("Knee points marked", f"{len(knee_points)}" if show_knees else "—")

    with st.expander("Per-cell path summary"):
        st.dataframe(pd.DataFrame([
            {
                "Cell": cid,
                "Cycles plotted": f"{p['n_plotted']:,}",
                "Cycle range": f"{p['first_cycle']:,.0f}–{p['last_cycle']:,.0f}",
                "SOH first → last": f"{p['first_soh']:.1f} → {p['latest_soh']:.1f}%",
                f"{z_label} first → last": f"{p['first_z']:.3f} → {p['last_z']:.3f}",
                "Rows set aside": p["n_dropped"],
            }
            for cid, p in paths.items()
        ]), use_container_width=True, hide_index=True)

    # ── Disclosures ─────────────────────────────────────────────────────────
    _dropped = sum(int(p["n_dropped"]) for p in paths.values())
    if _dropped:
        st.caption(
            f"{_dropped:,} cycle row(s) were set aside before plotting: a missing or "
            f"non-finite reading on one of the three axes"
            + (" (a non-positive resistance reading is a sentinel or parse artefact, not a "
               "measurement)" if spec["positive_only"] else "")
            + ". The per-cell counts are in the table above."
        )
    if lacking_axis:
        _names = ", ".join(lacking_axis[:6]) + (f" (+{len(lacking_axis) - 6} more)" if len(lacking_axis) > 6 else "")
        st.caption(
            f"{len(lacking_axis)} selected cell(s) have no {z_label} column at all and are "
            f"not plotted — {_names}. Their records simply do not carry this axis; nothing "
            "was estimated to fill it in."
        )
    if no_usable_rows:
        _names = ", ".join(no_usable_rows[:6]) + (f" (+{len(no_usable_rows) - 6} more)" if len(no_usable_rows) > 6 else "")
        st.caption(
            f"{len(no_usable_rows)} selected cell(s) have no finite {z_label} readings and are "
            f"not plotted — {_names}."
        )
    if has_knees and not show_knees:
        st.caption(
            "Knee markers are available for this fleet but switched off. The knee estimates "
            "themselves (cycle, SOH, confidence, phase) are tabulated on Fleet and Grading."
        )

    with st.expander("How to read this space"):
        st.markdown(
            f"- **Axes.** Cycle age (measured), SOH (derived — the cycle's measured discharge "
            f"capacity over the cell's own first measured capacity), {z_label} "
            f"({spec['basis']}).\n"
            f"- **The red translucent plane** is the platform's own end-of-life convention, "
            f"{EOL_PCT:.0f}% of each cell's first measured capacity — a defined threshold, not "
            "a manufacturer spec and not a measured failure point. Paths that cross it are "
            "past end-of-life under that convention.\n"
            "- **Marker colour** is the SOH at that cycle on a fixed 60–100% ramp, so equal "
            "colours mean equal SOH across cells and fleets.\n"
            "- **Knee markers** are the precomputed ``knee_detection`` estimates for each cell, "
            "plotted at the recorded cycle and interpolated on the third axis. A missing "
            "marker means no knee was detected in-window, which is the detector's honest "
            "output rather than a missing feature.\n"
            "- **Resistance** is a start-of-discharge-pulse estimate, the same proxy the "
            "platform's State-of-Power audit scopes at ±25% (\"proxy, not a measured power "
            "test\"). Read its *direction and shape* here, not its absolute value.\n"
            "- **Temperature** is measured where the records carry a real spread (NASA, "
            "Severson, the synthetic fleet). A flat ribbon means one logged set-point — Zhu "
            "2022's 25 °C cells will draw a plane, not a cloud.\n"
            "- **No predictions here.** This view is measured/derived history only; projected "
            "SOH and RUL live on the Cell Workbench's Health view and the Benchmark page."
        )

    st.caption(
        "Read-only view of the active fleet's own records. Rotate with the mouse, zoom with "
        "the wheel; the toolbar's camera icon exports a PNG."
    )
