"""
Virtual Pack Builder's 3D layout view — the selected cells drawn as physical
cylinders, coloured by SOH, with the pack's bottleneck called out.

The Pack Builder's bar chart answers "which cell is weakest" precisely; it does
not answer "what does that mean for the pack as it is actually wired", because
a bar chart has no notion of position or connection. This view gives the
selection a place and a topology:

- **Series** — a string: one row, in selection order, each cell's positive
  terminal joined to the next cell's negative.
- **Parallel** — a block: cells sharing the same two nodes.
- **XpYs** — groups strung along the series direction, each group's members
  side by side because they share a node pair. Two call-outs appear here that a
  flat topology cannot express: a translucent box around the **bottleneck
  group** (the string's capacity ceiling) and a halo around the **most-loaded
  cell** inside its group (the resistance-based current-sharing winner).

Honesty notes this view carries (stated on the chart, not just here):

- The cylinder height is a **visual metaphor** when the "scale height by SOH"
  toggle is on — a real 18650 is the same size at 100% and at 65% SOH. Colour
  is the measurement (the cell's own SOH, on the same 90/80 bands the rest of
  the app uses via ``_ui_helpers.soh_status()``); height only makes the spread
  readable at a glance.
- Cylinders are drawn as shells, not as the cells' real geometry: real cells
  differ in form factor between sources (NASA 18650s, CALCE prismatics), and
  the layout is a schematic of the wiring, not a CAD model of a specific pack.
- The interconnect lines show *which* terminals are joined (series chain, group
  ladders, or shared parallel nodes). They are not busbar routing or
  current-carrying sizing — nothing here is a pack-design output.
- Current sharing is the first-order resistance-only model documented on
  ``src/pack_builder.compute_group_current_sharing()``; the chart points at the
  model's scope rather than implying a thermal-electrical simulation.

The builders (``cylinder_mesh``, ``box_mesh``, ``layout_positions``,
``grouped_layout_positions``, ``cell_height``, ``soh_band_color``,
``build_pack_layout_figure``) are pure: they take plain dicts, numpy and plotly
only, and are unit-tested in tests/test_pack_layout_3d.py without a Streamlit
runtime.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

from _design_tokens import (
    SOH_DEGRADING_COLOR,
    SOH_EOL_COLOR,
    SOH_EOL_MIN,
    SOH_HEALTHY_COLOR,
    SOH_HEALTHY_MIN,
)

# ---------------------------------------------------------------------------
# Geometry (arbitrary but consistent units; the scene uses aspectmode="data",
# so proportions are the only thing that matters). A real 18650 is ~18 mm
# across and ~65 mm tall, hence radius 0.5 → height 3.6.
# ---------------------------------------------------------------------------

CELL_RADIUS = 0.5
CELL_HEIGHT = 3.6
CELL_SPACING = 1.35          # centre-to-centre within a group / string
GROUP_SPACING = 1.9          # centre-to-centre between parallel groups
TERMINAL_RADIUS = 0.17
TERMINAL_HEIGHT = 0.10
HALO_RADIUS = 0.62        # < CELL_SPACING / 2, so two halos never merge into one blob
HALO_BOTTOM = -0.06
HALO_TOP_PAD = 0.45
BOX_TOP_PAD = 0.5
SEGMENTS = 24                # 24-gon per ring: smooth at pack-view zoom levels
PARALLEL_ROW_WIDTH = 4       # cells per row in a flat parallel block
FLOOR_MARGIN = 1.1

SOH_HEIGHT_MIN = 0.25        # a cell at/below the floor draws at 25% height
# (25%, not a smaller value, so a fully-degraded cylinder still reads as a
# cylinder rather than a disc; the useful range spans SOH_HEIGHT_FLOOR..100%,
# which covers every pack this platform's fleets can actually produce.)
SOH_HEIGHT_FLOOR = 50.0      # SOH% at which the height metaphor bottoms out

# SOH colour bands — the same thresholds and palette as _design_tokens'
# SOH_HEALTHY_MIN/SOH_EOL_MIN and the three band colours, which is also what
# app/_ui_helpers.py's soh_status() (Healthy / Degrading / End of Life) reads.
# One definition, so a cell is never green on this chart and amber elsewhere.
SOH_BANDS: tuple[tuple[float, str, str], ...] = (
    (SOH_HEALTHY_MIN, SOH_HEALTHY_COLOR, f"Healthy (≥{SOH_HEALTHY_MIN:.0f}% SOH)"),
    (SOH_EOL_MIN, SOH_DEGRADING_COLOR, f"Degrading ({SOH_EOL_MIN:.0f}–{SOH_HEALTHY_MIN:.0f}% SOH)"),
    (float("-inf"), SOH_EOL_COLOR, f"End of Life (<{SOH_EOL_MIN:.0f}% SOH)"),
)

_INTERCONNECT_COLOR = "#8896a8"
_TERMINAL_COLOR = "#cbd5e0"
_BOTTLENECK_COLOR = "#fc8181"


def _finite_soh(soh: "float | None") -> "float | None":
    """``soh`` as a real number, or None when it is missing/non-numeric/NaN."""
    if soh is None:
        return None
    try:
        value = float(soh)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def soh_band_color(soh: "float | None") -> str:
    """The band colour for one cell's SOH (see SOH_BANDS)."""
    value = _finite_soh(soh)
    if value is None:
        return SOH_BANDS[-1][1]
    for threshold, color, _label in SOH_BANDS:
        if value >= threshold:
            return color
    return SOH_BANDS[-1][1]


def soh_band_label(soh: "float | None") -> str:
    """The band's human label for one cell's SOH."""
    value = _finite_soh(soh)
    if value is None:
        return "SOH unavailable"
    for _threshold, _color, label in SOH_BANDS:
        if value >= _threshold:
            return label
    return SOH_BANDS[-1][2]


def cell_height(soh: float, *, scale_by_soh: bool, base_height: float = CELL_HEIGHT) -> float:
    """Drawn height of one cell.

    With ``scale_by_soh``, height falls linearly from ``base_height`` at 100%
    SOH to ``SOH_HEIGHT_MIN * base_height`` at/below ``SOH_HEIGHT_FLOOR``. This
    is a visual metaphor, not a measurement — a real cell is the same size
    whatever its SOH. Without it every cell draws at ``base_height``.
    """
    if not scale_by_soh:
        return base_height
    try:
        value = float(soh)
    except (TypeError, ValueError):
        value = float("nan")
    if value != value:
        return base_height * SOH_HEIGHT_MIN
    frac = (value - SOH_HEIGHT_FLOOR) / (100.0 - SOH_HEIGHT_FLOOR)
    return base_height * min(1.0, max(SOH_HEIGHT_MIN, frac))


def layout_positions(
    n_cells: int,
    topology: str,
    *,
    per_row: int = PARALLEL_ROW_WIDTH,
    spacing: float = CELL_SPACING,
) -> list[tuple[float, float]]:
    """``(x, y)`` centre of each cell on the pack floor, in selection order.

    Series: a single row along X — a string, in the order the cells are wired.
    Parallel: a block, ``per_row`` cells across and wrapping into further rows;
    the order carries no electrical meaning there, since the cells share the
    same two nodes. The block is centred on the origin either way.
    """
    if n_cells <= 0:
        return []
    if topology == "Series":
        cols = n_cells
    else:
        cols = min(n_cells, max(1, int(per_row)))
    rows = math.ceil(n_cells / cols)
    width = (cols - 1) * spacing
    depth = (rows - 1) * spacing
    positions: list[tuple[float, float]] = []
    for idx in range(n_cells):
        row, col = divmod(idx, cols)
        positions.append((col * spacing - width / 2.0, depth / 2.0 - row * spacing))
    return positions


def grouped_layout_positions(
    group_sizes: list[int],
    *,
    cell_spacing: float = CELL_SPACING,
    group_spacing: float = GROUP_SPACING,
) -> "tuple[list[list[tuple[float, float]]], list[tuple[float, float, float, float]]]":
    """Floor positions for an XpYs build, plus each group's footprint.

    Groups are strung along X — the series direction, so wiring runs the way the
    string reads — and the members of a group sit side by side along Y, because
    they share one node pair. Every group is centred on the X axis, which is
    what lets the series links between groups draw as straight runs.

    Returns ``(group_positions, group_bounds)``: the first is a list of the
    ``(x, y)`` centres per group, the second the ``(x0, x1, y0, y1)`` footprint
    of each group on the floor (used to outline the bottleneck group).
    """
    if not group_sizes:
        return [], []
    total_width = (len(group_sizes) - 1) * group_spacing
    group_positions: list[list[tuple[float, float]]] = []
    group_bounds: list[tuple[float, float, float, float]] = []
    for group_index, size in enumerate(group_sizes):
        x = group_index * group_spacing - total_width / 2.0
        ys = [(member - (size - 1) / 2.0) * cell_spacing for member in range(size)]
        group_positions.append([(x, y) for y in ys])
        y0 = (ys[0] if ys else 0.0) - CELL_RADIUS
        y1 = (ys[-1] if ys else 0.0) + CELL_RADIUS
        group_bounds.append((x - CELL_RADIUS, x + CELL_RADIUS, y0, y1))
    return group_positions, group_bounds


def cylinder_mesh(
    cx: float, cy: float, z0: float, z1: float,
    radius: float, segments: int = SEGMENTS,
) -> dict:
    """Vertices + triangle indices for one upright cylinder (mesh3d form).

    Two rings plus one centre vertex per cap: ``2 * segments + 2`` vertices and
    ``4 * segments`` triangles (a quad per side segment, plus the two caps).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring_x = cx + radius * np.cos(theta)
    ring_y = cy + radius * np.sin(theta)
    x = np.concatenate([ring_x, ring_x, [cx, cx]])
    y = np.concatenate([ring_y, ring_y, [cy, cy]])
    z = np.concatenate([
        np.full(segments, float(z0)), np.full(segments, float(z1)),
        [float(z0), float(z1)],
    ])

    face_i: list[int] = []
    face_j: list[int] = []
    face_k: list[int] = []
    bottom_centre = 2 * segments
    top_centre = 2 * segments + 1
    for s in range(segments):
        nxt = (s + 1) % segments
        lo_s, lo_n, up_s, up_n = s, nxt, s + segments, nxt + segments
        face_i += [lo_s, lo_s]
        face_j += [lo_n, up_n]
        face_k += [up_n, up_s]
        # Caps, wound so both normals face outward.
        face_i += [bottom_centre, top_centre]
        face_j += [lo_n, up_s]
        face_k += [lo_s, up_n]

    return {"x": x, "y": y, "z": z, "i": face_i, "j": face_j, "k": face_k}


def box_mesh(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> dict:
    """Vertices + triangle indices for an axis-aligned box (mesh3d form).

    8 vertices, 12 triangles, wound outward — used for the translucent outline
    around a bottleneck group.
    """
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    faces = [
        (0, 2, 1), (0, 3, 2),   # bottom  (−z)
        (4, 5, 6), (4, 6, 7),   # top     (+z)
        (0, 1, 5), (0, 5, 4),   # −y side
        (2, 3, 7), (2, 7, 6),   # +y side
        (1, 2, 6), (1, 6, 5),   # +x side
        (3, 0, 4), (3, 4, 7),   # −x side
    ]
    return {
        "x": x, "y": y, "z": z,
        "i": [f[0] for f in faces], "j": [f[1] for f in faces], "k": [f[2] for f in faces],
    }


def _merge_meshes(meshes: list[dict]) -> dict | None:
    """One mesh3d built from many shapes (used for the terminal caps, where a
    single colour is right and one trace beats N)."""
    if not meshes:
        return None
    x: list = []
    y: list = []
    z: list = []
    i: list = []
    j: list = []
    k: list = []
    offset = 0
    for mesh in meshes:
        x.extend(mesh["x"])
        y.extend(mesh["y"])
        z.extend(mesh["z"])
        i.extend(int(v) + offset for v in mesh["i"])
        j.extend(int(v) + offset for v in mesh["j"])
        k.extend(int(v) + offset for v in mesh["k"])
        offset += len(mesh["x"])
    return {"x": x, "y": y, "z": z, "i": i, "j": j, "k": k}


def _fmt(value, scale: float = 1.0, unit: str = "", decimals: int = 1) -> str:
    """One measurement for a hover label; an em dash when it isn't available."""
    try:
        num = float(value) * scale
    except (TypeError, ValueError):
        return "—"
    if num != num:
        return "—"
    rendered = f"{num:,.{decimals}f}"
    if not unit:
        return rendered
    return f"{rendered}%" if unit == "%" else f"{rendered} {unit}"


def _cell_hover(cell: dict, flag: "str | None" = None) -> str:
    """Hover text for one cell — the measurements, plus one optional flag line."""
    rul = cell.get("rul_pred")
    if rul is None:
        rul_note = "RUL withheld"
    elif cell.get("rul_reliable", False):
        rul_note = f"RUL {_fmt(rul, 1.0, 'cy', 0)}"
    else:
        rul_note = f"RUL {_fmt(rul, 1.0, 'cy', 0)} (not calibrated)"
    return (
        f"<b>{cell.get('cell_id', '?')}</b><br>{soh_band_label(cell.get('soh_pct'))}"
        f"<br>SOH {_fmt(cell.get('soh_pct'), 1.0, '%')}"
        f"<br>Capacity {_fmt(cell.get('capacity_ah'), 1000.0, 'mAh', 0)}"
        f"<br>Resistance {_fmt(cell.get('resistance_ohm'), 1000.0, 'mΩ')}"
        f"<br>{rul_note}"
        + (f"<br><b>{flag}</b>" if flag else "")
        + "<extra></extra>"
    )


def _add_cylinder(
    fig: go.Figure, cell: dict, cx: float, cy: float, height: float,
    flag: "str | None" = None, opacity: float = 1.0,
) -> dict:
    """Draw one cell cylinder; returns the terminal-cap mesh to merge later."""
    mesh = cylinder_mesh(cx, cy, 0.0, height, CELL_RADIUS)
    fig.add_trace(go.Mesh3d(
        x=mesh["x"], y=mesh["y"], z=mesh["z"],
        i=mesh["i"], j=mesh["j"], k=mesh["k"],
        color=soh_band_color(cell.get("soh_pct")),
        opacity=opacity,
        flatshading=False,
        name=str(cell.get("cell_id", "?")),
        showlegend=False,
        hovertemplate=_cell_hover(cell, flag),
    ))
    return cylinder_mesh(cx, cy, height, height + TERMINAL_HEIGHT, TERMINAL_RADIUS, 12)


def _add_terminals(fig: go.Figure, terminal_meshes: list[dict]) -> None:
    merged = _merge_meshes(terminal_meshes)
    if merged is not None:
        fig.add_trace(go.Mesh3d(
            **merged, color=_TERMINAL_COLOR, opacity=1.0, flatshading=False,
            name="Cell terminals (+)", showlegend=False, hoverinfo="skip",
        ))


def _add_floor(fig: go.Figure, xs: list[float], ys: list[float], floor_color: str) -> None:
    if not xs or not ys:
        return
    pad = CELL_RADIUS + FLOOR_MARGIN
    fig.add_trace(go.Surface(
        x=[[min(xs) - pad, max(xs) + pad], [min(xs) - pad, max(xs) + pad]],
        y=[[min(ys) - pad, min(ys) - pad], [max(ys) + pad, max(ys) + pad]],
        z=[[-0.02, -0.02], [-0.02, -0.02]],
        showscale=False, opacity=0.12, hoverinfo="skip",
        colorscale=[[0.0, floor_color], [1.0, floor_color]],
        name="Pack floor", showlegend=False,
    ))


def _add_halo(fig: go.Figure, cx: float, cy: float, height: float,
              cell_id: str, label_suffix: str, legend_name: str, hover_text: str) -> None:
    """Translucent cylinder + floating label marking one cell."""
    halo = cylinder_mesh(cx, cy, HALO_BOTTOM, height + HALO_TOP_PAD, HALO_RADIUS)
    fig.add_trace(go.Mesh3d(
        x=halo["x"], y=halo["y"], z=halo["z"],
        i=halo["i"], j=halo["j"], k=halo["k"],
        color=_BOTTLENECK_COLOR, opacity=0.16, flatshading=False,
        name=legend_name, showlegend=True,
        hovertemplate=f"{hover_text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[height + HALO_TOP_PAD + 0.35],
        mode="text",
        text=[f"{cell_id} — {label_suffix}"],
        textfont=dict(size=11, color=_BOTTLENECK_COLOR),
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))


def _add_band_legend(fig: go.Figure) -> None:
    for _threshold, color, label in SOH_BANDS:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=8, color=color),
            name=label, hoverinfo="skip",
        ))


def _apply_layout(fig: go.Figure, font_color: str, height: int = 470) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=0.98, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=font_color),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.45, y=-1.45, z=0.72)),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_pack_layout_figure(
    cells: list[dict],
    topology: str,
    *,
    bottleneck_id: "str | None" = None,
    most_loaded_id: "str | None" = None,
    groups: "list[list[int]] | None" = None,
    group_metrics: "list[dict] | None" = None,
    scale_height_by_soh: bool = True,
    show_interconnects: bool = True,
    per_row: int = PARALLEL_ROW_WIDTH,
    layout: "list[tuple[float, float]] | None" = None,
    font_color: str = "#a0aec0",
    floor_color: str = "#2d3748",
) -> go.Figure:
    """The 3D pack layout.

    ``cells`` is the Pack Builder's own ``cell_stats`` list (order = selection
    order): ``{"cell_id", "soh_pct", "capacity_ah", "resistance_ohm",
    "rul_pred", "rul_reliable"}``.

    Pass ``groups`` (index lists into ``cells``, from
    ``src/pack_builder.build_parallel_groups``) for an XpYs build: the scene is
    then drawn group-aware — groups strung along the series direction, each
    group's members side by side, ladders inside groups and series links
    between them. Without ``groups`` the flat Series/Parallel layout is drawn.
    ``most_loaded_id`` haloes the cell carrying the largest share of its group's
    current; ``bottleneck_id`` haloes the cell that gates the pack.
    """
    if groups:
        return _build_grouped_figure(
            cells, groups,
            group_metrics=group_metrics,
            most_loaded_id=most_loaded_id,
            scale_height_by_soh=scale_height_by_soh,
            show_interconnects=show_interconnects,
            font_color=font_color,
            floor_color=floor_color,
        )

    positions = layout if layout is not None else layout_positions(
        len(cells), topology, per_row=per_row,
    )
    heights = [cell_height(c.get("soh_pct", float("nan")), scale_by_soh=scale_height_by_soh)
               for c in cells]

    fig = go.Figure()
    terminals: list[dict] = []
    for (cx, cy), height, cell in zip(positions, heights, cells):
        is_bottleneck = bottleneck_id is not None and str(cell.get("cell_id", "?")) == bottleneck_id
        terminals.append(_add_cylinder(
            fig, cell, cx, cy, height,
            flag="Pack bottleneck" if is_bottleneck else None,
        ))

    _add_terminals(fig, terminals)

    # ── Bottleneck call-out: a translucent halo plus a label ────────────────
    for idx, cell in enumerate(cells):
        cell_id = str(cell.get("cell_id", "?"))
        if bottleneck_id is None or cell_id != bottleneck_id:
            continue
        cx, cy = positions[idx]
        _add_halo(
            fig, cx, cy, heights[idx], cell_id,
            f"bottleneck {_fmt(cell.get('soh_pct'), 1.0, '%')}",
            f"Bottleneck cell — {cell_id}",
            f"<b>{cell_id}</b> — pack bottleneck at {_fmt(cell.get('soh_pct'), 1.0, '%')} SOH",
        )

    # ── Wiring: which terminals are joined ──────────────────────────────────
    if show_interconnects and len(cells) >= 2:
        line_x: list = []
        line_y: list = []
        line_z: list = []
        if topology == "Series":
            # + of one cell to − of the next, in selection order: the series path.
            for idx in range(len(cells) - 1):
                (x0, y0), (x1, y1) = positions[idx], positions[idx + 1]
                line_x += [x0, x1, None]
                line_y += [y0, y1, None]
                line_z += [heights[idx], 0.0, None]
            interconnect_name = "Series interconnects (+ of one cell → − of the next)"
        else:
            # Shared nodes: one busbar across the tops of each row and one across
            # the bottoms, per row of the block.
            group = min(len(cells), max(1, int(per_row)))
            for start in range(0, len(cells), group):
                row = positions[start:start + group]
                row_heights = heights[start:start + group]
                if len(row) < 2:
                    continue
                line_x += [row[0][0], row[-1][0], None]
                line_y += [row[0][1], row[-1][1], None]
                line_z += [max(row_heights), max(row_heights), None]
                line_x += [row[0][0], row[-1][0], None]
                line_y += [row[0][1], row[-1][1], None]
                line_z += [0.0, 0.0, None]
            interconnect_name = "Parallel busbars (shared + and − nodes)"
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color=_INTERCONNECT_COLOR, width=3),
            name=interconnect_name,
            hoverinfo="skip",
        ))

    _add_floor(fig, [p[0] for p in positions], [p[1] for p in positions], floor_color)
    _add_band_legend(fig)
    return _apply_layout(fig, font_color)


def _build_grouped_figure(
    cells: list[dict],
    groups: list[list[int]],
    *,
    group_metrics: "list[dict] | None" = None,
    most_loaded_id: "str | None" = None,
    scale_height_by_soh: bool = True,
    show_interconnects: bool = True,
    font_color: str = "#a0aec0",
    floor_color: str = "#2d3748",
) -> go.Figure:
    """The XpYs scene: groups along the series direction, members sharing a node pair."""
    # Direct indexing, not a filtered comprehension: a bad index should raise
    # rather than silently drop a cell out from under the positions that follow.
    ordered_cells = [cells[i] for group in groups for i in group]
    group_positions, group_bounds = grouped_layout_positions([len(g) for g in groups])
    flat_positions = [p for group_pos in group_positions for p in group_pos]

    group_spans: list[tuple[int, int]] = []
    offset = 0
    for group in groups:
        group_spans.append((offset, offset + len(group)))
        offset += len(group)

    metrics_by_group = {
        int(row.get("group", idx + 1)): row for idx, row in enumerate(group_metrics or [])
    }
    bottleneck_group_index = None
    for idx, row in enumerate(group_metrics or []):
        if row.get("is_bottleneck"):
            bottleneck_group_index = idx

    heights = [cell_height(c.get("soh_pct", float("nan")), scale_by_soh=scale_height_by_soh)
               for c in ordered_cells]

    fig = go.Figure()
    terminals: list[dict] = []
    for (cx, cy), height, cell in zip(flat_positions, heights, ordered_cells):
        cell_id = str(cell.get("cell_id", "?"))
        is_loaded = most_loaded_id is not None and cell_id == most_loaded_id
        terminals.append(_add_cylinder(
            fig, cell, cx, cy, height,
            flag="Most loaded cell in its group" if is_loaded else None,
        ))
    _add_terminals(fig, terminals)

    # ── Bottleneck group: a translucent box around its footprint ────────────
    if bottleneck_group_index is not None and bottleneck_group_index < len(group_bounds):
        x0, x1, y0, y1 = group_bounds[bottleneck_group_index]
        first, last = group_spans[bottleneck_group_index]
        member_heights = heights[first:last]
        box = box_mesh(x0, x1, y0, y1, -0.05, (max(member_heights) if member_heights else CELL_HEIGHT) + BOX_TOP_PAD)
        row = metrics_by_group.get(bottleneck_group_index + 1, {})
        fig.add_trace(go.Mesh3d(
            x=box["x"], y=box["y"], z=box["z"],
            i=box["i"], j=box["j"], k=box["k"],
            color=_BOTTLENECK_COLOR, opacity=0.10, flatshading=False,
            name=f"Bottleneck group {bottleneck_group_index + 1} "
                 f"({_fmt(row.get('capacity_ah'), 1000.0, 'mAh', 0)})",
            showlegend=True,
            hovertemplate=(
                f"<b>Bottleneck group {bottleneck_group_index + 1}</b><br>"
                f"Capacity {_fmt(row.get('capacity_ah'), 1000.0, 'mAh', 0)} — the series string's "
                f"ceiling<br>Weakest member {row.get('weakest_cell_id', '—')}"
                "<extra></extra>"
            ),
        ))

    # ── Most-loaded cell: halo + label (resistance-based current sharing) ───
    if most_loaded_id is not None:
        for idx, cell in enumerate(ordered_cells):
            if str(cell.get("cell_id", "?")) != most_loaded_id:
                continue
            cx, cy = flat_positions[idx]
            ratio = float("nan")
            for row in group_metrics or []:
                if row.get("max_share_cell_id") == most_loaded_id:
                    ratio = float(row.get("overload_ratio", float("nan")))
            ratio_text = f"{ratio:.2f}× fair share" if ratio == ratio else "current sharing"
            _add_halo(
                fig, cx, cy, heights[idx], most_loaded_id, ratio_text,
                f"Most-loaded cell — {most_loaded_id} ({ratio_text})",
                f"<b>{most_loaded_id}</b> — carries the largest share of its group's current "
                f"({ratio_text}), because it has the group's lowest resistance",
            )

    # ── Wiring: a ladder inside each group, a series link between groups ────
    if show_interconnects:
        line_x: list = []
        line_y: list = []
        line_z: list = []
        group_tops: list[tuple[float, float]] = []
        for first, last in group_spans:
            members = flat_positions[first:last]
            member_heights = heights[first:last]
            if not members:
                continue
            top = max(member_heights)
            group_tops.append((members[0][0], top))
            if len(members) >= 2:
                # Shared + node (tops) and shared − node (bases) of the group.
                line_x += [members[0][0], members[-1][0], None]
                line_y += [members[0][1], members[-1][1], None]
                line_z += [top, top, None]
                line_x += [members[0][0], members[-1][0], None]
                line_y += [members[0][1], members[-1][1], None]
                line_z += [0.0, 0.0, None]
        # The string: + of one group to − of the next.
        for idx in range(len(group_tops) - 1):
            (x0, top0), (x1, _top1) = group_tops[idx], group_tops[idx + 1]
            line_x += [x0, x1, None]
            line_y += [0.0, 0.0, None]
            line_z += [top0, 0.0, None]
        if line_x:
            fig.add_trace(go.Scatter3d(
                x=line_x, y=line_y, z=line_z,
                mode="lines",
                line=dict(color=_INTERCONNECT_COLOR, width=3),
                name="Group ladders (+) and series links between groups",
                hoverinfo="skip",
            ))

    _add_floor(fig, [p[0] for p in flat_positions], [p[1] for p in flat_positions], floor_color)
    _add_band_legend(fig)
    return _apply_layout(fig, font_color, height=500)


# ---------------------------------------------------------------------------
# Streamlit section
# ---------------------------------------------------------------------------

def render_pack_layout_3d(
    cells: list[dict],
    topology: str,
    metrics: dict,
    *,
    key_prefix: str = "pack",
    groups: "list[list[int]] | None" = None,
    cells_per_group: "int | None" = None,
) -> None:
    """The Pack Builder's 3D layout section (called from render_pack_builder)."""
    # Imported here, not at module scope: _ui_helpers installs its backward-
    # compatible re-exports of _pack_builder at the end of its own module, and
    # _pack_builder imports this module — a module-level import would run
    # mid-cycle (same reason _pack_builder reads the metrics helpers lazily).
    from _ui_helpers import base_layout

    if len(cells) < 2:
        return

    st.markdown("<h4 class='section-header'>Pack layout (3D)</h4>", unsafe_allow_html=True)

    if groups:
        _render_grouped_captions(cells, groups, metrics, cells_per_group or 0)
    else:
        bottleneck_id = metrics.get("bottleneck_cell_id")
        net_word = "string" if topology == "Series" else "block"
        st.caption(
            f"One cylinder per selected cell, laid out as a {topology.lower()} {net_word} — "
            + ("wired in selection order, each cell's positive terminal joined to the next "
               "cell's negative." if topology == "Series"
               else "cells sharing the same two nodes, so their order carries no electrical "
                    "meaning.")
            + " Colour is the cell's own SOH (same 90/80% bands as the rest of the app); "
            "the halo marks the pack bottleneck. Drag to rotate."
        )

    ctrl_a, ctrl_b = st.columns(2)
    scale_by_soh = ctrl_a.checkbox(
        "Scale cell height by SOH (visual metaphor)",
        value=True, key=f"{key_prefix}_pack3d_height",
        help="Drawn height falls with SOH so the spread is readable at a glance. "
             "A real cell is the same size at 100% and 65% SOH — colour is the "
             "measurement here, height is only a visual cue.",
    )
    show_interconnects = ctrl_b.checkbox(
        "Show terminal interconnects",
        value=True, key=f"{key_prefix}_pack3d_wiring",
        help="Draws which terminals are joined — the series chain, the parallel bus "
             "nodes, or (in an XpYs build) each group's ladder and the links between "
             "groups. Schematic, not busbar sizing.",
    )

    base = base_layout()
    font_color = (base.get("font") or {}).get("color", "#a0aec0")
    floor_color = "#cbd5e0" if st.session_state.get("light_mode", False) else "#2d3748"

    fig = build_pack_layout_figure(
        cells, topology,
        bottleneck_id=metrics.get("bottleneck_cell_id") if not groups else None,
        most_loaded_id=metrics.get("most_loaded_cell_id") if groups else None,
        groups=groups,
        group_metrics=metrics.get("groups") if groups else None,
        scale_height_by_soh=scale_by_soh,
        show_interconnects=show_interconnects,
        font_color=font_color,
        floor_color=floor_color,
    )
    st.plotly_chart(fig, use_container_width=True)

    if scale_by_soh:
        st.caption(
            "Cell height is scaled by SOH as a visual metaphor only — the physical cells are "
            "identical in size. Colour is the measured SOH band."
        )

    if groups:
        _render_grouped_notes(cells, metrics)
        return

    bottleneck_id = metrics.get("bottleneck_cell_id")
    if bottleneck_id and metrics.get("spread_level") == "Imbalanced":
        st.caption(
            f"**{bottleneck_id}** is the highlighted cell: in a series pack its SOH is the "
            f"pack's usable-capacity ceiling (see {metrics.get('pack_soh_label', 'pack SOH')} "
            "above), and it will be the first cylinder to need replacing."
        )
    elif bottleneck_id:
        st.caption(
            f"**{bottleneck_id}** is highlighted as the selection's weakest cell "
            f"({metrics.get('pack_soh_label', 'pack SOH')} = "
            f"{_fmt(metrics.get('pack_soh'), 1.0, '%')})."
        )

    st.caption(
        "Schematic of the wiring, not a CAD model or a pack-design output: cells are drawn as "
        "identical cylinders although the sources' form factors differ (NASA 18650s, CALCE "
        "prismatics), and the interconnects are not sized for current."
    )


def _render_grouped_captions(cells: list[dict], groups: list[list[int]],
                             metrics: dict, cells_per_group: int) -> None:
    """Header caption for an XpYs build, including the ragged-last-group case."""
    sizes = [len(g) for g in groups]
    config = (
        f"{sizes[0]}p{len(groups)}s" if len(set(sizes)) == 1
        else f"{min(sizes)}–{max(sizes)}p{len(groups)}s"
    )
    st.caption(
        f"One cylinder per selected cell, laid out as an XpYs build — {config}: "
        f"{len(groups)} parallel groups strung in series, the cells in each group sharing a "
        "node pair (capped at "
        f"{cells_per_group} in parallel). Colour is the cell's own SOH (same 90/80% bands as "
        "the rest of the app). The red box marks the bottleneck group, the halo the "
        "most-loaded cell in its group. Drag to rotate."
    )
    if len(set(sizes)) > 1:
        st.caption(
            f"⚠ Group sizes are not equal ({', '.join(f'{s} cell(s)' for s in sizes)}): "
            f"{len(cells)} cells at {cells_per_group}p does not divide evenly, so the smaller "
            "group(s) carry proportionally less charge and will gate the string. Choose a "
            "cells-per-group value that divides the selection to avoid it."
        )


def _render_grouped_notes(cells: list[dict], metrics: dict) -> None:
    """The XpYs reading notes: what gated the string, and what sharing says."""
    rows = metrics.get("groups") or []
    bottleneck_group = metrics.get("bottleneck_group")
    if bottleneck_group:
        row = next((r for r in rows if r.get("group") == bottleneck_group), None)
        if row:
            st.caption(
                f"**Group {bottleneck_group}** (boxed) is the bottleneck: at "
                f"{_fmt(row.get('capacity_ah'), 1000.0, 'mAh', 0)} it has the least charge of "
                f"the {len(rows)} groups, and a series string's usable capacity is its weakest "
                f"group's. Its weakest member is **{row.get('weakest_cell_id')}** "
                f"({_fmt(row.get('soh_pct'), 1.0, '%')} group SOH)."
            )

    loaded = metrics.get("most_loaded_cell_id")
    if loaded:
        ratio = metrics.get("max_overload_ratio", float("nan"))
        st.caption(
            f"**{loaded}** (haloed) carries the largest share of current in its group — "
            + (f"**{ratio:.2f}× its fair share**, " if ratio == ratio else "")
            + "because cells in parallel divide current inversely with resistance. At a group "
            "current that would be 1C per cell in a perfectly balanced group, that cell sees "
            "the ratio above as a multiple of 1C."
        )
    elif metrics.get("current_sharing_available") is False:
        st.caption(
            "Current sharing is not reported for this build: "
            f"{metrics.get('current_sharing_reason') or 'a parallel group needs at least two cells with resistance data'}."
        )

    if metrics.get("binning_level") and rows and len(rows) > 1:
        st.caption(
            f"Binning verdict: **{metrics.get('binning_level')}** — group capacities span "
            f"{_fmt(metrics.get('group_capacity_spread_ah'), 1000.0, 'mAh', 0)} "
            f"({_fmt(metrics.get('group_capacity_spread_pct'), 1.0, '%')} of the mean). Groups "
            "are packed to balance capacity (largest cells first, each into the lightest group); "
            "the packing shown is the one this tool produced, not an optimal solution."
        )

    st.caption(
        "Current sharing is a first-order, resistance-only, steady-state model: cells in "
        "parallel sit at one voltage, so current divides inversely with DC resistance. It does "
        "not model SOC-dependent OCV differences, temperature feedback or contact resistance, "
        "and it does not project the differential aging this loading causes. Note the direction "
        "it does capture — a cell whose resistance has risen with age draws *less* here, which "
        "is self-limiting."
    )
    st.caption(
        "Schematic of the wiring, not a CAD model or a pack-design output: cells are drawn as "
        "identical cylinders although the sources' form factors differ (NASA 18650s, CALCE "
        "prismatics), and the interconnects are not sized for current."
    )
