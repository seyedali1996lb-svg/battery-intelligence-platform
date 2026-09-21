"""Tests for the Virtual Pack Builder's 3D layout view (app/_pack_layout_3d.py).

The pure builders are the interesting part here: the geometry has to be
measurable (a cylinder is a real cylinder: 2n+2 vertices, 4n triangles), the
layout has to encode the topology (a series string is a row in selection order,
a parallel block wraps), and the honesty properties have to hold — the height
metaphor must be disclosed-as-a-metaphor (bounded and off with one toggle), the
bottleneck must be identifiable, and an uncalibrated RUL must say so.

One Streamlit AppTest at the end runs the section end-to-end inside Explore's
Pack Builder tab.
"""

import os as _os
import sys
import pathlib

import numpy as np
import pytest

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401
import db as db_module
from streamlit.testing.v1 import AppTest

from _pack_layout_3d import (
    CELL_HEIGHT,
    CELL_RADIUS,
    CELL_SPACING,
    GROUP_SPACING,
    SEGMENTS,
    SOH_BANDS,
    SOH_HEIGHT_FLOOR,
    SOH_HEIGHT_MIN,
    box_mesh,
    build_pack_layout_figure,
    cell_height,
    cylinder_mesh,
    grouped_layout_positions,
    layout_positions,
    soh_band_color,
    soh_band_label,
)
from _ui_helpers import soh_status
from pack_builder import build_parallel_groups, compute_xpys_metrics

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


def _cell(cell_id, soh, cap=2.0, res=0.05, rul=None, rul_ok=False):
    return {
        "cell_id": cell_id, "soh_pct": soh, "capacity_ah": cap,
        "resistance_ohm": res, "rul_pred": rul, "rul_reliable": rul_ok,
    }


def _mesh_traces(fig):
    return [t for t in fig.data if t.type == "mesh3d"]


def _line_traces(fig):
    return [t for t in fig.data if t.type == "scatter3d" and t.mode == "lines"]


def _text_traces(fig):
    return [t for t in fig.data if t.type == "scatter3d" and t.mode == "text"]


# ---------------------------------------------------------------------------
# cylinder_mesh — the geometry is real, not decorative
# ---------------------------------------------------------------------------

def test_cylinder_mesh_vertex_and_face_counts():
    mesh = cylinder_mesh(0.0, 0.0, 0.0, 3.0, CELL_RADIUS, segments=SEGMENTS)
    assert len(mesh["x"]) == len(mesh["y"]) == len(mesh["z"]) == 2 * SEGMENTS + 2
    assert len(mesh["i"]) == len(mesh["j"]) == len(mesh["k"]) == 4 * SEGMENTS


def test_cylinder_mesh_indices_stay_inside_the_vertex_array():
    mesh = cylinder_mesh(0.0, 0.0, 0.0, 3.0, CELL_RADIUS, segments=16)
    n_vertices = len(mesh["x"])
    assert min(mesh["i"] + mesh["j"] + mesh["k"]) >= 0
    assert max(mesh["i"] + mesh["j"] + mesh["k"]) < n_vertices


def test_cylinder_mesh_has_two_rings_and_two_cap_centres():
    mesh = cylinder_mesh(1.0, -2.0, 0.0, 2.5, CELL_RADIUS, segments=12)
    z = np.asarray(mesh["z"], dtype=float)
    assert set(np.round(z[:12], 6)) == {0.0}          # lower ring
    assert set(np.round(z[12:24], 6)) == {2.5}        # upper ring
    assert z[24] == pytest.approx(0.0)                # bottom cap centre
    assert z[25] == pytest.approx(2.5)                # top cap centre


def test_cylinder_mesh_ring_vertices_sit_on_the_requested_circle():
    mesh = cylinder_mesh(cx=3.0, cy=-1.0, z0=0.0, z1=1.0, radius=0.5, segments=24)
    r = np.hypot(np.asarray(mesh["x"][:24]) - 3.0, np.asarray(mesh["y"][:24]) + 1.0)
    assert np.allclose(r, 0.5)


# ---------------------------------------------------------------------------
# layout_positions — topology as geometry
# ---------------------------------------------------------------------------

def test_series_layout_is_a_single_row_in_selection_order():
    positions = layout_positions(4, "Series")
    assert len(positions) == 4
    xs = [p[0] for p in positions]
    assert all(p[1] == pytest.approx(0.0) for p in positions), "a series string is one row"
    assert xs == sorted(xs), "position order must follow selection order (the wiring order)"
    assert np.allclose(np.diff(xs), CELL_SPACING)
    assert sum(xs) == pytest.approx(0.0), "the string is centred on the origin"


def test_parallel_layout_wraps_into_a_block_and_stays_centred():
    positions = layout_positions(8, "Parallel", per_row=4)
    ys = sorted({round(p[1], 6) for p in positions})
    assert len(ys) == 2, "8 cells at 4 per row is a 2-row block"
    assert sum(p[0] for p in positions) == pytest.approx(0.0)
    assert sum(p[1] for p in positions) == pytest.approx(0.0)
    # Four columns per row, spaced evenly.
    row0 = [p[0] for p in positions[:4]]
    assert np.allclose(np.diff(sorted(row0)), CELL_SPACING)


def test_parallel_layout_does_not_wrap_a_short_block():
    positions = layout_positions(3, "Parallel", per_row=4)
    assert len(positions) == 3
    assert all(p[1] == pytest.approx(0.0) for p in positions)


def test_layout_positions_empty_for_no_cells():
    assert layout_positions(0, "Series") == []
    assert layout_positions(0, "Parallel") == []


# ---------------------------------------------------------------------------
# cell_height / soh bands — the parts that must stay honest
# ---------------------------------------------------------------------------

def test_cell_height_metaphor_is_monotonic_and_bounded():
    heights = [cell_height(s, scale_by_soh=True) for s in (100, 90, 80, 70, 60)]
    assert heights == sorted(heights, reverse=True), "lower SOH must never draw taller"
    assert heights[0] == pytest.approx(CELL_HEIGHT)
    assert all(SOH_HEIGHT_MIN * CELL_HEIGHT <= h <= CELL_HEIGHT for h in heights)


def test_cell_height_metaphor_bottoms_out_instead_of_vanishing():
    assert cell_height(SOH_HEIGHT_FLOOR, scale_by_soh=True) == pytest.approx(SOH_HEIGHT_MIN * CELL_HEIGHT)
    assert cell_height(0.0, scale_by_soh=True) == pytest.approx(SOH_HEIGHT_MIN * CELL_HEIGHT)


def test_cell_height_is_identical_for_every_cell_when_the_metaphor_is_off():
    assert {cell_height(s, scale_by_soh=False) for s in (100, 70, 55)} == {CELL_HEIGHT}


def test_cell_height_tolerates_a_missing_soh():
    assert cell_height(float("nan"), scale_by_soh=True) == pytest.approx(SOH_HEIGHT_MIN * CELL_HEIGHT)
    assert cell_height(None, scale_by_soh=True) == pytest.approx(SOH_HEIGHT_MIN * CELL_HEIGHT)


@pytest.mark.parametrize("soh,expected", [(95.0, "#68d391"), (85.0, "#f6ad55"), (70.0, "#fc8181")])
def test_soh_band_color_uses_this_apps_own_bands(soh, expected):
    assert soh_band_color(soh) == expected


@pytest.mark.parametrize("soh", [95.0, 90.0, 85.0, 80.0, 70.0, 55.0])
def test_soh_band_color_agrees_with_soh_status_on_the_band_boundaries(soh):
    """The 3D view must never paint a cell green while the rest of the app calls
    it degrading — both read the same 90/80% thresholds."""
    status, _css_class = soh_status(soh)
    label = soh_band_label(soh)
    assert status.split()[0].lower() in label.lower() or (
        status == "End of Life" and label.startswith("End of Life")
    )


def test_soh_band_color_and_label_handle_a_missing_soh():
    assert soh_band_color(float("nan")) == SOH_BANDS[-1][1]
    assert soh_band_label(None) == "SOH unavailable"


# ---------------------------------------------------------------------------
# build_pack_layout_figure
# ---------------------------------------------------------------------------

def test_figure_draws_a_cylinder_per_cell_plus_a_merged_terminal_trace():
    cells = [_cell("A", 95.0), _cell("B", 85.0), _cell("C", 70.0)]
    fig = build_pack_layout_figure(cells, "Series")
    meshes = _mesh_traces(fig)
    # One per cell, plus the merged terminal caps (no bottleneck halo: none given).
    assert len(meshes) == len(cells) + 1
    assert [t.name for t in meshes[:3]] == ["A", "B", "C"]
    assert meshes[-1].name == "Cell terminals (+)"


def test_figure_cylinder_colours_follow_the_soh_bands():
    cells = [_cell("A", 95.0), _cell("B", 85.0), _cell("C", 70.0)]
    fig = build_pack_layout_figure(cells, "Series")
    assert [t.color for t in _mesh_traces(fig)[:3]] == ["#68d391", "#f6ad55", "#fc8181"]


def test_figure_marks_the_bottleneck_with_a_halo_and_a_label():
    cells = [_cell("A", 94.0), _cell("B", 71.0), _cell("C", 88.0)]
    fig = build_pack_layout_figure(cells, "Series", bottleneck_id="B")
    halo = [t for t in _mesh_traces(fig) if "Bottleneck cell" in str(t.name)]
    assert len(halo) == 1
    assert "B" in str(halo[0].name)
    assert halo[0].opacity < 0.3, "the halo must read as a call-out, not a repainted cell"
    labels = _text_traces(fig)
    assert len(labels) == 1
    assert "B" in labels[0].text[0] and "bottleneck" in labels[0].text[0]


def test_figure_omits_the_halo_when_no_bottleneck_is_named():
    fig = build_pack_layout_figure([_cell("A", 94.0), _cell("B", 71.0)], "Series")
    assert not [t for t in _mesh_traces(fig) if "Bottleneck" in str(t.name)]
    assert not _text_traces(fig)


def test_figure_series_wiring_joins_each_cell_to_the_next():
    cells = [_cell("A", 95.0), _cell("B", 88.0), _cell("C", 80.0)]
    fig = build_pack_layout_figure(cells, "Series")
    lines = _line_traces(fig)
    assert len(lines) == 1
    assert "Series" in str(lines[0].name)
    xs = list(lines[0].x)
    zs = list(lines[0].z)
    assert len(xs) == 3 * (len(cells) - 1), "one 3-point gap-separated run between each pair"
    assert None in xs and None in zs
    assert zs[0] == pytest.approx(cell_height(95.0, scale_by_soh=True))
    assert zs[1] == pytest.approx(0.0), "+ of one cell runs to − (the base) of the next"


def test_figure_parallel_wiring_is_two_busbars_per_row():
    fig = build_pack_layout_figure([_cell(c, 90.0) for c in "ABCD"], "Parallel", per_row=4)
    lines = _line_traces(fig)
    assert len(lines) == 1
    assert "Parallel busbars" in str(lines[0].name)
    assert list(lines[0].z).count(None) == 2, "one shared top node and one shared bottom node"
    drawn = sorted(z for z in lines[0].z if z is not None)
    assert drawn[0] == pytest.approx(0.0), "the shared − node is the pack floor"
    assert drawn[-1] == pytest.approx(cell_height(90.0, scale_by_soh=True)), (
        "the shared + node sits on the tallest cylinder in the row, not at nominal height"
    )


def test_figure_parallel_wiring_covers_every_row_of_a_wrapped_block():
    fig = build_pack_layout_figure([_cell(c, 90.0) for c in "ABCDEF"], "Parallel", per_row=4)
    lines = _line_traces(fig)
    # Row 1 has four cells (two busbars); row 2 has two cells (two busbars).
    assert list(lines[0].z).count(None) == 4


def test_figure_can_draw_without_wiring_or_the_height_metaphor():
    cells = [_cell("A", 95.0), _cell("B", 70.0)]
    fig = build_pack_layout_figure(
        cells, "Series", scale_height_by_soh=False, show_interconnects=False,
    )
    assert not _line_traces(fig)
    tops = [float(np.max(t.z)) for t in _mesh_traces(fig)[:2]]
    assert tops[0] == pytest.approx(tops[1]) == pytest.approx(CELL_HEIGHT)


def test_figure_height_metaphor_is_reflected_in_the_drawn_cylinders():
    cells = [_cell("A", 100.0), _cell("B", 60.0)]
    fig = build_pack_layout_figure(cells, "Series", scale_height_by_soh=True)
    healthy, degraded = _mesh_traces(fig)[:2]
    assert np.max(healthy.z) > np.max(degraded.z)


def test_figure_legend_carries_the_three_soh_bands():
    fig = build_pack_layout_figure([_cell("A", 95.0), _cell("B", 70.0)], "Series")
    legend_names = [t.name for t in fig.data if t.type == "scatter3d" and t.mode == "markers"]
    assert len(legend_names) == len(SOH_BANDS)
    assert any("Healthy" in n for n in legend_names)
    assert any("Degrading" in n for n in legend_names)
    assert any("End of Life" in n for n in legend_names)


def test_figure_hover_discloses_an_uncalibrated_rul_and_a_withheld_one():
    cells = [_cell("A", 95.0, rul=120.0, rul_ok=False), _cell("B", 80.0, rul=None)]
    fig = build_pack_layout_figure(cells, "Series")
    hovers = [t.hovertemplate for t in _mesh_traces(fig)[:2]]
    assert "not calibrated" in hovers[0]
    assert "RUL withheld" in hovers[1]


def test_figure_hover_does_not_invent_missing_measurements():
    cells = [_cell("A", 95.0, cap=float("nan"), res=float("nan"))]
    fig = build_pack_layout_figure(cells, "Parallel")
    hover = _mesh_traces(fig)[0].hovertemplate
    assert "Capacity —" in hover and "Resistance —" in hover


def test_figure_serializes_for_the_browser():
    cells = [_cell("A", 95.0), _cell("B", 71.0), _cell("C", 88.0)]
    fig = build_pack_layout_figure(cells, "Parallel", bottleneck_id="B")
    assert fig.to_json()


# ---------------------------------------------------------------------------
# XpYs: grouped geometry
# ---------------------------------------------------------------------------

def test_box_mesh_is_a_closed_box():
    mesh = box_mesh(-1.0, 1.0, -0.5, 0.5, 0.0, 2.0)
    assert len(mesh["x"]) == len(mesh["y"]) == len(mesh["z"]) == 8
    assert len(mesh["i"]) == len(mesh["j"]) == len(mesh["k"]) == 12
    assert max(mesh["i"] + mesh["j"] + mesh["k"]) < 8
    assert min(mesh["x"]) == -1.0 and max(mesh["z"]) == 2.0


def test_grouped_layout_strands_groups_along_the_series_axis():
    group_positions, bounds = grouped_layout_positions([2, 2, 2])
    assert len(group_positions) == 3 and len(bounds) == 3
    xs = [group[0][0] for group in group_positions]
    assert xs == sorted(xs), "group order is the series order"
    assert all(np.diff(xs) == pytest.approx(GROUP_SPACING) for _ in [0])
    assert abs(sum(xs)) < 1e-9, "the string is centred on the origin"


def test_members_of_a_group_are_side_by_side_on_the_shared_node_axis():
    group_positions, _bounds = grouped_layout_positions([3, 1])
    assert len(group_positions[0]) == 3 and len(group_positions[1]) == 1
    ys = [p[1] for p in group_positions[0]]
    assert ys == sorted(ys)
    assert np.allclose(np.diff(ys), CELL_SPACING)
    assert sum(ys) == pytest.approx(0.0), "each group is centred on the X axis"


def test_group_bounds_contain_their_members_and_are_cell_sized_across():
    group_positions, bounds = grouped_layout_positions([2, 3])
    for group, (x0, x1, y0, y1) in zip(group_positions, bounds):
        assert x1 - x0 == pytest.approx(2 * CELL_RADIUS)
        for x, y in group:
            assert x0 <= x <= x1 and y0 <= y <= y1


def test_grouped_layout_handles_empty_input():
    assert grouped_layout_positions([]) == ([], [])


# ---------------------------------------------------------------------------
# XpYs: the grouped figure
# ---------------------------------------------------------------------------

def _xpys_cells():
    return [
        _cell("A", 92.0, 2.00, 0.050),
        _cell("B", 88.0, 1.90, 0.055),
        _cell("C", 80.0, 1.70, 0.070),
        _cell("D", 95.0, 2.05, 0.048),
    ]


def _grouped_figure(**kwargs):
    cells = _xpys_cells()
    groups = build_parallel_groups(cells, 2)
    metrics = compute_xpys_metrics(cells, groups)
    fig = build_pack_layout_figure(
        cells, "XpYs", groups=groups, group_metrics=metrics["groups"],
        most_loaded_id=metrics["most_loaded_cell_id"], **kwargs,
    )
    return fig, groups, metrics


def test_grouped_figure_draws_cells_in_group_order():
    fig, groups, _metrics = _grouped_figure()
    names = [t.name for t in _mesh_traces(fig)[:4]]
    cells_by_index = _xpys_cells()
    assert names == [cells_by_index[i]["cell_id"] for group in groups for i in group]
    assert _mesh_traces(fig)[4].name == "Cell terminals (+)", (
        "the merged terminal trace still follows the cells"
    )


def test_grouped_figure_outlines_the_bottleneck_group():
    fig, groups, metrics = _grouped_figure()
    boxes = [t for t in _mesh_traces(fig) if str(t.name).startswith("Bottleneck group")]
    assert len(boxes) == 1
    assert f"Bottleneck group {metrics['bottleneck_group']}" in str(boxes[0].name)
    assert boxes[0].opacity < 0.2, "the group outline must read as a call-out"


def test_grouped_figure_includes_group_ladders_and_series_links():
    fig, groups, _metrics = _grouped_figure()
    lines = _line_traces(fig)
    assert len(lines) == 1
    assert "Group ladders" in str(lines[0].name)
    # Per group: a top and a bottom busbar (2 runs) -> and one link between the
    # two groups (1 run). Each run is 3 points with a None separator.
    separators = [v for v in lines[0].x if v is None]
    assert len(separators) == 2 * len(groups) + (len(groups) - 1)


def test_grouped_figure_can_drop_the_wiring():
    fig, _groups, _metrics = _grouped_figure(show_interconnects=False)
    assert not _line_traces(fig)


def test_grouped_figure_haloes_the_most_loaded_cell_and_labels_its_load_ratio():
    fig, _groups, metrics = _grouped_figure()
    haloes = [t for t in _mesh_traces(fig) if "Most-loaded cell" in str(t.name)]
    assert len(haloes) == 1
    assert metrics["most_loaded_cell_id"] in str(haloes[0].name)
    labels = _text_traces(fig)
    assert len(labels) == 1
    assert "fair share" in labels[0].text[0]


def test_grouped_figure_does_not_halo_anything_when_sharing_is_unavailable():
    cells = [_cell("A", 92.0, 2.00, float("nan")), _cell("B", 88.0, 1.90, float("nan"))]
    groups = build_parallel_groups(cells, 2)
    metrics = compute_xpys_metrics(cells, groups)
    fig = build_pack_layout_figure(
        cells, "XpYs", groups=groups, group_metrics=metrics["groups"],
        most_loaded_id=metrics["most_loaded_cell_id"],
    )
    assert metrics["most_loaded_cell_id"] is None
    assert not [t for t in _mesh_traces(fig) if "Most-loaded" in str(t.name)]
    assert not _text_traces(fig)


def test_grouped_figure_still_carries_the_soh_band_legend_and_floor():
    fig, _groups, _metrics = _grouped_figure()
    legend_names = [t.name for t in fig.data if t.type == "scatter3d" and t.mode == "markers"]
    assert any("Healthy" in n for n in legend_names)
    assert any(t.type == "surface" for t in fig.data)


def test_grouped_figure_serializes_for_the_browser():
    fig, _groups, _metrics = _grouped_figure()
    assert fig.to_json()


# ---------------------------------------------------------------------------
# Page-level smoke: the section renders inside Explore's Pack Builder tab
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY=")
    monkeypatch.setattr(db_module, "_fernet", None)
    db_module.init_db()
    return db_module


def _pack_builder_app(topology: str = "Series") -> AppTest:
    at = AppTest.from_file(_MAIN_PY, default_timeout=180)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = "Engineer"
    at.session_state["page"] = "compare"
    at.session_state["data_mode"] = "nasa"
    at.session_state["explore_view_radio"] = "Pack Builder"
    at.session_state["explore_pack_cells"] = ["B0005", "B0006"]
    at.session_state["explore_pack_topology"] = topology
    return at


@pytest.mark.parametrize("topology", ["Series", "Parallel"])
def test_pack_layout_renders_in_the_pack_builder(isolated_db, topology):
    at = _pack_builder_app(topology)
    at.run()
    assert not at.exception, f"Pack Builder crashed rendering the 3D layout ({topology}): {at.exception}"
    text = "\n".join(m.value for m in at.markdown)
    assert "Pack layout (3D)" in text, "the 3D layout section didn't render"
    captions = "\n".join(c.value for c in at.caption)
    assert "visual metaphor" in captions.lower() or "SOH" in captions


@pytest.mark.parametrize("cells_per_group", [2, 3, 4])
def test_xpys_pack_layout_renders_end_to_end(isolated_db, cells_per_group):
    """4 NASA cells at 2p is two series groups, at 4p a single parallel block,
    and at 3p a ragged 3+1 build — the shapes the XpYs path has to handle in
    the real app, including the uneven one."""
    at = AppTest.from_file(_MAIN_PY, default_timeout=180)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = "Engineer"
    at.session_state["page"] = "compare"
    at.session_state["data_mode"] = "nasa"
    at.session_state["explore_view_radio"] = "Pack Builder"
    at.session_state["explore_pack_cells"] = ["B0005", "B0006", "B0007", "B0018"]
    at.session_state["explore_pack_topology"] = "XpYs"
    at.session_state["explore_pack_cells_per_group"] = cells_per_group
    at.run()
    assert not at.exception, f"XpYs Pack Builder crashed at {cells_per_group}p: {at.exception}"
    text = "\n".join(m.value for m in at.markdown)
    captions = "\n".join(c.value for c in at.caption)
    assert "Pack layout (3D)" in text
    assert "current sharing" in (text + captions).lower(), (
        "the XpYs build must report how current divides inside a group"
    )
    assert "Parallel groups & current sharing" in text
