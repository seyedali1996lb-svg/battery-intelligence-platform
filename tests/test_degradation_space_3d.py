"""Tests for Explore's 3D Degradation Space (app/_pages/_explore_3d.py).

Two layers, matching this repo's convention for a page-level feature:

- the pure builders (``axis_coverage``, ``build_cell_path``, ``z_at_cycle``,
  ``build_degradation_figure``) are tested directly on hand-built DataFrames --
  including the honesty properties the view depends on: unusable rows are
  *counted* rather than silently dropped, an axis a source does not carry is
  reported as absent instead of being estimated, and a knee marker outside the
  recorded range is refused rather than extrapolated;
- the page itself gets a Streamlit AppTest smoke run across data sources, the
  same pattern tests/test_app_state_combinations.py uses for its Explore tabs
  (nasa = 4 cells, below the "plot all" default; severson = 46 cells, above it).
"""

import os as _os
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401
import db as db_module
from streamlit.testing.v1 import AppTest

from _pages._explore_3d import (
    EOL_PCT,
    LEGEND_CELL_LIMIT,
    SOH_CBAR_MAX,
    SOH_CBAR_MIN,
    Z_AXIS_SPECS,
    axis_coverage,
    build_cell_path,
    build_degradation_figure,
    z_at_cycle,
)

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")

# What a source that carries no resistance column looks like (Zhu 2022's
# voltage-relaxation records store a protocol set-point temperature and no
# internal-resistance column at all -- confirmed against the real Parquet
# footers in data/cell_store/ when this view was written).
_ZHU_LIKE_COLUMNS = ["cycle_number", "soh_pct", "capacity_ah", "temperature_c", "fade_rate_30cy"]


def _cycles(n: int = 120, resistance: bool = True, temperature: float = 25.0) -> pd.DataFrame:
    cycle = np.arange(1, n + 1)
    soh = np.linspace(100.0, 70.0, n)
    data = {
        "cycle_number": cycle,
        "soh_pct": soh,
        "capacity_ah": 2.0 * soh / 100.0,
        "fade_rate_30cy": np.full(n, 0.004),
        "temperature_c": np.full(n, temperature),
    }
    if resistance:
        data["resistance_ohm"] = np.linspace(0.050, 0.080, n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# axis_coverage — what a source's records actually carry
# ---------------------------------------------------------------------------

def test_axis_coverage_counts_per_axis_in_preference_order():
    cols = [["cycle_number", "soh_pct", "resistance_ohm", "temperature_c"],
            ["cycle_number", "soh_pct", "resistance_ohm", "fade_rate_30cy"]]
    cov = axis_coverage(cols)
    assert list(cov) == list(Z_AXIS_SPECS), "coverage must preserve axis preference order"
    assert cov["resistance_ohm"] == 2
    assert cov["temperature_c"] == 1
    assert cov["fade_rate_30cy"] == 1


def test_axis_coverage_reports_zero_for_an_axis_no_record_carries():
    """Zhu-style records have no resistance column: the axis must read as
    absent (0 covered cells), which is what makes the view fall through to
    fade rate instead of plotting invented resistance values."""
    cov = axis_coverage([_ZHU_LIKE_COLUMNS, _ZHU_LIKE_COLUMNS])
    assert cov["resistance_ohm"] == 0
    assert cov["fade_rate_30cy"] == 2
    # The first *nonzero* axis is what the view defaults to.
    assert [c for c in Z_AXIS_SPECS if cov.get(c)][0] == "fade_rate_30cy"


# ---------------------------------------------------------------------------
# build_cell_path
# ---------------------------------------------------------------------------

def test_build_cell_path_returns_none_when_the_axis_column_is_absent():
    df = _cycles(resistance=False)
    assert build_cell_path(df, "resistance_ohm") is None


def test_build_cell_path_returns_none_for_empty_or_none_input():
    assert build_cell_path(None, "resistance_ohm") is None
    assert build_cell_path(pd.DataFrame(), "resistance_ohm") is None


def test_build_cell_path_scales_the_axis_into_display_units():
    path = build_cell_path(_cycles(), "resistance_ohm", scale=1000.0, positive_only=True)
    assert path is not None
    assert path["first_z"] == pytest.approx(50.0)
    assert path["last_z"] == pytest.approx(80.0)


def test_build_cell_path_sets_aside_non_positive_readings_and_counts_them():
    """A 0 Ω reading is a sentinel/parse artefact, not a measurement. It must
    be excluded AND counted, so the view can disclose it rather than drawing a
    tidy path that quietly lost rows."""
    df = _cycles(n=6)
    df.loc[2, "resistance_ohm"] = 0.0
    df.loc[3, "resistance_ohm"] = -0.01
    path = build_cell_path(df, "resistance_ohm", scale=1000.0, positive_only=True)
    assert path is not None
    assert path["n_plotted"] == 4
    assert path["n_dropped"] == 2
    assert path["n_series"] == 6
    assert np.all(np.asarray(path["z"], dtype=float) > 0)


def test_build_cell_path_keeps_non_positive_readings_when_the_axis_allows_them():
    """SOH and fade rate legitimately flatten or dip -- only axes declared
    positive_only get the artefact filter."""
    df = _cycles(n=5)
    df.loc[2, "fade_rate_30cy"] = 0.0
    df.loc[3, "fade_rate_30cy"] = -0.0005
    path = build_cell_path(df, "fade_rate_30cy", scale=1000.0, positive_only=False)
    assert path is not None
    assert path["n_dropped"] == 0
    assert path["n_plotted"] == 5


def test_build_cell_path_counts_rows_with_a_missing_coordinate():
    df = _cycles(n=10)
    df.loc[4, "soh_pct"] = np.nan
    df.loc[5, "resistance_ohm"] = np.nan
    path = build_cell_path(df, "resistance_ohm", scale=1000.0, positive_only=True)
    assert path is not None
    assert path["n_plotted"] == 8
    assert path["n_dropped"] == 2


def test_build_cell_path_downsamples_but_always_keeps_the_last_cycle():
    path = build_cell_path(_cycles(n=1200), "resistance_ohm", max_points=100)
    assert path is not None
    assert path["n_plotted"] == 1200            # the summary count is the real one
    assert len(path["cycle_number"]) <= 105     # the drawn path is bounded
    assert path["last_cycle"] == pytest.approx(1200.0)
    assert path["cycle_number"][-1] == pytest.approx(1200.0)


def test_build_cell_path_reports_first_and_latest_soh():
    path = build_cell_path(_cycles(n=50), "resistance_ohm")
    assert path is not None
    assert path["first_soh"] == pytest.approx(100.0)
    assert path["latest_soh"] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# z_at_cycle — knee placement without inventing values
# ---------------------------------------------------------------------------

def test_z_at_cycle_interpolates_between_recorded_points():
    path = build_cell_path(_cycles(n=101), "resistance_ohm", scale=1000.0)
    assert path is not None
    # Cycles are 1..101, resistance rises linearly 50 -> 80 mΩ, so cycle 51 is
    # the midpoint.
    assert z_at_cycle(path, 51.0) == pytest.approx(65.0, abs=0.5)


def test_z_at_cycle_refuses_a_cycle_outside_the_recorded_range():
    """A knee marker outside the recorded window cannot be read off the path;
    returning None keeps the view from extrapolating one."""
    path = build_cell_path(_cycles(n=40), "resistance_ohm")
    assert path is not None
    assert z_at_cycle(path, 900.0) is None
    assert z_at_cycle(path, 0.5) is None


def test_z_at_cycle_needs_at_least_two_points():
    single = {"cycle_number": np.array([5.0]), "z": np.array([50.0])}
    assert z_at_cycle(single, 5.0) is None


# ---------------------------------------------------------------------------
# build_degradation_figure
# ---------------------------------------------------------------------------

def _paths(n_cells: int = 3) -> dict:
    return {
        f"Cell{i}": build_cell_path(_cycles(n=60 + i), "resistance_ohm", scale=1000.0)
        for i in range(n_cells)
    }


def test_figure_has_one_path_per_cell_plus_the_eol_plane():
    fig = build_degradation_figure(_paths(4), "resistance_ohm")
    types = [t.type for t in fig.data]
    assert types.count("scatter3d") == 4
    assert types[-1] == "surface"
    plane = fig.data[-1]
    assert list(plane.y[0]) == [EOL_PCT, EOL_PCT], "the plane must sit at the 80% EOL convention"
    # …and that convention is the platform's own token, not a number this page
    # happens to hold. Both views of end-of-life are one number.
    from _design_tokens import SOH_EOL_MIN

    assert EOL_PCT == SOH_EOL_MIN


def test_figure_axes_are_cycle_age_soh_and_the_selected_axis():
    fig = build_degradation_figure(_paths(2), "fade_rate_30cy")
    assert fig.layout.scene.xaxis.title.text == "Cycle age"
    assert fig.layout.scene.yaxis.title.text == "SOH %"
    assert fig.layout.scene.zaxis.title.text == Z_AXIS_SPECS["fade_rate_30cy"]["label"]


def test_figure_colors_markers_by_soh_on_a_pinned_range():
    """The ramp is pinned, not fitted to the data drawn, so equal SOH means
    equal colour across cells and fleets."""
    fig = build_degradation_figure(_paths(2), "resistance_ohm")
    for trace in fig.data:
        if trace.type != "scatter3d":
            continue
        assert trace.marker.cmin == SOH_CBAR_MIN
        assert trace.marker.cmax == SOH_CBAR_MAX
        assert list(trace.marker.color) == pytest.approx(list(trace.y))


def test_figure_shows_one_colorbar_and_a_per_cell_legend_when_few_cells():
    fig = build_degradation_figure(_paths(LEGEND_CELL_LIMIT), "resistance_ohm")
    scatter = [t for t in fig.data if t.type == "scatter3d"]
    assert sum(1 for t in scatter if t.marker.showscale) == 1
    assert all(t.showlegend for t in scatter)
    assert len({t.line.color for t in scatter}) == LEGEND_CELL_LIMIT


def test_figure_drops_the_legend_and_shares_one_line_colour_past_the_cell_limit():
    fig = build_degradation_figure(_paths(LEGEND_CELL_LIMIT + 2), "resistance_ohm")
    scatter = [t for t in fig.data if t.type == "scatter3d"]
    assert not any(t.showlegend for t in scatter)
    assert len({t.line.color for t in scatter}) == 1, (
        "past the legend limit a per-cell palette is unreadable — hover carries the cell id"
    )


def test_figure_adds_a_knee_trace_only_when_knee_points_are_given():
    without = build_degradation_figure(_paths(2), "resistance_ohm")
    assert all(t.name != "Knee point (1)" for t in without.data)

    with_knees = build_degradation_figure(
        _paths(2), "resistance_ohm",
        knee_points=[("Cell0", 30.0, 90.0, 55.0), ("Cell1", 40.0, 88.0, 60.0)],
    )
    knee_traces = [t for t in with_knees.data if t.type == "scatter3d" and "Knee point" in str(t.name)]
    assert len(knee_traces) == 1
    assert knee_traces[0].name == "Knee point (2)"
    assert list(knee_traces[0].x) == [30.0, 40.0]


# ---------------------------------------------------------------------------
# Page-level AppTest smoke (matches test_app_state_combinations.py's pattern)
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


def _explore_app(data_mode: str, **extra_state) -> AppTest:
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
    at.session_state["data_mode"] = data_mode
    at.session_state["explore_view_radio"] = "3D Degradation Space"
    for k, v in extra_state.items():
        at.session_state[k] = v
    return at


@pytest.mark.parametrize("data_mode", ["nasa", "severson"])
def test_degradation_space_renders_without_exception(isolated_db, data_mode):
    """nasa (4 cells) exercises the plot-all default and the knee-marker path;
    severson (46 cells) exercises the multiselect branch and the fleet-scale
    pruned-read path."""
    at = _explore_app(data_mode)
    at.run()
    assert not at.exception, f"3D Degradation Space crashed in {data_mode} mode: {at.exception}"
    text = "\n".join(m.value for m in at.markdown)
    assert "Fleet degradation space" in text, f"3D view didn't render in {data_mode} mode"
    assert "No cells in the active fleet" not in text


def test_degradation_space_plots_an_explicit_cell_subset(isolated_db):
    at = _explore_app("nasa", explore3d_plot_all=False, explore3d_cells=["B0005", "B0006"])
    at.run()
    assert not at.exception, f"3D view crashed with an explicit subset: {at.exception}"
    text = "\n".join(m.value for m in at.markdown)
    assert "Fleet degradation space" in text


def test_degradation_space_empty_selection_shows_an_empty_state(isolated_db):
    at = _explore_app("nasa", explore3d_plot_all=False, explore3d_cells=[])
    at.run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "Select at least one cell" in text
