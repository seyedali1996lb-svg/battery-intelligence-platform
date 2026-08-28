"""AppTest page-level checks for the CellTwin physics-projection chart on
the Cell Workbench Health view (app/_pages/health.py's 12-Month Forecast
section): the projection chart (measured history + central + 2-sigma band)
renders beside the GBRT-fade forecast for a real NASA cell, the GBRT-vs-
physics RUL comparison strip appears, and the honest empty state renders
(without crashing) when no projection is available.

The twin math itself is covered in tests/test_digital_twin.py; this file's
job is UI-wiring correctness only, same split as the other AppTest files.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
import db as db_module
from streamlit.testing.v1 import AppTest

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(
            f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
        ),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _health_app(role: str = "Engineer", data_mode: str = "nasa", selected_cell: "str | None" = None) -> AppTest:
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
    at.session_state["user_role"] = role
    at.session_state["page"] = "health"
    at.session_state["data_mode"] = data_mode
    if selected_cell:
        at.session_state["selected_cell"] = selected_cell
    return at


def _all_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _physics_rul_value(text: str) -> "str | None":
    """The RUL tiles are HTML rendered via st.markdown (render_card +
    metric_tile_html, not st.metric), so the value lives in the markdown
    text right after the label's closing tag."""
    import re
    m = re.search(r"Physics RUL \(SEI √fade\)</div><div[^>]*>([^<]+)</div>", text)
    return m.group(1) if m else None


def test_health_twin_projection_chart_renders_for_nasa_cell(isolated_db):
    """A real NASA cell (168 capacity-bearing cycles) must render the twin
    projection chart — measured history + central + 2σ band — beside the
    existing forecast, plus the GBRT-vs-physics RUL comparison strip."""
    at = _health_app(role="Engineer", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception, f"Health page crashed with the twin chart: {at.exception}"

    text = _all_text(at)
    assert "What is degrading B0005" in text  # page_health() rendered for the right cell
    assert "GBRT RUL (LCO-validated)" in text
    assert "Physics RUL (SEI √fade)" in text

    phys_rul = _physics_rul_value(text)
    assert phys_rul not in (None, "—"), (
        "NASA cell should yield a physics RUL crossing from the SEI fit"
    )

    # The projection chart caption only renders when the projection exists.
    captions = [c.value for c in at.caption]
    assert any("Physics projection" in c for c in captions), captions
    assert any("2σ" in c for c in captions), captions


def test_health_twin_renders_for_non_engineer_role(isolated_db):
    """The twin chart lives in the always-visible forecast section, so a
    non-Engineer (Engineering details unchecked) still sees it."""
    at = _health_app(role="Fleet Manager", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception, f"Health page crashed for Fleet Manager: {at.exception}"
    text = _all_text(at)
    assert "Physics RUL (SEI √fade)" in text
    assert _physics_rul_value(text) not in (None, "—")
    captions = [c.value for c in at.caption]
    assert any("Physics projection" in c for c in captions), captions


def test_health_twin_honest_empty_state_when_no_projection(isolated_db, monkeypatch):
    """When no projection is available (fewer than 5 capacity-bearing
    cycles), the chart area must show the honest Unavailable card — never
    crash, and never draw a fake projection. Simulated by stubbing the
    twin's update() to return no projection."""
    import digital_twin as _dt

    monkeypatch.setattr(
        _dt.CellTwin, "update",
        lambda self, df: {"projection": None, "history": {"n_cycles": 3}},
    )
    at = _health_app(role="Engineer", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception, f"Health page crashed on the empty state: {at.exception}"

    text = _all_text(at)
    assert "Physics Projection" in text
    assert "Unavailable" in text
    assert "≥5 capacity-bearing cycles" in text
