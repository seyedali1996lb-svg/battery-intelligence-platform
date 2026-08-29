"""
AppTest page-level verification for the Grid Services page
(app/_pages/operations.py) — the Streamlit surface for the P1/P2 analytics
(health-aware dispatch, grid revenue, managed charging, fleet offers, ML
anomaly). Pure logic is covered by the module tests (test_health_aware_dispatch.py,
test_grid_services.py, test_managed_charging.py, test_fleet_aggregation.py,
test_ml_anomaly.py); this test's job is UI wiring: each tab renders without
exception and the honest labels are visible.
"""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
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
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _logged_in_app(page: str, data_mode: str) -> AppTest:
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
    at.session_state["user_role"] = "admin"
    at.session_state["page"] = page
    at.session_state["data_mode"] = data_mode
    return at


def test_operations_page_renders_all_tabs(isolated_db):
    at = _logged_in_app("operations", "synthetic")
    at.run()
    assert not at.exception, at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "Grid Services & Energy Operations" in text
    # Honest framing visible
    assert "not control signals" in text


def test_operations_dispatch_tab_metrics(isolated_db):
    at = _logged_in_app("operations", "synthetic")
    at.run()
    assert not at.exception, at.exception
    # Metrics from the health-aware dispatch render (revenue values present)
    metric_values = [m.value for m in at.metric]
    assert any("€" in str(v) for v in metric_values)
    # Band limitation caption rendered
    caption_text = "\n".join(c.value for c in at.caption)
    assert "Band:" in caption_text


def test_operations_nav_wiring(isolated_db):
    """The nav group gained a Grid Services entry — the button exists."""
    at = _logged_in_app("overview", "synthetic")
    at.run()
    assert not at.exception, at.exception
    assert any(b.label == "Grid Services" for b in at.button)
