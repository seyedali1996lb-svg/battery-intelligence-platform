"""
AppTest-based page-level verification for the Solar + Storage Sizing
expander added to app/_pages/consequences.py (see plan doc). Follows the
same pattern as tests/test_app_state_combinations.py: bypass login via
session_state, run the real app script, and assert no exception —
pure-logic math is already covered by tests/test_deployment_sizing.py and
tests/test_pvgis_client.py, so this test's job is UI-wiring correctness
only (the expander renders, the button/PVGIS-failure path doesn't crash),
per this project's standing "no live browser during build" rule.
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
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _logged_in_app(role: str, page: str, data_mode: str, **extra_state) -> AppTest:
    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = role
    at.session_state["page"] = page
    at.session_state["data_mode"] = data_mode
    for k, v in extra_state.items():
        at.session_state[k] = v
    return at


def _all_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_solar_storage_sizing_expander_renders_without_exception(isolated_db):
    """Regression guard for the new expander's default (not-yet-calculated)
    state — must render on a second-life-eligible cell without crashing."""
    at = _logged_in_app(
        role="Engineer", page="consequences", data_mode="nasa",
        selected_cell="B0006",  # NASA cell at ~58% SOH — solidly past the 85% primary-life gate
    )
    at.run()
    assert not at.exception, f"Consequences page crashed: {at.exception}"
    # The expander's title is an st.expander label, not an st.markdown element,
    # so it lives in at.expander (not _all_text(at), which only covers markdown).
    expander_labels = [e.label for e in at.expander]
    assert any("Solar" in label for label in expander_labels), expander_labels
    text = _all_text(at)
    assert "hour-by-hour" in text  # body text, confirms the hourly-engine copy actually rendered


def test_solar_storage_sizing_degrades_gracefully_when_pvgis_unavailable(isolated_db, monkeypatch):
    """Clicking Calculate when PVGIS is unreachable must show an honest
    empty state / warning, never an uncaught stack trace."""
    import pvgis_client

    def _always_fails(**kwargs):
        return {"error": "simulated PVGIS outage"}

    monkeypatch.setattr(pvgis_client, "fetch_pv_yield_hourly", _always_fails)

    at = _logged_in_app(
        role="Engineer", page="consequences", data_mode="nasa",
        selected_cell="B0006",  # NASA cell at ~58% SOH — solidly past the 85% primary-life gate
        # Unique, unlikely-to-collide-with-other-tests coordinates so
        # st.cache_data's process-wide cache can't mask this test behind an
        # earlier successful call for the same (lat, lon, tilt, azimuth, kWp, year).
        siz_lat=-71.234, siz_lon=169.876,
    )
    at.run()
    assert not at.exception, f"Consequences page crashed before Calculate: {at.exception}"

    calculate_btn = at.button(key="siz_calculate_btn")
    calculate_btn.click().run()
    assert not at.exception, f"Solar + Storage Sizing crashed on PVGIS failure: {at.exception}"

    # pv_kwp=0 is synthesized locally (never calls PVGIS), so a battery-only
    # near-miss/feasible result is still expected rather than a blank error —
    # either an explicit warning about PVGIS, or the honest empty-state path.
    text = _all_text(at)
    assert ("PVGIS" in text) or ("No sizing yet" not in text)


def test_solar_storage_sizing_calculate_flow_completes_promptly(isolated_db, monkeypatch):
    """Times the full Calculate-button flow with a mocked (fast, deterministic)
    PVGIS response — the hourly engine runs a real 8760-hour simulation per
    candidate (up to ~45 candidates with the two-phase refine), and this
    project's Streamlit Community Cloud deploy has hit resource limits before
    (see docs/history.md), so a latency regression here is worth catching in
    CI rather than only discovered live."""
    import time
    import pvgis_client

    def _fast_hourly(**kwargs):
        return {"pv_kwh": [1.0] * 8760}

    monkeypatch.setattr(pvgis_client, "fetch_pv_yield_hourly", _fast_hourly)

    at = _logged_in_app(
        role="Engineer", page="consequences", data_mode="nasa",
        selected_cell="B0006",
        siz_lat=12.345, siz_lon=67.891,  # unique coords, avoid cross-test cache hits
    )
    at.run()
    assert not at.exception

    t0 = time.time()
    at.button(key="siz_calculate_btn").click().run()
    elapsed = time.time() - t0
    assert not at.exception, f"Calculate flow crashed: {at.exception}"

    text = _all_text(at)
    assert "PV size" in text  # winner metric tile rendered -> a real result, not an empty state

    # Not a hard assertion (CI machines vary) — but flag loudly if this creeps
    # past a few seconds, since that's a genuine UX cost of the hourly upgrade.
    print(f"[timing] Solar + Storage Sizing Calculate flow: {elapsed:.2f}s (mocked PVGIS)")
    assert elapsed < 15.0, (
        f"Calculate flow took {elapsed:.2f}s even with PVGIS mocked out — "
        "the hourly dispatch simulation itself may need optimizing."
    )
