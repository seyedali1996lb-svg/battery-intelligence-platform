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

    def _fast_annual(**kwargs):
        return {"annual_kwh": 8760.0, "monthly_kwh": [730.0] * 12, "months": list(range(1, 13))}

    monkeypatch.setattr(pvgis_client, "fetch_pv_yield_hourly", _fast_hourly)
    monkeypatch.setattr(pvgis_client, "fetch_pv_yield", _fast_annual)

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


def test_solar_storage_sizing_csv_upload_without_file_shows_error_not_crash(isolated_db):
    """Selecting 'Upload hourly CSV' and clicking Calculate with no file
    chosen must show a validation error, never crash the page."""
    at = _logged_in_app(
        role="Engineer", page="consequences", data_mode="nasa",
        selected_cell="B0006",
        siz_shape="Upload hourly CSV",
    )
    at.run()
    assert not at.exception, f"Consequences page crashed: {at.exception}"

    at.button(key="siz_calculate_btn").click().run()
    assert not at.exception, f"Calculate crashed with no CSV uploaded: {at.exception}"
    # st.error() renders into at.error, not at.markdown.
    error_texts = [e.value for e in at.error]
    assert any("Upload a valid" in t for t in error_texts), error_texts


def test_parse_hourly_consumption_csv_valid_and_invalid_inputs():
    """Direct unit test of the CSV parser (no AppTest needed — file-upload
    widget simulation isn't reliably supported, so this exercises the pure
    parsing logic directly instead)."""
    import io
    from _pages.consequences import _parse_hourly_consumption_csv

    valid_no_header = io.StringIO("\n".join(str(1.0) for _ in range(8760)))
    assert _parse_hourly_consumption_csv(valid_no_header) == [1.0] * 8760

    valid_with_header = io.StringIO("kwh\n" + "\n".join(str(2.5) for _ in range(8760)))
    parsed = _parse_hourly_consumption_csv(valid_with_header)
    assert parsed == [2.5] * 8760

    wrong_length = io.StringIO("\n".join(str(1.0) for _ in range(100)))
    assert _parse_hourly_consumption_csv(wrong_length) is None

    garbage = io.StringIO("not,a,valid,csv,at,all\n" * 5)
    assert _parse_hourly_consumption_csv(garbage) is None


def test_parse_hourly_consumption_csv_two_column_timestamp_format():
    """Real smart-meter-style export: timestamp,kWh, hourly cadence, header row."""
    import io
    import pandas as pd
    from _pages.consequences import _parse_hourly_consumption_csv

    idx = pd.date_range("2023-01-01", periods=8760, freq="h")
    lines = ["timestamp,kWh"] + [f"{ts.isoformat()},1.5" for ts in idx]
    buf = io.StringIO("\n".join(lines))
    parsed = _parse_hourly_consumption_csv(buf)
    assert parsed is not None
    assert len(parsed) == 8760
    assert all(v == pytest.approx(1.5) for v in parsed)


def test_parse_hourly_consumption_csv_two_column_sub_hourly_resamples():
    """15-minute-interval readings should sum up to hourly totals, not be
    dropped or misinterpreted as a malformed file."""
    import io
    import pandas as pd
    from _pages.consequences import _parse_hourly_consumption_csv

    idx = pd.date_range("2023-01-01", periods=8760 * 4, freq="15min")
    lines = ["timestamp,kWh"] + [f"{ts.isoformat()},0.5" for ts in idx]
    buf = io.StringIO("\n".join(lines))
    parsed = _parse_hourly_consumption_csv(buf)
    assert parsed is not None
    assert len(parsed) == 8760
    assert all(v == pytest.approx(2.0) for v in parsed)  # 4 x 0.5 kWh per hour


def test_solar_storage_sizing_typical_year_and_region_preset_render_candidate_table(isolated_db, monkeypatch):
    """Exercises the TMY typical-year weather source, a non-default site
    region cost preset, and the candidate-grid table together — the
    combination of new controls added in this batch."""
    import pvgis_client

    def _fast_tmy(**kwargs):
        ghi = [100.0 if (h % 24) in range(8, 17) else 0.0 for h in range(8760)]
        return {"ghi_wm2": ghi, "temp_c": [15.0] * 8760}

    def _fast_annual(**kwargs):
        return {"annual_kwh": 1000.0, "monthly_kwh": [1000.0 / 12] * 12, "months": list(range(1, 13))}

    monkeypatch.setattr(pvgis_client, "fetch_tmy_ghi", _fast_tmy)
    monkeypatch.setattr(pvgis_client, "fetch_pv_yield", _fast_annual)

    at = _logged_in_app(
        role="Engineer", page="consequences", data_mode="nasa",
        selected_cell="B0006",
        siz_lat=33.111, siz_lon=44.222,  # unique coords, avoid cross-test cache hits
        siz_weather_source="Typical year (TMY shape)",
        siz_site_region="Germany",
    )
    at.run()
    assert not at.exception

    at.button(key="siz_calculate_btn").click().run()
    assert not at.exception, f"Calculate crashed with typical-year + region preset: {at.exception}"

    text = _all_text(at)
    assert "PV size" in text  # a real result rendered

    expander_labels = [e.label for e in at.expander]
    assert any("sizes explored" in label for label in expander_labels), expander_labels
