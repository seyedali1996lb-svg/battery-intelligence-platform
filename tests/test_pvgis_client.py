"""Unit tests for src/pvgis_client.py.

fetch_pv_yield() has no "not configured" state (PVGIS needs no API key) —
every call is attempted, and failures are caught and returned as an
{"error": ...} dict rather than raised, matching the never-raise contract
used by fetch_orion_bms()/circunomics_adapter.py/cmms_adapter.py (see
test_bms_connectors.py). compass_to_pvgis_azimuth() is tested separately
since a sign-convention bug there would silently produce a plausible-looking
but wrong yield number rather than an exception."""

from pvgis_client import compass_to_pvgis_azimuth, fetch_pv_yield, fetch_pv_yield_hourly, fetch_tmy_ghi


def test_compass_to_pvgis_azimuth_cardinal_points():
    # PVGIS's own field description: "0 = S, 90 = W, -90 = E"
    assert compass_to_pvgis_azimuth(180) == 0     # South -> 0
    assert compass_to_pvgis_azimuth(90) == -90    # East -> -90
    assert compass_to_pvgis_azimuth(270) == 90    # West -> +90
    assert abs(compass_to_pvgis_azimuth(0)) == 180  # North -> +-180


def test_compass_to_pvgis_azimuth_result_is_normalized():
    for compass_deg in range(0, 360, 15):
        result = compass_to_pvgis_azimuth(compass_deg)
        assert -180.0 < result <= 180.0


def test_fetch_pv_yield_never_raises_against_real_endpoint():
    # Real PVGIS call — may succeed (network available) or fail (sandboxed/offline
    # environment). Either way it must never raise; assert on the contract shape.
    result = fetch_pv_yield(
        lat=45.8, lon=15.98, peakpower_kwp=1.0, tilt_deg=30, azimuth_deg=0,
        timeout=5,
    )
    assert isinstance(result, dict)
    assert ("annual_kwh" in result and "monthly_kwh" in result) or "error" in result


def test_fetch_pv_yield_returns_error_dict_on_bad_host():
    # PVGIS_PVCALC_URL is looked up as a module global at call time, so
    # monkeypatching the module attribute redirects fetch_pv_yield() itself.
    import pvgis_client
    original_url = pvgis_client.PVGIS_PVCALC_URL
    try:
        pvgis_client.PVGIS_PVCALC_URL = "https://this-host-does-not-exist.invalid/v1"
        result = fetch_pv_yield(
            lat=45.8, lon=15.98, peakpower_kwp=1.0, tilt_deg=30, azimuth_deg=0,
            timeout=5,
        )
    finally:
        pvgis_client.PVGIS_PVCALC_URL = original_url

    assert isinstance(result, dict)
    assert "error" in result


def test_fetch_pv_yield_hourly_never_raises_against_real_endpoint():
    # Real PVGIS seriescalc call — may succeed or fail depending on network
    # availability, but must never raise. Confirms the 8760-length contract.
    result = fetch_pv_yield_hourly(
        lat=45.8, lon=15.98, peakpower_kwp=1.0, tilt_deg=30, azimuth_deg=0,
        timeout=15,
    )
    assert isinstance(result, dict)
    if "pv_kwh" in result:
        assert len(result["pv_kwh"]) == 8760
        assert all(v >= 0 for v in result["pv_kwh"])
        assert "temp_c" in result
        assert len(result["temp_c"]) == 8760
    else:
        assert "error" in result


def test_fetch_pv_yield_hourly_returns_error_dict_on_bad_host():
    import pvgis_client
    original_url = pvgis_client.PVGIS_SERIESCALC_URL
    try:
        pvgis_client.PVGIS_SERIESCALC_URL = "https://this-host-does-not-exist.invalid/v1"
        result = fetch_pv_yield_hourly(
            lat=45.8, lon=15.98, peakpower_kwp=1.0, tilt_deg=30, azimuth_deg=0,
            timeout=5,
        )
    finally:
        pvgis_client.PVGIS_SERIESCALC_URL = original_url

    assert isinstance(result, dict)
    assert "error" in result


def test_fetch_tmy_ghi_never_raises_against_real_endpoint():
    # Real PVGIS tmy call — may succeed or fail depending on network
    # availability, but must never raise. Confirms the 8760-length contract.
    result = fetch_tmy_ghi(lat=45.8, lon=15.98, timeout=15)
    assert isinstance(result, dict)
    if "ghi_wm2" in result:
        assert len(result["ghi_wm2"]) == 8760
        assert all(v >= 0 for v in result["ghi_wm2"])
        assert "temp_c" in result
        assert len(result["temp_c"]) == 8760
    else:
        assert "error" in result


def test_fetch_tmy_ghi_returns_error_dict_on_bad_host():
    import pvgis_client
    original_url = pvgis_client.PVGIS_TMY_URL
    try:
        pvgis_client.PVGIS_TMY_URL = "https://this-host-does-not-exist.invalid/v1"
        result = fetch_tmy_ghi(lat=45.8, lon=15.98, timeout=5)
    finally:
        pvgis_client.PVGIS_TMY_URL = original_url

    assert isinstance(result, dict)
    assert "error" in result
