"""
PVGIS PVcalc client — solar PV yield estimation for the Solar + Storage
Sizing calculator (Consequences page).

PVGIS (re.jrc.ec.europa.eu) is the European Commission's free, public,
no-API-key solar resource and PV performance tool. Its PVcalc endpoint
does the irradiance/geometry modelling server-side given only a location,
system size, and mounting geometry — so this app doesn't need pvlib or
any local solar-geometry modelling.

No credential is required, so unlike bms_connectors.py/circunomics_adapter.py/
cmms_adapter.py there is no "not configured -> return None" guard clause.
Every call is attempted; failures (unreachable host, bad response, timeout)
are caught and returned as {"error": str(e)} rather than raised, matching
the never-raise-on-failure contract used by fetch_orion_bms()/circunomics_adapter.py/
cmms_adapter.py, so a caller (the Streamlit page) never needs its own
try/except around this call.

Verified live against the real PVGIS v5.2 API while building this module:
GET https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat=..&lon=..&peakpower=..
    &loss=..&angle=..&aspect=..&outputformat=json
returns {"outputs": {"totals": {"fixed": {"E_y": <annual kWh>, ...}},
                     "monthly": {"fixed": [{"month": 1, "E_m": <kWh>, ...}, ...]}}}.

Also verified live: the seriescalc endpoint (same API family), with
pvcalculation=1, returns genuine HOURLY PV power output (field "P", in W)
for a full calendar year — 8760 records for a non-leap year — again
computed server-side, no local PV/irradiance modelling needed. Used by
the Solar + Storage Sizing calculator's hourly dispatch simulation
(src/deployment_sizing.py) instead of the monthly PVcalc totals.

IMPORTANT — verified live: seriescalc's hourly "time" field is in UTC,
not site-local time (confirmed via a known-longitude test: a Zagreb-area
site's summer solar-noon peak landed at UTC hour ~10, matching CEST
(UTC+2)'s ~10:30 UTC solar noon, not the ~12:xx it would show if
timestamps were already local). Callers building an hourly dispatch
simulation must correct for this — see deployment_sizing.utc_offset_hours()
/ shift_to_local_hours() — this module only fetches the raw UTC-indexed
series and does not attempt the correction itself.

Also verified live: PVGIS has a dedicated tmy (Typical Meteorological
Year) endpoint — a constructed "typical" year built by selecting the most
statistically representative real month from a multi-year dataset for
each calendar month, rather than one arbitrary historical year (see
fetch_tmy_ghi()). CONFIRMED LIVE that tmy does NOT support
pvcalculation=1 (the param is silently ignored) — it only returns raw
weather (global horizontal irradiance, ambient temperature), not PV power
output. Getting real PV power from TMY data would require this app to
reimplement irradiance-to-power modelling itself (plane-of-array
transposition, temperature coefficients, inverter curves) — exactly the
complexity avoided everywhere else in this module by leaning on PVGIS's
own PVcalc/seriescalc pvcalculation=1 support. fetch_tmy_ghi() is used
only as a SHAPE proxy (deployment_sizing.build_typical_year_pv_shape()
redistributes real PVcalc-derived monthly kWh totals across hours
proportional to TMY's irradiance pattern), not a PV-output source on its own.
"""

import requests

PVGIS_PVCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
PVGIS_SERIESCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/v5_2/tmy"

# 2013 verified non-leap (365 days -> exactly 8760 hourly records) AND
# verified live to return valid data across all 3 of PVGIS's regional
# radiation databases: PVGIS-SARAH2 (Europe/Africa), PVGIS-NSRDB (the
# Americas — whose valid seriescalc year range is only 2005-2015, so the
# previous default of 2019 would have failed there entirely, confirmed
# live via a Los Angeles test that returned "Please enter an integer
# between 2005 and 2015"), and PVGIS-ERA5 (elsewhere, e.g. Asia). Used as
# the single fixed reference year for the hourly dispatch simulation — NOT
# a claim that 2013's weather was a "typical" year; see
# fetch_pv_yield_hourly()'s docstring, and build_typical_year_pv_shape()
# in deployment_sizing.py for a better-shaped alternative.
HOURLY_REFERENCE_YEAR = 2013


def compass_to_pvgis_azimuth(compass_deg: float) -> float:
    """
    Convert a compass bearing (0=N, 90=E, 180=S, 270=W) to PVGIS's azimuth
    convention (0=S, -90=E, +90=W, +-180=N), confirmed against the live API's
    own field description: "Orientation (azimuth) angle ... 0 = S, 90 = W, -90 = E".

    Result is normalized to (-180, 180].
    """
    pvgis = (compass_deg - 180.0) % 360.0
    if pvgis > 180.0:
        pvgis -= 360.0
    return pvgis


def fetch_pv_yield(
    lat: float,
    lon: float,
    peakpower_kwp: float,
    tilt_deg: float,
    azimuth_deg: float,
    loss_pct: float = 14.0,
    timeout: int = 15,
) -> dict:
    """
    Fetch estimated PV energy yield from PVGIS for a fixed-mount system.

    azimuth_deg must already be in PVGIS convention (0=S, -90=E, +90=W) —
    call compass_to_pvgis_azimuth() first if the caller collected a compass
    bearing from the user.

    Never raises. Returns {"annual_kwh": float, "monthly_kwh": list[12 floats],
    "months": list[12 ints]} on success, or {"error": str} on any failure
    (network, timeout, unexpected response shape).
    """
    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": peakpower_kwp,
        "loss": loss_pct,
        "angle": tilt_deg,
        "aspect": azimuth_deg,
        "outputformat": "json",
    }

    try:
        resp = requests.get(PVGIS_PVCALC_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        outputs = payload["outputs"]
        annual_kwh = float(outputs["totals"]["fixed"]["E_y"])
        monthly_rows = sorted(outputs["monthly"]["fixed"], key=lambda r: r["month"])
        monthly_kwh = [float(r["E_m"]) for r in monthly_rows]
        months = [int(r["month"]) for r in monthly_rows]
    except Exception as e:
        return {"error": str(e)}

    if len(monthly_kwh) != 12:
        return {"error": f"PVGIS returned {len(monthly_kwh)} monthly records, expected 12"}

    return {"annual_kwh": annual_kwh, "monthly_kwh": monthly_kwh, "months": months}


def fetch_pv_yield_hourly(
    lat: float,
    lon: float,
    peakpower_kwp: float,
    tilt_deg: float,
    azimuth_deg: float,
    loss_pct: float = 14.0,
    year: int = HOURLY_REFERENCE_YEAR,
    timeout: int = 30,
) -> dict:
    """
    Fetch hourly PV power output from PVGIS's seriescalc endpoint for one
    full calendar year (pvcalculation=1 — PVGIS computes real PV output,
    not just raw irradiance).

    azimuth_deg must already be in PVGIS convention (0=S, -90=E, +90=W) —
    same as fetch_pv_yield(). The returned series is UTC-indexed (hour 0 =
    Jan 1 00:xx UTC) — see this module's docstring; callers building a
    local-time dispatch simulation must shift it themselves.

    This is ONE fixed historical year of measured/satellite-derived weather
    (PVGIS-SARAH2/ERA5), NOT the multi-year climate average fetch_pv_yield()'s
    PVcalc totals use — a single year's annual total can differ from a
    "typical" year by a non-trivial margin. Larger payload than fetch_pv_yield
    (8760 vs 12 records) — timeout defaults higher (30s vs 15s).

    Never raises. Returns {"pv_kwh": list[8760 floats], "temp_c": list[8760 floats]}
    on success (P in W for a 1-hour sample -> kWh via /1000; T2m is PVGIS's
    2m ambient air temperature in °C, included "for free" in the same
    response — used by deployment_sizing's temperature-aware power derating
    so no second API call is needed), or {"error": str} on any failure,
    including a response that isn't exactly 8760 records (catches a leap
    year, a partial response, or PVGIS changing its sampling resolution).
    """
    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": peakpower_kwp,
        "loss": loss_pct,
        "angle": tilt_deg,
        "aspect": azimuth_deg,
        "outputformat": "json",
        "pvcalculation": 1,
        "startyear": year,
        "endyear": year,
    }

    try:
        resp = requests.get(PVGIS_SERIESCALC_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        records = sorted(payload["outputs"]["hourly"], key=lambda r: r["time"])
        pv_kwh = [float(r["P"]) / 1000.0 for r in records]
        temp_c = [float(r["T2m"]) for r in records]
    except Exception as e:
        return {"error": str(e)}

    if len(pv_kwh) != 8760:
        return {"error": f"PVGIS returned {len(pv_kwh)} hourly records for {year}, expected 8760"}

    return {"pv_kwh": pv_kwh, "temp_c": temp_c}


def fetch_tmy_ghi(lat: float, lon: float, timeout: int = 30) -> dict:
    """
    Fetch PVGIS's Typical Meteorological Year (TMY) — see module docstring
    for what this is and why it's a shape-only input, not a PV-output source.

    Independent of PV system geometry (no peakpower/tilt/azimuth params —
    TMY is a weather dataset, not a PV-performance calculation), so unlike
    fetch_pv_yield_hourly() this only needs to be fetched ONCE per site,
    not once per candidate PV size.

    Never raises. Returns {"ghi_wm2": list[8760 floats], "temp_c": list[8760
    floats]} on success (G(h) = global horizontal irradiance in W/m²; T2m =
    2m ambient air temperature in °C), or {"error": str} on any failure,
    including a response that isn't exactly 8760 records.
    """
    params = {"lat": lat, "lon": lon, "outputformat": "json"}

    try:
        resp = requests.get(PVGIS_TMY_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        records = sorted(payload["outputs"]["tmy_hourly"], key=lambda r: r["time(UTC)"])
        ghi = [float(r["G(h)"]) for r in records]
        temp_c = [float(r["T2m"]) for r in records]
    except Exception as e:
        return {"error": str(e)}

    if len(ghi) != 8760:
        return {"error": f"PVGIS returned {len(ghi)} TMY hourly records, expected 8760"}

    return {"ghi_wm2": ghi, "temp_c": temp_c}
