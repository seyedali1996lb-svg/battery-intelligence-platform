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
"""

import requests

PVGIS_PVCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
PVGIS_SERIESCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"

# 2019 verified non-leap (365 days -> exactly 8760 hourly records) and within
# PVGIS-SARAH2's valid year range for European sites. Used as the single
# fixed reference year for the hourly dispatch simulation — NOT a claim that
# 2019's weather was a "typical" year; see fetch_pv_yield_hourly()'s docstring.
HOURLY_REFERENCE_YEAR = 2019


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

    Never raises. Returns {"pv_kwh": list[8760 floats]} on success (P in W
    for a 1-hour sample -> kWh via /1000), or {"error": str} on any failure,
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
    except Exception as e:
        return {"error": str(e)}

    if len(pv_kwh) != 8760:
        return {"error": f"PVGIS returned {len(pv_kwh)} hourly records for {year}, expected 8760"}

    return {"pv_kwh": pv_kwh}
