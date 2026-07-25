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
"""

import requests

PVGIS_PVCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"


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
