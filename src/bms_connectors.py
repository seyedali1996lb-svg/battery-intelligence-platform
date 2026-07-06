"""
Real BMS connector adapters.

These fetch cycle data from a real vendor API and reshape it into this
app's standard cycle-data schema (cell_id, cycle_number, capacity_ah,
resistance_ohm, temperature_c, test_date) — the same columns produced by
validate_upload()/adapt_upload_to_pipeline() for a manual CSV upload, so
a connector's output can be fed into the existing upload pipeline
unchanged.

No live vendor account exists for this project to test against. Each
fetch_* function is built against the vendor's publicly documented REST
API shape but is NOT a verified integration — it will only ever be
called with user-supplied credentials (never hardcoded), and returns
None rather than raising when unconfigured, so the UI can show a clear
empty state instead of a stack trace.
"""

import pandas as pd


def fetch_victron_vrm(base_url: str, api_token: str, installation_id: str) -> "pd.DataFrame | None":
    """
    Fetch battery cycle/telemetry data from a Victron VRM installation.

    base_url:         Victron VRM API base, e.g. "https://vrmapi.victronenergy.com/v2"
    api_token:         User-supplied VRM API access token (Settings -> Integrations in VRM).
    installation_id:   Numeric VRM installation ID.

    Returns None if base_url/api_token/installation_id are not all supplied —
    callers should treat None as "not configured" and show an empty state,
    not an error.

    Not tested against a live account (see module docstring). Built against
    the publicly documented VRM API shape: GET /installations/{id}/stats
    returns time-series records; this function maps the battery-relevant
    fields onto the app's standard cycle-data columns.
    """
    if not (base_url and api_token and installation_id):
        return None

    import requests

    url = f"{base_url.rstrip('/')}/installations/{installation_id}/stats"
    headers = {"X-Authorization": f"Token {api_token}"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    records = payload.get("records", payload.get("data", {}))
    rows = []
    for i, rec in enumerate(records if isinstance(records, list) else records.get("bs", [])):
        rows.append({
            "cell_id":        f"VRM-{installation_id}",
            "cycle_number":   i,
            "capacity_ah":    rec.get("consumed_ah") or rec.get("capacity_ah"),
            "resistance_ohm": rec.get("internal_resistance") or float("nan"),
            "temperature_c":  rec.get("battery_temperature") or rec.get("temperature"),
            "test_date":      rec.get("timestamp") or rec.get("time"),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def fetch_orion_bms(
    cell_id: str,
    api_key: str,
    api_base_url: str = "https://api.orion-bms.com/v1",
) -> "pd.DataFrame | None":
    """
    Fetch battery cycle/telemetry data for a cell monitored by an Orion Jr2
    BMS via its REST gateway.

    Returns None if api_key is not supplied — callers should treat this as
    "not configured" and show an empty state, not an error. Returns a dict
    with an "error" key on a non-2xx response or network failure (never
    raises) — a different contract than fetch_victron_vrm() above (which
    lets HTTP errors propagate for its caller to catch), but the same
    never-raise-on-failure guarantee as circunomics_adapter.py/cmms_adapter.py,
    chosen here so the Settings page's "Test connection" button doesn't need
    its own try/except.

    Not tested against a live account (see module docstring). Built against
    the publicly documented Orion Jr2 REST integration shape: GET
    /cells/{cell_id}/telemetry returns time-series records; this function
    maps the battery-relevant fields onto the app's standard cycle-data
    columns (same 6 columns as fetch_victron_vrm, for a consistent shape
    across BMS connectors).
    """
    if not api_key:
        return None

    import requests

    url = f"{api_base_url.rstrip('/')}/cells/{cell_id}/telemetry"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return {"error": str(e)}

    records = payload.get("records", payload.get("data", []))
    rows = []
    for i, rec in enumerate(records if isinstance(records, list) else []):
        rows.append({
            "cell_id":        cell_id,
            "cycle_number":   i,
            "capacity_ah":    rec.get("capacity_ah") or rec.get("remaining_capacity"),
            "resistance_ohm": rec.get("internal_resistance") or float("nan"),
            "temperature_c":  rec.get("temperature") or rec.get("cell_temperature"),
            "test_date":      rec.get("timestamp") or rec.get("time"),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)
