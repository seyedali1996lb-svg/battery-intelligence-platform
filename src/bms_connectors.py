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
