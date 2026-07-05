"""
Circunomics marketplace adapter.

Circunomics operates a real B2B platform for trading second-life batteries,
but publishes no self-serve public API documentation — access is provisioned
per-partner after a sales conversation. Same situation as Victron VRM in
bms_connectors.py: no live account exists for this project to test against.

This module builds a listing payload in the generic shape a battery-trading
marketplace API would expect (asset identity, SOH, chemistry, capacity,
asking price) and posts it to a partner-supplied endpoint. It returns None
rather than raising when unconfigured, and never raises on request failure —
callers should treat None/False as "not configured or not reachable" and
keep showing the existing demo-listing UI, not a stack trace.
"""

import datetime

import pandas as pd


def build_listing_payload(
    cell_id: str,
    soh_pct: float,
    chemistry: str,
    capacity_ah: float,
    asking_usd: float,
) -> dict:
    """
    Build the listing payload in the shape a Circunomics-style marketplace
    API would expect. Pure data-shaping, no network call — usable standalone
    for tests or for the demo listing UI's local session-state append.
    """
    return {
        "asset_id":       cell_id,
        "soh_pct":        round(soh_pct, 1),
        "chemistry":      chemistry,
        "capacity_ah":    round(capacity_ah, 2),
        "asking_usd":     round(asking_usd, 2),
        "listed_at":      datetime.datetime.now().isoformat(),
        "source_platform": "battery-intelligence-platform",
    }


def list_cell_on_circunomics(
    cell_id: str,
    soh_pct: float,
    chemistry: str,
    capacity_ah: float,
    asking_usd: float,
    api_key: str,
    api_base_url: str = "https://api.circunomics.com/v1",
) -> "dict | None":
    """
    Submit a second-life listing to Circunomics.

    Returns None if api_key is not supplied — callers should treat this as
    "not configured" and fall back to the local demo listing behavior.
    Returns the parsed JSON response dict on success, or a dict with an
    "error" key on a non-2xx response or network failure (never raises),
    so the UI can show a clear message either way.

    Not tested against a live account (see module docstring) — built
    against the generic REST shape most B2B marketplace APIs use
    (POST /listings with a bearer token). The exact endpoint path and
    payload schema will need adjusting once real API docs are available
    from a Circunomics partner account.
    """
    if not api_key:
        return None

    import requests

    payload = build_listing_payload(cell_id, soh_pct, chemistry, capacity_ah, asking_usd)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(f"{api_base_url.rstrip('/')}/listings", json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
