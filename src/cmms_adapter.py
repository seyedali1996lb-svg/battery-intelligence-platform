"""
CMMS/ERP maintenance write-back adapter.

No specific CMMS/ERP target system is named for this integration — per an
explicit product decision, this is a documented adapter *pattern* only,
following the exact shape already established by circunomics_adapter.py
and bms_connectors.py's Victron VRM connector: no live account exists to
test against, so this is built against the generic REST shape most
maintenance-ticketing systems (e.g. Maximo, SAP PM, Fiix, UpKeep) expose —
POST a ticket with an asset ID, title, description, and priority to a
partner-supplied endpoint, authenticated with a bearer token.

Returns None rather than raising when unconfigured, and never raises on
request failure — callers should treat None/an "error" key as "not
configured or not reachable" and keep the existing manual workflow, not
show a stack trace.

This is the write-back adapter contract now named explicitly in
src/adapter_contract.py — see that module before writing a new write-back
connector, and use its classify_result() helper at call sites instead of
re-deriving the None/"error"/success branch by hand.
"""

import datetime


def build_ticket_payload(
    cell_id: str,
    title: str,
    description: str,
    priority: str = "medium",
) -> dict:
    """
    Build the maintenance-ticket payload in the generic shape a CMMS/ERP
    REST API would expect. Pure data-shaping, no network call — usable
    standalone for tests.
    """
    return {
        "asset_id":     cell_id,
        "title":        title,
        "description":  description,
        "priority":     priority,
        "created_at":   datetime.datetime.now().isoformat(),
        "source_system": "battery-intelligence-platform",
    }


def create_maintenance_ticket(
    cell_id: str,
    title: str,
    description: str,
    priority: str,
    api_key: str,
    api_base_url: str = "https://api.example-cmms.com/v1",
) -> "dict | None":
    """
    Create a maintenance ticket for a cell in an external CMMS/ERP system.

    Returns None if api_key is not supplied — callers should treat this as
    "not configured" and keep the existing manual/decision-log workflow.
    Returns the parsed JSON response dict on success, or a dict with an
    "error" key on a non-2xx response or network failure (never raises).

    Not tested against a live account (see module docstring) — built
    against the generic REST shape most CMMS/ERP APIs use (POST /tickets
    with a bearer token). The exact endpoint path and payload schema will
    need adjusting once a real target system is chosen.
    """
    if not api_key:
        return None

    import requests

    payload = build_ticket_payload(cell_id, title, description, priority)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(f"{api_base_url.rstrip('/')}/tickets", json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
