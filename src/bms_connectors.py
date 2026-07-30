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

BMSAdapter (below) is this project's *object*-level connector contract —
a reusable, stateful instance for repeated fetches. See
src/adapter_contract.py for the sibling *function*-level contract used by
one-shot write-back connectors (circunomics_adapter.py, cmms_adapter.py);
that module's docstring explains when a new connector should use each shape.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BMSAdapter(Protocol):
    """
    Structural contract every BMS connector class in this module satisfies.

    Codifies what fetch_victron_vrm()/fetch_orion_bms() already do
    informally (credential guard -> vendor request -> map onto the app's
    standard cycle-data columns) so the next connector is "implement these
    3 members," not "read the other two and copy the pattern." This is a
    Protocol, not an ABC each adapter must inherit from -- any object with
    a matching `name`/`is_configured()`/`fetch()` shape satisfies it, which
    keeps the underlying fetch_*() functions (already exercised directly
    by app/_pages/settings.py and tests/test_bms_connectors.py) completely
    untouched.
    """

    name: str

    def is_configured(self) -> bool:
        """True if this adapter has everything it needs to attempt a fetch
        (credentials/IDs supplied), without making any network call."""
        ...

    def fetch(self) -> "pd.DataFrame | None":
        """Fetch cycle data in the app's standard schema (cell_id,
        cycle_number, capacity_ah, resistance_ohm, temperature_c,
        test_date), or None if not configured or the vendor returned no
        records. Each concrete adapter documents its own error-handling
        contract (raise vs. never-raise) -- see VictronVRMAdapter /
        OrionBMSAdapter below."""
        ...


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


# ---------------------------------------------------------------------------
# BMSAdapter-conforming wrapper classes
#
# These wrap the free functions above with a uniform stateful interface
# (credentials held on the instance, not re-passed to every call) and a
# uniform test_connection() contract -- new code, so unlike the wrapped
# functions it can standardize on "never raise, always return a result
# dict" regardless of which underlying fetch_*() contract it delegates to.
# The wrapped functions themselves are untouched: existing callers
# (app/_pages/settings.py, tests/test_bms_connectors.py) keep working as-is.
# ---------------------------------------------------------------------------

class VictronVRMAdapter:
    """BMSAdapter wrapping fetch_victron_vrm(). fetch_victron_vrm() itself
    lets HTTP errors propagate (its existing, tested contract) -- this
    class's fetch() catches them instead, so a caller of the class-based
    interface never needs its own try/except."""

    name = "Victron VRM"

    def __init__(self, base_url: str, api_token: str, installation_id: str):
        self.base_url = base_url
        self.api_token = api_token
        self.installation_id = installation_id

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_token and self.installation_id)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        try:
            return fetch_victron_vrm(self.base_url, self.api_token, self.installation_id)
        except Exception:
            return None

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        try:
            df = fetch_victron_vrm(self.base_url, self.api_token, self.installation_id)
        except Exception as e:
            return {"ok": False, "message": f"VRM connection failed: {e}", "n_records": None}
        if df is None or len(df) == 0:
            return {"ok": True, "message": "Connected, but no records were returned.", "n_records": 0}
        return {"ok": True, "message": f"Fetched {len(df)} records.", "n_records": len(df)}


class OrionBMSAdapter:
    """BMSAdapter wrapping fetch_orion_bms(). fetch_orion_bms() already
    never raises (returns a dict with an "error" key instead) -- this
    class's fetch() normalizes that into the shared None/DataFrame
    contract every BMSAdapter.fetch() promises."""

    name = "Orion Jr2"

    def __init__(self, cell_id: str, api_key: str, api_base_url: str = "https://api.orion-bms.com/v1"):
        self.cell_id = cell_id
        self.api_key = api_key
        self.api_base_url = api_base_url

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        result = fetch_orion_bms(self.cell_id, self.api_key, self.api_base_url)
        if isinstance(result, dict):  # {"error": ...} -- not a DataFrame
            return None
        return result

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        result = fetch_orion_bms(self.cell_id, self.api_key, self.api_base_url)
        if result is None:
            return {"ok": True, "message": "Connected, but no records were returned.", "n_records": 0}
        if isinstance(result, dict) and "error" in result:
            return {"ok": False, "message": f"Orion connection failed: {result['error']}", "n_records": None}
        return {"ok": True, "message": f"Fetched {len(result)} records.", "n_records": len(result)}
