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
        return {"error": str(e)}  # pyright: ignore[reportReturnType]

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


def fetch_modbus_bms(
    host: str,
    port: int,
    unit_id: int,
    register_map: dict,
    cell_id: str,
) -> "pd.DataFrame | None":
    """
    Read a single current-state snapshot from a Modbus TCP BMS/inverter.

    Modbus has no vendor-agnostic register layout the way Victron VRM /
    Orion Jr2's REST APIs do -- every vendor (Pylontech, Growatt, SMA,
    Deye, ...) assigns its own register addresses and scaling factors for
    capacity/resistance/temperature, and there is no real device available
    to this project to confirm one against (see module docstring). Rather
    than guess at a fixed layout that would silently misread a real
    device, the caller supplies register_map: {"capacity_ah": (address,
    scale), "resistance_ohm": (address, scale), "temperature_c": (address,
    scale)}. A field missing from the map reads as NaN, not an error --
    some devices don't expose internal resistance over Modbus at all.

    Unlike fetch_victron_vrm()/fetch_orion_bms(), a Modbus read returns
    holding-register values as they exist RIGHT NOW -- Modbus has no
    concept of historical time series the way a REST stats endpoint does.
    This returns exactly one row (cycle_number=0); building up per-cycle
    history from repeated reads (e.g. a scheduled poll appending rows over
    time) is the caller's responsibility, the same limitation any real
    Modbus BMS integration has.

    Returns None if host/port/register_map are not all supplied. Requires
    the optional `pymodbus` package -- raises ImportError with a clear
    message if it isn't installed, since this app doesn't otherwise depend
    on it (unlike `requests`, already a hard dependency for VRM/Orion).

    Not tested against a live device (see module docstring). Built against
    pymodbus's documented ModbusTcpClient API shape.
    """
    if not (host and port and register_map):
        return None

    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as e:
        raise ImportError(
            "Modbus BMS support requires the optional 'pymodbus' package "
            "(pip install pymodbus) -- not installed in this environment."
        ) from e

    client = ModbusTcpClient(host, port=port)
    if not client.connect():
        return None

    try:
        row = {"cell_id": cell_id, "cycle_number": 0, "test_date": None}
        for field in ("capacity_ah", "resistance_ohm", "temperature_c"):
            spec = register_map.get(field)
            if spec is None:
                row[field] = float("nan")
                continue
            address, scale = spec
            result = client.read_holding_registers(address, count=1, slave=unit_id)
            row[field] = float("nan") if result.isError() else result.registers[0] * scale
    finally:
        client.close()

    return pd.DataFrame([row])


def fetch_can_bus_bms(
    channel: str,
    bustype: str,
    node_id: int,
    pgn_map: dict,
    cell_id: str,
    timeout_s: float = 5.0,
) -> "pd.DataFrame | None":
    """
    Read a single current-state snapshot from a CAN bus BMS (common in EV
    packs) by listening for frames matching caller-supplied PGN/CAN IDs.

    Like Modbus, CAN has no single universal frame layout -- SAE J1939
    (the common heavy-EV/industrial convention), Tesla's, Orion Jr2's own
    CAN mode, and countless BMS vendors' proprietary DBC files all encode
    capacity/resistance/temperature at different CAN IDs with different
    byte layouts. This is deliberately the MOST assumption-heavy of this
    project's 3 new BMS adapters (Modbus/CAN/OCPP): pgn_map supplies which
    CAN arbitration ID carries which field and how to decode it:
    {"capacity_ah": (can_id, byte_offset, scale), "resistance_ohm": (...),
    "temperature_c": (...)}. With no real device or vendor DBC file to
    confirm frame layout against, this trusts the caller's map entirely --
    there is no fallback default to fall back to that wouldn't just be a
    guess.

    Listens for up to timeout_s seconds and returns whichever of the 3
    mapped fields it actually saw a frame for (NaN for the rest) -- a CAN
    bus is a continuous broadcast stream, not a request/response protocol,
    so "read now" means "listen briefly and take what arrives."

    Returns None if channel/bustype/pgn_map are not all supplied. Requires
    the optional `python-can` package -- raises ImportError with a clear
    message if it isn't installed.

    Not tested against a live device or a real vendor DBC file (see module
    docstring). Built against python-can's documented Bus/Message API shape.
    """
    if not (channel and bustype and pgn_map):
        return None

    try:
        import can
    except ImportError as e:
        raise ImportError(
            "CAN bus BMS support requires the optional 'python-can' package "
            "(pip install python-can) -- not installed in this environment."
        ) from e

    import time

    id_to_field = {spec[0]: (field, spec[1], spec[2]) for field, spec in pgn_map.items()}
    row = {"cell_id": cell_id, "cycle_number": 0, "test_date": None,
           "capacity_ah": float("nan"), "resistance_ohm": float("nan"), "temperature_c": float("nan")}

    bus = can.interface.Bus(channel=channel, bustype=bustype)
    try:
        deadline = time.monotonic() + timeout_s
        seen = set()
        while time.monotonic() < deadline and len(seen) < len(id_to_field):
            msg = bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None:
                break
            spec = id_to_field.get(msg.arbitration_id)
            if spec is None:
                continue
            field, byte_offset, scale = spec
            if byte_offset < len(msg.data):
                row[field] = msg.data[byte_offset] * scale
                seen.add(msg.arbitration_id)
    finally:
        bus.shutdown()

    return pd.DataFrame([row])


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


def fetch_ocpp_sessions(
    central_system_url: str,
    api_key: str,
    charge_point_id: str,
) -> "pd.DataFrame | None":
    """
    Fetch completed charging-session records from an OCPP Central System's
    REST query API (e.g. SteVe, or any Central System exposing transaction/
    MeterValues history over REST) for one charge point.

    A deliberate scope decision, not an oversight: OCPP itself (the
    protocol between a charger and its Central System) is WebSocket-based
    and stateful -- acting as a live Central System or Charge Point would
    mean running a persistent async connection, a materially different
    shape of integration than this app's other one-shot REST fetches, and
    a much larger undertaking to build untested against a real charge
    point. This instead queries a Central System's own REST reporting API
    for already-completed sessions, the same way an operator dashboard
    would, and is consistent with fetch_victron_vrm()/fetch_orion_bms()'s
    shape (credential guard -> one REST call -> map onto this app's
    columns).

    Charging-session data is NOT cell-level cycle data, and this function
    does not pretend otherwise: an OCPP transaction reports total energy
    delivered (kWh) and duration for a charging session at the CHARGE
    POINT level, not a per-cell capacity/resistance/temperature reading.
    Each row maps one completed session onto the app's standard columns
    as an approximation -- capacity_ah is derived from meterStop -
    meterStart (Wh) at an assumed nominal voltage (a real limitation
    inherent to what OCPP MeterValues expose, not this function's
    modeling choice), resistance_ohm is always NaN (OCPP does not report
    it at all), and cycle_number is the session's sequential index for
    this charge point. This makes OCPP data usable as an approximate
    capacity-trend input to the existing upload pipeline, not a
    substitute for real per-cell BMS telemetry.

    Returns None if central_system_url/api_key/charge_point_id are not
    all supplied. Requires no extra dependency beyond `requests` (already
    used by fetch_victron_vrm()) since this queries a REST reporting API,
    not the OCPP WebSocket protocol itself.

    Not tested against a live Central System (see module docstring). Built
    against the shape a REST transaction-history endpoint would plausibly
    expose (charge_point_id, transactions[] with meterStart/meterStop/
    startTimestamp/stopTimestamp) -- no single Central System implementation
    exposes an OCPP-standardized REST query API, so this is necessarily a
    best-effort shape, not a spec-verified one.
    """
    if not (central_system_url and api_key and charge_point_id):
        return None

    import requests

    url = f"{central_system_url.rstrip('/')}/chargepoints/{charge_point_id}/transactions"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    ASSUMED_NOMINAL_VOLTAGE = 3.7  # LiCoO2 nominal -- same assumption src/consequences.py's kWh math uses
    transactions = payload.get("transactions", payload.get("data", []))
    rows = []
    for i, tx in enumerate(transactions if isinstance(transactions, list) else []):
        meter_start = tx.get("meterStart")
        meter_stop = tx.get("meterStop")
        energy_wh = (meter_stop - meter_start) if (meter_start is not None and meter_stop is not None) else None
        rows.append({
            "cell_id":        charge_point_id,
            "cycle_number":   i,
            "capacity_ah":    (energy_wh / 1000.0 / ASSUMED_NOMINAL_VOLTAGE) if energy_wh is not None else float("nan"),
            "resistance_ohm": float("nan"),  # not reported by OCPP MeterValues
            "temperature_c":  tx.get("temperature", float("nan")),
            "test_date":      tx.get("stopTimestamp") or tx.get("startTimestamp"),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


class ModbusBMSAdapter:
    """BMSAdapter wrapping fetch_modbus_bms(). See that function's
    docstring for the register_map contract and why one is required."""

    name = "Modbus TCP"

    def __init__(self, host: str, port: int, unit_id: int, register_map: dict, cell_id: str):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.register_map = register_map or {}
        self.cell_id = cell_id

    def is_configured(self) -> bool:
        return bool(self.host and self.port and self.register_map)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        try:
            return fetch_modbus_bms(self.host, self.port, self.unit_id, self.register_map, self.cell_id)
        except Exception:
            return None

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        try:
            df = fetch_modbus_bms(self.host, self.port, self.unit_id, self.register_map, self.cell_id)
        except ImportError as e:
            return {"ok": False, "message": str(e), "n_records": None}
        except Exception as e:
            return {"ok": False, "message": f"Modbus connection failed: {e}", "n_records": None}
        if df is None or len(df) == 0:
            return {"ok": True, "message": "Connected, but no records were returned.", "n_records": 0}
        return {"ok": True, "message": f"Fetched {len(df)} record(s).", "n_records": len(df)}


class CANBusBMSAdapter:
    """BMSAdapter wrapping fetch_can_bus_bms(). See that function's
    docstring for the pgn_map contract -- the most assumption-heavy of
    this project's 3 new adapters since there's no real device or vendor
    DBC file to confirm frame layout against."""

    name = "CAN Bus"

    def __init__(self, channel: str, bustype: str, node_id: int, pgn_map: dict, cell_id: str):
        self.channel = channel
        self.bustype = bustype
        self.node_id = node_id
        self.pgn_map = pgn_map or {}
        self.cell_id = cell_id

    def is_configured(self) -> bool:
        return bool(self.channel and self.bustype and self.pgn_map)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        try:
            return fetch_can_bus_bms(self.channel, self.bustype, self.node_id, self.pgn_map, self.cell_id)
        except Exception:
            return None

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        try:
            df = fetch_can_bus_bms(self.channel, self.bustype, self.node_id, self.pgn_map, self.cell_id)
        except ImportError as e:
            return {"ok": False, "message": str(e), "n_records": None}
        except Exception as e:
            return {"ok": False, "message": f"CAN bus connection failed: {e}", "n_records": None}
        if df is None or len(df) == 0:
            return {"ok": True, "message": "Listened, but no matching frames arrived.", "n_records": 0}
        return {"ok": True, "message": f"Fetched {len(df)} record(s).", "n_records": len(df)}


class OCPPAdapter:
    """BMSAdapter wrapping fetch_ocpp_sessions(). See that function's
    docstring for why this queries a Central System's REST reporting API
    rather than speaking live OCPP, and for the charging-session-to-cycle-
    data mapping's real limitations (no resistance, capacity derived from
    energy at an assumed voltage)."""

    name = "OCPP Charge Point"

    def __init__(self, central_system_url: str, api_key: str, charge_point_id: str):
        self.central_system_url = central_system_url
        self.api_key = api_key
        self.charge_point_id = charge_point_id

    def is_configured(self) -> bool:
        return bool(self.central_system_url and self.api_key and self.charge_point_id)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        try:
            return fetch_ocpp_sessions(self.central_system_url, self.api_key, self.charge_point_id)
        except Exception:
            return None

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        try:
            df = fetch_ocpp_sessions(self.central_system_url, self.api_key, self.charge_point_id)
        except Exception as e:
            return {"ok": False, "message": f"OCPP Central System connection failed: {e}", "n_records": None}
        if df is None or len(df) == 0:
            return {"ok": True, "message": "Connected, but no sessions were returned.", "n_records": 0}
        return {"ok": True, "message": f"Fetched {len(df)} session(s).", "n_records": len(df)}


def _get_by_dot_path(data, path: str):
    """Walk a dotted path ("data.battery.capacity_ah") through nested
    dicts/lists. Returns None on any missing key/wrong type instead of
    raising -- a caller-supplied path pointing at a field a particular
    response happens not to include should degrade to NaN, not crash."""
    for part in path.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit():
            idx = int(part)
            data = data[idx] if 0 <= idx < len(data) else None
        else:
            return None
        if data is None:
            return None
    return data


def fetch_generic_rest_bms(
    base_url: str,
    auth_header_name: str,
    auth_header_value: str,
    field_map: dict,
    cell_id: str,
) -> "pd.DataFrame | None":
    """
    Fetch a single current-state snapshot from an arbitrary vendor's REST
    API, for BMS vendors not specifically coded for elsewhere in this
    module (Victron/Orion/Modbus/CAN/OCPP each target one real, named
    vendor/protocol shape; this one targets none in particular).

    field_map supplies which JSON field (dotted path, e.g.
    "data.battery.capacity_ah") carries each of the app's standard
    columns: {"capacity_ah": "...", "resistance_ohm": "...",
    "temperature_c": "..."}. A field missing from the map, or a path
    that doesn't resolve in a given response, reads as NaN rather than
    raising -- the same graceful-degradation contract every other
    adapter in this module uses.

    Returns None if base_url/field_map are not both supplied. auth_header_name/
    auth_header_value are optional (some internal/test APIs need no auth
    at all); when only one of the pair is set, no auth header is sent
    rather than sending a malformed one.

    Not tested against any live device (no specific vendor is targeted,
    so there's nothing to test against). Uses only `requests`, already a
    hard dependency for the other REST-based adapters in this module.
    """
    if not (base_url and field_map):
        return None

    import requests

    headers = {auth_header_name: auth_header_value} if (auth_header_name and auth_header_value) else {}
    resp = requests.get(base_url, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    row = {"cell_id": cell_id, "cycle_number": 0, "test_date": None}
    for field in ("capacity_ah", "resistance_ohm", "temperature_c"):
        path = field_map.get(field)
        value = _get_by_dot_path(payload, path) if path else None
        row[field] = float(value) if isinstance(value, (int, float)) else float("nan")

    return pd.DataFrame([row])


class GenericRESTBMSAdapter:
    """BMSAdapter wrapping fetch_generic_rest_bms(). See that function's
    docstring for the field_map contract. This is the adapter registered
    in src/plugin_registry.py to prove that pattern eliminates bespoke
    Settings UI code for a new integration — see that module's docstring."""

    name = "Generic REST BMS"

    def __init__(self, base_url: str, auth_header_name: str, auth_header_value: str, field_map: dict, cell_id: str):
        self.base_url = base_url
        self.auth_header_name = auth_header_name
        self.auth_header_value = auth_header_value
        self.field_map = field_map or {}
        self.cell_id = cell_id

    def is_configured(self) -> bool:
        return bool(self.base_url and self.field_map)

    def fetch(self) -> "pd.DataFrame | None":
        if not self.is_configured():
            return None
        try:
            return fetch_generic_rest_bms(self.base_url, self.auth_header_name, self.auth_header_value, self.field_map, self.cell_id)
        except Exception:
            return None

    def test_connection(self) -> dict:
        """Uniform {"ok", "message", "n_records"} result, never raises."""
        if not self.is_configured():
            return {"ok": False, "message": "Not configured.", "n_records": None}
        try:
            df = fetch_generic_rest_bms(self.base_url, self.auth_header_name, self.auth_header_value, self.field_map, self.cell_id)
        except Exception as e:
            return {"ok": False, "message": f"Connection failed: {e}", "n_records": None}
        if df is None or len(df) == 0:
            return {"ok": True, "message": "Connected, but no records were returned.", "n_records": 0}
        return {"ok": True, "message": f"Fetched {len(df)} record(s).", "n_records": len(df)}
