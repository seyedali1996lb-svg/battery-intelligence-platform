"""
Adapter plugin registry.

Every one of this platform's 6 BMS-fetch adapters (`src/bms_connectors.py`)
satisfies the same `BMSAdapter` Protocol (`name`/`is_configured()`/`fetch()`),
but until now, adding a new one meant hand-writing a whole new
`render_bms_connector_X()` function in `app/_pages/_settings_config.py` —
new `st.text_input()`/`st.text_area()` calls, a new `db.set_setting()`
loop, a new empty-state guard, a new "Test connection" button — every
time, even though that shape is close to identical across all 6
(confirmed by reading all 6 render functions before writing this module:
the only real variation is which fields exist and whether one of them is
a JSON blob).

`ADAPTER_REGISTRY` names that variation as data (`AdapterField` specs)
instead of code. `render_adapter_settings()` (in
`app/_pages/_settings_config.py`) reads any registered entry and renders
a working Settings section — widgets, persistence, empty state, test
button — with zero adapter-specific UI code.

**This does not retrofit the 5 pre-existing hand-written sections**
(Victron/Orion/Modbus/CAN/OCPP) — they already work, are already tested,
and rewriting working UI to prove a pattern would be pure churn with real
regression risk for no user-facing benefit. Their registry entries below
are metadata only (`rendered_via_registry=False`) — real, so
`ADAPTER_REGISTRY` is a genuinely complete listing of every BMS
integration this platform has, but not wired to the generic renderer.

**What actually proves the pattern eliminates bespoke code**: the 6th
adapter, `GenericRESTBMSAdapter` (`src/bms_connectors.py`) — a real,
new integration added with ONLY a registry entry below and zero new
Streamlit widget code. Any future adapter that fits the same
text/password/textarea-JSON field shape can be added the same way; one
that genuinely needs something the field-spec vocabulary below can't
express (a multi-step OAuth flow, for instance) should still get its
own hand-written function, same as the original 5 — this registry is a
tool for the common case, not a mandate for every future integration.
"""

from dataclasses import dataclass, field


@dataclass
class AdapterField:
    """One configuration input a generic Settings renderer can draw
    without adapter-specific code. `key` is the src/db.py setting key
    this field persists to."""
    key: str
    label: str
    kind: str = "text"          # "text" | "password" | "textarea_json" | "number"
    default: str = ""
    secret: bool = False        # must also be added to db._SECRET_SETTING_KEYS
    required: bool = True       # must be filled before the "Test connection" button appears
    help: "str | None" = None
    placeholder: "str | None" = None


@dataclass
class AdapterMeta:
    key: str                     # registry key, e.g. "generic_rest_bms"
    name: str                    # display name shown in Settings/an integrations overview
    kind: str                    # "bms" | "write_back"
    description: str
    adapter_class: type
    fields: list = field(default_factory=list)
    build_kwargs: "callable | None" = None   # dict[str, str] -> dict of adapter_class(**kwargs)
    rendered_via_registry: bool = False       # False = hand-written UI still owns this one (documentation-only entry)
    live_tested: bool = False


ADAPTER_REGISTRY: "dict[str, AdapterMeta]" = {}


def register_adapter(meta: AdapterMeta) -> None:
    if meta.key in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter {meta.key!r} is already registered.")
    ADAPTER_REGISTRY[meta.key] = meta


def _register_builtin_adapters() -> None:
    """Registers all 6 real BMS adapters' metadata. Called once at import
    time — Python only executes a module's top-level code once per
    process even if imported from multiple places, so no re-entrancy
    guard is needed here."""
    from bms_connectors import (
        VictronVRMAdapter, OrionBMSAdapter, ModbusBMSAdapter, CANBusBMSAdapter,
        OCPPAdapter, GenericRESTBMSAdapter,
    )

    register_adapter(AdapterMeta(
        key="victron_vrm", name="Victron VRM", kind="bms",
        description="Pulls real cycle data from a Victron VRM installation.",
        adapter_class=VictronVRMAdapter, rendered_via_registry=False,
    ))
    register_adapter(AdapterMeta(
        key="orion_bms", name="Orion Jr2", kind="bms",
        description="Pulls real cycle data from an Orion Jr2 BMS's REST gateway.",
        adapter_class=OrionBMSAdapter, rendered_via_registry=False,
    ))
    register_adapter(AdapterMeta(
        key="modbus_bms", name="Modbus TCP", kind="bms",
        description="Reads a live snapshot from a Modbus TCP BMS/inverter via a caller-supplied register map.",
        adapter_class=ModbusBMSAdapter, rendered_via_registry=False,
    ))
    register_adapter(AdapterMeta(
        key="can_bus_bms", name="CAN Bus", kind="bms",
        description="Listens for battery telemetry on a CAN bus via a caller-supplied PGN/CAN-ID map.",
        adapter_class=CANBusBMSAdapter, rendered_via_registry=False,
    ))
    register_adapter(AdapterMeta(
        key="ocpp", name="OCPP Charge Point", kind="bms",
        description="Pulls completed charging-session records from an OCPP Central System's REST reporting API.",
        adapter_class=OCPPAdapter, rendered_via_registry=False,
    ))

    register_adapter(AdapterMeta(
        key="generic_rest_bms", name="Generic REST BMS", kind="bms",
        description=(
            "For a BMS vendor with a proprietary REST API not specifically supported above — "
            "supply the endpoint, optional auth header, and a JSON field map (dotted paths) for "
            "capacity/resistance/temperature."
        ),
        adapter_class=GenericRESTBMSAdapter,
        fields=[
            AdapterField("generic_rest_base_url", "API endpoint URL", kind="text",
                         placeholder="https://api.example-bms.com/v1/cells/CELL-1"),
            AdapterField("generic_rest_auth_header_name", "Auth header name (optional)", kind="text",
                         placeholder="Authorization", required=False),
            AdapterField("generic_rest_auth_header_value", "Auth header value (optional)", kind="password", secret=True, required=False),
            AdapterField("generic_rest_cell_id", "Cell/pack ID to record readings under", kind="text"),
            AdapterField(
                "generic_rest_field_map", "Field map (JSON)", kind="textarea_json",
                placeholder='{"capacity_ah": "data.battery.capacity_ah", "resistance_ohm": "data.battery.resistance_ohm"}',
                help="One entry per field, a dotted JSON path into the API's response. Omit a field the API doesn't expose.",
            ),
        ],
        build_kwargs=lambda v: dict(
            base_url=v.get("generic_rest_base_url", ""),
            auth_header_name=v.get("generic_rest_auth_header_name", ""),
            auth_header_value=v.get("generic_rest_auth_header_value", ""),
            field_map=v.get("generic_rest_field_map") or {},
            cell_id=v.get("generic_rest_cell_id", ""),
        ),
        rendered_via_registry=True,
    ))


_register_builtin_adapters()
