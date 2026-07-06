"""Unit tests for src/cmms_adapter.py's guard-clause / payload-shaping logic.

No live CMMS/ERP account exists to test against (see module docstring) —
these tests cover the pure payload builder and the unconfigured/network-
failure guard clauses, matching the style used for the other adapters in
this codebase (bms_connectors.py, circunomics_adapter.py).
"""

from cmms_adapter import build_ticket_payload, create_maintenance_ticket


def test_build_ticket_payload_shape():
    payload = build_ticket_payload("B0005", "Replace Now — B0005", "SOH 62.3%", "high")
    assert payload["asset_id"] == "B0005"
    assert payload["title"] == "Replace Now — B0005"
    assert payload["description"] == "SOH 62.3%"
    assert payload["priority"] == "high"
    assert "created_at" in payload
    assert payload["source_system"] == "battery-intelligence-platform"


def test_create_maintenance_ticket_returns_none_without_api_key():
    result = create_maintenance_ticket("B0005", "title", "desc", "medium", api_key="")
    assert result is None


def test_create_maintenance_ticket_returns_error_dict_on_network_failure():
    # No real endpoint at this host — request should fail, and the adapter
    # must catch it and return an error dict rather than raising.
    result = create_maintenance_ticket(
        "B0005", "title", "desc", "medium", api_key="fake-key",
        api_base_url="https://this-host-does-not-exist.invalid/v1",
    )
    assert result is not None
    assert "error" in result
