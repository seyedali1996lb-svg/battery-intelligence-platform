"""Unit tests for src/adapter_contract.py — classify_result().

Also verifies the real circunomics_adapter.py/cmms_adapter.py write-back
functions actually produce results classify_result() handles correctly —
not just synthetic dicts — since the whole point of naming this contract
was that both modules already converged on it independently.
"""

from adapter_contract import classify_result
from circunomics_adapter import list_cell_on_circunomics
from cmms_adapter import create_maintenance_ticket


def test_classify_result_none_is_unconfigured():
    assert classify_result(None) == "unconfigured"


def test_classify_result_error_dict_is_error():
    assert classify_result({"error": "timeout"}) == "error"


def test_classify_result_other_dict_is_success():
    assert classify_result({"id": "12345", "status": "created"}) == "success"


def test_classify_result_empty_dict_is_success():
    # An empty-but-real response body is still a successful call, not an error.
    assert classify_result({}) == "success"


def test_circunomics_unconfigured_result_classifies_as_unconfigured():
    result = list_cell_on_circunomics("TEST", 80.0, "LiCoO2", 2.0, 50.0, api_key="")
    assert classify_result(result) == "unconfigured"


def test_cmms_unconfigured_result_classifies_as_unconfigured():
    result = create_maintenance_ticket("TEST", "title", "desc", "low", api_key="")
    assert classify_result(result) == "unconfigured"


def test_cmms_network_failure_classifies_as_error():
    """A configured call against an unreachable/fake endpoint must classify
    as "error", not raise — exercising the real never-raises contract, not
    just a synthetic {"error": ...} dict. Same reserved .invalid TLD
    convention (RFC 2606 -- guaranteed to never resolve) already used by
    test_cmms_adapter.py's own network-failure test."""
    result = create_maintenance_ticket(
        "TEST", "title", "desc", "low", api_key="fake-key",
        api_base_url="https://this-host-does-not-exist.invalid/v1",
    )
    assert classify_result(result) == "error"
