"""Unit tests for src/manufacturing_connector.py -- confirms this really is
an interface-only contract (no real system named, no implementation), the
same "documented adapter pattern only" precedent cmms_adapter.py set, one
level further: there is no concrete implementation at all here, only an
abstract contract a future connector would need to satisfy."""

import inspect

import pytest

from manufacturing_connector import ManufacturingDataConnector


def test_cannot_instantiate_directly():
    """abc.ABC with unimplemented abstractmethods must refuse
    instantiation -- proves this is a contract, not a usable connector."""
    with pytest.raises(TypeError):
        ManufacturingDataConnector()


def test_no_network_library_imported():
    """Unlike bms_connectors.py/cmms_adapter.py/circunomics_adapter.py
    (each of which imports requests inside its fetch/create function),
    this module must contain zero network code -- it's a docstring-only
    contract, not even a generic-REST-shape guess."""
    import manufacturing_connector
    source = inspect.getsource(manufacturing_connector)
    assert "import requests" not in source
    assert "requests." not in source


def test_abstract_methods_raise_not_implemented_when_called_via_subclass():
    """A minimal subclass that implements both abstract methods can be
    instantiated; calling super() from within it still hits the
    documented NotImplementedError body, proving the base class itself
    contributes no real behavior -- only the two subclass overrides do."""

    class _StubConnector(ManufacturingDataConnector):
        def fetch_production_batch(self, batch_id):
            return super().fetch_production_batch(batch_id)

        def fetch_cell_birth_certificate(self, cell_id):
            return super().fetch_cell_birth_certificate(cell_id)

    stub = _StubConnector()  # must not raise -- both abstract methods are implemented

    with pytest.raises(NotImplementedError):
        stub.fetch_production_batch("BATCH-1")
    with pytest.raises(NotImplementedError):
        stub.fetch_cell_birth_certificate("CELL-1")


def test_real_subclass_can_return_documented_shape():
    """A concrete implementation is free to actually return data -- the
    ABC only enforces that both methods exist, not that they must raise.
    Confirms the contract is usable once a real MES connector is written,
    not just theoretically abstract."""

    class _RealConnector(ManufacturingDataConnector):
        def fetch_production_batch(self, batch_id):
            return {"batch_id": batch_id, "line_id": "L1", "manufactured_at": "2026-01-01",
                     "chemistry": "LFP", "qc_pass_rate": 0.98, "n_cells": 500}

        def fetch_cell_birth_certificate(self, cell_id):
            return None  # e.g. a research cell with no manufacturing record

    connector = _RealConnector()
    batch = connector.fetch_production_batch("BATCH-1")
    assert batch["batch_id"] == "BATCH-1"
    assert connector.fetch_cell_birth_certificate("B0005") is None
