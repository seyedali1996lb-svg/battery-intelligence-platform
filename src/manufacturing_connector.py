"""
Manufacturing-data connector -- interface-only contract.

No real manufacturing execution system (MES) is named or targeted here.
Unlike bms_connectors.py's Victron VRM/Orion Jr2 adapters or
circunomics_adapter.py/cmms_adapter.py (each built against a real vendor's
publicly documented REST shape, even though none has a live account to
test against), no such public shape exists to build against for
manufacturing birth-certificate data -- every real MES (Siemens Opcenter,
SAP ME, Critical Manufacturing, a custom in-house system) exposes a
different, proprietary integration surface, usually only disclosed under
NDA to an actual manufacturing partner.

This module documents the *contract* a future connector would need to
satisfy -- an abstract base class with docstrings describing exactly what
each method should return and why the platform would want it -- with no
implementation at all (every method body is `raise NotImplementedError`,
no `requests` import, no network code). This mirrors the "documented
adapter pattern only, no real system named" precedent already set by
cmms_adapter.py's module docstring, one level further: cmms_adapter.py at
least implements a plausible generic REST shape, since maintenance-
ticketing APIs are broadly similar across vendors; manufacturing birth-
certificate data has no such common shape to guess at, so an honest
interface-only contract is the right amount of commitment until a real
manufacturing partner's actual API is in hand.
"""

import abc


class ManufacturingDataConnector(abc.ABC):
    """
    Contract a future manufacturing-data connector would need to
    implement. Cannot be instantiated directly (abc.ABC) -- a concrete
    subclass must implement both abstract methods before it can be used.
    """

    @abc.abstractmethod
    def fetch_production_batch(self, batch_id: str) -> "dict | None":
        """
        Fetch production-line metadata for one manufacturing batch.

        Intended return shape (once a real MES connector implements this):
            {
                "batch_id":        str,
                "line_id":         str,   # production line/cell identifier
                "manufactured_at": str,   # ISO-8601 batch production date
                "chemistry":       str,   # e.g. "LiCoO2", "LFP", "NCA"
                "qc_pass_rate":    float, # fraction of cells in this batch that passed QC, 0-1
                "n_cells":         int,   # cells produced in this batch
            }

        Should return None if the batch is not found -- callers should
        treat that as "no manufacturing record available," not an error.
        A real implementation should also never raise on network failure
        (same never-raise-on-failure convention as bms_connectors.py/
        cmms_adapter.py/circunomics_adapter.py), returning a dict with an
        "error" key instead.
        """
        raise NotImplementedError(
            "No manufacturing execution system is connected. This method documents the "
            "intended contract only -- see this class's docstring and manufacturing_"
            "connector.py's module docstring for why no implementation exists yet."
        )

    @abc.abstractmethod
    def fetch_cell_birth_certificate(self, cell_id: str) -> "dict | None":
        """
        Fetch the manufacturing "birth certificate" for one individual
        cell -- the per-cell record this platform would use to enrich a
        cell's EU Battery Passport (see src/passport_export.py) with real
        provenance instead of inferring chemistry/initial capacity from
        cycling data alone.

        Intended return shape (once a real MES connector implements this):
            {
                "cell_id":              str,
                "batch_id":             str,   # links to fetch_production_batch()
                "manufacturing_date":   str,   # ISO-8601
                "initial_capacity_ah":  float, # factory-measured, not cycling-inferred
                "qc_pass":              bool,
                "qc_notes":             str,
            }

        Should return None if the cell has no manufacturing record (e.g.
        a NASA/Severson/Oxford research cell manufactured long before this
        platform existed) -- callers must treat this as "provenance
        unavailable," never fabricate a birth certificate, and fall back
        to today's cycling-data-inferred fields.
        """
        raise NotImplementedError(
            "No manufacturing execution system is connected. This method documents the "
            "intended contract only -- see this class's docstring and manufacturing_"
            "connector.py's module docstring for why no implementation exists yet."
        )
