"""
Dependency-injection protocols for Battery Intelligence Platform src/ modules.

These Protocol classes define the *contracts* between subsystems so that:
  - Callers depend on an interface, not a concrete implementation.
  - Tests can substitute lightweight fakes without touching the real module.
  - A future refactor (e.g. swapping SQLite for Postgres) only needs one
    conforming class, not changes at every call site.

Usage::

    from protocols import BMSAdapter, MarketDataAdapter, DataStore

    def fetch_telemetry(adapter: BMSAdapter) -> pd.DataFrame:
        return adapter.fetch_cycles(cell_id="B0005")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# BMS / hardware adapters
# ---------------------------------------------------------------------------

@runtime_checkable
class BMSAdapter(Protocol):
    """Interface for Battery Management System data adapters.

    Every concrete adapter (Victron VRM, Orion Jr2, Modbus TCP, etc.)
    implements this protocol.  The adapter is responsible for:
      - Authenticating with the target system.
      - Fetching raw telemetry.
      - Reshaping it into the platform's standard cycle-data schema.
    """

    def fetch_cycles(
        self, cell_id: str, start: int | None = None, end: int | None = None
    ) -> pd.DataFrame:
        """Return a DataFrame in the platform's standard cycle schema."""
        ...

    def health_check(self) -> bool:
        """Return True if the adapter can reach its backend."""
        ...


# ---------------------------------------------------------------------------
# Market data adapters
# ---------------------------------------------------------------------------

@runtime_checkable
class MarketDataAdapter(Protocol):
    """Interface for electricity market price data sources.

    Concrete implementations: SyntheticMarketAdapter, EIAMarketAdapter,
   ENTSOMEMarketAdapter.
    """

    def fetch_prices(
        self, start: str, end: str, *, market: str = "day_ahead"
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: timestamp, price_per_kwh, currency."""
        ...

    @property
    def is_configured(self) -> bool:
        """True when the adapter has valid credentials / is reachable."""
        ...


# ---------------------------------------------------------------------------
# Data store abstraction
# ---------------------------------------------------------------------------

@runtime_checkable
class DataStore(Protocol):
    """Minimal interface for the persistence layer used by most src/ modules.

    This lets modules like recommendations, action_center, and
    marketplace_matching depend on a store interface rather than
    importing db.py directly.
    """

    def get_setting(self, org_id: int, key: str) -> Any:
        ...

    def set_setting(self, org_id: int, key: str, value: Any) -> None:
        ...

    def save_decision(self, org_id: int, decision: dict, *, caller_role: str = "admin") -> None:
        ...

    def load_decisions(self, org_id: int) -> list[dict]:
        ...


# ---------------------------------------------------------------------------
# Import / upload adapters
# ---------------------------------------------------------------------------

@runtime_checkable
class ImportAdapter(Protocol):
    """Interface for external-data import adapters (Circunomics, CMMS, etc.)."""

    def is_configured(self, org_id: int) -> bool:
        """Return True when the adapter's credentials are present in settings."""
        ...

    def submit(self, org_id: int, data: dict) -> dict:
        """Submit data to the external system; return a result dict."""
        ...


# ---------------------------------------------------------------------------
# Notification / webhook sender
# ---------------------------------------------------------------------------

@runtime_checkable
class NotificationSender(Protocol):
    """Interface for sending alerts (webhooks, email, etc.)."""

    def send(self, url: str, payload: dict, *, secret: str | None = None) -> bool:
        """Send a notification; return True on success."""
        ...


# ---------------------------------------------------------------------------
# Knowledge / retrieval
# ---------------------------------------------------------------------------

@runtime_checkable
class KnowledgeRetriever(Protocol):
    """Interface for the Copilot's knowledge-retrieval backend."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top-k relevant documents for a query."""
        ...

    def answer(self, query: str, *, context: list[dict] | None = None) -> str:
        """Generate a grounded answer from the knowledge base."""
        ...
