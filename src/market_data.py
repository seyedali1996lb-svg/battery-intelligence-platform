"""
Pluggable electricity market data adapters — prices and carbon intensity for
the Lifecycle Intelligence layer (health-aware dispatch, grid-services revenue,
managed charging, dynamic-LCA upgrade).

This is the P1 counterpart to bms_connectors.py's BMSAdapter: the same
object-level Protocol pattern, but for *market* telemetry instead of battery
telemetry. A dispatcher/valuation feature consumes a `MarketDataAdapter`
instead of hardcoding one price source, so the same health-aware dispatch
engine runs on:

  - "synthetic"  — deterministic, offline, always available (the demo/
                   test default; a two-peak daily price shape with a
                   controllable price-spike day, derived from real market
                   behavior qualitatively but from no live feed).
  - "eia"        — EIA Open Data API (api.eia.gov/v2) hourly electricity
                   prices for a balancing authority. Requires a free EIA
                   API key.
  - "entsoe"     — ENTSO-E Transparency Platform day-ahead prices for a
                   bidding zone. Requires a free ENTSO-E REST API key.
                   ENTSO-E's API returns XML (not JSON), parsed here with
                   only the stdlib.

Grid Status (gridstatus.io) is the cohort-adjacent commercial feed this
adapter was named for in the roadmap; its pricing/data is behind its own
subscription with no public API key flow, so it is deliberately NOT
implemented here — `register_market_adapter()` makes adding it later a
one-line registration once an account exists.

Contracts (same as this project's other external adapters, see
src/adapter_contract.py and src/pvgis_client.py):
  - Every `fetch_*` returns None when the adapter is not configured
    (missing API key), never raising for that case.
  - Returns {"error": str} on request failure — never raises.
  - Returns a result dict on success.
  - `MarketDataAdapter` is a typing.Protocol, like `BMSAdapter` — any
    object with the right members satisfies it; no inheritance required.

Honest scope, stated plainly: like the BMS connectors, no live EIA /
ENTSO-E account exists for this project to test against. The concrete
adapter request/response handling is built against each vendor's publicly
documented API shape but is NOT a verified integration; the synthetic
adapter is the tested, always-available path.

Carbon intensity: GRID_CARBON_INTENSITY (static regional averages) lives in
src/dynamic_circularity.py and stays the single source of truth for the
static table — this module imports it rather than duplicating it.
`resolve_carbon_intensity()` prefers a live per-hour feed from a configured
adapter when one exists, and falls back to the static table otherwise.
"""

from __future__ import annotations

import datetime as _dt
import random as _random
import xml.etree.ElementTree as _ET
from typing import Optional, Protocol, runtime_checkable

import requests

from dynamic_circularity import GRID_CARBON_INTENSITY  # single source of truth


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Structural contract every market-data connector in this module
    satisfies — the market-side sibling of bms_connectors.BMSAdapter.

    A concrete adapter is a stateful instance holding its credentials
    (e.g. an API key), with:

      name: str                    — human-readable feed name
      is_configured() -> bool      — True iff a fetch can be attempted
      fetch_hourly_prices(start, end) -> dict | None
      fetch_carbon_intensity(start, end) -> dict | None
    """

    name: str

    def is_configured(self) -> bool:
        """True if this adapter has everything it needs to attempt a fetch,
        without making any network call."""
        ...

    def fetch_hourly_prices(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        """Hourly price series for [start, end) (ISO-8601 strings, or a
        sensible adapter default when None). Returns None when not
        configured, {"error": str} on request failure, or
        {"adapter", "unit", "start", "hours", "prices"} on success.
        The `unit` field is declared honestly per feed ("EUR/kWh" or
        "USD/kWh"); use to_eur_per_kwh() before mixing feeds."""
        ...

    def fetch_carbon_intensity(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        """Hourly grid carbon intensity (g CO2e/kWh) for [start, end), same
        None / {"error"} / result contract. Adapters whose feed does not
        carry carbon data return {"error": ...} so resolve_carbon_intensity()
        can fall back to the static table."""
        ...


# ---------------------------------------------------------------------------
# Synthetic adapter (offline/demo default — the tested path)
# ---------------------------------------------------------------------------

def _synthetic_prices(n_hours: int, base_eur: float, amplitude_eur: float, seed: int) -> list:
    """Deterministic daily two-peak price shape (a qualitative, not fitted,
    model of real day-ahead curves: night trough, morning peak, midday
    shoulder, evening peak). Same seed -> same series, so tests and demo
    runs are reproducible."""
    rng = _random.Random(seed)
    prices = []
    for h in range(n_hours):
        hour_of_day = h % 24
        # Night trough (23-06), morning peak (07-10), shoulder (11-16),
        # evening peak (17-21), late-evening drop (22).
        if 23 <= hour_of_day or hour_of_day <= 6:
            shape = 0.2
        elif 7 <= hour_of_day <= 10:
            shape = 1.0
        elif 11 <= hour_of_day <= 16:
            shape = 0.55
        elif 17 <= hour_of_day <= 21:
            shape = 0.9
        else:
            shape = 0.4
        # Deterministic day-to-day jitter (same seed -> same jitter), so the
        # series is reproducible but not perfectly periodic.
        jitter = 0.92 + 0.16 * rng.random()
        prices.append(round(base_eur + amplitude_eur * shape * jitter, 4))
    return prices


class SyntheticMarketAdapter:
    """Deterministic, offline price feed — the default and the only adapter
    fully tested (no API key, no network). base_eur/amplitude_eur set the
    daily mean and swing; a configurable `spike_hour` injects a price spike
    (used by tests to verify the dispatch heuristic reacts to arbitrage
    opportunities)."""

    name = "Synthetic"

    def __init__(
        self,
        base_eur: float = 0.10,
        amplitude_eur: float = 0.08,
        seed: int = 7,
        spike_hour: "int | None" = None,
        spike_price_eur: float = 0.45,
    ):
        self.base_eur = base_eur
        self.amplitude_eur = amplitude_eur
        self.seed = seed
        self.spike_hour = spike_hour
        self.spike_price_eur = spike_price_eur

    def is_configured(self) -> bool:
        return True

    def fetch_hourly_prices(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        n_hours = _resolve_hours(start, end)
        prices = _synthetic_prices(n_hours, self.base_eur, self.amplitude_eur, self.seed)
        if self.spike_hour is not None and n_hours > self.spike_hour:
            prices[self.spike_hour] = self.spike_price_eur
        return {
            "adapter": self.name,
            "unit": "EUR/kWh",
            "start": _default_start_iso(),
            "hours": n_hours,
            "prices": prices,
        }

    def fetch_carbon_intensity(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        n_hours = _resolve_hours(start, end)
        # Deterministic daily carbon shape around a mean — a proxy for
        # "cleaner at night when wind is up", NOT a live feed. The mean
        # defaults to the EU average static value so resolve_carbon_intensity()
        # and the static table agree on average.
        rng = _random.Random(self.seed * 3)
        mean = GRID_CARBON_INTENSITY["EU_AVG"]
        series = []
        for h in range(n_hours):
            hour_of_day = h % 24
            night_factor = 0.85 if (hour_of_day >= 22 or hour_of_day <= 5) else 1.05
            series.append(round(mean * night_factor * (0.95 + 0.10 * rng.random()), 1))
        return {
            "adapter": self.name,
            "unit": "g CO2e/kWh",
            "start": _default_start_iso(),
            "hours": n_hours,
            "series": series,
        }


# ---------------------------------------------------------------------------
# EIA Open Data API (api.eia.gov/v2)
# ---------------------------------------------------------------------------

EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# Common EIA balancing-authority respondents. Not exhaustive — the EIA API
# accepts any registered respondent code.
EIA_DEFAULT_RESPONDENT = "PJM"

EIA_RESPONDENTS = {
    "PJM": "PJM Interconnection (US Mid-Atlantic)",
    "CAISO": "California ISO",
    "ERCO": "ERCOT (Texas)",
    "NYISO": "New York ISO",
    "ISNE": "ISO New England",
    "MISO": "Midcontinent ISO",
}


class EIAAdapter:
    """EIA Open Data API hourly wholesale electricity prices for one
    balancing authority (respondent). Built against the publicly documented
    v2 shape: GET /electricity/rto/region-data/data/?frequency=hourly&data[0]=value
    &facets[respondent][]=PJM&start=..&end=..&api_key=.. returns
    {"response": {"data": [{"period": "...", "value": ..}, ...]}} with value
    in $/MWh. Not verified against a live account (no API key exists for
    this project) — see module docstring.

    Prices are returned in USD/kWh (converted from the API's $/MWh);
    use to_eur_per_kwh() (with its documented FX assumption) before mixing
    with EUR-denominated feeds/assumptions."""

    name = "EIA Open Data"

    def __init__(self, api_key: str, respondent: str = EIA_DEFAULT_RESPONDENT):
        self.api_key = api_key
        self.respondent = respondent

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch_hourly_prices(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        if not self.is_configured():
            return None

        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": self.respondent,
            "api_key": self.api_key,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        try:
            resp = requests.get(EIA_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            records = resp.json()["response"]["data"]
        except Exception as e:
            return {"error": f"EIA request failed: {e}"}

        prices = []
        for rec in records:
            value = rec.get("value")
            if value is None:
                continue
            # EIA reports $/MWh; convert to $/kWh and keep the unit honest.
            prices.append(round(float(value) / 1000.0, 5))

        if not prices:
            return {"error": f"EIA returned no price records for respondent {self.respondent!r}."}

        return {
            "adapter": self.name,
            "unit": "USD/kWh",
            "start": records[0].get("period"),
            "hours": len(prices),
            "prices": prices,
            "respondent": self.respondent,
            "respondent_name": EIA_RESPONDENTS.get(self.respondent, self.respondent),
        }

    def fetch_carbon_intensity(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        if not self.is_configured():
            return None
        # The EIA v2 electricity API series used here (rto/region-data) does
        # not carry a CO2 emission-rate field; the separate emissions data
        # product is out of scope for this adapter. Honest explicit
        # non-support lets resolve_carbon_intensity() fall back cleanly.
        return {"error": "EIA rto/region-data feed does not expose carbon intensity."}


# ---------------------------------------------------------------------------
# ENTSO-E Transparency Platform (web-api.tp.entsoe.eu)
# ---------------------------------------------------------------------------

ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"

# Day-ahead price (documentType A44) per bidding zone. A few common zones;
# the ENTSO-E API accepts any registered zone code.
ENTSOE_BIDDING_ZONES = {
    "DE_LU": "Germany/Luxembourg",
    "FR": "France",
    "GB": "Great Britain",
    "NL": "Netherlands",
    "SE": "Sweden",
    "IT": "Italy",
    "NO": "Norway",
    "DK": "Denmark",
    "ES": "Spain",
}


class ENTSOEAdapter:
    """ENTSO-E Transparency Platform day-ahead prices for one bidding zone.
    The API is XML (not JSON); this adapter parses it with the stdlib only.
    Built against the documented shape: GET /api?securityToken=<key>
    &documentType=A44&in_Domain=<zone>&out_Domain=<zone>&periodStart=YYYYMMDDHH00
    &periodEnd=YYYYMMDDHH00 returns an XML `TimeSeries` whose
    `Period/Point/position`/`price.amount` pairs carry hourly €/MWh prices.
    Not verified against a live account — see module docstring.

    Prices are returned in EUR/kWh (converted from the API's €/MWh)."""

    name = "ENTSO-E Transparency"

    def __init__(self, api_key: str, bidding_zone: str = "DE_LU"):
        self.api_key = api_key
        self.bidding_zone = bidding_zone

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _domain(self) -> str:
        # ENTSO-E day-ahead prices use the zone's own EIC/domain code, which
        # differs per zone and is not derivable from the human-readable key.
        # Rather than fabricate codes, the caller supplies the real domain
        # code; the human-readable zone map is only for display.
        return self.bidding_zone

    def fetch_hourly_prices(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        if not self.is_configured():
            return None

        # ENTSO-E period params are YYYYMMDDHH00 with local-time semantics
        # per zone; default to a 48h window starting now (UTC) so an
        # unparameterized call still fetches something.
        period_start, period_end = _entsoe_period(start, end)

        params = {
            "securityToken": self.api_key,
            "documentType": "A44",
            "in_Domain": self._domain(),
            "out_Domain": self._domain(),
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        try:
            resp = requests.get(ENTSOE_API_URL, params=params, timeout=20)
            resp.raise_for_status()
            prices = _parse_entsoe_prices(resp.content)
        except Exception as e:
            return {"error": f"ENTSO-E request failed: {e}"}

        if not prices:
            return {"error": f"ENTSO-E returned no price points for domain {self._domain()!r}."}

        return {
            "adapter": self.name,
            "unit": "EUR/kWh",
            "start": period_start,
            "hours": len(prices),
            "prices": prices,
            "bidding_zone": self.bidding_zone,
            "bidding_zone_name": ENTSOE_BIDDING_ZONES.get(self.bidding_zone, self.bidding_zone),
        }

    def fetch_carbon_intensity(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "dict | None":
        if not self.is_configured():
            return None
        return {"error": "ENTSO-E day-ahead price feed does not expose carbon intensity."}


def _parse_entsoe_prices(xml_bytes: bytes) -> list:
    """Extract hourly €/MWh prices from an ENTSO-E A44 XML response,
    converting to €/kWh. Handles the standard TimeSeries/Period/Point
    nesting; returns [] on any malformed shape (caller reports it)."""
    root = _ET.fromstring(xml_bytes)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}
    prices = []
    for ts in root.findall(".//ns:TimeSeries", ns):
        period = ts.find("ns:Period", ns)
        if period is None:
            continue
        for point in period.findall("ns:Point", ns):
            amount = point.find("ns:price.amount", ns)
            if amount is None or amount.text is None:
                continue
            try:
                prices.append(round(float(amount.text) / 1000.0, 5))
            except ValueError:
                continue
    return prices


def _entsoe_period(start: Optional[str], end: Optional[str]) -> tuple:
    now = _dt.datetime.now(_dt.timezone.utc)
    if start:
        period_start = start
    else:
        period_start = (now - _dt.timedelta(hours=24)).strftime("%Y%m%d%H00")
    if end:
        period_end = end
    else:
        period_end = (now + _dt.timedelta(hours=24)).strftime("%Y%m%d%H00")
    return period_start, period_end


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_start_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="minutes")


def _resolve_hours(start: Optional[str], end: Optional[str]) -> int:
    if start and end:
        try:
            s = _dt.datetime.fromisoformat(start)
            e = _dt.datetime.fromisoformat(end)
            hours = int((e - s).total_seconds() // 3600)
            if hours > 0:
                return hours
        except ValueError:
            pass
    return 48


# Documented FX assumption, kept separate from any feed so it is one place
# to audit (see the ASSUMPTIONS convention in src/consequences.py).
USD_TO_EUR = 0.92  # "Illustrative — not sourced": mid-2020s EUR/USD level


def to_eur_per_kwh(result: dict) -> dict:
    """Normalize a fetch_hourly_prices() result to EUR/kWh so downstream
    engines (dispatch, revenue, charging) can consume any feed uniformly.
    EUR-denominated results pass through unchanged; USD results are
    converted at USD_TO_EUR, with the assumption recorded in the returned
    dict's `fx_assumption` field. A result dict already in EUR keeps its
    unit and gains no fx_assumption."""
    if result.get("unit") == "EUR/kWh":
        return dict(result)
    if result.get("unit") == "USD/kWh":
        out = dict(result)
        out["unit"] = "EUR/kWh"
        out["prices"] = [round(p * USD_TO_EUR, 5) for p in result["prices"]]
        out["fx_assumption"] = {
            "usd_to_eur": USD_TO_EUR,
            "label": "Illustrative — not sourced",
            "note": "Mid-2020s EUR/USD level; not a live FX feed.",
        }
        return out
    raise ValueError(f"Unknown price unit {result.get('unit')!r} in market data result.")


def resolve_carbon_intensity(
    region: str = "EU_AVG",
    adapter: "MarketDataAdapter | None" = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Best available grid carbon intensity: a live per-hour series from
    `adapter` when the adapter is configured AND returns data, otherwise the
    static regional table (src.dynamic_circularity.GRID_CARBON_INTENSITY,
    IEA/EEA-based). Returns {"source": "live"|"static", "g_co2_per_kwh":
    float (window mean for live), "per_hour": [...]|None, "region": ...}."""
    if adapter is not None and adapter.is_configured():
        live = adapter.fetch_carbon_intensity(start=start, end=end)
        if live and "error" not in live and live.get("series"):
            series = live["series"]
            return {
                "source": "live",
                "feed": live.get("adapter", adapter.name),
                "g_co2_per_kwh": round(float(sum(series)) / len(series), 1),
                "per_hour": series,
                "region": region,
            }
    static = GRID_CARBON_INTENSITY.get(region.upper(), GRID_CARBON_INTENSITY["EU_AVG"])
    return {
        "source": "static",
        "g_co2_per_kwh": float(static),
        "per_hour": None,
        "region": region,
        "note": "Static regional average (IEA/EEA). No live feed configured — "
                "configure a MarketDataAdapter to upgrade dynamic-LCA to live intensity.",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _default_adapters() -> dict:
    """The built-in adapter registry. `synthetic` is the tested, always-
    available default; `eia`/`entsoe` are registered with empty credentials
    so the API can resolve them and report an honest "not configured" state
    until a caller registers a keyed instance (see register_market_adapter
    and the Settings-style plugin pattern in src/plugin_registry.py)."""
    return {
        "synthetic": SyntheticMarketAdapter(),
        "eia": EIAAdapter(""),
        "entsoe": ENTSOEAdapter(""),
    }


_MARKET_ADAPTERS: dict = {}


def register_market_adapter(name: str, adapter: "MarketDataAdapter") -> None:
    """Register a MarketDataAdapter under a short key. Any object satisfying
    the MarketDataAdapter Protocol is accepted (no inheritance required) —
    the same plugin pattern src/plugin_registry.py proves for BMS adapters.
    Registration is additive: calling this twice with the same name replaces
    the earlier adapter."""
    _MARKET_ADAPTERS[name] = adapter


def get_market_adapter(name: str) -> "MarketDataAdapter":
    """Look up a registered adapter by key, falling back to the built-ins.
    Raises KeyError with the available keys for an unknown name — callers
    should catch it and show an empty state (same convention as
    bms_connectors' credential guards)."""
    if name in _MARKET_ADAPTERS:
        return _MARKET_ADAPTERS[name]
    builtins = _default_adapters()
    if name in builtins:
        return builtins[name]
    raise KeyError(
        f"Unknown market adapter {name!r}. Registered/built-in: {sorted(list(_MARKET_ADAPTERS) + list(builtins))}"
    )


def registered_market_adapters() -> list[str]:
    """All adapter keys get_market_adapter() can resolve."""
    return sorted(list(_MARKET_ADAPTERS) + list(_default_adapters()))


def make_eia_adapter(api_key: str, respondent: str = EIA_DEFAULT_RESPONDENT) -> EIAAdapter:
    """Convenience factory so callers don't need to import EIAAdapter and
    remember its default respondent separately."""
    return EIAAdapter(api_key=api_key, respondent=respondent)


def make_entsoe_adapter(api_key: str, bidding_zone: str = "DE_LU") -> ENTSOEAdapter:
    """Convenience factory, symmetric with make_eia_adapter()."""
    return ENTSOEAdapter(api_key=api_key, bidding_zone=bidding_zone)
