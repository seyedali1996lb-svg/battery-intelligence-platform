"""
Shared contract for this platform's "write-back" external adapters.

circunomics_adapter.py's list_cell_on_circunomics() and cmms_adapter.py's
create_maintenance_ticket() independently converged on the exact same
result shape (each documented it in its own module docstring) without
either being written down as one checkable contract. This module names
that contract explicitly, so the next write-back connector (a future ERP,
recycling-partner, or telemetry-webhook target) implements it by
construction instead of by copying the pattern from whichever module was
open — and gives every caller one shared helper instead of re-deriving the
same branch, which was found duplicated, identically, across 4 call sites
(app/_pages/decision.py's Circunomics listing + CMMS ticket buttons,
app/_pages/_settings_config.py's Circunomics + CMMS "Test connection"
buttons) before this module existed.

Contract every write-back adapter function in this project follows —
a single call, `do_thing(..., api_key: str, api_base_url: str = "...") ->
dict | None`:
  - Returns None when not configured (e.g. no API key supplied). Never
    raises for this case.
  - Returns a dict with an "error" key (str) on request failure (non-2xx
    response, network error, timeout). Never raises for this case either —
    callers must not need their own try/except.
  - Returns the parsed JSON response dict (whatever shape the target
    system's API returns) on success.

This is a *function*-level contract for one-shot actions (submit a
listing, create a ticket). It's deliberately distinct from
bms_connectors.py's BMSAdapter — an *object*-level typing.Protocol
(name/is_configured()/fetch()/test_connection()) for connectors that hold
credentials as state across repeated fetches. Don't conflate the two: a
write-back adapter here takes credentials fresh on every call and is used
directly as a free function; a BMSAdapter is instantiated once and reused.
If a future write-back adapter needs to hold state across calls too,
that's a sign it should become a BMSAdapter-style Protocol instead of
stretching this one — write-back and fetch-side connectors have earned
their own conventions in this project's history for a reason.
"""


def classify_result(result: "dict | None") -> str:
    """
    Classify a write-back adapter's return value into the one of three
    states every caller must handle:

      "unconfigured" -- result is None
      "error"        -- result is a dict with an "error" key
      "success"      -- any other dict

    Callers use this instead of re-deriving the same
    `if result is None: ... elif "error" in result: ...` branch by hand.
    """
    if result is None:
        return "unconfigured"
    if isinstance(result, dict) and "error" in result:
        return "error"
    return "success"
