"""
OPTIMADE-shaped battery cell data export.

Converts this platform's core per-cell health data — chemistry, SOH, RUL
quantile predictions, the Battery Knowledge Graph's mechanism verdict, and
batlab.datasets.schema.condition_completeness()'s measurement-condition
disclosure — into a JSON:API resource object shaped like an OPTIMADE entry
response, so a materials-database or interoperability tool that already
consumes OPTIMADE's response conventions can ingest it without hand-
transcribing values off this app's UI. Second-life economics (second_life
application fit, financial comparison, sustainability) stay out of scope
here — that's src/spine_export.py's job; this module covers the same
identity/health core the EU Passport already reports.

Schema confirmed against the real OPTIMADE specification
(https://www.optimade.org/specification/latest/, v1.3.0 at the time this
was written) before writing this module — not guessed from the
battery-data project's README summary alone: a JSON:API-compliant
envelope, `{"meta": {...}, "data": {"type", "id", "attributes"}}`, with
every non-standard attribute name required to carry a provider prefix
(the spec's own example: `_exmpl_custom_field`).

Explicit scope boundaries, same pattern as spine_export.py's own:
  - One-way, static export artifact only. This is NOT a live, filterable
    OPTIMADE API server — no `/info` discovery endpoint, no OPTIMADE filter
    query language, no pagination. Building a literal spec-compliant server
    would be a much larger, separate undertaking; this module produces a
    single entry resource object shaped like what such a server would
    return for one cell, for a user's own OPTIMADE-aware tooling to consume
    directly as a plain JSON file.
  - BattINFO (the other standards target originally considered alongside
    this) was researched and deliberately left out of scope: as of this
    writing it is not on PyPI (git-source-only install, pre-0.8 release per
    its own README) and its only documented usage example covers
    manufacturing specs (manufacturer/model/chemistry/nominal_capacity),
    not degradation data (SOH/RUL/cycle count) — this platform's actual
    domain. See pending_work for the reopening trigger.
"""

import datetime

# Every non-standard field name carries this prefix, per the OPTIMADE spec's
# own provider-prefix requirement for custom/non-base-spec attributes (its
# example: `_exmpl_custom_field`). "battery_cells" is likewise not one of
# OPTIMADE's base entry types (those are structure-database-oriented, e.g.
# "structures") — this platform defines its own.
PREFIX = "_batteryintel_"
ENTRY_TYPE = f"{PREFIX}battery_cells"


def build_battery_cell_entry(
    cell_id: str,
    source: str,
    chemistry: str,
    soh: float,
    capacity_ah: "float | None" = None,
    rul_reliable: bool = False,
    rul_q10: "float | None" = None,
    rul_pred: "float | None" = None,
    rul_q90: "float | None" = None,
    mechanism: "dict | None" = None,
    condition_completeness: "dict | None" = None,
) -> dict:
    """
    Build one OPTIMADE-shaped JSON:API resource object for a single cell.

    rul_q10/rul_pred/rul_q90 are only written when rul_reliable is True and
    each individual value is not None — same honest-omission contract as
    spine_export.build_second_life_export(): an unreliable-RUL cell exports
    everything except the RUL fields, never a fabricated number.

    condition_completeness, if given, is
    batlab.datasets.schema.condition_completeness()'s result for this
    cell's source DataFrame — written as a summary score plus the known-gap
    caveats, not the full per-field breakdown (that level of detail belongs
    in this app's own UI, not a materials-database attribute).
    """
    attributes = {
        f"{PREFIX}chemistry": chemistry,
        f"{PREFIX}soh_pct": round(float(soh), 2),
        f"{PREFIX}source_dataset": source,
    }
    if capacity_ah is not None:
        attributes[f"{PREFIX}capacity_ah"] = round(float(capacity_ah), 6)

    if rul_reliable:
        for value, key in (
            (rul_q10, "rul_cycles_p10"),
            (rul_pred, "rul_cycles_p50"),
            (rul_q90, "rul_cycles_p90"),
        ):
            if value is not None:
                attributes[f"{PREFIX}{key}"] = round(float(value), 1)

    if mechanism:
        attributes[f"{PREFIX}degradation_mechanism"] = mechanism.get("verdict", "insufficient_data")

    if condition_completeness is not None:
        attributes[f"{PREFIX}condition_completeness_score"] = condition_completeness.get("score")
        if condition_completeness.get("caveats"):
            attributes[f"{PREFIX}condition_completeness_caveats"] = list(condition_completeness["caveats"])

    return {"type": ENTRY_TYPE, "id": cell_id, "attributes": attributes}


def to_optimade_document(entry: dict, cell_id: str) -> dict:
    """
    Wrap build_battery_cell_entry()'s output with a `meta` block — same
    "wrap the core dict, don't reinvent it" pattern as
    spine_export.to_export_document() wrapping build_second_life_export().
    `data` is the confirmed OPTIMADE resource-object shape; `meta` is for a
    human (or a tool's own provenance handling) reading the file, not a
    claim of live-server spec conformance — see the module docstring's
    scope boundary.
    """
    return {
        "meta": {
            "cell_id": cell_id,
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "format": "optimade-entry-json",
            "format_reference": "https://www.optimade.org/specification/latest/",
            "disclaimer": (
                "Static, one-way export shaped like a single OPTIMADE entry "
                "resource object (JSON:API envelope, provider-prefixed custom "
                "attributes under the '_batteryintel_' namespace). This is NOT "
                "a live, filterable OPTIMADE API server -- no /info discovery "
                "endpoint, no OPTIMADE filter query language, no pagination. "
                "SOH/RUL/mechanism figures are this platform's own model "
                "outputs, not validated against an external reference."
            ),
        },
        "data": entry,
    }
