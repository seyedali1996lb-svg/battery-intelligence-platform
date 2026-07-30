"""
Spine-compatible second-life battery data export.

Converts this platform's second-life economics/health data — src/consequences.py's
application_fit()/financial_comparison()/sustainability_snapshot(), RUL quantile
predictions, and the Battery Knowledge Graph's mechanism verdict — into the JSON
format spinedb_api's import_data() (and the Spine Toolbox Data Store importer)
consume: entity_classes/entities/parameter_definitions/parameter_values/
alternatives. Lets a customer's grid-modeling team import a cell/pack's health
and second-life economics straight into a SpineDB-backed model (e.g. a
SpineOpt/FlexTool storage-flexibility study) instead of hand-transcribing values
off this app's UI.

Schema confirmed against the real Spine Toolbox / spinedb_api JSON format docs
(https://spine-toolbox.readthedocs.io/en/latest/spine_db_editor/importing_and_exporting_data.html,
https://spine-database-api.readthedocs.io/en/stable/user_guide.html) before writing
this module — not guessed from the EPRI Europe / GitHub README summary alone.

Explicit scope boundaries:
  - One-way export artifact only. No live SpineDB connection, no spinedb_api
    runtime dependency (this module produces a plain JSON-serializable dict —
    a user's own Spine workflow calls spinedb_api.import_data(db_map, **data)
    on it, or imports it via Spine Toolbox's Data Store JSON importer), no
    SpineOpt/Julia integration.
  - No historical SOH-vs-cycle trajectory export. This platform's cells are
    indexed by cycle_number, not calendar time — forcing that into SpineDB's
    native "time_series" parameter type (which requires real ISO timestamps)
    would misrepresent cycle count as elapsed time, the same class of
    methodological error this project's history already flagged once for the
    Oxford dataset (checkpoint_index vs. real cycle counts — see
    docs/history.md). SpineDB's "array" type would be the honest fit instead,
    but its exact JSON shape isn't confirmable from the published docs above
    without reading spinedb_api's source directly — left as explicit future
    work (see pending_work) rather than guessed here. This module exports the
    current-state snapshot only (SOH/RUL/mechanism/economics at the time of
    export), not a curve.
  - No deployment-sizing (PV+battery) figures — those are EUR-denominated
    (src/deployment_sizing.py) on a separate cost basis from consequences.py's
    USD figures; folding both into one export was left out of this first
    version to avoid conflating them.
"""

import datetime

ENTITY_CLASS = "battery_unit"

# SpineDB "alternatives" branch a parameter's value without duplicating the
# whole entity — a genuinely good fit for RUL's p10/p50/p90 quantile bounds,
# which are exactly "the same parameter under different assumptions".
RUL_ALTERNATIVES = [
    ["p10", "Conservative (10th percentile) RUL bound"],
    ["p50", "Median RUL point estimate"],
    ["p90", "Optimistic (90th percentile) RUL bound"],
]


def _parameter_definitions() -> list:
    """[entity_class, parameter_name, default_value, value_list_name, description]
    for every parameter this module writes — see module docstring for the
    confirmed 5-element shape."""
    defs = [
        ("chemistry", "Cell chemistry (e.g. LFP, NCA, LiCoO2)"),
        ("soh_pct", "State of health, percent of nominal capacity"),
        ("capacity_kwh", "Nominal capacity, kWh"),
        ("current_capacity_kwh", "Capacity remaining at current SOH, kWh"),
        ("rul_cycles", "Remaining useful life, cycles — see alternatives for p10/p50/p90 bounds"),
        ("degradation_mechanism", "Dominant degradation mechanism verdict (LLI/LAM/insufficient_data)"),
        ("second_life_application_fit", "Second-life application categories this cell fits, comma-separated"),
        ("second_life_net_value_usd", "Estimated second-life reuse net value after repack cost"),
        ("recycle_value_usd", "Estimated immediate-recycle value"),
        ("repack_cost_usd", "Estimated repack/BMS-integration labour cost"),
        ("co2_avoided_by_reuse_kg", "CO2 avoided by reuse vs. new-cell manufacture, kg"),
        ("source_experiment_run_id", "Provenance: platform ExperimentRun.run_id that produced these predictions"),
        ("source_git_commit", "Provenance: platform git commit at prediction time"),
    ]
    return [[ENTITY_CLASS, name, None, None, desc] for name, desc in defs]


def build_second_life_export(
    cell_id: str,
    source: str,
    chemistry: str,
    soh: float,
    fade_30_mah_cy: float,
    fleet_fade_median: "float | None",
    rul_reliable: bool,
    rul_q10: "float | None" = None,
    rul_pred: "float | None" = None,
    rul_q90: "float | None" = None,
    mechanism: "dict | None" = None,
    assumptions: "dict | None" = None,
    run_id: "str | None" = None,
    git_commit: "str | None" = None,
) -> dict:
    """
    Build the SpineDB-format dict for one cell's second-life snapshot.

    Reuses src/consequences.py's existing application_fit()/
    financial_comparison()/sustainability_snapshot() rather than
    re-deriving their math — this module only reshapes their output into
    the Spine entity/parameter format.

    assumptions, if given, overrides individual keys from
    consequences.ASSUMPTIONS (same {value,...} dict shape) — same override
    convention as consequences.py's own callers (e.g. the Decision page's
    sliders).

    rul_q10/rul_pred/rul_q90 are only written (as the p10/p50/p90
    alternatives on the rul_cycles parameter) when rul_reliable is True and
    each individual value is not None — an unreliable-RUL cell exports
    everything except rul_cycles, an honest omission rather than a
    fabricated number.
    """
    from consequences import ASSUMPTIONS, application_fit, financial_comparison, sustainability_snapshot

    a = {k: v["value"] for k, v in ASSUMPTIONS.items()}
    if assumptions:
        a.update(assumptions)

    fit_scores = application_fit(soh, fade_30_mah_cy, fleet_fade_median)
    financials = financial_comparison(
        soh, source, a["recycling_value"], a["new_cell_cost"],
        a["second_life_value_per_kwh"], a["repack_cost"],
    )
    sustainability = sustainability_snapshot(source, a["co2_manufacture"], a["material_recovery"])

    parameter_values = []

    def _pv(name, value, alternative_name=None):
        parameter_values.append([ENTITY_CLASS, cell_id, name, value, alternative_name])

    _pv("chemistry", chemistry)
    _pv("soh_pct", round(soh, 2))
    _pv("capacity_kwh", round(financials["cell_kwh"], 6))
    _pv("current_capacity_kwh", round(financials["current_kwh"], 6))
    _pv("second_life_net_value_usd", round(financials["sl_net"], 2))
    _pv("recycle_value_usd", round(financials["recycle_value"], 2))
    _pv("repack_cost_usd", round(financials["repack_cost"], 2))
    _pv("co2_avoided_by_reuse_kg", round(sustainability["co2_avoided_by_reuse"], 3))

    alternatives = []
    if rul_reliable:
        for value, alt_name in ((rul_q10, "p10"), (rul_pred, "p50"), (rul_q90, "p90")):
            if value is not None:
                _pv("rul_cycles", round(float(value), 1), alt_name)
        if any(v is not None for v in (rul_q10, rul_pred, rul_q90)):
            alternatives = [list(alt) for alt in RUL_ALTERNATIVES]

    if mechanism:
        _pv("degradation_mechanism", mechanism.get("verdict", "insufficient_data"))

    fitting_apps = [key for key, v in fit_scores.items() if v["fit"] == "fit"]
    _pv("second_life_application_fit", ", ".join(fitting_apps) if fitting_apps else "none")

    if run_id:
        _pv("source_experiment_run_id", run_id)
    if git_commit:
        _pv("source_git_commit", git_commit)

    return {
        "entity_classes": [
            [ENTITY_CLASS, [], "A battery cell or pack evaluated for second-life deployment", None],
        ],
        "entities": [[ENTITY_CLASS, cell_id, None]],
        "parameter_definitions": _parameter_definitions(),
        "parameter_values": parameter_values,
        "alternatives": alternatives,
    }


def to_export_document(spine_data: dict, cell_id: str) -> dict:
    """
    Wrap build_second_life_export()'s output with download-level metadata —
    same "wrap the core dict, don't reinvent it" pattern as
    passport_export.to_json_ld() wrapping build_passport(). The "data" key
    is the literal spinedb_api.import_data(db_map, **doc["data"]) payload;
    "metadata" is for a human reading the file, not part of the Spine schema.
    """
    return {
        "metadata": {
            "cell_id": cell_id,
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "format": "spine-toolbox-json",
            "format_reference": (
                "https://spine-toolbox.readthedocs.io/en/latest/"
                "spine_db_editor/importing_and_exporting_data.html"
            ),
            "disclaimer": (
                "One-way data export for use with Spine Toolbox / spinedb_api "
                "import_data(db_map, **data) or the Spine Toolbox Data Store "
                "JSON importer. Not a live database connection. Financial and "
                "sustainability figures are the same cited estimates or "
                "illustrative assumptions used elsewhere in this app (see "
                "src/consequences.py ASSUMPTIONS for sourcing) — not validated "
                "model outputs. No historical SOH-vs-cycle trajectory is "
                "included — see this module's docstring for why."
            ),
        },
        "data": spine_data,
    }
