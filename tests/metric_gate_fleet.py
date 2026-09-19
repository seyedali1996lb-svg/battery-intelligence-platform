"""
The CI metric-gate fleet — a deterministic synthetic population whose LCO
numbers are checked against declared floors/baselines on every test run.

Why a fixture fleet
-------------------
The real fleets are expensive (NASA/Severson/Zhu LCO runs take minutes) and
their raw data is not in CI. This fleet is: five of the platform's own
stress-profile cells, generated from fixed seeds (`load_or_generate_cell`),
small enough for a sub-minute LCO, deterministic enough that the only
legitimate reasons a headline number moves are a model/feature change or an
environment change — exactly the changes the gate must catch. The floors and
baselines in expectations.json are MEASURED, not aspirational: they were
recorded by running this exact fleet through the unmodified run_lco()
harness (see scripts/update_metric_baselines.py --initial).

The population deliberately spans the stress axes (20-40 C, 0.5-2C, 70-100%
DoD) so a stress-physics regression shows up as drift, not just a
capacity-fade regression.
"""

from __future__ import annotations

import json
from pathlib import Path

# Five cells spanning the stress axes; 5 folds keeps the full LCO well
# under a minute while remaining a real multi-cell evaluation.
GATE_FLEET_CELL_IDS = ["Cell1", "Cell3", "Cell5", "Cell6", "Cell8"]

# Metrics the gate tracks on the fixture fleet. Everything else run_lco()
# returns is surfaced as UNTRACKED (a new headline number should be a
# conscious decision to give it a floor).
GATE_TRACKED_METRICS = ["soh_r2", "rul_r2"]

_HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = _HERE / "metric_gate_expectations.json"


def build_gate_fleet() -> dict:
    """{cell_id: cycles DataFrame} for the gate fleet, via the app's own
    load path (build_battery: cached CSV -> _normalise_columns ->
    enrich_cycles) — NOT the bare load_or_generate_cell, whose cached CSVs
    lack the enriched columns (soh_pct etc.) every downstream consumer
    (and the fingerprint digest) assumes."""
    from src.data_loader import build_battery

    battery = build_battery(battery_id="CI_metric_gate", cell_ids=list(GATE_FLEET_CELL_IDS))
    return battery["cells"]


def run_gate_fleet_lco(cell_data: dict | None = None) -> dict:
    """run_lco() over the gate fleet — the numbers the gate checks.

    The fixture cells' RUL labels are extrapolated (no observed EOL), so
    rul_r2 is None under the v12 rules and the gate tracks the SOH head
    plus the fingerprint. The expectations file pins rul_r2 with
    allow_not_evaluable so a fleet that stops producing a usable SOH head
    is caught, and rul_evaluability_change documents why None is expected.
    """
    from batlab.validation.lco import run_lco

    # use_fold_cache=False: the gate exists to catch a CHANGE in what the code
    # computes, so it must re-derive the numbers. A cached fold (this runs as
    # a script in CI as well as under pytest) would let a regression pass by
    # replaying the pre-regression result.
    return run_lco(
        cell_data if cell_data is not None else build_gate_fleet(),
        use_fold_cache=False,
    )


def run_metric_gate(cell_data: dict | None = None) -> dict:
    """Build (or accept) the fixture fleet, run LCO, check the gate.

    Returns {gate, observed, lco} — `gate` is evaluate_gate()'s dict; the
    verdict string is gate["verdict"].
    """
    from batlab.validation.metric_gate import evaluate_gate, load_expectations

    observed_raw = run_gate_fleet_lco(cell_data)
    expectations = load_expectations(EXPECTATIONS_PATH)
    observed = {k: observed_raw.get(k) for k in GATE_TRACKED_METRICS}
    gate = evaluate_gate(observed, expectations)
    return {"gate": gate, "observed": observed, "lco": observed_raw}


def load_expectations_json() -> dict:
    """Raw expectations file access for the updater script."""
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
