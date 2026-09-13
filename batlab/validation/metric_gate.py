"""
The metric gate — a declared floor/ceiling (and baseline) per headline
number, checked mechanically so a silent accuracy regression is impossible.

Why this exists
---------------
Every other accuracy guard in this project *measures* honesty; this one
*enforces* it. The registry, manifests, and provenance chains make each
number explainable after the fact — but nothing stops a code change from
quietly moving a headline R² and the new number shipping as if it were the
old one. The gate closes that hole with three rules:

  1. FLOOR / CEILING  — a metric outside its declared range fails, period.
     A floor is a promise ("the model is at least this good"); letting CI
     pass under it means the promise was never real.
  2. BASELINE DRIFT   — within a declared tolerance, movement is expected
     noise (library patch versions, float summation order) and passes.
     Beyond tolerance it is an UNEXPLAINED CHANGE and fails — not because
     the new number is wrong, but because shipping it without saying so is
     exactly how a silent regression gets in. The fix is deliberate:
     scripts/update_metric_baselines.py records the change with a reason,
     producing a reviewable change log instead of a silent drift.
  3. NOT-EVALUABLE    — a metric that stopped being computable is itself a
     change. It fails unless the expectation explicitly allows it
     (allow_not_evaluable), because "the headline number disappeared"
     must be a decision, never an accident.

The gate is PURE LOGIC — no DB, no filesystem, no batlab model imports —
so it is directly unit-testable (tests/test_metric_gate.py) and usable
from any caller: the CI fixture-fleet check (tests/test_ci_metric_gate.py),
scripts/check_metric_gate.py over the registry's logged runs, and
drift_report() over metric history.
"""

from __future__ import annotations

import json
from pathlib import Path

GATE_SCHEMA_VERSION = 1

# A change within this fraction of the recorded baseline is noise, not news.
# Deliberately loose enough to absorb sklearn patch-version float jitter on
# an 8-cell fixture fleet (the CI population), tight enough that any real
# model/feature change clears it by an order of magnitude.
DEFAULT_BASELINE_TOLERANCE = 0.05

# Absolute epsilon for baselines near zero, where a relative tolerance is
# meaningless (|obs − base| ≤ 0.001 passes regardless of relative size).
_NEAR_ZERO_ABS_EPSILON = 1e-3

# Verdicts
PASS = "pass"
FAIL = "fail"
UNTRACKED = "untracked"  # observed metric with no expectation — surfaced, not fatal


def load_expectations(path: "str | Path") -> dict:
    """Load and validate an expectations JSON file.

    Raises ValueError with a specific message on structural problems — a
    malformed gate config must fail loudly at load, not silently gate
    nothing.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expectations file must be a JSON object.")
    if data.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError(
            f"Expectations schema_version {data.get('schema_version')!r} != "
            f"{GATE_SCHEMA_VERSION}; rewrite the file with "
            "scripts/update_metric_baselines.py."
        )
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Expectations file must declare a non-empty 'metrics' object.")
    for name, exp in metrics.items():
        if not isinstance(exp, dict):
            raise ValueError(f"Metric {name!r}: expectation must be an object.")
        if "floor" not in exp and "ceiling" not in exp:
            raise ValueError(f"Metric {name!r}: needs at least a 'floor' or a 'ceiling'.")
        for key in ("floor", "ceiling", "baseline", "tolerance"):
            if key in exp and not isinstance(exp[key], (int, float)):
                raise ValueError(f"Metric {name!r}: {key!r} must be a number.")
    return data


def check_metric(
    name: str,
    observed: "float | None",
    expectation: "dict | None",
) -> dict:
    """Check one observed metric against its expectation.

    Returns {name, verdict, observed, floor, ceiling, baseline, detail}.
    verdict is PASS, FAIL, or UNTRACKED (expectation None).
    """
    if expectation is None:
        return {
            "name": name,
            "verdict": UNTRACKED,
            "observed": observed,
            "floor": None,
            "ceiling": None,
            "baseline": None,
            "detail": "No expectation declared — add one or stop reporting it.",
        }

    floor = expectation.get("floor")
    ceiling = expectation.get("ceiling")
    baseline = expectation.get("baseline")
    tolerance = float(expectation.get("tolerance", DEFAULT_BASELINE_TOLERANCE))

    if observed is None:
        if expectation.get("allow_not_evaluable"):
            return {
                "name": name,
                "verdict": PASS,
                "observed": None,
                "floor": floor,
                "ceiling": ceiling,
                "baseline": baseline,
                "detail": "not evaluable — explicitly allowed by the expectation",
            }
        return {
            "name": name,
            "verdict": FAIL,
            "observed": None,
            "floor": floor,
            "ceiling": ceiling,
            "baseline": baseline,
            "detail": (
                "metric not evaluable — a headline number that stopped being "
                "computable is itself a change; allow it explicitly "
                "(allow_not_evaluable) or fix the evaluation"
            ),
        }

    observed = float(observed)

    # Rule 1: the declared range.
    if floor is not None and observed < float(floor):
        return {
            "name": name,
            "verdict": FAIL,
            "observed": observed,
            "floor": floor,
            "ceiling": ceiling,
            "baseline": baseline,
            "detail": f"{observed:.6g} below floor {floor}",
        }
    if ceiling is not None and observed > float(ceiling):
        return {
            "name": name,
            "verdict": FAIL,
            "observed": observed,
            "floor": floor,
            "ceiling": ceiling,
            "baseline": baseline,
            "detail": f"{observed:.6g} above ceiling {ceiling}",
        }

    # Rule 2: baseline drift vs approved change.
    if baseline is not None:
        baseline = float(baseline)
        delta = abs(observed - baseline)
        scale = max(abs(baseline), _NEAR_ZERO_ABS_EPSILON)
        # Two escapes, both deliberate: (a) relative tolerance scaled by the
        # baseline; (b) an absolute epsilon — for a metric on a ~±1 scale,
        # a 0.001 absolute move is noise even against a 0.0 baseline, where
        # any relative rule would explode.
        if delta > tolerance * scale and delta > _NEAR_ZERO_ABS_EPSILON:
            approved = expectation.get("approved_change") or {}
            if (
                approved.get("from") is not None
                and abs(float(approved["from"]) - baseline) <= tolerance * scale
                and str(approved.get("reason") or "").strip()
            ):
                return {
                    "name": name,
                    "verdict": PASS,
                    "observed": observed,
                    "floor": floor,
                    "ceiling": ceiling,
                    "baseline": baseline,
                    "detail": (
                        f"approved change from {baseline:.6g}: {approved['reason']}"
                    ),
                }
            return {
                "name": name,
                "verdict": FAIL,
                "observed": observed,
                "floor": floor,
                "ceiling": ceiling,
                "baseline": baseline,
                "detail": (
                    f"unexplained change: {baseline:.6g} -> {observed:.6g} "
                    f"(beyond the {tolerance:.0%} tolerance) — if intended, run "
                    "scripts/update_metric_baselines.py to record the reason"
                ),
            }

    return {
        "name": name,
        "verdict": PASS,
        "observed": observed,
        "floor": floor,
        "ceiling": ceiling,
        "baseline": baseline,
        "detail": "within declared range and baseline tolerance",
    }


def evaluate_gate(
    observed_metrics: dict,
    expectations: dict,
) -> dict:
    """Evaluate a full gate run.

    observed_metrics: {metric_name: value-or-None} — e.g. run_lco()'s
        soh_r2/rul_r2 keys.
    expectations: a load_expectations() dict (metrics keyed the same way).

    Returns {verdict, failures, results, untracked} where verdict is
    "pass" only when every tracked metric passed. Metrics present in
    observed but absent from expectations are listed as untracked — a new
    headline number should be a conscious decision to give it a floor.
    """
    exp_metrics = expectations.get("metrics", {})
    results = []
    for name, exp in exp_metrics.items():
        results.append(check_metric(name, observed_metrics.get(name), exp))
    untracked = [
        check_metric(name, observed_metrics.get(name), None)
        for name in sorted(observed_metrics)
        if name not in exp_metrics
    ]
    failures = [r for r in results if r["verdict"] == FAIL]
    return {
        "verdict": "pass" if not failures else "fail",
        "failures": failures,
        "results": results,
        "untracked": untracked,
    }


def update_baselines(
    expectations: dict,
    observed_metrics: dict,
    reason: str,
    commit: "str | None" = None,
) -> dict:
    """Record an explained change: baselines move to the observed values and
    an entry lands in the change log. Pure function — returns the new
    expectations dict for the caller (the script) to write.

    The reason is mandatory: an update with no reason would reintroduce the
    silent drift the gate exists to prevent.
    """
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError(
            "A baseline update requires a reason — that is the entire point "
            "of the gate. State why the number moved."
        )
    exp_metrics = expectations.get("metrics", {})
    entries = []
    for name, value in observed_metrics.items():
        if name not in exp_metrics or value is None:
            continue
        exp = exp_metrics[name]
        old = exp.get("baseline")
        new = float(value)
        if old is not None and abs(new - float(old)) <= 1e-12:
            continue  # unchanged — nothing to record
        exp["approved_change"] = {"from": old, "reason": reason}
        exp["baseline"] = new
        entries.append({"metric": name, "from": old, "to": new, "reason": reason})
    if commit:
        for entry in entries:
            entry["commit"] = commit
    if entries:
        log = expectations.setdefault("change_log", [])
        log.extend(entries)
        expectations["change_log"] = log[-100:]  # cap; the git history is the archive
    return expectations


def format_gate_report(gate: dict) -> str:
    """Human-readable one-line-per-metric report for CI logs and the UI."""
    lines = []
    for r in gate["results"]:
        marker = "PASS" if r["verdict"] == PASS else "FAIL"
        lines.append(f"  [{marker}] {r['name']}: {r['detail']}")
    for r in gate["untracked"]:
        lines.append(f"  [UNTRACKED] {r['name']}: {r['detail']}")
    header = "metric gate: PASS" if gate["verdict"] == "pass" else "metric gate: FAIL"
    return header + "\n" + "\n".join(lines)
