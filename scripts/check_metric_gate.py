"""
CI gate: the registry's logged headline numbers must stay inside their
declared floors/baselines (config/metric_gate_expectations.json).

The full-registry check
-----------------------
tests/test_ci_metric_gate.py runs the same gate logic against a small
deterministic fixture fleet on every pytest invocation. This script runs it
against the platform's REAL logged runs — the leaderboard the Benchmark page
actually shows — as a release gate. A regression that only shows on real
data (a loader fix, a feature change, a fleet that stopped producing
observed-EOL labels) fails here even though the fixture fleet stayed green.

Latest-run semantics: only the newest non-suffix run per (dataset,
model_kind) is gated — the registry keeps history (that is what
src/metric_history.py watches), but a gate compares the CURRENT claim
against the promise. An older run failing an updated floor is history,
not a regression.

Drift mode: --drift compares consecutive runs across history instead of
gating the latest run (src/metric_history.drift_report) — the audit view.

Run:
    python scripts/check_metric_gate.py            # gate latest runs (exit 1 on fail)
    python scripts/check_metric_gate.py --drift    # history drift audit (exit 1 on alerts)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))  # _paths.py lives at the root
for _p in ("src", "app", "scripts"):
    sys.path.insert(0, str(_root / _p))
import _paths  # noqa: F401  (side-effect: canonical sys.path bootstrap)

GATE_EXPECTATIONS = _root / "config" / "metric_gate_expectations.json"

# Suffixes/markers for study populations that are not accuracy claims
# (same exclusion rule as accuracy_by_source / model_kind_comparison).
# "_to_" is the cross-chemistry transfer naming infix (run_cross_chemistry_transfer
# logs under f"{train}_to_{eval}"); no reference dataset name contains it.
_STUDY_SUFFIXES = ("_robustness", "_prospective", "_transfer")


def _is_study_population(dataset: str) -> bool:
    return dataset.endswith(_STUDY_SUFFIXES) or "_to_" in dataset


def _latest_runs_by_dataset(runs: list) -> dict:
    """Newest non-study run per (dataset, model_kind), keyed for the gate."""
    latest: dict = {}
    for r in runs:
        dataset = r.get("dataset") or ""
        if _is_study_population(dataset):
            continue
        key = (dataset, r.get("model_kind", "gbrt"))
        prev = latest.get(key)
        if prev is None or str(r.get("timestamp") or "") > str(prev.get("timestamp") or ""):
            latest[key] = r
    return latest


def check_registry_gate(runs: list, expectations: dict) -> dict:
    """Gate the latest logged run per (dataset, model_kind).

    Returns {verdict, per_dataset: [{dataset, model_kind, run_id, gate}]}
    — one evaluate_gate() result per population, aggregated: pass only if
    every population passed. A population whose run carries no values at
    all for the tracked metrics is reported as failing (a dataset that
    stopped producing numbers is exactly what the gate must catch).
    """
    from batlab.validation.metric_gate import evaluate_gate

    tracked = list((expectations.get("metrics") or {}).keys())
    per_dataset = []
    any_fail = False
    for (dataset, model_kind), run in sorted(_latest_runs_by_dataset(runs).items()):
        observed = {m: run.get(m) for m in tracked}
        gate = evaluate_gate(observed, expectations)
        if gate["verdict"] != "pass":
            any_fail = True
        per_dataset.append({
            "dataset": dataset,
            "model_kind": model_kind,
            "run_id": run.get("run_id"),
            "gate": gate,
        })
    return {
        "verdict": "fail" if any_fail else "pass",
        "per_dataset": per_dataset,
    }


def main() -> int:
    import argparse

    import experiment_registry as reg
    from batlab.validation.metric_gate import (
        evaluate_gate,
        format_gate_report,
        load_expectations,
    )
    from metric_history import drift_report, format_drift_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drift", action="store_true",
        help="audit consecutive-run drift across history instead of gating the latest run",
    )
    parser.add_argument(
        "--org", type=int, default=None,
        help="org id scope (default: the platform's shared benchmark org)",
    )
    args = parser.parse_args()

    org = args.org if args.org is not None else reg.PLATFORM_ORG_ID
    # Default sort is rul_mae (numeric); timestamp sorting is applied here —
    # the registry's _sort_key negates values and cannot negate strings.
    runs = reg.leaderboard(org)
    runs.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    if not runs:
        print("metric gate: no logged runs found — nothing to gate (is the app database initialized?)")
        return 0

    if args.drift:
        report = drift_report(runs)
        print(format_drift_report(report))
        return 1 if report["n_alerts"] else 0

    if not GATE_EXPECTATIONS.exists():
        print(
            f"metric gate: {GATE_EXPECTATIONS} not found — real-run floors are not "
            "declared yet. The CI fixture-fleet gate (tests/test_ci_metric_gate.py) "
            "still enforces the fleet path; declare real-run expectations once the "
            "first benchmark cycle for each fleet is logged."
        )
        return 0

    expectations = load_expectations(GATE_EXPECTATIONS)
    result = check_registry_gate(runs, expectations)
    for entry in result["per_dataset"]:
        print(f"--- {entry['dataset']} ({entry['model_kind']}) run {entry['run_id']}")
        print(format_gate_report(entry["gate"]))
    print(f"metric gate (registry): {result['verdict'].upper()} "
          f"over {len(result['per_dataset'])} dataset populations")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
