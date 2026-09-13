"""
Record an explained headline-number change in a metric-gate expectations
file — the ONLY sanctioned way a baseline moves.

The gate (batlab.validation.metric_gate) fails CI on any metric outside its
declared range or beyond its baseline tolerance. Passing the gate after a
deliberate change is a two-step, reviewable act: run this script with a
reason, commit the diff. The reason is mandatory — a baseline update
without one would reintroduce the silent drift the gate exists to prevent.
Every change is appended to the file's change_log (and lives forever in
git history), so the headline numbers carry their own audit trail.

Examples:
    # The CI fixture fleet moved (a model or feature change landed):
    python scripts/update_metric_baselines.py --fleet --reason "switched GBRT subsample to 0.8; SOH R2 moved accordingly"

    # The real-run gate moved (registry numbers):
    python scripts/update_metric_baselines.py --registry --reason "Severson sentinel cleanup at load time changed the training pool"

    # Seed the floors/baselines the first time (no reason required only for --initial):
    python scripts/update_metric_baselines.py --fleet --initial
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))  # _paths.py lives at the root
for _p in ("src", "app", "scripts"):
    sys.path.insert(0, str(_root / _p))
import _paths  # noqa: F401  (side-effect: canonical sys.path bootstrap)

FLEET_EXPECTATIONS = _root / "tests" / "metric_gate_expectations.json"
REGISTRY_EXPECTATIONS = _root / "config" / "metric_gate_expectations.json"


def _git_commit() -> "str | None":
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip() or None
    except Exception:
        return None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_fleet_observed() -> dict:
    """Re-measure the fixture fleet now (real run_lco, ~1 minute)."""
    sys.path.insert(0, str(_root / "tests"))
    import io
    import contextlib

    import metric_gate_fleet as mgf  # tests/ is not a package

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = mgf.run_metric_gate()
    return res["observed"]


def collect_registry_observed(org_id) -> dict:
    """Latest non-study run per (dataset, model_kind), as the gate sees it.
    Registry metrics are keyed "<dataset>:<metric>" so each fleet carries
    its own baseline."""
    import experiment_registry as reg
    from check_metric_gate import _latest_runs_by_dataset

    runs = reg.leaderboard(org_id if org_id is not None else reg.PLATFORM_ORG_ID)
    runs.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    latest = _latest_runs_by_dataset(runs)
    observed: dict = {}
    for (dataset, _mk), run in sorted(latest.items()):
        if run.get("soh_r2") is not None:
            observed[f"{dataset}:soh_r2"] = run["soh_r2"]
        if run.get("rul_r2") is not None:
            observed[f"{dataset}:rul_r2"] = run["rul_r2"]
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", action="store_true", help="update tests/metric_gate_expectations.json (CI fixture fleet)")
    parser.add_argument("--registry", action="store_true", help="update config/metric_gate_expectations.json (real logged runs)")
    parser.add_argument("--reason", type=str, default="", help="WHY the number moved (mandatory unless --initial)")
    parser.add_argument("--initial", action="store_true", help="first-time seeding: measure and write floors/baselines without a reason")
    parser.add_argument("--org", type=int, default=None, help="org id for --registry (default: platform benchmark org)")
    args = parser.parse_args()

    if not (args.fleet or args.registry):
        parser.error("choose --fleet and/or --registry")
    if not args.initial and not args.reason.strip():
        parser.error(
            "--reason is required: a baseline update without a recorded reason "
            "is the silent drift the gate exists to prevent"
        )

    commit = _git_commit()
    now = datetime.now(timezone.utc).isoformat()

    if args.fleet:
        print("Re-measuring the fixture fleet (~1 minute)...")
        observed = collect_fleet_observed()
        if args.initial:
            print(f"Measured: {observed}")
            print(
                "Now set floors/baselines in tests/metric_gate_expectations.json "
                "from these values (floors deliberately below the measured number) "
                "and commit. --initial never writes the file for you: choosing a "
                "floor is a promise a human makes."
            )
            return 0
        data = _read(FLEET_EXPECTATIONS)
        from batlab.validation.metric_gate import update_baselines

        updated = update_baselines(data, observed, args.reason, commit)
        updated["measured_at"] = now[:10]
        _write(FLEET_EXPECTATIONS, updated)
        print(f"tests/metric_gate_expectations.json updated (reason recorded, commit {commit}).")
        print("Review the diff and commit it — that diff IS the audit trail.")

    if args.registry:
        observed = collect_registry_observed(args.org)
        if not observed:
            print("No logged registry runs found — nothing to update.")
            return 1
        reg_path = REGISTRY_EXPECTATIONS
        if not reg_path.exists():
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            _write(reg_path, {
                "schema_version": 1,
                "description": "Metric gate expectations for the platform's REAL logged runs (latest run per dataset, gated by scripts/check_metric_gate.py).",
                "metrics": {},
            })
            print(f"Created {reg_path} — declare floors for each metric, then re-run without --initial.")
        data = _read(reg_path)
        if args.initial:
            print(f"Measured (registry): {observed}")
            return 0
        from batlab.validation.metric_gate import update_baselines

        # Registry metrics are keyed "<dataset>:<metric>" so each fleet gets
        # its own baseline; unseen keys are appended with a None floor by the
        # operator afterwards (the gate treats floor-less entries as
        # baseline-only tracking).
        metrics = data.setdefault("metrics", {})
        for key, value in observed.items():
            if key not in metrics:
                metrics[key] = {"baseline": float(value), "note": "seeded by update_metric_baselines.py; add a floor"}
        updated = update_baselines(data, observed, args.reason, commit)
        _write(reg_path, updated)
        print(f"config/metric_gate_expectations.json updated (reason recorded, commit {commit}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
