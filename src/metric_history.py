"""
Continuous metric tracking with drift alerts — the registry's history,
read as a time series and checked for unexplained movement.

Why this exists
---------------
The metric gate (batlab.validation.metric_gate) catches drift at CI time
on a fixture fleet; this module catches it on the REAL fleets, whose
numbers are re-measured on every app load and logged permanently by the
experiment registry. Every (dataset, model_kind) run series is a tracking
chart nobody watches; drift_report() is the watcher. A headline number that
moves beyond its declared tolerance between consecutive runs is surfaced
with the magnitude and direction of the move — keyed by feature_version,
because a feature-set change is a legitimate reason for a number to move
and must not read as drift.

Pure logic over run dicts (the shape _experiment_run_row_to_dict returns);
DB access stays in the caller so this module is directly unit-testable and
usable from scripts, the API, or the Benchmark page.
"""

from __future__ import annotations

# Metrics tracked by default — the headline accuracy numbers every
# leaderboard already displays.
TRACKED_METRICS = ("soh_r2", "rul_r2", "baseline_soh_r2")

# Relative move (fraction of |previous|) considered alert-worthy when a run
# does not declare its own tolerance. Matches the gate's default: expected
# noise passes, real movement alerts.
DEFAULT_ALERT_TOLERANCE = 0.05

_NEAR_ZERO_ABS_EPSILON = 1e-3

# Runs are sorted by this field (ISO timestamps sort lexicographically).
_TIME_KEY = "timestamp"


def metric_history(
    runs: "list[dict]",
    dataset: str,
    model_kind: str = "gbrt",
    metric: str = "soh_r2",
) -> "list[dict]":
    """One metric's time series for a (dataset, model_kind) population.

    Returns [{run_id, timestamp, feature_version, value, git_commit}] in
    chronological order, skipping runs where the metric is missing/None.
    """
    series = []
    for r in runs:
        if r.get("dataset") != dataset or r.get("model_kind", "gbrt") != model_kind:
            continue
        value = r.get(metric)
        if value is None:
            continue
        series.append({
            "run_id": r.get("run_id"),
            "timestamp": r.get(_TIME_KEY),
            "feature_version": r.get("feature_version"),
            "value": float(value),
            "git_commit": r.get("git_commit"),
        })
    series.sort(key=lambda e: str(e.get(_TIME_KEY) or ""))
    return series


def drift_report(
    runs: "list[dict]",
    tolerance: float = DEFAULT_ALERT_TOLERANCE,
    metrics: "tuple[str, ...]" = TRACKED_METRICS,
) -> dict:
    """Scan every (dataset, model_kind) series for unexplained moves.

    A drift ALERT is a consecutive pair of runs, same feature_version,
    whose metric moved by more than tolerance (relative, with an absolute
    epsilon near zero). A feature_version CHANGE boundary is reported as a
    separate, non-alerting "explained" event: the feature code changed, so
    the number was always going to move — that is the registry working, not
    drift.

    Returns {datasets: [{dataset, model_kind, n_runs, alerts: [...],
    feature_changes: [...], latest: {...}}], n_alerts}.
    """
    populations: dict = {}
    for r in runs:
        dataset = r.get("dataset") or ""
        # Study populations are not accuracy claims: stress tests (suffix),
        # temporal holdouts (suffix), and cross-chemistry transfers (the
        # "_to_" infix run_cross_chemistry_transfer logs under).
        if dataset.endswith(("_robustness", "_prospective", "_transfer")) or "_to_" in dataset:
            continue
        key = (dataset, r.get("model_kind", "gbrt"))
        populations.setdefault(key, []).append(r)

    out_datasets = []
    n_alerts = 0
    for (dataset, model_kind), pop in sorted(populations.items()):
        alerts: list = []
        feature_changes: list = []
        latest: dict = {}
        for metric in metrics:
            series = metric_history(pop, dataset, model_kind, metric)
            if not series:
                continue
            latest[metric] = series[-1]["value"]
            for prev, curr in zip(series, series[1:]):
                if prev["feature_version"] != curr["feature_version"]:
                    feature_changes.append({
                        "metric": metric,
                        "from_version": prev["feature_version"],
                        "to_version": curr["feature_version"],
                        "from_value": prev["value"],
                        "to_value": curr["value"],
                    })
                    continue
                delta = curr["value"] - prev["value"]
                scale = max(abs(prev["value"]), _NEAR_ZERO_ABS_EPSILON)
                if abs(delta) > tolerance * scale:
                    n_alerts += 1
                    alerts.append({
                        "metric": metric,
                        "from_run": prev["run_id"],
                        "to_run": curr["run_id"],
                        "from_value": prev["value"],
                        "to_value": curr["value"],
                        "delta": delta,
                        "from_time": prev["timestamp"],
                        "to_time": curr["timestamp"],
                        "from_commit": prev["git_commit"],
                        "to_commit": curr["git_commit"],
                    })
        out_datasets.append({
            "dataset": dataset,
            "model_kind": model_kind,
            "n_runs": len(pop),
            "alerts": alerts,
            "feature_changes": feature_changes,
            "latest": latest,
        })

    return {"datasets": out_datasets, "n_alerts": n_alerts}


def format_drift_report(report: dict) -> str:
    """Human-readable drift report for CI logs and the Benchmark page."""
    if not report["datasets"]:
        return "metric history: no tracked runs"
    lines = []
    for d in report["datasets"]:
        head = f"{d['dataset']} ({d['model_kind']}): {d['n_runs']} runs"
        if d["latest"]:
            latest = ", ".join(f"{k}={v:.4g}" for k, v in sorted(d["latest"].items()))
            head += f" — latest {latest}"
        lines.append(head)
        for a in d["alerts"]:
            lines.append(
                f"  [DRIFT] {a['metric']}: {a['from_value']:.4g} -> "
                f"{a['to_value']:.4g} ({a['delta']:+.4g}) between "
                f"{a['from_time']} and {a['to_time']}"
            )
        for c in d["feature_changes"]:
            lines.append(
                f"  [feature change] {c['metric']}: {c['from_version']} -> "
                f"{c['to_version']} ({c['from_value']:.4g} -> {c['to_value']:.4g})"
            )
    if not report["n_alerts"]:
        lines.insert(0, "metric history: no drift alerts")
    return "\n".join(lines)
