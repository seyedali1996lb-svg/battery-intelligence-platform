"""
Unit tests for the Tier-6 verification layer's pure logic:
  - batlab.validation.metric_gate  (the accuracy regression gate)
  - batlab.validation.fingerprints (dataset/environment pinning)
  - src/metric_history             (continuous tracking + drift alerts)
  - batlab.validation.replication  (sealed third-party verification)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from batlab.validation.metric_gate import (
    DEFAULT_BASELINE_TOLERANCE,
    FAIL,
    GATE_SCHEMA_VERSION,
    PASS,
    UNTRACKED,
    check_metric,
    evaluate_gate,
    format_gate_report,
    load_expectations,
    update_baselines,
)
from batlab.validation.fingerprints import (
    cell_digest,
    compare_fingerprints,
    dataset_fingerprint,
    environment_snapshot,
)
from metric_history import (
    DEFAULT_ALERT_TOLERANCE,
    drift_report,
    format_drift_report,
    metric_history,
)
from batlab.validation.replication import (
    REPLICATION_SCHEMA,
    REPLICATION_SCHEMA_VERSION,
    load_bundle,
    verify_bundle,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _cycles_df(n: int = 100, fade: float = 0.0005, temp: float = 25.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n + 1)
    return pd.DataFrame({
        "cycle_number": cycles,
        "capacity_ah": 2.0 * (1 - fade * cycles) + rng.normal(0, 1e-4, n),
        "soh_pct": 100.0 * (1 - fade * cycles) + rng.normal(0, 0.01, n),
        "temperature_c": np.full(n, temp) + rng.normal(0, 0.1, n),
    })


def _expectation(**kw) -> dict:
    exp = {"floor": 0.5, "baseline": 0.9}
    exp.update(kw)
    return exp


# ── metric gate: check_metric ───────────────────────────────────────────


class TestCheckMetric:
    def test_below_floor_fails(self):
        r = check_metric("soh_r2", 0.4, _expectation())
        assert r["verdict"] == FAIL
        assert "floor" in r["detail"]

    def test_above_ceiling_fails(self):
        r = check_metric("soh_mae", 0.5, {"ceiling": 0.2})
        assert r["verdict"] == FAIL

    def test_within_floor_passes(self):
        assert check_metric("soh_r2", 0.91, _expectation())["verdict"] == PASS

    def test_baseline_drift_beyond_tolerance_fails(self):
        r = check_metric("soh_r2", 0.80, _expectation(tolerance=0.05))
        assert r["verdict"] == FAIL
        assert "unexplained change" in r["detail"]
        assert "update_metric_baselines" in r["detail"]

    def test_baseline_drift_within_tolerance_passes(self):
        r = check_metric("soh_r2", 0.87, _expectation(tolerance=0.05))
        assert r["verdict"] == PASS

    def test_approved_change_passes_and_says_why(self):
        r = check_metric("soh_r2", 0.80, _expectation(
            tolerance=0.05,
            approved_change={"from": 0.90, "reason": "switched to deeper trees"},
        ))
        assert r["verdict"] == PASS
        assert "approved" in r["detail"]
        assert "switched to deeper trees" in r["detail"]

    def test_approved_change_with_wrong_from_fails(self):
        """An approved_change whose recorded 'from' doesn't match the
        baseline is an inconsistent ledger — still fails."""
        r = check_metric("soh_r2", 0.80, _expectation(
            tolerance=0.05,
            approved_change={"from": 0.95, "reason": "irrelevant"},
        ))
        assert r["verdict"] == FAIL

    def test_approved_change_without_reason_fails(self):
        r = check_metric("soh_r2", 0.80, _expectation(
            tolerance=0.05,
            approved_change={"from": 0.90},
        ))
        assert r["verdict"] == FAIL

    def test_none_observed_fails_unless_allowed(self):
        assert check_metric("rul_r2", None, _expectation())["verdict"] == FAIL
        r = check_metric("rul_r2", None, _expectation(allow_not_evaluable=True))
        assert r["verdict"] == PASS
        assert "not evaluable" in r["detail"]

    def test_untracked_when_no_expectation(self):
        r = check_metric("new_metric", 0.7, None)
        assert r["verdict"] == UNTRACKED

    def test_nan_observed_is_not_evaluable_not_a_pass(self):
        """NaN compares False against every floor, ceiling and tolerance, so an
        unguarded one sails through the gate — the failure mode this module
        exists to prevent. It is treated exactly like None."""
        r = check_metric("soh_r2", float("nan"), _expectation())
        assert r["verdict"] == FAIL
        assert "not evaluable" in r["detail"]

    def test_nan_observed_passes_only_when_explicitly_allowed(self):
        r = check_metric(
            "rul_r2", float("nan"), _expectation(allow_not_evaluable=True)
        )
        assert r["verdict"] == PASS

    def test_inf_observed_is_not_evaluable(self):
        assert check_metric("soh_r2", float("inf"), _expectation())["verdict"] == FAIL

    def test_near_zero_baseline_uses_absolute_epsilon(self):
        r = check_metric("delta", 0.0005, {"baseline": 0.0, "tolerance": 0.05})
        assert r["verdict"] == PASS  # 0.0005 <= epsilon 1e-3 scaled


# ── metric gate: evaluate_gate / load_expectations / update ────────────


class TestEvaluateGate:
    def test_all_pass(self):
        gate = evaluate_gate(
            {"soh_r2": 0.91, "rul_r2": 0.3},
            {"metrics": {"soh_r2": _expectation(), "rul_r2": {"floor": 0.0}}},
        )
        assert gate["verdict"] == "pass"
        assert gate["failures"] == []

    def test_any_failure_fails_the_gate(self):
        gate = evaluate_gate(
            {"soh_r2": 0.91, "rul_r2": -0.5},
            {"metrics": {"soh_r2": _expectation(), "rul_r2": {"floor": 0.0}}},
        )
        assert gate["verdict"] == "fail"
        assert [f["name"] for f in gate["failures"]] == ["rul_r2"]

    def test_untracked_metrics_surfaced_not_fatal(self):
        gate = evaluate_gate(
            {"soh_r2": 0.91, "mystery": 0.7},
            {"metrics": {"soh_r2": _expectation()}},
        )
        assert gate["verdict"] == "pass"
        assert [u["name"] for u in gate["untracked"]] == ["mystery"]

    def test_report_renders_all_verdicts(self):
        gate = evaluate_gate(
            {"soh_r2": 0.4, "extra": 0.7},
            {"metrics": {"soh_r2": _expectation()}},
        )
        text = format_gate_report(gate)
        assert "FAIL" in text and "UNTRACKED" in text and "metric gate: FAIL" in text


class TestLoadExpectations:
    def _write(self, tmp_path, data) -> str:
        p = tmp_path / "exp.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_round_trip(self, tmp_path):
        p = self._write(tmp_path, {
            "schema_version": GATE_SCHEMA_VERSION,
            "metrics": {"soh_r2": {"floor": 0.5}},
        })
        data = load_expectations(p)
        assert data["metrics"]["soh_r2"]["floor"] == 0.5

    def test_wrong_schema_version_raises(self, tmp_path):
        p = self._write(tmp_path, {"schema_version": 99, "metrics": {"a": {"floor": 1}}})
        with pytest.raises(ValueError, match="schema_version"):
            load_expectations(p)

    def test_missing_metrics_raises(self, tmp_path):
        p = self._write(tmp_path, {"schema_version": GATE_SCHEMA_VERSION})
        with pytest.raises(ValueError, match="metrics"):
            load_expectations(p)

    def test_expectation_without_floor_or_ceiling_raises(self, tmp_path):
        p = self._write(tmp_path, {
            "schema_version": GATE_SCHEMA_VERSION,
            "metrics": {"a": {"baseline": 0.5}},
        })
        with pytest.raises(ValueError, match="floor"):
            load_expectations(p)

    def test_non_numeric_floor_raises(self, tmp_path):
        p = self._write(tmp_path, {
            "schema_version": GATE_SCHEMA_VERSION,
            "metrics": {"a": {"floor": "high"}},
        })
        with pytest.raises(ValueError, match="floor"):
            load_expectations(p)


class TestUpdateBaselines:
    def _exp(self) -> dict:
        return {"metrics": {"soh_r2": {"floor": 0.5, "baseline": 0.90}}}

    def test_moves_baseline_and_records_change(self):
        updated = update_baselines(self._exp(), {"soh_r2": 0.80}, "tuned trees")
        exp = updated["metrics"]["soh_r2"]
        assert exp["baseline"] == 0.80
        assert exp["approved_change"]["from"] == 0.90
        assert exp["approved_change"]["reason"] == "tuned trees"
        assert updated["change_log"][0]["metric"] == "soh_r2"

    def test_reason_is_mandatory(self):
        with pytest.raises(ValueError, match="reason"):
            update_baselines(self._exp(), {"soh_r2": 0.80}, "  ")

    def test_unchanged_value_records_nothing(self):
        updated = update_baselines(self._exp(), {"soh_r2": 0.90}, "no-op")
        assert updated.get("change_log") is None
        assert "approved_change" not in updated["metrics"]["soh_r2"]

    def test_commit_attached(self):
        updated = update_baselines(self._exp(), {"soh_r2": 0.80}, "why", commit="abc123")
        assert updated["change_log"][0]["commit"] == "abc123"

    def test_unknown_metrics_ignored(self):
        updated = update_baselines(self._exp(), {"not_tracked": 0.1}, "why")
        assert updated.get("change_log") is None

    def test_update_then_gate_passes(self):
        """The workflow loop closes: a recorded change lets the gate pass."""
        updated = update_baselines(self._exp(), {"soh_r2": 0.80}, "tuned trees")
        r = check_metric("soh_r2", 0.80, updated["metrics"]["soh_r2"])
        assert r["verdict"] == PASS


# ── fingerprints ────────────────────────────────────────────────────────


class TestFingerprints:
    def test_digest_stable_for_identical_data(self):
        a = cell_digest(_cycles_df(seed=1))
        b = cell_digest(_cycles_df(seed=1))
        assert a == b

    def test_digest_changes_when_values_change(self):
        a = cell_digest(_cycles_df(seed=1))
        b = cell_digest(_cycles_df(seed=1, fade=0.0006))
        assert a != b

    def test_digest_insensitive_to_row_order(self):
        df = _cycles_df(seed=1)
        a = cell_digest(df)
        b = cell_digest(df.iloc[::-1].reset_index(drop=True))
        assert a == b

    def test_environment_snapshot_covers_tracked_packages(self):
        env = environment_snapshot()
        for key in ("python", "platform", "numpy", "pandas", "scikit-learn"):
            assert key in env
            assert isinstance(env[key], str) and env[key]

    def test_dataset_fingerprint_order_independent(self):
        cells = {"A": _cycles_df(seed=1), "B": _cycles_df(seed=2)}
        fp1 = dataset_fingerprint(cells)
        fp2 = dataset_fingerprint({"B": cells["B"], "A": cells["A"]})
        assert fp1["dataset_sha256"] == fp2["dataset_sha256"]

    def test_compare_identical(self):
        fp = dataset_fingerprint({"A": _cycles_df(seed=1)})
        assert compare_fingerprints(fp, fp)["match"]

    def test_compare_detects_changed_cell(self):
        fp1 = dataset_fingerprint({"A": _cycles_df(seed=1), "B": _cycles_df(seed=2)})
        fp2 = dataset_fingerprint({"A": _cycles_df(seed=1, fade=0.001), "B": _cycles_df(seed=2)})
        cmp = compare_fingerprints(fp1, fp2)
        assert not cmp["match"]
        assert cmp["changed_cells"] == ["A"]

    def test_compare_detects_added_and_removed(self):
        fp1 = dataset_fingerprint({"A": _cycles_df(seed=1)})
        fp2 = dataset_fingerprint({"A": _cycles_df(seed=1), "B": _cycles_df(seed=2)})
        cmp = compare_fingerprints(fp1, fp2)
        assert cmp["added_cells"] == ["B"] and cmp["removed_cells"] == []
        cmp2 = compare_fingerprints(fp2, fp1)
        assert cmp2["removed_cells"] == ["B"]


# ── metric history / drift ──────────────────────────────────────────────


def _run(dataset, soh, rul=None, ts="2026-01-01T00:00:00", fv="v12", mk="gbrt", commit=None, run_id=None):
    return {
        "run_id": run_id or f"{dataset}_{ts}_{soh}",
        "dataset": dataset,
        "model_kind": mk,
        "soh_r2": soh,
        "rul_r2": rul,
        "baseline_soh_r2": None,
        "feature_version": fv,
        "timestamp": ts,
        "git_commit": commit,
    }


class TestMetricHistory:
    def test_series_sorted_chronologically(self):
        runs = [_run("nasa", 0.9, ts="2026-03-01"), _run("nasa", 0.8, ts="2026-01-01")]
        series = metric_history(runs, "nasa", "gbrt", "soh_r2")
        assert [e["value"] for e in series] == [0.8, 0.9]

    def test_filters_dataset_and_model_kind(self):
        runs = [_run("nasa", 0.9), _run("zhu", 0.99), _run("nasa", 0.7, mk="pinn")]
        series = metric_history(runs, "nasa", "gbrt", "soh_r2")
        assert len(series) == 1

    def test_skips_none_values(self):
        runs = [_run("nasa", None), _run("nasa", 0.9)]
        assert len(metric_history(runs, "nasa", "gbrt", "soh_r2")) == 1


class TestDriftReport:
    def test_no_drift_when_flat(self):
        runs = [_run("nasa", 0.90, ts="2026-01-01"), _run("nasa", 0.899, ts="2026-02-01")]
        report = drift_report(runs)
        assert report["n_alerts"] == 0

    def test_alert_on_beyond_tolerance_move(self):
        runs = [_run("nasa", 0.90, ts="2026-01-01"), _run("nasa", 0.70, ts="2026-02-01")]
        report = drift_report(runs)
        assert report["n_alerts"] == 1
        alert = report["datasets"][0]["alerts"][0]
        assert alert["metric"] == "soh_r2"
        assert alert["delta"] == pytest.approx(-0.20)

    def test_feature_version_change_is_not_drift(self):
        runs = [
            _run("nasa", 0.90, ts="2026-01-01", fv="v11"),
            _run("nasa", 0.70, ts="2026-02-01", fv="v12"),
        ]
        report = drift_report(runs)
        assert report["n_alerts"] == 0
        changes = report["datasets"][0]["feature_changes"]
        assert changes and changes[0]["from_version"] == "v11"

    def test_study_populations_excluded(self):
        runs = [
            _run("nasa_robustness", 0.90, ts="2026-01-01"),
            _run("nasa_robustness", 0.10, ts="2026-02-01"),
        ]
        assert drift_report(runs)["datasets"] == []

    def test_report_format_renders_alerts(self):
        runs = [_run("nasa", 0.90, ts="2026-01-01"), _run("nasa", 0.70, ts="2026-02-01")]
        text = format_drift_report(drift_report(runs))
        assert "[DRIFT]" in text


# ── replication ─────────────────────────────────────────────────────────


class TestReplication:
    def _bundle(self, tmp_path, cell_data) -> dict:
        """A minimal sealed bundle built the way the publisher builds it."""
        from batlab.validation.fingerprints import cell_digest, environment_snapshot

        digests = {cid: cell_digest(df) for cid, df in cell_data.items()}
        bench = {"metrics": {"soh_r2": 0.9}, "seed": 42}
        (tmp_path / "benchmark.json").write_text(json.dumps(bench), encoding="utf-8")
        bundle = {
            "schema": REPLICATION_SCHEMA,
            "schema_version": REPLICATION_SCHEMA_VERSION,
            "cell_ids": sorted(cell_data),
            "cell_digests": digests,
            "environment": environment_snapshot(),
            "reported": {"soh_r2": 0.9},
            "feature_version": "v-test",
            "seed": 42,
            "files": {"benchmark.json": self._sha(tmp_path / "benchmark.json")},
        }
        (tmp_path / "replication.json").write_text(json.dumps(bundle), encoding="utf-8")
        return bundle

    @staticmethod
    def _sha(p: Path) -> str:
        import hashlib

        return hashlib.sha256(p.read_bytes()).hexdigest()

    def test_load_bundle_rejects_wrong_schema(self, tmp_path):
        (tmp_path / "replication.json").write_text(json.dumps({"schema": "other"}))
        with pytest.raises(ValueError, match="schema"):
            load_bundle(tmp_path)

    def test_load_bundle_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle(tmp_path)

    def test_seal_passes_when_files_match(self, tmp_path):
        cell_data = {"A": _cycles_df(seed=1)}
        bundle = self._bundle(tmp_path, cell_data)
        result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path)
        seal = next(c for c in result["checks"] if c["name"] == "seal")
        assert seal["status"] == "pass"
        assert result["verdict"] == "pass"

    def test_seal_fails_when_file_tampered(self, tmp_path):
        cell_data = {"A": _cycles_df(seed=1)}
        self._bundle(tmp_path, cell_data)
        # Tamper after sealing.
        (tmp_path / "benchmark.json").write_text('{"metrics": {"soh_r2": 0.99}}', encoding="utf-8")
        result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path)
        seal = next(c for c in result["checks"] if c["name"] == "seal")
        assert seal["status"] == "fail"
        assert result["verdict"] == "fail"

    def test_data_identity_pass_and_fail(self, tmp_path):
        cell_data = {"A": _cycles_df(seed=1)}
        bundle = self._bundle(tmp_path, cell_data)
        ok = verify_bundle(bundle, cell_data=cell_data)
        assert next(c for c in ok["checks"] if c["name"] == "data-identity")["status"] == "pass"

        changed = {"A": _cycles_df(seed=1, fade=0.001)}
        bad = verify_bundle(bundle, cell_data=changed)
        ident = next(c for c in bad["checks"] if c["name"] == "data-identity")
        assert ident["status"] == "fail" and "changed cells" in ident["detail"]

    def test_missing_cell_fails_identity(self, tmp_path):
        bundle = self._bundle(tmp_path, {"A": _cycles_df(seed=1), "B": _cycles_df(seed=2)})
        result = verify_bundle(bundle, cell_data={"A": _cycles_df(seed=1)})
        ident = next(c for c in result["checks"] if c["name"] == "data-identity")
        assert ident["status"] == "fail" and "missing" in ident["detail"]

    def test_environment_mismatch_warns_not_fails(self, tmp_path):
        cell_data = {"A": _cycles_df(seed=1)}
        bundle = self._bundle(tmp_path, cell_data)
        bundle["environment"] = dict(bundle["environment"], numpy="0.0.1")
        result = verify_bundle(bundle, cell_data=cell_data)
        env = next(c for c in result["checks"] if c["name"] == "environment")
        assert env["status"] == "warn"
        assert result["verdict"] == "pass"  # warn alone doesn't fail

    def test_recompute_skipped_without_data(self, tmp_path):
        bundle = self._bundle(tmp_path, {"A": _cycles_df(seed=1)})
        result = verify_bundle(bundle, recompute=True)
        rec = next(c for c in result["checks"] if c["name"] == "recompute")
        assert rec["status"] == "warn"

    def test_recompute_feature_version_mismatch_fails(self, tmp_path):
        cell_data = {"A": _cycles_df(seed=1), "B": _cycles_df(seed=2)}
        bundle = self._bundle(tmp_path, cell_data)
        bundle["feature_version"] = "v-ancient"
        result = verify_bundle(bundle, cell_data=cell_data, recompute=True)
        rec = next(c for c in result["checks"] if c["name"] == "recompute")
        assert rec["status"] == "fail"

    def test_recompute_matches_reported(self, tmp_path):
        """The end-to-end promise: a bundle recomputes from the third
        party's byte-identical data. run_lco's own reported number is used
        as the claim, so the check must reproduce it exactly."""
        from batlab.validation.lco import run_lco

        cell_data = {"A": _cycles_df(n=120, seed=1), "B": _cycles_df(n=120, seed=2, fade=0.0007)}
        result = run_lco(cell_data, seed=42)
        reported = {"soh_r2": result["soh_r2"]}
        bundle = {
            "schema": REPLICATION_SCHEMA,
            "schema_version": REPLICATION_SCHEMA_VERSION,
            "cell_ids": sorted(cell_data),
            "cell_digests": {cid: cell_digest(df) for cid, df in cell_data.items()},
            "environment": environment_snapshot(),
            "reported": reported,
            "feature_version": __import__("batlab.features.engineering", fromlist=["FEATURE_VERSION"]).FEATURE_VERSION,
            "seed": 42,
        }
        ver = verify_bundle(bundle, cell_data=cell_data, recompute=True)
        rec = next(c for c in ver["checks"] if c["name"] == "recompute")
        assert rec["status"] == "pass", rec["detail"]
