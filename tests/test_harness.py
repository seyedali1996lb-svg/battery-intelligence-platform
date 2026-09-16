"""Tests for batlab.harness — the model-agnostic validation harness.

What these tests defend, in order of how much damage their absence would do:

1. The seam does not move the numbers. validate_forecaster() with no model
   passed must reproduce run_lco() / run_lco_quantiles() EXACTLY, or the
   platform's published figures stop describing the code that ships.
2. The factory contract holds: one fresh model per fold and per target, so no
   fold can be fitted on another fold's state.
3. A pre-fitted estimator is refused rather than deep-copied.
4. The honesty rules survive a foreign model: label provenance, the gate's
   not-evaluable handling (NaN must not read as "passed its floor"), withheld
   claims, and a distribution-free interval for a point-only model.
5. A sealed bundle verifies end to end, and a bundle published for a
   non-default model cannot be silently recomputed with the default one.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_cycles_df
from sklearn.linear_model import Ridge

from batlab.harness import (
    CallableForecaster,
    SklearnForecaster,
    as_factory,
    default_forecaster,
    default_interval_forecaster,
    format_report,
    has_predict_interval,
    seal_bundle,
    torch_forecaster,
    validate_forecaster,
)
from batlab.validation.calibration import run_lco_quantiles
from batlab.validation.lco import run_lco
from batlab.validation.replication import load_bundle, verify_bundle


# ---------------------------------------------------------------------------
# Fleets
# ---------------------------------------------------------------------------

def _eol_fleet(n_cells: int = 3, n_cycles: int = 180, fade: float = 0.005) -> dict:
    """Cells that reach the 80% EOL threshold in-window (measured RUL labels)."""
    return {
        f"C{i}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=fade * (1.0 + 0.08 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


def _censored_fleet(n_cells: int = 3, n_cycles: int = 340) -> dict:
    """Cells that never reach EOL in-window — the label-provenance refusal case."""
    return {
        f"C{i}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=0.0004 * (1.0 + 0.1 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


class _FitCounter:
    """Counts fit() calls on the models a factory produces."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self.fits = 0

    def __call__(self):
        inner = self._factory()
        outer = self

        class _Counted:
            def fit(self, X, y):
                outer.fits += 1
                inner.fit(X, y)
                return self

            def predict(self, X):
                return inner.predict(X)

        return _Counted()


@pytest.fixture(scope="module")
def eol_fleet():
    return _eol_fleet()


@pytest.fixture(scope="module")
def censored_fleet():
    return _censored_fleet()


@pytest.fixture(scope="module")
def default_report(eol_fleet):
    """The full default run: LCO + prospective + intervals + baselines."""
    return validate_forecaster(eol_fleet, dataset="synth-eol")


@pytest.fixture(scope="module")
def ridge_report(eol_fleet):
    """A foreign point-only model, LCO split only (fast, exercises conformal)."""
    return validate_forecaster(
        eol_fleet, model=Ridge(alpha=1.0), dataset="synth-eol", splits=("lco",)
    )


# ---------------------------------------------------------------------------
# 1. The seam does not move the platform's numbers
# ---------------------------------------------------------------------------

def test_default_model_reproduces_run_lco_exactly(eol_fleet, default_report):
    reference = run_lco(eol_fleet, seed=42)
    assert default_report["lco"]["soh_r2"] == pytest.approx(reference["soh_r2"], abs=0.0)
    assert default_report["lco"]["rul_r2"] == pytest.approx(reference["rul_r2"], abs=0.0)
    assert default_report["lco"]["soh_mae"] == pytest.approx(reference["soh_mae"], abs=0.0)
    assert default_report["lco"]["rul_reliable"] == reference["rul_reliable"]
    assert default_report["lco"]["confidence_intervals"] == reference["confidence_intervals"]


def test_default_run_uses_the_platforms_own_interval_path(default_report):
    """model=None must reproduce the platform's PUBLISHED calibration.

    The default call has no model to take an interval from, and the platform
    has always published a quantile + conformal calibration (not a residual
    interval) — so the harness must use the platform's own interval triple
    rather than inventing a second, unpublished number for the same model.
    """
    assert default_report["model"]["interval_capable"] is True
    assert default_report["calibration"]["status"] == "quantile_interval"
    assert "platform default" in default_report["model"]["interval_source"]
    assert default_report["calibration"]["calibrated_coverage"] is not None


def test_default_interval_model_reproduces_run_lco_quantiles(eol_fleet):
    report = validate_forecaster(
        eol_fleet,
        model=default_interval_forecaster(seed=42),
        dataset="synth-eol",
        splits=("lco",),
    )
    reference = run_lco_quantiles(eol_fleet, seed=42)
    assert report["model"]["interval_capable"] is True
    assert report["calibration"]["status"] == "quantile_interval"
    assert report["calibration"]["raw_coverage"] == pytest.approx(
        reference["rul_interval_coverage"], abs=0.0
    )
    assert report["calibration"]["calibrated_coverage"] == pytest.approx(
        reference["recalibrated_coverage"], abs=0.0
    )
    assert report["calibration"]["e_star_cycles"] == pytest.approx(
        reference["global_e_star"], abs=0.0
    )


def test_run_lco_default_path_has_no_predictions_key(eol_fleet):
    """include_predictions is opt-in, so existing callers see the same dict."""
    fold = next(iter(run_lco(eol_fleet, seed=42)["per_cell"].values()))
    assert "predictions" not in fold


def test_harness_drops_out_of_fold_predictions_from_the_report(ridge_report):
    """They are an input to calibration, not part of the report (or a bundle)."""
    for fold in ridge_report["lco"]["per_cell"].values():
        assert "predictions" not in fold
    json.dumps(ridge_report)  # also proves the report is JSON-serializable


# ---------------------------------------------------------------------------
# 2 + 3. Factory contract and prefit refusal
# ---------------------------------------------------------------------------

def test_factory_is_called_once_per_fold_and_target(eol_fleet):
    counted = _FitCounter(default_forecaster(seed=42))
    report = validate_forecaster(
        eol_fleet,
        model=counted,
        intervals=False,
        splits=("lco",),
        include_baselines=False,
    )
    n_cells = len(eol_fleet)
    assert counted.fits == n_cells * 2, (
        "each leave-cell-out fold must fit one fresh SOH model and one fresh RUL "
        "model — a fold fitted on another fold's state is the leak this guards"
    )
    assert report["lco"]["per_cell"] and len(report["lco"]["per_cell"]) == n_cells


def test_prefitted_estimator_is_refused(eol_fleet):
    X = np.arange(20, dtype=float).reshape(-1, 1)
    fitted = Ridge(alpha=1.0).fit(X, X.ravel())
    with pytest.raises(ValueError, match="already fitted"):
        validate_forecaster(eol_fleet, model=fitted)


def test_as_factory_normalizes_shapes():
    assert as_factory(None) is None
    assert as_factory(None, default=default_forecaster()) is not None
    factory = as_factory(default_interval_forecaster())
    assert has_predict_interval(factory())
    assert not has_predict_interval(as_factory(Ridge())())
    with pytest.raises(TypeError):
        as_factory("not a model")


def test_callable_forecaster_works_and_reports_its_label():
    def fit_fn(X, y):
        mean = float(np.asarray(y, dtype=float).mean())
        return {"mean": mean}

    def predict_fn(state, X):
        return np.full(len(X), state["mean"])

    model = CallableForecaster(fit_fn, predict_fn, label="mean-baseline")
    report = validate_forecaster(
        _eol_fleet(),
        model=model,
        intervals=False,
        splits=("lco",),
        include_baselines=False,
        gate={"metrics": {"soh_r2": {"floor": 0.9}}},
    )
    # A training-mean predictor is the anti-model: ~0 R², so a 0.9 floor fails.
    assert report["gate"]["verdict"] == "fail"
    assert report["verdict"]["status"] == "fail"
    assert report["model"]["identity"]["class"] == "CallableForecaster"


def test_torch_forecaster_contract():
    """Either torch is installed (and the loop trains) or the ImportError names it."""
    def module_factory(in_features: int):
        import torch

        return torch.nn.Sequential(torch.nn.Linear(in_features, 8), torch.nn.Linear(8, 1))

    factory = torch_forecaster(module_factory, epochs=2, seed=0)
    cells = _eol_fleet(n_cells=2, n_cycles=120)
    try:
        report = validate_forecaster(
            cells, model=factory, intervals=False, splits=("lco",), include_baselines=False
        )
    except ImportError as exc:
        assert "torch" in str(exc)  # batlab must not require PyTorch
    else:
        assert np.isfinite(report["lco"]["soh_r2"])


# ---------------------------------------------------------------------------
# 4. The honesty rules survive a foreign model
# ---------------------------------------------------------------------------

def test_leakage_lint_runs_and_passes(default_report):
    assert default_report["leakage_lint"]["ok"] is True
    assert len(default_report["leakage_lint"]["checked"]) == 2


def test_label_provenance_reports_measured_fraction(default_report):
    prov = default_report["label_provenance"]
    assert prov["observed_rows"] > 0
    assert prov["extrapolated_rows"] == 0
    assert prov["observed_fraction"] == 1.0


def test_censored_fleet_withholds_the_rul_claim(censored_fleet):
    report = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), dataset="synth-censored"
    )
    prov = report["label_provenance"]
    assert prov["observed_rows"] == 0
    assert prov["extrapolated_rows"] > 0
    assert report["lco"]["rul_reliable"] is False
    assert any("no measured end-of-life rows" in w for w in report["verdict"]["withheld"])
    # A NaN metric must survive as null, not as an invalid JSON literal.
    json.dumps(report)
    assert report["lco"]["rul_r2"] is None


def test_gate_reads_a_nan_metric_as_not_evaluable(censored_fleet):
    """nan < floor is False — an unnormalized gate would call this a pass."""
    floor = {"metrics": {"rul_r2": {"floor": 0.3}}}
    report = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), gate=floor, include_baselines=False
    )
    assert report["gate"]["verdict"] == "fail"
    failure = next(r for r in report["gate"]["results"] if r["name"] == "rul_r2")
    assert "not evaluable" in failure["detail"]

    allowed = {"metrics": {"rul_r2": {"floor": 0.3, "allow_not_evaluable": True}}}
    report_ok = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), gate=allowed, include_baselines=False
    )
    assert report_ok["gate"]["verdict"] == "pass"


def test_gate_reports_not_checked_without_expectations(default_report):
    assert default_report["gate"]["verdict"] == "not_checked"
    assert default_report["verdict"]["status"] == "not_checked"
    assert "NOT CHECKED" in default_report["verdict"]["summary"]
    assert set(default_report["gate"]["untracked"]) >= {"soh_r2", "rul_r2"}


def test_conformal_residual_interval_for_a_point_only_model(ridge_report):
    calibration = ridge_report["calibration"]
    assert calibration["status"] == "conformal_residual"
    assert calibration["raw_coverage"] is None  # the model claims no interval
    assert 0.0 <= calibration["calibrated_coverage"] <= 1.0
    # Nominal 80% on a smooth synthetic fleet: the measured coverage must be in
    # the right neighbourhood, and it is reported as measured, not as nominal.
    assert 0.6 <= calibration["calibrated_coverage"] <= 1.0
    assert calibration["per_cell"]
    assert all(v["mean_width_cycles"] >= 0.0 for v in calibration["per_cell"].values())
    assert any("measured coverage" in c for c in ridge_report["verdict"]["claims"])


def test_interval_skipped_when_disabled(censored_fleet):
    report = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), include_baselines=False
    )
    assert report["calibration"]["status"] == "skipped"


def test_split_selection_is_disclosed(censored_fleet):
    lco_only = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), include_baselines=False
    )
    assert lco_only["prospective"]["status"] == "skipped"
    assert any("forecasting" in w for w in lco_only["verdict"]["withheld"])

    prospective_only = validate_forecaster(
        censored_fleet, intervals=False, splits=("prospective",), include_baselines=False
    )
    assert prospective_only["lco"] is None
    assert prospective_only["calibration"]["status"] == "skipped"

    # With intervals ON but no LCO section, calibration has no folds to
    # calibrate against — a distinct, disclosed reason.
    no_folds = validate_forecaster(
        censored_fleet, intervals=True, splits=("prospective",), include_baselines=False
    )
    assert no_folds["calibration"]["status"] == "not_evaluable"
    assert "lco" in no_folds["calibration"]["reason"]


def test_baselines_can_be_withheld_but_say_so(censored_fleet):
    report = validate_forecaster(
        censored_fleet, intervals=False, splits=("lco",), include_baselines=False
    )
    assert report["baselines"]["status"] == "withheld"
    assert "no floor beside them" in report["baselines"]["reason"]


def test_tie_comparison_does_not_claim_a_marginal_win():
    """1.000 vs 1.000 must read as a tie, not as a win on float noise."""
    from batlab.harness.harness import _compare

    assert _compare(1.0, 1.0) == "matches"
    assert _compare(1.0, 1.0 - 1e-12) == "matches"
    assert _compare(0.9, 0.5) == "beats"
    assert _compare(0.2, 0.7) == "LOSES to"
    assert _compare(None, 0.7) == "not comparable to"


def test_baselines_and_the_verdict_agree_about_losing(default_report):
    """The baselines exist to make a losing model read as losing.

    On this steeply-fading synthetic fleet the default GBRT interpolates well
    under leave-cell-out and extrapolates badly under the prospective split —
    it LOSES to a per-cell straight line. The verdict is required to say so in
    the claim itself rather than leaving the reader to compare two numbers.
    """
    baselines = default_report["baselines"]
    assert baselines["status"] == "computed"
    assert baselines["lco_trend_r2"] is not None
    assert baselines["lco_rul_formula_r2"] is not None
    assert baselines["prospective_trend_r2"] is not None

    claims = " | ".join(default_report["verdict"]["claims"])
    assert "trend baseline" in claims
    assert "LOSES to" in claims or "matches" in claims


def test_text_report_covers_every_section(default_report):
    text = format_report(default_report)
    for expected in (
        "batlab-forecaster-harness",
        "model:",
        "lint:    PASS",
        "labels:",
        "leave-cell-out",
        "baselines",
        "interval calibration",
        "prospective",
        "metric gate: NOT_CHECKED",
        "verdict:",
    ):
        assert expected in text


def test_unknown_split_and_single_cell_are_rejected():
    with pytest.raises(ValueError, match="Unknown split"):
        validate_forecaster(_eol_fleet(2), splits=("holdout",))
    with pytest.raises(ValueError, match="at least 2 cells"):
        validate_forecaster({"only": make_cycles_df(n_cycles=120)})


# ---------------------------------------------------------------------------
# 5. Sealing and independent verification
# ---------------------------------------------------------------------------

def test_sealed_bundle_verifies_with_the_same_model(tmp_path, eol_fleet):
    model = default_forecaster(seed=42)
    report = validate_forecaster(
        eol_fleet, model=model, intervals=False, splits=("lco",), include_baselines=False
    )
    out = tmp_path / "bundle"
    replication = seal_bundle(report, eol_fleet, out, dataset="synth-eol")
    assert set(replication["files"]) == {"benchmark.json", "harness_report.json"}
    assert replication["model"]["factory"]  # model identity travels with the bundle

    bundle = load_bundle(out)
    result = verify_bundle(bundle, bundle_dir=out, cell_data=eol_fleet, recompute=True,
                           forecaster=model)
    assert result["verdict"] == "pass", result
    checks = {c["name"]: c["status"] for c in result["checks"]}
    assert checks["seal"] == "pass"
    assert checks["data-identity"] == "pass"
    assert checks["recompute"] == "pass"


def test_bundle_for_a_non_default_model_refuses_the_default_recompute(tmp_path, eol_fleet):
    report = validate_forecaster(
        eol_fleet, model=Ridge(alpha=1.0), intervals=False, splits=("lco",),
        include_baselines=False,
    )
    out = tmp_path / "bundle-ridge"
    seal_bundle(report, eol_fleet, out, dataset="synth-eol")

    # Recomputing a non-default bundle without saying which model is not a
    # number mismatch to blame on the data — it is a missing input.
    without_model = verify_bundle(load_bundle(out), bundle_dir=out, cell_data=eol_fleet,
                                  recompute=True)
    assert without_model["verdict"] == "fail"
    recompute = next(c for c in without_model["checks"] if c["name"] == "recompute")
    assert "non-default model" in recompute["detail"]

    # With the model that was actually published, it verifies.
    with_model = verify_bundle(load_bundle(out), bundle_dir=out, cell_data=eol_fleet,
                               recompute=True, forecaster=Ridge(alpha=1.0))
    assert with_model["verdict"] == "pass", with_model

    # And the default-model bundle above verifies without any --model.
    default_report = validate_forecaster(
        eol_fleet, intervals=False, splits=("lco",), include_baselines=False
    )
    default_out = tmp_path / "bundle-default"
    seal_bundle(default_report, eol_fleet, default_out, dataset="synth-eol")
    default_result = verify_bundle(load_bundle(default_out), bundle_dir=default_out,
                                   cell_data=eol_fleet, recompute=True)
    assert default_result["verdict"] == "pass", default_result


def test_seal_requires_the_lco_section(tmp_path, eol_fleet):
    report = validate_forecaster(
        eol_fleet, intervals=False, splits=("prospective",), include_baselines=False
    )
    with pytest.raises(ValueError, match="needs an LCO section"):
        seal_bundle(report, eol_fleet, tmp_path / "nope")


def test_report_is_not_mutated_by_formatting(default_report):
    before = copy.deepcopy(default_report)
    format_report(default_report)
    assert default_report == before


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CLI_MODULE = '''
import numpy as np
import pandas as pd


def load_cells():
    cells = {}
    for i in range(3):
        n = 180
        cycles = np.arange(1, n + 1)
        capacity = 2.0 - 0.005 * (1.0 + 0.08 * i) * cycles
        df = pd.DataFrame({
            "cycle_number": cycles,
            "capacity_ah": capacity,
            "resistance_ohm": 0.05 + 0.004 * i + 5e-5 * cycles,
            "temperature_c": np.full(n, 25.0),
        })
        df["soh_pct"] = df["capacity_ah"] / 2.0 * 100.0
        cells[f"CLI{i}"] = df
    return cells
'''


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    from batlab.harness.harness import main as harness_main

    module = tmp_path / "cli_fixture_mod.py"
    module.write_text(_CLI_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    json_out = tmp_path / "report.json"
    seal_dir = tmp_path / "sealed"
    code = harness_main([
        "--loader", "cli_fixture_mod:load_cells",
        "--no-intervals",
        "--seed", "42",
        "--json", str(json_out),
        "--seal", str(seal_dir),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "batlab-forecaster-harness" in printed
    assert "metric gate: NOT_CHECKED" in printed

    report = json.loads(Path(json_out).read_text())
    assert report["dataset"]["n_cells"] == 3
    assert (seal_dir / "replication.json").exists()


def test_cli_exits_nonzero_when_a_floor_fails(tmp_path, monkeypatch, capsys):
    from batlab.harness.harness import main as harness_main

    # A distinct module name per test: 'module:function' specs are cached in
    # sys.modules, so reusing the name would import the previous test's module.
    module = tmp_path / "cli_gate_fixture_mod.py"
    module.write_text(
        _CLI_MODULE
        + '''

def make_model():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)
''',
        encoding="utf-8",
    )
    # A ceiling of 0 on an MAE is unattainable by anything that is not
    # exact — the point is the exit code, not the model's quality (a Ridge
    # fit on this smooth synthetic fleet clears any realistic R² floor).
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "schema_version": 1,
        "metrics": {"soh_mae": {"ceiling": 0.0}},
    }), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    code = harness_main([
        "--loader", "cli_gate_fixture_mod:load_cells",
        "--model", "cli_gate_fixture_mod:make_model",
        "--gate", str(gate),
        "--no-intervals",
    ])
    capsys.readouterr()
    assert code == 1
