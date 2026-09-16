"""
The model-agnostic validation harness — grade any forecaster by this
platform's honesty rules, in one call.

Why this exists
---------------
Everything in batlab.validation was built to answer one question honestly:
*will this model perform on a battery it has never seen?* Leave-cell-out
folds, per-row label provenance, conformal interval calibration, a declared
floor/ceiling on every headline number, and a sealed bundle a third party can
re-derive. Those rules were never GBRT-specific — but they were welded to
this platform's GBRT, which made them unusable for anyone else's model, and
made "our methodology is sound" a claim about a dataset rather than about a
method.

This module is the seam removed. One call:

    from batlab.harness import validate_forecaster

    report = validate_forecaster(cells, model=my_model, gate="floors.json")

runs, in the order a skeptic would insist on:

  1. LEAKAGE LINT     — can any model feature be read by the RUL label's own
                        generating expression? (the defect that produced the
                        platform's former 0.9994 RUL R², caught mechanically)
  2. LABEL PROVENANCE — how many RUL rows carry a MEASURED end-of-life label,
                        and how many carry a closed-form extrapolation? A
                        model scored on extrapolated labels is graded on
                        formula recovery, so those rows are reported
                        separately and never allowed to set the headline.
  3. LEAVE-CELL-OUT   — new-cell generalization, folds per cell, with the
                        trivial trend baseline beside it so "how much of this
                        R² is the shape of aging curves?" is answerable.
  4. INTERVAL         — nominal-vs-measured coverage of an 80% interval, with
                        conformal recalibration fitted on the other folds
                        only. A point-only model still gets an interval: the
                        harness builds a distribution-free one from that
                        model's own out-of-fold residuals.
  5. PROSPECTIVE      — the same-cell forecasting split: train on each cell's
                        first half, score the second. This is the only test
                        that separates forecasting from curve interpolation,
                        and the only one where most models look bad.
  6. METRIC GATE      — declared floors/ceilings, enforced. Optional, and
                        its absence is reported as NOT CHECKED rather than
                        as a pass.

What it refuses to claim
------------------------
The report carries an explicit `withheld` list. A fleet whose cells never
reach end-of-life in-window gets *no* RUL number (not a caveated one); a
model that loses to the closed-form fade formula is reported as losing; an
interval whose measured coverage is 62% is reported at 62%, not at its
nominal 80%. If a number cannot be measured honestly, the harness says so
instead of producing one — that behaviour is the product, not a limitation of
it.

What it is not
--------------
Not a leaderboard and not a hyperparameter search. It grades one model on one
fleet and tells you what that grade does and does not support. Numbers here
describe cells that look like the ones you validated on, which is why the
fingerprint of the dataset it ran on travels in every report and every sealed
bundle.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from batlab.harness.forecaster import (
    ForecasterLike,
    as_factory,
    default_forecaster,
    default_interval_forecaster,
    forecaster_identity,
    has_predict_interval,
)
from batlab.validation.calibration import (
    NOMINAL_INTERVAL_COVERAGE,
    empirical_coverage,
    interval_width_mean,
    run_lco_quantiles,
)
from batlab.validation.fingerprints import dataset_fingerprint, environment_snapshot
from batlab.validation.leakage_lint import run_lint
from batlab.validation.lco import RUL_RELIABLE_FLOOR, run_lco, unwrap_cell_data
from batlab.validation.manifest import export_benchmark_results
from batlab.validation.metric_gate import evaluate_gate, format_gate_report, load_expectations
from batlab.validation.prospective import (
    DEFAULT_TRAIN_FRACTION,
    run_prospective,
    rul_formula_baseline_prospective,
    trivial_soh_baseline_prospective,
)
from batlab.validation.trivial_baseline import baseline_lco_r2, rul_formula_baseline_lco

__all__ = [
    "HARNESS_SCHEMA",
    "HARNESS_SCHEMA_VERSION",
    "DEFAULT_SPLITS",
    "format_report",
    "seal_bundle",
    "validate_forecaster",
]

HARNESS_SCHEMA = "batlab-forecaster-harness"
HARNESS_SCHEMA_VERSION = 1

DEFAULT_SPLITS = ("lco", "prospective")

# Minimum rows for a fold's residual pool to define quantiles at all. Below
# this the fold is skipped and disclosed, never padded with a fabricated
# quantile from too little evidence.
_MIN_CALIBRATION_ROWS = 5


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate_forecaster(
    cell_data: dict,
    model: ForecasterLike = None,
    *,
    seed: int = 42,
    featured: "dict | None" = None,
    splits: "tuple[str, ...]" = DEFAULT_SPLITS,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    intervals: bool = True,
    nominal_coverage: float = NOMINAL_INTERVAL_COVERAGE,
    gate: "dict | None" = None,
    gate_path: "str | Path | None" = None,
    include_baselines: bool = True,
    dataset: "str | None" = None,
) -> dict:
    """Grade `model` on `cell_data` under every honesty rule this platform has.

    Parameters
    ----------
    cell_data : {cell_id: cycles DataFrame}. Either shape the platform's
        loaders and build_battery() produce is accepted (a {"cycles": df}
        wrapper is unwrapped), and each frame must carry the standardized
        cycle schema — the harness builds features from raw cycles itself, so
        what it grades is the model, not a pre-processed matrix you could
        have shaped favourably.
    model : what to grade. None = the platform's own configuration, in which
        case the default call reproduces this platform's published point
        metrics AND its published calibration (point + Q10/Q90 quantile
        regressors for the interval section). Otherwise anything
        batlab.harness.as_factory() accepts: an unfitted sklearn estimator, a
        zero-argument factory returning a fresh model, a CallableForecaster
        (PyTorch, a custom numerical fit), or an interval model exposing
        predict_interval().
    seed : every fold's random_state, and the bootstrap seed. Stamped into
        the report so the run is reproducible.
    featured : optional prebuilt {cell_id: featured DataFrame} cache (the app
        holds one across pages). Omit and the harness builds it once for all
        six sections, so no section can see different features than another.
    splits : which evaluations to run — any of ("lco", "prospective"). The
        LCO number is new-cell generalization; the prospective number is
        same-cell forecasting. They are complements; reporting one without
        the other is how interpolation gets mistaken for skill.
    train_fraction : prospective train window per cell (default 0.5).
    intervals : measure interval coverage. An interval-capable model is
        calibrated with conformal quantile recalibration; a point-only model
        gets a distribution-free constant-width interval built from its own
        out-of-fold residuals. False skips the section entirely.
    nominal_coverage : the interval the model claims (0.80 = Q10/Q90).
    gate : expectations dict (see batlab.validation.metric_gate), or
    gate_path : path to an expectations JSON file. Neither supplied means the
        gate section reports NOT CHECKED — declared floors are the only thing
        that can turn a measured number into a failure, and their absence is
        not silently read as success.
    include_baselines : also run the trivial-trend and closed-form RUL
        baselines under the identical splits. Off is allowed, but the report
        then says which baselines were withheld — a bare R² invites exactly
        the misreading those baselines exist to prevent.
    dataset : a label for the report/bundle (e.g. "nasa"). Cosmetic; the
        dataset FINGERPRINT, not this name, is what makes two runs
        comparable.

    Returns
    -------
    A JSON-safe report dict: schema/version, dataset fingerprint +
    environment, model identity, and one section per check above, plus
    `verdict` with the claims this run supports and the claims it withholds.
    """
    cells = unwrap_cell_data(cell_data)
    if len(cells) < 2:
        raise ValueError(
            f"validate_forecaster needs at least 2 cells (got {len(cells)}): a "
            "leave-cell-out fold structure with one cell has no training data."
        )
    unknown = [s for s in splits if s not in ("lco", "prospective")]
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; supported: 'lco', 'prospective'.")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be strictly between 0 and 1.")

    featured_cache = featured if featured is not None else _build_features(cells)

    # The point metrics always come from the model under test. The INTERVAL
    # section is a separate question, and when no model was supplied the
    # honest answer is the platform's own interval configuration (point +
    # Q10/Q90 quantile regressors, conformally recalibrated) rather than a
    # residual interval the platform has never published — otherwise the
    # default call would report a calibration number that is not the one this
    # platform's own documentation states. The point estimates are identical
    # either way: the interval triple uses the same GBRT_PARAMS for its point
    # regressor.
    interval_model = model if model is not None else default_interval_forecaster(seed)
    interval_factory = as_factory(interval_model, default=default_forecaster(seed))
    if interval_factory is None:  # pragma: no cover - interval_model is never None
        raise ValueError("no interval forecaster available")
    interval_capable = has_predict_interval(interval_factory())

    report: dict = {
        "harness_schema": HARNESS_SCHEMA,
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset,
            "n_cells": len(cells),
            "fingerprint": dataset_fingerprint(cells),
            "environment": environment_snapshot(),
        },
        "model": {
            "identity": forecaster_identity(model),
            "interval_capable": interval_capable,
            "interval_source": (
                "platform default point + Q10/Q90 quantile regressors"
                if model is None
                else "the supplied model"
            ),
            "note": (
                "The model is re-fitted from scratch inside every fold from a "
                "factory call; a pre-fitted estimator is rejected at the seam."
            ),
        },
        "config": {
            "seed": seed,
            "splits": list(splits),
            "train_fraction": train_fraction,
            "intervals": intervals,
            "nominal_coverage": nominal_coverage,
            "include_baselines": include_baselines,
        },
    }

    # 1. Leakage lint — mechanical, cheap, and runs first: a leaking label
    #    formula makes every number below it meaningless, so there is no
    #    point spending fold fits before knowing.
    sample = next(iter(featured_cache.values()), None)
    lint = run_lint(featured_sample=sample)
    report["leakage_lint"] = {
        "ok": lint["ok"],
        "violations": lint["violations"],
        "checked": [
            "no quantity read by the RUL label's generating expression is a model feature",
            "build_features() emits rul_label_kind, so label provenance is checkable",
        ],
    }

    # 2 + 3. Leave-cell-out, with out-of-fold predictions retained when the
    #    calibration section will need them.
    need_predictions = intervals and not interval_capable
    if "lco" in splits:
        lco = run_lco(
            cells,
            seed=seed,
            featured=featured_cache,
            forecaster=model,
            include_predictions=need_predictions,
        )
        report["lco"] = lco
        report["label_provenance"] = _provenance_section(lco)
        report["baselines"] = _baseline_sections(
            cells, featured_cache, splits, train_fraction, include_baselines
        )
    else:
        report["lco"] = None
        report["label_provenance"] = {
            "status": "not_evaluated",
            "reason": "the 'lco' split was not requested",
        }
        report["baselines"] = _baseline_sections(
            cells, featured_cache, splits, train_fraction, include_baselines
        )

    # 4. Interval calibration.
    calibration = _calibration_section(
        cells, report["lco"], featured_cache, interval_model, seed, intervals,
        interval_capable, nominal_coverage,
    )
    # The out-of-fold predictions were an INPUT to the calibration section,
    # not part of the report: they are fold-sized arrays that would bloat a
    # sealed bundle and every JSON round trip. Dropped now that they are used.
    if isinstance(report.get("lco"), dict):
        for fold in (report["lco"].get("per_cell") or {}).values():
            if isinstance(fold, dict):
                fold.pop("predictions", None)
    report["calibration"] = calibration

    # 5. Prospective (same-cell forecasting).
    report["prospective"] = _prospective_section(
        cells, featured_cache, splits, train_fraction, seed, model
    )

    # 6. Metric gate.
    report["gate"] = _gate_section(report, gate, gate_path)

    report["verdict"] = _verdict_section(report)
    return _json_safe(report)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _build_features(cells: dict) -> dict:
    """Build the featured frame for every cell exactly once.

    One cache shared by all six sections: if each section built its own
    features, a subtle difference between two builds (a PyBaMM calibration
    that ran in one and not the other) would be invisible in the report while
    changing the numbers in it.
    """
    from batlab.features.engineering import build_features

    return {cid: build_features(df, cell_id=cid) for cid, df in cells.items()}


def _provenance_section(lco: dict) -> dict:
    """How many RUL rows carry a measured vs an extrapolated label."""
    observed = int(lco.get("n_rul_observed_rows") or 0)
    extrapolated = int(lco.get("n_rul_extrapolated_rows") or 0)
    total = observed + extrapolated
    coverage = float(lco.get("rul_label_coverage") or 0.0)
    return {
        "observed_rows": observed,
        "extrapolated_rows": extrapolated,
        "observed_fraction": coverage,
        "rul_reliable": bool(lco.get("rul_reliable")),
        "rul_reliable_floor": RUL_RELIABLE_FLOOR,
        "precedence": (
            "RUL is scored on measured-EOL rows only"
            if observed
            else "no measured-EOL rows: RUL is not evaluable on this fleet"
        ),
        "note": (
            "Extrapolated labels are generated by a closed-form fade formula "
            "(batlab.features.engineering), so a model scored on them is being "
            "graded on formula recovery. They never enter the headline RUL number."
        ),
    }


def _baseline_sections(
    cells: dict,
    featured: dict,
    splits: "tuple[str, ...]",
    train_fraction: float,
    include_baselines: bool,
) -> dict:
    """The floors a model has to beat, under the identical splits."""
    if not include_baselines:
        return {
            "status": "withheld",
            "reason": "include_baselines=False — the R² numbers below have no floor beside them",
        }

    out: dict = {"status": "computed"}
    if "lco" in splits:
        try:
            trend = baseline_lco_r2(cells, featured=featured)
            out["lco_trend_r2"] = trend.get("baseline_lco_r2", trend.get("baseline_soh_r2"))
            out["lco_trend_per_cell"] = {
                cid: v.get("r2") for cid, v in (trend.get("per_cell") or {}).items()
            }
        except Exception as exc:  # a baseline that cannot run is disclosed, not hidden
            out["lco_trend_r2"] = None
            out["lco_trend_error"] = f"{type(exc).__name__}: {exc}"
        try:
            formula = rul_formula_baseline_lco(cells, featured=featured)
            out["lco_rul_formula_r2"] = formula.get("rul_formula_baseline_r2")
        except Exception as exc:
            out["lco_rul_formula_r2"] = None
            out["lco_rul_formula_error"] = f"{type(exc).__name__}: {exc}"

    if "prospective" in splits:
        try:
            out["prospective_trend_r2"] = (
                trivial_soh_baseline_prospective(
                    cells, featured=featured, train_fraction=train_fraction
                ).get("baseline_soh_r2")
            )
        except Exception as exc:
            out["prospective_trend_r2"] = None
            out["prospective_trend_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["prospective_rul_formula_r2"] = (
                rul_formula_baseline_prospective(
                    cells, featured=featured, train_fraction=train_fraction
                ).get("rul_formula_baseline_r2")
            )
        except Exception as exc:
            out["prospective_rul_formula_r2"] = None
            out["prospective_rul_formula_error"] = f"{type(exc).__name__}: {exc}"

    out["note"] = (
        "A trend baseline is a per-cell straight line through cycle number with "
        "no engineered features; the RUL formula baseline is the closed form that "
        "generates extrapolated labels. A model that cannot beat the former has "
        "learned nothing beyond linear extrapolation; one that cannot beat the "
        "latter has not out-forecast a fade formula."
    )
    return out


def _calibration_section(
    cells: dict,
    lco: "dict | None",
    featured: dict,
    model: ForecasterLike,
    seed: int,
    intervals: bool,
    interval_capable: bool,
    nominal: float,
) -> dict:
    if not intervals:
        return {
            "status": "skipped",
            "reason": "intervals=False — interval coverage was not measured",
        }
    if lco is None:
        return {
            "status": "not_evaluable",
            "reason": "the 'lco' split was not requested, and conformal calibration needs its folds",
        }

    if interval_capable:
        try:
            quantile = run_lco_quantiles(
                cells, seed=seed, featured=featured, forecaster=model
            )
        except Exception as exc:
            return {
                "status": "not_evaluable",
                "method": "quantile_interval",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if not np.isfinite(quantile.get("rul_interval_coverage", float("nan"))):
            return {
                "status": "not_evaluable",
                "method": "quantile_interval",
                "nominal": nominal,
                "reason": (
                    "no measured-EOL rows: an interval around a formula-generated "
                    "target would measure how tightly the model reproduces the "
                    "formula, not calibrated uncertainty"
                ),
            }
        return {
            "status": "quantile_interval",
            "label": "quantile interval + conformal recalibration",
            "method": (
                "the model's own predict_interval(), evaluated under leave-cell-out, "
                "then conformal quantile recalibration (Romano et al. 2019) fitted "
                "per fold on the OTHER folds' conformity scores only"
            ),
            "nominal": nominal,
            "raw_coverage": _none_if_nan(quantile.get("rul_interval_coverage")),
            "e_star_cycles": _none_if_nan(quantile.get("global_e_star")),
            "calibrated_coverage": _none_if_nan(quantile.get("recalibrated_coverage")),
            "mean_width_cycles": _none_if_nan(quantile.get("recalibrated_width_mean")),
            "per_cell": {
                cid: {
                    "raw_coverage": _none_if_nan(fold.get("rul_interval_coverage")),
                    "mean_width_cycles": _none_if_nan(fold.get("rul_interval_width_mean")),
                }
                for cid, fold in (quantile.get("per_cell") or {}).items()
            },
            "scope": (
                "marginal coverage over this calibration population — not a "
                "per-cell conditional guarantee, and on a small fold population "
                "an estimate rather than a certificate"
            ),
        }

    return _residual_conformal_section(lco, nominal)


def _residual_conformal_section(lco: dict, nominal: float) -> dict:
    """Constant-width conformal intervals for a POINT-ONLY model.

    The model has no interval head, so one is built from evidence it already
    produced: for each fold, the quantiles of the residuals of the OTHER
    folds — each of which came from a model that never trained on them, and
    never on the fold being covered. That makes the interval
    distribution-free in the same way the platform's quantile path is, at the
    cost of a constant width (no heteroscedasticity is modelled, so a model
    whose errors grow with the horizon is under-covered at long range and
    over-covered at short range — which is exactly what the measured coverage
    below discloses rather than hides).
    """
    alpha = 1.0 - nominal
    folds: dict = {}
    skipped: dict = {}
    for cid, entry in (lco.get("per_cell") or {}).items():
        preds = entry.get("predictions") if isinstance(entry, dict) else None
        if not preds:
            continue
        obs = np.asarray(preds["rul_label_observed"], dtype=bool)
        if int(obs.sum()) < 2:
            skipped[cid] = "fewer than 2 measured-EOL rows in this fold"
            continue
        y_true = np.asarray(preds["rul_true"], dtype=float)[obs]
        y_pred = np.asarray(preds["rul_pred"], dtype=float)[obs]
        folds[cid] = {"true": y_true, "pred": y_pred, "residual": y_true - y_pred}

    if len(folds) < 2:
        return {
            "status": "not_evaluable",
            "method": "conformal_residual",
            "nominal": nominal,
            "reason": (
                "needs at least 2 folds with measured-EOL rows to calibrate one "
                "fold from the others"
            ),
            "skipped": skipped,
        }

    pooled_true, pooled_lo, pooled_hi = [], [], []
    per_cell: dict = {}
    for cid, fold in folds.items():
        others = np.concatenate(
            [f["residual"] for other, f in folds.items() if other != cid]
        )
        if len(others) < _MIN_CALIBRATION_ROWS:
            skipped[cid] = (
                f"only {len(others)} residual row(s) in the other folds — below "
                f"the {_MIN_CALIBRATION_ROWS}-row floor for a conformal quantile"
            )
            continue
        n = len(others)
        # Conformal-corrected levels: ceil(alpha (n+1)) / n, the same finite-
        # sample correction the platform's quantile path uses, so a small
        # calibration pool widens rather than under-covers.
        lo_q = min(max(math.ceil((alpha / 2) * (n + 1)) / n, 0.0), 1.0)
        hi_q = min(math.ceil((1 - alpha / 2) * (n + 1)) / n, 1.0)
        lo_offset = float(np.quantile(others, lo_q, method="lower"))
        hi_offset = float(np.quantile(others, hi_q, method="higher"))
        lo = np.clip(fold["pred"] + lo_offset, 0.0, None)
        hi = np.clip(fold["pred"] + hi_offset, 0.0, None)
        pooled_true.append(fold["true"])
        pooled_lo.append(lo)
        pooled_hi.append(hi)
        per_cell[cid] = {
            "coverage": empirical_coverage(fold["true"], lo, hi),
            "mean_width_cycles": interval_width_mean(lo, hi),
            "n_rows": int(len(fold["true"])),
            "lower_offset": lo_offset,
            "upper_offset": hi_offset,
        }

    if not pooled_true:
        return {
            "status": "not_evaluable",
            "method": "conformal_residual",
            "nominal": nominal,
            "reason": "no fold had enough calibration rows in the others",
            "skipped": skipped,
        }

    y_all = np.concatenate(pooled_true)
    lo_all = np.concatenate(pooled_lo)
    hi_all = np.concatenate(pooled_hi)
    return {
        "status": "conformal_residual",
        "label": "conformal residual interval (point-only model)",
        "method": (
            "distribution-free constant-width conformal interval built from the "
            "model's own out-of-fold residuals: each fold's interval comes from "
            "residual quantiles of the OTHER folds, one fold at a time"
        ),
        "nominal": nominal,
        "raw_coverage": None,
        "e_star_cycles": None,
        "calibrated_coverage": empirical_coverage(y_all, lo_all, hi_all),
        "mean_width_cycles": interval_width_mean(lo_all, hi_all),
        "n_rows": int(len(y_all)),
        "per_cell": per_cell,
        "skipped": skipped,
        "n_calibration_folds": len(folds),
        "scope": (
            "marginal coverage over this calibration population, calibrated from "
            f"{len(folds)} fold(s) and not a certificate; constant width, so a "
            "model whose error grows with the horizon will be under-covered at "
            "long range even when the pooled number looks nominal"
        ),
    }


def _prospective_section(
    cells: dict,
    featured: dict,
    splits: "tuple[str, ...]",
    train_fraction: float,
    seed: int,
    model: ForecasterLike,
) -> dict:
    if "prospective" not in splits:
        return {
            "status": "skipped",
            "reason": (
                "the 'prospective' split was not requested — without it the report "
                "shows new-cell interpolation but nothing about forecasting"
            ),
        }
    result = run_prospective(
        cells, featured=featured, train_fraction=train_fraction, seed=seed, forecaster=model
    )
    result["status"] = "computed"
    return result


def _gate_section(report: dict, gate: "dict | None", gate_path: "str | Path | None") -> dict:
    observed = _observed_metrics(report)
    if gate is None and gate_path is None:
        return {
            "verdict": "not_checked",
            "reason": (
                "no expectations supplied (gate= / gate_path=): every number above "
                "is measured but nothing was enforced. NOT CHECKED is not a pass — "
                "declare floors to make this section mean something."
            ),
            "observed": observed,
            "results": [],
            "untracked": sorted(observed),
        }
    expectations = gate if isinstance(gate, dict) else load_expectations(gate_path)  # type: ignore[arg-type]
    result = evaluate_gate(observed, expectations)
    result["observed"] = observed
    result["report"] = format_gate_report(result)
    return result


def _observed_metrics(report: dict) -> dict:
    """The headline numbers the gate can act on — NaN normalized to None.

    NaN matters here: a missing RUL R² arrives as float('nan'), and
    `nan < floor` is False, so an unnormalized gate would read a metric it
    never measured as passing its floor. None routes into the gate's
    explicit not-evaluable rule instead, where it fails unless the
    expectation allows it.
    """
    observed: dict = {}
    lco = report.get("lco") or {}
    for key in ("soh_r2", "soh_mae", "rul_r2", "rul_mae"):
        if key in lco:
            observed[key] = _none_if_nan(lco.get(key))

    prospective = report.get("prospective") or {}
    if prospective.get("status") == "computed":
        observed["prospective_soh_r2"] = _none_if_nan(prospective.get("soh_r2"))
        observed["prospective_rul_r2"] = _none_if_nan(prospective.get("rul_r2"))

    calibration = report.get("calibration") or {}
    if calibration.get("status") in ("quantile_interval", "conformal_residual"):
        observed["interval_coverage"] = _none_if_nan(calibration.get("calibrated_coverage"))

    return observed


def _verdict_section(report: dict) -> dict:
    claims: list = []
    withheld: list = []

    lint = report.get("leakage_lint") or {}
    if not lint.get("ok", False):
        withheld.append(
            "leakage lint FAILED — every accuracy number in this report is "
            "suspect until it passes: " + "; ".join(lint.get("violations") or [])
        )

    lco = report.get("lco") or {}
    baselines = report.get("baselines") or {}
    if lco:
        soh_r2 = _none_if_nan(lco.get("soh_r2"))
        if soh_r2 is not None:
            ci = (lco.get("confidence_intervals") or {}).get("soh_r2")
            bracket = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if _is_pair(ci) else ""
            trend = _none_if_nan(baselines.get("lco_trend_r2"))
            lift = (
                f" ({_compare(soh_r2, trend)} the {trend:.3f} trend baseline)"
                if trend is not None
                else ""
            )
            claims.append(f"SOH on unseen cells: R² = {soh_r2:.3f}{bracket}{lift}")
        else:
            withheld.append("SOH R² not evaluable on this fleet")

        coverage = float(lco.get("rul_label_coverage") or 0.0)
        rul_r2 = _none_if_nan(lco.get("rul_r2"))
        if lco.get("rul_reliable") and rul_r2 is not None:
            claims.append(
                f"RUL on unseen cells (measured-EOL rows only): R² = {rul_r2:.3f} "
                f"over {lco.get('n_rul_observed_rows')} rows"
            )
        elif coverage <= 0.0:
            withheld.append(
                "RUL: no measured end-of-life rows on this fleet — the harness "
                "refuses to score a model against closed-form extrapolated labels"
            )
        else:
            withheld.append(
                f"RUL: below the reliability floor (R² = {rul_r2!r}, observed-label "
                f"coverage {coverage:.0%}, floor {RUL_RELIABLE_FLOOR})"
            )

    calibration = report.get("calibration") or {}
    if calibration.get("status") in ("quantile_interval", "conformal_residual"):
        cov = _none_if_nan(calibration.get("calibrated_coverage"))
        nominal = calibration.get("nominal")
        if cov is not None:
            claims.append(
                f"{nominal:.0%} interval: measured coverage {cov:.1%} "
                f"({calibration.get('label') or calibration.get('method', 'conformal')})"
            )
    elif calibration.get("status") == "not_evaluable":
        withheld.append(f"interval coverage: {calibration.get('reason')}")

    prospective = report.get("prospective") or {}
    if prospective.get("status") == "computed":
        pro_soh = _none_if_nan(prospective.get("soh_r2"))
        pro_trend = _none_if_nan((report.get("baselines") or {}).get("prospective_trend_r2"))
        if pro_soh is not None:
            comparison = (
                f" — {_compare(pro_soh, pro_trend)} the per-cell trend baseline "
                f"({pro_trend:.3f})"
                if pro_trend is not None
                else ""
            )
            claims.append(
                f"same-cell forecasting (train on the first "
                f"{prospective.get('train_fraction', 0.5):.0%} of each cell): "
                f"SOH R² = {pro_soh:.3f}{comparison}"
            )
        pro_rul = _none_if_nan(prospective.get("rul_r2"))
        pro_formula = _none_if_nan(
            (report.get("baselines") or {}).get("prospective_rul_formula_r2")
        )
        if pro_rul is not None and pro_formula is not None:
            claims.append(
                f"prospective RUL R² = {pro_rul:.3f} — {_compare(pro_rul, pro_formula)} "
                f"the closed-form fade formula ({pro_formula:.3f})"
            )
    elif prospective.get("status") == "skipped":
        withheld.append("same-cell forecasting: the prospective split was not run")

    gate = report.get("gate") or {}
    status = gate.get("verdict", "not_checked")
    return {
        "status": status,
        "claims": claims,
        "withheld": withheld,
        "gate_detail": gate.get("report"),
        "summary": (
            "PASS — every declared floor/ceiling held"
            if status == "pass"
            else "FAIL — a declared metric expectation was violated"
            if status == "fail"
            else "NOT CHECKED — no floors were declared, so nothing was enforced"
        ),
    }


# ---------------------------------------------------------------------------
# Sealing: a third party can re-derive these numbers
# ---------------------------------------------------------------------------

def seal_bundle(
    report: dict,
    cell_data: dict,
    out_dir: "str | Path",
    *,
    loader: Any = None,
    loader_kwargs: "dict | None" = None,
    dataset: "str | None" = None,
) -> dict:
    """Write a sealed, verifiable bundle for a harness report.

    Three files: `benchmark.json` (fold structure + reported metrics, the
    schema batlab-lco-benchmark), `harness_report.json` (the full report), and
    `replication.json` — the root of trust, carrying the per-cell content
    digests, the environment, the model identity, and the SHA-256 of the other
    files (the seal).

    A third party then runs:

        python -m batlab.validation.replication <bundle-dir> \\
            --loader batlab.datasets.nasa:load_nasa_cells \\
            --model my_pkg.models:make_forecaster --recompute

    and gets pass/fail per check: seal, data-identity, environment, recompute.
    For a bundle published with the platform's default model, `--model` is
    omitted. `loader` may be a 'module:function' spec string or a callable,
    recorded in the bundle so the verifier knows your data path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lco = report.get("lco")
    if not lco:
        raise ValueError(
            "seal_bundle needs an LCO section to seal (validate_forecaster with "
            "splits including 'lco'); the replication bundle's recompute check "
            "re-derives the LCO number."
        )

    cells = unwrap_cell_data(cell_data)
    seed = int((report.get("config") or {}).get("seed", 42))
    cell_ids = sorted(cells)

    benchmark = export_benchmark_results(
        {k: v for k, v in lco.items() if k != "predictions"},
        out / "benchmark.json",
        cell_ids=cell_ids,
        seed=seed,
    )

    (out / "harness_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, default=str) + "\n", encoding="utf-8"
    )

    if callable(loader):
        loader_record = {"function": getattr(loader, "__name__", "<callable>"),
                         "kwargs": loader_kwargs or {}}
    elif loader:
        module_name, _, func_name = str(loader).partition(":")
        if not module_name or not func_name:
            raise ValueError("loader must be 'module.path:function' or a callable")
        loader_record = {"module": module_name, "function": func_name,
                         "kwargs": loader_kwargs or {}}
    else:
        loader_record = {}

    replication = {
        "schema": "batlab-replication-bundle",
        "schema_version": 1,
        "dataset": dataset or report.get("dataset", {}).get("name"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_version": benchmark["feature_version"],
        "cell_ids": cell_ids,
        "cell_digests": report["dataset"]["fingerprint"]["cell_digests"],
        "environment": report["dataset"]["environment"],
        "reported": {k: lco.get(k) for k in ("soh_r2", "soh_mae", "rul_r2", "rul_mae", "rul_reliable")},
        "loader": loader_record,
        "model": report.get("model", {}).get("identity", {}),
        "harness": {
            "schema": HARNESS_SCHEMA,
            "schema_version": HARNESS_SCHEMA_VERSION,
            "verdict": (report.get("verdict") or {}).get("status"),
            "label_provenance": report.get("label_provenance"),
            "calibration": {
                "status": (report.get("calibration") or {}).get("status"),
                "nominal": (report.get("calibration") or {}).get("nominal"),
                "calibrated_coverage": (report.get("calibration") or {}).get("calibrated_coverage"),
            },
            "prospective": {
                "status": (report.get("prospective") or {}).get("status"),
                "soh_r2": (report.get("prospective") or {}).get("soh_r2"),
                "rul_r2": (report.get("prospective") or {}).get("rul_r2"),
            },
        },
        "notes": (
            "Recompute with python -m batlab.validation.replication <dir> "
            "--loader <module:function> --recompute (add --model <module:function> "
            "when the bundle's recorded model is not the platform default)."
        ),
    }

    # Seal LAST: every other file is digested into the root of trust, which
    # cannot contain its own hash.
    (out / "replication.json").write_text(
        json.dumps(replication, indent=2, default=str) + "\n", encoding="utf-8"
    )
    import hashlib

    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "replication.json"
    }
    replication["files"] = files
    (out / "replication.json").write_text(
        json.dumps(replication, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return replication


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def format_report(report: dict) -> str:
    """The report as terminal text — every claim, plus what was withheld."""
    lines: list = []
    dataset = report.get("dataset") or {}
    lines.append(
        f"{HARNESS_SCHEMA}: {dataset.get('name') or '<unnamed dataset>'} "
        f"({dataset.get('n_cells')} cells, sha256 {str((dataset.get('fingerprint') or {}).get('dataset_sha256'))[:12]})"
    )
    identity = (report.get("model") or {}).get("identity") or {}
    model_label = identity.get("class") or identity.get("source") or "unrecorded"
    if identity.get("source"):
        model_label = f"{identity['source']} ({model_label})"
    elif identity.get("factory"):
        model_label = f"{identity['factory']} → {model_label}"
    interval_flag = "interval-capable" if (report.get("model") or {}).get("interval_capable") else "point-only"
    lines.append(f"model:   {model_label} [{interval_flag}]")

    lint = report.get("leakage_lint") or {}
    lines.append(f"lint:    {'PASS' if lint.get('ok') else 'FAIL'} — {len(lint.get('checked') or [])} invariant(s)")
    for violation in lint.get("violations") or []:
        lines.append(f"         {violation}")

    prov = report.get("label_provenance") or {}
    if "observed_rows" in prov:
        lines.append(
            f"labels:  {prov['observed_rows']} measured-EOL / "
            f"{prov['extrapolated_rows']} extrapolated RUL rows "
            f"({prov['observed_fraction']:.0%} measured)"
        )

    lco = report.get("lco")
    if lco:
        ci = (lco.get("confidence_intervals") or {}).get("soh_r2")
        bracket = f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]" if _is_pair(ci) else ""
        lines.append("leave-cell-out (new-cell generalization)")
        lines.append(f"  SOH  R² {_fmt(lco.get('soh_r2'))}  MAE {_fmt(lco.get('soh_mae'))}{bracket}")
        lines.append(f"  RUL  R² {_fmt(lco.get('rul_r2'))}  {'reliable' if lco.get('rul_reliable') else 'NOT reliable/not evaluable'}")

    baselines = report.get("baselines") or {}
    if baselines.get("status") == "computed":
        lines.append("baselines (same folds)")
        lines.append(f"  trend  R² {_fmt(baselines.get('lco_trend_r2'))} (LCO)  "
                     f"{_fmt(baselines.get('prospective_trend_r2'))} (prospective)")
        lines.append(f"  formula R² {_fmt(baselines.get('lco_rul_formula_r2'))} (LCO)  "
                     f"{_fmt(baselines.get('prospective_rul_formula_r2'))} (prospective)")

    calibration = report.get("calibration") or {}
    if calibration.get("status") in ("quantile_interval", "conformal_residual"):
        lines.append(f"interval calibration (nominal {_pct(calibration.get('nominal'))})")
    else:
        lines.append("interval calibration")
    if calibration.get("status") == "quantile_interval":
        lines.append(
            f"  raw {_pct(calibration.get('raw_coverage'))} → calibrated "
            f"{_pct(calibration.get('calibrated_coverage'))} "
            f"(E* = {_fmt(calibration.get('e_star_cycles'))} cycles, "
            f"mean width {_fmt(calibration.get('mean_width_cycles'))})"
        )
    elif calibration.get("status") == "conformal_residual":
        lines.append(
            f"  conformal residual interval: measured coverage "
            f"{_pct(calibration.get('calibrated_coverage'))}, "
            f"mean width {_fmt(calibration.get('mean_width_cycles'))} cycles"
        )
    else:
        lines.append(f"  {calibration.get('status')}: {calibration.get('reason', '')}")

    prospective = report.get("prospective") or {}
    if prospective.get("status") == "computed":
        lines.append("prospective (same-cell forecasting)")
        lines.append(f"  SOH  R² {_fmt(prospective.get('soh_r2'))}  MAE {_fmt(prospective.get('soh_mae'))}")
        lines.append(f"  RUL  R² {_fmt(prospective.get('rul_r2'))}  "
                     f"({'reliable' if prospective.get('rul_reliable') else 'not reliable/not evaluable'})")
    else:
        lines.append(f"prospective: {prospective.get('status')} — {prospective.get('reason', '')}")

    gate = report.get("gate") or {}
    lines.append(f"metric gate: {str(gate.get('verdict', 'not_checked')).upper()}")
    for r in gate.get("results") or []:
        lines.append(f"  [{'PASS' if r.get('verdict') == 'pass' else 'FAIL'}] {r.get('name')}: {r.get('detail')}")
    for r in gate.get("untracked") or []:
        lines.append(f"  [UNTRACKED] {r.get('name') if isinstance(r, dict) else r}")
    if gate.get("verdict") == "not_checked":
        lines.append(f"  {gate.get('reason', '')}")

    verdict = report.get("verdict") or {}
    lines.append(f"verdict: {verdict.get('summary', '')}")
    for claim in verdict.get("claims") or []:
        lines.append(f"  supported: {claim}")
    for withhold in verdict.get("withheld") or []:
        lines.append(f"  withheld:  {withhold}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _none_if_nan(value: Any) -> "float | None":
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _compare(metric: "float | None", baseline: "float | None", tolerance: float = 0.005) -> str:
    """"beats" / "matches" / "loses to" — with a tolerance for ties.

    A bare `metric > baseline` prints "beats" for a 1e-9 difference on a
    fleet where the two numbers are the same number. Since the whole point of
    the baseline is to make the model's real contribution legible, a tie has
    to read as a tie rather than as a marginal win in either direction.
    """
    if metric is None or baseline is None:
        return "not comparable to"
    delta = metric - baseline
    if abs(delta) <= tolerance:
        return "matches"
    return "beats" if delta > 0 else "LOSES to"


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(v, (int, float)) for v in value
    )


def _fmt(value: Any) -> str:
    number = _none_if_nan(value)
    return "n/a" if number is None else f"{number:.3f}"


def _pct(value: Any) -> str:
    number = _none_if_nan(value)
    return "n/a" if number is None else f"{number:.1%}"


def _json_safe(obj: Any) -> Any:
    """Recursively make a report JSON-serializable.

    numpy scalars → Python scalars, arrays → lists, NaN → None (so
    "not evaluable" survives a JSON round trip as null rather than as an
    invalid NaN literal that would make the bundle unreadable).
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main(argv: "list[str] | None" = None) -> int:
    """CLI: grade a forecaster from the command line. See `python -m batlab.harness -h`."""
    import argparse

    from batlab.harness.__main__ import run_cli

    parser = argparse.ArgumentParser(
        prog="python -m batlab.harness",
        description="Grade any battery-degradation forecaster with batlab's validation harness.",
    )
    parser.add_argument("--loader", required=True,
                        help="'module:function' returning {cell_id: DataFrame} "
                             "(e.g. batlab.datasets.nasa:load_nasa_cells)")
    parser.add_argument("--loader-kwargs", default="{}",
                        help="JSON dict of keyword args for the loader")
    parser.add_argument("--model", default=None,
                        help="'module:function' yielding the model to grade: a factory "
                             "returning a fresh model, an unfitted estimator, or a "
                             "batlab.harness adapter. Omit to grade the platform default.")
    parser.add_argument("--gate", default=None, help="expectations JSON (metric gate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--no-prospective", action="store_true",
                        help="skip the same-cell forecasting split")
    parser.add_argument("--no-intervals", action="store_true",
                        help="skip interval coverage measurement")
    parser.add_argument("--seal", default=None, help="write a sealed replication bundle here")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the full report JSON here")
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
