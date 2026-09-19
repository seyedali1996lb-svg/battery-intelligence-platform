"""
Quantile-interval calibration for leave-cell-out RUL predictions.

The Q10/Q90 quantile GBRT models in batlab.models.gbrt report an 80%
prediction interval, and train_models() already measures its empirical
coverage on a chronological holdout. This module asks the harder,
leave-cell-out version of the same question, and adds an honest,
leakage-free recalibration step:

  - run_lco_quantiles() trains the RUL point + Q10/Q90 quantile models
    under LCO folds (train on N-1 cells, predict on the held-out one) and
    reports per-fold and aggregate empirical coverage of the 80% interval
    on cells never seen in training — the same honesty upgrade LCO gives
    to the point estimate, applied to the interval.
  - recalibrate_lco_intervals() applies conformal quantile recalibration
    (Romano, Patterson & Candès, "Conformalized Quantile Regression",
    NeurIPS 2019): for each fold, the conformity score
    E_i = max(q10_i - y_i, y_i - q90_i) — how far the true value falls
    outside the raw interval — is computed on the OTHER folds only, and
    the held-out fold's interval is widened to
    [q10 - E*, q90 + E*] where E* is the (1-alpha)(n+1)/n empirical
    quantile of those scores. The recalibrator never sees the cell it is
    applied to (no leakage by construction), and the method carries a
    distribution-free marginal coverage guarantee at the nominal level —
    which the isotonic-CDF alternative does not (fitting (F_i(y_i), y_i)
    pairs trivially recovers the model's own inverse CDF, a fixed point).

Honest scope: conformal recalibration guarantees MARGINAL coverage over
the calibration population, not per-cell conditional coverage; with LCO's
small fold populations, the corrected coverage should be read as an
estimate, not a certificate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
from batlab._parallel import map_folds
from batlab.harness.forecaster import (
    ForecasterLike,
    as_factory,
    default_interval_forecaster,
    fit_forecaster,
    has_predict_interval,
)

# The Q10/Q90 pair nominally brackets an 80% interval.
NOMINAL_INTERVAL_COVERAGE = 0.80


def empirical_coverage(y_true, q10, q90) -> float:
    """Fraction of true values falling inside the [q10, q90] interval."""
    y_true = np.asarray(y_true, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true >= q10) & (y_true <= q90)))


def interval_width_mean(q10, q90) -> float:
    """Mean interval width in the same units as the predictions (cycles)."""
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    if len(q10) == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, q90 - q10)))


def _observed_label_total(featured_cache: dict) -> int:
    """How many evaluated RUL rows anywhere carry an OBSERVED (measured-EOL)
    label. Cheap and label-only: it reads `rul_label_kind`, never a model.

    A frame whose kind column is missing counts as zero (a pre-v12 frame
    cannot say which labels are measured, so the fitted path's mask is
    all-False for it too — both paths agree, see _not_evaluable_result).
    Named (and module-level) so the not-evaluable short-circuit below is
    testable: a test monkeypatches this to 1 to force the fitted path and
    compare its numbers against the short-circuit's, proving equivalence
    rather than asserting it in a comment.
    """
    from batlab.validation.lco import _LABEL_OBSERVED  # avoid circular import

    total = 0
    for _cid, (_X, _y, kinds) in featured_cache.items():
        if kinds is None:
            continue
        total += int((kinds == _LABEL_OBSERVED).sum())
    return total


def _not_evaluable_result(cell_ids: list, featured_cache: dict) -> dict:
    """The result of an all-extrapolated fleet, assembled WITHOUT fitting.

    Every interval metric in this module is computed on OBSERVED-EOL rows:
    an interval around a formula-generated target measures how tightly the
    model reproduces the extrapolation formula, not calibrated uncertainty
    around a measured quantity (see this module's own docstring). So when no
    row anywhere carries an observed label, every failing field below is
    not-evaluable BY CONSTRUCTION — nothing a fit produces is ever read.

    The scalars returned here are therefore identical to the fitted path's
    (all nan / None / False / 0.0, `pooled_conformity_scores` empty), which
    is what makes skipping the fits a pure time saving rather than a
    methodology change. The one thing not materialised is each fold's raw
    Q10/Q90 prediction array: it is the only fit-dependent content, its
    observed-row mask is all-False, and no reported number reads it. An
    empty array plus `not_evaluable` says that plainly instead of serving
    a prediction that no metric consumes.

    Why this is worth a named function: on the 46-cell Severson fleet
    (38,765 rows, every one of them formula-extrapolated) this path used to
    run 46 folds x 3 GBRT fits — measured 132 s per fold, ~13 minutes of a
    ~26-minute cold boot — to produce nothing but the not-evaluable values
    below.
    """
    per_cell = {}
    for cid in cell_ids:
        X, y_rul, _kinds = featured_cache[cid]
        per_cell[cid] = {
            "rul_true": np.asarray(y_rul.to_numpy(dtype=float), dtype=float),
            "rul_q10": np.array([], dtype=float),
            "rul_q90": np.array([], dtype=float),
            "rul_label_observed": np.zeros(len(X), dtype=bool),
            "rul_interval_coverage": None,
            "rul_interval_width_mean": None,
            "rul_mae": None,
            "rul_r2": None,
        }
    return {
        "rul_interval_coverage": float("nan"),
        "rul_interval_width_mean": float("nan"),
        "pooled_conformity_scores": np.array([]),
        "global_e_star": None,
        "recalibrated_coverage": float("nan"),
        "recalibrated_width_mean": float("nan"),
        "rul_r2": float("nan"),
        "rul_mae": float("nan"),
        "rul_reliable": False,
        "rul_label_coverage": 0.0,
        "per_cell": per_cell,
        "not_evaluable": "no observed end-of-life labels in any cell",
    }


def run_lco_quantiles(
    cell_data: dict,
    seed: int = 42,
    featured: "dict | None" = None,
    forecaster: "ForecasterLike | None" = None,
) -> dict[str, Any]:
    """
    Leave-cell-out evaluation that also trains the Q10/Q90 RUL quantile
    models, so the 80% prediction interval can be checked on cells never
    seen in training — not just on a chronological holdout of cells the
    model has already partly seen.

    Args:
        cell_data: {cell_id: DataFrame} of raw cycle-level DataFrames
                   (batlab.datasets.schema kind="cycle").
        seed: random seed for every fold's GradientBoostingRegressor.
        forecaster: the interval model to calibrate. None = the platform's
                   own point + Q10/Q90 GBRT triple (the configuration every
                   published interval here used). Whatever is passed must
                   expose predict_interval(X) -> (q10, q90) after fit();
                   batlab.harness.SklearnIntervalForecaster and
                   CallableForecaster(interval_fn=...) both do.

    Returns:
        {
          "rul_interval_coverage":   float,  # empirical coverage of [Q10, Q90], OBSERVED-EOL rows only
          "rul_interval_width_mean": float,  # mean interval width (cycles), observed rows
          "rul_r2":                  float,  # point-estimate RUL R2, observed-EOL rows only (nan when none)
          "rul_mae":                 float,
          "rul_reliable":            bool,   # observed-row rul_r2 >= RUL_RELIABLE_FLOOR AND coverage >= floor
          "rul_label_coverage":      float,  # fraction of evaluated RUL rows with observed labels
          "global_e_star":           float,  # pooled cross-cell conformal correction (cycles);
                                             # clamped at 0 (deployment only widens); None when no
                                             # observed rows — the correction the SERVE path applies
          "recalibrated_coverage":   float,  # MEASURED coverage of the widened interval at nominal 80%
          "recalibrated_width_mean": float,  # mean width after widening
          "pooled_conformity_scores": np.ndarray,  # signed E_i per row (negative = covered), unclamped
          "per_cell": {
              cell_id: {
                  "rul_true":  np.ndarray,   # observed RUL on the held-out fold
                  "rul_q10":   np.ndarray,   # predicted Q10
                  "rul_q90":   np.ndarray,   # predicted Q90
                  "rul_label_observed": np.ndarray,  # bool mask: label is measured, not formula-extrapolated
                  "rul_interval_coverage":   float,  # observed rows only (None when no observed rows)
                  "rul_interval_width_mean": float,
                  "rul_mae":  float,   # observed rows only (None when none)
                  "rul_r2":   float,
              }, ...
          },
        }

    The pooled calibration set is honest by construction: every row's
    conformity score comes from a model that never trained on its cell, so
    global_e_star is a leakage-free correction for a production model that
    likewise never sees the cells it serves. app/_data.py stamps it on the
    bundle as interval_e_star; predict() widens the served Q10/Q90 by it.

    Like run_lco(), interval metrics are computed on OBSERVED-EOL rows only:
    an interval around a formula-generated target would measure how tightly
    the model reproduces the extrapolation formula, not calibrated
    uncertainty around a measured quantity. A fleet with zero observed rows
    returns global_e_star=None and NaN coverage — not evaluable, never a
    number computed against formula-generated labels.

    Because of that rule, a fleet with zero observed rows costs no fitting
    at all: _not_evaluable_result() returns the same numbers (every one of
    them nan/None/False/0.0) without training a single model, and reports
    `not_evaluable` as the reason. The decision reads only label provenance,
    so it is forecaster-independent — any interval model evaluates to the
    same not-evaluable fields on a fleet whose labels are all extrapolated.
    """
    from batlab.validation.lco import (
        RUL_RELIABLE_FLOOR, MIN_RUL_LABEL_OBSERVED_FRACTION,
        _LABEL_OBSERVED,
    )  # avoid circular import

    featured_cache = {}
    for cell_id, df in cell_data.items():
        df_feat = (featured or {}).get(cell_id)
        if df_feat is None or isinstance(df_feat, tuple) or "rul_label_kind" not in df_feat.columns:
            df_feat = build_features(df, cell_id=cell_id)
        X, _y_soh, y_rul = get_model_matrix(df_feat)
        kinds = get_rul_label_kinds(df_feat)
        featured_cache[cell_id] = (X, y_rul, kinds)

    cell_ids = list(featured_cache.keys())
    if len(cell_ids) < 2:
        return {
            "rul_interval_coverage": float("nan"), "rul_interval_width_mean": float("nan"),
            "rul_r2": float("nan"), "rul_mae": float("nan"),
            "rul_reliable": False, "per_cell": {},
            "rul_label_coverage": 0.0,
        }

    # ── Not-evaluable short-circuit: do not fit what cannot be measured ──
    # Checked on LABEL PROVENANCE alone (no model, no fold), and after the
    # n<2 guard above so that tiny fleet keeps its own (smaller) shape.
    if _observed_label_total(featured_cache) == 0:
        return _not_evaluable_result(cell_ids, featured_cache)

    factory = as_factory(forecaster, default=default_interval_forecaster(seed))
    if factory is None:  # pragma: no cover - default_interval_forecaster() is never None
        raise ValueError(
            "run_lco_quantiles needs an interval-capable forecaster or the "
            "platform default; both were None."
        )

    def _run_fold(test_cell: str) -> dict:
        """One fold: point + Q10/Q90 models fit on the other cells, scored
        on this one. Pure, so folds run concurrently (batlab._parallel)."""
        train_cells = [c for c in cell_ids if c != test_cell]
        X_train = pd.concat([featured_cache[c][0] for c in train_cells])
        y_train = pd.concat([featured_cache[c][1] for c in train_cells])
        X_test, y_test, _kinds = featured_cache[test_cell]

        # One fresh interval model per fold, straight from the factory. The
        # default factory is the point + Q10/Q90 GBRT triple sharing one
        # scaler — the configuration every published interval number here
        # was measured under.
        model = fit_forecaster(factory(), X_train, y_train)
        if not has_predict_interval(model):
            raise ValueError(
                "run_lco_quantiles() needs an interval-capable model: "
                "predict_interval(X) -> (q10, q90) after fit(). For a "
                "point-only model, batlab.harness.validate_forecaster(...) "
                "builds a distribution-free interval from out-of-fold "
                "residuals instead, under the same folds."
            )

        rul_true = y_test.to_numpy(dtype=float)
        q10, q90 = model.predict_interval(X_test)
        q10 = np.asarray(q10, dtype=float)
        q90 = np.asarray(q90, dtype=float)

        kinds = featured_cache[test_cell][2]
        obs_mask = (
            (kinds.reindex(X_test.index) == _LABEL_OBSERVED).to_numpy()
            if kinds is not None else np.zeros(len(X_test), dtype=bool)
        )
        n_obs = int(obs_mask.sum())

        fold_true, fold_q10, fold_q90 = (
            rul_true[obs_mask], q10[obs_mask], q90[obs_mask]
        ) if n_obs >= 2 else ([], [], [])

        fold_mae = fold_r2 = None
        if n_obs >= 2:
            rul_pred_obs = np.asarray(model.predict(X_test), dtype=float)[obs_mask]
            fold_mae = mean_absolute_error(fold_true, rul_pred_obs)
            fold_r2 = r2_score(fold_true, rul_pred_obs)

        return dict(
            rul_true=rul_true, q10=q10, q90=q90, obs_mask=obs_mask, n_obs=n_obs,
            fold_true=np.asarray(fold_true, dtype=float),
            fold_q10=np.asarray(fold_q10, dtype=float),
            fold_q90=np.asarray(fold_q90, dtype=float),
            fold_mae=fold_mae, fold_r2=fold_r2,
        )

    # Independent folds, run concurrently, re-assembled in cell order so
    # the pooled arrays and per-fold lists match the serial loop exactly.
    fold_results = map_folds(_run_fold, cell_ids)

    all_true, all_q10, all_q90, all_mae, all_r2 = [], [], [], [], []
    per_cell = {}

    for test_cell, fr in zip(cell_ids, fold_results):
        n_obs = fr["n_obs"]
        fold_true, fold_q10, fold_q90 = fr["fold_true"], fr["fold_q10"], fr["fold_q90"]
        all_true.append(fold_true)
        all_q10.append(fold_q10)
        all_q90.append(fold_q90)
        if n_obs >= 2:
            all_mae.append(fr["fold_mae"])
            if not np.isnan(fr["fold_r2"]):
                all_r2.append(fr["fold_r2"])

        per_cell[test_cell] = {
            "rul_true": fr["rul_true"],
            "rul_q10": fr["q10"],
            "rul_q90": fr["q90"],
            "rul_label_observed": fr["obs_mask"],
            "rul_interval_coverage": (
                empirical_coverage(fold_true, fold_q10, fold_q90) if n_obs >= 2 else None
            ),
            "rul_interval_width_mean": (
                interval_width_mean(fold_q10, fold_q90) if n_obs >= 2 else None
            ),
            "rul_mae": all_mae[-1] if n_obs >= 2 else None,
            "rul_r2": (all_r2[-1] if (n_obs >= 2 and all_r2 and not np.isnan(all_r2[-1])) else None),
        }

    y_all = np.concatenate([np.asarray(t, dtype=float) for t in all_true]) if all_true else np.array([])
    q10_all = np.concatenate([np.asarray(t, dtype=float) for t in all_q10]) if all_q10 else np.array([])
    q90_all = np.concatenate([np.asarray(t, dtype=float) for t in all_q90]) if all_q90 else np.array([])
    mean_rul_r2 = float(np.mean(all_r2)) if all_r2 else float("nan")

    n_obs_total = int(sum(len(t) for t in all_true))
    # Count evaluated rows from the per-fold masks (NOT `~mask.sum()`, which
    # bitwise-NOTs the sum — a real bug this comment now documents).
    n_rows_total = int(sum(len(per_cell[c]["rul_label_observed"]) for c in per_cell))
    n_ext_total = n_rows_total - n_obs_total
    coverage = (n_obs_total / n_rows_total) if n_rows_total > 0 else 0.0

    # ── Global conformal correction for the PRODUCTION model ────────────────
    # Every fold's E_i = max(q10 - y, y - q90) is computed on rows whose cell
    # was NOT in that fold's training set, so the pooled scores are an honest
    # calibration set for a correction applied to a model that similarly
    # never saw the cells it will serve. This is the number the bundle takes
    # so the SERVED 80% interval is the one whose coverage was actually
    # measured — not the raw quantile regressor's nominal claim.
    alpha = 1.0 - NOMINAL_INTERVAL_COVERAGE
    pooled_scores = np.concatenate([
        np.maximum(np.asarray(q10, dtype=float) - np.asarray(y, dtype=float),
                   np.asarray(y, dtype=float) - np.asarray(q90, dtype=float))
        for y, q10, q90 in zip(all_true, all_q10, all_q90)
        if len(y) > 0
    ]) if any(len(t) > 0 for t in all_true) else np.array([])
    if pooled_scores.size >= 2:
        level = min(np.ceil((1.0 - alpha) * (pooled_scores.size + 1)) / pooled_scores.size, 1.0)
        # Clamp at 0: the deployment only ever WIDENS (predict() applies the
        # correction only when > 0), so the measured coverage reported here
        # must be the coverage of exactly that clamped correction. When the
        # raw interval already over-covers (signed quantile < 0) the served
        # interval is the raw one and its measured coverage is reported as-is.
        global_e_star = max(0.0, float(np.quantile(pooled_scores, level, method="higher")))
        recal_cov = empirical_coverage(
            y_all,
            np.clip(q10_all - global_e_star, 0, None),
            q90_all + global_e_star,
        ) if len(y_all) >= 2 else float("nan")
        recal_wid = interval_width_mean(
            np.clip(q10_all - global_e_star, 0, None), q90_all + global_e_star,
        ) if len(q10_all) else float("nan")
    else:
        global_e_star = None
        recal_cov = float("nan")
        recal_wid = float("nan")

    return {
        "rul_interval_coverage": (
            empirical_coverage(y_all, q10_all, q90_all)
            if len(y_all) >= 2 else float("nan")
        ),
        "rul_interval_width_mean": (
            interval_width_mean(q10_all, q90_all) if len(q10_all) else float("nan")
        ),
        "pooled_conformity_scores": pooled_scores,
        "global_e_star": global_e_star,
        "recalibrated_coverage": recal_cov,
        "recalibrated_width_mean": recal_wid,
        "rul_r2": mean_rul_r2,
        "rul_mae": float(np.mean(all_mae)) if all_mae else float("nan"),
        "rul_reliable": bool(
            all_r2
            and mean_rul_r2 >= RUL_RELIABLE_FLOOR
            and coverage >= MIN_RUL_LABEL_OBSERVED_FRACTION
        ),
        "rul_label_coverage": float(coverage),
        "per_cell": per_cell,
    }


def recalibrate_lco_intervals(quantile_result: dict) -> dict[str, Any]:
    """
    Conformal quantile recalibration of each fold's Q10/Q90 interval.

    For a held-out fold, the conformity scores
    E_i = max(q10_i - y_i, y_i - q90_i) are computed from the OTHER folds
    only (each other-fold point against its own predicted interval), and
    the held-out fold's interval becomes [q10 - E*, q90 + E*], where E* is
    the (1-alpha)(n+1)/n empirical quantile of those scores (alpha = 1 -
    nominal coverage, i.e. 0.2 for the default 80% interval). The held-out
    cell never contributes to the E* that is applied to it, so the
    reported corrected coverage is honest LCO coverage — and, marginally
    over the calibration population, guaranteed at the nominal level.

    Consumes the result dict from run_lco_quantiles() (a fabricated dict
    with the same shape works too — useful for testing the recalibrator in
    isolation). Returns:

        {
          "nominal": 0.80,
          "raw": {"rul_interval_coverage": float, "rul_interval_width_mean": float,
                  "per_cell": {...coverage/width per cell...}},
          "recalibrated": {"rul_interval_coverage": float, "rul_interval_width_mean": float,
                           "per_cell": {...}},
          "skipped": {cell_id: bool},   # True when a fold had no calibration data
        }
    """
    per_cell = quantile_result.get("per_cell", {})
    cell_ids = list(per_cell.keys())
    alpha = 1.0 - NOMINAL_INTERVAL_COVERAGE

    raw_covs, raw_wids = {}, {}
    new_covs, new_wids, skipped, e_stars = {}, {}, {}, {}
    all_raw_y, all_raw_q10, all_raw_q90 = [], [], []
    all_new_q10, all_new_q90 = [], []

    for cell in cell_ids:
        fold = per_cell[cell]
        others = [c for c in cell_ids if c != cell]

        scores = np.concatenate([
            np.maximum(per_cell[oc]["rul_q10"] - per_cell[oc]["rul_true"],
                       per_cell[oc]["rul_true"] - per_cell[oc]["rul_q90"])
            for oc in others
        ]) if others else np.array([])

        if len(scores) == 0:
            skipped[cell] = True
            e_stars[cell] = 0.0
        else:
            skipped[cell] = False
            level = min(np.ceil((1.0 - alpha) * (len(scores) + 1)) / len(scores), 1.0)
            e_stars[cell] = float(np.quantile(scores, level, method="higher"))

        q10_new = np.clip(fold["rul_q10"] - e_stars[cell], 0, None)
        q90_new = fold["rul_q90"] + e_stars[cell]

        raw_covs[cell] = empirical_coverage(fold["rul_true"], fold["rul_q10"], fold["rul_q90"])
        raw_wids[cell] = interval_width_mean(fold["rul_q10"], fold["rul_q90"])
        new_covs[cell] = empirical_coverage(fold["rul_true"], q10_new, q90_new)
        new_wids[cell] = interval_width_mean(q10_new, q90_new)

        all_raw_y.append(fold["rul_true"])
        all_raw_q10.append(fold["rul_q10"])
        all_raw_q90.append(fold["rul_q90"])
        all_new_q10.append(q10_new)
        all_new_q90.append(q90_new)

    y_all = np.concatenate(all_raw_y)
    return {
        "nominal": NOMINAL_INTERVAL_COVERAGE,
        "raw": {
            "rul_interval_coverage": empirical_coverage(y_all, np.concatenate(all_raw_q10), np.concatenate(all_raw_q90)),
            "rul_interval_width_mean": interval_width_mean(np.concatenate(all_raw_q10), np.concatenate(all_raw_q90)),
            "per_cell": raw_covs,
        },
        "recalibrated": {
            "rul_interval_coverage": empirical_coverage(y_all, np.concatenate(all_new_q10), np.concatenate(all_new_q90)),
            "rul_interval_width_mean": interval_width_mean(np.concatenate(all_new_q10), np.concatenate(all_new_q90)),
            "per_cell": new_covs,
        },
        "skipped": skipped,
    }
