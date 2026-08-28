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

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from batlab.features.engineering import build_features, get_model_matrix
from batlab.models.gbrt import GBRT_PARAMS, GBRT_QUANTILE_PARAMS

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


def run_lco_quantiles(cell_data: dict, seed: int = 42) -> dict:
    """
    Leave-cell-out evaluation that also trains the Q10/Q90 RUL quantile
    models, so the 80% prediction interval can be checked on cells never
    seen in training — not just on a chronological holdout of cells the
    model has already partly seen.

    Args:
        cell_data: {cell_id: DataFrame} of raw cycle-level DataFrames
                   (batlab.datasets.schema kind="cycle").
        seed: random seed for every fold's GradientBoostingRegressor.

    Returns:
        {
          "rul_interval_coverage":   float,  # empirical coverage of [Q10, Q90] across ALL folds
          "rul_interval_width_mean": float,  # mean interval width (cycles)
          "rul_r2":                  float,  # point-estimate RUL R2 across all folds
          "rul_mae":                 float,
          "rul_reliable":            bool,   # rul_r2 >= RUL_RELIABLE_FLOOR
          "per_cell": {
              cell_id: {
                  "rul_true":  np.ndarray,   # observed RUL on the held-out fold
                  "rul_q10":   np.ndarray,   # predicted Q10
                  "rul_q90":   np.ndarray,   # predicted Q90
                  "rul_interval_coverage":   float,
                  "rul_interval_width_mean": float,
                  "rul_mae":  float,
                  "rul_r2":   float,
              }, ...
          },
        }
    """
    from batlab.validation.lco import RUL_RELIABLE_FLOOR  # avoid circular import

    featured = {}
    for cell_id, df in cell_data.items():
        df_feat = build_features(df, cell_id=cell_id)
        X, _y_soh, y_rul = get_model_matrix(df_feat)
        featured[cell_id] = (X, y_rul)

    cell_ids = list(featured.keys())
    if len(cell_ids) < 2:
        return {
            "rul_interval_coverage": float("nan"), "rul_interval_width_mean": float("nan"),
            "rul_r2": float("nan"), "rul_mae": float("nan"),
            "rul_reliable": False, "per_cell": {},
        }

    point_params = {**GBRT_PARAMS, "random_state": seed}
    quantile_params = {**GBRT_QUANTILE_PARAMS, "random_state": seed}

    all_true, all_q10, all_q90, all_mae, all_r2 = [], [], [], [], []
    per_cell = {}

    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]
        X_train = pd.concat([featured[c][0] for c in train_cells])
        y_train = pd.concat([featured[c][1] for c in train_cells])
        X_test, y_test = featured[test_cell]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        rul_m = GradientBoostingRegressor(**point_params).fit(Xtr, y_train)
        q10_m = GradientBoostingRegressor(loss="quantile", alpha=0.10, **quantile_params).fit(Xtr, y_train)
        q90_m = GradientBoostingRegressor(loss="quantile", alpha=0.90, **quantile_params).fit(Xtr, y_train)

        rul_true = y_test.to_numpy(dtype=float)
        q10 = np.clip(q10_m.predict(Xte), 0, None)
        q90 = np.clip(q90_m.predict(Xte), 0, None)

        all_true.append(rul_true)
        all_q10.append(q10)
        all_q90.append(q90)
        all_mae.append(mean_absolute_error(rul_true, rul_m.predict(Xte)))
        all_r2.append(r2_score(rul_true, rul_m.predict(Xte)))

        per_cell[test_cell] = {
            "rul_true": rul_true,
            "rul_q10": q10,
            "rul_q90": q90,
            "rul_interval_coverage": empirical_coverage(rul_true, q10, q90),
            "rul_interval_width_mean": interval_width_mean(q10, q90),
            "rul_mae": all_mae[-1],
            "rul_r2": all_r2[-1],
        }

    y_all = np.concatenate(all_true)
    q10_all = np.concatenate(all_q10)
    q90_all = np.concatenate(all_q90)
    mean_rul_r2 = float(np.mean(all_r2))
    return {
        "rul_interval_coverage": empirical_coverage(y_all, q10_all, q90_all),
        "rul_interval_width_mean": interval_width_mean(q10_all, q90_all),
        "rul_r2": mean_rul_r2,
        "rul_mae": float(np.mean(all_mae)),
        "rul_reliable": mean_rul_r2 >= RUL_RELIABLE_FLOOR,
        "per_cell": per_cell,
    }


def recalibrate_lco_intervals(quantile_result: dict) -> dict:
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
