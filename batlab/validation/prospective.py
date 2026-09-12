"""Prospective evaluation — the only test that separates forecasting from
curve-fitting.

Why this exists
---------------
Leave-cell-out answers "does the model generalize to a cell it has never
seen?" using data whose FUTURE the model has already seen: the held-out
cell's full aging curve, including its last cycles, is in the evaluation
pool. For deployed use that is not the deployment question. A real deployment
asks: this cell has produced 200 cycles so far — what happens NEXT, beyond
everything we have observed? No leave-cell-out or random row-split ever tests
that, because the evaluation rows always come from the same recorded window
as the training rows.

The prospective split makes the time axis the boundary: train ONLY on each
cell's first `train_fraction` of cycles, evaluate ONLY on the remainder. The
model is never shown a cycle from the future it is scored on. This is
strictly harder than LCO — the model must extrapolate in cycle number, not
just interpolate between cells — and a model that only interpolates aging
curves will look dramatically worse here. That drop is the measurement:
the gap between an LCO R² and a prospective R² IS the amount of
curve-fitting that was silently riding along in the evaluation setup.

Conventions (stated because they change the number):
- The split is PER CELL (each cell's own cycle range, by relative fraction),
  not a global cycle cutoff — cells in one fleet start and age at different
  rates, and a global cutoff would silently give fast-aging cells an empty
  test window.
- Training and evaluation pools each contain ALL cells; the same cell
  contributes its early cycles to training and its late cycles to testing.
  This deliberately mirrors deployment (you have history for the very cell
  you're predicting) and is the opposite axis from LCO, which holds out
  whole CELLS. The two numbers are complements, not competitors:
  LCO = new-cell generalization, prospective = future-horizon forecasting.
- Features are computed per row from PAST-ONLY windows (rolling means/fade
  rates use data up to and including the current cycle — the leakage lint
  enforces the target side; the feature windows were already causal by
  construction).
- RUL follows the same v12 honesty rules as run_lco(): scored on
  observed-EOL rows only, the extrapolated pool reported separately, and
  the formula baseline (the closed form that generates extrapolated labels)
  runs under the identical split so "can the model beat the formula?" is
  answered on the same population.
- Bootstrap CIs reuse batlab.validation.bootstrap unchanged. The resampling
  unit here is the CELL's test window (one unit per cell), the closest
  analogue to LCO's fold.

A trivial SOH baseline (per-cell linear fit on the TRAIN window only,
extrapolated forward) runs under the identical split — the honest floor for
"how much of the prospective number is just extending each cell's own early
trend?", which is exactly the null hypothesis this evaluation exists to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from batlab.features.engineering import (
    FEATURE_COLUMNS,
    build_features,
    get_model_matrix,
    get_rul_label_kinds,
)
from batlab.models.gbrt import GBRT_PARAMS
from batlab.validation.bootstrap import lco_confidence_intervals
from batlab.validation.lco import RUL_RELIABLE_FLOOR

DEFAULT_TRAIN_FRACTION = 0.5

_LABEL_OBSERVED = "observed"
_LABEL_EXTRAPOLATED = "extrapolated"

# (X, y_soh, y_rul, obs_mask) per window — obs_mask marks observed-EOL rows.
_Matrix = tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]


def _safe_r2(y_true, y_pred) -> "float | None":
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or float(np.var(y_true)) <= 0.0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return None
    return 1.0 - ss_res / ss_tot


def _build_featured(cell_data: dict, featured: "dict | None") -> dict:
    """Featured frames per cell, honouring a precomputed cache and rebuilding
    pre-v12 frames (no rul_label_kind column) from raw data."""
    out: dict = {}
    for cell_id, df in cell_data.items():
        frame = (featured or {}).get(cell_id)
        if frame is None or isinstance(frame, tuple) or "rul_label_kind" not in frame.columns:
            frame = build_features(df, cell_id=cell_id)
        out[cell_id] = frame
    return out


def split_prospective(
    df_feat: pd.DataFrame,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> "tuple[pd.DataFrame, pd.DataFrame] | None":
    """Split one cell's FEATURED frame by cycle position: the earliest
    `train_fraction` of its cycles are the train window, the rest the test
    window. Ordered by the canonical cycle_number axis. Returns
    (train_df, test_df), or None when the cell has too few rows to produce a
    meaningful split (<10 rows total, or a window of fewer than 3 rows) —
    such a cell carries no forecastable future and is excluded rather than
    padded."""
    if "cycle_number" not in df_feat.columns or len(df_feat) < 10:
        return None
    df_sorted = df_feat.sort_values("cycle_number", kind="stable")
    n_train = int(len(df_sorted) * train_fraction)
    n_test = len(df_sorted) - n_train
    if n_test < 3 or n_train < 3:
        return None
    return df_sorted.iloc[:n_train], df_sorted.iloc[n_train:]


def _window_matrix(df_window: pd.DataFrame) -> _Matrix:
    """get_model_matrix + observed-EOL mask for one window, aligned to the
    same rows the matrix keeps."""
    X, y_soh, y_rul = get_model_matrix(df_window)
    kinds = get_rul_label_kinds(df_window)
    kinds = kinds.reindex(X.index) if kinds is not None else None
    obs_mask = (
        (kinds == _LABEL_OBSERVED).to_numpy()
        if kinds is not None else np.zeros(len(X), dtype=bool)
    )
    return X, y_soh, y_rul, obs_mask


def _pool(matrices: "list[_Matrix]") -> "tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]":
    """Concatenate window matrices; returns X, y_soh, y_rul, obs_mask, and a
    row→cell-position index (into the caller's list) for per-cell metrics."""
    X = pd.concat([m[0] for m in matrices])
    y_soh = pd.concat([m[1] for m in matrices])
    y_rul = pd.concat([m[2] for m in matrices])
    obs = np.concatenate([m[3] for m in matrices]) if matrices else np.zeros(0, dtype=bool)
    cell_of = np.concatenate([
        np.full(len(m[0]), i, dtype=int) for i, m in enumerate(matrices)
    ]) if matrices else np.zeros(0, dtype=int)
    return X, y_soh, y_rul, obs, cell_of


def run_prospective(
    cell_data: dict,
    featured: "dict | None" = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = 42,
) -> dict:
    """Prospective evaluation over a fleet.

    Parameters
    ----------
    cell_data : {cell_id: raw_cycles_DataFrame} (used when `featured` omits
        a cell — features are built from the raw frame).
    featured : optional {cell_id: already-built featured DataFrame}. Callers
        holding the features cache pass it to avoid the expensive
        PyBaMM-backed rebuild; any cell missing from it is built here.
    train_fraction : fraction of each cell's cycles given to the TRAIN
        window (default 0.5 — the model sees the first half of life and
        must forecast the second half).
    seed : GBRT random_state; also the bootstrap seed via the CI module.

    Returns
    -------
    {
      "train_fraction", "n_cells_evaluated", "n_cells_skipped",
      "soh_r2", "soh_mae",
      "rul_r2", "rul_mae",            # observed-EOL rows only (nan when none)
      "rul_reliable",
      "rul_label_coverage", "n_rul_observed_rows", "n_rul_extrapolated_rows",
      "rul_extrapolated_r2",
      "confidence_intervals",
      "per_cell",                      # per-cell test-window metrics
      "n_train_rows", "n_test_rows",
      "train_fraction_note",
    }
    """
    featured_in = _build_featured(cell_data, featured)

    train_mats: list[_Matrix] = []
    test_mats: list[_Matrix] = []
    cell_ids_in_order: list[str] = []
    skipped: list[str] = []
    raw_by_cell: dict = {}
    for cell_id, frame in featured_in.items():
        split = split_prospective(frame, train_fraction)
        if split is None:
            skipped.append(cell_id)
            continue
        tr, te = split
        train_mats.append(_window_matrix(tr))
        test_mats.append(_window_matrix(te))
        cell_ids_in_order.append(cell_id)
        raw_by_cell[cell_id] = frame

    empty = {
        "train_fraction": train_fraction, "n_cells_evaluated": 0,
        "n_cells_skipped": len(skipped),
        "soh_r2": float("nan"), "soh_mae": float("nan"),
        "rul_r2": float("nan"), "rul_mae": float("nan"),
        "rul_reliable": False,
        "rul_label_coverage": 0.0,
        "n_rul_observed_rows": 0, "n_rul_extrapolated_rows": 0,
        "rul_extrapolated_r2": None,
        "confidence_intervals": None,
        "per_cell": {}, "n_train_rows": 0, "n_test_rows": 0,
        "train_fraction_note": "",
    }
    if len(train_mats) < 2:
        return empty

    X_train, y_soh_train, y_rul_train, _obs_tr, _ = _pool(train_mats)
    X_test, y_soh_test, y_rul_test, obs_mask, cell_of = _pool(test_mats)

    common = [c for c in FEATURE_COLUMNS if c in X_train.columns and c in X_test.columns]
    if not common:
        return empty
    Xtr, Xte = X_train[common], X_test[common]

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    params = {**GBRT_PARAMS, "random_state": seed}
    soh_m = GradientBoostingRegressor(**params).fit(Xtr_sc, y_soh_train)
    rul_m = GradientBoostingRegressor(**params).fit(Xtr_sc, y_rul_train)

    y_soh = np.asarray(y_soh_test, dtype=float)
    y_rul = np.asarray(y_rul_test, dtype=float)
    soh_pred = np.asarray(soh_m.predict(Xte_sc), dtype=float)
    rul_pred = np.asarray(rul_m.predict(Xte_sc), dtype=float)

    soh_r2 = _safe_r2(y_soh, soh_pred)
    soh_mae = float(mean_absolute_error(y_soh, soh_pred))

    n_obs = int(obs_mask.sum())
    n_ext = int(len(obs_mask) - n_obs)
    rul_r2 = _safe_r2(y_rul[obs_mask], rul_pred[obs_mask]) if n_obs >= 2 else None
    rul_mae = (
        float(mean_absolute_error(y_rul[obs_mask], rul_pred[obs_mask]))
        if n_obs >= 2 else float("nan")
    )
    ext_r2 = _safe_r2(y_rul[~obs_mask], rul_pred[~obs_mask]) if n_ext >= 2 else None

    # Per-cell test-window metrics (the CI's resampling units).
    per_cell: dict = {}
    cell_soh_r2s, cell_soh_maes, cell_obs_r2s, cell_obs_maes = [], [], [], []
    for pos, cid in enumerate(cell_ids_in_order):
        rows = np.where(cell_of == pos)[0]
        if rows.size < 3:
            continue
        c_soh_r2 = _safe_r2(y_soh[rows], soh_pred[rows])
        c_soh_mae = float(mean_absolute_error(y_soh[rows], soh_pred[rows]))
        entry = {
            "n_test_rows": int(rows.size),
            "soh_r2": c_soh_r2,
            "soh_mae": c_soh_mae,
            "rul_r2": None, "rul_mae": None,
            "rul_label_kinds": {
                _LABEL_OBSERVED: int(obs_mask[rows].sum()),
                _LABEL_EXTRAPOLATED: int((~obs_mask[rows]).sum()),
            },
        }
        if c_soh_r2 is not None:
            cell_soh_r2s.append(c_soh_r2)
            cell_soh_maes.append(c_soh_mae)
        c_obs = obs_mask[rows]
        if c_obs.sum() >= 2:
            c_r2 = _safe_r2(y_rul[rows][c_obs], rul_pred[rows][c_obs])
            c_mae = float(mean_absolute_error(y_rul[rows][c_obs], rul_pred[rows][c_obs]))
            entry["rul_r2"], entry["rul_mae"] = c_r2, c_mae
            if c_r2 is not None:
                cell_obs_r2s.append(c_r2)
                cell_obs_maes.append(c_mae)
        per_cell[cid] = entry

    # Fold-level CIs over per-cell test-window metrics (cell = resampling unit,
    # mirroring run_lco's fold-level convention).
    ci = lco_confidence_intervals(
        soh_r2s=cell_soh_r2s,
        soh_maes=cell_soh_maes,
        obs_r2s=cell_obs_r2s, obs_maes=cell_obs_maes,
        seed=seed,
    )

    coverage = (n_obs / (n_obs + n_ext)) if (n_obs + n_ext) > 0 else 0.0

    return {
        "train_fraction": train_fraction,
        "features": common,
        "n_cells_evaluated": len(cell_ids_in_order),
        "n_cells_skipped": len(skipped),
        "soh_r2": float(soh_r2) if soh_r2 is not None else float("nan"),
        "soh_mae": soh_mae,
        "rul_r2": float(rul_r2) if rul_r2 is not None else float("nan"),
        "rul_mae": rul_mae,
        "rul_reliable": bool(
            rul_r2 is not None
            and rul_r2 >= RUL_RELIABLE_FLOOR
            and coverage >= 0.5
        ),
        "rul_label_coverage": float(coverage),
        "n_rul_observed_rows": n_obs,
        "n_rul_extrapolated_rows": n_ext,
        "rul_extrapolated_r2": ext_r2,
        "confidence_intervals": ci,
        "per_cell": per_cell,
        "n_train_rows": int(len(Xtr)),
        "n_test_rows": int(len(Xte)),
        "train_fraction_note": (
            f"Train window = first {train_fraction:.0%} of each cell's cycles; "
            "evaluation only on the remainder. The model never sees a cycle "
            "from the window it is scored on — the LCO-vs-prospective gap "
            "measures how much of the LCO number was interpolation, not "
            "forecasting."
        ),
    }


def trivial_soh_baseline_prospective(
    cell_data: dict,
    featured: "dict | None" = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict:
    """Per-cell linear trend, fit on the TRAIN window only, extrapolated
    across the TEST window — under the identical prospective split.

    This is the null hypothesis the GBRT must beat here: "you don't need a
    model, just extend each cell's own early fade line forward". A model
    that cannot beat this baseline has learned nothing beyond linear
    extrapolation of the training window.
    """
    from sklearn.linear_model import LinearRegression

    per_cell_r2s: list[float] = []
    per_cell: dict = {}
    for cell_id, frame in _build_featured(cell_data, featured).items():
        split = split_prospective(frame, train_fraction)
        if split is None:
            continue
        tr, te = split
        cy_tr = tr["cycle_number"].to_numpy(dtype=float).reshape(-1, 1)
        soh_tr = tr["soh_pct"].to_numpy(dtype=float)
        cy_te = te["cycle_number"].to_numpy(dtype=float).reshape(-1, 1)
        soh_te = te["soh_pct"].to_numpy(dtype=float)
        if len(cy_tr) < 3 or float(np.var(soh_te)) <= 0.0:
            continue
        lin = LinearRegression().fit(cy_tr, soh_tr)
        soh_hat = lin.predict(cy_te)
        r2 = _safe_r2(soh_te, soh_hat)
        if r2 is None:
            continue
        per_cell_r2s.append(r2)
        per_cell[cell_id] = {
            "r2": r2,
            "mae": float(mean_absolute_error(soh_te, soh_hat)),
            "n_test_rows": int(len(cy_te)),
        }

    return {
        "baseline_soh_r2": float(np.mean(per_cell_r2s)) if per_cell_r2s else None,
        "per_cell": per_cell,
        "n_cells": len(per_cell_r2s),
    }


def rul_formula_baseline_prospective(
    cell_data: dict,
    featured: "dict | None" = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict:
    """The RUL closed form (remaining headroom ÷ fade rate) under the
    identical prospective split, scored on observed-EOL rows of each cell's
    TEST window only. Answers: does the GBRT beat the formula that generates
    extrapolated labels when BOTH are denied access to the future?"""
    per_cell: dict = {}
    obs_r2s: list[float] = []
    for cell_id, frame in _build_featured(cell_data, featured).items():
        split = split_prospective(frame, train_fraction)
        if split is None:
            continue
        _tr, te = split
        X_te, _ys, y_rul, obs_mask = _window_matrix(te)
        if int(obs_mask.sum()) < 2:
            continue
        initial_cap = float(frame["capacity_ah"].iloc[0])
        eol_capacity = initial_cap * 0.80
        fade = te.loc[X_te.index, "fade_rate_50cy"].clip(lower=1e-6).to_numpy(dtype=float)
        rul_pred = (
            (te.loc[X_te.index, "capacity_ah"].to_numpy(dtype=float) - eol_capacity) / fade
        )
        rul_pred = np.clip(rul_pred, 0.0, None)
        y_true = np.asarray(y_rul, dtype=float)
        r2 = _safe_r2(y_true[obs_mask], rul_pred[obs_mask])
        if r2 is None:
            continue
        obs_r2s.append(r2)
        per_cell[cell_id] = {
            "r2": r2,
            "mae": float(mean_absolute_error(y_true[obs_mask], rul_pred[obs_mask])),
            "n_obs_rows": int(obs_mask.sum()),
        }

    return {
        "rul_formula_baseline_r2": float(np.mean(obs_r2s)) if obs_r2s else None,
        "per_cell": per_cell,
        "n_cells": len(obs_r2s),
    }
