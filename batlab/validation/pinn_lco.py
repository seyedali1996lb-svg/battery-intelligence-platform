"""
Leave-cell-out benchmark for the PINN physics estimator.

Why this module exists
----------------------
The GBRT has a published, honest LCO number (see batlab.validation.lco). The
PINN (`src/pinn_model.py`) is described as a physics-regularized estimator but
had no comparable number, so "which model should we use?" was answered by
assertion rather than measurement.

`run_pinn_lco()` runs the PINN through the *same* harness shape as the GBRT:

  - identical leave-one-cell-out folds (same population, same held-out cells);
  - identical targets (`soh_pct` and `rul` from get_model_matrix());
  - identical metrics (mean soh_r2/soh_mae/rul_r2/rul_mae across folds);
  - the same trivial baseline denominator available via
    batlab.validation.trivial_baseline.baseline_lco_r2().

The only thing that differs between the two reported numbers is the model.

Honest scope / conventions (stated, not hidden)
-----------------------------------------------
  - The PINN's degradation law is shared across cells
    (soh(n) = s0 - beta_sei*sqrt(n) - beta_lam*n**gamma). It is fitted on the
    TRAINING cells only. Each cell (train and test) is anchored at its own
    first observed SOH, which is the level information a deployment genuinely
    has when a cell enters service — and the GBRT likewise consumes the test
    cell's measured per-cycle features. This is documented on the result as
    `initial_condition_convention` so the comparison can't be over-read.
  - RUL is derived by projecting the fitted curve to the EOL threshold, which
    can be far outside the observed window. When the fused curve does not
    reach EOL within the projection horizon the fold is marked
    `eol_reached=False` and the RUL error is an extrapolation artefact, not a
    measured miss. That flag is returned per cell rather than swallowed.
  - `lambda_physics` is exposed so the physics-loss weight is part of the
    experiment record, exactly like the GBRT's hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error, r2_score

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
from batlab.validation.lco import RUL_RELIABLE_FLOOR, MIN_RUL_LABEL_OBSERVED_FRACTION, _safe_r2

MODEL_KIND = "pinn"

# Fitted on training cells only; see module docstring.
PARAM_BOUNDS = [(1e-7, 0.05), (1e-9, 0.005), (0.7, 1.8)]
PARAM_INITIAL = [0.003, 0.00005, 1.0]

# Concrete (non-formal) default: the PINN's own default in src/pinn_model.py.
LAMBDA_PHYSICS_DEFAULT = 0.25


def pinn_hyperparams(lambda_physics: float = LAMBDA_PHYSICS_DEFAULT) -> dict:
    """The recorded 'hyperparameters' of a PINN fit, mirroring GBRT_PARAMS so
    both model kinds carry a comparable, replayable parameter record."""
    return {
        "model_kind": MODEL_KIND,
        "lambda_physics": float(lambda_physics),
        "param_initial": list(PARAM_INITIAL),
        "param_bounds": [list(b) for b in PARAM_BOUNDS],
        "degradation_law": "s0 - beta_sei*sqrt(n) - beta_lam*n**gamma",
        "anchor": "first observed SOH per cell",
    }


def _soh_curve(cycles: np.ndarray, s0_frac: float, p: np.ndarray) -> np.ndarray:
    b_sei, b_lam, gamma = p
    return s0_frac - b_sei * np.sqrt(np.maximum(cycles, 0.0)) - b_lam * (np.maximum(cycles, 0.0) ** gamma)


def _fit_shared_params(train: list, lambda_physics: float) -> np.ndarray:
    """Fit one shared (beta_sei, beta_lam, gamma) to every training cell,
    each anchored at its own first observed SOH."""

    def _loss(p: np.ndarray) -> float:
        total = 0.0
        n_pts = 0
        mono_penalty = 0.0
        for cycles, soh_frac, s0 in train:
            pred = _soh_curve(cycles, s0, p)
            total += float(np.sum((pred - soh_frac) ** 2))
            n_pts += len(cycles)
            # Physics penalty: monotonicity of the degradation curve.
            d = np.diff(pred)
            mono_penalty += float(np.sum(np.maximum(0.0, d) ** 2))
        if n_pts == 0:
            return 1e9
        mse = total / n_pts
        reg = 0.01 * (p[0] ** 2 + (p[1] * 100) ** 2)
        return float(mse + lambda_physics * mono_penalty + reg)

    try:
        res = opt.minimize(_loss, PARAM_INITIAL, bounds=PARAM_BOUNDS, method="L-BFGS-B")
        if res.success or res.fun < 1.0:
            return np.asarray(res.x, dtype=np.float64)
    except Exception:
        pass
    return np.asarray(PARAM_INITIAL, dtype=np.float64)


def _predicted_rul_curve(
    cycles: np.ndarray,
    s0_frac: float,
    p: np.ndarray,
    eol_soh_pct: float,
) -> tuple:
    """Project the fitted curve forward and return (rul_pred, eol_reached).

    Uses one dense grid rather than a per-cycle search, so this is O(N) in the
    horizon instead of O(N * horizon)."""
    max_cycle = int(np.max(cycles)) if len(cycles) else 0
    horizon = max(max_cycle * 4 + 20000, 5000)
    grid = np.arange(0, horizon + 1, dtype=np.float64)
    soh_grid = _soh_curve(grid, s0_frac, p) * 100.0
    # Curve is (penalized toward) monotonically decreasing; searchsorted on the
    # ascending negation finds the first crossing of the EOL threshold.
    idx = np.searchsorted(-soh_grid, -eol_soh_pct, side="left")
    if idx >= len(grid):
        eol_reached = False
        eol_cycle = float(horizon)
    else:
        eol_reached = True
        eol_cycle = float(grid[idx])
    rul_pred = np.clip(eol_cycle - cycles, 0.0, None)
    return rul_pred, eol_reached


def run_pinn_lco(
    cell_data: dict,
    seed: int = 42,
    eol_soh_pct: float = 80.0,
    lambda_physics: float = LAMBDA_PHYSICS_DEFAULT,
    featured: "dict | None" = None,
) -> dict:
    """Leave-cell-out benchmark of the PINN physics estimator.

    Args:
        cell_data: {cell_id: raw cycle DataFrame}, same contract as
                   batlab.validation.lco.run_lco().
        seed:      accepted for signature parity with run_lco(); the PINN fit
                   is deterministic (L-BFGS-B from a fixed initial point), so
                   this does not currently change the result.
        featured:  optional {cell_id: df_feat} already built by the caller, so
                   the (expensive) feature build is not repeated.

    Returns the same keys as run_lco(), plus `model_kind`, `hyperparams`, and
    `pinn_extrapolation_flags` (per-cell `eol_reached`).
    """
    del seed  # deterministic fit; kept for a uniform harness signature

    featured_frames = {}
    for cell_id, df in (cell_data or {}).items():
        if featured is not None and cell_id in featured:
            frame = featured[cell_id]
            # Callers may hand the model-matrix tuple (X, y_soh, y_rul); the
            # physics fit needs the label columns, so rebuild in that case.
            if isinstance(frame, tuple):
                df_feat = build_features(df, cell_id=cell_id)
            else:
                df_feat = frame
        else:
            df_feat = build_features(df, cell_id=cell_id)
        _X, y_soh, y_rul = get_model_matrix(df_feat)
        kinds = get_rul_label_kinds(df_feat)
        if "cycle_number" not in df_feat.columns:
            continue
        cycles = df_feat.loc[y_soh.index, "cycle_number"].to_numpy(dtype=np.float64)
        featured_frames[cell_id] = (
            cycles,
            y_soh.to_numpy(dtype=np.float64) / 100.0,
            y_soh.to_numpy(dtype=np.float64),
            y_rul.to_numpy(dtype=np.float64),
            kinds,
        )

    cell_ids = list(featured_frames.keys())
    empty = {
        "soh_r2": float("nan"), "soh_mae": float("nan"),
        "rul_r2": float("nan"), "rul_mae": float("nan"),
        "rul_reliable": False, "per_cell": {},
        "model_kind": MODEL_KIND,
        "hyperparams": pinn_hyperparams(lambda_physics),
        "pinn_extrapolation_flags": {},
        "initial_condition_convention": "first observed SOH per cell",
    }
    if len(cell_ids) < 2:
        return empty

    soh_maes, soh_r2s, rul_maes, rul_r2s = [], [], [], []
    per_cell, flags = {}, {}
    n_obs_rows = 0
    n_ext_rows = 0

    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]

        train = [
            (featured_frames[c][0], featured_frames[c][1], float(featured_frames[c][2][0]) / 100.0)
            for c in train_cells
        ]
        params = _fit_shared_params(train, lambda_physics)

        cycles, _soh_frac, y_soh, y_rul, kinds = featured_frames[test_cell]
        s0 = float(y_soh[0]) / 100.0
        soh_pred = _soh_curve(cycles, s0, params) * 100.0
        rul_pred, eol_reached = _predicted_rul_curve(cycles, s0, params, eol_soh_pct)

        # Observed-EOL rows only for the headline RUL metrics — same rule as
        # run_lco(); the PINN competes on the identical population.
        kinds = kinds.reindex(y_soh.index if hasattr(y_soh, "index") else range(len(y_soh))) if kinds is not None else None
        obs_mask = (
            (kinds == "observed").to_numpy() if kinds is not None else np.zeros(len(cycles), dtype=bool)
        )
        n_obs = int(obs_mask.sum())
        n_obs_rows += n_obs
        n_ext_rows += int(len(cycles) - n_obs)

        if len(y_soh) >= 2:
            soh_r2 = float(r2_score(y_soh, soh_pred))
        else:
            soh_r2 = float("nan")
        fold_rul_r2 = _safe_r2(np.asarray(y_rul)[obs_mask], np.asarray(rul_pred)[obs_mask]) if n_obs >= 2 else None
        rul_r2 = fold_rul_r2 if fold_rul_r2 is not None else float("nan")
        soh_mae = float(mean_absolute_error(y_soh, soh_pred))
        rul_mae = (
            float(mean_absolute_error(np.asarray(y_rul)[obs_mask], np.asarray(rul_pred)[obs_mask]))
            if n_obs >= 2 else float("nan")
        )

        soh_maes.append(soh_mae); soh_r2s.append(soh_r2)
        if fold_rul_r2 is not None:
            rul_maes.append(rul_mae)
            rul_r2s.append(fold_rul_r2)
        per_cell[test_cell] = dict(
            soh_mae=soh_mae, soh_r2=soh_r2, rul_mae=rul_mae, rul_r2=rul_r2,
            eol_reached=eol_reached,
            beta_sei=float(params[0]), beta_lam=float(params[1]), gamma_lam=float(params[2]),
        )
        flags[test_cell] = {"eol_reached": eol_reached}

    mean_rul_r2 = float(np.mean(rul_r2s)) if rul_r2s else float("nan")
    n_rows_total = n_obs_rows + n_ext_rows
    coverage = (n_obs_rows / n_rows_total) if n_rows_total > 0 else 0.0
    return {
        "soh_r2":       float(np.nanmean(soh_r2s)) if len(soh_r2s) else float("nan"),
        "soh_mae":      float(np.nanmean(soh_maes)) if len(soh_maes) else float("nan"),
        "rul_r2":       mean_rul_r2,
        "rul_mae":      float(np.mean(rul_maes)) if rul_maes else float("nan"),
        "rul_reliable": bool(
            rul_r2s
            and mean_rul_r2 >= RUL_RELIABLE_FLOOR
            and coverage >= MIN_RUL_LABEL_OBSERVED_FRACTION
        ),
        "rul_label_coverage": float(coverage),
        "per_cell":     per_cell,
        "model_kind":   MODEL_KIND,
        "hyperparams":  pinn_hyperparams(lambda_physics),
        "pinn_extrapolation_flags": flags,
        "initial_condition_convention": "first observed SOH per cell",
    }
