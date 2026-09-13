"""
Hierarchical / partial-pooling fade model — a new cell borrows strength
from its chemistry and its own early cycles.

Why this exists
---------------
The GBRT is the strongest interpolator (leave-cell-out SOH R² 0.96-0.999)
but structurally cannot forecast (Tier-2 prospective split: it collapses
below a straight line when denied the future, because a regression tree
cannot predict outside its training label range). The PINN can forecast
but fits ONE shared fade law for the whole fleet. What a real deployment
actually has for a NEW cell is neither extreme:

  - the cell's own EARLY cycling history (level + local fade behaviour);
  - a fleet of same-chemistry cells whose fade rates inform how bad this
    cell's fade could plausibly get.

`run_hierarchical_lco()` implements the classic empirical-Bayes
partial-pooling answer:

  1. Each training cell contributes a log fade-rate observation
     theta_j = log(slope of its per-cycle capacity loss), together with
     the within-cell sampling variance of that estimate.
  2. The fleet prior N(mu, tau^2) is estimated from the training cells
     (mu = mean log theta, tau^2 = var(log theta) minus the mean sampling
     variance — the classic method-of-moments shrinkage target, floored
     at a small positive value so a suspiciously consistent fleet cannot
     produce a zero-variance prior).
  3. The held-out cell contributes ONLY its early window (first
     `min_history_fraction` of its recorded cycles — what a deployment
     genuinely has when a cell enters service). Its local log fade rate
     is shrunk toward the fleet prior with precision weights:

         theta_hat = (w_loc * theta_loc + w_pri * mu) / (w_loc + w_pri)

     A short or noisy early window pulls the estimate toward the fleet
     mean (borrows strength from the chemistry); a long, informative
     window stays with the cell's own data.
  4. SOH is predicted as a straight line from the cell's own early-window
     anchor with the shrunk slope; RUL projects that line to the EOL
     threshold.

Honest scope (stated, not hidden)
---------------------------------
  - The fade law is LINEAR in cycle number. Real cells have a sqrt(n)
    SEI phase and a knee; this model deliberately does not pretend to
    capture curvature. Its value is an honest, calibrated FORECASTING
    baseline with pooled uncertainty — the bar every forecasting model
    must beat (the per-cell trend baseline of the prospective split is
    its no-pooling cousin; this adds the fleet prior).
  - The held-out cell's early window is used for anchoring, exactly like
    the PINN's first-SOH anchor and the GBRT's consumption of the test
    cell's per-cycle features. It is recorded as
    `initial_condition_convention`.
  - RUL metrics follow the v12 honesty rules: observed-EOL rows only,
    with label coverage disclosed.
  - When multiple reference fleets share a chemistry (NASA + CALCE, both
    LiCoO2), the prior is estimated per chemistry across the POOLED
    fleets — that is item 2's chemistry-conditional backbone. With one
    fleet per chemistry the prior is fleet-local and the two items
    coincide; `prior_scope` records which happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
from batlab.validation.lco import unwrap_cell_data
from batlab.validation.lco import RUL_RELIABLE_FLOOR, MIN_RUL_LABEL_OBSERVED_FRACTION, _safe_r2

MODEL_KIND = "hierarchical"

# The fraction of the held-out cell's recorded cycles the model may consume
# as its "early window" — the deployment-realistic information set.
MIN_HISTORY_FRACTION = 0.3
MIN_HISTORY_FLOOR_CYCLES = 10

# Method-of-moments floor on the prior variance: a fleet of near-identical
# cells must not produce a zero-variance prior (that would make the
# shrinkage divide by zero and over-trust the fleet mean).
TAU2_FLOOR = 1e-4


def hierarchical_hyperparams() -> dict:
    """The recorded hyperparameters, mirroring GBRT_PARAMS / pinn_hyperparams."""
    return {
        "model_kind": MODEL_KIND,
        "fade_law": "linear per-cycle capacity loss, log-rate Gaussian prior",
        "min_history_fraction": MIN_HISTORY_FRACTION,
        "min_history_floor_cycles": MIN_HISTORY_FLOOR_CYCLES,
        "tau2_floor": TAU2_FLOOR,
        "estimator": "empirical-Bayes method of moments + precision-weighted shrinkage",
    }


def _cell_local_stats(cycles: np.ndarray, cap_ah: np.ndarray, early_only: bool) -> "tuple[float, float, float, float] | None":
    """(log local fade rate, its sampling variance, anchor cycle,
    anchor capacity) from a cell's data. With `early_only` only the first
    MIN_HISTORY_FRACTION of the recorded cycles are consumed (the
    deployment information set). The anchor is the window's FIRST row —
    the (cycle, capacity) point the predicted line passes through."""
    if len(cycles) < 4:
        return None
    if early_only:
        n_hist = max(MIN_HISTORY_FLOOR_CYCLES, int(len(cycles) * MIN_HISTORY_FRACTION))
        n_hist = min(n_hist, len(cycles))
        cycles, cap_ah = cycles[:n_hist], cap_ah[:n_hist]
    # OLS slope of capacity on cycle number, plus its sampling variance
    # sigma^2 / Sxx where sigma^2 is the residual variance.
    x = cycles.astype(float)
    y = cap_ah.astype(float)
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0:
        return None
    slope = float(((x - xbar) * (y - y.mean())).sum() / sxx)
    # Capacity DECREASES with cycles, so the OLS slope is negative; the
    # model works on the per-cycle capacity LOSS (positive convention).
    loss = -slope
    if loss <= 1e-9:
        # A non-fading (or charging-up) window carries no degradation info.
        return None
    resid = y - (y.mean() + slope * (x - xbar))
    dof = max(1, len(x) - 2)
    sigma2 = float((resid ** 2).sum()) / dof
    var_slope = sigma2 / sxx
    # var_slope == 0 (noiseless data, e.g. a synthetic generator) means
    # PERFECT local information — floor it rather than reject the cell.
    # Variance is sign-invariant, so this is the loss-scale variance.
    var_slope = max(var_slope, 1e-12)
    return float(np.log(loss)), float(var_slope / (loss ** 2)), float(x[0]), float(y[0])


def _prior_from_fleet(fleet_stats: list) -> "tuple[float, float]":
    """Empirical-Bayes prior N(mu, tau^2) over LOG fade rates from the
    training cells' (log theta, sampling variance) pairs."""
    logs = np.array([s[0] for s in fleet_stats], dtype=float)
    vs = np.array([s[1] for s in fleet_stats], dtype=float)
    mu = float(np.mean(logs))
    tau2 = float(np.var(logs, ddof=1) - np.mean(vs)) if len(logs) >= 2 else 0.0
    return mu, max(tau2, TAU2_FLOOR)


def _shrunk_log_rate(local: "tuple[float, float, float]", prior: "tuple[float, float]") -> float:
    """Precision-weighted shrinkage of the cell's local log fade rate
    toward the fleet prior."""
    th_loc, v_loc = local[0], local[1]
    mu, tau2 = prior
    w_loc = 1.0 / max(v_loc, 1e-12)
    w_pri = 1.0 / max(tau2, 1e-12)
    return (w_loc * th_loc + w_pri * mu) / (w_loc + w_pri)


def run_hierarchical_lco(
    cell_data: dict,
    seed: int = 42,
    eol_soh_pct: float = 80.0,
    featured: "dict | None" = None,
    chemistry_by_cell: "dict | None" = None,
) -> dict:
    """Leave-cell-out benchmark of the hierarchical partial-pooling model.

    Same harness contract as run_lco()/run_pinn_lco(): identical folds,
    identical targets, identical observed-EOL RUL honesty rules, identical
    metric definitions — the only difference between its numbers and the
    GBRT's/PINN's is the model.

    Args:
        cell_data: {cell_id: raw cycle DataFrame}.
        seed:      accepted for signature parity; the estimator is
                   deterministic (closed-form OLS + moment matching).
        featured:  optional pre-built feature frames {cell_id: df_feat}.
        chemistry_by_cell: optional {cell_id: chemistry} enabling the
                   chemistry-conditional prior (item 2): the prior is then
                   estimated within chemistry groups (pooling every fleet
                   supplied), and `prior_scope` reports "per-chemistry".

    Returns the same keys as run_lco(), plus `model_kind`, `hyperparams`,
    `prior_scope`, and per-cell shrinkage diagnostics (`per_cell_theta`).
    """
    del seed  # deterministic estimator; kept for a uniform harness signature
    cell_data = unwrap_cell_data(cell_data)

    featured_in: dict = {}
    for cell_id, df in (cell_data or {}).items():
        if featured is not None and cell_id in featured:
            frame = featured[cell_id]
            if isinstance(frame, tuple) or "rul_label_kind" not in frame.columns:
                frame = build_features(df, cell_id=cell_id)
        else:
            frame = build_features(df, cell_id=cell_id)
        X, y_soh, y_rul = get_model_matrix(frame)
        kinds = get_rul_label_kinds(frame)
        if "cycle_number" not in frame.columns or "capacity_ah" not in frame.columns:
            continue
        sub = frame.loc[y_soh.index]
        featured_in[cell_id] = (
            sub["cycle_number"].to_numpy(dtype=np.float64),
            sub["capacity_ah"].to_numpy(dtype=float),
            y_soh.to_numpy(dtype=float),
            y_rul.to_numpy(dtype=float),
            kinds.reindex(y_soh.index) if kinds is not None else None,
        )

    cell_ids = list(featured_in.keys())
    empty = {
        "soh_r2": float("nan"), "soh_mae": float("nan"),
        "rul_r2": float("nan"), "rul_mae": float("nan"),
        "rul_reliable": False, "per_cell": {},
        "rul_label_coverage": 0.0,
        "n_rul_observed_rows": 0, "n_rul_extrapolated_rows": 0,
        "model_kind": MODEL_KIND,
        "hyperparams": hierarchical_hyperparams(),
        "prior_scope": "none",
        "per_cell_theta": {},
        "initial_condition_convention": (
            f"first {MIN_HISTORY_FRACTION:.0%} of the cell's recorded cycles "
            "(level + local fade) anchored at its own early capacity"
        ),
    }
    if len(cell_ids) < 2:
        return empty

    chem = chemistry_by_cell or {}

    soh_maes, soh_r2s = [], []
    per_cell, per_theta = {}, {}
    n_obs_rows = 0
    n_ext_rows = 0
    obs_r2s, obs_maes = [], []

    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]
        group = chem.get(test_cell)
        # Chemistry-conditional prior when the caller supplies chemistries:
        # pool every TRAINING cell of the same chemistry across all fleets
        # in cell_data; fall back to all training cells when chemistries
        # are unknown (fleet-local prior, the item-1 special case).
        pool = [c for c in train_cells if (chem.get(c) == group)] if (group and chem) else train_cells
        prior_scope = "per-chemistry" if (group and chem) else "fleet-local"
        if len(pool) < 1:
            pool = train_cells

        fleet_stats = [
            s for s in (
                _cell_local_stats(featured_in[c][0], featured_in[c][1], early_only=False)
                for c in pool
            ) if s is not None
        ]
        prior = _prior_from_fleet(fleet_stats) if fleet_stats else (float("nan"), TAU2_FLOOR)

        cycles, cap_ah, y_soh, y_rul, kinds = featured_in[test_cell]
        local = _cell_local_stats(cycles, cap_ah, early_only=True)
        if local is None or not np.isfinite(prior[0]):
            # No usable local window and/or prior: fold is not evaluable,
            # surfaced as None metrics rather than a fabricated number.
            per_cell[test_cell] = dict(
                soh_mae=None, soh_r2=None, rul_mae=None, rul_r2=None,
                rul_label_kinds={"evaluable": 0},
            )
            continue

        th_hat = _shrunk_log_rate(local, prior)
        slope_hat = float(np.exp(th_hat))  # Ah lost per cycle
        anchor_cycle, anchor_cap = local[2], local[3]  # the cell's own early (cycle, capacity) point
        # Predict the straight line through the anchor with the shrunk
        # slope. Cycles before the anchor are the consumed window itself —
        # the line fits that window by construction, honest to score.
        # SOH converts projected capacity through the cell's initial
        # capacity: pred_cap / cap_init * 100, never pred_cap * 100.
        cap_init = anchor_cap
        pred_cap = anchor_cap - slope_hat * (cycles - anchor_cycle)
        soh_pred_full = np.clip(pred_cap / cap_init * 100.0, 0.0, None)

        soh_mae = float(mean_absolute_error(y_soh, soh_pred_full))
        soh_r2 = _safe_r2(y_soh, soh_pred_full)
        soh_maes.append(soh_mae)
        if soh_r2 is not None:
            soh_r2s.append(soh_r2)

        # RUL: cycles from each row until the linear projection crosses EOL.
        # The EOL threshold resolves against the cell's fresh capacity —
        # the anchor capacity (its first observed measurement).
        eol_cap = float(anchor_cap) * (eol_soh_pct / 100.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rul_pred = np.where(
                slope_hat > 0,
                (cap_ah - eol_cap) / slope_hat,
                np.nan,
            )
        rul_pred = np.clip(rul_pred, 0.0, None)

        obs_mask = (kinds == "observed").to_numpy() if kinds is not None else np.zeros(len(cycles), dtype=bool)
        n_obs = int(obs_mask.sum())
        n_ext = int((len(cycles)) - n_obs) if kinds is not None else 0
        n_obs_rows += n_obs
        n_ext_rows += n_ext
        rul_true = y_rul
        fold_obs_r2 = fold_obs_mae = None
        if n_obs >= 2:
            fold_obs_r2 = _safe_r2(rul_true[obs_mask], rul_pred[obs_mask])
            fold_obs_mae = float(mean_absolute_error(rul_true[obs_mask], rul_pred[obs_mask]))
            if fold_obs_r2 is not None:
                obs_r2s.append(fold_obs_r2)
                obs_maes.append(fold_obs_mae)

        per_cell[test_cell] = dict(
            soh_mae=soh_mae, soh_r2=soh_r2,
            rul_mae=fold_obs_mae, rul_r2=fold_obs_r2,
            rul_label_kinds={"observed": n_obs, "extrapolated": n_ext},
        )
        per_theta[test_cell] = {
            "local_log_fade": local[0],
            "shrunk_log_fade": th_hat,
            "prior_mu": prior[0],
            "prior_tau2": prior[1],
            "shrinkage_weight_prior": 1.0 / max(prior[1], 1e-12)
            / (1.0 / max(local[1], 1e-12) + 1.0 / max(prior[1], 1e-12)),
        }

    n_rul_rows = n_obs_rows + n_ext_rows
    coverage = (n_obs_rows / n_rul_rows) if n_rul_rows > 0 else 0.0
    mean_obs_r2 = float(np.mean(obs_r2s)) if obs_r2s else float("nan")

    return {
        "soh_r2": float(np.mean(soh_r2s)) if soh_r2s else float("nan"),
        "soh_mae": float(np.mean(soh_maes)) if soh_maes else float("nan"),
        "rul_r2": mean_obs_r2,
        "rul_mae": float(np.mean(obs_maes)) if obs_maes else float("nan"),
        "rul_reliable": bool(
            obs_r2s
            and mean_obs_r2 >= RUL_RELIABLE_FLOOR
            and coverage >= MIN_RUL_LABEL_OBSERVED_FRACTION
        ),
        "rul_label_coverage": float(coverage),
        "n_rul_observed_rows": n_obs_rows,
        "n_rul_extrapolated_rows": n_ext_rows,
        "per_cell": per_cell,
        "per_cell_theta": per_theta,
        "model_kind": MODEL_KIND,
        "hyperparams": hierarchical_hyperparams(),
        "prior_scope": prior_scope if cell_ids else "none",
        "initial_condition_convention": empty["initial_condition_convention"],
    }
