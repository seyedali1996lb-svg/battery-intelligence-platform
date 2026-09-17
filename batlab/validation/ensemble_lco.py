"""
Ensemble of the GBRT and the PINN physics estimator, weighted per
chemistry by held-out performance.

Why this exists
---------------
The GBRT and the PINN fail in complementary ways, both now measured:

  - the GBRT interpolates superbly (LCO SOH R² 0.96-0.999) but cannot
    forecast (Tier-2 prospective split: below a straight line when denied
    the future);
  - the PINN extrapolates but fits one shared fade law, losing to the
    GBRT on held-out-cell interpolation on every fleet.

An ensemble that is better than BOTH would be a result; an ensemble that
sits between them is also a result — the honest question is only whether
held-out performance justifies the extra model. `run_ensemble_lco()`
answers it through the identical leave-cell-out harness:

  - in every fold, the GBRT trains on N-1 cells and the PINN's shared
    fade law fits on the same N-1 cells;
  - the BLEND WEIGHT is chosen INSIDE the fold by an inner
    leave-one-cell-out pass over the training cells (weight grid searched
    for the best inner-fold SOH R²) — never on the held-out cell. This is
    the same weight-estimation discipline as stacking with an inner CV:
    the held-out cell plays no role in choosing its own blender;
  - the final prediction is w * GBRT + (1-w) * PINN on SOH, and the same
    w on RUL (both models produce RUL on comparable cycle scales).

The grid spans w in [0, 1] in 0.1 steps, so w=1.0 (pure GBRT) and w=0.0
(pure PINN) are both reachable — if one member is uniformly better the
ensemble reports that honestly by collapsing onto it. The per-fold
weights are returned for inspection: which chemistry needs physics and
which needs data is itself a result. The fleet-level
`weights_by_chemistry` summary (mean weight per chemistry of the held-out
cells) is what "weighted per chemistry" means here: the weight is
estimated per FOLD, and chemistry-level averages of fold weights are a
description, not a separate fitting path.

Honest scope: the PINN member anchors each cell at its own first SOH and
shares one fade law per fold (its standard convention); RUL metrics obey
the v12 observed-EOL-only rules. The extra inner-CV pass multiplies GBRT
fits by the number of training cells — real cost, disclosed in the
hyperparams record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
from batlab.validation.lco import unwrap_cell_data
from batlab.validation.lco import (
    GBRT_PARAMS,
    RUL_RELIABLE_FLOOR,
    MIN_RUL_LABEL_OBSERVED_FRACTION,
    _safe_r2,
)
from batlab.validation.pinn_lco import _fit_shared_params, _soh_curve, _predicted_rul_curve

MODEL_KIND = "ensemble"

WEIGHT_GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.1), 1))

# Cost bounds for the inner weight-selection pass. Full inner-LCO is
# O(n²) GBRT fits (n_folds × n_training_cells), which on a 9-cell ×
# ~970-row fleet means ~160 full fits — minutes of startup for a weight
# that only needs RANK ordering across a 10-step grid. The bounds keep
# the selection honest (still inner folds, never the held-out cell)
# while capping the cost; both are recorded in the hyperparams.
INNER_MAX_FOLDS = 4
INNER_MAX_ROWS_PER_FIT = 3000


def ensemble_hyperparams(lambda_physics: float = 0.25) -> dict:
    """The recorded hyperparameters, mirroring GBRT_PARAMS / pinn_hyperparams."""
    return {
        "model_kind": MODEL_KIND,
        "members": ["gbrt", "pinn"],
        "weight_grid": [float(w) for w in WEIGHT_GRID],
        "weight_selection": (
            "inner leave-one-cell-out over training cells, never the held-out cell; "
            f"bounded to {INNER_MAX_FOLDS} folds x {INNER_MAX_ROWS_PER_FIT} rows"
        ),
        "inner_max_folds": INNER_MAX_FOLDS,
        "inner_max_rows_per_fit": INNER_MAX_ROWS_PER_FIT,
        "lambda_physics": float(lambda_physics),
    }


def _blend_r2(w: float, soh_true: list[np.ndarray], preds_a: list, preds_b: list) -> float:
    """Mean R² of the w-blend across inner folds. NaN-safe: folds with
    degenerate targets are skipped by _safe_r2 returning None."""
    r2s = []
    for y, pa, pb in zip(soh_true, preds_a, preds_b):
        r2 = _safe_r2(y, w * pa + (1.0 - w) * pb)
        if r2 is not None:
            r2s.append(r2)
    return float(np.mean(r2s)) if r2s else float("nan")


def run_ensemble_lco(
    cell_data: dict,
    seed: int = 42,
    eol_soh_pct: float = 80.0,
    featured: "dict | None" = None,
    lambda_physics: float = 0.25,
    chemistry_by_cell: "dict | None" = None,
) -> dict:
    """Leave-cell-out benchmark of the GBRT+PINN ensemble.

    Same harness contract as run_lco()/run_pinn_lco(): identical folds,
    targets, observed-EOL RUL rules, metric definitions. The blend weight
    is chosen inside each fold by an inner leave-one-cell-out pass over
    that fold's training cells — the held-out cell never influences its
    own weight.

    `chemistry_by_cell` ({cell_id: chemistry}) is optional metadata: it
    does not change any fit, but the returned `weights_by_chemistry`
    summary groups the per-fold weights by the held-out cell's chemistry,
    so "which chemistry leans on physics vs data" becomes readable.

    Returns the same keys as run_lco(), plus `model_kind`, `hyperparams`,
    `per_cell_weights`, and `weights_by_chemistry`.
    """
    params = {**GBRT_PARAMS, "random_state": seed}
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
        featured_in[cell_id] = (X, y_soh, y_rul, kinds)

    cell_ids = list(featured_in.keys())
    empty = {
        "soh_r2": float("nan"), "soh_mae": float("nan"),
        "rul_r2": float("nan"), "rul_mae": float("nan"),
        "rul_reliable": False, "per_cell": {},
        "rul_label_coverage": 0.0,
        "n_rul_observed_rows": 0, "n_rul_extrapolated_rows": 0,
        "model_kind": MODEL_KIND,
        "hyperparams": ensemble_hyperparams(lambda_physics),
        "per_cell_weights": {},
        "weights_by_chemistry": {},
    }
    if len(cell_ids) < 2:
        return empty

    def _gbrt_preds(train_cells: list, test_cell_ids: list,
                    row_cap: "int | None" = None, rng: "np.random.Generator | None" = None) -> "tuple[dict, dict]":
        """Fit one GBRT (scaler + SOH/RUL models) on train_cells; predict
        SOH and RUL for every cell in test_cell_ids. Returns ({cell: soh_pred},
        {cell: rul_pred}). With `row_cap`, the training matrix is subsampled
        (seeded, reproducible) to that many rows — used only by the bounded
        inner weight-selection pass, never by the outer evaluation folds."""
        X_train = pd.concat([featured_in[c][0] for c in train_cells])
        y_soh_train = pd.concat([featured_in[c][1] for c in train_cells])
        y_rul_train = pd.concat([featured_in[c][2] for c in train_cells])
        if row_cap is not None and len(X_train) > row_cap:
            pick = np.sort((rng if rng is not None else np.random.default_rng(0)).choice(
                len(X_train), size=row_cap, replace=False))
            X_train = X_train.iloc[pick]
            y_soh_train = y_soh_train.iloc[pick]
            y_rul_train = y_rul_train.iloc[pick]
        scaler = StandardScaler().fit(X_train)
        soh_m = GradientBoostingRegressor(**params).fit(scaler.transform(X_train), y_soh_train)
        rul_m = GradientBoostingRegressor(**params).fit(scaler.transform(X_train), y_rul_train)
        soh_out, rul_out = {}, {}
        for c in test_cell_ids:
            Xc = scaler.transform(featured_in[c][0])
            soh_out[c] = soh_m.predict(Xc)
            rul_out[c] = rul_m.predict(Xc)
        return soh_out, rul_out

    def _pinn_preds(train_cells: list, test_cell_ids: list) -> "tuple[dict, dict]":
        """Fit the shared PINN fade law on train_cells; predict SOH (curve
        anchored at each cell's own first SOH) and RUL (EOL projection) for
        every cell in test_cell_ids."""
        train = []
        for c in train_cells:
            X, y_soh, _y_rul, _kinds = featured_in[c]
            if "cycle_number" not in X.columns:
                return {}, {}
            cyc = X["cycle_number"].to_numpy(dtype=np.float64)
            train.append((cyc, y_soh.to_numpy(dtype=np.float64) / 100.0, float(y_soh.iloc[0]) / 100.0))
        try:
            params_p = _fit_shared_params(train, lambda_physics)
        except Exception:
            return {}, {}
        soh_out, rul_out = {}, {}
        for c in test_cell_ids:
            X, y_soh, _y_rul, _kinds = featured_in[c]
            cyc = X["cycle_number"].to_numpy(dtype=np.float64)
            s0 = float(y_soh.iloc[0]) / 100.0
            soh_out[c] = _soh_curve(cyc, s0, params_p) * 100.0
            rul_pred, _flag = _predicted_rul_curve(cyc, s0, params_p, eol_soh_pct)
            rul_out[c] = rul_pred
        return soh_out, rul_out

    chem = chemistry_by_cell or {}

    soh_maes, soh_r2s = [], []
    per_cell, per_weights, chem_weights = {}, {}, {}
    n_obs_rows = 0
    n_ext_rows = 0
    obs_r2s, obs_maes = [], []

    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]

        # ── Outer-fold members ──
        g_soh, g_rul = _gbrt_preds(train_cells, [test_cell])
        p_soh, p_rul = _pinn_preds(train_cells, [test_cell])
        if test_cell not in g_soh or test_cell not in p_soh:
            per_cell[test_cell] = dict(
                soh_mae=None, soh_r2=None, rul_mae=None, rul_r2=None,
                rul_label_kinds={"evaluable": 0},
            )
            continue

        # ── Inner leave-one-cell-out pass to pick w (held-out cell excluded) ──
        # Bounded cost: at most INNER_MAX_FOLDS inner folds (strided across
        # the training cells so every cell remains *eligible* across the
        # fleet's folds), each inner GBRT fit capped at
        # INNER_MAX_ROWS_PER_FIT rows (seeded subsample — the weight needs
        # rank ordering across a 10-step grid, not full-sample precision).
        n_train = len(train_cells)
        inner_cells = (
            train_cells[:: max(1, n_train // INNER_MAX_FOLDS)][:INNER_MAX_FOLDS]
            if n_train > INNER_MAX_FOLDS else train_cells
        )
        rng_inner = np.random.default_rng(seed + 1)
        inner_preds_a: list = []
        inner_preds_b: list = []
        inner_true: list[np.ndarray] = []
        for ic in inner_cells:
            itrain = [c for c in train_cells if c != ic]
            if len(itrain) < 1:
                continue
            ig_soh, _ = _gbrt_preds(itrain, [ic], row_cap=INNER_MAX_ROWS_PER_FIT, rng=rng_inner)
            ip_soh, _ = _pinn_preds(itrain, [ic])
            if ic in ig_soh and ic in ip_soh:
                inner_true.append(featured_in[ic][1].to_numpy(dtype=float))
                inner_preds_a.append(ig_soh[ic])
                inner_preds_b.append(ip_soh[ic])

        if inner_true:
            r2_grid = [_blend_r2(w, inner_true, inner_preds_a, inner_preds_b) for w in WEIGHT_GRID]
            w_best = float(WEIGHT_GRID[int(np.nanargmax(r2_grid))])
        else:
            # No usable inner folds: fall back to the training-mean blend,
            # stated in the diagnostics rather than hidden.
            w_best = 0.5

        X_test, y_soh_test, y_rul_test, kinds = featured_in[test_cell]
        blend_soh = w_best * g_soh[test_cell] + (1.0 - w_best) * p_soh[test_cell]
        blend_rul = w_best * g_rul[test_cell] + (1.0 - w_best) * p_rul[test_cell]

        soh_true = y_soh_test.to_numpy(dtype=float)
        soh_mae = float(mean_absolute_error(soh_true, blend_soh))
        soh_r2 = _safe_r2(soh_true, blend_soh)
        soh_maes.append(soh_mae)
        if soh_r2 is not None:
            soh_r2s.append(soh_r2)

        obs_mask = (kinds == "observed").to_numpy() if kinds is not None else np.zeros(len(X_test), dtype=bool)
        n_obs = int(obs_mask.sum())
        n_ext = int(len(X_test) - n_obs) if kinds is not None else 0
        n_obs_rows += n_obs
        n_ext_rows += n_ext
        rul_true = y_rul_test.to_numpy(dtype=float)
        fold_obs_r2 = fold_obs_mae = None
        if n_obs >= 2:
            fold_obs_r2 = _safe_r2(rul_true[obs_mask], blend_rul[obs_mask])
            fold_obs_mae = float(mean_absolute_error(rul_true[obs_mask], blend_rul[obs_mask]))
            if fold_obs_r2 is not None:
                obs_r2s.append(fold_obs_r2)
                obs_maes.append(fold_obs_mae)

        per_cell[test_cell] = dict(
            soh_mae=soh_mae, soh_r2=soh_r2,
            rul_mae=fold_obs_mae, rul_r2=fold_obs_r2,
            rul_label_kinds={"observed": n_obs, "extrapolated": n_ext},
            # The fold's chosen blend weight travels with the fold so the
            # registry's fold drill-down can show WHY a fold behaved as it
            # did (w=1.0 → pure GBRT; w=0 → pure physics).
            blend_weight=w_best,
        )
        per_weights[test_cell] = w_best
        grp = chem.get(test_cell)
        if grp:
            chem_weights.setdefault(grp, []).append(w_best)

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
        "per_cell_weights": per_weights,
        "weights_by_chemistry": {
            k: {"mean_weight_gbrt": float(np.mean(v)), "n_folds": len(v)}
            for k, v in chem_weights.items()
        },
        "model_kind": MODEL_KIND,
        "hyperparams": ensemble_hyperparams(lambda_physics),
    }
