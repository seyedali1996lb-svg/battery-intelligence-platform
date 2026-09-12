"""
Leave-cell-out cross-validation.

Exported function: run_lco(cell_data) -> dict of LCO metrics.

Why LCO instead of a row-level holdout split?
  A chronological split on the concatenated multi-cell dataset puts the tail
  cycles of the last cell in the test set. The model has seen all of the
  other cells entirely during training, so it's not being tested on an unseen
  cell -- just unseen cycle indices. LCO is the correct evaluation for asking
  "does this model generalise to a cell it has never seen?"

Why RUL is scored on TWO populations (and only one is a skill claim)
--------------------------------------------------------------------
The RUL target built by build_features() has two fundamentally different
label kinds, recorded per row in `rul_label_kind` (FEATURE_VERSION >= v12):

  "observed"     — the cell reached the EOL threshold inside its recorded
                   window, so RUL = (EOL cycle − current cycle) is MEASURED.
  "extrapolated" — the cell never reached EOL in-window, so RUL is a
                   closed-form linear extrapolation from fade_rate_50cy —
                   which is itself FEATURE_COLUMNS entry #3. A model scored
                   on these labels is graded on recovering the formula that
                   generated its own target: R² ≈ 1 there is an identity, not
                   skill, over horizons up to 10× the observed data (this is
                   exactly what Severson's former 0.9994 RUL R² was).

So run_lco() reports:

  rul_r2 / rul_mae        — OBSERVED-EOL rows only. This is the headline
                            RUL number and the only one `rul_reliable` can
                            be built from. None/nan when no fold has
                            observed rows.
  rul_extrapolated_r2/mae — extrapolated-label rows, reported separately as
                            a formula-recovery diagnostic. Never mixed into
                            the headline, never allowed to set rul_reliable.

`rul_reliable` additionally requires that at least
MIN_RUL_LABEL_OBSERVED_FRACTION of the evaluated RUL rows carry observed
labels — a population where only 1 of 12 folds can be validated must not
present a mean over that one fold as a dataset-level claim.

SOH is unaffected: the SOH target is always measured.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
from batlab.models.gbrt import GBRT_PARAMS

RUL_RELIABLE_FLOOR = 0.3   # LCO R2 below this -> show "Not calibrated" in UI

# Headline rul_reliable additionally requires this fraction of evaluated RUL
# rows to carry OBSERVED (measured-EOL) labels. Below it, the population is
# too formula-dominated for even a good observed-subset R² to be a
# dataset-level claim.
MIN_RUL_LABEL_OBSERVED_FRACTION = 0.5

_LABEL_OBSERVED = "observed"
_LABEL_EXTRAPOLATED = "extrapolated"


def _safe_r2(y_true, y_pred) -> "float | None":
    """R² that returns None instead of raising/NaN when the subset is too
    small or has zero variance to define R² at all."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or float(np.var(y_true)) <= 0.0:
        return None
    return float(r2_score(y_true, y_pred))


def run_lco(cell_data: dict, seed: int = 42, featured: "dict | None" = None) -> dict:
    """
    Run leave-cell-out cross-validation on a dict of cell DataFrames.

    Args:
        cell_data: {cell_id: DataFrame} where each DataFrame is the raw
                   cycle-level DataFrame (batlab.datasets.schema kind="cycle").
                   Features are built internally per cell.
        seed:      Random seed for every fold's GradientBoostingRegressor —
                   parameterized (rather than hardcoded) so a split manifest
                   (see batlab.validation.manifest) can reproduce an exact
                   prior run.
        featured:  optional {cell_id: already-built feature DataFrame} from
                   the SAME FEATURE_VERSION — callers holding raw_fdfs (e.g.
                   app/_data.py's train_and_predict) pass it so the
                   (expensive) feature pipeline, including the PyBaMM-backed
                   physics calibration, is not re-run per cell. Frames missing
                   rul_label_kind (pre-v12) are rebuilt from cell_data.

    Returns:
        {
          "soh_r2":      float,   # mean LCO R2 across all folds (SOH labels are always measured)
          "soh_mae":     float,   # mean LCO MAE across all folds (%)
          "rul_r2":      float,   # mean LCO R2 for RUL on OBSERVED-EOL rows only (nan when none)
          "rul_mae":     float,   # mean LCO MAE for RUL on OBSERVED-EOL rows only (cycles)
          "rul_reliable": bool,   # observed-row rul_r2 >= RUL_RELIABLE_FLOOR AND observed coverage
          "rul_label_coverage": float,  # fraction of evaluated RUL rows with observed labels
          "n_rul_observed_rows": int,
          "n_rul_extrapolated_rows": int,
          "rul_extrapolated_r2": float|None,  # formula-recovery diagnostic, separate pool
          "rul_extrapolated_mae": float|None,
          "per_cell":    dict,    # per-fold breakdown incl. per-cell label kinds
        }
    """
    params = {**GBRT_PARAMS, "random_state": seed}

    # Build feature matrices per cell
    featured_in = {}
    for cell_id, df in cell_data.items():
        if featured is not None and cell_id in featured:
            frame = featured[cell_id]
            # A pre-v12 frame cannot say which labels are observed; rebuild.
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
        "rul_extrapolated_r2": None, "rul_extrapolated_mae": None,
    }
    if len(cell_ids) < 2:
        return empty

    soh_maes, soh_r2s = [], []
    per_cell = {}
    n_obs_rows = 0
    n_ext_rows = 0
    obs_r2s, obs_maes = [], []
    ext_r2s, ext_maes = [], []

    for test_cell in cell_ids:
        X, y_soh, y_rul, kinds = featured_in[test_cell]
        train_cells = [c for c in cell_ids if c != test_cell]

        X_train     = pd.concat([featured_in[c][0] for c in train_cells])
        y_soh_train = pd.concat([featured_in[c][1] for c in train_cells])
        y_rul_train = pd.concat([featured_in[c][2] for c in train_cells])
        X_test      = X
        y_soh_test  = y_soh
        y_rul_test  = y_rul

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_train)
        Xte_sc = scaler.transform(X_test)

        soh_m = GradientBoostingRegressor(**params).fit(Xtr_sc, y_soh_train)
        rul_m = GradientBoostingRegressor(**params).fit(Xtr_sc, y_rul_train)

        soh_pred = soh_m.predict(Xte_sc)
        rul_pred = rul_m.predict(Xte_sc)

        soh_mae = mean_absolute_error(y_soh_test, soh_pred)
        soh_r2  = _safe_r2(y_soh_test, soh_pred)
        soh_maes.append(soh_mae)
        if soh_r2 is not None:
            soh_r2s.append(soh_r2)

        # ── Split this fold's RUL rows by label provenance ──
        kinds = kinds.reindex(X.index) if kinds is not None else None
        obs_mask = (kinds == _LABEL_OBSERVED).to_numpy() if kinds is not None else np.zeros(len(X), dtype=bool)
        ext_mask = (kinds == _LABEL_EXTRAPOLATED).to_numpy() if kinds is not None else np.zeros(len(X), dtype=bool)

        n_obs = int(obs_mask.sum())
        n_ext = int(ext_mask.sum())
        n_unk = len(X) - n_obs - n_ext
        n_obs_rows += n_obs
        n_ext_rows += n_ext

        rul_true = np.asarray(y_rul_test, dtype=float)

        fold_obs_r2 = fold_ext_r2 = None
        fold_obs_mae = fold_ext_mae = None
        if n_obs >= 2:
            fold_obs_r2 = _safe_r2(rul_true[obs_mask], rul_pred[obs_mask])
            fold_obs_mae = float(mean_absolute_error(rul_true[obs_mask], rul_pred[obs_mask]))
            if fold_obs_r2 is not None:
                obs_r2s.append(fold_obs_r2)
                obs_maes.append(fold_obs_mae)
        if n_ext >= 2:
            fold_ext_r2 = _safe_r2(rul_true[ext_mask], rul_pred[ext_mask])
            fold_ext_mae = float(mean_absolute_error(rul_true[ext_mask], rul_pred[ext_mask]))
            if fold_ext_r2 is not None:
                ext_r2s.append(fold_ext_r2)
                ext_maes.append(fold_ext_mae)

        per_cell[test_cell] = dict(
            soh_mae=soh_mae, soh_r2=soh_r2,
            # Headline per-cell RUL metrics: observed rows only. None when
            # this fold has no observed-EOL rows — the fold's RUL cannot be
            # validated, which is a fact to surface, not a zero to hide.
            rul_mae=fold_obs_mae, rul_r2=fold_obs_r2,
            rul_mae_extrapolated=fold_ext_mae, rul_r2_extrapolated=fold_ext_r2,
            rul_label_kinds={
                _LABEL_OBSERVED: n_obs,
                _LABEL_EXTRAPOLATED: n_ext,
                "unknown": n_unk,
            },
        )

    n_rul_rows = n_obs_rows + n_ext_rows
    coverage = (n_obs_rows / n_rul_rows) if n_rul_rows > 0 else 0.0

    mean_obs_r2 = float(np.mean(obs_r2s)) if obs_r2s else float("nan")
    mean_ext_r2 = float(np.mean(ext_r2s)) if ext_r2s else None

    reliable = (
        bool(obs_r2s)
        and mean_obs_r2 >= RUL_RELIABLE_FLOOR
        and coverage >= MIN_RUL_LABEL_OBSERVED_FRACTION
    )

    return {
        "soh_r2":       float(np.mean(soh_r2s)) if soh_r2s else float("nan"),
        "soh_mae":      float(np.mean(soh_maes)) if soh_maes else float("nan"),
        "rul_r2":       mean_obs_r2,
        "rul_mae":      float(np.mean(obs_maes)) if obs_maes else float("nan"),
        "rul_reliable": bool(reliable),
        "rul_label_coverage": float(coverage),
        "n_rul_observed_rows": n_obs_rows,
        "n_rul_extrapolated_rows": n_ext_rows,
        "rul_extrapolated_r2": mean_ext_r2,
        "rul_extrapolated_mae": float(np.mean(ext_maes)) if ext_maes else None,
        "per_cell":     per_cell,
    }
