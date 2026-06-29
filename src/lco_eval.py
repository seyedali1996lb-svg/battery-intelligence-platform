"""
Leave-cell-out cross-validation.

Exported function: run_lco(cell_data) -> dict of LCO metrics.
Can also be run directly as a CLI script for the synthetic cells.

Why LCO instead of a row-level holdout split?
  A chronological split on the concatenated multi-cell dataset puts the tail
  cycles of the last cell in the test set. The model has seen all of the
  other cells entirely during training, so it's not being tested on an unseen
  cell -- just unseen cycle indices. LCO is the correct evaluation for asking
  "does this model generalise to a cell it has never seen?"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from features import build_features, get_model_matrix

GBRT_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42,
)

RUL_RELIABLE_FLOOR = 0.3   # LCO R2 below this -> show "Not calibrated" in UI


def run_lco(cell_data: dict) -> dict:
    """
    Run leave-cell-out cross-validation on a dict of cell DataFrames.

    Args:
        cell_data: {cell_id: DataFrame} where each DataFrame is the raw
                   enriched cycles DataFrame (output of enrich_cycles()).
                   Features are built internally per cell.

    Returns:
        {
          "soh_r2":      float,   # mean LCO R2 across all folds
          "soh_mae":     float,   # mean LCO MAE across all folds (%)
          "rul_r2":      float,   # mean LCO R2 for RUL
          "rul_mae":     float,   # mean LCO MAE for RUL (cycles)
          "rul_reliable": bool,   # rul_r2 >= RUL_RELIABLE_FLOOR
          "per_cell":    dict,    # per-fold breakdown
        }
    """
    # Build feature matrices per cell
    featured = {}
    for cell_id, df in cell_data.items():
        if "soh_pct" not in df.columns:
            # Already enriched; build features only
            df_feat = build_features(df)
        else:
            df_feat = build_features(df)
        X, y_soh, y_rul = get_model_matrix(df_feat)
        featured[cell_id] = (X, y_soh, y_rul)

    cell_ids = list(featured.keys())
    if len(cell_ids) < 2:
        return {"soh_r2": float("nan"), "soh_mae": float("nan"),
                "rul_r2": float("nan"), "rul_mae": float("nan"),
                "rul_reliable": False, "per_cell": {}}

    soh_maes, soh_r2s, rul_maes, rul_r2s = [], [], [], []
    per_cell = {}

    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]

        X_train     = pd.concat([featured[c][0] for c in train_cells])
        y_soh_train = pd.concat([featured[c][1] for c in train_cells])
        y_rul_train = pd.concat([featured[c][2] for c in train_cells])
        X_test      = featured[test_cell][0]
        y_soh_test  = featured[test_cell][1]
        y_rul_test  = featured[test_cell][2]

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_train)
        Xte_sc = scaler.transform(X_test)

        soh_m = GradientBoostingRegressor(**GBRT_PARAMS).fit(Xtr_sc, y_soh_train)
        rul_m = GradientBoostingRegressor(**GBRT_PARAMS).fit(Xtr_sc, y_rul_train)

        soh_pred = soh_m.predict(Xte_sc)
        rul_pred = rul_m.predict(Xte_sc)

        soh_mae = mean_absolute_error(y_soh_test, soh_pred)
        soh_r2  = r2_score(y_soh_test, soh_pred)
        rul_mae = mean_absolute_error(y_rul_test, rul_pred)
        rul_r2  = r2_score(y_rul_test, rul_pred)

        soh_maes.append(soh_mae); soh_r2s.append(soh_r2)
        rul_maes.append(rul_mae); rul_r2s.append(rul_r2)
        per_cell[test_cell] = dict(soh_mae=soh_mae, soh_r2=soh_r2,
                                   rul_mae=rul_mae, rul_r2=rul_r2)

    mean_rul_r2 = float(np.mean(rul_r2s))
    return {
        "soh_r2":       float(np.mean(soh_r2s)),
        "soh_mae":      float(np.mean(soh_maes)),
        "rul_r2":       mean_rul_r2,
        "rul_mae":      float(np.mean(rul_maes)),
        "rul_reliable": mean_rul_r2 >= RUL_RELIABLE_FLOOR,
        "per_cell":     per_cell,
    }


# ---------------------------------------------------------------------------
# CLI: validate synthetic cells
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import build_battery, CELL_STRESS_PROFILES

    battery = build_battery("Oxford_B1", list(CELL_STRESS_PROFILES.keys()))
    cell_data = {cid: cell["cycles"] for cid, cell in battery["cells"].items()}

    print("Leave-Cell-Out Cross-Validation -- Synthetic cells")
    print("=" * 60)
    print(f"  {'Test cell':<10}  {'SOH MAE':>8}  {'SOH R2':>8}  {'RUL MAE':>9}  {'RUL R2':>8}")
    print("-" * 60)

    result = run_lco(cell_data)

    for cid, m in result["per_cell"].items():
        print(f"  {cid:<10}  {m['soh_mae']:>7.3f}%  {m['soh_r2']:>8.4f}  "
              f"{m['rul_mae']:>8.1f}cy  {m['rul_r2']:>8.4f}")

    print("-" * 60)
    print(f"  {'MEAN':<10}  {result['soh_mae']:>7.3f}%  {result['soh_r2']:>8.4f}  "
          f"{result['rul_mae']:>8.1f}cy  {result['rul_r2']:>8.4f}")
    print()
    print(f"SOH generalises: {'yes' if result['soh_r2'] > 0.5 else 'no'} (R2={result['soh_r2']:.3f})")
    print(f"RUL reliable:    {'yes' if result['rul_reliable'] else 'no -- show Calibrating'} (R2={result['rul_r2']:.3f})")
