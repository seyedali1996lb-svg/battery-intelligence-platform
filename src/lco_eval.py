"""Leave-cell-out cross-validation — run directly to check for data leakage."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from data_loader import build_battery, CELL_STRESS_PROFILES
from features import build_features, get_model_matrix

battery = build_battery("Oxford_B1", list(CELL_STRESS_PROFILES.keys()))

cell_data = {}
for cell_id, cell in battery["cells"].items():
    df_feat = build_features(cell["cycles"])
    X, y_soh, y_rul = get_model_matrix(df_feat)
    cell_data[cell_id] = (X, y_soh, y_rul)

cell_ids = list(cell_data.keys())

GBRT_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                   subsample=0.8, random_state=42)

print("Leave-Cell-Out Cross-Validation")
print("=" * 60)
print(f"  {'Test cell':<10}  {'SOH MAE':>8}  {'SOH R2':>8}  {'RUL MAE':>9}  {'RUL R2':>8}")
print("-" * 60)

soh_maes, soh_r2s, rul_maes, rul_r2s = [], [], [], []

for test_cell in cell_ids:
    train_cells = [c for c in cell_ids if c != test_cell]

    X_train     = pd.concat([cell_data[c][0] for c in train_cells])
    y_soh_train = pd.concat([cell_data[c][1] for c in train_cells])
    y_rul_train = pd.concat([cell_data[c][2] for c in train_cells])
    X_test      = cell_data[test_cell][0]
    y_soh_test  = cell_data[test_cell][1]
    y_rul_test  = cell_data[test_cell][2]

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

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

    print(f"  {test_cell:<10}  {soh_mae:>7.3f}%  {soh_r2:>8.4f}  {rul_mae:>8.1f}cy  {rul_r2:>8.4f}")

print("-" * 60)
print(f"  {'MEAN':<10}  {np.mean(soh_maes):>7.3f}%  {np.mean(soh_r2s):>8.4f}  {np.mean(rul_maes):>8.1f}cy  {np.mean(rul_r2s):>8.4f}")
print(f"  {'STD':<10}  {np.std(soh_maes):>7.3f}%  {np.std(soh_r2s):>8.4f}  {np.std(rul_maes):>8.1f}cy  {np.std(rul_r2s):>8.4f}")
print()
print("Verdict:")
mean_soh_r2 = np.mean(soh_r2s)
if mean_soh_r2 > 0.85:
    print(f"  SOH R2 = {mean_soh_r2:.3f} under LCO — original 0.96 is real generalisation.")
elif mean_soh_r2 > 0.50:
    print(f"  SOH R2 = {mean_soh_r2:.3f} under LCO — partial generalisation, some leakage in original.")
else:
    print(f"  SOH R2 = {mean_soh_r2:.3f} under LCO — original 0.96 was memorisation, not generalisation.")
