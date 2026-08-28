"""
Per-prediction local feature attribution (occlusion-based, SHAP-style).

batlab.models.gbrt exposes global feature_importances_ — how much each
feature reduced training loss across the whole ensemble. That answers "in
general, what drives this model?" but not "why did THIS cell's RUL come
out at 512 cycles?" This module adds the local, per-prediction answer.

The occlusion method is deliberately dependency-free (no `shap` package):
for one prediction x and one feature j, draw n_samples values of feature j
from a reference distribution, substitute each into x one at a time, and
record the mean absolute change in the model's prediction. A feature the
model leans on for THIS row shows a large mean change; an irrelevant
feature leaves the prediction untouched. This is a SHAP-style local
attribution in the sense that it attributes the prediction to features via
counterfactual perturbation — it is not the exact, game-theoretic TreeSHAP
values (which require the `shap` package's tree-path algorithm and are
exact for tree ensembles in a way perturbation sampling is not).

Honest scope: occlusion attribution measures sensitivity, not a Shapley
value; with n_samples finite it is a Monte Carlo estimate, so it is
seeded and should be read as a ranking signal, not a precise number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MODEL_KEYS = {"rul": "rul_model", "soh": "soh_model"}


def occlusion_attribution(
    model_bundle: dict,
    X: pd.DataFrame,
    X_ref: pd.DataFrame | None = None,
    n_samples: int = 25,
    seed: int = 42,
    model: str = "rul",
) -> pd.DataFrame:
    """
    Per-row, per-feature occlusion attribution for one fitted model.

    Args:
        model_bundle: dict returned by batlab.models.gbrt.train_models().
        X: feature matrix (one row per prediction to explain). Must contain
           the bundle's `feature_names` columns.
        X_ref: reference DataFrame whose feature distributions supply the
               substitution values. Defaults to X itself (self-referencing,
               fine for exploration; pass an explicit reference population
               for a stricter counterfactual).
        n_samples: Monte Carlo draws per (row, feature) pair.
        seed: RNG seed — results are deterministic for a fixed seed.
        model: "rul" or "soh" — which of the bundle's two point models to
               attribute.

    Returns:
        DataFrame with one row per row of X, one column per feature, values
        = mean absolute prediction change when that feature is replaced by
        random draws from X_ref's distribution.
    """
    if model not in _MODEL_KEYS:
        raise ValueError(f"model must be one of {sorted(_MODEL_KEYS)}, got {model!r}")
    key = _MODEL_KEYS[model]
    if key not in model_bundle:
        raise ValueError(f"model_bundle has no {key!r} — is this a train_models() bundle?")
    feature_names = list(model_bundle["feature_names"])
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"X is missing feature columns required by the bundle: {missing}")

    estimator = model_bundle[key]
    scaler = model_bundle["scaler"]
    ref = X_ref if X_ref is not None else X
    if list(ref.columns) != list(X.columns):
        raise ValueError("X_ref must have the same columns as X.")

    rng = np.random.default_rng(seed)
    Xw = X[feature_names]
    X_scaled = scaler.transform(Xw)
    base = estimator.predict(X_scaled)

    out = np.empty((len(Xw), len(feature_names)), dtype=float)
    for i in range(len(Xw)):
        row = Xw.iloc[[i]]
        for j, col in enumerate(feature_names):
            draws = ref[col].to_numpy()[rng.integers(0, len(ref), size=n_samples)]
            perturbed = pd.DataFrame(np.repeat(row.to_numpy(), n_samples, axis=0), columns=Xw.columns)
            perturbed[col] = draws
            preds = estimator.predict(scaler.transform(perturbed))
            out[i, j] = float(np.mean(np.abs(preds - base[i])))

    return pd.DataFrame(out, index=Xw.index, columns=feature_names)


def mean_attribution(attr_df: pd.DataFrame) -> pd.Series:
    """Average each feature's attribution across all explained rows — a
    local-first alternative to global feature_importances_."""
    return attr_df.mean(axis=0).sort_values(ascending=False)


def top_attributions(attr_df: pd.DataFrame, top_n: int = 5) -> list[list[dict]]:
    """
    The top `top_n` features per explained row, as [{feature, attribution}]
    dicts sorted descending — the shape a UI can render as a "why this
    prediction?" list per cell.
    """
    result = []
    for _, row in attr_df.iterrows():
        ranked = row.sort_values(ascending=False).head(top_n)
        result.append([{"feature": f, "attribution": float(v)} for f, v in ranked.items()])
    return result
