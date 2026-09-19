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

Grading a model that is not this platform's GBRT
-----------------------------------------------
The methodology above is model-agnostic; only the model was not. Pass
`forecaster=` to swap it without touching a single rule: None keeps the
platform's own GBRT (the default, and the configuration every published
number here was measured under), while anything
batlab.harness.as_factory() understands is graded identically — an unfitted
sklearn estimator, a zero-argument factory returning a fresh model, or a
CallableForecaster wrapping your own fit/predict (PyTorch, an ODE fit, a
subprocess call to MATLAB).

The factory is called once per fold AND once per target, so a fold can
never be fitted on another fold's state — and a pre-fitted estimator is
rejected at the seam rather than deep-copied, because a model that already
saw the held-out cell would produce a meaningless, beautiful R².
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from batlab.features.engineering import build_features, get_model_matrix, get_rul_label_kinds
# GBRT_PARAMS stays importable from here as well as from batlab.models.gbrt:
# src/experiment_registry.py reads it from this module to pin the
# hyperparameters each logged run was trained with.
from batlab.models.gbrt import GBRT_PARAMS
from batlab._parallel import map_folds
from batlab.validation import fold_cache as fold_cache_mod
from batlab.harness.forecaster import (
    ForecasterLike,
    as_factory,
    default_forecaster,
    fit_forecaster,
)

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


def unwrap_cell_data(cell_data: dict) -> dict:
    """Normalize a cell-data dict to {cell_id: raw cycles DataFrame}.

    Callers build this map in two wrapper shapes — `build_battery()` cells
    ({"cell_id": ..., "cycles": df}) and the dataset loaders' dicts
    ({"cycles": df}) — and any harness that calls build_features() on a
    value directly would crash on (or worse, misread) a wrapper. Both
    shapes are unwrapped here; plain DataFrames pass through untouched.
    """
    out: dict = {}
    for cid, val in (cell_data or {}).items():
        if isinstance(val, pd.DataFrame):
            out[cid] = val
        elif isinstance(val, dict) and isinstance(val.get("cycles"), pd.DataFrame):
            out[cid] = val["cycles"]
        # Anything else is not a cell record; drop it rather than crash
        # the whole fleet's evaluation on one malformed entry.
    return out


def run_lco(
    cell_data: dict,
    seed: int = 42,
    featured: "dict | None" = None,
    forecaster: "ForecasterLike | None" = None,
    include_predictions: bool = False,
    use_fold_cache: bool = True,
) -> dict:
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
        forecaster: the model to grade. None = the platform's own GBRT (the
                   configuration the published numbers used). Anything
                   batlab.harness.as_factory() accepts is graded by the same
                   folds, metrics, and label-provenance rules.
        include_predictions: also return each fold's out-of-fold rows
                   (y_true / y_pred / observed-label mask) under the fold's
                   'predictions' key. Off by default: the arrays are
                   fold-sized and only the harness's conformal interval
                   calibration consumes them.
        use_fold_cache: replay folds this exact configuration has already
                   computed, instead of refitting them (see "Repeating a
                   run" below). True is right for measurement pipelines and
                   for anything the app boots; a caller whose PURPOSE is to
                   re-derive the numbers from code — an independent
                   replication, a metric regression gate — passes False, so
                   it cannot pass by replaying a stored answer. Ignored when
                   a `forecaster` was supplied, which disables the cache
                   outright.

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
          "confidence_intervals": dict|None,  # fold-level bootstrap CIs, see
                                # batlab.validation.bootstrap — None on n<2 folds
          "per_cell":    dict,    # per-fold breakdown incl. per-cell label kinds
                                # (+ a 'predictions' key when include_predictions)
          "fold_cache":  dict,    # {mode, enabled, key, dir, hits, fitted} —
                                # how much of this run was replayed rather than
                                # refitted (batlab.validation.fold_cache)
        }

    Repeating a run
    ---------------
    Every fold is a pure function of the pool's cells, their feature frames,
    the hyperparameters and the seed, so this function caches each completed
    fold's result on disk and replays it when those ingredients are unchanged
    (batlab.validation.fold_cache; `BATLAB_LCO_CACHE=off` disables it). That is
    what makes an interrupted 46-cell run resumable and a repeated boot cheap:
    replayed folds are the same numbers the refit would have produced, because
    the fold is pure. A caller-supplied `forecaster=` turns the cache off
    entirely — its folds are not keyed on anything this module can vouch for —
    and so does `use_fold_cache=False`, which is what a verification caller
    (batlab.validation.replication's recompute, the metric regression gate)
    passes: a check that replays a stored answer is not a check.
    """
    factory = as_factory(forecaster, default=default_forecaster(seed))
    if factory is None:  # pragma: no cover - default_forecaster() never returns None
        raise ValueError(
            "run_lco needs either a forecaster or the platform default; both "
            "were None, which means as_factory() was called with neither."
        )
    cell_data = unwrap_cell_data(cell_data)

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

    # ── Fingerprint + per-fold cache ───────────────────────────────────────
    # The dataset/environment fingerprint is computed BEFORE the folds now
    # rather than after: it is what a fold's cache key is built from (see
    # batlab.validation.fold_cache), and hashing the cells once up front costs
    # a fraction of a second against the minutes the key exists to save. The
    # value returned below is exactly the one this function always returned —
    # same normalization, same digest source — just computed earlier.
    from batlab.validation import fingerprints as _fp
    raw_frames = unwrap_cell_data(cell_data)
    fingerprint = {
        "dataset": _fp.dataset_fingerprint(raw_frames),
        "environment": _fp.environment_snapshot(),
    }
    fold_cache = fold_cache_mod.for_run(
        dataset_sha256=fingerprint["dataset"]["dataset_sha256"],
        env_snapshot=fingerprint["environment"],
        cell_ids=cell_ids,
        seed=seed,
        include_predictions=include_predictions,
        params=GBRT_PARAMS,
        # A caller-supplied forecaster is a model this module has no identity
        # for; caching its folds would risk grading the wrong estimator on a
        # hit. Supplying one therefore turns the cache off completely, as does
        # an explicit use_fold_cache=False (see the docstring: verifiers such
        # as replication's recompute must re-derive, never replay).
        enabled=forecaster is None and use_fold_cache,
    )

    def _run_fold(test_cell: str) -> dict:
        """One leave-cell-out fold: fit on every other cell, score this one.
        Pure (no shared state written), so folds can run concurrently --
        see batlab._parallel.

        A completed fold is REPLAYED from batlab.validation.fold_cache when
        the same cells, features, hyperparameters, seed and numeric stack have
        already produced it -- which is what lets an interrupted run resume
        and a repeated boot skip the refit entirely. The fold is pure, so
        replay and refit are the same numbers by construction."""
        cached = fold_cache.get(test_cell)
        if cached is not None:
            return cached

        X, y_soh, y_rul, kinds = featured_in[test_cell]
        train_cells = [c for c in cell_ids if c != test_cell]

        X_train     = pd.concat([featured_in[c][0] for c in train_cells])
        y_soh_train = pd.concat([featured_in[c][1] for c in train_cells])
        y_rul_train = pd.concat([featured_in[c][2] for c in train_cells])
        X_test      = X
        y_soh_test  = y_soh
        y_rul_test  = y_rul

        # One fresh model per (fold, target), straight from the factory —
        # see the module docstring's "Grading a model that is not this
        # platform's GBRT". The default factory is the scaled-GBRT
        # configuration every published number here was measured under.
        soh_pred = np.asarray(
            fit_forecaster(factory(), X_train, y_soh_train).predict(X_test), dtype=float
        )
        rul_pred = np.asarray(
            fit_forecaster(factory(), X_train, y_rul_train).predict(X_test), dtype=float
        )

        soh_mae = mean_absolute_error(y_soh_test, soh_pred)
        soh_r2  = _safe_r2(y_soh_test, soh_pred)

        # ── Split this fold's RUL rows by label provenance ──
        kinds = kinds.reindex(X.index) if kinds is not None else None
        obs_mask = (kinds == _LABEL_OBSERVED).to_numpy() if kinds is not None else np.zeros(len(X), dtype=bool)
        ext_mask = (kinds == _LABEL_EXTRAPOLATED).to_numpy() if kinds is not None else np.zeros(len(X), dtype=bool)

        n_obs = int(obs_mask.sum())
        n_ext = int(ext_mask.sum())
        n_unk = len(X) - n_obs - n_ext

        predictions = None
        if include_predictions:
            predictions = {
                "soh_true": np.asarray(y_soh_test, dtype=float),
                "soh_pred": soh_pred,
                "rul_true": np.asarray(y_rul_test, dtype=float),
                "rul_pred": rul_pred,
                "rul_label_observed": obs_mask,
            }

        rul_true = np.asarray(y_rul_test, dtype=float)

        fold_obs_r2 = fold_ext_r2 = None
        fold_obs_mae = fold_ext_mae = None
        if n_obs >= 2:
            fold_obs_r2 = _safe_r2(rul_true[obs_mask], rul_pred[obs_mask])
            fold_obs_mae = float(mean_absolute_error(rul_true[obs_mask], rul_pred[obs_mask]))
        if n_ext >= 2:
            fold_ext_r2 = _safe_r2(rul_true[ext_mask], rul_pred[ext_mask])
            fold_ext_mae = float(mean_absolute_error(rul_true[ext_mask], rul_pred[ext_mask]))

        result = dict(
            soh_mae=soh_mae, soh_r2=soh_r2,
            n_obs=n_obs, n_ext=n_ext, n_unk=n_unk,
            fold_obs_r2=fold_obs_r2, fold_obs_mae=fold_obs_mae,
            fold_ext_r2=fold_ext_r2, fold_ext_mae=fold_ext_mae,
            predictions=predictions,
        )
        # Durable BEFORE the aggregate is assembled: a run killed after this
        # fold finishes keeps it, so the next run fits only what never landed.
        fold_cache.put(test_cell, result)
        return result

    # Folds are independent; run them concurrently and re-assemble in the
    # original cell order so every list below is identical to the serial
    # loop's (means, per_cell insertion order, and the bootstrap CIs all
    # depend on that order).
    fold_results = map_folds(_run_fold, cell_ids)

    soh_maes, soh_r2s = [], []
    per_cell = {}
    n_obs_rows = 0
    n_ext_rows = 0
    obs_r2s, obs_maes = [], []
    ext_r2s, ext_maes = [], []

    for test_cell, fr in zip(cell_ids, fold_results):
        soh_maes.append(fr["soh_mae"])
        if fr["soh_r2"] is not None:
            soh_r2s.append(fr["soh_r2"])
        n_obs_rows += fr["n_obs"]
        n_ext_rows += fr["n_ext"]
        if fr["fold_obs_r2"] is not None:
            obs_r2s.append(fr["fold_obs_r2"])
            obs_maes.append(fr["fold_obs_mae"])
        if fr["fold_ext_r2"] is not None:
            ext_r2s.append(fr["fold_ext_r2"])
            ext_maes.append(fr["fold_ext_mae"])

        per_cell[test_cell] = dict(
            soh_mae=fr["soh_mae"], soh_r2=fr["soh_r2"],
            # Headline per-cell RUL metrics: observed rows only. None when
            # this fold has no observed-EOL rows — the fold's RUL cannot be
            # validated, which is a fact to surface, not a zero to hide.
            rul_mae=fr["fold_obs_mae"], rul_r2=fr["fold_obs_r2"],
            rul_mae_extrapolated=fr["fold_ext_mae"], rul_r2_extrapolated=fr["fold_ext_r2"],
            rul_label_kinds={
                _LABEL_OBSERVED: fr["n_obs"],
                _LABEL_EXTRAPOLATED: fr["n_ext"],
                "unknown": fr["n_unk"],
            },
        )
        if include_predictions:
            per_cell[test_cell]["predictions"] = fr["predictions"]

    n_rul_rows = n_obs_rows + n_ext_rows
    coverage = (n_obs_rows / n_rul_rows) if n_rul_rows > 0 else 0.0

    mean_obs_r2 = float(np.mean(obs_r2s)) if obs_r2s else float("nan")
    mean_ext_r2 = float(np.mean(ext_r2s)) if ext_r2s else None

    reliable = (
        bool(obs_r2s)
        and mean_obs_r2 >= RUL_RELIABLE_FLOOR
        and coverage >= MIN_RUL_LABEL_OBSERVED_FRACTION
    )

    # Fold-level bootstrap CIs (Tier-1 #7): the aggregate is a MEAN over
    # leave-cell-out folds, and on a 4-cell fleet that mean has a wide
    # sampling distribution the point estimate hides. Computed on the same
    # per-fold lists the means use, so the interval describes exactly the
    # population the headline averages. n=1 fleets return None and the UI
    # renders "—" rather than a fabricated bracket.
    from batlab.validation.bootstrap import lco_confidence_intervals
    confidence_intervals = lco_confidence_intervals(
        soh_r2s=soh_r2s, soh_maes=soh_maes,
        obs_r2s=obs_r2s, obs_maes=obs_maes,
    )

    # Tier-6 fingerprints: WHAT DATA and WHAT ENVIRONMENT produced these
    # numbers, hashed per cell (holdout's digest normalization) so two runs
    # claiming the same dataset are byte-comparable. Logged with every run
    # via the registry; the Tier-4 NASA loader incident is the failure mode
    # this retires (same name, different bytes, silently incomparable rows).
    # Computed above (the fold cache keys on it); returned here unchanged.

    return {
        "fingerprint": fingerprint,
        "fold_cache": fold_cache.summary(),
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
        "confidence_intervals": confidence_intervals,
        "per_cell":     per_cell,
    }
