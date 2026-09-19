"""
Hierarchical partial-pooling estimator — the production "what happens next"
model.

Why this exists
---------------
The GBRT (batlab.models.gbrt) is the platform's strongest INTERPOLATOR
(leave-cell-out SOH R² 0.96–0.999) but structurally cannot forecast: the
prospective split shows it collapsing below a straight line on 2 of 3 real
fleets once it is denied the future, because a regression tree cannot
predict outside its training label range. What a real deployment asks —
"this cell has N cycles so far, what happens NEXT?" — is exactly the
question the GBRT cannot answer.

This module IS that answer, as a first-class production model rather than a
benchmark candidate:

  - each cell's fade rate (log per-cycle capacity loss) is shrunk toward
    its chemistry's fleet prior with empirical-Bayes precision weights —
    a short/noisy early window borrows strength from the fleet; a long,
    informative window stays with the cell's own data;
  - SOH is forecast as the shrunk straight line from the cell's own
    early-window anchor, extended beyond the last observed cycle;
  - RUL is forecast from the same line to the EOL threshold, with an
    80% interval derived from the posterior uncertainty of the shrunk
    log fade rate (an accelerated-failure-time-style interval: fade-rate
    uncertainty propagates to time-to-EOL through the closed form).

Honest scope (stated, not hidden)
---------------------------------
  - The fade law is LINEAR in cycle number. It cannot follow knee-type
    (LFP) degradation — measured honestly as SOH R² = 0.318 on Severson
    against the GBRT's 0.986 in the LCO benchmark. The routing layer
    (src/forecast_routing.py) picks GBRT vs. hierarchical per cell from
    the measured folds; this module never claims to be the right model
    everywhere.
  - The served RUL interval is a POSTERIOR interval from the shrunk fade
    rate, NOT a conformally calibrated one: its coverage has not been
    measured on unseen cells the way the GBRT's Q10/Q90 has (see
    batlab.validation.calibration). Callers must disclose that —
    `RUL_INTERVAL_CONVENTION` is the exact sentence to show.
  - The per-row forecast consumes only the cell's early window
    (MIN_HISTORY_FRACTION of recorded cycles) — the deployment-realistic
    information set, identical to the validation harness's convention so
    benchmark numbers and served forecasts describe the same estimator.

This module owns the estimation math; batlab/validation/hierarchical_lco.py
imports it so the LCO benchmark and the production estimator cannot drift
apart (the "same estimator, two implementations" failure mode).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The fraction of a cell's recorded cycles the model consumes as its
# "early window" — the deployment-realistic information set. The
# validation harness (batlab.validation.hierarchical_lco) re-exports these.
MIN_HISTORY_FRACTION = 0.3
MIN_HISTORY_FLOOR_CYCLES = 10

# Method-of-moments floor on the prior variance: a fleet of near-identical
# cells must not produce a zero-variance prior (that would make the
# shrinkage divide by zero and over-trust the fleet mean).
TAU2_FLOOR = 1e-4

# z-quantile of the standard normal for a central 80% interval.
_Z80 = 1.2815515655446004

MODEL_KIND = "hierarchical"

RUL_INTERVAL_CONVENTION = (
    "RUL interval from the hierarchical model's posterior fade-rate "
    "uncertainty (AFT-style: fade-rate quantiles propagated through the "
    "closed form to EOL) — posterior, NOT conformally calibrated; its "
    "coverage has not been measured on unseen cells."
)


def hierarchical_hyperparams() -> dict:
    """The recorded hyperparameters, mirroring GBRT_PARAMS / pinn_hyperparams."""
    return {
        "model_kind": MODEL_KIND,
        "fade_law": "linear per-cycle capacity loss, log-rate Gaussian prior",
        "min_history_fraction": MIN_HISTORY_FRACTION,
        "min_history_floor_cycles": MIN_HISTORY_FLOOR_CYCLES,
        "tau2_floor": TAU2_FLOOR,
        "estimator": "empirical-Bayes method of moments + precision-weighted shrinkage",
        "rul_interval": "posterior fade-rate quantiles (AFT-style), uncalibrated",
    }


# ---------------------------------------------------------------------------
# Cell-level estimation (shared with the LCO validation harness)
# ---------------------------------------------------------------------------

def cell_local_stats(
    cycles: np.ndarray, cap_ah: np.ndarray, early_only: bool,
) -> "tuple[float, float, float, float] | None":
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


def prior_from_fleet(fleet_stats: list) -> "tuple[float, float]":
    """Empirical-Bayes prior N(mu, tau^2) over LOG fade rates from the
    cells' (log theta, sampling variance) pairs. Both shapes the callers
    actually pass are accepted: the harness's positional tuples and this
    module's own {log_fade, log_fade_sampling_var} dicts (a dict is not a
    tuple — indexing it with 0 raises KeyError, which is exactly the bug
    this accept-both contract retired)."""
    def _pair(s):
        if isinstance(s, dict):
            return float(s["log_fade"]), float(s["log_fade_sampling_var"])
        return float(s[0]), float(s[1])
    pairs = [_pair(s) for s in fleet_stats]
    logs = np.array([p[0] for p in pairs], dtype=float)
    vs = np.array([p[1] for p in pairs], dtype=float)
    mu = float(np.mean(logs))
    tau2 = float(np.var(logs, ddof=1) - np.mean(vs)) if len(logs) >= 2 else 0.0
    return mu, max(tau2, TAU2_FLOOR)


def shrunk_log_rate(local: Sequence[float], prior: "tuple[float, float]") -> float:
    """Precision-weighted shrinkage of the cell's local log fade rate
    toward the fleet prior."""
    th_loc, v_loc = local[0], local[1]
    mu, tau2 = prior
    w_loc = 1.0 / max(v_loc, 1e-12)
    w_pri = 1.0 / max(tau2, 1e-12)
    return (w_loc * th_loc + w_pri * mu) / (w_loc + w_pri)


def posterior_variance(local: Sequence[float], prior: "tuple[float, float]") -> float:
    """Variance of the posterior over the cell's log fade rate:
    1 / (w_loc + w_pri). Drives the served RUL interval."""
    _, v_loc = local[0], local[1]
    _, tau2 = prior
    w_loc = 1.0 / max(v_loc, 1e-12)
    w_pri = 1.0 / max(tau2, 1e-12)
    return 1.0 / (w_loc + w_pri)


# ---------------------------------------------------------------------------
# Production fit
# ---------------------------------------------------------------------------

def fit_hierarchical(
    cell_data: dict,
    chemistry_by_cell: "dict | None" = None,
    eol_soh_pct: float = 80.0,
) -> dict:
    """Fit the hierarchical model on a fleet: per-cell full-history fade
    statistics + per-chemistry empirical-Bayes priors.

    Args:
        cell_data: {cell_id: raw cycle DataFrame} (both wrapper shapes the
            validation harness unwraps are accepted).
        chemistry_by_cell: optional {cell_id: chemistry} — the prior pools
            within chemistry groups (per-chemistry scope); falls back to a
            single fleet-local prior when chemistries are unknown or only
            one chemistry is present.
        eol_soh_pct: the EOL threshold the RUL forecast projects to.

    Returns a JSON-safe dict of per-cell parameters and priors — the
    "hierarchical_fit" a model bundle carries. Deliberately stores no
    cycle arrays: a bundle is a model, not a copy of the training data.
    """
    from batlab.validation.lco import unwrap_cell_data

    cell_data = unwrap_cell_data(cell_data)
    chem = chemistry_by_cell or {}

    cell_stats: dict = {}
    for cid, df in (cell_data or {}).items():
        if not isinstance(df, pd.DataFrame) or "cycle_number" not in df.columns or "capacity_ah" not in df.columns:
            continue
        s = df.sort_values("cycle_number", kind="stable")
        stats = cell_local_stats(
            s["cycle_number"].to_numpy(dtype=np.float64),
            s["capacity_ah"].to_numpy(dtype=float),
            early_only=False,  # the FIT uses full history, exactly like the harness's priors
        )
        if stats is None:
            continue
        cell_stats[cid] = {
            "log_fade": stats[0],
            "log_fade_sampling_var": stats[1],
            "anchor_cycle": stats[2],
            "anchor_capacity_ah": stats[3],
            "n_cycles": int(len(s)),
            "final_cycle": float(s["cycle_number"].iloc[-1]),
            "final_capacity_ah": float(s["capacity_ah"].iloc[-1]),
            "chemistry": chem.get(cid),
        }
        # The RUL forecast resolves EOL against the cell's fresh capacity —
        # its first observed measurement (same convention as the harness).
        cell_stats[cid]["initial_capacity_ah"] = float(s["capacity_ah"].iloc[0])

    if not cell_stats:
        return {"cells": {}, "priors": {}, "prior_scope": "none", "eol_soh_pct": eol_soh_pct,
                "hyperparams": hierarchical_hyperparams()}

    # Priors: per-chemistry when the caller supplies chemistries AND more
    # than one distinct chemistry is present; otherwise one fleet-local
    # prior (the item-1 special case, recorded as such).
    chems_present = {c.get("chemistry") for c in cell_stats.values()} - {None}
    per_chemistry = bool(chem and len(chems_present) > 1)

    priors: dict = {}
    if per_chemistry:
        for chemistry in sorted(chems_present):
            stats = [
                {"log_fade": c["log_fade"], "log_fade_sampling_var": c["log_fade_sampling_var"]}
                for c in cell_stats.values() if c.get("chemistry") == chemistry
            ]
            mu, tau2 = prior_from_fleet(stats)
            priors[chemistry] = {"mu": mu, "tau2": tau2, "n_cells": len(stats)}
    else:
        stats = [
            {"log_fade": c["log_fade"], "log_fade_sampling_var": c["log_fade_sampling_var"]}
            for c in cell_stats.values()
        ]
        mu, tau2 = prior_from_fleet(stats)
        priors["__fleet__"] = {"mu": mu, "tau2": tau2, "n_cells": len(stats)}

    return {
        "cells": cell_stats,
        "priors": priors,
        "prior_scope": "per-chemistry" if per_chemistry else "fleet-local",
        "eol_soh_pct": float(eol_soh_pct),
        "hyperparams": hierarchical_hyperparams(),
        "rul_interval_convention": RUL_INTERVAL_CONVENTION,
    }


# ---------------------------------------------------------------------------
# Production forecast
# ---------------------------------------------------------------------------

def forecast_soh(
    fit: "dict | None",
    cell_id: str,
    cycles: np.ndarray,
    capacities: np.ndarray,
) -> "dict | None":
    """Per-row hierarchical SOH/RUL forecast for one cell.

    The cell's own early window (first MIN_HISTORY_FRACTION of the GIVEN
    rows — the same population the validation harness consumes) provides
    the local log fade rate; it is shrunk toward the fit's prior and the
    resulting straight line is the forecast. Rows inside the consumed
    window are marked "insample" (the line fits them by construction);
    rows beyond it are the actual forecast ("extrapolated").

    Returns None when the cell has no usable window or the fit has no
    prior — callers must render "hierarchical forecast unavailable" rather
    than fall back silently to the GBRT's served columns.

    Returned arrays are aligned 1:1 with the input rows. Includes:
        soh_forecast   — % SOH on the shrunk line, clipped at 0
        forecast_kind  — "insample" | "extrapolated" per row
        rul_forecast   — cycles to EOL on the shrunk line (per row)
        rul_q10/rul_q90 — posterior AFT-style interval (see the module
                          docstring's honesty note)
        shrinkage_weight_prior — how much of the slope came from the fleet
    """
    if not fit or not isinstance(fit, dict):
        return None
    cells = fit.get("cells") or {}
    cell = cells.get(cell_id)
    priors = fit.get("priors") or {}
    if cell is not None:
        prior_key = cell.get("chemistry") if fit.get("prior_scope") == "per-chemistry" else "__fleet__"
        prior = priors.get(prior_key) or priors.get("__fleet__")
    else:
        prior = None

    x = np.asarray(cycles, dtype=float)
    y = np.asarray(capacities, dtype=float)
    if x.size < 4 or prior is None:
        return None

    local = cell_local_stats(x, y, early_only=True)
    if local is None or not np.isfinite(prior.get("mu", float("nan"))):
        return None

    th_hat = shrunk_log_rate(local, (prior["mu"], prior["tau2"]))
    slope = float(np.exp(th_hat))
    v_post = posterior_variance(local, (prior["mu"], prior["tau2"]))
    slope_lo = float(np.exp(th_hat - _Z80 * np.sqrt(v_post)))
    slope_hi = float(np.exp(th_hat + _Z80 * np.sqrt(v_post)))

    anchor_cycle, anchor_cap = local[2], local[3]
    cap_init = anchor_cap
    pred_cap = anchor_cap - slope * (x - anchor_cycle)
    soh_pred = np.clip(pred_cap / cap_init * 100.0, 0.0, None)

    # Early-window mask, in row space (the given rows are the full recorded
    # population — same convention as the harness's featured_in).
    n_hist = max(MIN_HISTORY_FLOOR_CYCLES, int(len(x) * MIN_HISTORY_FRACTION))
    kinds = np.where(np.arange(len(x)) < n_hist, "insample", "extrapolated")

    # RUL: cycles from each row until the line crosses EOL. The threshold
    # resolves against the cell's fresh capacity (its first measurement) —
    # identical to the validation harness.
    eol_cap = cap_init * (fit.get("eol_soh_pct", 80.0) / 100.0)
    headroom = pred_cap - eol_cap
    with np.errstate(divide="ignore", invalid="ignore"):
        rul = np.where(slope > 0, headroom / slope, np.nan)
        rul_q10 = np.where(slope_hi > 0, headroom / slope_hi, np.nan)  # faster fade → sooner EOL
        rul_q90 = np.where(slope_lo > 0, headroom / slope_lo, np.nan)
    rul = np.clip(rul, 0.0, None)
    rul_q10 = np.clip(rul_q10, 0.0, None)
    rul_q90 = np.clip(rul_q90, 0.0, None)

    return {
        "soh_forecast": soh_pred,
        "forecast_kind": kinds,
        "rul_forecast": rul,
        "rul_q10": rul_q10,
        "rul_q90": rul_q90,
        "shrinkage_weight_prior": float(
            (1.0 / max(prior["tau2"], 1e-12))
            / (1.0 / max(local[1], 1e-12) + 1.0 / max(prior["tau2"], 1e-12))
        ),
        "local_log_fade": local[0],
        "shrunk_log_fade": th_hat,
        "prior_mu": float(prior["mu"]),
        "prior_tau2": float(prior["tau2"]),
        "prior_scope": fit.get("prior_scope", "fleet-local"),
        "rul_interval_convention": RUL_INTERVAL_CONVENTION,
    }


def project_future_soh(fit, cell_id: str, horizon_cycles: int) -> "dict | None":
    """Extend the shrunk line `horizon_cycles` beyond the cell's last
    recorded cycle, with the posterior 80% band — the projection the
    Health page's 12-month forecast consumes.

    Returns {cycles, soh_pct, soh_q10_pct, soh_q90_pct} or None (same
    unavailability contract as forecast_soh).
    """
    if not fit or not isinstance(fit, dict):
        return None
    cell = (fit.get("cells") or {}).get(cell_id)
    priors = fit.get("priors") or {}
    prior = None
    if cell is not None:
        prior_key = cell.get("chemistry") if fit.get("prior_scope") == "per-chemistry" else "__fleet__"
        prior = priors.get(prior_key) or priors.get("__fleet__")
    if cell is None or prior is None:
        return None

    # The forecast line continues from the LAST RECORDED cycle outward;
    # the shrunk slope is the fit's estimate (full-history anchors), the
    # posterior variance combines the fit's sampling variance with the
    # prior exactly as forecast_soh does for the early window.
    th = cell["log_fade"]
    v_loc = cell["log_fade_sampling_var"]
    th_hat = shrunk_log_rate((th, v_loc), (prior["mu"], prior["tau2"]))
    v_post = posterior_variance((th, v_loc), (prior["mu"], prior["tau2"]))
    slope = float(np.exp(th_hat))
    slope_lo = float(np.exp(th_hat - _Z80 * np.sqrt(v_post)))
    slope_hi = float(np.exp(th_hat + _Z80 * np.sqrt(v_post)))

    cap_init = cell["initial_capacity_ah"]
    last_cycle = cell["final_cycle"]
    eol_cap = cap_init * (fit.get("eol_soh_pct", 80.0) / 100.0)
    headroom0 = cell["final_capacity_ah"] - eol_cap

    k = np.arange(1, int(horizon_cycles) + 1, dtype=float)
    cycles = last_cycle + k
    soh_c = np.clip((cell["final_capacity_ah"] - slope * k) / cap_init * 100.0, 0.0, None)
    soh_hi = np.clip((cell["final_capacity_ah"] - slope_lo * k) / cap_init * 100.0, 0.0, None)
    soh_lo = np.clip((cell["final_capacity_ah"] - slope_hi * k) / cap_init * 100.0, 0.0, None)

    def _cross(sl: float) -> "float | None":
        if sl <= 0:
            return None
        rul = headroom0 / sl
        return float(last_cycle + rul) if rul >= 0 else None

    return {
        "cycles": cycles,
        "soh_pct": soh_c,
        "soh_q10_pct": soh_lo,   # faster fade → lower SOH path
        "soh_q90_pct": soh_hi,
        "eol_cycle_central": _cross(slope),
        "eol_cycle_q10": _cross(slope_hi),
        "eol_cycle_q90": _cross(slope_lo),
        "rul_interval_convention": RUL_INTERVAL_CONVENTION,
        "shrinkage_weight_prior": float(
            (1.0 / max(prior["tau2"], 1e-12))
            / (1.0 / max(v_loc, 1e-12) + 1.0 / max(prior["tau2"], 1e-12))
        ),
    }


def export_priors(fit: dict) -> dict:
    """The fit's priors in a portable {chemistry: {mu, tau2, n_cells}} form —
    what a deployment caches so a chemistry-matched reference prior can be
    handed to an uploaded fleet too small to fit its own."""
    out = {}
    for key, p in (fit.get("priors") or {}).items():
        out[key] = {"mu": p.get("mu"), "tau2": p.get("tau2"), "n_cells": p.get("n_cells")}
    return out


def priors_for_chemistry(reference_fit: "dict | None", chemistry: "str | None") -> "tuple[float, float] | None":
    """The (mu, tau2) prior a reference fit holds for `chemistry`, or the
    fleet-local one when the fit is single-chemistry. None when nothing
    matches — the honest answer for an unrepresented chemistry."""
    if not reference_fit:
        return None
    priors = reference_fit.get("priors") or {}
    if chemistry and reference_fit.get("prior_scope") == "per-chemistry":
        p = priors.get(chemistry)
        if p:
            return float(p["mu"]), float(p["tau2"])
        return None
    p = priors.get("__fleet__")
    return (float(p["mu"]), float(p["tau2"])) if p else None
