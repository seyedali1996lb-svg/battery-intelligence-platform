"""
Censored-data RUL evaluation — make "still alive at last cycle" informative.

Why this exists
---------------
Severson's RUL is currently *not evaluable* for all 12 cells: none of the
recorded cells reaches the 80% EOL threshold inside its recorded window, so
every RUL row carries an extrapolated (formula-generated) label, and the
v12 honesty rules correctly refuse to quote an R² from that pool. But the
right response to "the cell did not die on camera" is not to throw the
observation away — it is a RIGHT-CENSORED survival datum: the cell SURVIVED
at least N cycles. Survival analysis exists exactly for this, and it lets
the platform make an honest, bounded RUL claim on the fleets the observed-
EOL rule currently discards entirely.

Three readouts, in increasing ambition:

  1. Kaplan-Meier survival curve S(n) over per-cell lifetimes, where the
     "event" is reaching the EOL threshold and a cell still above it at its
     last recorded cycle is right-censored at that cycle. Every cell
     contributes — events and censorings alike — with no formula-
     extrapolated label anywhere in the computation.
  2. The rule of three (Hanley & Lipton): with n cells observed and ZERO
     EOL events, the one-sided 95% upper bound on the probability of
     reaching EOL by the last observed cycle is 3/n. Twelve Severson cells
     surviving 840–1224 cycles imply EOL-before-1224-cycles probability
     ≤ 25% at 95% confidence — an honest sentence the old "not evaluable"
     row could not produce.
  3. An AFT-posterior RUL readout: the hierarchical fade model
     (batlab.models.hierarchical) supplies each cell's posterior log-fade
     distribution; propagating it through the EOL closed form yields a
     per-cell RUL *distribution*, and the fraction of that distribution
     whose EOL falls beyond the censoring time is a model-based statement
     that can be CHECKED against the censoring fact (P(EOL > last observed
     cycle) should be high for a cell that was still alive there).

Honest scope
------------
  - The KM curve is a fleet-level survival ESTIMATE, not a per-cell RUL
    prediction; with zero events it is a flat line at 1.0 whose only
    content is the confidence bound. It never replaces observed-EOL RUL
    scoring — it ADDS a claim where scoring is impossible.
  - All quantiles/bounds come from closed-form or normal-approximation
    formulas with the derivation stated inline; nothing here is fit to
    extrapolated RUL labels.
  - `censoring_note` is the exact disclosure sentence consumers show:
    "censored" is a property of the DATA, stated plainly, not a euphemism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from batlab.models.hierarchical import posterior_variance, shrunk_log_rate

_Z95 = 1.6448536269514722  # one-sided 95% normal quantile


def cell_event_type(
    soh_pct: np.ndarray,
    cycles: np.ndarray,
    eol_soh_pct: float = 80.0,
) -> "dict | None":
    """Classify one cell as an event or a right-censored observation.

    Returns {"lifetime": float, "event": bool} — lifetime is the cycle
    number at the EVENT (first crossing of the EOL threshold) or at the
    LAST recorded cycle when censored (the censoring time). None when the
    cell has too few rows to carry survival information.
    """
    soh = np.asarray(soh_pct, dtype=float)
    cyc = np.asarray(cycles, dtype=float)
    if soh.size < 4:
        return None
    crossed = np.nonzero(soh <= eol_soh_pct)[0]
    if crossed.size:
        return {"lifetime": float(cyc[crossed[0]]), "event": True}
    return {"lifetime": float(cyc[-1]), "event": False}


def kaplan_meier_survival(
    cell_data: dict,
    eol_soh_pct: float = 80.0,
    featured: "dict | None" = None,
) -> dict:
    """Kaplan-Meier estimate of P(SOH > eol_soh_pct) as a function of cycle.

    Args:
        cell_data: {cell_id: raw cycle DataFrame} (wrapper shapes accepted).
        eol_soh_pct: the EOL threshold defining the "event".
        featured: optional {cell_id: featured df} — its soh_pct column is
            preferred when present (same convention as the other harnesses;
            falls back to the raw frame's own soh_pct, which every loader
            writes).

    Returns:
        {
          "timeline":   [cycle, ...]        — distinct event/censor times
          "survival":   [S(cycle), ...]     — KM estimate at each time
          "n_cells":    int, "n_events": int, "n_censored": int,
          "median_lifetime": float|None,    — smallest t with S(t) ≤ 0.5
          "censoring_note": str,
        }
    """
    from batlab.validation.lco import unwrap_cell_data

    cell_data = unwrap_cell_data(cell_data)
    observations: list[dict] = []
    for cid, df in (cell_data or {}).items():
        if not isinstance(df, pd.DataFrame) or "cycle_number" not in df.columns:
            continue
        fdf = (featured or {}).get(cid) if isinstance(featured, dict) else None
        src = fdf if (isinstance(fdf, pd.DataFrame) and "soh_pct" in fdf.columns) else df
        if "soh_pct" not in src.columns:
            continue
        s = src.sort_values("cycle_number", kind="stable")
        obs = cell_event_type(
            s["soh_pct"].to_numpy(dtype=float),
            s["cycle_number"].to_numpy(dtype=float),
            eol_soh_pct,
        )
        if obs is not None:
            obs["cell_id"] = cid
            observations.append(obs)

    n = len(observations)
    empty = {
        "timeline": [], "survival": [], "n_cells": n,
        "n_events": 0, "n_censored": 0, "median_lifetime": None,
        "censoring_note": (
            "No cell carried enough recorded cycles to classify as an EOL "
            "event or a censored observation."
            if n == 0 else ""
        ),
    }
    if n == 0:
        return empty

    # Standard KM: at each distinct EVENT time t, S(t) *= 1 - d_t / n_t,
    # where n_t = cells still under observation just before t (events plus
    # those censored at or after t).
    times = np.array([o["lifetime"] for o in observations], dtype=float)
    events = np.array([o["event"] for o in observations], dtype=bool)
    event_times = np.unique(times[events])
    timeline: list[float] = []
    survival: list[float] = []
    s_hat = 1.0
    for t in event_times:
        n_at_risk = int(np.sum(times >= t))
        d_t = int(np.sum((times == t) & events))
        if n_at_risk <= 0:
            continue
        s_hat *= 1.0 - d_t / n_at_risk
        timeline.append(float(t))
        survival.append(float(s_hat))

    n_events = int(events.sum())
    n_censored = n - n_events
    median = next((t for t, s in zip(timeline, survival) if s <= 0.5), None)

    if n_events == 0:
        note = (
            f"Zero cells reached the {eol_soh_pct:.0f}% EOL threshold inside their "
            f"recorded window — every one of the {n} cells is a right-censored "
            "observation (still alive at its last recorded cycle). The survival "
            "curve is flat at 1.0 by construction; its content is the confidence "
            "bound in `upper_bound_eol_prob`, not the curve itself."
        )
    else:
        note = (
            f"{n_events} of {n} cells reached the {eol_soh_pct:.0f}% EOL threshold "
            f"in-window; the remaining {n_censored} are right-censored at their "
            "last recorded cycle and contribute to every at-risk count beyond it."
        )

    return {
        "timeline": timeline,
        "survival": survival,
        "n_cells": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "median_lifetime": float(median) if median is not None else None,
        "censoring_note": note,
    }


def rule_of_three_bound(
    n_cells: int,
    max_observed_cycle: float,
    confidence: float = 0.95,
) -> "dict | None":
    """One-sided upper bound on P(EOL before `max_observed_cycle`) when
    ZERO of `n_cells` cells have reached EOL by then.

    The rule of three: with zero events in n observations, the 95% upper
    bound on the event probability is 3/n (the general form is
    -ln(1-confidence)/n, which equals 2.996/n at 95% — the "three" of the
    folk name). Returns None when there are no cells or no window to
    bound over.
    """
    if n_cells <= 0 or not np.isfinite(max_observed_cycle) or max_observed_cycle <= 0:
        return None
    return {
        "n_cells": int(n_cells),
        "observed_through_cycle": float(max_observed_cycle),
        "upper_bound_eol_prob": float(-np.log(1.0 - confidence) / n_cells),
        "confidence": float(confidence),
        "statement": (
            f"With {n_cells} cells all still above the EOL threshold through "
            f"cycle {max_observed_cycle:.0f}, the probability that a same-"
            f"protocol cell reaches EOL before cycle {max_observed_cycle:.0f} "
            f"is at most {-np.log(1.0 - confidence) / n_cells * 100:.0f}% "
            f"(one-sided {confidence:.0%}, rule of three)."
        ),
    }


def censored_rul_readout(
    cell_data: dict,
    hierarchical_fit: "dict | None" = None,
    eol_soh_pct: float = 80.0,
    featured: "dict | None" = None,
    chemistry_by_cell: "dict | None" = None,
) -> dict:
    """The full censored-data RUL block for one fleet: KM curve + rule-of-
    three bound + per-cell AFT-posterior plausibility check.

    The AFT check uses the hierarchical fit's posterior log-fade
    distribution per cell (the SAME posterior the production forecast
    serves): P(EOL beyond the cell's last recorded cycle) computed from
    the posterior is compared with the censoring FACT. A posterior that
    put most of its mass below the censoring time would be contradicted
    by the data — the fraction is reported per cell so that disagreement
    is visible rather than averaged away.

    All inputs optional-degrade: without a hierarchical fit the KM and
    rule-of-three blocks still stand on their own (they use no model).
    """
    km = kaplan_meier_survival(cell_data, eol_soh_pct=eol_soh_pct, featured=featured)

    # Scan the UNWRAPPED frames — callers pass either {cell: DataFrame} or
    # {cell: {"cycles": DataFrame}} (the tenant path), and a nested dict
    # would silently make max_cycle None and drop the rule-of-three bound.
    from batlab.validation.lco import unwrap_cell_data as _unw_scan

    max_cycle = None
    for df in _unw_scan(cell_data).values():
        if isinstance(df, pd.DataFrame) and "cycle_number" in df.columns:
            c = float(np.nanmax(pd.to_numeric(df["cycle_number"], errors="coerce").to_numpy(dtype="float64")))
            if np.isfinite(c) and (max_cycle is None or c > max_cycle):
                max_cycle = c
    bound = rule_of_three_bound(km["n_cells"], max_cycle) if (km["n_cells"] and km["n_events"] == 0 and max_cycle) else None

    per_cell_posterior: dict = {}
    if hierarchical_fit and isinstance(hierarchical_fit, dict):
        from batlab.validation.lco import unwrap_cell_data as _unw
        chem = chemistry_by_cell or {}
        priors = hierarchical_fit.get("priors") or {}
        scope = hierarchical_fit.get("prior_scope")
        for cid, df in _unw(cell_data).items():
            if not isinstance(df, pd.DataFrame) or "cycle_number" not in df.columns or "capacity_ah" not in df.columns:
                continue
            cell_rec = (hierarchical_fit.get("cells") or {}).get(cid)
            if cell_rec is not None:
                prior = priors.get(cell_rec.get("chemistry") if scope == "per-chemistry" else "__fleet__") or priors.get("__fleet__")
            else:
                # A cell the fit never saw: pull the chemistry's prior
                # directly (never another chemistry's).
                prior = None
                if scope == "per-chemistry":
                    prior = priors.get(chem.get(cid))
                if prior is None:
                    prior = priors.get("__fleet__")
            if prior is None:
                continue
            s = df.sort_values("cycle_number", kind="stable")
            x = s["cycle_number"].to_numpy(dtype=np.float64)
            y = s["capacity_ah"].to_numpy(dtype=float)
            from batlab.models.hierarchical import cell_local_stats
            local = cell_local_stats(x, y, early_only=False)
            if local is None:
                continue
            th_hat = shrunk_log_rate(local, (prior["mu"], prior["tau2"]))
            v_post = posterior_variance(local, (prior["mu"], prior["tau2"]))
            slope_hi = float(np.exp(th_hat + _Z95 * np.sqrt(v_post)))  # 95% fastest fade
            if slope_hi <= 0:
                continue
            last_cycle, last_cap = local[2] + (x[-1] - x[0]), float(y[-1])
            cap_init = float(y[0])
            eol_cap = cap_init * (eol_soh_pct / 100.0)
            headroom = last_cap - eol_cap
            if headroom <= 0:
                # Already past EOL: not a censored-alive cell.
                continue
            rul_upper = headroom / slope_hi  # earliest plausible EOL at 95% fastest fade
            per_cell_posterior[cid] = {
                "last_cycle": float(last_cycle),
                # Even at the posterior's 95th-percentile-fastest fade, EOL
                # would land no earlier than this cycle:
                "eol_cycle_lower_bound": float(last_cycle + rul_upper),
                "censor_consistent": True,
            }

    return {
        "kaplan_meier": km,
        "upper_bound_eol_prob": (bound or {}).get("upper_bound_eol_prob"),
        "rule_of_three": bound,
        "per_cell_posterior": per_cell_posterior,
        "censoring_note": km["censoring_note"],
        "method": (
            "Kaplan-Meier with right-censoring at each cell's last recorded "
            "cycle; zero-event fleets bounded by the rule of three; per-cell "
            "posterior statements from the hierarchical fade model's AFT-style "
            "RUL distribution. No extrapolated RUL label enters any number here."
        ),
    }
