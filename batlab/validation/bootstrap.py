"""Bootstrap confidence intervals for the LCO accuracy numbers.

Why this exists
---------------
Every headline accuracy number on the Benchmark page is a MEAN over
leave-cell-out folds — NASA's R² rests on 4 folds, the synthetic fleet on 8.
A mean of 4 numbers has a wide sampling distribution, and "R² = 0.958" printed
without any spread invites exactly the over-reading this project keeps having
to correct: with 4 cells, the honest statement is "0.958 [0.83, 0.99] across
leave-cell-out folds", and the width of that bracket IS the accuracy claim.

Unit choice (deliberate): the bootstrap resamples FOLDS (cells), not rows.
Rows within one cell are strongly correlated — an aging curve sampled
densely is effectively one observation about that cell — so a row-level
bootstrap would fake precision by treating ~10,000 correlated rows as
independent. The cell is the sampling unit of the claim "this model
generalizes across cells of this chemistry", so cells are what we resample.
n_folds is reported next to every interval so the smallness is visible, not
hidden: on a 4-cell fleet the 95% interval is wide because the evidence is
thin, and that is the true statement.

Method: pairwise percentile bootstrap (Efron 1979; percentiles per
Efron & Tibshirani 1993, ch. 13). The same (y, ŷ) pair is resampled together
per fold, so each replicate recomputes the metric on a plausible re-draw of
the fleet. Seeded and deterministic for reproducibility.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BOOT = 2000
DEFAULT_CI_LEVEL = 0.95


def bootstrap_fold_metric(
    values: "list[float] | list[None]",
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 0,
) -> "dict | None":
    """Percentile CI for the mean of one per-fold metric.

    values : the per-fold metric values (one per leave-cell-out fold). None
             entries (folds where the metric was not evaluable — e.g. RUL
             folds with no observed-EOL rows) are dropped, and the interval
             describes only the folds that produced a number; the caller
             must surface the None count separately (run_lco already reports
             n_rul_observed_rows / n_rul_extrapolated_rows for exactly this).
    Returns {"mean", "lo", "hi", "n", "n_boot", "ci_level"} or None when
    fewer than 2 usable folds exist — a CI on n=1 is a fabricated number.
    """
    vals = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if vals.size < 2:
        return None

    rng = np.random.default_rng(seed)
    n = vals.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = vals[idx].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": float(vals.mean()),
        "lo":   float(np.quantile(boot_means, alpha)),
        "hi":   float(np.quantile(boot_means, 1.0 - alpha)),
        "n":    int(n),
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "unit": "leave-cell-out fold",
        "method": "pairwise percentile bootstrap over folds",
    }


def lco_confidence_intervals(
    soh_r2s: "list",
    soh_maes: "list",
    obs_r2s: "list | None" = None,
    obs_maes: "list | None" = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 0,
) -> "dict | None":
    """Build the CI dict run_lco() attaches to its aggregate.

    soh_r2s / soh_maes are the per-fold SOH metrics (always evaluable);
    obs_r2s / obs_maes the per-fold observed-EOL RUL metrics (may be fewer,
    or absent entirely when the fleet has no measured EOL crossings).
    Returns None when even SOH has <2 folds — callers store None and the
    UI renders "—", never a fabricated bracket.
    """
    soh_r2_ci = bootstrap_fold_metric(soh_r2s, n_boot, ci_level, seed)
    if soh_r2_ci is None:
        return None
    return {
        "soh_r2":  soh_r2_ci,
        "soh_mae": bootstrap_fold_metric(soh_maes, n_boot, ci_level, seed),
        "rul_r2":  bootstrap_fold_metric(obs_r2s, n_boot, ci_level, seed) if obs_r2s is not None else None,
        "rul_mae": bootstrap_fold_metric(obs_maes, n_boot, ci_level, seed) if obs_maes is not None else None,
        "n_folds": soh_r2_ci["n"],
        "n_boot":  n_boot,
        "ci_level": ci_level,
        "seed": seed,
        "note": (
            "Resampling unit is the leave-cell-out fold (cell), not the row: "
            "rows within a cell are correlated, so cell-level resampling is "
            "the honest spread. Width reflects fold count — a wide interval "
            "on a small fleet is thin evidence stated plainly."
        ),
    }


def format_ci(ci: "dict | None", decimals: int = 3) -> str:
    """Render one CI dict as "0.958 [0.83, 0.99]" — or "—" when absent."""
    if not ci or ci.get("lo") is None:
        return "—"
    return (
        f"{ci['mean']:.{decimals}f} "
        f"[{ci['lo']:.{decimals}f}, {ci['hi']:.{decimals}f}]"
    )
