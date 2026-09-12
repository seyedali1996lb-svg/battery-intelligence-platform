"""Tests for batlab.validation.bootstrap — fold-level bootstrap CIs.

The property that matters: the interval brackets the point mean when folds
are well-behaved, widens as evidence shrinks, and never fabricates a bracket
from a single fold (a CI on n=1 would be fake precision on the exact
small-fleet case the module exists to be honest about).
"""

from batlab.validation.bootstrap import (
    bootstrap_fold_metric,
    lco_confidence_intervals,
    format_ci,
)


def test_ci_brackets_mean_for_stable_folds():
    # Ten folds tightly clustered — the 95% CI must contain the mean.
    vals = [0.90, 0.91, 0.92, 0.90, 0.93, 0.91, 0.92, 0.90, 0.91, 0.92]
    ci = bootstrap_fold_metric(vals, seed=42)
    assert ci is not None
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    # Tight cluster → narrow interval (not proof, but guards a gross bug).
    assert ci["hi"] - ci["lo"] < 0.1


def test_ci_widens_with_fewer_folds():
    wide = bootstrap_fold_metric([0.80, 0.95, 0.70, 0.90], seed=7)
    # Same population shape, but resampling a 2-fold mean swings far more
    # than a 4-fold mean — compare against a large-n draw of similar spread.
    vals = [0.70, 0.80, 0.90, 0.95] * 5
    narrow = bootstrap_fold_metric(vals, seed=7)
    assert wide is not None and narrow is not None
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])


def test_single_fold_returns_none_no_fabricated_bracket():
    assert bootstrap_fold_metric([0.95]) is None
    assert bootstrap_fold_metric([]) is None


def test_none_folds_dropped_and_counted():
    # RUL folds with no observed-EOL rows report None — they must not
    # silently become zeros (that would drag the mean and fake a tight CI).
    vals = [0.5, None, 0.7, None]
    ci = bootstrap_fold_metric(vals, seed=1)
    assert ci is not None
    assert ci["n"] == 2
    assert ci["mean"] == 0.6


def test_deterministic_for_fixed_seed():
    vals = [0.4, 0.6, 0.5, 0.7, 0.3]
    a = bootstrap_fold_metric(vals, seed=123)
    b = bootstrap_fold_metric(vals, seed=123)
    assert a == b


def test_lco_confidence_intervals_none_when_soh_has_one_fold():
    out = lco_confidence_intervals(soh_r2s=[0.9], soh_maes=[0.1])
    assert out is None


def test_lco_confidence_intervals_shape():
    out = lco_confidence_intervals(
        soh_r2s=[0.95, 0.93, 0.97, 0.94],
        soh_maes=[0.5, 0.6, 0.4, 0.55],
        obs_r2s=[0.6, 0.4],
        obs_maes=[8.0, 12.0],
    )
    assert out is not None
    assert out["n_folds"] == 4
    assert out["soh_r2"]["n"] == 4
    assert out["rul_r2"]["n"] == 2  # RUL CI describes only evaluable folds
    assert out["soh_r2"]["lo"] <= out["soh_r2"]["mean"] <= out["soh_r2"]["hi"]
    assert "fold" in out["note"]


def test_lco_confidence_intervals_rul_none_when_no_observed_folds():
    out = lco_confidence_intervals(
        soh_r2s=[0.9, 0.85, 0.88], soh_maes=[0.5, 0.6, 0.5],
        obs_r2s=[], obs_maes=[],
    )
    assert out is not None
    assert out["rul_r2"] is None
    assert out["rul_mae"] is None


def test_format_ci():
    assert format_ci({"mean": 0.958, "lo": 0.83, "hi": 0.99, "n": 4}) == "0.958 [0.830, 0.990]"
    assert format_ci(None) == "—"
    assert format_ci({}) == "—"
