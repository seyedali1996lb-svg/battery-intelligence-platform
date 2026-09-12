"""Tests for batlab.validation.leakage_lint — the mechanical Tier-0 guard.

The lint exists because the worst number this platform ever reported
(Severson RUL R² = 0.9994) came from `fade_rate_50cy` — a FEATURE_COLUMNS
entry — being also the denominator of the expression that GENERATED the
extrapolated RUL labels. These tests pin the two invariants mechanically so
the bug class cannot return silently.
"""

import pandas as pd

from batlab.features.engineering import build_features, FEATURE_COLUMNS
from batlab.validation.leakage_lint import (
    check_rul_formula_isolation,
    check_label_provenance_coverage,
    rul_label_expression_vars,
    run_lint,
)


def _cycles(n: int, fade: float):
    """A minimal fading cycle table (soh_pct computed to match
    enrich_cycles()'s output shape, as build_features expects)."""
    return pd.DataFrame({
        "cycle_number": range(1, n + 1),
        "capacity_ah": [2.0 - fade * i for i in range(1, n + 1)],
        "resistance_ohm": [0.05] * n,
        "temperature_c": [25.0] * n,
        "soh_pct": [(2.0 - fade * i) / 2.0 * 100.0 for i in range(1, n + 1)],
    })


def test_fade_rate_50cy_is_not_a_feature_anymore():
    """The historical leak: fade_rate_50cy generated the labels it was fed
    as a feature. It must not be in FEATURE_COLUMNS — this assertion is the
    regression test for the audit finding."""
    assert "fade_rate_50cy" not in FEATURE_COLUMNS


def test_lint_passes_on_current_code():
    # No featured sample in a unit context: pass None so only the source-
    # parsing invariant runs (the provenance-column invariant needs a frame).
    report = run_lint(None)
    assert report["ok"], report["violations"]


def test_lint_finds_the_label_expression():
    """The parser must actually locate the df["rul"] assignments — if the
    code shape changes so the lint can't find them, that's a lint failure,
    not a silent pass."""
    vars_read = rul_label_expression_vars()
    assert vars_read["found"]
    # The known formula quantities appear in the read-set:
    assert {"capacity_ah", "fade_rate_50cy"} <= vars_read["observed"]


def test_lint_catches_a_reintroduced_leak(monkeypatch):
    """Simulate someone adding fade_rate_50cy back to FEATURE_COLUMNS: the
    lint must flag it as a violation."""
    import batlab.validation.leakage_lint as lint
    monkeypatch.setattr(lint, "FEATURE_COLUMNS", [*FEATURE_COLUMNS, "fade_rate_50cy"])
    violations = lint.check_rul_formula_isolation()
    assert violations
    assert any("fade_rate_50cy" in v for v in violations)


def test_label_provenance_column_exists_and_is_well_typed():
    df = build_features(_cycles(50, 0.001), cell_id="LintCell1")
    assert "rul_label_kind" in df.columns
    assert set(df["rul_label_kind"].unique()) <= {"observed", "extrapolated"}
    assert check_label_provenance_coverage(df) == []


def test_label_provenance_missing_is_a_violation():
    df = build_features(_cycles(50, 0.001), cell_id="LintCell2")
    violations = check_label_provenance_coverage(df.drop(columns=["rul_label_kind"]))
    assert violations and "rul_label_kind" in violations[0]
