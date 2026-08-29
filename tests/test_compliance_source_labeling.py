"""
Regression tests for app/_pages/compliance.py's page_reports() and
page_sustainability() -- both used to resolve a "source" string as a 2-way
is_nasa/else branch (no Severson case), so a real Severson cell fed
financial_comparison()/sustainability_snapshot() with the synthetic fleet's
CELL_NOMINAL_KWH["synth"] constant instead of its own ["severson"] value.
Same fix pattern already applied to src/passport.py, app/_pages/overview.py,
and app/_pages/decision.py in this session.

These call the page functions directly (not via Streamlit's AppTest) since
bare `st.*` calls outside a script run context are safe no-ops (they log a
"missing ScriptRunContext" warning and return their default value) -- this
lets the test control the input DataFrame precisely (needed for
page_reports(), whose financial_comparison() call is gated behind
`soh <= 85.0`) without depending on a specific real cell's actual SOH.
"""

import sys
import pathlib
from unittest.mock import patch

import pandas as pd

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from _pages import compliance
import consequences


def _sample_df(last_soh: float = 97.0):
    return pd.DataFrame({
        "soh_pct":        [100, 99, 98, last_soh],
        "cycle_number":   [1, 2, 3, 4],
        "resistance_ohm": [0.05, 0.051, 0.052, 0.053],
        "fade_rate_30cy": [0.001] * 4,
        "capacity_ah":    [2.0, 1.98, 1.96, 1.94],
        "rul_pred":       [None, None, None, 600.0],
    })


def test_page_reports_severson_cell_financial_comparison_uses_severson_source():
    """
    Regression: page_reports() had `source = "nasa" if is_nasa else "synth"`
    feeding financial_comparison(source=source), reached whenever a cell is
    at/below 85% SOH. A degraded Severson cell's second-life financial
    figures were silently computed off the synthetic fleet's capacity
    constant. Wraps the real financial_comparison() (rather than replacing
    it with a canned return value) so the rest of page_reports' rendering
    still gets a fully valid result -- only the `source` kwarg is captured.
    """
    real_financial_comparison = consequences.financial_comparison
    captured = {}

    def _spy(*args, **kwargs):
        captured["source"] = kwargs.get("source")
        return real_financial_comparison(*args, **kwargs)

    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    df = _sample_df(last_soh=70.0)  # <= 85.0 to trigger the second-life branch

    with patch("consequences.financial_comparison", side_effect=_spy), \
         patch("report_pdf.build_report_pdf", return_value=(b"", "doc1")), \
         patch("passport_export.to_json_ld", return_value={}):
        compliance.page_reports("S-b1c2", df, bundle, rul_reliable=True)

    assert captured.get("source") == "severson", (
        f"page_reports() must resolve a Severson cell to source='severson' for "
        f"financial_comparison(), not {captured.get('source')!r}"
    )


def test_page_sustainability_severson_cell_uses_severson_source():
    """
    Regression: page_sustainability() had the same `source = "nasa" if
    is_nasa else "synth"` 2-way branch feeding both CELL_NOMINAL_KWH[source]
    (nominal capacity for the CO2 use-phase calc) and
    sustainability_snapshot(source=source) -- a Severson cell's carbon and
    material-recovery figures used the synthetic fleet's tiny capacity
    constant instead of Severson's own. Wraps the real sustainability_snapshot()
    so page rendering still gets a valid result; only the `source` kwarg is
    captured.
    """
    real_sustainability_snapshot = consequences.sustainability_snapshot
    captured = {}

    def _spy(*args, **kwargs):
        captured["source"] = kwargs.get("source")
        return real_sustainability_snapshot(*args, **kwargs)

    df = _sample_df()

    with patch("consequences.sustainability_snapshot", side_effect=_spy):
        compliance.page_sustainability("S-b1c2", df)

    assert captured.get("source") == "severson", (
        f"page_sustainability() must resolve a Severson cell to source='severson', "
        f"not {captured.get('source')!r}"
    )
