"""Unit tests for app/utils.py's shared render primitives.

app/utils.py is an app-layer module (imports streamlit at module level),
which is why nothing in tests/ has unit-tested it directly before now --
everything else about it is implicitly covered via the AppTest
integration suite (tests/test_app_state_combinations.py). metric_tile_html()
is a pure string-returning function though (no st.* calls), so it's
directly testable the same way src/ pure-logic modules are.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))

from utils import metric_tile_html


def test_metric_tile_html_includes_label_and_value():
    html = metric_tile_html("Total Cells", "12")
    assert "Total Cells" in html
    assert ">12<" in html


def test_metric_tile_html_omits_sub_div_when_sub_is_empty():
    """No sub-caption given -> no dangling empty <div> for it, so a page
    composing several tiles doesn't get stray empty elements."""
    html = metric_tile_html("Total Cells", "12")
    assert "margin-top:2px" not in html  # that's the sub-caption's div


def test_metric_tile_html_includes_sub_when_given():
    html = metric_tile_html("Total Cells", "12", sub="12 Severson real")
    assert "12 Severson real" in html
    assert "margin-top:2px" in html


def test_metric_tile_html_uses_given_value_color():
    html = metric_tile_html("Fleet Health", "85.9%", value_color="#fc8181")
    assert "#fc8181" in html
