"""The Streamlit host of the 3D cell scene (app/_scene_view.py, app/_pages/battery3d.py).

Two layers, matching this repo's convention for a page-level feature:

* the **iframe document builder** is tested directly and without a browser —
  what it embeds must round-trip as the same spec, the bundle must be linked by
  the same static route `app/static/theme.css` already uses, and a payload
  containing `</script>` must not be able to end the element early;
* the **page** gets a Streamlit AppTest smoke run, the pattern
  tests/test_degradation_space_3d.py uses, because a page that raises on its
  first render is exactly the failure a unit test of its helpers cannot see.
"""

from __future__ import annotations

import json
import os as _os
import pathlib
import re
import sys

import pytest

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401
import db as db_module
from streamlit.testing.v1 import AppTest

from _scene_view import BUNDLE_URL, scene_iframe_html, theme_from_app
from cell_scene import default_theme

MAIN_PY = str(pathlib.Path(_root) / "app" / "main.py")
SPEC_URL_TAG = re.compile(
    r'<script id="cell-scene-spec" type="application/json">(.*?)</script>', re.S
)


def _spec() -> dict:
    """A real spec, built the way the page builds it (synthetic frame → no data needed)."""
    import numpy as np
    import pandas as pd
    from cell_scene import build_cell_scene

    n = 120
    cycles = np.arange(1, n + 1, dtype=float)
    soh = 100.0 - 0.01 * cycles
    frame = pd.DataFrame({
        "cycle_number": cycles,
        "soh_pct": soh,
        "capacity_ah": 2.0 * soh / 100.0,
        "resistance_ohm": 0.045 + 0.0001 * cycles,
        "temperature_c": np.full(n, 30.0),
        "sop_pct": 100.0 * 0.045 / (0.045 + 0.0001 * cycles),
        "fade_rate_30cy": np.full(n, 0.002),
    })
    frame["resistance_normalized"] = frame["resistance_ohm"] / frame["resistance_ohm"].iloc[0]
    return build_cell_scene("B0005", frame, theme=theme_from_app())


# ---------------------------------------------------------------------------
# The iframe document
# ---------------------------------------------------------------------------

def test_the_iframe_links_the_bundle_over_the_route_streamlit_already_serves():
    """`app/static/theme.css` (app/main.py) is fetched this exact way, so the
    scene's 578 kB renderer travels the route that is already known to work —
    and is cached by the browser instead of re-sent on every rerun."""
    html = scene_iframe_html(_spec())
    assert f'<script src="{BUNDLE_URL}"></script>' in html
    assert BUNDLE_URL == "app/static/cell_scene/cell_scene.js"
    assert (pathlib.Path(_root) / BUNDLE_URL).is_file()


def test_the_document_embeds_the_spec_verbatim_and_reads_back_identical():
    spec = _spec()
    payload = SPEC_URL_TAG.search(scene_iframe_html(spec))
    assert payload, "the spec script tag is missing or malformed"
    assert json.loads(payload.group(1)) == spec


def test_a_payload_containing_a_script_tag_cannot_end_the_element_early():
    spec = _spec()
    spec["disclosures"] = ["this text contains </script> and would break a naive embed"]
    html = scene_iframe_html(spec)
    payload = SPEC_URL_TAG.search(html)
    assert payload
    assert json.loads(payload.group(1))["disclosures"] == spec["disclosures"]
    # Exactly one spec script tag, and the document still ends where it should.
    assert html.count('id="cell-scene-spec"') == 1
    assert html.rstrip().endswith("</html>")


def test_the_document_carries_the_controls_and_the_honesty_notes():
    html = scene_iframe_html(_spec())
    for control in ('id="explode"', 'id="cursor"', 'id="data-scaled"', 'id="annotations"',
                    'id="strip-casing"', 'id="play"', 'id="reset"'):
        assert control in html, f"the control strip lost {control}"
    # The fallback sentences are the difference between an empty box and a bug report.
    assert "window.CellScene" in html
    assert "did not load" in html
    assert "Every number" in html


def test_the_cursor_starts_at_today_and_spans_the_projection():
    spec = _spec()
    html = scene_iframe_html(spec)
    measured = len(spec["series"]["cycles"])
    assert f'id="cursor" type="range" min="0" max="{measured - 1}" value="{measured - 1}"' in html


def test_a_non_spec_argument_is_refused_rather_than_rendered_blank():
    for bad in ({}, None, "not a spec", {"parts": []}):
        with pytest.raises(ValueError):
            scene_iframe_html(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The theme
# ---------------------------------------------------------------------------

def test_light_mode_restyles_the_surface_without_touching_the_health_bands():
    dark = theme_from_app(light_mode=False)
    light = theme_from_app(light_mode=True)
    assert dark == default_theme()
    assert light["background"] != dark["background"]
    assert light["text"] != dark["text"]
    # The bands are semantics, not decoration: a cell the app calls Degrading
    # must be amber in both modes.
    assert light["sohBands"] == dark["sohBands"]


# ---------------------------------------------------------------------------
# Wiring: a page nobody can reach is a page nobody has
# ---------------------------------------------------------------------------

def test_the_page_is_in_the_navigation():
    from _sidebar import NAV_GROUPS

    entries = {key: label for _group, items in NAV_GROUPS for label, key in items}
    assert entries.get("battery3d") == "Battery 3D"


def test_the_router_dispatches_the_page_with_the_cell_it_needs():
    source = (pathlib.Path(_root) / "app" / "_router.py").read_text(encoding="utf-8")
    assert "from _pages.battery3d import page_battery3d" in source
    assert '"page_battery3d":     page_battery3d' in source
    assert 'page == "battery3d"' in source
    assert 'pages["page_battery3d"](cell_ids, active_fdfs, bundles, selected' in source


# ---------------------------------------------------------------------------
# Page-level AppTest smoke
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY=")
    monkeypatch.setattr(db_module, "_fernet", None)
    db_module.init_db()
    return db_module


def _battery3d_app(data_mode: str) -> AppTest:
    at = AppTest.from_file(MAIN_PY, default_timeout=180)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = "Engineer"
    at.session_state["page"] = "battery3d"
    at.session_state["data_mode"] = data_mode
    return at


def test_the_page_renders_for_a_real_fleet_without_raising(isolated_db):
    at = _battery3d_app("nasa").run()
    assert not at.exception, at.exception
    text = " ".join(block.value for block in at.markdown)
    assert "MEASURED" in text or "SYNTHETIC" in text or "SIMULATED" in text


def test_the_page_offers_the_spec_as_a_download(isolated_db):
    """The document is the artifact a third-party renderer consumes; a host
    that shows the scene but cannot hand out the document is a dead end."""
    at = _battery3d_app("nasa").run()
    assert not at.exception
    labels = [button.label for button in at.get("download_button")]
    assert any("CellSceneSpec" in label for label in labels), labels


def test_the_page_renders_for_a_second_fleet_too(isolated_db):
    """Two data sources, two chemistries (LFP vs LCO), one page: the branches
    that only appear on one of them (a refused forecast, an inapplicable dQ/dV
    model) must not take the page down on the other."""
    at = _battery3d_app("severson").run()
    assert not at.exception, at.exception
