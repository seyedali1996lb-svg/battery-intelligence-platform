"""
AppTest page-level verification for the "Bring your own model" page
(app/_pages/model_validation.py).

The logic behind this page — upload refusal, module validation, report
summarization, fleet probes — is covered by tests/test_harness_models.py, and
the harness itself by tests/test_harness.py. This test's job is UI wiring:
the page renders, it does NOT run an expensive harness for free, the upload
mode refuses to run until a file has been read and acknowledged, and a real
run puts the verdict, the supported claims, and the withheld claims on screen.

The one click-through test uses the cheapest honest configuration available:
8 synthetic cells, the training-mean baseline, leave-cell-out only, intervals
off (~1 s), while the page's default configuration on the same fleet takes
about a minute.
"""

import os
import pathlib
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401
import pytest
import db as db_module
from streamlit.testing.v1 import AppTest

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _logged_in_app(page: str = "model_validation", **session) -> AppTest:
    at = AppTest.from_file(_MAIN_PY, default_timeout=180)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = "admin"
    at.session_state["page"] = page
    at.session_state["data_mode"] = "synthetic"
    for key, value in session.items():
        at.session_state[key] = value
    return at


def _all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _all_text(at: AppTest) -> str:
    """Every text-bearing element, not just markdown.

    The page's verdict is a st.error/st.warning/st.success banner and its
    blocking conditions are st.info, so a markdown-only assertion would miss
    exactly the sentences that matter.
    """
    parts = [
        str(el.value)
        for kind in (at.markdown, at.caption, at.info, at.warning, at.error,
                     at.success, at.text, at.code)
        for el in kind
    ]
    return "\n".join(parts)


def test_page_renders_without_grading_anything(isolated_db):
    at = _logged_in_app()
    at.run()
    assert not at.exception, at.exception

    text = _all_markdown(at)
    assert "Bring Your Own Model" in text
    # The verdict/claims surfaces must NOT exist before a run.
    assert "Supported claims" not in text
    assert "Withheld claims" not in text
    # ... and the page says why nothing has been graded, rather than looking broken.
    assert any("Nothing has been graded yet" in c.value for c in at.caption)
    assert "byom_result" not in at.session_state


def test_page_offers_a_run_button_and_the_reference_fleets(isolated_db):
    at = _logged_in_app()
    at.run()
    assert not at.exception, at.exception
    run_buttons = [b for b in at.button if b.key == "byom_run"]
    assert run_buttons, "the Run button is missing"
    assert run_buttons[0].disabled is False

    # Fleet selection is populated from what this deployment can actually
    # reload, defaults to the real NASA cells (the fleet with measured
    # end-of-life, where RUL is scorable), and excludes anything unavailable.
    fleets = [s for s in at.selectbox if s.key == "byom_fleet"]
    assert fleets, "the fleet selector is missing"
    assert fleets[0].value == "nasa"
    options = " | ".join(str(o) for o in fleets[0].options)
    assert "Synthetic fleet" in options and "NASA PCoE" in options
    # CALCE is not cached on this deployment, so it must not be offered —
    # selecting it would only fail after the click.
    assert "CALCE" not in options

    # The "not reloadable here" reason is stated rather than left to a missing option.
    assert "Not reloadable here" in _all_text(at)


def test_upload_mode_blocks_the_run_until_a_module_is_read_and_acknowledged(isolated_db):
    at = _logged_in_app(byom_model="uploaded_module")
    at.run()
    assert not at.exception, at.exception
    text = _all_text(at)
    assert "no model module uploaded yet" in text
    # The starter template is offered so a user can see the expected shape.
    assert any(b.key == "byom_starter_download" for b in at.download_button)
    assert "byom_result" not in at.session_state


def test_no_floors_declared_means_not_checked_not_a_pass(isolated_db):
    """Turning the gate off must say NOT CHECKED — never imply a pass."""
    at = _logged_in_app(
        byom_fleet="synth",
        byom_model="mean_baseline",
        byom_splits_lco=True,
        byom_splits_prospective=False,
        byom_intervals=False,
        byom_enforce_gate=False,
    )
    at.run()
    assert not at.exception, at.exception
    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception

    text = _all_text(at)
    assert "NOT CHECKED" in text
    assert "Supported claims" in text
    assert "Withheld claims" in text


def test_run_renders_verdict_supported_claims_and_withheld_claims(isolated_db):
    """A real, cheap run: the training-mean anti-model must FAIL a declared floor.

    Uses the page's own default floors with the cheapest configuration that
    still exercises the full render path (leave-cell-out, no intervals). The
    point is not the numbers — it is that a FAIL, the claims it supports, and
    the claims it withholds all reach the screen instead of an exception.
    """
    at = _logged_in_app(
        byom_fleet="synth",
        byom_model="mean_baseline",
        byom_splits_lco=True,
        byom_splits_prospective=False,
        byom_intervals=False,
    )
    at.run()
    assert not at.exception, at.exception
    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception

    text = _all_text(at)
    assert "FAIL" in text
    assert "Supported claims" in text
    assert "Withheld claims" in text
    # The withheld list is populated for this model/fleet: synthetic cells never
    # reach measured end-of-life, so RUL is refused rather than scored.
    assert any("RUL" in str(w.value) for w in at.warning) or "RUL" in text

    result = at.session_state["byom_result"]
    assert result["summary"]["status"] == "fail"
    assert result["summary"]["claims"], "the run produced no claims at all"
    assert result["summary"]["withheld"], "the synthetic fleet should withhold RUL"
    assert result["summary"]["lint_ok"] is True
    # A sealed bundle is offered because leave-cell-out ran. Sealing is a
    # SEPARATE failure mode from the run itself (its error is reported without
    # discarding the report), so it is asserted explicitly rather than assumed.
    assert result["seal_error"] is None, result["seal_error"]
    assert result["zip_bytes"]
    assert any(b.key == "byom_download_zip" for b in at.download_button)


def test_changing_the_configuration_does_not_relabel_a_stale_result(isolated_db):
    """Results carry the fleet/model they were measured on.

    Streamlit re-runs the script on every interaction, so a stored result can
    outlive the widgets that produced it. The page must say so rather than
    showing one fleet's numbers under another fleet's selector.
    """
    at = _logged_in_app(
        byom_fleet="synth",
        byom_model="mean_baseline",
        byom_splits_prospective=False,
        byom_intervals=False,
    )
    at.run()
    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception
    stored_fleet = at.session_state["byom_result"]["summary"]["dataset_name"]
    assert stored_fleet == "synth"

    at.session_state["byom_model"] = "ridge"
    at.run()
    assert not at.exception, at.exception
    assert any(
        "configuration changed since the last run" in c.value for c in at.caption
    )


def test_a_binary_file_renamed_to_py_is_refused_before_it_runs():
    """The extension check is not the whole gate.

    A pickle or a compiled blob renamed to .py passes the suffix check, so the
    page must also refuse to run anything whose bytes are not readable UTF-8
    text — showing it is a precondition of ticking the acknowledgement.
    """
    from _pages.model_validation import _read_upload_source

    class _FakeUpload:
        name = "sneaky.py"

        def getvalue(self):
            return b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00pickle-ish"

    source, error = _read_upload_source(_FakeUpload())
    assert source == ""
    assert error and "not UTF-8 text" in error

    class _TextUpload:
        name = "fine.py"

        def getvalue(self):
            return b"def make_model():\n    return None\n"

    source, error = _read_upload_source(_TextUpload())
    assert error is None and "make_model" in source


def test_nav_wiring(isolated_db):
    """The Analyse nav group gained this page — the button exists and routes."""
    at = _logged_in_app()
    at.run()
    assert not at.exception, at.exception
    nav_buttons = [b for b in at.sidebar.button if b.key == "nav_model_validation"]
    assert nav_buttons, "the sidebar nav entry for the new page is missing"
