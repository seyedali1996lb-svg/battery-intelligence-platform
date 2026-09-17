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


@pytest.fixture(autouse=True)
def isolated_upload_store(tmp_path, monkeypatch):
    """Point the raw-cycle store at a temp dir for EVERY test in this file.

    The page's fleet picker lists the org's persisted uploads, so without this
    a developer who has analysed an upload in the app would change what these
    tests see (and their real store would be read by the suite).
    """
    import uploaded_store as us

    monkeypatch.setattr(us, "UPLOADED_STORE_DIR", tmp_path / "uploaded_fleets")
    return us


def _tenant_fleet(n_cells: int = 3) -> dict:
    from conftest import make_cycles_df

    return {
        f"UP{i + 1}": make_cycles_df(
            n_cycles=120,
            fade_per_cycle=0.004 * (1.0 + 0.1 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


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


def test_upload_mode_states_what_the_sandbox_enforces(isolated_db):
    """The disclosure is on screen BEFORE a file is chosen.

    It is a property of this deployment, not of the module about to be uploaded:
    someone deciding whether to upload at all needs it first. The two sentences
    that must not disappear are the one about what is NOT enforced (the half a
    reader is most likely to assume away) and the measured cost of serialized
    folding, which is a real change from the in-process path.
    """
    at = _logged_in_app(byom_model="uploaded_module")
    at.run()
    assert not at.exception, at.exception

    text = _all_text(at)
    assert "The sandbox this deployment runs your module in" in text
    assert "What is NOT enforced" in text
    assert "not a container" in text  # the honest half, verbatim
    assert "The sandbox answers one fit or predict at a time" in text
    assert "≈4 s sandboxed versus ≈2.5 s in-process" in text

    # The table is the detail behind those sentences, and every row the sandbox
    # can report is rendered — a row that vanished would silently shrink the
    # disclosure, which is the failure mode this table exists to avoid.
    table = at.dataframe[0].value
    controls = list(table["Control"])
    assert controls == [
        "Where the model runs", "Wall clock", "Memory", "CPU",
        "Imports it is refused", "Operations it is refused", "File writes",
        "**Not enforced**",
    ]
    rendered = " | ".join(table["What happens"])
    assert "audit hook" in rendered and "scratch directory" in rendered
    assert "separate child process" in rendered
    assert "OS-level boundary" in rendered

    # And the same page must NOT claim a sandbox where the model is the
    # platform's own: the disclosure belongs to the upload path.
    other = _logged_in_app(byom_model="ridge")
    other.run()
    assert not other.exception, other.exception
    assert "The sandbox this deployment runs your module in" not in _all_text(other)


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


def test_an_uploaded_module_is_graded_in_a_sandbox_end_to_end(isolated_db):
    """The upload path, driven for real: page logic -> sandbox -> harness -> seal.

    AppTest cannot put a file into Streamlit's uploader, so this calls the page's
    own `_run` with an upload source — the same function the Run button calls —
    and checks the things the page has to get right: the module is imported in
    another process, the result is JSON-safe (no live sandbox left in the
    session), the child is stopped when the run ends, and the sealed bundle
    still carries the model's source so a reviewer can recompute it.
    """
    import json as _json
    import pathlib

    from _pages.model_validation import _run

    source = (
        "from sklearn.linear_model import Ridge\n"
        "from batlab.harness import SklearnForecaster\n"
        "\n"
        "print('module imported in the sandbox')\n"
        "\n"
        "def make_model():\n"
        "    return SklearnForecaster(Ridge(alpha=1.0), scale=True)\n"
    )

    fleet = next(f for f in __import__("harness_models").available_fleets() if f["key"] == "synth")
    cfg = {
        "fleet_key": "synth",
        "fleet": fleet,
        "org_id": 1,
        "seed": 42,
        "model_key": "uploaded_module",
        "upload_source": source,
        "upload_name": "my_model.py",
        "splits_lco": True,
        "splits_prospective": False,
        "intervals": False,
        "enforce_gate": False,
        "gate_text": "{}",
        "embed_data": False,
        "include_model_source": True,
    }
    result = _run(cfg)

    probe = result["uploaded_module"]
    assert probe["sandboxed"] is True
    assert probe["probe_class"] == "SklearnForecaster"
    assert "separate child process" in probe["sandbox"]["process"]
    # JSON-safe: a live sandbox in the result would be kept alive by the session
    # state that stores it, one process per re-run.
    assert "factory" not in probe and "close" not in probe
    assert _json.loads(_json.dumps(probe))["sandbox"]["not_enforced"]
    assert "module imported in the sandbox" in probe["output"]

    # What was graded is what the child held, and the report says so.
    assert result["report"]["model"]["identity"]["sandboxed"] is True
    assert result["report"]["model"]["identity"]["class"] == "SklearnForecaster"

    # The child is gone by the time the result exists (its scratch directory is
    # removed with it; test_sandbox.py asserts the process tree itself).
    assert not pathlib.Path(probe["path"]).parent.exists()

    # ... and the sealed bundle still carries the model, so the printed command
    # can re-derive the number somewhere else.
    import io
    import zipfile

    assert result["seal_error"] is None, result["seal_error"]
    with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
        assert "model/module.py" in archive.namelist()
        assert archive.read("model/module.py").decode("utf-8") == source


def _render_sandbox_only(result: dict) -> None:
    """Module-level so AppTest.from_function can read the function's source."""
    from _pages.model_validation import _render_sandbox

    _render_sandbox(result)


def test_the_post_run_sandbox_section_renders_what_was_enforced():
    """The section that shows, AFTER the run, where the model actually ran.

    Driven directly rather than through a full page run: Streamlit's file
    uploader cannot be filled by AppTest, so the only other way to reach this
    code path would be to fake the whole configuration key. Two cases matter —
    a sandboxed result renders the enforcement table and the captured output,
    and a run that graded no upload renders nothing at all (the section belongs
    to the upload path, not to every result).
    """
    from batlab.harness.sandbox import describe_enforcement

    result = {
        "uploaded_module": {
            "path": "C:/tmp/batlab_sandbox_abc/uploaded_model.py",
            "probe_class": "SklearnForecaster",
            "sandbox": describe_enforcement(),
            "output": "[stdout] module imported in the sandbox",
        }
    }
    at = AppTest.from_function(_render_sandbox_only, kwargs={"result": result}).run()
    assert not at.exception, at.exception
    assert [m.value for m in at.markdown] == ["#### The sandbox this model ran in"]
    caption = " ".join(c.value for c in at.caption)
    assert "scratch file" in caption and "never in this app's process" in caption
    assert "removed when the run ended" in caption
    assert "SklearnForecaster" in caption
    assert list(at.dataframe[0].value["Control"]) == [
        "Where the model runs", "Wall clock", "Memory", "CPU",
        "Imports it is refused", "Operations it is refused", "File writes",
        "**Not enforced**",
    ]
    assert [e.label for e in at.expander] == [
        "What the module itself printed (captured in the sandbox)"
    ]

    for empty in ({"uploaded_module": None}, {}, {"uploaded_module": {}}):
        quiet = AppTest.from_function(_render_sandbox_only, kwargs={"result": empty}).run()
        assert not quiet.exception, quiet.exception
        assert not quiet.markdown and not quiet.dataframe


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


def test_the_orgs_own_upload_is_offered_first_and_can_be_graded(isolated_db, isolated_upload_store):
    """The end state of persisting raw cycles: your own fleet, on this page.

    Cheapest honest configuration (3 cells, training-mean floor, leave-cell-out
    only, no intervals) — the point is that a tenant's OWN cells reach the
    verdict, the claims, and a sealed bundle that carries them.
    """
    import io
    import zipfile

    key = "upload-0f0f0f0f0f0f0f0f0f0f"
    isolated_upload_store.save_uploaded_cell_data(
        1, key, _tenant_fleet(3), meta={"n_cells": 3}
    )

    at = _logged_in_app(
        byom_model="mean_baseline",
        byom_splits_lco=True,
        byom_splits_prospective=False,
        byom_intervals=False,
    )
    at.run()
    assert not at.exception, at.exception

    fleet_box = [s for s in at.selectbox if s.key == "byom_fleet"][0]
    # Offering it is the feature; offering it FIRST is the courtesy — if you
    # have just uploaded your own cells, that is the fleet you came for.
    assert fleet_box.value == key
    assert "Your uploaded fleet" in str(fleet_box.options[0])
    assert "3 cells" in str(fleet_box.options[0])
    assert "cells — 3 cells" not in str(fleet_box.options[0])  # count stated once

    # The privacy trade is stated BEFORE the run, not after the download.
    assert any(c.key == "byom_embed_data" for c in at.checkbox)
    assert "sharing the bundle shares the data" in _all_text(at).lower()

    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception

    result = at.session_state["byom_result"]
    assert result["summary"]["dataset_name"].startswith("uploaded-")
    assert result["summary"]["n_cells"] == 3
    assert result["embed_data"] is True
    assert result["seal_error"] is None, result["seal_error"]

    with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
        names = archive.namelist()
    assert any(n.startswith("cells/") and n.endswith(".csv") for n in names)
    # ... and the panel says which loader verifies it, matching what was sealed.
    assert result["loader"]["loader"] == "batlab.validation.bundle_data:load_bundle_cells"
    assert "load_bundle_cells" in _all_text(at)

    # The model travelled too: this run was graded with the training-mean
    # baseline, whose code used to be unobtainable — so the sealed bundle now
    # carries it and the printed command is the complete, runnable one.
    assert result["default_model"] is False
    assert result["model_source_embedded"] is True
    assert any(n == "model/module.py" for n in names)
    command = "\n".join(c.value for c in at.code)
    assert "load_bundle_cells" in command
    assert "--model batlab.harness.model_source:load_bundle_model" in command
    assert "--recompute" in command
    assert "Re-deriving the number needs the model" not in _all_text(at)

    # The whole promise, end to end: extract the zip into an empty directory
    # and verify with NOTHING else — the data comes from `cells/`, the model
    # from `model/module.py`, and the store this deployment keeps is never
    # consulted. This is what a recipient of the bundle can do.
    import importlib
    import tempfile

    from batlab.validation.replication import (
        _load_cell_data, _load_model, _recorded_loader_kwargs, _recorded_model_kwargs,
        load_bundle, verify_bundle,
    )

    extracted = tempfile.mkdtemp(prefix="byom_zip_only_")
    with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
        archive.extractall(extracted)
    loaded = load_bundle(extracted)

    # Delete the tenant's cycles before verifying, so a bundle that secretly
    # leaned on this deployment fails instead of passing by accident.
    isolated_upload_store.clear_uploaded_cell_data(key)

    data_loader = loaded["loader"]
    cell_data = _load_cell_data(f"{data_loader['module']}:{data_loader['function']}",
                                _recorded_loader_kwargs(loaded), bundle_dir=extracted)
    model_loader = loaded["model_source"]["loader"]
    factory = _load_model(f"{model_loader['module']}:{model_loader['function']}",
                          bundle_dir=extracted,
                          recorded_kwargs=_recorded_model_kwargs(loaded))
    verification = verify_bundle(loaded, bundle_dir=extracted, cell_data=cell_data,
                                recompute=True, forecaster=factory)
    assert verification["verdict"] == "pass", verification["checks"]
    # The carried model is the one that was graded, not merely a source file:
    # run_lco fits it per fold, so the recomputed number has to match exactly.
    assert any(c["name"] == "recompute" and c["status"] == "pass"
               for c in verification["checks"]), verification["checks"]
    assert importlib.import_module("batlab.validation.bundle_data") is not None


def test_keeping_the_raw_cycles_home_seals_a_bundle_without_them(isolated_db, isolated_upload_store):
    """The other side of the trade, and it must be the other side.

    Unchecking the box has to change BOTH what is sealed and what the page tells
    you to run — a page that printed one command and sealed the other would be
    worse than no option at all.
    """
    import io
    import zipfile

    key = "upload-1e1e1e1e1e1e1e1e1e1e"
    isolated_upload_store.save_uploaded_cell_data(1, key, _tenant_fleet(3))

    at = _logged_in_app(
        byom_fleet=key,
        byom_model="mean_baseline",
        byom_splits_lco=True,
        byom_splits_prospective=False,
        byom_intervals=False,
        byom_embed_data=False,
    )
    at.run()
    assert not at.exception, at.exception
    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception

    result = at.session_state["byom_result"]
    assert result["embed_data"] is False
    assert result["loader"]["loader"] == "src.uploaded_store:load_uploaded_cell_data"
    assert result["loader"]["loader_kwargs"] == {"upload_key": key}
    assert result["seal_error"] is None, result["seal_error"]

    with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
        names = archive.namelist()
    assert not any(n.startswith("cells/") for n in names)
    assert "does not carry your data" in _all_text(at)


def test_the_verify_command_recomputes_when_it_can(isolated_db):
    """The platform's own model IS reproducible from the repo — so that command
    carries --recompute, and the bundle with the tenant's cycles verifies
    end to end from the printed lines alone."""
    import io
    import zipfile

    at = _logged_in_app(
        byom_fleet="synth",
        byom_model="platform_default",
        byom_splits_lco=True,
        byom_splits_prospective=False,
        byom_intervals=False,
        byom_enforce_gate=False,
    )
    at.run()
    assert not at.exception, at.exception
    at.button(key="byom_run").click().run()
    assert not at.exception, at.exception

    result = at.session_state["byom_result"]
    assert result["default_model"] is True
    command = "\n".join(c.value for c in at.code)
    assert "--recompute" in command
    assert "Re-deriving the number needs the model" not in _all_text(at)

    # Run the printed loader against the sealed bundle, in-process, exactly as
    # the CLI would: it is the same call the verifier makes.
    import importlib
    import tempfile

    from batlab.validation.replication import _load_cell_data, _recorded_loader_kwargs

    bundle = result["report"]
    with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
        extracted = tempfile.mkdtemp(prefix="byom_verify_")
        archive.extractall(extracted)
    from batlab.validation.replication import load_bundle, verify_bundle

    loaded = load_bundle(extracted)
    loader = getattr(importlib.import_module(loaded["loader"]["module"]),
                     loaded["loader"]["function"])
    cell_data = _load_cell_data(f"{loaded['loader']['module']}:{loaded['loader']['function']}",
                                _recorded_loader_kwargs(loaded), bundle_dir=extracted)
    verification = verify_bundle(loaded, bundle_dir=extracted, cell_data=cell_data,
                                recompute=True)
    assert verification["verdict"] == "pass", verification["checks"]
    assert loader is not None and bundle["lco"] is not None


def test_nav_wiring(isolated_db):
    """The Analyse nav group gained this page — the button exists and routes."""
    at = _logged_in_app()
    at.run()
    assert not at.exception, at.exception
    nav_buttons = [b for b in at.sidebar.button if b.key == "nav_model_validation"]
    assert nav_buttons, "the sidebar nav entry for the new page is missing"
