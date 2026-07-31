"""
Cold-start regression guard (generalized class, not one instance).

tests/test_db_init_ordering_guard.py already guards the *specific* bug
found on a live Streamlit Cloud deployment (main() reaching
load_everything() -- which writes to experiment_runs/cell_summaries --
without db.init_db() ever having run, because render_login()'s own
init_db() call is skipped whenever session_state["authenticated"] is
already True). That guard is a structural AST check for one function's
statement ordering. It would NOT catch a similarly-shaped bug in a
different function.

Every AppTest-based test elsewhere in this suite uses the isolated_db
fixture, which calls db.init_db() directly during fixture setup, and
reads the real repo's on-disk bundle_cache/cell_store caches (usually
warm from ordinary local development/manual app runs). Both of those are
exactly the two things a genuinely fresh Streamlit Cloud deployment does
NOT have. This module closes that blind spot generally: it runs main.py's
real script body against a DB that has never had init_db() called on it
and cache/store directories that have never been written to, so any
*future* cold-start-only assumption -- not just the one already found --
surfaces as a failing test here instead of a live crash report.

Severson training is deliberately forced off for this test (via
any_cached() monkeypatched to False, the same signal load_everything()
itself checks) -- with cold bundle_cache/features_cache, training all 3
reference pipelines (synth + NASA + Severson's 12 real cells, each an
LCO cross-validation with one GBRT fit per left-out cell) genuinely took
over 12 minutes and still didn't finish inside a 180s per-run AppTest
timeout. That cost is about Severson's real data volume, not about the
cold-start ordering bug this test exists to catch -- NASA (4 cells) +
synthetic (8 cells) already exercise the exact same load_everything() /
_persist_cell_data() / db.init_db() code path Severson does, just
faster. Severson's own training correctness is already covered
elsewhere (test_app_state_combinations.py's parametrized data_mode
tests), just not from a cold DB/cache -- that combination isn't the
concern this module is guarding.

Even with Severson skipped, a genuinely cold NASA (4 cells) + synthetic
(8 cells) run -- 2 full train_models() fits plus a 4-fold and an 8-fold
LCO cross-validation, ~24 GradientBoostingRegressor fits total, all
cache misses -- took longer than the 120s default AppTest script-run
timeout every other AppTest test in this suite uses (safe there only
because those tests hit a warm bundle_cache and never actually retrain).
default_timeout=300 below reflects that real cost, not a fixed bug.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import bundle_cache
import cell_store
import db as db_module
from batlab.datasets import severson as severson_module

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def cold_start_env(tmp_path, monkeypatch):
    """Point every on-disk cache/DB this app touches at fresh, empty
    directories. Unlike isolated_db (test_app_state_combinations.py),
    this deliberately does NOT call db.init_db() -- the whole point is to
    verify main.py's own script body creates the schema itself, before
    it's needed, the way a brand-new deployment actually experiences it.
    """
    test_db_path = tmp_path / "cold_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY=")
    monkeypatch.setattr(db_module, "_fernet", None)

    monkeypatch.setattr(bundle_cache, "CACHE_DIR", tmp_path / "cache_bundles")
    monkeypatch.setattr(bundle_cache, "TENANT_BUNDLE_DIR", tmp_path / "tenant_bundles")
    monkeypatch.setattr(cell_store, "CELL_STORE_DIR", tmp_path / "cell_store")

    # See module docstring: Severson's real-cell training cost is not what
    # this test guards against, so keep load_everything() to synth + NASA.
    monkeypatch.setattr(severson_module, "any_cached", lambda: False)

    # cell_store's in-process LRU and Streamlit's own @st.cache_resource
    # results (load_everything() etc.) are BOTH process-global -- they
    # outlive any one test's monkeypatched paths above. Clear both before
    # (so this test can't silently reuse another test's warm result and
    # skip the cold path entirely) and after (so the next test doesn't
    # inherit a cached result pointing at this test's about-to-be-deleted
    # tmp_path).
    cell_store.clear_lru()
    st.cache_resource.clear()
    yield db_module
    cell_store.clear_lru()
    st.cache_resource.clear()


def _cold_logged_in_app(role: str = "Engineer", page: str = "overview", data_mode: str = "nasa") -> AppTest:
    """Bypasses the login *form* via session_state, same as every other
    AppTest test in this suite -- login.py's own logic isn't what this
    test is checking. This is deliberate, not a shortcut around the bug:
    it means render_login()'s conditional db.init_db() call is skipped
    too, exactly like the real live bug, so the only thing that can make
    the schema exist is main()'s own unconditional init_db() call."""
    at = AppTest.from_file(_MAIN_PY, default_timeout=300)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = role
    at.session_state["page"] = page
    at.session_state["data_mode"] = data_mode
    return at


def _all_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_boots_from_genuinely_cold_state(cold_start_env):
    """Full script body, truly cold DB + truly cold caches: must not
    crash, and must actually render real content (not just an empty/blank
    page) -- proving load_everything()'s cache-miss training path and
    _persist_cell_data()'s DB writes both ran successfully against a
    schema that main() itself had to create on the fly.

    A second page (Fleet, the heaviest reader of the CellSummary rows
    _persist_cell_data() just wrote) is checked in the same test,
    deliberately reusing the now-warm in-process @st.cache_resource
    result from the first at.run() above rather than forcing a second
    full cold retrain -- the thing under test (does the schema exist
    before it's needed) was already proven by the first run; this second
    check only needs the DB rows to be readable, not to retrain again."""
    at = _cold_logged_in_app(page="overview", data_mode="nasa")
    at.run()

    assert not at.exception, f"App crashed on a cold boot: {at.exception}"
    text = _all_text(at)
    assert text.strip(), "Expected real page content to render on a cold boot, got nothing"

    at_fleet = _cold_logged_in_app(role="Fleet Manager", page="fleet", data_mode="nasa")
    at_fleet.run()
    assert not at_fleet.exception, f"Fleet page crashed reading cold-boot data: {at_fleet.exception}"
    fleet_text = _all_text(at_fleet)
    assert "Fleet Health" in fleet_text, "Fleet exec bar didn't render after a cold boot"
