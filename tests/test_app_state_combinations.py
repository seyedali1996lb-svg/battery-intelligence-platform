"""
Integration tests for app/main.py using Streamlit's AppTest harness.

Unlike the rest of tests/ (pure-logic src/ functions, fast), these actually
run the Streamlit script end-to-end and inspect the rendered output. This
is the test class tests/README.md flagged as a follow-up: several real
bugs found during a product review were state-COMBINATION bugs (role X +
data source Y + page Z) that no pure-logic unit test could have caught,
since each individual function was correct in isolation — the bug was in
which function got called with which arguments for a specific combination.

Each test bypasses the login form by setting session_state directly
(authenticated=True + org/role fields) rather than simulating keystrokes —
faster, and login.py's own logic isn't what these tests are checking.
The DB is monkeypatched to a throwaway SQLite file per test (same pattern
as tests/test_db.py) so nothing here touches the real data/app.db.

Slower than the rest of the suite (each test runs the full app script,
~3-10s including model loading) — this is the tradeoff tests/README.md
already flagged for this class of test.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
import db as db_module
from streamlit.testing.v1 import AppTest

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point src/db.py at a throwaway SQLite file for this test only."""
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _logged_in_app(role: str, page: str, data_mode: str, **extra_state) -> AppTest:
    """AppTest instance pre-authenticated as the Demo Org admin, past the
    onboarding overlays, on a specific page/data-source combination."""
    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = role
    at.session_state["page"] = page
    at.session_state["data_mode"] = data_mode
    for k, v in extra_state.items():
        at.session_state[k] = v
    return at


def _all_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.parametrize("data_mode", ["nasa", "severson", "synthetic"])
def test_fleet_page_no_crash_across_data_sources(isolated_db, data_mode):
    """
    Regression test for the exact bug found in review: the Fleet page's
    executive KPI bar and its ranking table used to be two independent
    computations that could disagree — in Severson mode specifically, the
    KPI bar showed real numbers while the ranking table said "No cells
    loaded" underneath it, because _bundle_for_cell() had no Severson
    branch. Now that both read from one shared `rows` list, this should
    be structurally impossible for any data source.
    """
    at = _logged_in_app(role="Fleet Manager", page="fleet", data_mode=data_mode)
    at.run()
    assert not at.exception, f"Fleet page raised an exception in {data_mode} mode: {at.exception}"
    text = _all_text(at)
    assert "Fleet Health" in text, f"Fleet exec bar didn't render in {data_mode} mode"
    assert "No cells loaded" not in text, (
        f"Fleet page showed the empty state in {data_mode} mode even though "
        f"the exec bar above it has data — the exact contradiction bug from review"
    )


def test_nasa_cells_never_labeled_nca_chemistry(isolated_db):
    """
    Regression test: NASA cells were mislabeled "NCA chemistry" in 5 places
    across the app (mode switcher, chemistry-label dicts, Live Monitor,
    Settings), contradicting LiCoO2 everywhere else — including this app's
    own chemistry_profiles.py. This directly undercut the app's core
    chemistry-honesty positioning.
    """
    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "NCA chemistry" not in text, "NASA cells must never be labeled NCA — they are LiCoO2"


def test_decide_and_ask_no_crash_for_degraded_cell(isolated_db):
    """
    Regression test for the TypeError crash found in review: Application
    Fit Scores on the Decide & Ask page treated application_fit()'s
    categorical {"fit": "marginal", ...} dicts as if they were numeric
    scores (`_score >= 70`), crashing on every visit for any cell under
    90% SOH. B0006 (NASA) is ~58% SOH — squarely in the crashing range
    before the fix.
    """
    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="nasa",
        selected_cell="B0006",
    )
    at.run()
    assert not at.exception, f"Decide & Ask crashed for a degraded cell: {at.exception}"


def test_decide_and_ask_no_crash_for_severson_cell(isolated_db):
    """
    Regression test for two masked bugs found while fixing the above: (1)
    breakeven_curve()'s np.arange(soh_current, soh_min-0.1, -0.5) produced
    an empty array whenever a cell's SOH was already below the hardcoded
    soh_min=62.0 floor; (2) page_consequences()'s NPV Scenario Planner used
    CELL_NOMINAL_KWH (a dict keyed by source) directly in arithmetic
    instead of indexing it, and `source` itself was only ever "nasa" or
    "synth" — never "severson" — so every Severson cell got the wrong
    capacity value even where the dict *was* indexed correctly elsewhere.
    Both bugs were invisible until the earlier Application Fit Scores
    crash (fixed above) stopped masking the code paths after it.
    """
    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="severson",
        selected_cell="S-b1c2",
    )
    at.run()
    assert not at.exception, f"Decide & Ask crashed for a Severson cell: {at.exception}"


def test_explore_compare_tab_survives_data_source_switch(isolated_db):
    """
    Regression test: Explore's Compare tab selectboxes (compare_cell_a/b)
    kept a stale cell ID in session_state across a data-source switch,
    which Streamlit's selectbox widget can't reconcile against a new
    options list that no longer contains it.
    """
    at = _logged_in_app(
        role="Engineer", page="compare", data_mode="severson",
        compare_cell_a="B0005", compare_cell_b="B0006",  # stale NASA IDs, but mode is severson
    )
    at.run()
    assert not at.exception, f"Explore Compare tab crashed on a stale cell selection: {at.exception}"


# ---------------------------------------------------------------------------
# Cell Workbench (Health + Decide & Ask merge — UX/Workflow review finding)
# ---------------------------------------------------------------------------

def test_workbench_health_page_key_defaults_to_mechanism_view(isolated_db):
    """
    Health and Decide & Ask used to be two separate page navigations for
    the same cell -- now merged into one workbench (app/_pages/workbench.py)
    with a radio between "Mechanism (Health)" and "Decision (Decide & Ask)".
    Arriving via the "health" page-key (sidebar nav, Fleet's per-cell
    quick-jump) must default to the Mechanism view.
    """
    at = _logged_in_app(role="Engineer", page="health", data_mode="nasa")
    at.run()
    assert not at.exception, f"Workbench crashed defaulting to Mechanism view: {at.exception}"
    radios = [r for r in at.radio if r.key == "workbench_view_radio"]
    assert radios, "Workbench view radio not found"
    assert radios[0].value == "Mechanism (Health)"
    text = _all_text(at)
    assert "What is degrading" in text  # page_health()'s own header


def test_workbench_decision_page_key_defaults_to_decision_view(isolated_db):
    """Arriving via the "decision" page-key (nav, trajectory-match card's
    "Go to Decide & Ask", the mechanism verdict button) must default to the
    Decision view, not silently reopen Mechanism."""
    at = _logged_in_app(role="Engineer", page="decision", data_mode="nasa")
    at.run()
    assert not at.exception, f"Workbench crashed defaulting to Decision view: {at.exception}"
    radios = [r for r in at.radio if r.key == "workbench_view_radio"]
    assert radios, "Workbench view radio not found"
    assert radios[0].value == "Decision (Decide & Ask)"
    text = _all_text(at)
    assert "What should I do" in text  # page_decision()'s own header


def test_workbench_manual_view_switch_has_no_crash(isolated_db):
    """Switching the in-page radio (browsing between the two lenses on the
    same cell without a fresh navigation) must render the other view with
    no exception and no duplicate-widget-key collision between
    page_health() and page_decision()."""
    at = _logged_in_app(role="Engineer", page="health", data_mode="nasa")
    at.run()
    radio = [r for r in at.radio if r.key == "workbench_view_radio"][0]
    radio.set_value("Decision (Decide & Ask)").run()
    assert not at.exception, f"Manual view switch crashed: {at.exception}"
    assert "What should I do" in _all_text(at)


def test_decide_and_ask_shows_mechanism_caution_note_when_signals_disagree(isolated_db):
    """
    Regression test (Decision Support review finding): classify() picks the
    recommended action from SOH/fade/RUL/fit-scores alone -- it never sees
    the mechanism classifier's LLI/LAM verdict, so a cell recommended a
    lower-urgency action ("continue"/"inspect") could still show an
    accelerating (LAM) degradation pattern with zero cross-referencing
    between the two surfaces. S-b1c2 (Severson) is a real cell where this
    disagreement actually occurs: LAM mechanism at Medium confidence,
    "continue" action -- the caution note must render on the hero card.
    """
    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="severson",
        selected_cell="S-b1c2",
    )
    at.run()
    assert not at.exception, f"Decide & Ask crashed: {at.exception}"
    text = _all_text(at)
    assert "elevated caution" in text
    assert "LAM" in text


def test_health_mechanism_verdict_visible_to_non_engineer_by_default(isolated_db):
    """
    Regression test (Engineering Usability review finding): the mechanism
    verdict card used to only render once the "Engineering details"
    checkbox was ticked, which defaults on only for the Engineer role --
    Fleet Manager/Executive/Compliance Officer never saw *why* a decision
    was made unless they knew to look for a checkbox. The compact verdict
    (icon/label/confidence) must now render for every role regardless of
    that checkbox; only the deep signal-by-signal breakdown stays gated.
    """
    at = _logged_in_app(
        role="Fleet Manager", page="health", data_mode="nasa",
        selected_cell="B0006",
    )
    at.run()
    assert not at.exception, f"Health page crashed for Fleet Manager: {at.exception}"
    text = _all_text(at)
    assert "Degradation Mechanism" in text
    assert "confidence</span>" in text
    checkboxes = [c for c in at.checkbox if "Engineering details" in (c.label or "")]
    assert checkboxes and checkboxes[0].value is False, (
        "Engineering details should still default off for non-Engineer roles"
    )
