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


def test_overview_hero_severson_cell_not_mislabeled_synthetic(isolated_db):
    """
    Regression test: page_overview()'s hero card built its source tag from
    an is_nasa boolean with no Severson/Oxford/uploaded branch -- every
    non-NASA cell fell into the else branch and was tagged "Synthetic ·
    Stress Nx baseline", even a real measured Severson cell. Live-reproduced
    before the fix: the Overview hero for S-b1c2 (Severson mode) read
    "Synthetic · Stress 1.00x baseline". Source tag is now resolved via
    ChemistryProfile.for_cell(), the same registry used by src/passport.py's
    equivalent fix.
    """
    at = _logged_in_app(
        role="Engineer", page="overview", data_mode="severson",
        selected_cell="S-b1c2",
    )
    at.run()
    assert not at.exception, f"Overview crashed for a Severson cell: {at.exception}"
    text = _all_text(at)
    assert "Synthetic" not in text, (
        "A real measured Severson cell's Overview hero card must never show "
        "a Synthetic source tag"
    )


def test_decide_and_ask_npv_uses_severson_capacity_not_synth(isolated_db, monkeypatch):
    """
    Regression test: page_decision()'s Financial Decision NPV table resolved
    `source` as "nasa" if is_nasa else "synth" -- a 2-way branch with no
    Severson case -- so a real Severson cell's NPV figures were silently
    computed from the synthetic fleet's nominal capacity constant
    (CELL_NOMINAL_KWH["synth"]) instead of Severson's own
    (CELL_NOMINAL_KWH["severson"]). Force the two constants far apart so
    the wrong branch would be unmistakable in the rendered $ figure: the
    "Replace Now" 5-yr NPV is a pure function of nominal capacity (does not
    depend on the cell's actual SOH/RUL), so with severson=500.0 the correct
    value renders as "$159,558" -- a value the buggy "synth" branch (with
    synth forced to 0.001) could never produce.
    """
    import consequences
    monkeypatch.setitem(consequences.CELL_NOMINAL_KWH, "severson", 500.0)
    monkeypatch.setitem(consequences.CELL_NOMINAL_KWH, "synth", 0.001)

    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="severson",
        selected_cell="S-b1c2",
    )
    at.run()
    assert not at.exception, f"Decide & Ask crashed for a Severson cell: {at.exception}"
    text = _all_text(at)
    assert "159,558" in text, (
        "Decide & Ask's NPV table must use CELL_NOMINAL_KWH['severson'] for a "
        "Severson cell, not ['synth'] -- expected the forced-huge severson "
        "capacity to show up as $159,558 somewhere in the rendered NPV figures"
    )


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


@pytest.mark.parametrize("data_mode,cell,expected_n", [
    ("nasa", "B0005", "n=4"),
    ("severson", "S-b1c2", "n=12"),
])
def test_overview_hero_card_shows_lco_sample_size(isolated_db, data_mode, cell, expected_n):
    """
    Regression test (Battery Engineering Accuracy review finding): n=4
    (NASA) / n=12 (Severson) leave-cell-out validation is a thin population
    for any fleet-scale reliability claim -- this used to only be
    discoverable in a settings-page footnote. The confidence badge next to
    every RUL number on Overview must now carry the actual cell count.
    """
    at = _logged_in_app(
        role="Engineer", page="overview", data_mode=data_mode,
        selected_cell=cell,
    )
    at.run()
    assert not at.exception, f"Overview crashed for {cell} in {data_mode} mode: {at.exception}"
    text = _all_text(at)
    assert expected_n in text
    assert "thin population" in text


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


def _synthetic_telemetry(n: int, cell_id: str = "B0005") -> list:
    """Synthetic telemetry readings shaped like mqtt_stream.py's real
    publisher payload -- used to exercise Live Monitor's rendering logic
    without needing a live MQTT broker connection."""
    return [
        {
            "cell_id": cell_id, "cycle": i + 1, "seq": i, "ts": "2026-01-01T00:00:00Z",
            "voltage_v": 3.9, "current_a": -2.0, "temperature_c": 24.0,
            "capacity_ah": 2.0 - i * 0.002,
            "soh_pct": 100.0 - i * 0.3,
        }
        for i in range(n)
    ]


def test_live_monitor_physics_twin_check_waits_for_minimum_readings(isolated_db):
    """
    Regression test (Digital Twin Quality review finding): PyBaMM previously
    only ran once, offline, against a cell's full historical data -- never
    against telemetry as it streams in. With fewer than 5 readings (project_
    rul()'s own minimum for a fade-curve fit), the physics check must show
    an honest "waiting for more data" message, not crash or fabricate a
    result from insufficient data.
    """
    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005", lm_telemetry=_synthetic_telemetry(3), lm_anomalies=[],
    )
    at.run()
    assert not at.exception, f"Live Monitor crashed with 3 readings: {at.exception}"
    captions = [c.value for c in at.caption]
    assert any("waiting for more" in c for c in captions)


def test_live_monitor_physics_twin_check_runs_against_streamed_telemetry(isolated_db):
    """
    With enough streamed readings, the physics-consistency re-check must
    actually run PyBaMM against only the telemetry received so far (not
    the cell's full historical dataset) and render a real RUL estimate,
    chemistry label, and an honest caveat that this still uses a fixed
    parameter set -- not a live-synced digital twin.
    """
    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005", lm_telemetry=_synthetic_telemetry(20), lm_anomalies=[],
    )
    at.run()
    assert not at.exception, f"Live Monitor crashed with 20 readings: {at.exception}"
    metrics = {m.label: m.value for m in at.metric}
    assert "Physics RUL estimate" in metrics
    assert metrics["Physics RUL estimate"] != "—"
    assert "NCA" in metrics.get("Chemistry model", "")
    captions = [c.value for c in at.caption]
    assert any("not a live-synced digital twin" in c for c in captions)


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
