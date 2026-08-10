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
    """Point src/db.py at a throwaway SQLite file for this test only, with
    a real (test-only) SETTINGS_ENCRYPTION_KEY set so tests that write a
    fresh secret-setting value (e.g. the webhook regression test below)
    don't hit InsecureCredentialStorageError, which is meant to guard the
    unset-key case, not ordinary test fixtures. _fernet is a process-global
    cache in db.py -- reset it too, or a previous test's key could leak in."""
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
    at.session_state["mode_chosen"] = True
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


def test_explore_reference_datasets_condition_completeness_renders(isolated_db):
    """
    Smoke test for the new test-condition-documentation disclosure
    (batlab.datasets.schema.condition_completeness(), rendered via
    app/_pages/explore.py's _render_condition_completeness()) on the
    Explore > Reference Datasets tab (Oxford NCA cells).
    """
    at = _logged_in_app(role="Engineer", page="compare", data_mode="nasa", explore_view_radio="Reference Datasets")
    at.run()
    assert not at.exception, f"Reference Datasets tab crashed: {at.exception}"
    labels = [e.label for e in at.expander]
    assert any("Test-condition documentation" in label for label in labels)


@pytest.mark.parametrize("data_mode", ["nasa", "severson"])
def test_decide_and_ask_spine_export_button_renders(isolated_db, data_mode):
    """
    Smoke test for the new Spine-compatible second-life export
    (src/spine_export.py): the "Export for grid modeling" download button
    must render on the Decide & Ask page without an exception across data
    sources, including a degraded NASA cell (the same crash class as
    test_decide_and_ask_no_crash_for_degraded_cell above).
    """
    at = _logged_in_app(role="Engineer", page="decision", data_mode=data_mode)
    at.run()
    assert not at.exception, f"Decide & Ask crashed building the Spine export in {data_mode} mode: {at.exception}"
    labels = [b.label for b in at.button] + [d.label for d in at.download_button]
    assert "Export for grid modeling" in labels


@pytest.mark.parametrize("data_mode", ["nasa", "severson"])
def test_decide_and_ask_optimade_export_button_renders(isolated_db, data_mode):
    """
    Smoke test for the OPTIMADE-shaped export (src/optimade_export.py),
    the second static export target added alongside the Spine one — same
    coverage shape as test_decide_and_ask_spine_export_button_renders above.
    """
    at = _logged_in_app(role="Engineer", page="decision", data_mode=data_mode)
    at.run()
    assert not at.exception, f"Decide & Ask crashed building the OPTIMADE export in {data_mode} mode: {at.exception}"
    labels = [b.label for b in at.button] + [d.label for d in at.download_button]
    assert "Export for materials database" in labels
    assert "Export for grid modeling" in labels


def test_decision_cmms_ticket_button_reports_error_for_unreachable_endpoint(isolated_db):
    """
    Click-through regression test for the src/adapter_contract.py refactor
    of decision.py's "Create CMMS ticket" button: with a CMMS API key
    configured but pointed at an unreachable endpoint, the button must
    still classify the result as "error" (via classify_result(), not the
    old hand-rolled `if _cmms_result is None: ... elif "error" in
    _cmms_result: ...` branch) and show the same user-facing message as
    before the refactor.
    """
    # Reserved .invalid TLD (RFC 2606, guaranteed never to resolve) — same
    # convention tests/test_cmms_adapter.py's own network-failure test uses.
    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="nasa", selected_cell="B0006",
        cmms_api_key="fake-test-key",
        cmms_api_base_url="https://this-host-does-not-exist.invalid/v1",
    )
    at.run()
    assert not at.exception
    at.button(key="dec_cmms_btn").click().run()
    assert not at.exception, f"CMMS ticket button crashed: {at.exception}"
    # st.error()/st.success() render as their own element types, not
    # markdown -- _all_text() (which only joins at.markdown) won't see them.
    errors = [e.value for e in at.error]
    assert any("CMMS ticket creation failed" in e for e in errors), errors


def test_settings_cmms_test_connection_reports_error_for_unreachable_endpoint(isolated_db):
    """
    Same refactor, same coverage for the other of the 4 call sites that
    used to duplicate the None/"error"/success branch by hand — Settings'
    "Test CMMS connection" button.
    """
    isolated_db.set_setting(1, "cmms_api_base_url", "https://this-host-does-not-exist.invalid/v1", role="admin")
    at = _logged_in_app(
        role="Fleet Manager", page="settings", data_mode="synthetic",
        cmms_api_key="fake-test-key", cmms_api_base_url="https://this-host-does-not-exist.invalid/v1",
    )
    at.run()
    assert not at.exception
    at.button(key="cmms_test_btn").click().run()
    assert not at.exception, f"CMMS test-connection button crashed: {at.exception}"
    errors = [e.value for e in at.error]
    assert any("CMMS connection failed" in e for e in errors), errors


def test_settings_circunomics_test_connection_reports_error_for_unreachable_endpoint(isolated_db, monkeypatch):
    """
    Same refactor, same coverage for Settings' "Test Circunomics
    connection" button. Unlike the CMMS call sites, circunomics_adapter.py's
    api_base_url isn't exposed as a Settings field (always the module's
    real-domain default) -- monkeypatch requests.post itself instead of
    relying on real network resolution, so this stays fully offline.
    """
    import requests

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(requests, "post", _raise)

    at = _logged_in_app(
        role="Fleet Manager", page="settings", data_mode="synthetic",
        circunomics_api_key="fake-test-key",
    )
    at.run()
    assert not at.exception
    at.button(key="circ_test_btn").click().run()
    assert not at.exception, f"Circunomics test-connection button crashed: {at.exception}"
    errors = [e.value for e in at.error]
    assert any("Circunomics connection failed" in e for e in errors), errors


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


def test_live_monitor_ingestion_faults_panel_renders_with_no_faults(isolated_db):
    """The new Ingestion Faults expander must render an honest empty state
    when nothing has tripped a fault check yet, not crash or show stale
    placeholder content."""
    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005", lm_telemetry=_synthetic_telemetry(5),
        lm_anomalies=[], lm_faults=[],
    )
    at.run()
    assert not at.exception, f"Live Monitor crashed: {at.exception}"
    assert any("Ingestion Faults" in (e.label or "") for e in at.expander)
    assert "No ingestion faults detected" in _all_text(at)


def test_live_monitor_ingestion_faults_panel_renders_seeded_faults(isolated_db):
    """With pre-seeded fault entries (as mqtt_stream.drain_faults() would
    have produced), the panel must show them -- kind, cell_id, detail --
    and offer a CSV export, mirroring the existing Anomaly Log."""
    seeded_faults = [
        {
            "cell_id": "B0005", "kind": "IMPLAUSIBLE_CAPACITY",
            "detail": "capacity_ah=1823.0 is outside any physically plausible range",
            "severity": "warning", "ts": "2026-01-01T00:00:05Z", "seq": 5,
        },
    ]
    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005", lm_telemetry=_synthetic_telemetry(5),
        lm_anomalies=[], lm_faults=seeded_faults,
    )
    at.run()
    assert not at.exception, f"Live Monitor crashed: {at.exception}"
    assert "IMPLAUSIBLE_CAPACITY" in _all_text(at)
    assert any("Export ingestion fault log CSV" in (b.label or "") for b in at.download_button)


def test_live_monitor_anomaly_log_groups_consecutive_same_kind_readings(isolated_db):
    """A sustained condition (the same anomaly kind tripping on many
    consecutive readings) must render as one episode card with a reading
    count and status, not one near-identical card per reading -- the exact
    duplication a design review flagged from a live screenshot (two
    identical OVERVOLTAGE cards seconds apart). A different kind, and the
    same kind recurring non-consecutively, must still form separate cards."""
    seeded_anomalies = [
        {"cell_id": "B0005", "kind": "OVERVOLTAGE", "detail": "4.182 V above max 3.65 V",
         "severity": "critical", "ts": "2026-01-01T00:00:00Z", "seq": 0},
        {"cell_id": "B0005", "kind": "OVERVOLTAGE", "detail": "4.185 V above max 3.65 V",
         "severity": "critical", "ts": "2026-01-01T00:00:01Z", "seq": 1},
        {"cell_id": "B0005", "kind": "OVERVOLTAGE", "detail": "4.190 V above max 3.65 V",
         "severity": "critical", "ts": "2026-01-01T00:00:02Z", "seq": 2},
        {"cell_id": "B0005", "kind": "TEMP_RATE_HIGH", "detail": "Temperature rose 2.1°C in one reading",
         "severity": "warning", "ts": "2026-01-01T00:00:03Z", "seq": 3},
        {"cell_id": "B0005", "kind": "OVERVOLTAGE", "detail": "4.170 V above max 3.65 V",
         "severity": "critical", "ts": "2026-01-01T00:00:04Z", "seq": 4},
        {"cell_id": "B0005", "kind": "OVERVOLTAGE", "detail": "4.172 V above max 3.65 V",
         "severity": "critical", "ts": "2026-01-01T00:00:05Z", "seq": 5},
    ]
    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005", lm_telemetry=_synthetic_telemetry(5),
        lm_anomalies=seeded_anomalies, lm_faults=[],
    )
    at.run()
    assert not at.exception, f"Live Monitor crashed: {at.exception}"
    text = _all_text(at)
    # Two separate OVERVOLTAGE episodes (3 consecutive, then 2 consecutive
    # after the TEMP_RATE_HIGH break) -- five raw readings must collapse
    # into two cards, not five.
    assert text.count("OVERVOLTAGE") == 2
    assert "TEMP_RATE_HIGH" in text
    assert "3 readings" in text
    assert "2 readings" in text
    # No live subscriber is connected in this test, so every episode --
    # including the most recent -- must read as resolved, not active.
    assert "ACTIVE" not in text
    assert text.count("RESOLVED") == 3


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


def test_settings_webhook_widgets_do_not_trigger_session_state_duplication_warning(isolated_db, monkeypatch):
    """
    Regression test: settings.py's webhook_url/webhook_secret/webhook_events
    widgets used to pass both key= (pre-seeded by main.py's hydration block,
    which reads persisted values from the DB into session_state before the
    Settings page ever renders) AND a value=/default= kwarg re-reading the
    same session_state key -- the exact anti-pattern Streamlit warns about
    via streamlit.elements.lib.policies.check_session_state_rules(). Only
    reproduces once the DB actually has persisted values for these keys
    (hydration is a no-op otherwise), matching how it originally surfaced.

    Streamlit only logs this warning once per process
    (_shown_default_value_warning is a module-level flag), so an earlier
    test tripping it would hide a regression here -- reset it first.
    """
    import streamlit.elements.lib.policies as _policies
    monkeypatch.setattr(_policies, "_shown_default_value_warning", False)
    calls = []
    monkeypatch.setattr(_policies._LOGGER, "warning", lambda *a, **k: calls.append(a))

    isolated_db.set_setting(1, "webhook_url", "https://example.com/hook", role="admin")
    isolated_db.set_setting(1, "webhook_secret", "s3cr3t", role="admin")
    isolated_db.set_setting(1, "webhook_events", ["CAPACITY_PLUNGE"], role="admin")

    at = _logged_in_app(role="Fleet Manager", page="settings", data_mode="synthetic")
    at.run()

    assert not at.exception, f"Settings page crashed: {at.exception}"
    duplication_warnings = [c for c in calls if "default value" in str(c[0])]
    assert duplication_warnings == [], f"session-state duplication warning fired: {duplication_warnings}"


def test_settings_sites_and_fleets_panel_renders_for_admin(isolated_db):
    """The new FleetAsset "Sites & Fleets" panel (admin-only) must render
    without crashing, showing the Default Site/Fleet every org is seeded
    with by db.init_db()."""
    at = _logged_in_app(role="Fleet Manager", page="settings", data_mode="synthetic")
    at.run()

    assert not at.exception, f"Settings page crashed: {at.exception}"
    assert "Sites & Fleets" in _all_text(at)
    assert any("Default Site" in list(w.options) for w in at.selectbox if w.key == "sf_site_select")


def test_settings_sites_and_fleets_panel_hidden_for_non_admin(isolated_db):
    """Same admin-only gate as the existing Team Members section."""
    at = _logged_in_app(role="Fleet Manager", page="settings", data_mode="synthetic",
                         auth_role="engineer")
    at.run()

    assert not at.exception, f"Settings page crashed: {at.exception}"
    assert "Sites & Fleets" not in _all_text(at)


def test_live_monitor_fragment_does_not_crash_with_an_active_connection(isolated_db, monkeypatch):
    """
    Regression test for a production crash on Streamlit Cloud: Live
    Monitor's telemetry fragment used to call
    time.sleep(...); st.rerun(scope="fragment") manually inside its two
    polling sites. st.rerun(scope="fragment") raises StreamlitAPIException
    the very first time the fragment executes as part of a full script run
    -- scope="fragment" is only valid from inside an already-in-progress
    fragment rerun, which the first execution never is. This never
    reproduced locally: every other Live Monitor test sets lm_telemetry
    directly in session_state without ever starting a real subscriber, so
    is_subscriber_connected() was always False and the guarded rerun call
    never actually ran. It broke the first time a real deployment had an
    active connection. Fixed by switching to @st.fragment(run_every=...),
    which needs no manual rerun call -- this test simulates the exact
    connected state that crashed in production to catch a regression.
    """
    import mqtt_stream
    monkeypatch.setattr(mqtt_stream, "is_subscriber_connected", lambda: True)
    monkeypatch.setattr(mqtt_stream, "publisher_running", lambda: True)
    monkeypatch.setattr(mqtt_stream, "drain_telemetry", lambda max_msgs=200: [])
    monkeypatch.setattr(mqtt_stream, "drain_anomalies", lambda max_msgs=100: [])

    at = _logged_in_app(
        role="Engineer", page="live_monitor", data_mode="nasa",
        lm_replay_cell="B0005",
    )
    at.run()

    assert not at.exception, f"Live Monitor crashed with an active connection: {at.exception}"


# ---------------------------------------------------------------------------
# Use-case landing picker (Diagnose / Monitor / Plan)
# ---------------------------------------------------------------------------

def test_use_case_picker_renders_when_mode_not_yet_chosen(isolated_db):
    """role_chosen=True but mode_chosen unset must show the new interstitial
    (not fall through to a page, not crash) -- the sequencing this whole
    feature depends on (_FIRST_RUN_OVERLAYS in app/main.py)."""
    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", mode_chosen=False)
    at.run()
    assert not at.exception, f"Use-case picker crashed: {at.exception}"
    text = _all_text(at)
    assert "What are you here to do?" in text
    assert "Diagnose a battery" in text
    assert "Monitor live telemetry" in text
    assert "Plan a storage deployment" in text


def test_use_case_picker_diagnose_lands_on_overview(isolated_db):
    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", mode_chosen=False)
    at.run()
    at.button(key="onboard_mode_diagnose").click().run()
    assert not at.exception
    assert at.session_state["page"] == "overview"
    assert at.session_state["mode_chosen"] is True


def test_use_case_picker_monitor_lands_on_live_monitor(isolated_db):
    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", mode_chosen=False)
    at.run()
    at.button(key="onboard_mode_monitor").click().run()
    assert not at.exception
    assert at.session_state["page"] == "live_monitor"


def test_use_case_picker_plan_lands_on_decision_with_expanders_open(isolated_db):
    """The one case with a side effect beyond page routing: both nested
    expanders (decision.py's outer one, consequences.py's inner
    Solar + Storage Sizing one) must open on this arrival, and the
    mode_landing_ess flag must be gone afterward -- proving the
    read-then-pop sequencing works and won't force them open again on a
    later, unrelated visit."""
    at = _logged_in_app(
        role="Engineer", page="overview", data_mode="nasa", mode_chosen=False,
        selected_cell="B0006",  # NASA cell at ~58% SOH -- past the second-life gate, so the
                                 # inner expander actually renders (not the "Still in Primary Life" stub)
    )
    at.run()
    at.button(key="onboard_mode_plan").click().run()
    assert not at.exception, f"Plan landing crashed: {at.exception}"
    assert at.session_state["page"] == "decision"
    assert "mode_landing_ess" not in at.session_state

    expander_labels = [e.label for e in at.expander]
    assert any("Full economics" in label for label in expander_labels), expander_labels
    assert any("Solar" in label for label in expander_labels), expander_labels
    for e in at.expander:
        if "Full economics" in e.label or "Solar" in e.label:
            assert e.proto.expanded, f"Expander '{e.label}' should be open on the Plan landing"


def test_use_case_picker_does_not_reopen_expanders_on_a_later_unrelated_visit(isolated_db):
    """Regression guard for the pop-once behavior: visiting the Decision page
    normally (mode_landing_ess never set) must NOT force either expander open."""
    at = _logged_in_app(
        role="Engineer", page="decision", data_mode="nasa",
        selected_cell="B0006",
    )
    at.run()
    assert not at.exception
    for e in at.expander:
        if "Full economics" in e.label or "Solar" in e.label:
            assert not e.proto.expanded, f"Expander '{e.label}' should stay collapsed on a normal visit"


def test_benchmark_page_empty_state_no_crash(isolated_db):
    """No experiment runs logged yet in this isolated DB -- the Benchmark
    page must show its empty state, not crash trying to build a leaderboard
    table/filters from zero rows."""
    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "No experiment runs logged yet" in text


def test_benchmark_page_renders_leaderboard_with_logged_runs(isolated_db):
    """Once a run is logged (the registry's normal auto-logging path, here
    seeded directly), the Benchmark page must render a leaderboard table
    and a fold-level drill-down for it without crashing."""
    import experiment_registry as reg

    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number", "fade_rate_30cy"],
        feature_version="v9-crate-stress-index-dod-proxy",
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=["B0005", "B0006"], n_rows=300,
        lco_metrics={
            "soh_mae": 1.1, "soh_r2": 0.83, "rul_mae": 22.0, "rul_r2": 0.55,
            "rul_reliable": True,
            "per_cell": {"B0005": {"soh_mae": 1.0, "soh_r2": 0.9, "rul_mae": 20, "rul_r2": 0.6}},
        },
    )

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "Model Benchmark" in text
    assert len(at.dataframe) == 2  # leaderboard table + fold drill-down table
    assert "nasa" in at.dataframe[0].value["Dataset"].values


def test_benchmark_page_physics_divergence_button_runs_for_nasa(isolated_db):
    """Clicking "Run held-out-cell divergence check" on a NASA run's
    drill-down must render the physics-vs-GBRT divergence table with no
    exception, using real locally-cached NASA cell data reloaded via
    experiment_registry.reload_reference_cell_data() -- not a mock."""
    import experiment_registry as reg

    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number", "fade_rate_30cy"],
        feature_version="v10-physics-calibration",
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=["B0005", "B0006"], n_rows=300,
        lco_metrics={
            "soh_mae": 1.1, "soh_r2": 0.83, "rul_mae": 22.0, "rul_r2": 0.55,
            "rul_reliable": True,
            "per_cell": {"B0005": {"soh_mae": 1.0, "soh_r2": 0.9, "rul_mae": 20, "rul_r2": 0.6}},
        },
    )

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception

    btn = next(b for b in at.button if "divergence check" in b.label.lower())
    btn.click().run(timeout=180)
    assert not at.exception, f"Divergence check raised: {at.exception}"
    all_captions = "\n".join(c.value for c in at.caption)
    assert "not measuring the same thing" in all_captions
    assert len(at.dataframe) >= 1
    assert "B0005" in at.dataframe[-1].value["Cell"].values or "B0006" in at.dataframe[-1].value["Cell"].values


def test_benchmark_page_no_physics_divergence_section_for_uploaded_dataset(isolated_db):
    """Uploaded-data runs have no NASA/Severson physics calibration path --
    the divergence section must not appear at all, not show an error."""
    import experiment_registry as reg

    reg.log_run(
        org_id=1, dataset="uploaded", chemistry="Custom",
        feature_set=["cycle_number"], feature_version="v10-physics-calibration",
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=["MyCell1"], n_rows=100,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.8, "rul_mae": 10.0, "rul_r2": 0.5,
                     "rul_reliable": True, "per_cell": {}},
    )
    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "Physics vs GBRT" not in text


def test_overview_shows_no_cross_chem_badge_when_no_transfer_study_logged(isolated_db):
    """Baseline: with nothing logged in the registry, the Overview page
    must render normally with no cross-dataset transfer badge -- the
    badge is opt-in-by-evidence, never a placeholder."""
    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "CROSS-CHEM TRANSFER" not in text


def test_overview_shows_cross_chem_transfer_error_badge_when_logged(isolated_db):
    """Once a cross-chemistry generalization study has been logged for this
    cell's own dataset (nasa -> severson), the Overview page must show the
    real transfer error next to the RUL sample-size badge -- same "report
    the honest number" principle as the sample-size badge itself."""
    import experiment_registry as reg

    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa_to_severson", chemistry="LiCoO2 -> LFP",
        feature_set=["cycle_number"], feature_version="v9-crate-stress-index-dod-proxy",
        hyperparams={"random_state": 42}, seed=42, cell_ids=["B0005", "S-b1c2"], n_rows=100,
        lco_metrics={
            "soh_mae": 22.0, "soh_r2": -34.6, "rul_mae": 1350.9, "rul_r2": -1.14,
            "rul_reliable": False, "per_cell": {},
        },
        notes="Cross-chemistry generalization study: trained on nasa, zero-shot evaluated on severson.",
    )

    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "CROSS-CHEM TRANSFER: WEAK" in text
    assert "severson" in text.lower()


def test_overview_shows_not_evaluated_cross_chem_badge_for_incompatible_pairing(isolated_db):
    """The Oxford schema-incompatibility case: log_cross_chemistry_unavailable()
    records an honest "not evaluated" row (no fabricated number) -- the
    Overview badge must surface that disclosure, not silently omit it."""
    import experiment_registry as reg

    reg.log_cross_chemistry_unavailable(
        "nasa", "oxford", "Oxford schema incompatible with the per-cycle feature pipeline.",
        org_id=reg.PLATFORM_ORG_ID,
    )

    at = _logged_in_app(role="Engineer", page="overview", data_mode="nasa", selected_cell="B0005")
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "CROSS-CHEM TRANSFER: NOT EVALUATED" in text
    assert "oxford" in text.lower()


def test_regenerate_report_button_replays_and_shows_recorded_vs_reproduced(isolated_db, monkeypatch):
    """utils.render_regenerate_report_button() end-to-end: given a bundle
    carrying a logged experiment_run_id, clicking Regenerate must replay
    the recorded pipeline and render both the recorded and reproduced
    metrics. Tested in isolation (AppTest.from_string(), not the full
    main.py) with a tiny synthetic cell population and
    reload_reference_cell_data() monkeypatched to avoid touching the real
    NASA/synthetic/Severson loaders -- this is a UI-integration test for
    the registry's replay path, not a re-test of replay_run() itself
    (already covered by tests/test_experiment_registry.py)."""
    import experiment_registry as reg
    from conftest import make_cycles_df
    from batlab.validation.lco import run_lco
    from batlab.features.engineering import FEATURE_VERSION

    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }
    lco = run_lco(cell_data, seed=42)
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number", "fade_rate_30cy"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=list(cell_data.keys()), n_rows=400, lco_metrics=lco,
    )
    monkeypatch.setattr(reg, "reload_reference_cell_data", lambda dataset, cell_ids=None: cell_data)

    app_dir = str(pathlib.Path(__file__).parent.parent / "app")
    src_dir = str(pathlib.Path(__file__).parent.parent / "src")
    script = f"""
import sys
sys.path.insert(0, {app_dir!r})
sys.path.insert(0, {src_dir!r})
from utils import render_regenerate_report_button
bundle = {{"metrics": {{"experiment_run_id": {run_id!r}}}}}
render_regenerate_report_button(bundle, org_id=1, key_suffix="test")
"""
    at = AppTest.from_string(script)
    at.run()
    assert not at.exception, f"Initial render raised: {at.exception}"

    btn = next(b for b in at.button if "Regenerate" in b.label)
    btn.click().run()
    assert not at.exception, f"Click raised: {at.exception}"

    texts = _all_text(at)
    assert "Recorded" in texts
    assert "Reproduced now" in texts
    assert f"{lco['rul_mae']:.1f}" in texts  # the reproduced number matches the recorded one exactly

    # This test's hyperparams={"random_state": 42} is a partial fixture dict
    # (not a real dict(GBRT_PARAMS) copy), so the hyperparams-drift warning
    # (src/experiment_registry.py's "The replay contract" section) must fire
    # -- st.warning() is its own AppTest element type, not markdown.
    warnings_ = [w.value for w in at.warning]
    assert any("GBRT hyperparameters" in w for w in warnings_), warnings_


# ---------------------------------------------------------------------------
# Phase 6 — Physics-Informed Battery Intelligence (src/physics_calibration.py)
# ---------------------------------------------------------------------------

def test_health_page_shows_physics_sei_lam_decomposition_for_nasa_cell(isolated_db):
    """
    Health page's Model Comparison expander must show the physics
    calibration's SEI/LAM decomposition (beta_sei, beta_lam, resistance
    growth k_r, dominant mode) for a real NASA cell -- not just the
    pre-existing single-beta PyBaMM fit. Regression guard for the physics
    calibration -> GBRT/UI wiring (src/physics_calibration.py).
    """
    at = _logged_in_app(role="Engineer", page="health", data_mode="nasa")
    at.run()
    assert not at.exception, f"Health page raised an exception: {at.exception}"

    metric_labels = {m.label for m in at.metric}
    assert "β SEI (√n)" in metric_labels
    assert "β LAM (linear)" in metric_labels
    assert "SEI resistance growth (k_r)" in metric_labels
    assert "Physics dominant mode" in metric_labels

    dominant = next(m.value for m in at.metric if m.label == "Physics dominant mode")
    assert dominant in ("LLI — Loss of Lithium Inventory", "LAM — Loss of Active Material", "Mixed LLI + LAM")


def test_health_page_no_physics_metrics_for_synthetic_cell(isolated_db):
    """Synthetic cells are not calibration-eligible (NASA/Severson only) --
    the SEI/LAM decomposition block must not appear at all for them,
    rather than showing empty/placeholder physics metrics."""
    at = _logged_in_app(role="Engineer", page="health", data_mode="synthetic")
    at.run()
    assert not at.exception, f"Health page raised an exception: {at.exception}"
    metric_labels = {m.label for m in at.metric}
    assert "β SEI (√n)" not in metric_labels


def test_copilot_page_renders_without_exception(isolated_db):
    """Basic smoke test -- this page had zero AppTest coverage before the
    real tool-use Copilot build (src/copilot_agent.py); confirms the page
    loads cleanly with no query selected (the empty-state prompt) and no
    Anthropic API key configured."""
    at = _logged_in_app(role="Engineer", page="copilot", data_mode="nasa")
    at.run()
    assert not at.exception, f"Copilot page raised an exception: {at.exception}"


def test_copilot_free_text_falls_back_to_template_without_api_key(isolated_db):
    """Real tool-use only runs when a personal Anthropic API key is
    configured (copilot_agent.answer_with_tools() returns (None, []) with
    no key) -- without one, the free-text ask box must fall back to
    exactly the pre-existing template+retrieval path, not crash or
    silently show nothing."""
    at = _logged_in_app(
        role="Engineer", page="copilot", data_mode="nasa",
        copilot_free_text="Why is the RUL uncertain?",
    )
    at.run()
    assert not at.exception, f"Copilot free-text fallback raised an exception: {at.exception}"
    text = "\n".join(m.value for m in at.markdown)
    assert "Template fallback" in text
    assert "Claude Sonnet 5" not in text

