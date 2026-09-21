"""Tests for the CellSceneSpec builder (src/cell_scene.py).

Three kinds of test, in order of how much they protect:

1. **Contract tests** — the spec validates against docs/cell_scene.schema.json,
   carries exactly one card per anatomy mesh (the bijection both the renderer
   and this builder assert), every series is aligned to one shared index, and
   the whole thing is JSON round-trippable. These are what stop data and
   geometry drifting apart in a view nobody can unit-test by looking at it.
2. **Value tests** — each card's number equals the value the platform computes
   elsewhere (the same column, the same library function), so the 3D view can
   never quietly disagree with the page beside it.
3. **Refusal tests** — the branches where the honest answer is "no number":
   an unidentifiable SEI/LAM split, a refused forecast, an inapplicable
   chemistry, a dead model column, and a frame with nothing in it.

The refusal tests matter most. Every one of them pins a case where a plausible
implementation would have drawn something confident and wrong.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from cell_scene import (
    FORM_FACTOR_CYLINDRICAL,
    FORM_FACTOR_PRISMATIC,
    FORM_FACTOR_UNKNOWN,
    MESH_PART_IDS,
    PhysicalModel,
    SCENE_SCHEMA_VERSION,
    build_cell_scene,
    build_cell_scene_from_sources,
    form_factor_for,
    physics_series,
    resolve_bundle_key,
)

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "docs" / "cell_scene.schema.json"


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _frame(
    n=180, *, fade_per_cycle=0.12, resistance_rise_per_cycle=0.00008,
    temperature=32.0, dqdv=True, dead_columns=False, start_resistance=0.05,
):
    """A featured-shaped frame: the columns src/cell_scene.py reads, nothing more."""
    cycles = np.arange(1, n + 1, dtype=float)
    soh = 100.0 - fade_per_cycle * (cycles - 1)
    capacity = 2.0 * soh / 100.0
    frame = pd.DataFrame({
        "cycle_number": cycles,
        "soh_pct": soh,
        "capacity_ah": capacity,
        "resistance_ohm": start_resistance + resistance_rise_per_cycle * cycles,
        "temperature_c": np.full(n, temperature),
        "sop_pct": 100.0 * start_resistance / (start_resistance + resistance_rise_per_cycle * cycles),
        "fade_rate_30cy": np.full(n, fade_per_cycle / 100.0 * 2.0),
        "physics_beta_sei": np.full(n, 0.002),
        "physics_beta_lam": np.full(n, 0.0002),
        "physics_fit_r2": np.full(n, 0.95),
        "physics_spm_capacity_ah": np.full(n, 0.42),
    })
    frame["resistance_normalized"] = frame["resistance_ohm"] / frame["resistance_ohm"].iloc[0]
    if dqdv:
        frame["dqdv_sim_peak_value"] = 140.0 + 60.0 * (cycles - 1) / max(n - 1, 1)
        frame["dqdv_sim_peak_soc"] = np.full(n, 0.0251)
        frame["dqdv_sim_area"] = -4.0 + 1.0 * (cycles - 1) / max(n - 1, 1)
        frame["dqdv_sim_fwhm"] = np.zeros(n)
    if dead_columns:
        # A flat cell: no fade to fit, so the two-term fit cannot converge.
        frame["soh_pct"] = np.full(n, 100.0)
        frame["capacity_ah"] = np.full(n, 2.0)
    return frame


def _spec(cell_id="B0005", *, frame=None, horizon_cycles=None, bundle=None, bundles=None,
          **frame_kwargs):
    """A spec built from a synthetic frame: frame kwargs pass through to _frame."""
    df = frame if frame is not None else _frame(**frame_kwargs)
    extra = {}
    if horizon_cycles is not None:
        extra["horizon_cycles"] = horizon_cycles
    if bundle is not None:
        extra["bundle"] = bundle
    if bundles is not None:
        extra["bundles"] = bundles
    return build_cell_scene(cell_id, df, **extra)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_the_spec_validates_against_the_published_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    spec = _spec()
    jsonschema.validate(spec, schema)


def test_the_schema_version_is_the_one_both_sides_declare():
    spec = _spec()
    assert spec["schemaVersion"] == SCENE_SCHEMA_VERSION == 1
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schemaVersion"]["const"] == SCENE_SCHEMA_VERSION


def test_there_is_exactly_one_card_per_anatomy_mesh():
    """The contract the renderer relies on: no mesh without a card, no card
    without a mesh. A new layer must add both or the suite fails."""
    spec = _spec()
    ids = [p["id"] for p in spec["parts"]]
    assert len(ids) == len(set(ids)), "two cards for one part"
    assert set(ids) == set(MESH_PART_IDS)


def test_the_schema_enum_lists_exactly_the_mesh_parts_this_builder_emits():
    """Third surface for the same bijection: a third-party renderer reads the
    schema, so its enum must not drift from the builder's part list."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(
        schema["properties"]["parts"]["items"]["properties"]["id"]["enum"]
    )
    assert enum == set(MESH_PART_IDS)


def test_every_series_is_aligned_to_one_shared_index():
    """The renderer addresses every array with a single cursor."""
    spec = _spec()
    n = len(spec["series"]["cycles"])
    assert n > 0
    for name, values in spec["series"].items():
        assert len(values) == n, f"{name} is {len(values)} long, not {n}"
    for part in spec["parts"]:
        if part["series"] is not None:
            assert len(part["series"]) == n, f"part {part['id']} series is misaligned"


def test_the_spec_round_trips_as_json_with_no_numpy_types():
    spec = _spec()
    encoded = json.dumps(spec)
    assert json.loads(encoded) == spec
    # NaN would serialize as the invalid literal NaN; None must be used instead.
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_a_long_record_is_sampled_but_keeps_both_endpoints():
    spec = _spec(frame=_frame(n=1500))
    cycles = spec["series"]["cycles"]
    assert spec["record"]["sampled"] is True
    assert spec["record"]["nRecordedRows"] == 1500
    assert len(cycles) <= 240
    assert cycles[0] == 1.0 and cycles[-1] == 1500.0
    assert spec["record"]["lastCycle"] == 1500.0


def test_a_short_record_is_not_resampled():
    spec = _spec(frame=_frame(n=120))
    assert spec["record"]["sampled"] is False
    assert spec["record"]["nRecordedRows"] == len(spec["series"]["cycles"]) == 120


# ---------------------------------------------------------------------------
# Values: every card points at a number the platform already computed
# ---------------------------------------------------------------------------

def test_the_casing_card_is_the_measured_temperature():
    frame = _frame(temperature=41.5)
    spec = build_cell_scene("B0005", frame)
    can = next(p for p in spec["parts"] if p["id"] == "can")
    assert can["value"] == pytest.approx(41.5)
    assert can["provenance"] == "measured"
    assert can["available"] is True
    assert can["series"] == [pytest.approx(41.5)] * len(frame)


def test_the_terminal_card_is_the_measured_resistance():
    frame = _frame()
    spec = build_cell_scene("B0005", frame)
    terminal = next(p for p in spec["parts"] if p["id"] == "terminal_pos")
    assert terminal["value"] == pytest.approx(float(frame["resistance_ohm"].iloc[-1]))
    assert terminal["provenance"] == "measured"


def test_the_power_card_is_the_platform_s_own_sop_column():
    frame = _frame()
    spec = build_cell_scene("B0005", frame)
    tab = next(p for p in spec["parts"] if p["id"] == "tab_neg")
    assert tab["value"] == pytest.approx(float(frame["sop_pct"].iloc[-1]))
    assert tab["provenance"] == "derived"


def test_the_resistance_growth_card_is_the_normalised_column():
    frame = _frame()
    spec = build_cell_scene("B0005", frame)
    tab = next(p for p in spec["parts"] if p["id"] == "tab_pos")
    assert tab["value"] == pytest.approx(float(frame["resistance_normalized"].iloc[-1]))


def test_the_fitted_physics_comes_from_the_library_fit_not_a_reimplementation():
    """The two betas must equal what physics_calibration.fit_two_term_fade returns
    for this exact frame — not something this module derived its own way."""
    from physics_calibration import fit_two_term_fade
    frame = _frame(n=200)
    fit = fit_two_term_fade(frame["cycle_number"].to_numpy(float), frame["soh_pct"].to_numpy(float))
    spec = build_cell_scene("B0005", frame)
    assert spec["physics"]["betaSei"] == pytest.approx(fit["beta_sei"])
    assert spec["physics"]["betaLam"] == pytest.approx(fit["beta_lam"])
    assert spec["physics"]["fitR2"] == pytest.approx(fit["r2"])
    assert spec["physics"]["nCyclesUsed"] == fit["n_cycles_used"]


def test_the_physics_law_is_the_one_the_library_fits():
    # The library's own model function — private, so imported from the library
    # rather than through the app's src/physics_calibration.py shim, whose
    # __all__ is the public surface the application is allowed to call.
    from batlab.features.physics_calibration import _two_term_fade_model
    cycles = np.array([1.0, 50.0, 100.0, 154.0])
    phys = physics_series(cycles.tolist(), 0.002, 0.0002)
    for i, cy in enumerate(cycles):
        n = float(cy)  # first cycle is 1, so n == cycle here
        expected = (1.0 - _two_term_fade_model(n, 0.002, 0.0002)) * 100.0
        assert phys["seiPct"][i] + phys["lamPct"][i] == pytest.approx(expected)


def test_both_fitted_terms_are_monotone_over_the_record():
    """A film cannot shrink: the geometry is only drawable because a single
    full-history fit makes both terms monotone in n (the per-window columns in
    the frame do not — see the module docstring)."""
    spec = _spec(n=300)
    sei = [v for v in spec["series"]["seiPct"] if v is not None]
    lam = [v for v in spec["series"]["lamPct"] if v is not None]
    assert all(b >= a for a, b in zip(sei, sei[1:]))
    assert all(b >= a for a, b in zip(lam, lam[1:]))


def test_the_identified_total_is_the_sum_of_the_two_channels():
    spec = _spec()
    for sei, lam, total in zip(
        spec["series"]["seiPct"], spec["series"]["lamPct"], spec["series"]["fadeModelPct"]
    ):
        if None not in (sei, lam, total):
            assert total == pytest.approx(sei + lam)


def test_measured_fade_publishes_the_reconciliation_with_the_fit():
    frame = _frame(n=200)
    spec = build_cell_scene("B0005", frame)
    assert spec["physics"]["measuredFadePct"] == pytest.approx(
        100.0 - float(frame["soh_pct"].iloc[-1]))
    assert spec["physics"]["modelTotalFadePct"] is not None


def test_the_electrode_cards_are_the_dqdv_proxy_columns_and_say_so():
    spec = _spec()
    cathode = next(p for p in spec["parts"] if p["id"] == "cathode_sheet")
    anode = next(p for p in spec["parts"] if p["id"] == "anode_sheet")
    assert cathode["value"] == pytest.approx(200.0, abs=1.0)  # 140 -> 200 across the record
    assert anode["value"] == pytest.approx(-3.0, abs=0.1)
    assert cathode["provenance"] == "derived"
    assert "proxy" in cathode["meaning"].lower() or "proxy" in anode["meaning"].lower()


def test_the_dead_dqdv_columns_are_never_presented_as_cell_state():
    """peak_soc is a model constant and fwhm is degenerate; both must be absent
    from the scene rather than drawn as this cell's condition."""
    spec = _spec()
    drawn = {p["id"]: p for p in spec["parts"]}
    positions = {p["value"] for p in spec["parts"] if p["value"] is not None}
    assert 0.0251 not in positions
    assert 0.0 not in positions
    assert "peak_soc" not in json.dumps(spec)
    assert "fwhm" not in json.dumps(spec)
    assert drawn["anode_sheet"]["law"].startswith("simulated dQ/dV")


def test_the_knee_verdict_is_the_platform_knee_detector_s():
    from batlab.features.knee_detection import detect_knee
    frame = _frame(n=400, fade_per_cycle=0.05)
    frame.loc[300:, "soh_pct"] -= np.linspace(0, 12, 100)  # force a knee
    expected = detect_knee(frame["soh_pct"], frame["cycle_number"])
    spec = build_cell_scene("B0005", frame)
    assert spec["knee"]["detected"] == bool(expected["detected"])
    assert spec["knee"]["phase"] == expected["phase"]


# ---------------------------------------------------------------------------
# Refusals — the branches where a plausible implementation lies
# ---------------------------------------------------------------------------

def test_an_unidentifiable_channel_split_is_refused_not_drawn():
    """The case that is true of every cell this platform ships: sqrt(n) and n
    are collinear over a few hundred cycles, so one channel fits far below its
    own standard error. The scene must draw no film thickness."""
    spec = _spec(n=200)
    assert spec["physics"]["splitIdentified"] is False
    assert spec["physics"]["splitReason"]
    for part_id in ("sei_film", "particles"):
        part = next(p for p in spec["parts"] if p["id"] == part_id)
        assert part["available"] is False
        assert part["value"] is None
        assert part["series"] is None
        assert "cannot separate" in (part["unavailableReason"] or "")


def test_an_unidentifiable_split_does_not_claim_agreement_between_verdicts():
    """Two verdicts agreeing is only evidence if they are independent; a physics
    verdict read off an unidentified ridge is not."""
    spec = _spec(n=200)
    assert spec["mechanism"]["agree"] is None
    assert spec["mechanism"]["agreementIsEvidence"] is False
    assert "nothing independent to agree with" in spec["mechanism"]["note"]


def test_the_emphasis_follows_the_ml_verdict_when_the_physics_cannot_speak():
    spec = _spec(n=200)
    assert spec["mechanism"]["emphasisSource"] in ("ml", "none")
    if spec["mechanism"]["emphasisSource"] == "ml":
        assert spec["mechanism"]["emphasis"] in ("sei", "particles")


def test_a_separated_split_is_drawn_and_agreement_becomes_evidence(monkeypatch):
    """The other branch, reached by making the fit identifiable: both channels
    above their own sigma. Confirms the refusal above is a data test and not a
    hard-coded refusal."""
    import cell_scene
    real = cell_scene.fit_physics
    monkeypatch.setattr(cell_scene, "fit_physics", lambda df: {
        **real(df), "splitIdentified": True, "splitReason": None,
        "seiTStat": 4.0, "lamTStat": 3.0,
    })
    spec = _spec(n=200)
    assert spec["physics"]["splitIdentified"] is True
    sei = next(p for p in spec["parts"] if p["id"] == "sei_film")
    assert sei["available"] is True and sei["series"] is not None
    assert spec["mechanism"]["emphasisSource"] in ("physics", "none")


def test_the_forecast_is_refused_with_a_reason_when_the_router_refuses():
    """A refused route must produce no future at all — the scene's whole
    honesty claim rests on not drawing a confident line where the platform
    declines to forecast."""
    spec = _spec(n=200, bundles={
        "nasa": {"metrics": {"forecast_routing": [
            {"cell_id": "B0005", "served": "refuse", "reason": "no regime row clears the floor"},
        ]}},
    })
    assert spec["projection"]["available"] is False
    assert spec["projection"]["reason"]


def test_no_forecast_fit_means_no_future_and_the_scene_says_why():
    spec = _spec(n=200)
    assert spec["projection"]["available"] is False
    assert "hierarchical forecast fit" in spec["projection"]["reason"]
    assert any("stops at the last measured cycle" in d for d in spec["disclosures"])


def test_a_forecast_fit_produces_a_banded_projection():
    from batlab.models.hierarchical import fit_hierarchical
    frame = _frame(n=200)
    raw = {"B0005": frame[["cycle_number", "capacity_ah", "soh_pct"]]}
    fit = fit_hierarchical(raw, chemistry_by_cell={"B0005": "LCO"})
    spec = build_cell_scene("B0005", frame, bundle={"hierarchical_fit": fit},
                            horizon_cycles=120)
    proj = spec["projection"]
    assert proj["available"] is True
    assert len(proj["cycles"]) == 120
    assert proj["cycles"][0] > spec["record"]["lastCycle"]
    assert proj["kind"] == "projected"
    assert proj["modelLabel"]
    assert len(proj["sohQ10Pct"]) == len(proj["sohPct"]) == len(proj["sohQ90Pct"])
    # The band brackets the central line.
    for q10, mid, q90 in zip(proj["sohQ10Pct"], proj["sohPct"], proj["sohQ90Pct"]):
        if None not in (q10, mid, q90):
            assert q10 <= mid + 1e-9 and mid <= q90 + 1e-9


def test_an_inapplicable_chemistry_gets_no_electrode_numbers():
    """Severson is LFP: the platform's LiCoO2-shaped dQ/dV model is declared
    inapplicable, even though the frame carries the columns."""
    spec = _spec(cell_id="S-b1c0", n=200)
    assert spec["cell"]["dqdvApplicable"] is False
    for part_id in ("cathode_sheet", "anode_sheet"):
        part = next(p for p in spec["parts"] if p["id"] == part_id)
        assert part["available"] is False
        assert part["value"] is None
        assert "dQ/dV" in (part["unavailableReason"] or "")


def test_a_frame_with_no_fade_has_no_fitted_layer_and_a_reason():
    spec = _spec(n=200, dead_columns=True)
    assert spec["physics"]["available"] is False
    for part_id in ("sei_film", "particles"):
        part = next(p for p in spec["parts"] if p["id"] == part_id)
        assert part["available"] is False
        assert part["unavailableReason"]


def test_an_empty_frame_does_not_raise_and_reports_nothing_measured():
    frame = _frame(n=1)
    frame = frame.iloc[0:0]
    spec = build_cell_scene("B0005", frame)
    assert spec["record"]["nCycles"] == 0
    assert all(p["available"] is False for p in spec["parts"])
    assert spec["series"]["sohPct"] == []


def test_a_missing_column_never_becomes_a_zero():
    """Dropping temperature must remove the can's number, not print 0.0 °C on
    a scene where the casing colour means temperature."""
    frame = _frame().drop(columns=["temperature_c"])
    spec = build_cell_scene("B0005", frame)
    can = next(p for p in spec["parts"] if p["id"] == "can")
    assert can["available"] is False
    assert can["value"] is None
    assert can["series"] is None
    assert can["unavailableReason"]


def test_unavailable_parts_always_explain_themselves():
    spec = _spec()
    for part in spec["parts"]:
        if not part["available"]:
            assert part["unavailableReason"], f"{part['id']} is unavailable with no reason"
            assert part["meaning"], f"{part['id']} is unavailable with no explanation"
        else:
            assert part["provenance"] in ("measured", "derived", "fitted", "projected")


def test_the_disclosures_cover_the_claims_the_scene_makes():
    spec = _spec()
    text = " ".join(spec["disclosures"]).lower()
    assert "schematic" in text
    assert "anatomical proportions" in text
    assert "measured, derived, fitted or projected" in text
    assert "does not feed" in text or "nothing in this view feeds" in text


# ---------------------------------------------------------------------------
# Form factor and source resolution
# ---------------------------------------------------------------------------

class _Profile:
    def __init__(self, passport, dqdv=True):
        self.passport_chemistry = passport
        self.dqdv_applicable = dqdv


def test_form_factor_is_read_from_the_chemistry_profile():
    assert form_factor_for(_Profile("LiCoO₂ (lithium cobalt oxide), 18650 cylindrical"))[0] == FORM_FACTOR_CYLINDRICAL
    assert form_factor_for(_Profile("LiCoO₂ (lithium cobalt oxide), 1.1 Ah prismatic"))[0] == FORM_FACTOR_PRISMATIC
    assert form_factor_for(_Profile("unknown chemistry"))[0] == FORM_FACTOR_UNKNOWN
    assert form_factor_for(None)[0] == FORM_FACTOR_UNKNOWN


def test_an_unknown_form_factor_is_disclosed_instead_of_guessed():
    spec = _spec(cell_id="S-b1c0")
    assert spec["cell"]["formFactor"] in (FORM_FACTOR_CYLINDRICAL, FORM_FACTOR_PRISMATIC, FORM_FACTOR_UNKNOWN)
    joined = " ".join(spec["disclosures"]).lower()
    assert "drawn as" in joined or "form factor" in joined


def test_a_prismatic_source_says_so_in_the_spec_and_the_disclosures():
    spec = _spec(cell_id="MLP-a1")
    if spec["cell"]["formFactor"] == FORM_FACTOR_PRISMATIC:
        assert any("prismatic" in d.lower() for d in spec["disclosures"])
    else:
        # Non-prismatic sources must still declare a form factor honestly.
        assert spec["cell"]["formFactor"] in (FORM_FACTOR_CYLINDRICAL, FORM_FACTOR_UNKNOWN)


def test_bundle_keys_resolve_for_real_and_uploaded_cells():
    assert resolve_bundle_key("B0005") == "nasa"
    assert resolve_bundle_key("S-b1c0") == "severson"
    assert resolve_bundle_key("Cell1") == "synth"
    assert resolve_bundle_key("someone-elses-cell-42") == "upload"


def test_the_sources_wrapper_finds_the_frame_and_builds_the_same_spec():
    frame = _frame()
    spec = build_cell_scene_from_sources("B0005", {"B0005": frame})
    assert spec is not None
    assert spec["cell"]["id"] == "B0005"
    assert spec["schemaVersion"] == SCENE_SCHEMA_VERSION


def test_the_sources_wrapper_returns_none_for_a_cell_it_does_not_have():
    assert build_cell_scene_from_sources("nope", {"B0005": _frame()}) is None
    assert build_cell_scene_from_sources("nope", {}) is None


def test_a_null_valued_column_is_reported_as_unavailable_not_as_a_number():
    frame = _frame()
    frame["temperature_c"] = np.nan
    spec = build_cell_scene("B0005", frame)
    can = next(p for p in spec["parts"] if p["id"] == "can")
    assert can["available"] is False
    assert can["value"] is None
    assert can["series"] is None or all(v is None for v in can["series"])


def test_geometry_scales_name_the_series_they_read_and_promise_nothing_else():
    spec = _spec()
    scales = spec["geometryScales"]
    assert "sei_film" in scales and "fade_model" in scales
    for key, scale in scales.items():
        assert scale["target"]
        assert scale["from"] in ("series.seiPct", "series.lamPct", "series.fadeModelPct",
                                 "series.temperatureC", "series.sopPct", "series.resistanceOhm",
                                 "series.seiThicknessNm"), key
        assert scale["note"], f"{key} has no disclosed mapping"
    # Every scale that points at a per-cycle series must point at one that exists.
    series_names = set(spec["series"].keys())
    for scale in scales.values():
        assert scale["from"].split(".", 1)[1] in series_names


# ---------------------------------------------------------------------------
# Theme: the scene's colours are the platform's own vocabulary, not new ones
# ---------------------------------------------------------------------------

def test_the_scene_soh_bands_are_the_platform_s_own_bands():
    """A cell the app calls Degrading must not be drawn green here.

    The bands are the one place a second implementation would silently diverge:
    `app/_ui_helpers.soh_status()` decides the label and the CSS class, and
    `app/static/theme.css` decides what colour that class paints. This asserts
    the scene's default theme agrees with both — thresholds, labels and hex.
    """
    import re
    import pathlib

    from _ui_helpers import soh_status
    from cell_scene import default_theme

    theme = default_theme()
    css = (pathlib.Path(__file__).resolve().parents[1] / "app" / "static" / "theme.css").read_text(
        encoding="utf-8"
    )
    for soh in (99.0, 95.0, 90.0, 89.9, 85.0, 80.0, 79.9, 60.0):
        label, css_class = soh_status(soh)
        band = next(
            b for b in theme["sohBands"]
            if soh >= b["min"] and (b["max"] is None or soh < b["max"])
        )
        assert band["label"] == label, f"SOH {soh}: scene says {band['label']}, app says {label}"
        hex_code = re.search(rf"\.{css_class}\s*\{{\s*color:\s*(#[0-9a-fA-F]{{6}})", css)
        assert hex_code, f"{css_class} has no colour in app/static/theme.css"
        assert band["color"].lower() == hex_code.group(1).lower(), (
            f"SOH {soh}: the scene paints {band['color']}, the app paints {hex_code.group(1)}"
        )


def test_the_default_theme_is_a_copy_the_caller_cannot_poison():
    from cell_scene import default_theme

    first = default_theme()
    first["sohBands"][0]["color"] = "#000000"
    assert default_theme()["sohBands"][0]["color"] != "#000000"


def test_a_host_theme_is_merged_over_the_defaults_rather_than_replacing_them():
    frame = _frame()
    spec = build_cell_scene("B0005", frame, theme={"accent": "#ff00ff"})
    assert spec["theme"]["accent"] == "#ff00ff"
    assert spec["theme"]["sohBands"]  # still there: a partial theme is not a wipe


def test_a_router_verdict_carrying_a_model_still_serializes(monkeypatch):
    """The document is a contract, so it must survive `json.dumps` — including
    when the forecast router hands back the fitted estimators behind its
    decision. That was a real failure: the endpoint and the Streamlit iframe
    both serialize this spec, and a live `GradientBoostingRegressor` in the
    route turned the whole view into a TypeError.
    """
    import forecast_routing

    import batlab.models.hierarchical as hierarchical

    class _Model:
        pass

    monkeypatch.setattr(
        forecast_routing, "route_forecast_for_cell",
        lambda *a, **kw: {"served": "hierarchical", "reason": "regime clears both floors",
                          "model_label": "Hierarchical", "estimator": _Model()},
    )
    monkeypatch.setattr(
        hierarchical, "project_future_soh",
        lambda fit, cell_id, horizon: {
            "cycles": list(range(1, horizon + 1)),
            "soh_pct": [95.0 - 0.01 * i for i in range(horizon)],
            "soh_q10_pct": [94.0 - 0.01 * i for i in range(horizon)],
            "soh_q90_pct": [96.0 - 0.01 * i for i in range(horizon)],
            "shrinkage_weight_prior": 0.5,
        },
    )
    with_fit = build_cell_scene(
        "B0005", _frame(n=200),
        bundle={"hierarchical_fit": {"pooled": True}},
        bundles={"nasa": {"hierarchical_fit": {"pooled": True}}},
        horizon_cycles=5,
    )
    encoded = json.dumps(with_fit)
    route = with_fit["projection"].get("route")
    assert route is not None
    assert route["reason"] == "regime clears both floors"
    assert "estimator" not in route
    assert route["unserializedKeys"] == ["estimator"]
    assert json.loads(encoded) == with_fit


def test_a_refused_route_keeps_its_own_words_in_the_spec(monkeypatch):
    import forecast_routing

    monkeypatch.setattr(
        forecast_routing, "route_forecast_for_cell",
        lambda *a, **kw: {"served": "refuse", "reason": "no regime row clears the floor"},
    )
    spec = build_cell_scene(
        "B0005", _frame(n=200),
        bundle={"hierarchical_fit": {"pooled": True}},
        bundles={"nasa": {"hierarchical_fit": {"pooled": True}}},
    )
    assert spec["projection"]["available"] is False
    assert "no regime row clears the floor" in spec["projection"]["reason"]
    assert any("stops at the last measured cycle" in text for text in spec["disclosures"])


# ---------------------------------------------------------------------------
# The physical model: the winding is drawn from these millimetres
# ---------------------------------------------------------------------------

def test_the_physical_block_declares_a_dimension_for_every_drawn_size():
    spec = _spec()
    physical = spec["physical"]
    assert physical["formFactor"] == FORM_FACTOR_CYLINDRICAL
    cyl = physical["cylindrical"]
    assert cyl["diameterMm"] == pytest.approx(18.4)
    assert cyl["heightMm"] == pytest.approx(65.0)
    assert physical["unitsMmPerCellUnit"] == pytest.approx(cyl["heightMm"])
    roll = physical["roll"]
    # The stack tiles the pitch exactly: the three drawn ribbons are the anode
    # coating on its copper, the separator, and the cathode coating on its
    # aluminium, and nothing is left over.
    stack = roll["stackMm"]
    assert roll["pitchMm"] == pytest.approx(sum(stack.values()))
    assert roll["drawnRibbonsMm"]["anode"] == pytest.approx(stack["copperFoil"] + stack["anodeCoating"])
    assert roll["drawnRibbonsMm"]["separator"] == pytest.approx(stack["separator"])
    assert roll["drawnRibbonsMm"]["cathode"] == pytest.approx(
        stack["aluminiumFoil"] + stack["cathodeCoating"]
    )
    assert sum(roll["drawnRibbonsMm"].values()) == pytest.approx(roll["pitchMm"])


def test_the_turn_count_is_derived_from_the_dimensions_not_chosen():
    roll = _spec()["physical"]["roll"]
    span = roll["envelopeDiameterMm"] / 2 - roll["mandrelDiameterMm"] / 2
    expected = int(span // roll["pitchMm"])
    assert roll["turns"] == expected
    # …and the roll it describes fills that envelope without leaving it.
    assert roll["drawnOuterDiameterMm"] <= roll["envelopeDiameterMm"]
    assert roll["drawnOuterDiameterMm"] > 0.95 * roll["envelopeDiameterMm"]
    # What the winding implies for the electrode, reported for checking: the sum
    # of the turn circumferences, which for an 18650-class stack and envelope is
    # a metre of electrode, not a few centimetres.
    assert 0.5 < roll["electrodeLengthM"] < 2.0


def test_the_roll_may_not_be_drawn_wider_than_the_can_that_holds_it():
    cyl = _spec()["physical"]["cylindrical"]
    roll = _spec()["physical"]["roll"]
    assert cyl["rollClearanceMm"] > 0, "a roll that touches the wall cannot be inserted"
    assert roll["envelopeDiameterMm"] + 2 * cyl["wallMm"] + 2 * cyl["rollClearanceMm"] == pytest.approx(
        cyl["diameterMm"]
    )


def test_every_declared_dimension_says_where_it_came_from():
    physical = _spec()["physical"]
    assert physical["provenance"]["diameterMm"] == "format-standard"
    assert physical["provenance"]["turns"] == "derived"
    assert physical["provenance"]["mandrelDiameterMm"] == "assumed"
    assert "format-standard" not in physical["provenance"]["stackMm"]
    assert physical["note"], "the block must explain what it is"
    assert physical["schematic"], "and name what is not to a datasheet"


def test_a_prismatic_cell_declares_its_envelope_and_admits_its_stack_is_a_diagram():
    physical = PhysicalModel(FORM_FACTOR_PRISMATIC, "1.1 Ah prismatic").block()
    assert physical["formFactor"] == FORM_FACTOR_PRISMATIC
    assert physical["prismatic"]["widthMm"] == pytest.approx(20.5)
    assert physical["prismatic"]["thicknessMm"] == pytest.approx(5.4)
    assert physical["cylindrical"] is None
    assert physical["roll"] is None, "a prismatic cell is not a winding"
    assert any("stack" in item for item in physical["schematic"])
    assert "diagram of a stacked cell" in physical["note"]
    # A source that really is prismatic reaches that same block through the spec.
    spec = _spec(cell_id="MLP-a1")
    if spec["cell"]["formFactor"] == FORM_FACTOR_PRISMATIC:
        assert spec["physical"]["roll"] is None
        assert any("diagram of a stacked cell" in d for d in spec["disclosures"])


def test_an_undeclared_form_factor_is_drawn_as_the_default_and_says_so():
    physical = PhysicalModel(FORM_FACTOR_UNKNOWN, "").block()
    assert physical["formFactor"] == FORM_FACTOR_UNKNOWN
    assert physical["roll"] is not None, "the fallback envelope is still declared"
    # …and it is the same 18650 block a declared cylindrical cell gets, so the
    # drawing is unchanged — only the claim about it is.
    assert physical["roll"]["turns"] == PhysicalModel(FORM_FACTOR_CYLINDRICAL, "x").block()["roll"]["turns"]


def test_the_disclosure_states_the_winding_it_actually_drew():
    spec = _spec()
    roll = spec["physical"]["roll"]
    sentence = next(d for d in spec["disclosures"] if "foil-to-foil stack" in d)
    assert f"{roll['pitchMm']} mm foil-to-foil stack" in sentence
    assert f"{roll['turns']} turns" in sentence
    assert f"{roll['electrodeLengthM']} m electrode" in sentence
    # The film used to be disclosed as a legibility-scaled metaphor; what must
    # be in this sentence now is that its *thickness* is derived and its drawn
    # layer magnified — the two facts a reader has to keep apart.
    assert "SEI film's thickness *is* derived" in sentence
    assert "magnified" in sentence


# ---------------------------------------------------------------------------
# The SEI film's thickness: derived from the fitted loss, magnified to be seen
# ---------------------------------------------------------------------------

def _identified(monkeypatch, beta_sei=0.01, beta_lam=0.0002):
    """Reach the identified branch with a substantial √n term.

    The fit on the synthetic frame puts β_sei below its own standard error (which
    is why the refusal tests above are real data tests), so the identified branch
    supplies the betas the fit would have returned for a cell whose film is
    actually identified.
    """
    import cell_scene
    real = cell_scene.fit_physics
    monkeypatch.setattr(cell_scene, "fit_physics", lambda df: {
        **real(df), "splitIdentified": True, "splitReason": None,
        "seiTStat": 9.0, "lamTStat": 4.0,
        "betaSei": beta_sei, "betaLam": beta_lam,
    })


def test_the_film_chain_is_arithmetic_a_reader_can_re_do():
    """The nm-per-percent factor must fall out of the stated assumptions and the
    cell's own capacity and area. That is the whole difference between a derived
    thickness and a drawing constant."""
    import cell_scene
    physical = PhysicalModel(FORM_FACTOR_CYLINDRICAL, "cylindrical").block()
    film = cell_scene.film_model(physical, 1.8026, 20.0)
    assert film["available"] and film["identified"]
    a, d = film["assumptions"], film["derivation"]
    for key in a:
        assert film["provenance"].get(key), f"{key} is assumed without saying so"
    molar_volume = a["molarMassGPerMol"] / a["densityGPerCm3"]
    area_cm2 = a["anodeFaces"] * (d["electrodeLengthM"] * 100.0) * (a["coatedWidthMm"] / 10.0)
    mol_film = (0.01 * d["capacity0Ah"] * 3600.0 / 96485.33212) / a["lithiumPerFormulaUnit"]
    # The block rounds what it reports; the chain is checked at that precision.
    assert d["molarVolumeCm3PerMol"] == pytest.approx(molar_volume, rel=1e-4)
    assert d["anodeAreaCm2"] == pytest.approx(area_cm2, rel=1e-4)
    assert d["nmPerPctLli"] == pytest.approx(mol_film * molar_volume / area_cm2 * 1e7, rel=1e-6)
    assert d["maxNm"] == pytest.approx(a["initialNm"] + 20.0 * d["nmPerPctLli"], rel=1e-5)
    assert d["chain"]
    # A plausible order of magnitude for a wound graphite anode: tens to
    # thousands of nanometres, not ångströms and not millimetres.
    assert 10.0 < d["nmPerPctLli"] < 1000.0


def test_the_thickness_series_is_the_fitted_term_rescaled(monkeypatch):
    _identified(monkeypatch)
    spec = _spec()
    sei = spec["series"]["seiPct"]
    nm = spec["series"]["seiThicknessNm"]
    a = spec["physical"]["film"]["assumptions"]
    per_pct = spec["physical"]["film"]["derivation"]["nmPerPctLli"]
    assert len(nm) == len(sei)
    for pct, thickness in zip(sei, nm):
        if pct is None:
            assert thickness is None, "a gap in the fit must stay a gap in the thickness"
        else:
            assert thickness == pytest.approx(a["initialNm"] + pct * per_pct)
    assert nm[-1] > 100.0, "no film growth in the fixture to draw"


def test_the_film_card_reports_nanometres(monkeypatch):
    _identified(monkeypatch)
    spec = _spec()
    card = next(p for p in spec["parts"] if p["id"] == "sei_film")
    nm = [v for v in spec["series"]["seiThicknessNm"] if v is not None]
    assert card["value"] == pytest.approx(nm[-1])
    assert card["unit"].startswith("nm")
    assert card["provenance"] == "derived", "a derived number is not a fitted one"
    assert card["series"] == spec["series"]["seiThicknessNm"]
    assert card["law"] == spec["physical"]["film"]["derivation"]["chain"]
    assert card["available"] is True


def test_the_disclosure_states_the_thickness_and_the_magnification_separately(monkeypatch):
    """Two questions, two sentences: how thick the film is, and what the layer a
    viewer is looking at is worth. Conflating them is what made the old drawing
    a metaphor."""
    _identified(monkeypatch)
    spec = _spec()
    film = spec["physical"]["film"]
    d, display = film["derivation"], film["display"]
    thickness = next(x for x in spec["disclosures"] if "SEI film's thickness is derived" in x)
    assert f"{d['nmPerPctLli']} nm" in thickness
    assert f"{d['anodeAreaCm2']} cm²" in thickness
    assert film["note"] in thickness
    drawn = next(x for x in spec["disclosures"] if "drawn SEI layer" in x)
    assert "magnification" in drawn
    assert display["note"] in drawn


def test_the_drawn_band_and_its_magnification_are_the_documents_own_numbers(monkeypatch):
    _identified(monkeypatch)
    spec = _spec()
    film = spec["physical"]["film"]
    d, display = film["derivation"], film["display"]
    assert display["drawnMaxMm"] > display["drawnMinMm"] > 0
    assert display["drawnMaxNm"] == pytest.approx(display["drawnMaxMm"] * 1e6, rel=1e-9)
    assert display["magnificationAtMaxX"] == pytest.approx(display["drawnMaxNm"] / d["maxNm"], rel=1e-3)
    # The magnification is printed, not left to be assumed away: it is far from
    # one, which is exactly why a reader has to be told it.
    assert display["magnificationAtMaxX"] > 10.0
    assert "magnification" in display["note"]
    assert f"{display['drawnMaxMm']} mm" in display["note"]


def test_the_film_scale_reads_nanometres_and_keeps_two_cells_comparable(monkeypatch):
    _identified(monkeypatch)
    young, old = _spec(n=60), _spec(n=200)
    scale = old["geometryScales"]["sei_film"]
    a, d = old["physical"]["film"]["assumptions"], old["physical"]["film"]["derivation"]
    assert scale["from"] == "series.seiThicknessNm"
    assert scale["unit"].startswith("nm")
    assert scale["displayMin"] == pytest.approx(a["initialNm"])
    assert scale["displayMax"] == pytest.approx(d["displayMaxNm"])
    assert d["displayMaxNm"] == pytest.approx(
        a["initialNm"] + d["displayMaxPctLli"] * d["nmPerPctLli"])
    assert d["chain"] in scale["note"] and "magnification" in scale["note"]
    # The scale is a share of initial capacity, not this record's own maximum:
    # a young cell and an old one draw to the same scale, so their films can be
    # compared by eye — the one comparison normalising to the record's own max
    # would destroy.
    assert young["geometryScales"]["sei_film"]["displayMax"] == pytest.approx(scale["displayMax"])
    assert d["maxNm"] > young["physical"]["film"]["derivation"]["maxNm"]


def test_an_unidentified_channel_withholds_a_thickness_instead_of_a_zero():
    """The chain's constants are still this cell's, but the quantity they scale
    is not identified — so the scale factor is reported and the thickness is
    not."""
    spec = _spec()
    film = spec["physical"]["film"]
    assert film["available"] is True
    assert film["identified"] is False
    assert film["derivation"]["maxNm"] is None
    assert film["derivation"]["nmPerPctLli"] > 0, "the chain is still this cell's"
    assert film["reason"] and "does not identify" in film["reason"]
    assert film["display"]["magnificationAtMaxX"] is None
    card = next(p for p in spec["parts"] if p["id"] == "sei_film")
    assert card["value"] is None and card["available"] is False
    assert spec["geometryScales"]["sei_film"]["from"] == "series.seiPct", (
        "with no thickness to scale, the mapping must stay on the fitted share"
    )
    # The series keeps the document's shape (like seiPct, which is emitted even
    # when the split is refused) but carries no growth a renderer could draw:
    # the gate is physics.splitIdentified, and the values are the formation film
    # plus a fit that put essentially no lithium in the film.
    nm = [v for v in spec["series"]["seiThicknessNm"] if v is not None]
    assert nm, "the series must be emitted with the shape a renderer indexes"
    assert max(nm) - film["assumptions"]["initialNm"] < 1.0
    refused = next(x for x in spec["disclosures"] if "would* be derived" in x)
    assert "does not identify" in refused


def test_a_stacked_prismatic_cell_has_no_winding_to_derive_a_thickness_from():
    import cell_scene
    film = cell_scene.film_model(
        PhysicalModel(FORM_FACTOR_PRISMATIC, "1.1 Ah prismatic").block(), 1.1, 20.0)
    assert film["available"] is False
    assert film["derivation"] is None and film["display"] is None
    assert "prismatic" in film["reason"]
    assert film["assumptions"], "the assumptions are stated even when they are not used"


def test_a_missing_capacity_is_a_refusal_not_a_thickness():
    import cell_scene
    physical = PhysicalModel(FORM_FACTOR_CYLINDRICAL, "cylindrical").block()
    film = cell_scene.film_model(physical, None, 20.0)
    assert film["available"] is False and film["derivation"] is None
    assert "capacity" in film["reason"]
    # …and the same is true of a document carrying no physical block at all.
    assert cell_scene.film_model(None, 2.0, 20.0)["available"] is False


# ---------------------------------------------------------------------------
# The top of the cell: declared millimetres, plain-language titles, two new parts
# ---------------------------------------------------------------------------

def test_the_top_assembly_is_declared_with_provenance():
    physical = PhysicalModel(FORM_FACTOR_CYLINDRICAL, "x").block()
    top = physical["topAssembly"]
    assert top is not None and len(top) == 10
    # Every figure is a positive millimetre size and every one is stated to be
    # typical for the format, not a datasheet row for this cell.
    assert all(isinstance(v, (int, float)) and v > 0 for v in top.values())
    assert "typical for the format" in physical["provenance"]["topAssembly"]
    # A prismatic cell carries no cylindrical top assembly.
    assert PhysicalModel(FORM_FACTOR_PRISMATIC, "x").block().get("topAssembly") is None


def test_the_schematic_list_no_longer_names_the_cap():
    """The cap/vent/terminal sizes used to be the renderer's guess; now they are
    the document's own declared millimetres, so they leave the schematic list."""
    physical = PhysicalModel(FORM_FACTOR_CYLINDRICAL, "x").block()
    assert not any("cap" in item for item in physical["schematic"])
    assert any("film" in item for item in physical["schematic"])


def test_every_part_card_carries_a_plain_language_title():
    spec = _spec()
    assert len(spec["parts"]) == 19
    for part in spec["parts"]:
        assert part["title"], f"{part['id']} has no plain-language title"
        # The title answers "what am I looking at"; the label stays anatomical.
        assert part["title"] != part["label"] or part["id"] == "can"


def test_the_two_new_parts_are_declared_unavailable_not_zero():
    spec = _spec()
    by_id = {p["id"]: p for p in spec["parts"]}
    for part_id in ("wrap", "crimp"):
        part = by_id[part_id]
        assert part["available"] is False
        assert part["value"] is None
        assert "no measurement" in part["unavailableReason"]


def test_the_mesh_part_registry_and_the_cards_stay_a_bijection():
    spec = _spec()
    assert tuple(p["id"] for p in spec["parts"]) == MESH_PART_IDS
    schema = json.load(open("docs/cell_scene.schema.json", encoding="utf-8"))
    enum = set(schema["properties"]["parts"]["items"]["properties"]["id"]["enum"])
    assert enum == set(MESH_PART_IDS)


def test_every_part_carries_a_category_from_the_declared_taxonomy():
    from cell_scene import _CATEGORY
    spec = _spec()
    taxonomy = {"SHELL", "INSULATION", "SEAL", "SAFETY", "TERMINAL",
                "WINDING", "ELECTRODE", "SEPARATOR", "ELECTROLYTE", "DEGRADATION"}
    assert len(spec["parts"]) == 19
    for part in spec["parts"]:
        assert part["category"] in taxonomy, part["id"]
        assert part["category"] == _CATEGORY[part["id"]]
