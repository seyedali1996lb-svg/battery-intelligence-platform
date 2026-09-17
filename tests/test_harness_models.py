"""
Unit tests for src/harness_models.py — the logic behind the app's
"Bring your own model" page (app/_pages/model_validation.py).

The page's rendering is covered separately by tests/test_model_validation_page.py
(AppTest, UI wiring). Everything here is the part that can actually be WRONG:
which uploads are refused and why, whether an uploaded module is validated the
same way the harness will judge it, whether a report is summarized without
inventing anything, and whether the fleet probes tell the truth about what this
deployment can reload without starting a download.
"""

import json
import sys
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _paths  # noqa: F401

from harness_models import (  # noqa: E402
    BUNDLE_LOADER_SPEC,
    BUILTIN_MODELS,
    FLEET_LOADER_SPEC,
    UPLOADED_LOADER_SPEC,
    ModelModuleError,
    STARTER_MODULE_SOURCE,
    available_fleets,
    builtin_model,
    example_gate_expectations,
    fleet_loader_record,
    fleet_plan,
    catalogue_module_source,
    load_model_module,
    model_bundle_source,
    seal_zip_bytes,
    summarise_report,
    unsupported_upload_reason,
    uploaded_fleet_entries,
    verify_command,
)


# ---------------------------------------------------------------------------
# Upload gate: what is refused, and the reason the user sees
# ---------------------------------------------------------------------------

def test_serialized_model_formats_are_refused_with_the_real_reason():
    """A pickle cannot be read before it runs — the refusal must say that."""
    for name in ("model.pkl", "model.PICKLE", "model.joblib", "net.pt", "net.pth", "m.onnx"):
        reason = unsupported_upload_reason(name)
        assert reason, name
        assert "executes code from the file as part of loading it" in reason
        assert ".py" in reason  # tells the user what IS accepted


def test_python_module_and_unknown_suffixes():
    assert unsupported_upload_reason("my_model.py") is None
    reason = unsupported_upload_reason("model.txt")
    assert reason and "make_model()" in reason


# ---------------------------------------------------------------------------
# Loading an uploaded module
# ---------------------------------------------------------------------------

def test_loads_an_sklearn_module_and_returns_a_factory():
    source = (
        "from sklearn.linear_model import Ridge\n"
        "def make_model():\n"
        "    return Ridge(alpha=1.0)\n"
    )
    module = load_model_module(source)
    assert module["entry_point"] == "make_model"
    assert module["interval_capable"] is False
    assert module["probe_class"] == "Ridge"
    assert "make_model" in module["callables"]
    # The factory must hand back a NEW object each call — one fold must never
    # inherit another fold's fit.
    assert module["factory"]() is not module["factory"]()
    assert pathlib.Path(module["path"]).exists()


def test_interval_capable_module_is_detected():
    source = (
        "from batlab.harness import CallableForecaster\n"
        "def _fit(X, y):\n"
        "    return float(y.mean())\n"
        "def _pred(state, X):\n"
        "    return [state] * len(X)\n"
        "def _interval(state, X):\n"
        "    return ([state - 1.0] * len(X), [state + 1.0] * len(X))\n"
        "def make_model():\n"
        "    return CallableForecaster(_fit, _pred, interval_fn=_interval)\n"
    )
    assert load_model_module(source)["interval_capable"] is True


def test_prefitted_model_is_refused_not_silently_accepted():
    """The one failure this whole harness exists to prevent, refused at the door.

    A module whose make_model() returns an already-fitted estimator would score
    beautifully on cells it memorized. as_factory() refuses it; the page must
    surface that refusal as a ModelModuleError rather than letting the run
    start.
    """
    source = (
        "from sklearn.linear_model import Ridge\n"
        "_MODEL = Ridge()\n"
        "_MODEL.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])\n"
        "def make_model():\n"
        "    return _MODEL\n"
    )
    with pytest.raises(ModelModuleError) as excinfo:
        load_model_module(source)
    assert "already fitted" in str(excinfo.value)


def test_named_failure_modes_carry_actionable_messages():
    with pytest.raises(ModelModuleError, match="empty"):
        load_model_module("   \n")

    with pytest.raises(ModelModuleError, match="not valid Python"):
        load_model_module("def make_model(:\n    pass\n")

    with pytest.raises(ModelModuleError, match="does not define make_model"):
        load_model_module("x = 1\n")

    with pytest.raises(ModelModuleError, match="raised ZeroDivisionError"):
        load_model_module("def make_model():\n    return 1 / 0\n")

    with pytest.raises(ModelModuleError, match="returned None"):
        load_model_module("def make_model():\n    return None\n")

    with pytest.raises(ModelModuleError, match="Importing the uploaded module raised"):
        load_model_module("import a_module_that_does_not_exist_anywhere\n")


def test_uploaded_module_errors_are_never_a_bare_traceback():
    """Every failure path raises ModelModuleError, so the page can show the text."""
    for source in ("", "def nope(", "import sys\n", "def make_model():\n    raise SystemExit(2)\n"):
        try:
            load_model_module(source)
        except ModelModuleError:
            pass


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def test_catalogue_defaults_to_the_platform_model():
    entry = builtin_model("platform_default")
    assert "model" in entry and entry["model"] is None
    assert entry["kind"] == "platform"


def test_every_builtin_model_resolves_to_a_fresh_factory():
    for key, spec in BUILTIN_MODELS.items():
        entry = builtin_model(key)
        assert entry["label"] == spec["label"]
        if spec["kind"] == "upload":
            assert entry["model"] is None  # the page supplies the module instead
            continue
        if spec["kind"] == "platform":
            assert entry["model"] is None
            continue
        first, second = entry["model"](), entry["model"]()
        assert first is not second


def test_unknown_model_key_is_an_error_not_a_fallback():
    """Silently substituting a different model would mislabel every number below."""
    with pytest.raises(KeyError):
        builtin_model("not_a_model")


# ---------------------------------------------------------------------------
# Fleets: availability must not start a download
# ---------------------------------------------------------------------------

def test_fleet_probe_reports_every_fleet_with_a_reason_when_unavailable():
    fleets = available_fleets()
    keys = [f["key"] for f in fleets]
    assert keys == ["nasa", "synth", "severson", "zhu2022", "calce"]
    for fleet in fleets:
        assert fleet["label"] and fleet["detail"]
        if fleet["available"]:
            assert fleet["n_cells"]
        else:
            assert fleet["unavailable_reason"]


def test_nasa_fleet_reports_the_tracked_cells():
    """The four NASA summaries are committed, so this deployment can reload them."""
    nasa = next(f for f in available_fleets() if f["key"] == "nasa")
    assert nasa["available"] is True
    assert nasa["n_cells"] == 4


def test_synthetic_fleet_is_always_available():
    synth = next(f for f in available_fleets() if f["key"] == "synth")
    assert synth["available"] is True
    assert synth["n_cells"] and synth["n_cells"] >= 2


def test_partially_cached_datasets_are_not_offered():
    """A partial cache makes these loaders DOWNLOAD — never offer that on a click.

    any_cached() answers "is at least one file here", which is not the question
    a picker needs. The probes use the loaders' full-cache guarantees, so a
    dehydrated severson/zhu cache must read as unavailable with that reason.
    """
    from batlab.datasets import severson, zhu2022

    sev = next(f for f in available_fleets() if f["key"] == "severson")
    if severson._all_cached():  # pragma: no cover - depends on the checkout
        assert sev["available"] is True
        assert sev["n_cells"] == len(severson.SEVERSON_CELL_IDS)
    else:
        assert sev["available"] is False
        assert "not fully cached" in sev["unavailable_reason"]

    zhu = next(f for f in available_fleets() if f["key"] == "zhu2022")
    if zhu2022._all_summaries_present():  # pragma: no cover - depends on the checkout
        assert zhu["available"] is True
    else:
        assert zhu["available"] is False


def test_fleet_loader_record_is_the_enriched_reference_reloader():
    """The recorded loader returns the SAME frames that were graded.

    cell_digest hashes every column, so recording a raw dataset loader (whose
    frames lack the enrichment step's columns) would make a correct bundle fail
    its own data-identity check.
    """
    record = fleet_loader_record("nasa")
    assert record["loader"] == "experiment_registry:reload_reference_cell_data"
    assert record["loader_kwargs"] == {"dataset": "nasa"}


# ---------------------------------------------------------------------------
# Report summary: pass-through, no invented numbers
# ---------------------------------------------------------------------------

_MINIMAL_REPORT = {
    "dataset": {
        "name": "nasa",
        "n_cells": 4,
        "fingerprint": {"dataset_sha256": "ab" * 32, "cell_digests": {"B0005": "cd" * 32}},
        "environment": {"python": "3.11.0"},
    },
    "model": {"identity": {"factory": "make_forecaster", "class": "Ridge"},
              "interval_capable": False, "interval_source": "the supplied model"},
    "config": {"seed": 42},
    "leakage_lint": {"ok": True, "violations": []},
    "label_provenance": {"observed_rows": 12, "extrapolated_rows": 3,
                         "observed_fraction": 0.8, "rul_reliable": True},
    "lco": {"soh_r2": 0.9, "soh_mae": 1.0, "rul_r2": 0.7, "rul_mae": 20.0,
            "rul_reliable": True, "per_cell": {}},
    "baselines": {"status": "computed", "lco_trend_r2": 0.5, "note": "why"},
    "calibration": {"status": "conformal_residual", "nominal": 0.8,
                    "calibrated_coverage": 0.79},
    "prospective": {"status": "computed", "soh_r2": 0.4, "rul_r2": -0.2},
    "gate": {"verdict": "pass", "results": [], "untracked": [], "observed": {"soh_r2": 0.9}},
    "verdict": {"status": "pass", "summary": "PASS — every declared floor/ceiling held",
                "claims": ["SOH on unseen cells: R² = 0.900"],
                "withheld": ["RUL: below the reliability floor"]},
}


def test_summarise_report_surfaces_verdict_claims_and_withheld():
    summary = summarise_report(_MINIMAL_REPORT)
    assert summary["status"] == "pass"
    assert summary["claims"] == ["SOH on unseen cells: R² = 0.900"]
    assert summary["withheld"] == ["RUL: below the reliability floor"]
    assert summary["dataset_name"] == "nasa"
    assert summary["dataset_sha256"] == "ab" * 32
    assert summary["lint_ok"] is True
    assert summary["model_label"] == "make_forecaster → Ridge"
    assert summary["interval_source"] == "the supplied model"


def test_summarise_report_does_not_invent_a_pass_without_a_gate():
    """No gate section means NOT CHECKED — never a pass, never a fail."""
    report = {k: v for k, v in _MINIMAL_REPORT.items() if k != "gate"}
    report["verdict"] = {"status": "not_checked", "claims": [], "withheld": []}
    summary = summarise_report(report)
    assert summary["status"] == "not_checked"
    assert "NOT CHECKED" in summary["status_label"]
    assert summary["claims"] == [] and summary["withheld"] == []


def test_summarise_report_tolerates_a_bare_report():
    """A report missing optional sections must not crash the page's render."""
    summary = summarise_report({"dataset": {}, "verdict": {}})
    assert summary["status"] == "not_checked"
    assert summary["claims"] == [] and summary["withheld"] == []
    assert summary["lco"] is None


# ---------------------------------------------------------------------------
# Gate expectations offered as defaults
# ---------------------------------------------------------------------------

def test_example_expectations_are_usable_and_enforced():
    """The defaults must be a real gate: parseable, and able to FAIL.

    They are chosen so the platform's own model clears them on NASA while a
    training-mean baseline does not — otherwise the page could never show a
    FAIL, and a gate nobody has watched fail is indistinguishable from a gate
    that does nothing.
    """
    from batlab.validation.metric_gate import evaluate_gate

    expectations = example_gate_expectations()
    assert json.loads(json.dumps(expectations)) == expectations
    assert expectations["metrics"]

    good = evaluate_gate({"soh_r2": 0.9, "rul_r2": 0.7, "prospective_soh_r2": 0.4},
                         expectations)
    assert good["verdict"] == "pass"

    bad = evaluate_gate({"soh_r2": 0.0, "rul_r2": None, "prospective_soh_r2": -3.0},
                        expectations)
    assert bad["verdict"] == "fail"
    failed = {r["name"] for r in bad["failures"]}
    assert "soh_r2" in failed


def test_example_expectations_allow_a_fleet_without_measured_eol_rows():
    """Synthetic fleets have no observed EOL rows: that is allowed, not a failure."""
    from batlab.validation.metric_gate import evaluate_gate

    result = evaluate_gate(
        {"soh_r2": 0.95, "rul_r2": None, "prospective_soh_r2": 0.5},
        example_gate_expectations(),
    )
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Sealing: refuses what it cannot verify
# ---------------------------------------------------------------------------

def test_sealing_needs_an_lco_section():
    """Sealing a report with no leave-cell-out number would ship a bundle whose
    recompute check has nothing to re-derive."""
    with pytest.raises(ValueError, match="LCO section"):
        seal_zip_bytes({"lco": None}, {"B0005": None}, dataset="nasa")


def test_replication_cli_honours_the_recorded_loader_kwargs():
    """The kwargs a bundle records must reach the loader on verification.

    The reader used to look only for a top-level "loader_kwargs" key that no
    writer has ever produced, so a loader whose signature REQUIRES an argument
    (experiment_registry:reload_reference_cell_data(dataset)) was called with
    none and raised TypeError — a correct bundle reported as unverifiable.
    """
    from batlab.validation import replication

    nested = {"loader": {"module": "m", "function": "f", "kwargs": {"dataset": "nasa"}}}
    assert replication._recorded_loader_kwargs(nested) == {"dataset": "nasa"}

    # Hand-written bundles using the top-level spelling still work.
    assert replication._recorded_loader_kwargs({"loader_kwargs": {"dataset": "synth"}}) == {
        "dataset": "synth"
    }
    # And a bundle that records no kwargs at all asks for none.
    assert replication._recorded_loader_kwargs({}) == {}
    assert replication._recorded_loader_kwargs({"loader": {"module": "m"}}) == {}


def test_sealed_bundle_reproduces_through_the_recorded_loader(tmp_path):
    """The strongest claim the page makes, actually executed.

    Seal a run the way the page does, then verify it the way a third party
    would: reload the fleet from the loader string RECORDED IN THE BUNDLE (not
    the frames this process happened to hold), and recompute the leave-cell-out
    number. If the recorded loader were the wrong data path, or its kwargs were
    dropped on the way back in, this fails — which is exactly the failure mode
    a "download the evidence" button is worthless without.
    """
    import importlib

    import experiment_registry as reg
    from batlab.harness import validate_forecaster
    from batlab.validation.replication import load_bundle, verify_bundle

    fleet = "synth"
    graded = reg.reload_reference_cell_data(fleet)
    entry = builtin_model("mean_baseline")
    report = validate_forecaster(
        graded, model=entry["model"], splits=("lco",), intervals=False, dataset=fleet
    )
    record = fleet_loader_record(fleet)
    blob = seal_zip_bytes(report, graded, dataset=fleet, loader=record["loader"],
                          loader_kwargs=record["loader_kwargs"])
    assert blob[:2] == b"PK"

    import zipfile
    with zipfile.ZipFile(__import__("io").BytesIO(blob)) as archive:
        archive.extractall(tmp_path)
    bundle = load_bundle(tmp_path)

    # Resolve the loader the way the verifier CLI does — from the record.
    from batlab.validation.replication import _recorded_loader_kwargs
    module_name, _, func_name = bundle["loader"]["module"], None, bundle["loader"]["function"]
    loader = getattr(importlib.import_module(module_name), func_name)
    reloaded = loader(**_recorded_loader_kwargs(bundle))

    result = verify_bundle(bundle, bundle_dir=tmp_path, cell_data=reloaded,
                           recompute=True, forecaster=entry["model"])
    assert result["verdict"] == "pass", result["checks"]
    checks = {c["name"]: c["status"] for c in result["checks"]}
    assert checks["seal"] == "pass"
    assert checks["data-identity"] == "pass"
    assert checks["recompute"] == "pass"

    # The environment check is allowed to WARN (a different machine is not a
    # failure) but must never silently disappear — and the bundle must record
    # the model it was published for, or recompute could not be attempted.
    assert "environment" in checks
    assert bundle["model"]["class"]


def test_starter_module_is_itself_a_valid_upload():
    """The template the page hands out must load — a broken starter wastes a
    user's first five minutes and teaches them the tool is broken."""
    module = load_model_module(STARTER_MODULE_SOURCE, module_name="batlab_starter_probe")
    assert module["factory"]() is not None
    assert "make_model" in module["callables"]


# ---------------------------------------------------------------------------
# Grading a tenant's OWN fleet: the catalogue row, and the two sealing modes
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated raw-cycle store — never a developer's real uploads."""
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


def test_an_orgs_uploads_appear_as_fleets(store):
    fleet = _tenant_fleet(2)
    store.save_uploaded_cell_data(1, "upload-0123456789abcdef0123", fleet,
                                  meta={"n_cells": 2})
    store.save_uploaded_cell_data(99, "upload-fedcba9876543210fedc", fleet)

    entries = uploaded_fleet_entries(1)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["key"] == "upload-0123456789abcdef0123"
    assert entry["kind"] == "uploaded"
    assert entry["available"] is True
    assert entry["n_cells"] == 2
    assert "2 cells" not in entry["label"]  # the picker appends the count once
    # Another org's upload is not offered, and its absence is not an error.
    assert all(e["key"] != "upload-fedcba9876543210fedc" for e in entries)


def test_no_uploads_means_no_extra_picker_rows(store):
    """The default page must be unchanged for a tenant that never uploaded."""
    assert uploaded_fleet_entries(1) == []


def test_fleet_plan_for_a_reference_fleet_records_the_public_loader():
    entry = next(f for f in available_fleets() if f["key"] == "synth")
    plan = fleet_plan(entry, org_id=1)

    assert plan["loader"] == FLEET_LOADER_SPEC
    assert plan["loader_kwargs"] == {"dataset": "synth"}
    assert plan["embed_data"] is False          # public data: nothing to embed
    assert plan["dataset"] == "synth"
    assert set(plan["reload"]()) >= {"Cell1"}


def test_fleet_plan_for_an_upload_defaults_to_embedding_and_can_be_turned_off(store):
    key = "upload-abcabcabcabcabcabcab"
    store.save_uploaded_cell_data(1, key, _tenant_fleet(2))
    entry = uploaded_fleet_entries(1)[0]

    embedded = fleet_plan(entry, org_id=1, embed_data=True)
    assert embedded["loader"] == BUNDLE_LOADER_SPEC
    assert embedded["loader_kwargs"] == {"cells_dir": "cells"}
    assert embedded["embed_data"] is True
    assert "contain your raw cycle tables" in embedded["privacy_note"]
    assert embedded["dataset"].startswith("uploaded-")
    assert set(embedded["reload"]()) == {"UP1", "UP2"}

    # The default is to embed — an artifact the recipient cannot check is the
    # weaker claim, so it is not the default.
    assert fleet_plan(entry, org_id=1)["embed_data"] is True

    kept_home = fleet_plan(entry, org_id=1, embed_data=False)
    assert kept_home["loader"] == UPLOADED_LOADER_SPEC
    assert kept_home["loader_kwargs"] == {"upload_key": key}
    assert kept_home["embed_data"] is False
    assert "will NOT contain your data" in kept_home["privacy_note"]


def test_a_plan_for_another_orgs_upload_fails_at_reload_naming_the_org(store):
    key = "upload-11111111111111111111"
    store.save_uploaded_cell_data(7, key, _tenant_fleet(2))

    plan = fleet_plan({"key": key, "kind": "uploaded", "label": "theirs"}, org_id=1)
    with pytest.raises(Exception) as exc:
        plan["reload"]()
    assert "another organization" in str(exc.value)


def test_a_plan_for_a_deleted_upload_says_so_instead_of_returning_nothing(store):
    key = "upload-22222222222222222222"
    store.save_uploaded_cell_data(1, key, _tenant_fleet(2))
    plan = fleet_plan({"key": key, "kind": "uploaded", "label": "mine"}, org_id=1)
    store.clear_uploaded_cell_data(key)

    with pytest.raises(Exception) as exc:
        plan["reload"]()
    assert "No persisted raw cycles" in str(exc.value)


# ---------------------------------------------------------------------------
# The model that travels with the evidence
# ---------------------------------------------------------------------------

def test_generated_catalogue_source_grades_the_same_model():
    """A template that drifted from the catalogue would recompute a different
    number from an equally plausible model — so both are fitted and compared.

    The templates are text and the factories are closures, so nothing but this
    test keeps them in step.
    """
    import numpy as np
    import pandas as pd

    from batlab.harness.forecaster import forecaster_identity

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 1.5 - X["b"])

    for key, seed in (("ridge", 42), ("random_forest", 7), ("mean_baseline", 42)):
        from_bundle = load_model_module(catalogue_module_source(key, seed))["factory"]()
        from_catalogue = builtin_model(key, seed=seed)["model"]()

        assert forecaster_identity(from_bundle) == forecaster_identity(from_catalogue), key
        from_bundle.fit(X, y)
        from_catalogue.fit(X, y)
        assert np.array_equal(from_bundle.predict(X), from_catalogue.predict(X)), key


def test_catalogue_source_refuses_keys_it_has_no_template_for():
    with pytest.raises(KeyError):
        catalogue_module_source("uploaded_module")
    with pytest.raises(KeyError):
        catalogue_module_source("platform_default")


def test_model_bundle_source_travels_for_the_right_reasons():
    platform = model_bundle_source("platform_default")
    assert platform["source"] is None
    assert platform["optional"] is False
    # The reason has to name where a verifier gets it instead of the bundle.
    assert "default_forecaster" in platform["note"]

    ridge = model_bundle_source("ridge", seed=42)
    assert ridge["source"] and "Ridge" in ridge["source"]
    assert ridge["entry_point"] == "make_model"
    assert ridge["optional"] is False  # a 20-line baseline is not a secret

    # A seed is a configuration value: the forest's template has to carry the
    # one that was actually graded, or the recompute is a different model.
    assert "random_state=7" in model_bundle_source("random_forest", seed=7)["source"]

    upload = "def make_model():\n    return None\n"
    carried = model_bundle_source("uploaded_module", upload_source=upload,
                                  include_upload=True)
    assert carried["source"] == upload
    assert carried["optional"] is True
    assert "will contain your model's source" in carried["note"]

    kept_home = model_bundle_source("uploaded_module", upload_source=upload,
                                    include_upload=False)
    assert kept_home["source"] is None
    assert "not its code" in kept_home["note"]
    # No upload at all is the same answer as declining, never a crash.
    assert model_bundle_source("uploaded_module", upload_source=None)["source"] is None


def _result_for(model_key, *, embedded, default, carried, embed_data, loader):
    """The slice of a run result the printed command is decided from."""
    return {
        "default_model": default,
        "model_source_embedded": carried,
        "embed_data": embed_data,
        "loader": {"loader": loader, "loader_kwargs": {}},
        "summary": {"model_label": model_key},
    }


def test_the_printed_command_is_only_what_can_actually_be_run():
    """Three branches, and the two ways of getting one wrong are both lies.

    --recompute next to a model the verifier cannot obtain fails on the number
    it was printed beside; a missing --recompute when the model DOES travel
    undersells an artifact that re-derives its own number.
    """
    platform = verify_command(_result_for(
        "platform default", embedded=True, default=True, carried=False,
        embed_data=False, loader="batlab.datasets.nasa:load_nasa_cells"))
    assert platform["recomputes"] is True and platform["carries_model"] is False
    assert "--recompute" in platform["command"]
    assert "--model" not in platform["command"]
    assert platform["reason"] is None

    catalogue = verify_command(_result_for(
        "ridge", embedded=True, default=False, carried=True,
        embed_data=False, loader="experiment_registry:reload_reference_cell_data"))
    assert "--recompute" in catalogue["command"]
    assert "--model batlab.harness.model_source:load_bundle_model" in catalogue["command"]
    assert catalogue["reason"] is None

    withheld = verify_command(_result_for(
        "uploaded module (mine.py)", embedded=False, default=False, carried=False,
        embed_data=False, loader="src.uploaded_store:load_uploaded_cell_data"))
    assert "--recompute" not in withheld["command"]
    assert "--model" not in withheld["command"]
    assert "Re-deriving the number needs the model" in withheld["reason"]
    # The sentence names what WAS graded, so a reader can see the claim even
    # without the code that made it.
    assert "uploaded module (mine.py)" in withheld["reason"]


def test_the_printed_command_unzips_exactly_when_the_data_is_inside():
    """Embedded data ⇒ the command must unpack the bundle first, and name the
    same file the download button offers; kept-home data ⇒ it must not pretend
    a directory it never created exists."""
    embedded = verify_command(
        _result_for("ridge", embedded=True, default=False, carried=True,
                    embed_data=True,
                    loader="batlab.validation.bundle_data:load_bundle_cells"),
        bundle_file_name="sealed_bundle_upload-a.csv.zip",
    )
    assert embedded["unzips"] is True
    lines = embedded["command"].splitlines()
    assert lines[0] == "unzip sealed_bundle_upload-a.csv.zip -d bundle"
    # Shell line continuations, one trailing backslash each: pasted as a block,
    # the command is a single command rather than three.
    assert lines[1] == "python -m batlab.validation.replication bundle " + chr(92)
    assert lines[2] == "    --loader batlab.validation.bundle_data:load_bundle_cells " + chr(92)
    assert lines[3].endswith("--recompute")
    assert "<unzipped-bundle>" not in embedded["command"]

    kept_home = verify_command(
        _result_for("ridge", embedded=True, default=False, carried=True,
                    embed_data=False,
                    loader="batlab.datasets.nasa:load_nasa_cells"))
    assert kept_home["unzips"] is False
    assert kept_home["command"].startswith("python -m batlab.validation.replication <unzipped-bundle>")
    assert not kept_home["command"].startswith("unzip")


def test_seal_zip_bytes_carries_the_model_when_asked(store, tmp_path):
    """The page's own export path, with a model inside it."""
    import io
    import zipfile

    from batlab.harness import validate_forecaster
    from batlab.validation.replication import (
        _load_model, _recorded_loader_kwargs, _recorded_model_kwargs, load_bundle,
        verify_bundle,
    )

    key = "upload-44444444444444444444"
    fleet = _tenant_fleet(3)
    store.save_uploaded_cell_data(1, key, fleet)
    entry = uploaded_fleet_entries(1)[0]
    plan = fleet_plan(entry, org_id=1, embed_data=True)
    graded = plan["reload"]()

    source_plan = model_bundle_source("mean_baseline")
    report = validate_forecaster(graded, model=builtin_model("mean_baseline")["model"],
                                 splits=("lco",), intervals=False, dataset=plan["dataset"])
    blob = seal_zip_bytes(report, graded, dataset=plan["dataset"], loader=plan["loader"],
                          loader_kwargs=plan["loader_kwargs"], embed_data=plan["embed_data"],
                          model_source=source_plan["source"],
                          model_entry_point=source_plan["entry_point"])

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
    assert "model/module.py" in names
    assert "model/index.json" in names

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(tmp_path)
    bundle = load_bundle(tmp_path)
    # The bundle names its own model loader, exactly as the printed --model does.
    loader = bundle["model_source"]["loader"]
    assert loader["module"] == "batlab.harness.model_source"
    assert loader["function"] == "load_bundle_model"
    factory = _load_model(f"{loader['module']}:{loader['function']}", bundle_dir=tmp_path,
                          recorded_kwargs=_recorded_model_kwargs(bundle))

    import importlib

    data_loader = bundle["loader"]
    reloaded = getattr(importlib.import_module(data_loader["module"]),
                       data_loader["function"])(**_recorded_loader_kwargs(bundle), bundle_dir=tmp_path)

    # Neither the data nor the model comes from this deployment: both come out
    # of the zip, together, which is the whole point of carrying the source.
    result = verify_bundle(bundle, bundle_dir=tmp_path, cell_data=reloaded,
                           recompute=True, forecaster=factory)
    assert result["verdict"] == "pass", result["checks"]


def test_an_uploaded_fleet_seals_into_a_bundle_that_verifies_on_its_own(store, tmp_path):
    """The end the whole change exists for: a tenant's own evidence, checkable
    by someone who was given only the zip.

    Sealed the way the page does (plan -> seal_zip_bytes), verified the way a
    third party would (extract, resolve the RECORDED loader, recompute). The
    data path is inside the bundle, so nothing here depends on this deployment.
    """
    import importlib
    import zipfile

    from batlab.harness import validate_forecaster
    from batlab.validation.replication import _recorded_loader_kwargs, load_bundle, verify_bundle

    key = "upload-33333333333333333333"
    fleet = _tenant_fleet(3)
    store.save_uploaded_cell_data(1, key, fleet)
    entry = uploaded_fleet_entries(1)[0]
    plan = fleet_plan(entry, org_id=1, embed_data=True)

    graded = plan["reload"]()
    model_entry = builtin_model("mean_baseline")
    report = validate_forecaster(graded, model=model_entry["model"], splits=("lco",),
                                 intervals=False, dataset=plan["dataset"])
    blob = seal_zip_bytes(report, graded, dataset=plan["dataset"], loader=plan["loader"],
                          loader_kwargs=plan["loader_kwargs"], embed_data=plan["embed_data"])
    assert blob[:2] == b"PK"

    import io

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        archive.extractall(tmp_path)
    assert any(n.startswith("cells/") and n.endswith(".csv") for n in names)

    bundle = load_bundle(tmp_path)
    module_name = bundle["loader"]["module"]
    func_name = bundle["loader"]["function"]
    loader = getattr(importlib.import_module(module_name), func_name)
    kwargs = _recorded_loader_kwargs(bundle)
    assert "cells_dir" in kwargs
    # The verifier CLI injects its own bundle path for a loader that declares
    # one; do the same here rather than relying on the working directory.
    reloaded = loader(**kwargs, bundle_dir=tmp_path)

    result = verify_bundle(bundle, bundle_dir=tmp_path, cell_data=reloaded,
                           recompute=True, forecaster=model_entry["model"])
    assert result["verdict"] == "pass", result["checks"]
    assert bundle["embedded_cells"]["n_cells"] == 3
    # The bundles's own digests describe the frames that came back out of it.
    from batlab.validation.fingerprints import cell_digest
    for cell_id, digest in bundle["cell_digests"].items():
        assert cell_digest(reloaded[cell_id]) == digest
