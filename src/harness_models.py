"""
Model catalogue, uploaded-model loading, and report summarization for the
Model Validation page (app/_pages/model_validation.py).

Why this is a src/ module and not page code
-------------------------------------------
Everything here is pure logic with no Streamlit import: which candidate
models the page can grade, whether an uploaded .py file is a usable model
module, how a sealed bundle is packaged for download, and how a harness
report is flattened for rendering. Keeping it out of the page means the
repo's own unit tests can exercise the parts that can actually be wrong —
the upload validation and the candidate registry — instead of only
smoke-testing rendered HTML.

The page itself only picks options and draws. That split is the same one the
rest of this app follows (src/ holds the logic, app/_pages/ renders it).

On uploaded models, honestly
----------------------------
A model IS code: grading someone's forecaster on this platform's fleets means
executing their fit()/predict() inside this process. The supported upload is a
single .py file defining `make_model()`, because a text file can be *read*
before it is run — the page shows its source and requires an explicit
acknowledgement first. Pickled models are refused rather than supported: a
pickle executes on LOAD, so it cannot be inspected before it runs, and
"download this .pkl and open it" is a security habit this platform should not
teach. See unsupported_upload_reason() for the user-facing wording of each
refusal.

Nothing here sandboxes an uploaded module. That is stated on the page, not
glossed over: an uploaded module runs with this process's privileges.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from batlab.harness import as_factory, has_predict_interval, seal_bundle
from batlab.harness.forecaster import SklearnForecaster

__all__ = [
    "BUILTIN_MODELS",
    "FLEET_LOADER_SPEC",
    "ModelModuleError",
    "STARTER_MODULE_SOURCE",
    "available_fleets",
    "builtin_model",
    "example_gate_expectations",
    "fleet_loader_record",
    "load_model_module",
    "seal_zip_bytes",
    "summarise_report",
    "unsupported_upload_reason",
]

# Filename suffixes refused by the uploader, with the reason shown to the user.
REFUSED_UPLOAD_SUFFIXES = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".onnx", ".zip")


class ModelModuleError(ValueError):
    """An uploaded model module could not be turned into something gradable.

    Always carries a specific, user-actionable message (which line failed,
    which function is missing, why the returned object was refused) — the
    page shows it verbatim instead of a stack trace.
    """


# ---------------------------------------------------------------------------
# Candidate models the page can grade without any upload
# ---------------------------------------------------------------------------

def _ridge_factory(seed: int) -> Callable[[], Any]:
    def _make():
        from sklearn.linear_model import Ridge

        # scale=True: a linear model on unscaled features (cycle counts in the
        # hundreds next to ohms) is a strawman, not a baseline. The harness's
        # scaler keeps this candidate honest without pretending the platform
        # preprocesses models it does not own.
        return SklearnForecaster(Ridge(alpha=1.0, random_state=None), scale=True)

    return _make


def _forest_factory(seed: int) -> Callable[[], Any]:
    def _make():
        from sklearn.ensemble import RandomForestRegressor

        return SklearnForecaster(
            RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
            scale=False,
        )

    return _make


def _mean_factory(seed: int) -> Callable[[], Any]:
    def _make():
        from sklearn.dummy import DummyRegressor

        # The anti-model: predicts the training mean for every row. It exists
        # so a declared floor can be seen FAILING on the page — a gate nobody
        # has ever watched fail is indistinguishable from a gate that does
        # nothing.
        return SklearnForecaster(DummyRegressor(strategy="mean"), scale=False)

    return _make


# Ordered: the platform's own model first (the reference), then a cheap
# linear baseline, then a different model family, then the floor.
BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "platform_default": {
        "label": "Platform GBRT (this deployment's own model)",
        "detail": (
            "The configuration every number on the Benchmark page was measured "
            "under: gradient-boosted trees, one scaler shared across folds, "
            "leave-cell-out validated. Its interval section uses the platform's "
            "own Q10/Q90 quantile regressors plus conformal recalibration."
        ),
        "kind": "platform",
        "factory": None,          # None => pass model=None to the harness
        "interval_capable": True,
    },
    "ridge": {
        "label": "Ridge regression (linear baseline)",
        "detail": (
            "A scaled linear model. If a boosted-tree model cannot beat this, "
            "the complicated model is not earning its complexity."
        ),
        "kind": "builtin",
        "factory": _ridge_factory,
        "interval_capable": False,
    },
    "random_forest": {
        "label": "Random forest (different model family)",
        "detail": (
            "200 trees, no scaling needed. Same folds, same features, same "
            "label-provenance rules as the platform model."
        ),
        "kind": "builtin",
        "factory": _forest_factory,
        "interval_capable": False,
    },
    "mean_baseline": {
        "label": "Training-mean baseline (the floor)",
        "detail": (
            "Predicts the training mean for every row — no features used. "
            "Deliberately here so you can watch a declared floor fail."
        ),
        "kind": "builtin",
        "factory": _mean_factory,
        "interval_capable": False,
    },
    "uploaded_module": {
        "label": "Upload a model module (.py)",
        "detail": (
            "A single Python file defining make_model(), returning a fresh "
            "unfitted model. Read the file before you run it — an uploaded "
            "module executes inside this app's process."
        ),
        "kind": "upload",
        "factory": None,
        "interval_capable": None,  # discovered from the module
    },
}


def builtin_model(key: str, seed: int = 42) -> dict:
    """Resolve a catalogue key into what the harness needs.

    Returns {key, label, model, factory, interval_capable, detail} where
    `model` is what to pass to validate_forecaster() — None for the platform
    default (so the harness applies its own interval configuration) or a
    factory for everything else.
    """
    entry = BUILTIN_MODELS.get(key)
    if entry is None:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(BUILTIN_MODELS)}")
    factory_fn = entry.get("factory")
    factory = factory_fn(seed) if callable(factory_fn) else None
    return {
        "key": key,
        "label": entry["label"],
        "detail": entry["detail"],
        "kind": entry["kind"],
        "model": None if entry["kind"] == "platform" else factory,
        "factory": factory,
        "interval_capable": entry.get("interval_capable"),
    }


def example_gate_expectations() -> dict:
    """The floors the page offers as a starting point.

    Measured, not aspirational, where it matters: the platform's own reference
    runs clear the soh_r2 floor and the prospective floor, while the
    training-mean anti-model fails soh_r2 immediately — so the default
    expectations demonstrate both a PASS and a FAIL on the same fleet without
    being edited.
    """
    return {
        "schema_version": 1,
        "metrics": {
            "soh_r2": {"floor": 0.5, "ceiling": 1.0},
            "rul_r2": {"floor": 0.0, "allow_not_evaluable": True},
            "prospective_soh_r2": {"floor": -1.0, "allow_not_evaluable": True},
        },
    }


# ---------------------------------------------------------------------------
# Fleets this page can grade against
# ---------------------------------------------------------------------------

# The data path used for every run on this page, and recorded in the sealed
# bundle. It is the platform's own reference-fleet reloader — the same one
# Benchmark's replay/audit/divergence features use — rather than a raw dataset
# loader, because it returns the frames the app actually evaluates: the cycle
# tables AFTER enrich_cycles() adds the derived columns the feature builder
# reads. Recording `batlab.datasets.nasa:load_nasa_cells` instead would look
# more portable and verify WORSE: a third party loading the raw frame would
# digest a different table than the one that produced the number (cell_digest
# hashes every column), and the data-identity check would fail on a bundle that
# is in fact correct.
FLEET_LOADER_SPEC = "experiment_registry:reload_reference_cell_data"


def fleet_loader_record(fleet_key: str) -> dict:
    """{loader, loader_kwargs} for sealing a run on this fleet.

    Passed straight to seal_bundle(), which records it in replication.json so
    the download can be re-derived by someone who has neither the app nor the
    report — only the repo and the public dataset.
    """
    return {"loader": FLEET_LOADER_SPEC, "loader_kwargs": {"dataset": fleet_key}}


def _probe(name: str, probe_fn: Callable[[], Any]) -> tuple[Any, "str | None"]:
    """Run one availability probe, never letting it break the page."""
    try:
        return probe_fn(), None
    except Exception as exc:  # a missing dataset must not be a traceback
        return None, f"{type(exc).__name__}: {exc}"


def available_fleets() -> list[dict]:
    """Reference fleets this deployment can reload raw cycles for.

    Every entry is returned, available or not: "CALCE is not on this
    deployment" is information the user needs, and a fleet that quietly
    disappears from a picker reads as a bug rather than as a missing download.

    Each entry: {key, label, detail, n_cells (int|None), available,
    unavailable_reason (str|None)}. n_cells is None when knowing it cheaply
    would mean loading the fleet, which is the very cost this page defers
    until the user presses Run; the report states the real count afterwards.
    """
    fleets: list[dict] = []

    def _nasa() -> list[str]:
        from batlab.datasets.nasa import CELL_IDS, DATA_DIR

        return [c for c in CELL_IDS if (Path(DATA_DIR) / f"{c}_summary.csv").exists()]

    cached, error = _probe("nasa", _nasa)
    fleets.append({
        "key": "nasa",
        "label": "NASA PCoE — LiCoO₂, real measured (B0005–B0018)",
        "detail": (
            "Four real 18650 cells at 24 °C / 2 A. These cells reach measured "
            "end-of-life, so the RUL claim is scored on observed labels rather "
            "than on formula-extrapolated ones."
        ),
        "n_cells": len(cached) if cached else None,
        "available": bool(cached),
        "unavailable_reason": error or (
            "No parsed NASA CSVs in data/raw — run `python -m batlab.datasets.nasa` once."
        ),
    })

    def _synth() -> int:
        from data_loader import CELL_STRESS_PROFILES

        return len(CELL_STRESS_PROFILES)

    n_synth, error = _probe("synth", _synth)
    fleets.append({
        "key": "synth",
        "label": "Synthetic fleet — physics-informed (not measured)",
        "detail": (
            "A deterministic generator, always available, useful for a fast "
            "first run. No cell reaches measured end-of-life, so the harness "
            "withholds RUL here — deliberately, see the run's withheld list."
        ),
        "n_cells": n_synth,
        "available": n_synth is not None,
        "unavailable_reason": error,
    })

    # Both of these loaders DOWNLOAD when their cache is incomplete (Severson's
    # batch is ~2.9 GB, Zhu 2022 is a zip extraction), so availability is
    # deliberately not any_cached(): offering a fleet whose one click starts a
    # multi-gigabyte download is not a feature. A complete cache is required,
    # and the reason is stated on the page if it is missing.
    def _severson() -> list[str]:
        from batlab.datasets.severson import fully_cached_cell_ids

        return fully_cached_cell_ids()

    cached, error = _probe("severson", _severson)
    fleets.append({
        "key": "severson",
        "label": "Severson 2019 — LFP, real measured",
        "detail": (
            "The platform's largest real fleet and a different chemistry from "
            "NASA, so it is the fleet that shows what a model does on cells "
            "its features were not tuned for. A full run takes minutes, not "
            "seconds."
        ),
        "n_cells": len(cached) if cached else None,
        "available": bool(cached),
        "unavailable_reason": error or (
            "Severson is not fully cached here. Only a complete cache counts: "
            "a partial one makes the loader download the remaining batch "
            "(~2.9 GB), which this page will not start on a click — run "
            "`python -m batlab.datasets.severson` first."
        ),
    })

    def _zhu() -> list[str]:
        from batlab.datasets.zhu2022 import fully_cached_cell_stems

        return fully_cached_cell_stems()

    cached, error = _probe("zhu2022", _zhu)
    fleets.append({
        "key": "zhu2022",
        "label": "Zhu 2022 — NCM+NCA, real measured",
        "detail": "A third chemistry: nickel-rich cells, fast-charging protocols.",
        "n_cells": len(cached) if cached else None,
        "available": bool(cached),
        "unavailable_reason": error or (
            "Zhu 2022 summaries are only partly cached here, and a partial "
            "cache makes its loader download the dataset — so this page "
            "offers it only when the cache is complete."
        ),
    })

    def _calce() -> bool:
        from batlab.datasets.calce import any_cached

        return bool(any_cached())

    cached, error = _probe("calce", _calce)
    fleets.append({
        "key": "calce",
        "label": "CALCE CS2 — LiCoO₂ prismatic, real measured",
        "detail": "Prismatic cells; requires a manual dataset download.",
        "n_cells": None,
        "available": bool(cached),
        "unavailable_reason": error or (
            "CALCE needs a manual download on this deployment — see "
            "batlab/datasets/calce.py."
        ),
    })
    return fleets


# ---------------------------------------------------------------------------
# Uploaded model modules
# ---------------------------------------------------------------------------

def unsupported_upload_reason(filename: str) -> "str | None":
    """Why this upload is refused, or None when it is acceptable.

    Only plain .py modules are accepted. Serialized model formats are refused
    on purpose, and the reason is stated rather than implied: loading a pickle
    or a torch checkpoint executes code from the file before anything can
    inspect it, so it cannot be read-then-run the way a .py module can.
    """
    name = str(filename or "")
    lowered = name.lower()
    for suffix in REFUSED_UPLOAD_SUFFIXES:
        if lowered.endswith(suffix):
            return (
                f"{suffix} is not accepted. Loading a serialized model executes "
                "code from the file as part of loading it, so there is no point "
                "at which it can be inspected before it runs. Upload a .py module "
                "that defines make_model() instead — you can read that first."
            )
    if not lowered.endswith(".py"):
        return "Only .py model modules are accepted (a file defining make_model())."
    return None


def load_model_module(
    source: str,
    *,
    module_name: str = "batlab_uploaded_model",
    entry_point: str = "make_model",
) -> dict:
    """Turn uploaded module source into a gradable factory.

    Writes the source to a temporary file, imports it under `module_name`, and
    validates the entry point by CALLING it once and running the result
    through batlab.harness.as_factory() — the same normalization the harness
    uses, so "the page accepted it" and "the harness can grade it" cannot
    disagree.

    Raises ModelModuleError with a specific message for every failure mode:
    syntax error (with line), import-time exception, missing entry point,
    entry point that raises, or a returned object the harness refuses (most
    importantly a PRE-FITTED estimator — see as_factory's docstring for why
    that is a refusal and not a convenience).

    Returns {factory, module_name, path, entry_point, interval_capable,
    probe_class, callables}. `factory` is what the page passes to
    validate_forecaster().
    """
    if not isinstance(source, str) or not source.strip():
        raise ModelModuleError("The uploaded file is empty.")

    try:
        compile(source, f"<{module_name}>", "exec")
    except SyntaxError as exc:
        raise ModelModuleError(
            f"The uploaded file is not valid Python — line {exc.lineno}: {exc.msg}"
        ) from exc

    tmpdir = Path(tempfile.mkdtemp(prefix="batlab_model_"))
    path = tmpdir / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ModelModuleError("Could not load the uploaded file as a Python module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses/nested classes need this
    try:
        spec.loader.exec_module(module)
    except KeyboardInterrupt:  # a user interrupting the app is not a model error
        raise
    except BaseException as exc:
        # BaseException, not Exception: a module that calls sys.exit() would
        # otherwise take down the whole Streamlit script with SystemExit, and
        # the user would see a dead app instead of "your module exited".
        raise ModelModuleError(
            f"Importing the uploaded module raised {type(exc).__name__}: {exc}"
        ) from exc

    entry = getattr(module, entry_point, None)
    if not callable(entry):
        raise ModelModuleError(
            f"The uploaded module does not define {entry_point}(). Expected a "
            f"function {entry_point}() that returns a fresh, unfitted model — "
            "for example `def make_model(): return Ridge()`."
        )

    try:
        probe = entry()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # incl. SystemExit — see the exec_module note
        raise ModelModuleError(
            f"{entry_point}() raised {type(exc).__name__}: {exc}"
        ) from exc

    if probe is None:
        raise ModelModuleError(
            f"{entry_point}() returned None — it must return a model (an unfitted "
            "sklearn-style estimator, a batlab.harness adapter, or a factory "
            "returning one)."
        )

    try:
        factory = as_factory(probe)
    except Exception as exc:
        raise ModelModuleError(f"{entry_point}() returned a model the harness refuses: {exc}") from exc
    if factory is None:  # pragma: no cover - as_factory only returns None for None
        raise ModelModuleError(f"{entry_point}() returned nothing gradable.")

    return {
        "factory": factory,
        "module_name": module_name,
        "path": str(path),
        "entry_point": entry_point,
        "interval_capable": has_predict_interval(probe),
        "probe_class": type(probe).__name__,
        "callables": sorted(
            name for name in vars(module)
            if callable(getattr(module, name, None)) and not name.startswith("_")
        ),
    }


STARTER_MODULE_SOURCE = '''\
"""A starter model module for the platform's "Bring your own model" page.

make_model() must return a FRESH, UNFITTED model every time it is called.
The harness calls it once per fold AND once per target (SOH, RUL), so a
module that returns a cached global object would let one fold inherit another
fold's fit. The harness refuses pre-fitted estimators outright for exactly
that reason — it is the failure mode the whole validation layer exists to
prevent, so it is refused rather than quietly refitted.

Three shapes are accepted — use whichever fits your model:

  1. An sklearn-compatible estimator (fit(X, y) / predict(X)):

         from sklearn.linear_model import Ridge

         def make_model():
             return Ridge(alpha=1.0)

     X is a DataFrame of engineered features, y a Series (SOH in %, or RUL in
     cycles). No scaling is applied for you; wrap the estimator in
     SklearnForecaster(est, scale=True) — as make_model() below does — to fit
     one scaler on each fold's training rows before the estimator sees them.

  2. A CallableForecaster, for anything not sklearn-shaped (PyTorch, a
     hand-written ODE fit, a subprocess call):

         from batlab.harness import CallableForecaster

         def make_model():
             return CallableForecaster(my_fit, my_predict, label="my-ode-fit")
         #   my_fit(X, y) -> state        (whatever my_predict needs)
         #   my_predict(state, X) -> array of predictions

     Add interval_fn(state, X) -> (lower, upper) to make the model
     interval-capable; the harness then measures that interval's real
     coverage instead of building one from your residuals.

  3. A zero-argument factory returning any of the above.

Everything here runs inside the app's own process. There is no sandbox.
"""

from batlab.harness import SklearnForecaster
from sklearn.linear_model import Ridge


def make_model():
    """Return a fresh, unfitted model. Called once per fold and target."""
    return SklearnForecaster(Ridge(alpha=1.0), scale=True)
'''


# ---------------------------------------------------------------------------
# Report summary + sealed download
# ---------------------------------------------------------------------------

def summarise_report(report: dict) -> dict:
    """Flatten a harness report into what the page draws.

    Pure pass-through plus labels/formatting — no recomputation of any metric,
    so whatever the harness decided is what the page shows.
    """
    dataset = report.get("dataset") or {}
    model = report.get("model") or {}
    verdict = report.get("verdict") or {}
    gate = report.get("gate") or {}
    calibration = report.get("calibration") or {}
    identity = model.get("identity") or {}

    model_label = identity.get("class") or identity.get("source") or "unrecorded"
    if identity.get("source"):
        model_label = f"{identity['source']} ({model_label})"
    elif identity.get("factory"):
        model_label = f"{identity['factory']} → {model_label}"

    # Resolved ONCE: an absent gate section means NOT CHECKED, and the label
    # must follow the status rather than being looked up a second time from a
    # missing key (which rendered as the string "None" beside a status of
    # "not_checked" — exactly the kind of mislabelling this page exists to
    # prevent).
    status = str(gate.get("verdict") or "not_checked")
    status_labels = {
        "pass": "PASS — every declared floor/ceiling held",
        "fail": "FAIL — a declared metric expectation was violated",
        "not_checked": "NOT CHECKED — no floors were declared, so nothing was enforced",
    }

    return {
        "status": status,
        "status_label": status_labels.get(status, status),
        "summary": verdict.get("summary", ""),
        "claims": list(verdict.get("claims") or []),
        "withheld": list(verdict.get("withheld") or []),
        "model_label": model_label,
        "interval_capable": bool(model.get("interval_capable")),
        "interval_source": model.get("interval_source"),
        "dataset_name": dataset.get("name"),
        "dataset_sha256": ((dataset.get("fingerprint") or {}).get("dataset_sha256") or ""),
        "n_cells": dataset.get("n_cells"),
        "environment": dataset.get("environment") or {},
        "fingerprint": dataset.get("fingerprint") or {},
        "lint_ok": bool((report.get("leakage_lint") or {}).get("ok")),
        "lint_violations": list((report.get("leakage_lint") or {}).get("violations") or []),
        "provenance": report.get("label_provenance") or {},
        "lco": report.get("lco"),
        "baselines": report.get("baselines") or {},
        "calibration": calibration,
        "prospective": report.get("prospective") or {},
        "gate": gate,
        "config": report.get("config") or {},
    }


def seal_zip_bytes(
    report: dict,
    cell_data: dict,
    *,
    dataset: "str | None" = None,
    loader: Any = None,
    loader_kwargs: "dict | None" = None,
) -> bytes:
    """Seal a report into an in-memory zip a third party can verify.

    Zips exactly what seal_bundle() writes (benchmark.json, harness_report.json,
    and the sealed replication.json), so the downloaded artifact is the same
    one the CLI produces — the page does not have a second, weaker export path.
    Raises ValueError when the report has no LCO section, which is the same
    condition seal_bundle refuses for (its recompute check re-derives the LCO
    number).
    """
    with tempfile.TemporaryDirectory(prefix="batlab_seal_") as tmp:
        seal_bundle(report, cell_data, tmp, loader=loader, loader_kwargs=loader_kwargs,
                    dataset=dataset)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(tmp).iterdir()):
                if path.is_file():
                    archive.write(path, path.name)
    return buffer.getvalue()
