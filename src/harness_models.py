"""
Model catalogue, uploaded-model loading, and report summarization for the
Model Validation page (app/_pages/model_validation.py).

Why this is a src/ module and not page code
-------------------------------------------
Everything here is pure logic with no Streamlit import: which candidate
models the page can grade, which fleets it can grade them on — the deployment's
reloadable reference datasets AND this org's own persisted uploads — whether an
uploaded .py file is a usable model module, how a sealed bundle is packaged for
download, and how a harness report is flattened for rendering. Keeping it out
of the page means the repo's own unit tests can exercise the parts that can
actually be wrong — upload validation, the candidate registry, and the two
sealing modes — instead of only smoke-testing rendered HTML.

Two kinds of fleet, two kinds of seal
-------------------------------------
A reference fleet is public and reproducibly reloadable, so its sealed bundle
records the dataset's own loader and stays small. A tenant's upload is neither:
since 2026-09-17 the raw cycles are persisted at import time
(src/uploaded_store.py), which makes the fleet gradable, and a bundle over it
can either EMBED the cycles (verifiable by anyone holding the artifact — the
data travels) or point at this deployment's store (the data stays home, and the
verifier needs the store). fleet_plan() is the single place that decides which,
so the page cannot show one command and seal the other.

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

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from batlab.harness import seal_bundle
from batlab.harness.forecaster import SklearnForecaster
from batlab.harness.model_source import BUNDLE_MODEL_LOADER, ModuleSourceError
from batlab.harness.sandbox import (
    SandboxError,
    SandboxLimits,
    SandboxedModelFactory,
    describe_enforcement,
)
from batlab.validation.bundle_data import BUNDLE_CELLS_LOADER

__all__ = [
    "BUNDLE_LOADER_SPEC",
    "BUNDLE_MODEL_LOADER",
    "BUILTIN_MODELS",
    "FLEET_LOADER_SPEC",
    "ModelModuleError",
    "STARTER_MODULE_SOURCE",
    "UPLOADED_LOADER_SPEC",
    "available_fleets",
    "builtin_model",
    "catalogue_module_source",
    "describe_enforcement",
    "describe_module",
    "example_gate_expectations",
    "fleet_loader_record",
    "fleet_plan",
    "load_model_module",
    "model_bundle_source",
    "seal_zip_bytes",
    "summarise_report",
    "verify_command",
    "unsupported_upload_reason",
    "uploaded_fleet_entries",
]

# The loader a sealed bundle records when it CARRIES the raw cycles (see
# batlab/validation/bundle_data.py). Verifiable anywhere, by anyone holding the
# artifact — the only honest option for data that is not public.
BUNDLE_LOADER_SPEC = BUNDLE_CELLS_LOADER

# The loader a sealed bundle records for a tenant upload when the cycles are
# deliberately NOT embedded: it reads this deployment's own store
# (data/uploaded_fleets), so verification needs access to that store — and the
# seal then certifies the number against data the verifier must obtain from the
# publisher separately.
UPLOADED_LOADER_SPEC = "src.uploaded_store:load_uploaded_cell_data"

# The loader a sealed bundle records when it CARRIES the model it was published
# for, as source (see batlab/harness/model_source.py). Like the data path it is
# a separate module so a verifier needs only this repo; unlike the data path it
# is never applied implicitly, because importing a module runs it.
BUNDLE_MODEL_LOADER_SPEC = BUNDLE_MODEL_LOADER

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

# The catalogue models' configuration, as constants the factories AND the
# bundle-embedded source templates below both read. A number written twice is a
# number that eventually disagrees with itself — and here it would disagree
# between the model that was graded and the model the verifier re-runs.
RIDGE_ALPHA = 1.0
FOREST_ESTIMATORS = 200


def _ridge_factory(seed: int) -> Callable[[], Any]:
    def _make():
        from sklearn.linear_model import Ridge

        # scale=True: a linear model on unscaled features (cycle counts in the
        # hundreds next to ohms) is a strawman, not a baseline. The harness's
        # scaler keeps this candidate honest without pretending the platform
        # preprocesses models it does not own.
        return SklearnForecaster(Ridge(alpha=RIDGE_ALPHA, random_state=None), scale=True)

    return _make


def _forest_factory(seed: int) -> Callable[[], Any]:
    def _make():
        from sklearn.ensemble import RandomForestRegressor

        return SklearnForecaster(
            RandomForestRegressor(n_estimators=FOREST_ESTIMATORS, random_state=seed, n_jobs=1),
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
        # (kind="upload") The one catalogue entry whose model is not the
        # platform's own code and not reproducible from the repo — so it is the
        # only one whose embedding is optional: the source may be proprietary,
        # and "share the bundle" then means "share your model".
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


# ---------------------------------------------------------------------------
# The model a bundle can carry, as source
# ---------------------------------------------------------------------------

def catalogue_module_source(key: str, seed: int = 42) -> str:
    """The source of a catalogue model, as a module a verifier can run.

    Written for the two candidates that are neither the platform's own GBRT
    (reproducible from `batlab.harness.default_forecaster`) nor somebody's
    upload: a bundle published for a Ridge, a random forest or the mean
    baseline would otherwise record an identity the verifier cannot obtain,
    and its recompute check could never run.

    Every configuration number here comes from the same constant the factory
    reads (RIDGE_ALPHA, FOREST_ESTIMATORS) — a value written twice is a value
    that eventually disagrees — and tests/test_harness_models.py fits both
    models on identical data and requires identical predictions, so a template
    that drifts from the catalogue fails loudly instead of silently grading a
    different model.
    """
    entry = BUILTIN_MODELS.get(key) or {}
    if key == "ridge":
        body = (
            "from sklearn.linear_model import Ridge\n\n"
            "from batlab.harness import SklearnForecaster\n\n\n"
            "def make_model():\n"
            f"    return SklearnForecaster(Ridge(alpha={RIDGE_ALPHA!r}, random_state=None), scale=True)\n"
        )
    elif key == "random_forest":
        body = (
            "from sklearn.ensemble import RandomForestRegressor\n\n"
            "from batlab.harness import SklearnForecaster\n\n\n"
            "def make_model():\n"
            f"    return SklearnForecaster(RandomForestRegressor(n_estimators={FOREST_ESTIMATORS},\n"
            f"                                             random_state={int(seed)}, n_jobs=1),\n"
            "                             scale=False)\n"
        )
    elif key == "mean_baseline":
        body = (
            "from sklearn.dummy import DummyRegressor\n\n"
            "from batlab.harness import SklearnForecaster\n\n\n"
            "def make_model():\n"
            '    return SklearnForecaster(DummyRegressor(strategy="mean"), scale=False)\n'
        )
    else:
        raise KeyError(
            f"{key!r} has no bundle source template — only the catalogue models that are "
            "neither the platform default nor an upload need one"
        )

    header = (
        '"""The model this bundle was published for, as runnable source.\n\n'
        f'Written by this platform so a sealed result can be re-derived by whoever\n'
        f'receives the bundle. Importing this module EXECUTES it — read the file\n'
        f'first.\n\n'
        f'Catalogue entry: {key}\n'
        f'Label: {entry.get("label", key)}\n'
        f'Seed: {int(seed)}\n'
        '"""\n'
    )
    return header + body


def model_bundle_source(
    model_key: str,
    *,
    seed: int = 42,
    upload_source: "str | None" = None,
    include_upload: bool = True,
) -> dict:
    """What model source (if any) should travel inside a bundle for this run.

    Returns {source, entry_point, label, is_user_code, note, optional}:

    - the platform's own model → no source at all. It is reproducible from the
      repo (`batlab.harness.default_forecaster`), so embedding it would add
      bytes and no new capability;
    - a catalogue baseline (ridge / random forest / mean) → generated source,
      always embedded: the numbers are the platform's own and a 20-line module
      is what makes those bundles self-verifying;
    - an uploaded module → the user's text, embedded unless they turned it off.
      This is the only entry where the source may be someone's property, and
      the only one where the page offers a choice.

    `optional` is True only for the uploaded case: the page shows a checkbox
    for it and a statement for every kind.
    """
    entry = BUILTIN_MODELS.get(model_key)
    if entry is None:
        raise KeyError(f"unknown model key {model_key!r}")
    label = entry["label"]

    if model_key == "platform_default":
        return {
            "source": None,
            "entry_point": "make_model",
            "label": label,
            "is_user_code": False,
            "optional": False,
            "note": (
                "This is the platform's own configuration, so it does not need to "
                "travel: a verifier re-creates it from the repo "
                "(batlab.harness.default_forecaster, same GBRT_PARAMS)."
            ),
        }

    if model_key == "uploaded_module":
        wanted = bool(include_upload) and bool(upload_source)
        return {
            "source": upload_source if wanted else None,
            "entry_point": "make_model",
            "label": label,
            "is_user_code": True,
            "optional": True,
            "note": (
                "The bundle will contain your model's source, so the number can be "
                "re-derived by whoever you hand it to — and they will have your "
                "model code. Verifying it imports that file, which executes it; "
                "the bundle says so and the checker refuses a file whose bytes "
                "changed after sealing."
                if wanted else
                "The bundle will record what model was graded (its class and "
                "identity) but not its code, so it cannot be recomputed without "
                "the module — the seal and the data identity still hold."
            ),
        }

    return {
        "source": catalogue_module_source(model_key, seed),
        "entry_point": "make_model",
        "label": label,
        "is_user_code": False,
        "optional": False,
        "note": (
            "The bundle carries this baseline's source, so the recompute check can "
            "re-run the exact configuration that was graded instead of trusting "
            "the recorded class name."
        ),
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
    """{loader, loader_kwargs} for sealing a run on a REFERENCE fleet.

    Passed straight to seal_bundle(), which records it in replication.json so
    the download can be re-derived by someone who has neither the app nor the
    report — only the repo and the public dataset.
    """
    return {"loader": FLEET_LOADER_SPEC, "loader_kwargs": {"dataset": fleet_key}}


def uploaded_fleet_entries(org_id: int) -> list[dict]:
    """This org's own uploaded fleets, in the same shape as available_fleets().

    new-cell generalization and closing-form baselines were the only questions
    this page could ask. A tenant's own cells are the fleet most relevant to
    their decision, and from 2026-09-17 the raw cycles are persisted at import
    time (src/uploaded_store.py), so they can be graded and sealed like any
    other fleet.

    Every entry carries kind="uploaded", the upload_key, and the manifest's
    fingerprint summary. Unreadable or foreign-store entries never appear —
    uploaded_store.list_uploaded_fleets() already filters by org.
    """
    from uploaded_store import list_uploaded_fleets

    entries: list[dict] = []
    for manifest in list_uploaded_fleets(org_id):
        key = str(manifest.get("upload_key"))
        created = str(manifest.get("created_utc") or "")
        entries.append({
            "key": key,
            "kind": "uploaded",
            # No cell count in here: the picker's format_func appends it, and
            # "…3 cells — 3 cells" is exactly the kind of unread label that
            # makes a list look machine-generated.
            "label": (
                "Your uploaded fleet"
                + (f" ({created[:10]})" if created else "")
            ),
            "detail": (
                "The cells you uploaded to the Import page, reloaded from the "
                "raw cycles this deployment persisted for that upload "
                f"(key {key}). Nothing here is extrapolated from a reference "
                "fleet: these are your cells, your chemistry, your labels."
            ),
            "n_cells": manifest.get("n_cells"),
            "available": True,
            "unavailable_reason": None,
            "created_utc": created,
            "cell_ids": list(manifest.get("cell_ids") or []),
            "n_rows": manifest.get("n_rows"),
        })
    return entries


def fleet_plan(entry: dict, *, org_id: int, embed_data: "bool | None" = None) -> dict:
    """How to reload a fleet and how to seal a run on it.

    `entry` is one row from uploaded_fleet_entries() or available_fleets() —
    the page's picker already holds it, so this does not re-probe.

    Returns {key, kind, label, dataset, reload, loader, loader_kwargs,
    embed_data, embedded, privacy_note}. `reload()` returns
    {cell_id: cycles DataFrame}; `dataset` is the label the report and the
    bundle carry (and the downloaded file names are built from).

    `embed_data` only has a meaning for an uploaded fleet: True puts the raw
    cycles inside the sealed bundle (verifiable by anyone holding it, and the
    data travels with it), False records this deployment's own store loader
    instead (the data stays home, and verifying needs the store). It defaults
    to True — an artifact nobody else can check is the weaker claim, and the
    page states the trade before the run rather than after the download.
    """
    kind = entry.get("kind", "reference")
    key = str(entry["key"])

    if kind == "uploaded":
        from uploaded_store import load_uploaded_cell_data

        embedded = True if embed_data is None else bool(embed_data)

        def _reload() -> dict:
            return load_uploaded_cell_data(key, expected_org_id=org_id)

        if embedded:
            loader, loader_kwargs = BUNDLE_LOADER_SPEC, {"cells_dir": "cells"}
            privacy_note = (
                "The sealed bundle will contain your raw cycle tables, so your "
                "own model's numbers can be re-derived by whoever you hand the "
                "bundle to. Sharing the bundle shares the data in it."
            )
        else:
            loader = UPLOADED_LOADER_SPEC
            loader_kwargs = {"upload_key": key}
            privacy_note = (
                "The sealed bundle will NOT contain your data. It records this "
                "deployment's store as the data path, so verifying it needs "
                "access to that store — the seal still proves the numbers came "
                "from the digests recorded in it, and the digests travel."
            )
        return {
            "key": key,
            "kind": kind,
            "label": entry["label"],
            "dataset": f"uploaded-{key.rsplit('-', 1)[-1][:8]}",
            "reload": _reload,
            "loader": loader,
            "loader_kwargs": loader_kwargs,
            "embed_data": embedded,
            "privacy_note": privacy_note,
        }

    from experiment_registry import reload_reference_cell_data

    record = fleet_loader_record(key)
    return {
        "key": key,
        "kind": kind,
        "label": entry["label"],
        "dataset": key,
        "reload": lambda: reload_reference_cell_data(key),
        "loader": record["loader"],
        "loader_kwargs": record["loader_kwargs"],
        "embed_data": False,
        "privacy_note": (
            "The bundle records the public dataset's own loader, not your data: "
            "these are published reference cells."
        ),
    }


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


def _model_module_error(exc: ModuleSourceError) -> ModelModuleError:
    """The message the page shows for each way a module can fail to load.

    The mechanism lives in batlab (batlab.harness.model_source) so this page
    and the replication verifier can never disagree about what counts as a
    gradable model; the WORDING lives here, because the audiences differ —
    "your upload" at a web form, "the module this bundle carries" at a
    command line.
    """
    entry = exc.entry_point
    if exc.kind == "empty":
        return ModelModuleError("The uploaded file is empty.")
    if exc.kind == "syntax":
        detail = getattr(exc.exc, "msg", None) or "it does not compile"
        return ModelModuleError(
            f"The uploaded file is not valid Python — line {exc.lineno}: {detail}"
        )
    if exc.kind == "import":
        return ModelModuleError(
            f"Importing the uploaded module raised {type(exc.exc).__name__}: {exc.exc}"
        )
    if exc.kind == "missing_entry_point":
        return ModelModuleError(
            f"The uploaded module does not define {entry}(). Expected a "
            f"function {entry}() that returns a fresh, unfitted model — "
            "for example `def make_model(): return Ridge()`."
        )
    if exc.kind == "entry_raised":
        return ModelModuleError(f"{entry}() raised {type(exc.exc).__name__}: {exc.exc}")
    if exc.kind == "returned_none":
        return ModelModuleError(
            f"{entry}() returned None — it must return a model (an unfitted "
            "sklearn-style estimator, a batlab.harness adapter, or a factory "
            "returning one)."
        )
    return ModelModuleError(f"{entry}() returned a model the harness refuses: {exc.exc}")


def load_model_module(
    source: str,
    *,
    module_name: str = "batlab_uploaded_model",
    entry_point: str = "make_model",
    limits: Any = None,
    process_limits: Any = None,
) -> dict:
    """Turn uploaded module source into a gradable factory, IN A SANDBOX.

    The module is imported and fitted in a child process
    (batlab.harness.sandbox) with a policy against reaching outside it and
    with wall-clock/memory/CPU caps, and the factory returned here is the
    proxy the harness grades through. Nothing in this process ever imports the
    uploaded code: an upload can no longer read this app's files, open a
    socket, spawn a process, or take the page down with it.

    The compile-import-probe mechanism is still batlab's single implementation
    of "what counts as a gradable model" (batlab.harness.model_source, run
    inside the child), so "the page accepted it" and "the harness can grade
    it" remain one decision made once. What changed is where it is made.

    Raises ModelModuleError with a specific, user-actionable message for every
    failure mode: syntax error (with line), import-time exception, a module the
    policy refused, a missing entry point, an entry point that raises, a
    returned object the harness refuses (most importantly a PRE-FITTED
    estimator — see as_factory's docstring), and the caps themselves (a module
    that hangs, or asks for more memory/CPU than the sandbox allows).

    Returns {factory, module_name, path, entry_point, interval_capable,
    probe_class, callables, sandboxed, sandbox, close}. `factory` is what the
    page passes to validate_forecaster(); `close()` stops the child and should
    be called when the run is over; `sandbox` is what the deployment actually
    enforced (see sandbox.describe_enforcement), stated so the page can show it
    rather than promise it.
    """
    factory = None
    try:
        factory = SandboxedModelFactory(
            source, entry_point=entry_point, limits=limits or process_limits
        )
        factory.start()
    except SandboxError as exc:
        # Includes the constructor's own refusals (empty source), so every way
        # this can fail reaches the page as a ModelModuleError.
        if factory is not None:
            factory.close()
        raise _sandbox_error(exc) from exc
    return {
        "factory": factory,
        "module_name": module_name,
        "path": factory.module_path,
        "entry_point": entry_point,
        "interval_capable": factory.interval_capable,
        "probe_class": factory.probe_class,
        "callables": factory.callables,
        "sandboxed": True,
        "sandbox": factory.enforced,
        "limits": (limits or SandboxLimits()).to_dict(),
        "output": factory.output,
        "close": factory.close,
    }


def describe_module(module: dict) -> dict:
    """The JSON-safe part of load_model_module's answer.

    A run's result is kept in the page's session state so it survives Streamlit
    re-runs; a live sandbox process (and its factory) must not be kept alive by
    a dictionary sitting in a session, so the result carries this instead of
    the module itself.
    """
    return {
        key: module.get(key)
        for key in (
            "module_name", "path", "entry_point", "interval_capable",
            "probe_class", "callables", "sandboxed", "sandbox", "limits", "output",
        )
    }


def _sandbox_error(exc: "SandboxError") -> ModelModuleError:
    """The message the page shows when the SANDBOX refused to grade a module.

    Two families, because they need different sentences: a policy refusal (the
    module tried to reach outside the process, and the sandbox says what) and a
    limit (it hung, or asked for more resources than the sandbox grants). Both
    name the specific thing rather than blaming "your model".
    """
    if exc.kind == "policy":
        detail = str(exc).rstrip(".")
        return ModelModuleError(
            f"The sandbox refused to run this module — {detail}. Nothing outside "
            "the sandbox was touched, and the run was stopped before the model ran."
        )
    if exc.kind in _LIMIT_MESSAGES:
        return ModelModuleError(
            f"{_LIMIT_MESSAGES[exc.kind]} ({exc}) The model was stopped by the "
            "sandbox's caps, not by an error in the harness."
        )
    # Everything else is the module's own failure, in the wording this page has
    # always used for it (one place, two entry points: here and the verifier).
    # The child reports the structured fields it knows (which line, which entry
    # point), so the message keeps the detail the earlier in-process version had
    # instead of degrading to "it does not compile".
    return _model_module_error(
        ModuleSourceError(
            exc.kind,
            str(exc),
            entry_point=str(exc.extra.get("entry_point") or "make_model"),
            lineno=exc.extra.get("lineno"),
            exc=exc,
        )
    )


_LIMIT_MESSAGES = {
    "timeout": "The module did not finish inside the sandbox's wall-clock limit.",
    "memory": "The module exceeded the sandbox's memory limit.",
    "cpu": "The module exceeded the sandbox's CPU limit.",
    "too_large": "The module's inputs grew past the sandbox's message limit.",
    "spawn": "The sandbox process could not be started on this deployment.",
    "child_gone": "The sandbox process died while the module was running",
    "protocol": "The sandbox and the module stopped making sense to each other:",
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

This module is run in a SANDBOX: a separate process with an audit-hook policy
(no network, no subprocesses, no ctypes, writes confined to its own scratch
directory) and wall-clock/memory/CPU caps. The process it runs in cannot read
this app's files or take the page down, and the page lists exactly which of
those controls are in force on the deployment you are using. Read the file
first anyway: the code still runs, and it can still read what is inside the
sandbox.
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


def verify_command(result: dict, *, bundle_file_name: str = "sealed_bundle.zip") -> dict:
    """The shell command that verifies this run's sealed bundle, and whether it
    can re-derive the number.

    Pure, because the rule has three branches and getting one wrong is a lie in
    the form of a command:

    - the platform's own model → no `--model`, `--recompute` is valid (the
      verifier re-creates it from the repo);
    - a model the bundle CARRIES (a catalogue baseline, or an upload the user
      chose to include) → `--model <bundle model loader> --recompute`;
    - a non-default model the bundle does NOT carry → no `--model` and NO
      `--recompute`, because the check would fail on the number it was printed
      next to.

    Returns {command, carries_model, recomputes, unzips, reason}: `reason` is
    None when the command is complete, and the sentence to show beside it when
    the recompute step is deliberately left out.
    """
    record = result.get("loader") or {}
    loader_spec = record.get("loader") or "<module:function>"
    carries_model = bool(result.get("model_source_embedded"))
    recomputes = bool(result.get("default_model")) or carries_model
    model_flag = f" \\\n    --model {BUNDLE_MODEL_LOADER}" if carries_model else ""
    recompute_flag = " --recompute" if recomputes else ""

    if result.get("embed_data"):
        command = (
            f"unzip {bundle_file_name} -d bundle\n"
            f"python -m batlab.validation.replication bundle \\\n"
            f"    --loader {loader_spec}{model_flag}{recompute_flag}"
        )
    else:
        command = (
            f"python -m batlab.validation.replication <unzipped-bundle> \\\n"
            f"    --loader {loader_spec}{model_flag}{recompute_flag}"
        )

    reason = None
    if not recomputes:
        reason = (
            "**Re-deriving the number needs the model, so this command does "
            "not re-derive it.** The bundle records the model it was published "
            "for ("
            f"`{(result.get('summary') or {}).get('model_label')}`) but not its "
            "code; pass `--model module:factory` (a zero-argument factory "
            "returning a fresh, unfitted copy of it) to add the recompute "
            "check. Without the model you still get the checks that do not "
            "depend on its code — the seal, the data identity, and the "
            "environment — and the bundle records that identity, so a reader "
            "can see exactly what was claimed."
        )
    return {
        "command": command,
        "carries_model": carries_model,
        "recomputes": recomputes,
        "unzips": bool(result.get("embed_data")),
        "reason": reason,
    }


def seal_zip_bytes(
    report: dict,
    cell_data: dict,
    *,
    dataset: "str | None" = None,
    loader: Any = None,
    loader_kwargs: "dict | None" = None,
    embed_data: bool = False,
    cells_dir: str = "cells",
    model_source: "str | None" = None,
    model_entry_point: str = "make_model",
) -> bytes:
    """Seal a report into an in-memory zip a third party can verify.

    Zips exactly what seal_bundle() writes (benchmark.json, harness_report.json,
    and the sealed replication.json — plus `cells/` when the raw cycles are
    embedded and `model/` when the model's source is), so the downloaded
    artifact is the same one the CLI produces: the page does not have a second,
    weaker export path.
    Raises ValueError when the report has no LCO section, which is the same
    condition seal_bundle refuses for (its recompute check re-derives the LCO
    number).
    """
    with tempfile.TemporaryDirectory(prefix="batlab_seal_") as tmp:
        seal_bundle(report, cell_data, tmp, loader=loader, loader_kwargs=loader_kwargs,
                    dataset=dataset, embed_data=embed_data, cells_dir=cells_dir,
                    model_source=model_source, model_entry_point=model_entry_point)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            # rglob, not iterdir: an embedded-cells bundle carries a `cells/`
            # subdirectory, and a zip that silently dropped the evidence and
            # kept the report would export a bundle that cannot verify at all.
            root = Path(tmp)
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()
