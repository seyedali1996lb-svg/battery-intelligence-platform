"""
A model as readable source — written into a bundle, and loaded back out.

Why this exists
---------------
`embed_data=True` closed half of a gap: a tenant's cycles could travel inside
the sealed bundle, so the number re-derived from the artifact. The other half
was still missing. A bundle published for anything but the platform's own GBRT
recorded only an *identity* (`SklearnForecaster`, a class name) and no way to
obtain it, so the recompute check — the check that actually re-derives the
headline metric — could not run. The page told the truth about it and printed a
subset; the honest fix is to let the model travel too.

So the bundle can now carry the module that produced the number, as source,
under `model/`, in the same way it carries the cycles:

    unzip sealed_bundle.zip -d bundle
    python -m batlab.validation.replication bundle \\
        --loader batlab.validation.bundle_data:load_bundle_cells \\
        --model batlab.harness.model_source:load_bundle_model --recompute

Read before run, and not by accident
------------------------------------
Importing a Python module executes it. That is why the app accepts `.py` and
refuses pickles (a pickle executes as part of *loading* it, so there is no
moment at which it can be inspected first) — and it is exactly why a
bundle-carried model is shipped as plain text and **never loaded implicitly**:
the verifier has to name `load_bundle_model` on the command line, which is the
deliberate act, and `model/index.json` + the bundle's seal make the bytes they
are about to run checkable first. `load_bundle_model()` verifies the file
against its recorded digest BEFORE importing it, so a model edited after
sealing is refused rather than executed.

The source→factory mechanism lives here once
--------------------------------------------
`import_module_source()` is the compile-import-probe path both callers use: the
app's uploader — inside its sandbox child, via
`src/harness_models.load_model_module` — and this module's bundle loader. They disagree about wording — "your upload" versus "the module this
bundle carries" — but never about what counts as a gradable model, which is the
part that has to match. Two implementations of that would eventually disagree,
and the disagreement would show up as a bundle that the page could grade and
the verifier could not load.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from batlab.harness.forecaster import as_factory, has_predict_interval

__all__ = [
    "BUNDLE_MODEL_LOADER",
    "BUNDLE_MODEL_SCHEMA",
    "BUNDLE_MODEL_SCHEMA_VERSION",
    "MODEL_INDEX_NAME",
    "ModuleSourceError",
    "import_module_source",
    "load_bundle_model",
    "write_bundle_model",
]

BUNDLE_MODEL_SCHEMA = "batlab-bundle-model"
BUNDLE_MODEL_SCHEMA_VERSION = 1
MODEL_INDEX_NAME = "index.json"

# The 'module:function' spec a bundle records when it carries its model's
# source. Recorded rather than hard-coded at each call site so the writer, the
# page's printed command and these docs cannot spell it three ways.
BUNDLE_MODEL_LOADER = "batlab.harness.model_source:load_bundle_model"

DEFAULT_ENTRY_POINT = "make_model"


class ModuleSourceError(ValueError):
    """Module source that cannot be turned into a gradable factory.

    `kind` is the stable part — branch on it, never on the message, which is
    phrased for whoever is reading it (an uploader at a web form, or a reviewer
    verifying a bundle):

      empty | syntax | import | missing_entry_point | entry_raised |
      returned_none | not_gradable
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        entry_point: str = DEFAULT_ENTRY_POINT,
        lineno: "int | None" = None,
        exc: "BaseException | None" = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.entry_point = entry_point
        self.lineno = lineno
        self.exc = exc


def import_module_source(
    source: str,
    *,
    module_name: str = "batlab_model_module",
    entry_point: str = DEFAULT_ENTRY_POINT,
    path: "str | Path | None" = None,
) -> dict:
    """Compile `source`, import it, and validate its entry point.

    Writes the source to a temporary file and imports it (or imports `path`
    directly when one is given — the bundle loader passes the file it has
    already digest-checked), calls the entry point once, and normalizes the
    result with `as_factory()` — the same normalization the harness uses to
    decide what it can grade, so acceptance here and gradability there cannot
    disagree.

    Raises ModuleSourceError with a `kind` for every failure mode, including a
    module that calls `sys.exit()` (BaseException, not Exception: a SystemExit
    would otherwise take down the host process — a Streamlit script, or the
    verifier's CLI — instead of reporting itself).

    Returns {module, module_name, path, entry_point, probe, factory,
    interval_capable, probe_class, callables}.
    """
    if not isinstance(source, str) or not source.strip():
        raise ModuleSourceError("empty", "the module source is empty")

    try:
        compile(source, f"<{module_name}>", "exec")
    except SyntaxError as exc:
        raise ModuleSourceError(
            "syntax",
            f"the module does not compile — line {exc.lineno}: {exc.msg}",
            entry_point=entry_point,
            lineno=exc.lineno,
            exc=exc,
        ) from exc

    if path is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="batlab_model_"))
        path = tmpdir / f"{module_name}.py"
        # write_bytes: no platform newline translation, so what is imported is
        # byte-for-byte what the digest was taken over.
        path.write_bytes(source.encode("utf-8"))
    path = Path(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ModuleSourceError("import", "could not be loaded as a Python module", entry_point=entry_point)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses/nested classes need this
    try:
        spec.loader.exec_module(module)
    except KeyboardInterrupt:  # a user interrupting the host is not a model error
        raise
    except BaseException as exc:  # incl. SystemExit — see the docstring
        raise ModuleSourceError(
            "import",
            f"importing it raised {type(exc).__name__}: {exc}",
            entry_point=entry_point,
            exc=exc,
        ) from exc

    entry = getattr(module, entry_point, None)
    if not callable(entry):
        raise ModuleSourceError(
            "missing_entry_point",
            f"it does not define {entry_point}()",
            entry_point=entry_point,
        )

    try:
        probe = entry()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # incl. SystemExit — see the docstring
        raise ModuleSourceError(
            "entry_raised",
            f"{entry_point}() raised {type(exc).__name__}: {exc}",
            entry_point=entry_point,
            exc=exc,
        ) from exc

    if probe is None:
        raise ModuleSourceError(
            "returned_none",
            f"{entry_point}() returned None",
            entry_point=entry_point,
        )

    try:
        factory = as_factory(probe)
    except Exception as exc:
        raise ModuleSourceError(
            "not_gradable",
            f"{entry_point}() returned a model the harness refuses: {exc}",
            entry_point=entry_point,
            exc=exc,
        ) from exc
    if factory is None:  # pragma: no cover - as_factory only returns None for None
        raise ModuleSourceError("not_gradable", f"{entry_point}() returned nothing gradable",
                                entry_point=entry_point)

    return {
        "module": module,
        "module_name": module_name,
        "path": str(path),
        "entry_point": entry_point,
        "probe": probe,
        "factory": factory,
        "interval_capable": has_predict_interval(probe),
        "probe_class": type(probe).__name__,
        "callables": sorted(
            name for name in vars(module)
            if callable(getattr(module, name, None)) and not name.startswith("_")
        ),
    }


def write_bundle_model(
    source: str,
    out_dir: "str | Path",
    *,
    file_name: str = "module.py",
    entry_point: str = DEFAULT_ENTRY_POINT,
    label: "str | None" = None,
) -> dict:
    """Write `source` into a bundle directory, with an index that pins it.

    Deliberately does NOT import the source: writing an artifact must not run
    the code in it, and the publisher has already graded with this module, so a
    broken one cannot reach here in the normal path. The index records the
    file's own sha256 (checkable with `sha256sum`) and what `load_bundle_model`
    will require — so a model edited after sealing is refused before it runs.

    `file_name` must be a plain file name: it becomes a path inside the
    artifact, and a name that walks out of the bundle is a bug in the caller,
    not something to resolve quietly.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("model source is empty — nothing to embed")
    if Path(file_name).name != file_name or file_name in (".", ".."):
        raise ValueError(f"{file_name!r} is not a plain file name")
    if not str(entry_point).strip():
        raise ValueError("entry_point must be a function name")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = source.encode("utf-8")
    (out / file_name).write_bytes(raw)

    index = {
        "schema": BUNDLE_MODEL_SCHEMA,
        "schema_version": BUNDLE_MODEL_SCHEMA_VERSION,
        "file": file_name,
        "entry_point": str(entry_point),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_lines": len(source.splitlines()),
        "label": label,
        "note": (
            "The model module this bundle was published for, as source. It is "
            "NOT loaded automatically: importing a Python module executes it, "
            "so read this file before you run it — "
            f"--model {BUNDLE_MODEL_LOADER} does exactly that, after checking "
            "these bytes against this digest."
        ),
    }
    (out / MODEL_INDEX_NAME).write_bytes((json.dumps(index, indent=2) + "\n").encode("utf-8"))
    return index


def _resolve_model_path(source_file: str, bundle_dir: "str | Path | None") -> Path:
    """Resolve a recorded model path inside a bundle, refusing anything else.

    The traversal check is the same one `load_bundle_cells` applies to cell
    files: a bundle is attacker-controlled input at verification time, and an
    index that names `../../etc/passwd` (or writes to it) must be refused
    rather than trusted.
    """
    relative = Path(source_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"the recorded model path {source_file!r} is not inside the bundle — refusing to read it"
        )
    base = Path(bundle_dir) if bundle_dir is not None else Path(".")
    return base / relative


def load_bundle_model(
    source_file: str = "model/module.py",
    entry_point: str = DEFAULT_ENTRY_POINT,
    bundle_dir: "str | Path | None" = None,
) -> Any:
    """Load the model a bundle carries — the bundle-relative model path.

    Returns the entry-point FUNCTION (a zero-argument factory returning a fresh
    model), which is what `--model` / `forecaster=` expects. The function is
    probed once here so a module that cannot produce a model fails at load with
    a named reason instead of inside a fold.

    `bundle_dir` is supplied by the verifier (batlab.validation.replication
    injects its own bundle path for a model loader that declares it), so the
    bundle verifies from wherever it was unzipped. `source_file` and
    `entry_point` are the values the bundle recorded, passed back in.

    Raises ValueError naming the reason when the index is missing, records a
    different schema, disagrees with the file's bytes, or the module cannot
    produce a gradable model. **Importing the file executes it** — see the
    module docstring.
    """
    path = _resolve_model_path(source_file, bundle_dir)
    if not path.is_file():
        raise ValueError(
            f"no model module at {path} — this bundle does not carry its model's "
            "source, so recompute needs the model from wherever it was published"
        )

    index_path = path.parent / MODEL_INDEX_NAME
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{index_path} is not readable JSON: {exc}") from exc
        if index.get("schema") != BUNDLE_MODEL_SCHEMA:
            raise ValueError(
                f"{index_path} declares schema {index.get('schema')!r}, not {BUNDLE_MODEL_SCHEMA!r}"
            )
        recorded = index.get("sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded and recorded != actual:
            raise ValueError(
                f"{path.name} does not match the digest its own index records "
                f"(recorded {str(recorded)[:16]}…, found {actual[:16]}…) — the "
                "model was changed after this bundle was sealed."
            )
        entry_point = index.get("entry_point") or entry_point

    try:
        probe = import_module_source(
            path.read_text(encoding="utf-8"),
            module_name=f"batlab_bundle_model_{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}",
            entry_point=entry_point,
            path=path,
        )
    except ModuleSourceError as exc:
        raise ValueError(f"the model module this bundle carries is not usable — {exc}") from exc
    return getattr(probe["module"], str(entry_point))
