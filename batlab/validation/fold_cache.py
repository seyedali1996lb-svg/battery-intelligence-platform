"""Durable, per-fold cache for leave-cell-out validation.

Why this exists
---------------
An LCO fold is a pure function of (the pool's cells, their feature frames, the
model configuration, the seed) — the property batlab._parallel already leans
on to run folds concurrently ("each fold is seeded independently, the fold
function is pure"). Nothing about a fold's result depends on when, or how
often, it is computed. So a completed fold can be REPLAYED instead of refitted
— which is what makes a 46-cell fleet's validation cheaper to repeat than to
interrupt.

Measured motivation (2026-09-19, 46-cell Severson fleet = 35 822 feature rows):
one `run_lco` is 92 exact-splitter GBRT fits at ~53 s each, **710 s of wall
clock** on the 8-worker budget. Before this module, an interrupted run threw
all of that away and a repeated boot refit all 46 folds even though 45 of them
had already produced their numbers minutes earlier. With the cache, a kill
during fold 40 leaves 39 fold results on disk, and the next run fits only the
7 that never finished.

What a fold's result is keyed on
--------------------------------
`fold_key()` hashes: the per-cell CONTENT digests
(fingerprints.dataset_fingerprint — a loader fix, a sentinel cleanup or one
extra row all change it), the held-out cell id, FEATURE_VERSION, GBRT_PARAMS,
the seed, the tracked numeric-stack versions (a different scikit-learn fits a
different model), whether out-of-fold predictions were requested (they change
the stored shape), and FOLD_CACHE_CODE_VERSION below.

Deliberately NOT keyed on a dataset *name*: "Severson" (or any other label) is
a string, and the Tier-4 NASA incident is what that costs — the registry's
per-cell digests exist for the same reason. It is also not keyed on the cell's
own `attrs`, which are provenance for humans and never enter a fit.

One assumption is load-bearing and worth stating: when a caller passes
pre-built feature frames (`featured=`), the key assumes they were derived from
the SAME cells by the same FEATURE_VERSION. That invariant is what the app's
features cache already relies on (src/bundle_cache keys on FEATURE_VERSION plus
a data signature), so it holds everywhere in this repo — but a caller who
hand-edits feature frames while keeping the raw cells somehow identical would
be replaying a fold that describes different features.

Scope of the seam
-----------------
Only the platform's own default forecaster is cached. A fold graded with a
caller-supplied `forecaster=` is neither read from nor written to this cache:
this module has no way to key on a model it does not own, and a stale hit
there would silently grade the wrong estimator. `for_run(enabled=False)` is
how a caller opts out, and run_lco() passes it exactly when `forecaster` was
supplied.

Failure policy
--------------
Every filesystem error is swallowed and treated as a miss — a cache is an
optimization and must never be able to fail a training run. Writes go to a
temporary file and are renamed into place, so a process killed mid-write
leaves either the previous result or nothing, never a truncated one: the
interrupted-run case this module exists for is precisely the case in which a
half-written fold is possible.

Knobs
-----
BATLAB_LCO_CACHE      on (default) | off | refresh
                      off     — neither read nor write
                      refresh — ignore existing entries, rewrite them
BATLAB_LCO_CACHE_DIR  cache root override (tests use it for hermeticity)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import time
from typing import Any, Sequence

import joblib

# Bump when a fold's MODEL or METRICS change in a way that leaves its key
# ingredients identical — a new criterion, a changed metric definition, an
# extra column in the fold dict. Nothing else in the codebase covers that
# (FEATURE_VERSION covers feature values, GBRT_PARAMS covers hyperparameters),
# which is the same gap bundle_cache.MODEL_VERSION documents for trained
# bundles. A key that cannot see a code change is how a cache starts serving
# last week's answer.
FOLD_CACHE_CODE_VERSION = "v1-gbrt-lco-folds"

CACHE_MODE_ENV = "BATLAB_LCO_CACHE"
CACHE_DIR_ENV = "BATLAB_LCO_CACHE_DIR"

_MODE_ON = "on"
_MODE_OFF = "off"
_MODE_REFRESH = "refresh"

# batlab/validation/fold_cache.py -> the checkout root two levels up. Used only
# to keep the cache next to .cache/bundles in a source checkout; a `pip
# install batlab` (where this resolves inside site-packages) falls back to the
# user cache dir instead of writing into the installed package.
_CHECKOUT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def cache_mode() -> str:
    """BATLAB_LCO_CACHE, normalized. Anything unrecognized means `on`."""
    mode = (os.environ.get(CACHE_MODE_ENV) or _MODE_ON).strip().lower()
    return mode if mode in (_MODE_ON, _MODE_OFF, _MODE_REFRESH) else _MODE_ON


def cache_dir() -> pathlib.Path:
    """Where fold results live: BATLAB_LCO_CACHE_DIR, else the checkout's
    .cache/lco, else the user cache dir (for a pip-installed batlab)."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override)
    if (_CHECKOUT_ROOT / "pyproject.toml").exists():
        return _CHECKOUT_ROOT / ".cache" / "lco"
    return pathlib.Path.home() / ".cache" / "batlab" / "lco"


def _safe_name(cell_id: str) -> str:
    """A filesystem-safe file name for a cell id.

    Cell ids are human labels (`B0005`, `S-b1c13`, `slow_2`) and are kept
    readable on purpose — the cache directory is meant to be inspectable. Any
    character that could change the path (separators, drive colons) is
    replaced; a collision between two ids that differ only in those
    characters is possible in principle and harmless in practice, because the
    fold's own key still pins which cell it belongs to.
    """
    safe = "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in str(cell_id))
    return safe or "_"


def fold_key(base: dict, cell_id: str) -> str:
    """Full key for one fold: sha256 over the base ingredients + the cell id."""
    payload = {**base, "fold": str(cell_id)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def cache_key_id(base: dict) -> str:
    """Short directory name for one (dataset, features, model, seed) run."""
    return hashlib.sha256(json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()[:20]


class FoldCache:
    """Per-fold result store for one run_lco() call.

    `hits` counts folds replayed from disk, `fitted` counts folds this call
    computed. Both are reported in run_lco()'s result under `fold_cache`, so a
    boot can say whether its validation layer was reused or refitted instead
    of leaving it to be inferred from the clock.
    """

    def __init__(
        self,
        base: dict,
        root: pathlib.Path,
        key_dir: pathlib.Path,
        enabled: bool,
        read: bool,
        store: bool,
    ) -> None:
        self._base = dict(base)
        self._root = root
        self._dir = key_dir
        self._enabled = enabled
        self._read = read
        self._store = store
        self.hits = 0
        self.fitted = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def _path(self, cell_id: str) -> pathlib.Path:
        return self._dir / f"{_safe_name(cell_id)}.joblib"

    def get(self, cell_id: str) -> "dict | None":
        """The stored fold result, or None on any miss (absent, unreadable,
        truncated, wrong shape — all of which mean \"fit it again\").

        `refresh` mode reads nothing while still writing, so a repopulated
        cache is rebuilt from the current code rather than from itself."""
        if not (self._enabled and self._read):
            return None
        try:
            result = joblib.load(self._path(cell_id))
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        self.hits += 1
        return result

    def put(self, cell_id: str, result: dict) -> None:
        """Persist one fold's result, atomically. Never raises."""
        self.fitted += 1
        if not self._store:
            return
        path = self._path(cell_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_meta_once(path.parent)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            os.close(fd)
            try:
                joblib.dump(result, tmp, compress=3)
                os.replace(tmp, path)  # atomic: readers see old or new, never half
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception:
            # A cache that cannot write is a slower run, not a failed one.
            return

    def _write_meta_once(self, key_dir: pathlib.Path) -> None:
        meta = key_dir / "meta.json"
        if meta.exists():
            return
        meta.write_text(json.dumps({
            "key": cache_key_id(self._base),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ingredients": self._base,
        }, indent=2, default=str), encoding="utf-8")

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        """What this run did with the cache — travels in run_lco()'s result.

        `dir` is the cache ROOT (the same thing describe() reports, and what a
        human inspects); `key` names the key directory inside it holding this
        run's folds.
        """
        return {
            "mode": cache_mode(),
            "enabled": self._enabled,
            "key": cache_key_id(self._base) if self._enabled else None,
            "dir": str(self._root),
            "hits": self.hits,
            "fitted": self.fitted,
        }


def for_run(
    *,
    dataset_sha256: str,
    env_snapshot: dict,
    cell_ids: Sequence[str],
    seed: int,
    include_predictions: bool,
    params: dict,
    enabled: bool = True,
) -> FoldCache:
    """A FoldCache for one run_lco() call.

    `enabled=False` (a caller-supplied forecaster) returns a cache that reads
    and writes nothing, so the fold path is byte-identical to the pre-cache
    behaviour and summary() still reports honestly why.
    """
    mode = cache_mode()
    base: dict[str, Any] = {
        "code": FOLD_CACHE_CODE_VERSION,
        "dataset_sha256": dataset_sha256,
        "cells": sorted(str(c) for c in cell_ids),
        "features": _feature_version(),
        "params": params,
        "seed": int(seed),
        "predictions": bool(include_predictions),
        "env": env_snapshot,
    }
    root = cache_dir()
    key_dir = root / cache_key_id(base)
    if not enabled or mode == _MODE_OFF:
        # Nothing read, nothing written: the fold path is then byte-identical
        # to the pre-cache behaviour, and summary() says so.
        return FoldCache(base, root, key_dir, enabled=False, read=False, store=False)
    return FoldCache(
        base, root, key_dir,
        enabled=True,
        read=(mode != _MODE_REFRESH),
        store=True,
    )


def _feature_version() -> str:
    """FEATURE_VERSION, imported lazily so this module stays import-light."""
    from batlab.features.engineering import FEATURE_VERSION

    return FEATURE_VERSION


def clear(directory: "pathlib.Path | None" = None) -> int:
    """Delete cached folds (all of them, or one key directory). Returns the
    number of files removed. Used by tests and by anyone who wants to force a
    clean refit; never called on a training path."""
    root = directory or cache_dir()
    if not root.exists():
        return 0
    removed = 0
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                path.rmdir()
        except Exception:
            continue
    return removed


def describe() -> dict:
    """A read-only snapshot of what is on disk: mode, root, and per-key fold
    counts with the dataset/params each key was built from. Purely for
    inspection (a cold-boot log line, a support question, a script)."""
    root = cache_dir()
    keys = []
    if root.exists():
        for key_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            meta = {}
            try:
                meta = json.loads((key_dir / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            folds = len(list(key_dir.glob("*.joblib")))
            ingredients = meta.get("ingredients") or {}
            keys.append({
                "key": key_dir.name,
                "folds": folds,
                "n_cells": len(ingredients.get("cells") or []),
                "features": ingredients.get("features"),
                "seed": ingredients.get("seed"),
                "dataset_sha256": ingredients.get("dataset_sha256"),
                "mtime": key_dir.stat().st_mtime,
            })
    return {"mode": cache_mode(), "dir": str(root), "keys": keys}
