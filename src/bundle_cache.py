"""
Disk-based bundle cache.

Training the GBRT + quantile + LCO suite takes 20–60 s on a first run.
save_cached()/load_cached() are shape-agnostic joblib wrappers — they
persist whatever object a caller hands them, so cold starts after the
first run are instant regardless of what's actually cached.

app/main.py's load_everything() (the shared reference-fleet pipeline)
caches (bundle, split_cycles) pairs — per-cell full DataFrames are no
longer part of this cache; they're persisted separately, per cell, by
src/cell_store.py (see that module's docstring for why: this cache used
to also hold featured_dfs, which meant every warm restart reconstituted
every cell's full per-cycle DataFrame into RAM just to satisfy this
cache's return shape). save_tenant_bundle()/load_tenant_bundle() (the
separate uploaded-fleet path, below) still cache the full
(featured_dfs, bundle, split_cycles) triple — uploaded fleets are
user-bounded by what one person actually uploads in a session, not the
"low thousands of reference cells" scaling concern cell_store.py exists
for, so that path is intentionally left as-is.

Cache is invalidated automatically when the input data changes —
detected by a signature derived from cell IDs + cycle counts.
"""

import hashlib
import json
import pathlib

import joblib

from batlab.features.engineering import FEATURE_VERSION

CACHE_DIR = pathlib.Path(__file__).parent.parent / ".cache" / "bundles"

# FEATURE_VERSION (imported above) is the single source of truth for
# batlab.features.engineering.build_features() changes — see that module.
# MODEL_VERSION covers the other things this cache stores: trained model
# bundles, and the shape of the cached object itself. Bump it when
# batlab/models/gbrt.py changes in a way that makes a cached bundle stale
# WITHOUT the features themselves changing (e.g. new hyperparameters, a new
# model class) — nothing else tracks that, so it can't be centralized the
# way FEATURE_VERSION was — OR when what save_cached()/load_cached() are
# actually asked to persist changes shape (e.g. the reference-fleet callers
# switching from a (bundle, featured_dfs, split_cycles) triple to a
# (bundle, split_cycles) pair once src/cell_store.py took over per-cell
# persistence): without a bump here, a pre-existing on-disk cache from
# before the shape change would still match on cell IDs/cycle counts and
# get treated as a hit, and the caller's tuple-unpacking would crash on the
# old shape instead of cleanly missing and rebuilding.
MODEL_VERSION = "v2-gbrt-quantile-lazy-cell-store"

CACHE_VERSION = f"{FEATURE_VERSION}+{MODEL_VERSION}"


def _signature(battery_dict: dict) -> str:
    """
    Cache signature: cell IDs + cycle counts + cache version.
    Changing CACHE_VERSION above busts the cache across all cells.
    """
    sig = {cid: len(cell["cycles"]) for cid, cell in sorted(battery_dict.items())}
    sig["__cache_version__"] = CACHE_VERSION  # pyright: ignore[reportArgumentType]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:20]


def load_cached(key: str, battery_dict: dict):
    """
    Load whatever was cached under `key` by save_cached() — a
    (bundle, split_cycles) pair for the shared reference-fleet callers in
    app/main.py's load_everything(). Returns None if no cache exists or the
    signature doesn't match.

    key:          Short string identifying which model/dataset this is (e.g. "nasa", "synth")
    battery_dict: The raw battery dict used to check if data has changed.
    """
    meta_path   = CACHE_DIR / f"{key}.meta.json"
    bundle_path = CACHE_DIR / f"{key}.joblib"
    if not meta_path.exists() or not bundle_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if meta.get("sig") != _signature(battery_dict):
        return None
    try:
        return joblib.load(bundle_path)
    except Exception:
        return None


def load_cached_unchecked(key: str):
    """
    Load whatever was cached under `key`, with NO signature check — for
    callers that have no live battery_dict to check
    a signature against in the first place (e.g. src/api.py running as a
    bare process, with no Streamlit app import available to rebuild the
    battery_dict). Returns None if no cache file exists for this key, or
    if the cached file fails to load.

    Prefer load_cached() over this whenever a battery_dict is available —
    this skips the staleness check load_cached() exists to provide.
    """
    bundle_path = CACHE_DIR / f"{key}.joblib"
    if not bundle_path.exists():
        return None
    try:
        return joblib.load(bundle_path)
    except Exception:
        return None


def save_cached(key: str, battery_dict: dict, data):
    """
    Persist `data` to disk under `key` (shape-agnostic — a (bundle,
    split_cycles) pair for the shared reference-fleet callers in
    app/main.py's load_everything()). Overwrites any existing cache for
    this key.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path   = CACHE_DIR / f"{key}.meta.json"
    bundle_path = CACHE_DIR / f"{key}.joblib"
    sig = _signature(battery_dict)
    meta_path.write_text(json.dumps({"sig": sig, "key": key}))
    joblib.dump(data, bundle_path, compress=3)


def load_features_cached(key: str, battery_dict: dict):
    """Load (raw_fdfs, model_inputs) cached independently of the model bundle.

    raw_fdfs:     {cell_id: df_feat} — build_features() output, no predictions
    model_inputs: {cell_id: (X, y_soh, y_rul)} — ready for train_models()

    Returns None if no cache exists or the signature doesn't match.
    """
    meta_path = CACHE_DIR / f"{key}-features.meta.json"
    data_path = CACHE_DIR / f"{key}-features.joblib"
    if not meta_path.exists() or not data_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if meta.get("sig") != _signature(battery_dict):
        return None
    try:
        return joblib.load(data_path)
    except Exception:
        return None


def save_features_cached(key: str, battery_dict: dict, raw_fdfs: dict, model_inputs: dict):
    """Save (raw_fdfs, model_inputs) independently of the model bundle.

    Allows the model to be retrained (e.g. hyperparameter tuning) without
    re-running the full feature engineering pipeline.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / f"{key}-features.meta.json"
    data_path = CACHE_DIR / f"{key}-features.joblib"
    sig = _signature(battery_dict)
    meta_path.write_text(json.dumps({"sig": sig, "key": key, "type": "features"}))
    joblib.dump((raw_fdfs, model_inputs), data_path, compress=3)


TENANT_BUNDLE_DIR = pathlib.Path(__file__).parent.parent / "data" / "tenant_bundles"


def save_tenant_bundle(org_id: int, triple: tuple) -> None:
    """
    Persist an organization's uploaded ("My Data") (featured_dfs, bundle,
    split_cycles) triple to disk, keyed by org_id — not content-addressed
    like the caches above (there's no original CSV kept to regenerate this
    from), so it's simply overwritten on each new upload for that org.
    """
    TENANT_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(triple, TENANT_BUNDLE_DIR / f"{org_id}.joblib", compress=3)


def load_tenant_bundle(org_id: int):
    """Returns the org's persisted (featured_dfs, bundle, split_cycles) triple, or None."""
    path = TENANT_BUNDLE_DIR / f"{org_id}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def clear_cache(key: str | None = None, features_only: bool = False):
    """Delete cached files.

    key=None clears all bundles.  features_only=True preserves model bundles
    and only removes feature caches — useful when retraining without changing
    the feature engineering pipeline.
    """
    if not CACHE_DIR.exists():
        return
    if key is None:
        for f in CACHE_DIR.iterdir():
            if features_only and "-features." not in f.name:
                continue
            f.unlink(missing_ok=True)
    else:
        if features_only:
            suffixes = ("-features.meta.json", "-features.joblib")
        else:
            suffixes = (".meta.json", ".joblib", "-features.meta.json", "-features.joblib")
        for suffix in suffixes:
            (CACHE_DIR / f"{key}{suffix}").unlink(missing_ok=True)
