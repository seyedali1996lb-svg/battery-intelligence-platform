"""
Disk-based bundle cache.

Training the GBRT + quantile + LCO suite takes 20–60 s on a first run.
This module persists the (bundle, featured_dfs, split_cycles) tuple to disk
using joblib so cold starts after the first run are instant.

Cache is invalidated automatically when the input data changes —
detected by a signature derived from cell IDs + cycle counts.
"""

import hashlib
import json
import pathlib

import joblib

CACHE_DIR = pathlib.Path(__file__).parent.parent / ".cache" / "bundles"

# Bump this string whenever feature engineering or model code changes in a way
# that makes cached featured_dfs or bundles incompatible with the current code.
# The version is mixed into the cache signature so old caches are automatically
# invalidated and rebuilt on the next app start.
FEATURE_VERSION = "v9-crate-stress-index-dod-proxy"


def _signature(battery_dict: dict) -> str:
    """
    Cache signature: cell IDs + cycle counts + feature version.
    Changing FEATURE_VERSION above busts the cache across all cells.
    """
    sig = {cid: len(cell["cycles"]) for cid, cell in sorted(battery_dict.items())}
    sig["__feature_version__"] = FEATURE_VERSION
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:20]


def load_cached(key: str, battery_dict: dict):
    """
    Load a cached (bundle, featured_dfs, split_cycles) triple.
    Returns None if no cache exists or the signature doesn't match.

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


def save_cached(key: str, battery_dict: dict, triple: tuple):
    """
    Persist a (bundle, featured_dfs, split_cycles) triple to disk.
    Overwrites any existing cache for this key.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path   = CACHE_DIR / f"{key}.meta.json"
    bundle_path = CACHE_DIR / f"{key}.joblib"
    sig = _signature(battery_dict)
    meta_path.write_text(json.dumps({"sig": sig, "key": key}))
    joblib.dump(triple, bundle_path, compress=3)


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
