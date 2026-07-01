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


def _signature(battery_dict: dict) -> str:
    """
    Fast data signature: cell IDs + cycle counts per cell.
    If any cell's data changes length (new cycles added), the sig changes
    and the cache is invalidated. Deliberately ignores actual values for
    speed — a length change is the normal signal that data was updated.
    """
    sig = {cid: len(cell["cycles"]) for cid, cell in sorted(battery_dict.items())}
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


def clear_cache(key: str | None = None):
    """
    Delete cached files. If key is None, clears all bundles.
    Called from Settings page when the user wants to force a retrain.
    """
    if not CACHE_DIR.exists():
        return
    if key is None:
        for f in CACHE_DIR.iterdir():
            f.unlink(missing_ok=True)
    else:
        for suffix in (".meta.json", ".joblib"):
            (CACHE_DIR / f"{key}{suffix}").unlink(missing_ok=True)
