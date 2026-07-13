"""Unit tests for src/bundle_cache.py — disk cache round-trip and invalidation."""

import pytest
import bundle_cache as bc


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "CACHE_DIR", tmp_path / "bundles")
    return bc


def test_load_cached_returns_none_when_absent(isolated_cache):
    battery = {"CellA": {"cycles": list(range(100))}}
    assert isolated_cache.load_cached("nasa", battery) is None


def test_save_then_load_round_trip(isolated_cache):
    battery = {"CellA": {"cycles": list(range(100))}}
    triple = ({"metrics": {"soh_r2": 0.9}}, {"CellA": "df_placeholder"}, {"CellA": 80})
    isolated_cache.save_cached("nasa", battery, triple)
    loaded = isolated_cache.load_cached("nasa", battery)
    assert loaded == triple


def test_cache_invalidates_on_cycle_count_change(isolated_cache):
    battery_v1 = {"CellA": {"cycles": list(range(100))}}
    isolated_cache.save_cached("nasa", battery_v1, ("bundle", {}, {}))
    battery_v2 = {"CellA": {"cycles": list(range(150))}}  # cycle count changed
    assert isolated_cache.load_cached("nasa", battery_v2) is None


def test_features_cache_round_trip_independent_of_bundle_cache(isolated_cache):
    battery = {"CellA": {"cycles": list(range(100))}}
    raw_fdfs = {"CellA": "feat_df"}
    model_inputs = {"CellA": ("X", "y_soh", "y_rul")}
    isolated_cache.save_features_cached("upload-key", battery, raw_fdfs, model_inputs)
    loaded = isolated_cache.load_features_cached("upload-key", battery)
    assert loaded == (raw_fdfs, model_inputs)
    # The regular bundle cache for the same key must remain untouched
    assert isolated_cache.load_cached("upload-key", battery) is None


def test_clear_cache_features_only_preserves_bundle(isolated_cache):
    battery = {"CellA": {"cycles": list(range(100))}}
    isolated_cache.save_cached("nasa", battery, ("bundle", {}, {}))
    isolated_cache.save_features_cached("nasa", battery, {}, {})
    isolated_cache.clear_cache("nasa", features_only=True)
    assert isolated_cache.load_cached("nasa", battery) is not None
    assert isolated_cache.load_features_cached("nasa", battery) is None


def test_load_cached_unchecked_returns_none_when_absent(isolated_cache):
    assert isolated_cache.load_cached_unchecked("nasa") is None


def test_load_cached_unchecked_ignores_signature_mismatch(isolated_cache):
    """The whole point of load_cached_unchecked(): a caller with no live
    battery_dict (src/api.py's standalone-process fallback) still gets the
    cached triple back, unlike load_cached() which would reject it."""
    battery_v1 = {"CellA": {"cycles": list(range(100))}}
    triple = ({"metrics": {"soh_r2": 0.9}}, {"CellA": "df_placeholder"}, {"CellA": 80})
    isolated_cache.save_cached("nasa", battery_v1, triple)

    battery_v2 = {"CellA": {"cycles": list(range(150))}}  # would invalidate load_cached()
    assert isolated_cache.load_cached("nasa", battery_v2) is None
    assert isolated_cache.load_cached_unchecked("nasa") == triple
