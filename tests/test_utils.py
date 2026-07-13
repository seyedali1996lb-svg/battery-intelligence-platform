"""Unit tests for app/utils.py's shared render primitives.

app/utils.py is an app-layer module (imports streamlit at module level),
which is why nothing in tests/ has unit-tested it directly before now --
everything else about it is implicitly covered via the AppTest
integration suite (tests/test_app_state_combinations.py). metric_tile_html()
is a pure string-returning function though (no st.* calls), so it's
directly testable the same way src/ pure-logic modules are.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from utils import metric_tile_html, load_tenant_bundle_cached, cached_detect_knee


def test_metric_tile_html_includes_label_and_value():
    html = metric_tile_html("Total Cells", "12")
    assert "Total Cells" in html
    assert ">12<" in html


def test_metric_tile_html_omits_sub_div_when_sub_is_empty():
    """No sub-caption given -> no dangling empty <div> for it, so a page
    composing several tiles doesn't get stray empty elements."""
    html = metric_tile_html("Total Cells", "12")
    assert "margin-top:2px" not in html  # that's the sub-caption's div


def test_metric_tile_html_includes_sub_when_given():
    html = metric_tile_html("Total Cells", "12", sub="12 Severson real")
    assert "12 Severson real" in html
    assert "margin-top:2px" in html


def test_metric_tile_html_uses_given_value_color():
    html = metric_tile_html("Fleet Health", "85.9%", value_color="#fc8181")
    assert "#fc8181" in html


# ---------------------------------------------------------------------------
# load_tenant_bundle_cached() — regression coverage for the fix that removed
# an org's uploaded fleet from st.session_state (previously duplicated there
# AND on disk simultaneously). This is a real @st.cache_resource function
# (works fine called directly outside a live Streamlit run, verified before
# writing these as committed tests) — these confirm caching and .clear()
# invalidation actually work, not just that the disk-loading logic is right
# (that part is bundle_cache.load_tenant_bundle()'s own, already-tested job).
# ---------------------------------------------------------------------------

def test_load_tenant_bundle_cached_only_hits_disk_once_per_org(monkeypatch):
    load_tenant_bundle_cached.clear()  # isolate from any other test's cached entries
    calls = []

    def fake_load_tenant_bundle(org_id):
        calls.append(org_id)
        return (f"fdfs-{org_id}", f"bndl-{org_id}", f"sc-{org_id}")

    import bundle_cache
    monkeypatch.setattr(bundle_cache, "load_tenant_bundle", fake_load_tenant_bundle)

    result1 = load_tenant_bundle_cached(42)
    result2 = load_tenant_bundle_cached(42)
    assert result1 == result2 == ("fdfs-42", "bndl-42", "sc-42")
    assert calls == [42]  # second call served from cache, not a second disk read


def test_load_tenant_bundle_cached_clear_forces_a_fresh_disk_read(monkeypatch):
    """The actual regression this exists to prevent: after a new upload is
    saved to disk, .clear() must make the next call see it — not keep
    serving a stale cached triple from before the upload."""
    load_tenant_bundle_cached.clear()
    calls = []

    def fake_load_tenant_bundle(org_id):
        calls.append(org_id)
        return (f"fdfs-v{len(calls)}", f"bndl-v{len(calls)}", f"sc-v{len(calls)}")

    import bundle_cache
    monkeypatch.setattr(bundle_cache, "load_tenant_bundle", fake_load_tenant_bundle)

    first = load_tenant_bundle_cached(7)
    load_tenant_bundle_cached.clear()
    second = load_tenant_bundle_cached(7)
    assert first != second
    assert calls == [7, 7]


def test_load_tenant_bundle_cached_keys_by_org_id(monkeypatch):
    load_tenant_bundle_cached.clear()

    import bundle_cache
    monkeypatch.setattr(bundle_cache, "load_tenant_bundle", lambda org_id: (org_id, org_id, org_id))

    assert load_tenant_bundle_cached(1) == (1, 1, 1)
    assert load_tenant_bundle_cached(2) == (2, 2, 2)  # different org -> not the org-1 cached value


# ---------------------------------------------------------------------------
# cached_detect_knee() — regression coverage for the Fleet page recomputing
# knee detection for every cell on every single rerun. st.cache_data has
# built-in pandas Series hashing, so this needs no manual cache-key
# construction like load_tenant_bundle_cached()/PDF caching did.
# ---------------------------------------------------------------------------

def _knee_series():
    import numpy as np
    import pandas as pd
    n, knee_at = 300, 220
    cyc = np.arange(1, n + 1)
    soh = np.where(cyc < knee_at, 100 - (cyc / knee_at) * 5, 95 - (cyc - knee_at) * 0.15)
    return pd.Series(soh), pd.Series(cyc)


def test_cached_detect_knee_matches_uncached_result():
    cached_detect_knee.clear()
    from batlab.features.knee_detection import detect_knee

    soh, cyc = _knee_series()
    assert cached_detect_knee(soh, cyc) == detect_knee(soh, cyc)


def test_cached_detect_knee_only_computes_once_for_identical_series(monkeypatch):
    cached_detect_knee.clear()
    import utils
    from batlab.features import knee_detection as kd_module

    calls = []
    real_detect_knee = kd_module.detect_knee

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_detect_knee(*args, **kwargs)

    monkeypatch.setattr(utils, "_detect_knee", _spy)

    soh, cyc = _knee_series()
    r1 = cached_detect_knee(soh, cyc)
    r2 = cached_detect_knee(soh.copy(), cyc.copy())  # equal content, different objects
    assert r1 == r2
    assert calls == [1], f"expected exactly 1 real computation across 2 identical-content calls, got {len(calls)}"


def test_cached_detect_knee_recomputes_for_different_series():
    cached_detect_knee.clear()
    soh, cyc = _knee_series()
    soh_other = soh * 0.5  # genuinely different SOH curve
    assert cached_detect_knee(soh, cyc) != cached_detect_knee(soh_other, cyc)
