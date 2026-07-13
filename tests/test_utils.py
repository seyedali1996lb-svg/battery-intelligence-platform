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

from utils import metric_tile_html, load_tenant_bundle_cached


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
