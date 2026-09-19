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

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
import pandas as pd
from utils import (
    metric_tile_html, load_tenant_bundle_cached, cached_detect_knee, cached_match_fleet,
    _rul_interval_band_svg,
)


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


# ---------------------------------------------------------------------------
# cached_match_fleet() — regression coverage for TrajectoryMemory.match_fleet()
# being recomputed up to 3x per Fleet-page render, plus once on EVERY page in
# the app (main.py calls it unconditionally before page routing just to size
# a sidebar badge). This was the highest-impact finding of the whole perf
# review since it taxed every single interaction anywhere, not just Fleet.
# ---------------------------------------------------------------------------

class _FakeTrajectoryMemory:
    """Minimal stand-in with the one method cached_match_fleet() calls --
    doesn't need real TrajectoryMemory's cosine-similarity machinery, just
    something whose call count is observable."""
    def __init__(self):
        self.calls = 0

    def match_fleet(self, all_featured_dfs):
        self.calls += 1
        return {cid: f"match-for-{cid}" for cid in all_featured_dfs}


def _fleet_dfs():
    import pandas as pd
    return {
        "CellA": pd.DataFrame({"soh_pct": [100.0, 95.0], "cycle_number": [1, 2]}),
        "CellB": pd.DataFrame({"soh_pct": [100.0, 90.0], "cycle_number": [1, 2]}),
    }


def test_cached_match_fleet_matches_uncached_result():
    cached_match_fleet.clear()
    tm = _FakeTrajectoryMemory()
    dfs = _fleet_dfs()
    assert cached_match_fleet(tm, dfs) == tm.match_fleet(dfs)


def test_cached_match_fleet_only_computes_once_across_repeated_calls():
    """The actual regression: main.py's sidebar-badge call and Fleet page's
    own call, on the same featured_dfs content, must hit match_fleet() once
    combined, not twice -- simulating exactly that scenario here."""
    cached_match_fleet.clear()
    tm = _FakeTrajectoryMemory()
    dfs = _fleet_dfs()

    sidebar_badge_result = cached_match_fleet(tm, dfs)       # main.py's call
    fleet_page_result = cached_match_fleet(tm, dfs.copy())   # fleet.py's call, same content

    assert sidebar_badge_result == fleet_page_result
    assert tm.calls == 1, f"expected exactly 1 real match_fleet() call across both call sites, got {tm.calls}"


def test_cached_match_fleet_recomputes_for_different_fleet_content():
    cached_match_fleet.clear()
    tm = _FakeTrajectoryMemory()
    dfs = _fleet_dfs()
    dfs_smaller = {"CellA": dfs["CellA"]}

    cached_match_fleet(tm, dfs)
    cached_match_fleet(tm, dfs_smaller)
    assert tm.calls == 2, f"different fleet content must trigger a real recompute, got {tm.calls} calls"


# ---------------------------------------------------------------------------
# RUL interval band (the hero's calibrated-interval chart)
# ---------------------------------------------------------------------------

def _rul_inputs(n: int = 120, end_rul: float = 30.0):
    """A per-cycle RUL series: remaining life shrinking to `end_rul`."""
    cycles = pd.Series(range(1, n + 1))
    rul = pd.Series([float(end_rul + n - i) for i in range(n)])
    return cycles, rul


def test_interval_band_draws_the_served_interval_as_a_shaded_band():
    """The band is the served interval itself, so the picture and the quoted
    Q10/Q90 cannot disagree, and the legend says which line is which."""
    cycles, rul = _rul_inputs()
    svg = _rul_interval_band_svg(cycles, rul_pred=rul, rul_q10=rul - 40, rul_q90=rul + 40)
    assert "<polygon" in svg, "no band drawn despite a served interval"
    assert "band = served 80% interval" in svg
    assert "<polyline" in svg, "the point forecast must be drawn alongside it"
    assert "aria-label" in svg and "served 80% interval is drawn" in svg


def test_interval_band_draws_a_zero_lower_bound_instead_of_hiding_it():
    """Q10 == 0 is the served lower bound on every cycle of some fleets: an
    interval that cannot rule out that the cell is already at end of life.
    That is the case a user most needs to see, so it is drawn -- not dropped as
    degenerate -- and the accessible description quotes it."""
    cycles, rul = _rul_inputs()
    zeros = pd.Series([0.0] * len(rul))
    svg = _rul_interval_band_svg(cycles, rul_pred=rul, rul_q10=zeros, rul_q90=rul + 40)
    assert "<polygon" in svg
    assert "0 to " in svg, "aria must quote the served (open) lower bound"


def test_interval_band_draws_no_band_when_the_interval_is_withheld():
    """A withheld interval must render as no band, not a plausible-looking cone
    — that is the whole honesty rule the hero is supposed to show."""
    cycles, rul = _rul_inputs()
    svg = _rul_interval_band_svg(cycles, rul_pred=rul)
    assert svg, "the point forecast should still render"
    assert "<polygon" not in svg
    assert "<polyline" in svg
    assert "no interval served" in svg


def test_interval_band_renders_nothing_when_there_is_no_forecast_at_all():
    cycles, _ = _rul_inputs()
    assert _rul_interval_band_svg(cycles) == ""
    # ...and nothing to say when no cycle has any remaining life to plot.
    assert _rul_interval_band_svg(cycles, rul_pred=pd.Series([0.0] * len(cycles))) == ""


def test_interval_band_rejects_an_inverted_or_empty_interval():
    cycles, rul = _rul_inputs()
    assert "<polygon" not in _rul_interval_band_svg(
        cycles, rul_pred=rul, rul_q10=rul + 10, rul_q90=rul)
    assert "<polygon" not in _rul_interval_band_svg(
        cycles, rul_pred=rul, rul_q10=rul, rul_q90=rul)


def test_interval_band_clamps_a_negative_lower_bound_to_zero():
    """A negative Q10 is the model saying the cell may already be past the
    threshold, not a remaining life of -5 cycles. It clamps to an open band
    (the honest reading) instead of tearing a hole in the middle of the tail."""
    cycles, rul = _rul_inputs()
    negatives = pd.Series([-5.0] * len(rul))
    svg = _rul_interval_band_svg(cycles, rul_pred=rul, rul_q10=negatives, rul_q90=rul + 10)
    assert svg.count("<polygon") == 1, "clamping must keep the band in one piece"
    assert "-5" not in svg


def test_interval_band_draws_the_observed_outcome_only_when_handed_one():
    """The chart never invents an outcome to compare against: a formula
    extrapolation arrives as None and nothing is drawn for it."""
    cycles, rul = _rul_inputs()
    with_outcome = _rul_interval_band_svg(
        cycles, rul_pred=rul, rul_q10=rul - 40, rul_q90=rul + 40, observed_rul=rul - 5)
    assert "dashed = observed outcome" in with_outcome
    assert "against the observed remaining life" in with_outcome

    without_outcome = _rul_interval_band_svg(
        cycles, rul_pred=rul, rul_q10=rul - 40, rul_q90=rul + 40)
    assert "observed" not in without_outcome


def test_interval_band_does_not_bridge_a_gap_in_the_served_interval():
    """A hole in the served interval (or in the uploaded data) must not be
    covered by a straight edge implying an estimate nobody made."""
    cycles, rul = _rul_inputs(n=60)
    q10 = (rul - 20).astype(float)
    q90 = (rul + 20).astype(float)
    q10.iloc[20:30] = None
    q90.iloc[20:30] = None
    svg = _rul_interval_band_svg(cycles, rul_pred=rul, rul_q10=q10, rul_q90=q90)
    assert svg.count("<polygon") == 2, "the gap must split the band, not be spanned"


def test_interval_band_handles_a_series_too_short_to_plot():
    assert _rul_interval_band_svg(pd.Series([1]), rul_pred=pd.Series([10.0])) == ""


def test_interval_band_tolerates_cycles_without_a_number():
    """Uploaded data can arrive without a cycle_number column; the chart falls
    back to positional x rather than collapsing or dropping points."""
    _, rul = _rul_inputs()
    svg = _rul_interval_band_svg(
        pd.Series([None] * len(rul)), rul_pred=rul, rul_q10=rul - 40, rul_q90=rul + 40)
    assert "<polyline" in svg and "<polygon" in svg


def test_interval_band_downsamples_a_long_history_to_a_bounded_point_count():
    """A 1,000-cycle cell must not emit a 1,000-point polyline into the hero."""
    cycles = pd.Series(range(1, 1001))
    rul = pd.Series([float(1000 - i) for i in range(1000)])
    svg = _rul_interval_band_svg(
        cycles, rul_pred=rul, rul_q10=rul - 50, rul_q90=rul + 50, max_points=220)
    lines = [seg.split('"')[0] for seg in svg.split('<polyline points="')[1:]]
    bands = [seg.split('"')[0] for seg in svg.split('<polygon points="')[1:]]
    assert lines and all(len(pts.split()) <= 220 for pts in lines)
    assert bands and all(len(pts.split()) <= 2 * 220 for pts in bands)
