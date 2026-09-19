"""Production hierarchical estimator, censored-data survival evaluation,
and regime-based forecast routing — the "accuracy vs. real-world" batch.

Covers:
  - batlab.models.hierarchical: fit/forecast/project + AFT-style posterior
    RUL intervals, and estimation math shared with the validation harness.
  - The harness refactor: run_hierarchical_lco() must return byte-identical
    numbers after the math moved into the production module.
  - batlab.validation.survival: KM with right-censoring, rule of three,
    and the per-cell posterior check.
  - src/forecast_routing: gbrt / hierarchical / refuse per cell.
"""

import numpy as np
import pandas as pd
import pytest


def _lin(n_cycles=300, fade=0.0008, cap0=2.0, noise=0.0, seed=0, temp=25.0, r0=0.05, rg=5e-5):
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    capacity = cap0 - fade * cycles + (rng.normal(0, noise, n_cycles) if noise else 0.0)
    df = pd.DataFrame({
        "cycle_number": cycles,
        "capacity_ah": capacity,
        "resistance_ohm": r0 + rg * cycles,
        "temperature_c": np.full(n_cycles, temp),
    })
    df["soh_pct"] = df["capacity_ah"] / cap0 * 100.0
    return df


# ---------------------------------------------------------------------------
# batlab.models.hierarchical — production fit/forecast
# ---------------------------------------------------------------------------

def test_fit_hierarchical_priors_and_scope():
    from batlab.models.hierarchical import fit_hierarchical
    cells = {
        **{f"A{i}": _lin(fade=0.0008 + i * 0.0002, seed=i) for i in range(4)},
        **{f"B{i}": _lin(fade=0.0020 + i * 0.0003, seed=10 + i) for i in range(4)},
    }
    chem = {**{f"A{i}": "LiCoO2" for i in range(4)}, **{f"B{i}": "LFP" for i in range(4)}}
    fit = fit_hierarchical(cells, chemistry_by_cell=chem)
    assert fit["prior_scope"] == "per-chemistry"
    assert set(fit["priors"]) == {"LiCoO2", "LFP"}
    # A-fades ~0.0009-0.0015/cy, B-fades ~0.002-0.003: the LFP prior sits above.
    assert fit["priors"]["LFP"]["mu"] > fit["priors"]["LiCoO2"]["mu"]
    # Single-chemistry fleets get a fleet-local prior, recorded as such.
    fit2 = fit_hierarchical({f"C{i}": _lin(fade=0.0008 + i * 0.0002, seed=i) for i in range(4)},
                            chemistry_by_cell={f"C{i}": "LiCoO2" for i in range(4)})
    assert fit2["prior_scope"] == "fleet-local"
    assert "__fleet__" in fit2["priors"]


def test_prior_from_fleet_accepts_tuples_and_dicts():
    """The harness passes positional tuples; the production fit passes
    dicts. Indexing a dict with [0] used to raise KeyError — both shapes
    must produce the SAME prior."""
    from batlab.models.hierarchical import prior_from_fleet
    tup = prior_from_fleet([(np.log(0.001), 1e-4), (np.log(0.002), 1e-4)])
    dct = prior_from_fleet([{"log_fade": np.log(0.001), "log_fade_sampling_var": 1e-4},
                            {"log_fade": np.log(0.002), "log_fade_sampling_var": 1e-4}])
    assert tup == dct


def test_forecast_soh_shape_kinds_and_interval_ordering():
    from batlab.models.hierarchical import fit_hierarchical, forecast_soh
    cells = {f"N{i}": _lin(fade=0.0005 + 2e-5 * i, n_cycles=400, noise=0.003, seed=i) for i in range(6)}
    fit = fit_hierarchical(cells, chemistry_by_cell={c: "LFP" for c in cells})
    df = cells["N3"].sort_values("cycle_number")
    fc = forecast_soh(fit, "N3", df["cycle_number"].to_numpy(float), df["capacity_ah"].to_numpy(float))
    assert fc is not None
    n_hist = max(10, int(len(df) * 0.3))
    kinds = fc["forecast_kind"]
    assert (kinds[:n_hist] == "insample").all()
    assert (kinds[n_hist:] == "extrapolated").all()
    # A fading cell must show decreasing forecast SOH and positive RUL.
    assert fc["soh_forecast"][-1] < fc["soh_forecast"][n_hist]
    assert fc["rul_forecast"][-1] > 0
    # Posterior interval: q10 (sooner EOL) <= central <= q90.
    mask = np.isfinite(fc["rul_forecast"]) & np.isfinite(fc["rul_q10"]) & np.isfinite(fc["rul_q90"])
    assert mask.any()
    assert (fc["rul_q10"][mask] <= fc["rul_forecast"][mask] + 1e-9).all()
    assert (fc["rul_forecast"][mask] <= fc["rul_q90"][mask] + 1e-9).all()
    # The uncalibrated-interval disclosure travels with the forecast.
    assert "NOT conformally calibrated" in fc["rul_interval_convention"]
    # Unknown cell / missing prior -> None, never a fabricated number.
    assert forecast_soh(fit, "not_a_cell", df["cycle_number"].to_numpy(float), df["capacity_ah"].to_numpy(float)) is None
    assert forecast_soh(None, "N3", df["cycle_number"].to_numpy(float), df["capacity_ah"].to_numpy(float)) is None


def test_project_future_soh_extends_beyond_last_cycle_with_band():
    from batlab.models.hierarchical import fit_hierarchical, project_future_soh
    cells = {f"N{i}": _lin(fade=0.0006 + 2e-5 * i, n_cycles=300, noise=0.002, seed=i) for i in range(5)}
    fit = fit_hierarchical(cells, chemistry_by_cell={c: "LFP" for c in cells})
    proj = project_future_soh(fit, "N2", horizon_cycles=365)
    assert proj is not None
    last_cycle = 300.0
    assert proj["cycles"][0] == last_cycle + 1 and proj["cycles"][-1] == last_cycle + 365
    # Band ordering: q10 path (fast fade) below central below q90 path.
    assert (proj["soh_q10_pct"] <= proj["soh_pct"] + 1e-9).all()
    assert (proj["soh_pct"] <= proj["soh_q90_pct"] + 1e-9).all()
    # Fast-enough fade: an EOL crossing is produced inside the horizon.
    assert proj["eol_cycle_central"] is not None and proj["eol_cycle_central"] > last_cycle


# ---------------------------------------------------------------------------
# Harness equivalence: validation must not move when the math moved
# ---------------------------------------------------------------------------

def test_run_hierarchical_lco_equivalent_after_shared_math_refactor():
    """run_hierarchical_lco() now imports cell_local_stats/prior_from_fleet/
    shrunk_log_rate from batlab.models.hierarchical. A verbatim re-run of
    the estimation math over the harness's OWN data path (featured frames,
    get_model_matrix row filtering — the same rows the pre-refactor
    implementation consumed) must agree with the module's result to
    machine precision."""
    from batlab.validation.hierarchical_lco import run_hierarchical_lco
    from batlab.features.engineering import build_features, get_model_matrix
    cells = {f"N{i}": _lin(fade=0.0006 + 2e-5 * i, n_cycles=350, noise=0.002, seed=i) for i in range(5)}
    got = run_hierarchical_lco(cells)
    assert np.isfinite(got["soh_r2"]) and got["soh_r2"] > 0.9

    TAU2_FLOOR, MIN_HIST_FRACTION, MIN_HIST_FLOOR = 1e-4, 0.3, 10

    def local_stats(cycles, cap, early_only):
        if early_only:
            n_hist = max(MIN_HIST_FLOOR, int(len(cycles) * MIN_HIST_FRACTION))
            cycles, cap = cycles[:n_hist], cap[:n_hist]
        x, y = cycles.astype(float), cap.astype(float)
        xbar = x.mean()
        sxx = ((x - xbar) ** 2).sum()
        slope = ((x - xbar) * (y - y.mean())).sum() / sxx
        loss = -slope
        resid = y - (y.mean() + slope * (x - xbar))
        sigma2 = (resid ** 2).sum() / max(1, len(x) - 2)
        var_slope = max(sigma2 / sxx, 1e-12)
        return np.log(loss), var_slope / loss ** 2, x[0], y[0]

    def prior(stats):
        logs = np.array([s[0] for s in stats]); vs = np.array([s[1] for s in stats])
        tau2 = np.var(logs, ddof=1) - np.mean(vs)
        return np.mean(logs), max(tau2, TAU2_FLOOR)

    def shrink(local, pr):
        w_loc = 1.0 / max(local[1], 1e-12); w_pri = 1.0 / max(pr[1], 1e-12)
        return (w_loc * local[0] + w_pri * pr[0]) / (w_loc + w_pri)

    # The harness's exact data path: featured frame -> get_model_matrix row
    # filtering -> cycle_number/capacity_ah subselection on y_soh.index.
    series_by_cell = {}
    y_soh_by_cell = {}
    for cid, df in cells.items():
        frame = build_features(df, cell_id=cid)
        _X, y_soh, _y_rul = get_model_matrix(frame)
        sub = frame.loc[y_soh.index]
        series_by_cell[cid] = (
            sub["cycle_number"].to_numpy(np.float64),
            sub["capacity_ah"].to_numpy(float),
        )
        y_soh_by_cell[cid] = y_soh.to_numpy(float)

    cell_ids = list(cells)
    soh_r2s = []
    for test_cell in cell_ids:
        train = [c for c in cell_ids if c != test_cell]
        stats = [local_stats(*series_by_cell[c], early_only=False) for c in train]
        pr = prior(stats)
        x, y = series_by_cell[test_cell]
        loc = local_stats(x, y, early_only=True)
        th = shrink(loc, pr)
        slope = np.exp(th)
        pred = loc[3] - slope * (x - loc[2])
        soh_pred = np.clip(pred / loc[3] * 100.0, 0, None)
        y_soh = y_soh_by_cell[test_cell]
        ss_res = ((y_soh - soh_pred) ** 2).sum(); ss_tot = ((y_soh - y_soh.mean()) ** 2).sum()
        soh_r2s.append(1 - ss_res / ss_tot)
    assert abs(float(np.mean(soh_r2s)) - got["soh_r2"]) < 1e-9


# ---------------------------------------------------------------------------
# Censored-data survival
# ---------------------------------------------------------------------------

def test_kaplan_meier_all_censored_zero_events():
    """The Severson shape: no cell reaches EOL in-window. Every cell is a
    right-censored observation; the KM curve is flat at 1.0 and the fleet
    still gets an honest bounded claim."""
    from batlab.validation.survival import kaplan_meier_survival, rule_of_three_bound
    cells = {f"C{i}": _lin(300, 0.0002 + i * 1e-5, seed=i) for i in range(5)}
    km = kaplan_meier_survival(cells)
    assert km["n_cells"] == 5 and km["n_events"] == 0 and km["n_censored"] == 5
    assert km["survival"] == [] or all(s == 1.0 for s in km["survival"])
    assert "right-censored" in km["censoring_note"]
    bound = rule_of_three_bound(5, 300.0)
    assert bound is not None
    assert abs(bound["upper_bound_eol_prob"] - 3.0 / 5.0) < 0.01  # ~3/n at 95%
    assert "at most" in bound["statement"]


def test_survival_nested_cell_path_still_yields_rule_of_three():
    """Regression: censored_rul_readout() scanned the RAW dict for max_cycle,
    so the tenant-path shape {cell: {'cycles': df}} made max_cycle None and
    silently dropped the rule-of-three bound — the exact claim the survival
    module exists to make on never-EOL fleets. Both input shapes must
    produce the identical bound."""
    from batlab.validation.survival import censored_rul_readout
    df = _lin(300, 0.0002)
    nested = {"C1": {"cycles": df}}
    flat = {"C1": df}
    a = censored_rul_readout(nested)
    b = censored_rul_readout(flat)
    assert a["kaplan_meier"]["n_events"] == 0
    assert a["rule_of_three"] is not None
    assert a["rule_of_three"] == b["rule_of_three"]
    assert a["upper_bound_eol_prob"] == pytest.approx(-np.log(0.05) / 1)


def test_severson_sized_fleet_produces_an_honest_bound():
    """The real claim shape: a fleet where no cell reaches EOL still gets a
    quantitative survival statement — 3/n at 95% — not a bare
    'not evaluable'."""
    from batlab.validation.survival import censored_rul_readout
    cells = {f"C{i}": _lin(500 + i, 0.00008) for i in range(46)}
    surv = censored_rul_readout(cells)
    assert surv["kaplan_meier"]["n_censored"] == 46
    assert surv["upper_bound_eol_prob"] == pytest.approx(-np.log(0.05) / 46, rel=1e-6)
    stmt = surv["rule_of_three"]["statement"]
    assert "46" in stmt and "7%" in stmt


def test_kaplan_meier_mixed_events_and_censoring():
    from batlab.validation.survival import kaplan_meier_survival
    fast = {f"A{i}": _lin(300, 0.0030 + i * 0.0002, seed=i) for i in range(3)}
    slow = {f"B{i}": _lin(300, 0.0004 + i * 0.0001, seed=10 + i) for i in range(3)}
    km = kaplan_meier_survival({**fast, **slow})
    assert km["n_events"] == 3 and km["n_censored"] == 3
    # Censored-alive cells must remain at risk BEYOND their censoring time:
    # with 3 events and 3 censored, S at the last event is (1-1/6)(1-1/5)(1-1/4).
    expected = (1 - 1 / 6) * (1 - 1 / 5) * (1 - 1 / 4)
    assert abs(km["survival"][-1] - expected) < 1e-9
    assert km["median_lifetime"] is not None


def test_censored_rul_readout_with_hierarchical_posterior():
    from batlab.validation.survival import censored_rul_readout
    from batlab.models.hierarchical import fit_hierarchical
    # Never-EOL fleet (fade ~0.02%/cy over 300 cycles: final SOH ~94%).
    cells = {f"S{i}": _lin(300, 0.0002 + i * 2e-5, seed=i) for i in range(4)}
    fit = fit_hierarchical(cells, chemistry_by_cell={c: "LFP" for c in cells})
    out = censored_rul_readout(cells, hierarchical_fit=fit)
    assert out["kaplan_meier"]["n_events"] == 0
    assert out["rule_of_three"] is not None
    assert out["censoring_note"]
    # The posterior check: every still-alive cell's earliest plausible EOL
    # (at the 95th-percentile-fastest fade) lands beyond its last cycle —
    # consistent with the censoring fact by construction.
    pc = out["per_cell_posterior"]
    assert set(pc) == set(cells)
    for cid, rec in pc.items():
        assert rec["censor_consistent"] is True
        assert rec["eol_cycle_lower_bound"] > rec["last_cycle"]
    assert "No extrapolated RUL label" in out["method"] or "extrapolated RUL label" in out["method"]


# ---------------------------------------------------------------------------
# Regime-based forecast routing
# ---------------------------------------------------------------------------

class _FakeBundle:
    """Minimal bundle dict-shape for routing tests (no training)."""


def _mk_bundle(cells, with_hier=True, hier_soh_r2=0.75):
    from batlab.models.hierarchical import fit_hierarchical
    bndl = {
        "metrics": {
            "lco_per_cell": {c: {"soh_r2": 0.9, "rul_r2": None} for c in cells},
            "rul_reliable": False,
        }
    }
    if with_hier:
        chem = {c: "LFP" for c in cells}
        bndl["hierarchical_fit"] = fit_hierarchical({c: _lin(fade=0.0004, seed=0) for c in cells}, chemistry_by_cell=chem)
        bndl["metrics"]["hierarchical_validation"] = {
            "per_cell": {c: {"soh_r2": hier_soh_r2} for c in cells},
        }
        bndl["metrics"]["hierarchical_available"] = True
    return bndl


def test_route_gbrt_when_fold_passes_floor():
    from forecast_routing import route_forecast_for_cell
    cells = ["A", "B", "C"]
    bndl = _mk_bundle(cells)
    bndl["metrics"]["lco_per_cell"]["A"]["rul_r2"] = 0.6
    dfA = _lin(300, 0.001, seed=0)
    dfA["rul_pred"], dfA["rul_q10"], dfA["rul_q90"] = 100.0, 80.0, 120.0
    r = route_forecast_for_cell("A", {"src": bndl}, featured_dfs={"A": dfA})
    assert r["served"] == "gbrt"
    assert r["rul_pred"] == 100.0
    assert "calibrated" in r["interval_convention"]


def test_route_hierarchical_when_gbrt_rul_not_evaluable():
    from forecast_routing import route_forecast_for_cell
    cells = ["A", "B", "C"]
    bndl = _mk_bundle(cells)
    r = route_forecast_for_cell("A", {"src": bndl}, featured_dfs=None)
    assert r["served"] == "hierarchical"
    assert "not evaluable" in r["reason"]
    assert "NOT conformally calibrated" in r["interval_convention"]


def test_route_hierarchical_blocked_by_bad_hier_soh():
    """A knee-type fleet where the hierarchical linear fade fails (SOH R2
    below zero): the router must refuse, not serve a measured-bad model."""
    from forecast_routing import route_forecast_for_cell
    cells = ["A", "B", "C"]
    bndl = _mk_bundle(cells, with_hier=True, hier_soh_r2=-0.5)
    r = route_forecast_for_cell("A", {"src": bndl}, featured_dfs=None)
    assert r["served"] == "refuse"
    assert "withheld" in r["reason"].lower()


def test_route_refuse_without_any_model():
    from forecast_routing import route_forecast_for_cell
    r = route_forecast_for_cell("X", {}, featured_dfs=None)
    assert r["served"] == "refuse"
    assert r["rul_pred"] is None


def test_route_hierarchical_blocked_without_served_forecast_column():
    """When the caller supplies a featured frame but this cell's own
    forecast column is absent (no usable early window), the fleet-level
    hierarchical availability must NOT leak into a per-cell answer."""
    from forecast_routing import route_forecast_for_cell
    cells = ["A", "B", "C"]
    bndl = _mk_bundle(cells)
    dfA = _lin(300, 0.001, seed=0)  # no rul_forecast column on purpose
    r = route_forecast_for_cell("A", {"src": bndl}, featured_dfs={"A": dfA})
    assert r["served"] == "refuse"


# ---------------------------------------------------------------------------
# prospective baseline vs glitched rows — the NaN-row regression
# ---------------------------------------------------------------------------

def test_trivial_soh_baseline_skips_nonfinite_soh_rows():
    """One cycler-glitch row (NaN SOH — the loader's sentinel for impossible
    capacity readings) inside an otherwise clean fleet used to raise inside
    sklearn, the prospective runner's silent except then dropped the whole
    dataset from the benchmark. The baseline must skip non-finite rows the
    same way the GBRT path does."""
    from batlab.validation.prospective import trivial_soh_baseline_prospective
    cells = {
        **{f"C{i}": _lin(300, fade=0.0008 + i * 0.0002, seed=i) for i in range(3)},
        "G": _lin(300, fade=0.0014, seed=7).assign(
            soh_pct=lambda d: d["soh_pct"].mask(d.index == 5),
        ),
    }
    out = trivial_soh_baseline_prospective(cells, featured=None, train_fraction=0.5)
    assert out["baseline_soh_r2"] is not None
    assert out["n_cells"] == 4
