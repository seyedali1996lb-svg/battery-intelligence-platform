"""Where leave-cell-out validation's wall clock actually goes.

Why this exists: on the 46-cell Severson fleet a cold boot's `validation`
layer (app/_data.py) was measured at "not finished in 25 minutes" while
train_models() on the same data takes 43.5 s, and nothing said which part of
run_lco() was responsible. Guessing at that is how a fix becomes a rewrite, so
this script decomposes the layer instead:

    python scripts/profile_lco.py severson
    python scripts/profile_lco.py severson --phase folds     # just the folds
    python scripts/profile_lco.py severson --folds 6         # first 6 cells only

Phases, in the order run_lco() runs them:

    features    get_model_matrix() per cell (run_lco()'s featured_in build)
    folds       map_folds() over the leave-cell-out folds — printed per fold,
                so a single pathological cell cannot hide inside a mean
    fingerprint dataset/environment digest over the raw frames
    baselines   baseline_lco_r2() + rul_formula_baseline_lco()
    envelope    compute_envelope() + regime_reliability()
    quantiles   run_lco_quantiles() — the calibration layer's measurement

Nothing is written: no bundle cache, no cell store, no registry row. Feature
frames come from the same cache a boot uses (bundle_cache.load_features_cached)
so the numbers are comparable with the boot profile's.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _paths import setup  # noqa: E402

setup()

import _data  # noqa: E402
from bundle_cache import load_features_cached  # noqa: E402

import batlab.validation.lco as lco_mod  # noqa: E402

PHASES = ("features", "folds", "fingerprint", "baselines", "envelope", "quantiles")


def _cells(key: str) -> dict:
    if key == "severson":
        from batlab.datasets.severson import load_severson_cells

        return {
            cid: {"cycles": df}
            for cid, df in load_severson_cells(status_fn=lambda _m: None).items()
        }
    if key == "zhu2022":
        from batlab.datasets.zhu2022 import load_zhu2022_cells

        return {
            cid: {"cycles": df}
            for cid, df in load_zhu2022_cells(status_fn=lambda _m: None).items()
        }
    if key == "nasa":
        from data_loader import build_battery

        ids = _data.nasa_cells_available()
        return build_battery(battery_id="NASA_B1", cell_ids=ids)["cells"]
    raise SystemExit(f"unknown fleet '{key}' (nasa, severson, zhu2022)")


def _instrument_folds() -> "tuple[dict, Callable[[], None]]":
    """Time every fold and every model fit inside run_lco() by wrapping the
    functions it calls — so the measurement cannot drift from the code."""
    stats: dict = {"folds": [], "fits": []}
    real_map_folds = lco_mod.map_folds
    real_fit = lco_mod.fit_forecaster

    def timed_fit(model, X, y):
        t0 = time.perf_counter()
        out = real_fit(model, X, y)
        stats["fits"].append((len(X), time.perf_counter() - t0))
        return out

    def timed_map_folds(fn, items):
        def wrapped(item):
            t0 = time.perf_counter()
            out = fn(item)
            stats["folds"].append((item, time.perf_counter() - t0))
            print(f"      fold {item:<12} {stats['folds'][-1][1]:7.2f}s", flush=True)
            return out

        return real_map_folds(wrapped, items)

    lco_mod.map_folds = timed_map_folds
    lco_mod.fit_forecaster = timed_fit

    def restore() -> None:
        lco_mod.map_folds = real_map_folds
        lco_mod.fit_forecaster = real_fit

    return stats, restore


def profile(key: str, phases: set[str], max_folds: int | None) -> None:
    t0 = time.perf_counter()
    cells = _cells(key)
    print(f"\n=== {key}: {len(cells)} cells (loaded in {time.perf_counter() - t0:.1f}s) ===", flush=True)

    t0 = time.perf_counter()
    cached = load_features_cached(key, cells)
    if cached is None:
        raw_fdfs, model_inputs = _data.compute_features_only(cells)
        print(f"    features (cold)        {time.perf_counter() - t0:8.1f}s", flush=True)
    else:
        raw_fdfs, model_inputs = cached
        print(f"    features (cache hit)   {time.perf_counter() - t0:8.1f}s", flush=True)

    cell_cycles = {cid: c["cycles"] for cid, c in cells.items()}
    if max_folds:
        keep = list(cell_cycles)[:max_folds]
        cell_cycles = {k: cell_cycles[k] for k in keep}
        raw_fdfs = {k: v for k, v in raw_fdfs.items() if k in keep}
        print(f"    (subset: first {len(cell_cycles)} cells)", flush=True)

    n_rows = sum(len(f) for f in raw_fdfs.values())
    print(f"    featured rows          {n_rows:8d} across {len(raw_fdfs)} cells", flush=True)

    # ── features: exactly what run_lco() does before it can run a fold ──
    if "features" in phases:
        from batlab.features.engineering import get_model_matrix, get_rul_label_kinds

        t0 = time.perf_counter()
        n_missing_kind = 0
        for cid, frame in raw_fdfs.items():
            if "rul_label_kind" not in frame.columns:
                n_missing_kind += 1
            get_model_matrix(frame)
            get_rul_label_kinds(frame)
        print(
            f"    model matrices          {time.perf_counter() - t0:8.1f}s"
            f"  (frames missing rul_label_kind: {n_missing_kind})",
            flush=True,
        )

    if "folds" in phases:
        stats, restore = _instrument_folds()
        print("    folds:", flush=True)
        t0 = time.perf_counter()
        try:
            lco = lco_mod.run_lco(cell_cycles, featured=raw_fdfs)
        finally:
            restore()
        fold_s = time.perf_counter() - t0
        folds = sorted(s for _c, s in stats["folds"])
        fits = sorted(s for _n, s in stats["fits"])
        print(f"    folds total            {fold_s:8.1f}s   ({len(folds)} folds)", flush=True)
        if folds:
            print(
                f"      fold  min/median/max  {folds[0]:6.2f} / {statistics.median(folds):6.2f} / {folds[-1]:6.2f}s",
                flush=True,
            )
        if fits:
            print(
                f"      fit   min/median/max  {fits[0]:6.2f} / {statistics.median(fits):6.2f} / {fits[-1]:6.2f}s"
                f"   ({len(fits)} fits, rows {fits and stats['fits'][0][0]})",
                flush=True,
            )
        print(f"      soh_r2={lco['soh_r2']:.4f} rul_r2={lco['rul_r2']}", flush=True)
        cache = lco.get("fold_cache") or {}
        print(
            f"      fold cache         {cache.get('hits')} reused / {cache.get('fitted')} fitted"
            f"  (mode={cache.get('mode')}, key={cache.get('key')})",
            flush=True,
        )

    if "fingerprint" in phases:
        from batlab.validation import fingerprints as _fp

        t0 = time.perf_counter()
        _fp.dataset_fingerprint(cell_cycles)
        print(f"    fingerprint             {time.perf_counter() - t0:8.1f}s", flush=True)

    if "baselines" in phases:
        from batlab.validation.trivial_baseline import (
            baseline_lco_r2,
            rul_formula_baseline_lco,
        )

        t0 = time.perf_counter()
        baseline_lco_r2(cell_cycles, featured=raw_fdfs)
        print(f"    baseline (linear SOH)   {time.perf_counter() - t0:8.1f}s", flush=True)
        t0 = time.perf_counter()
        rul_formula_baseline_lco(cell_cycles, featured=raw_fdfs)
        print(f"    baseline (RUL formula)  {time.perf_counter() - t0:8.1f}s", flush=True)

    if "envelope" in phases:
        from domain_validity import compute_envelope, regime_reliability

        t0 = time.perf_counter()
        compute_envelope(cell_cycles, cell_data=cell_cycles, featured=raw_fdfs)
        print(f"    envelope                {time.perf_counter() - t0:8.1f}s", flush=True)
        t0 = time.perf_counter()
        regime_reliability({}, cell_data=cell_cycles, featured=raw_fdfs)
        print(f"    regime reliability      {time.perf_counter() - t0:8.1f}s", flush=True)

    if "quantiles" in phases:
        from batlab.validation.calibration import run_lco_quantiles

        t0 = time.perf_counter()
        cal = run_lco_quantiles(cell_cycles, featured=raw_fdfs)
        print(f"    quantile LCO            {time.perf_counter() - t0:8.1f}s", flush=True)
        cov = cal.get("rul_interval_coverage_calibrated")
        print(f"      calibrated coverage   {cov}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fleet", help="nasa, severson, zhu2022")
    ap.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    ap.add_argument("--folds", type=int, default=None, help="use only the first N cells")
    ap.add_argument("--workers", type=int, default=None, help="BATLAB_FOLD_WORKERS override")
    ap.add_argument("--cache-info", action="store_true", help="print what is in the fold cache and exit")
    args = ap.parse_args()

    if args.cache_info:
        from batlab.validation import fold_cache

        info = fold_cache.describe()
        print(f"mode={info['mode']}  dir={info['dir']}")
        for entry in info["keys"]:
            print(
                f"  {entry['key']}  {entry['folds']:>3} fold(s)  "
                f"{entry['n_cells']} cell(s)  features={entry['features']}  seed={entry['seed']}"
            )
        return

    if args.workers is not None:
        os.environ["BATLAB_FOLD_WORKERS"] = str(args.workers)

    from batlab._parallel import cpu_budget

    print(f"cpu_budget={cpu_budget()} (os.cpu_count={os.cpu_count()})", flush=True)
    profile(args.fleet, set(args.phase) if args.phase else set(PHASES), args.folds)


if __name__ == "__main__":
    main()
