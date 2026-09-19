"""One-off: rerun the Severson benchmark studies on the expanded 46-cell
population, through the exact runner functions the app background thread
uses, reusing the features cache. The 12-cell study rows must not sit next
to the 46-cell GBRT row on the Benchmark page — mixed populations would be
the same comparability failure this repo documents elsewhere."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

import db  # noqa: E402
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    import experiment_registry as reg
    from bundle_cache import load_features_cached
    from batlab.datasets.severson import load_severson_cells

    sev = load_severson_cells(status_fn=lambda m: None)
    cells = {cid: {"cycles": df} for cid, df in sev.items()}
    print(f"severson cells: {len(cells)}")

    feat = load_features_cached("severson", cells)
    df_featured, model_inputs = (None, None)
    if feat is not None:
        df_featured, model_inputs = feat
        print("features cache hit")
    else:
        import app._data as app_data
        df_featured, model_inputs = app_data.compute_features_only(cells)
        from bundle_cache import save_features_cached
        save_features_cached("severson", cells, df_featured, model_inputs)
        print("features rebuilt")

    datasets = {"severson": cells}
    featured_lco = {"severson": model_inputs}
    featured_df = {"severson": df_featured}

    print("== prospective ==")
    for row in reg.run_prospective_benchmark_study(
        datasets, featured=featured_df, refresh=True,
    ):
        print("  ", row)

    print("== modeling (hierarchical + ensemble) ==")
    for row in reg.run_modeling_benchmark_study(
        datasets, featured=featured_df,
        baselines={"severson": None}, refresh=True,
    ):
        print("  ", row)

    print("== robustness ==")
    for row in reg.run_robustness_study(datasets, refresh=True):
        print("  ", row)

    print("\n== registry state (severson, v12) ==")
    for r in reg.leaderboard(tenant_org_id=None):
        ds = r.get("dataset") or ""
        if ds.startswith("severson") and r.get("feature_version", "").startswith("v12"):
            soh = r.get("soh_r2")
            print(f"{ds:<24} {(r.get('model_kind') or 'gbrt'):<13} "
                  f"soh={None if soh is None else round(soh, 4)} "
                  f"{str(r.get('timestamp'))[:19]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
