"""
Publish a sealed replication bundle for one dataset's LCO number.

A replication bundle is everything a skeptical third party needs to verify
a headline number WITHOUT access to this repo's working state: the fold
structure and reported metrics (benchmark.json), the per-cell content
digests and environment snapshot (from run_lco's fingerprint), and the
SHA-256 of every file in the bundle itself (replication.json's "files"
map — the seal; same discipline as the Tier-0 holdout manifest).

The third party runs:
    python -m batlab.validation.replication <bundle-dir> \
        --loader batlab.datasets.<mod>:<fn> --recompute

and gets pass/fail per check: seal (files unmodified), data-identity
(their copy of the public data is byte-identical), environment (version
differences listed), recompute (the number re-derives from their data).

Example:
    python scripts/publish_replication_bundle.py --dataset zhu2022 \
        --loader batlab.datasets.zhu2022:load_zhu2022_cells --out replication/zhu2022
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))  # _paths.py lives at the root
for _p in ("src", "app", "scripts"):
    sys.path.insert(0, str(_root / _p))
import _paths  # noqa: F401  (side-effect: canonical sys.path bootstrap)

REPLICATION_SCHEMA = "batlab-replication-bundle"
REPLICATION_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(dataset: str, loader, loader_kwargs: dict, seed: int, out: Path) -> dict:
    """`loader` is either a 'module.path:function' spec string (the CLI path)
    or a callable returning {cell_id: DataFrame} (programmatic use)."""
    from batlab.validation.fingerprints import dataset_fingerprint, environment_snapshot
    from batlab.validation.lco import unwrap_cell_data, run_lco
    from batlab.validation.manifest import export_benchmark_results

    if callable(loader):
        loader_spec = getattr(loader, "__name__", "<callable>")
        loader_record = {"function": loader_spec, "kwargs": loader_kwargs or {}}
    else:
        module_name, _, func_name = str(loader).partition(":")
        if not module_name or not func_name:
            raise ValueError("--loader must be 'module.path:function'")
        loader = getattr(__import__(module_name, fromlist=[func_name]), func_name)
        loader_spec = str(loader)
        loader_record = {"module": module_name, "function": func_name, "kwargs": loader_kwargs or {}}

    print(f"Loading {dataset} via {loader_spec} ...")
    loaded = loader(**(loader_kwargs or {}))
    if not isinstance(loaded, dict):
        raise TypeError(f"loader returned {type(loaded).__name__}, expected {{cell_id: DataFrame}}")
    cell_data = unwrap_cell_data(loaded)
    if len(cell_data) < 2:
        raise ValueError(f"{dataset} loaded {len(cell_data)} cells — a bundle needs at least 2")

    print(f"Running leave-cell-out on {len(cell_data)} cells ...")
    result = run_lco(cell_data, seed=seed)
    fingerprint = result["fingerprint"]

    benchmark = export_benchmark_results(
        result, out / "benchmark.json",
        cell_ids=sorted(cell_data), seed=seed,
    )

    reported = {k: result.get(k) for k in ("soh_r2", "soh_mae", "rul_r2", "rul_mae", "rul_reliable")}
    replication = {
        "schema": REPLICATION_SCHEMA,
        "schema_version": REPLICATION_SCHEMA_VERSION,
        "dataset": dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_version": benchmark["feature_version"],
        "cell_ids": sorted(cell_data),
        "cell_digests": fingerprint["dataset"]["cell_digests"],
        "environment": fingerprint["environment"],
        "reported": reported,
        "loader": loader_record,
        "notes": (
            "Fold structure and metrics in benchmark.json. Recompute with "
            "python -m batlab.validation.replication <dir> --loader "
            f"{loader_spec} --recompute"
        ),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "replication.json").write_text(
        json.dumps(replication, indent=2, default=str) + "\n", encoding="utf-8")

    # Seal LAST: every file in the bundle except replication.json itself is
    # digested into replication.json (the manifest cannot contain its own
    # hash; it is the root of trust).
    files = {p.name: _sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "replication.json"}
    replication["files"] = files
    (out / "replication.json").write_text(
        json.dumps(replication, indent=2, default=str) + "\n", encoding="utf-8")
    return replication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset label recorded in the bundle")
    parser.add_argument("--loader", required=True,
                        help="'module:function' returning {cell_id: DataFrame}")
    parser.add_argument("--loader-kwargs", default="{}",
                        help="JSON dict of keyword args for the loader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="output bundle directory")
    args = parser.parse_args()

    loader_kwargs = json.loads(args.loader_kwargs)
    replication = build_bundle(args.dataset, args.loader, loader_kwargs, args.seed, Path(args.out))

    print(f"\nBundle written to {args.out}")
    print(f"  reported: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in replication['reported'].items()} }")
    print(f"  sealed:   {len(replication['files'])} file(s)")
    print(f"  verify:   python -m batlab.validation.replication {args.out} "
          f"--loader {args.loader} --recompute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
