"""``python -m batlab`` — the library's command line.

Four verbs, all of which are thin wrappers over the public API rather than a
second implementation of it:

    batlab version
    batlab cite [--dataset nasa]
    batlab datasets [--load]
    batlab benchmark --dataset nasa [--out report.json]

``benchmark`` accepts either a friendly ``--dataset`` alias or the explicit
``--loader module:function`` form the validation harness already uses, so the
two CLIs in this package are spelled the same way:

    batlab benchmark --loader batlab.datasets.severson:load_severson_cells

Everything prints to stdout and exits 0/1; ``--out`` writes JSON (numpy scalars
and arrays are encoded, so a report is diff-able and re-loadable). Grading a
model you wrote lives in ``python -m batlab.harness``, which is a bigger tool
with its own flags — this CLI deliberately does not duplicate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Friendly alias -> the loaders in batlab.datasets. Same mapping the docs and
# README use, so a copy-pasted command keeps working.
_DATASET_LOADERS: dict[str, str] = {
    "nasa": "batlab.datasets.nasa:load_nasa_cells",
    "severson": "batlab.datasets.severson:load_severson_cells",
    "zhu2022": "batlab.datasets.zhu2022:load_zhu2022_cells",
    "oxford": "batlab.datasets.oxford:load_oxford_cells",
    "calce": "batlab.datasets.calce:load_calce_cells",
}


def _json_default(value: Any) -> Any:
    """Encode the numeric types a report actually contains.

    numpy scalars/arrays are the whole reason this exists: a report that dies
    with ``Object of type float32 is not JSON serializable`` is not a report.
    """
    if hasattr(value, "tolist"):  # numpy scalar or array
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _resolve_loader(spec: str) -> Any:
    """``module:function`` -> the callable, with a useful error if it is not."""
    import importlib

    if ":" not in spec:
        raise SystemExit(
            f"loader must be 'module:function', got {spec!r} "
            f"(e.g. batlab.datasets.nasa:load_nasa_cells)"
        )
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import loader module {module_name!r}: {exc}") from exc
    loader = getattr(module, func_name, None)
    if loader is None:
        raise SystemExit(f"{module_name!r} has no attribute {func_name!r}")
    return loader


def _cmd_version(_args: argparse.Namespace) -> int:
    import batlab

    print(f"batlab {batlab.__version__}")
    return 0


def _cmd_cite(args: argparse.Namespace) -> int:
    import batlab

    print(batlab.cite(dataset=args.dataset) if args.dataset else batlab.cite())
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    from batlab.datasets import (
        load_calce_cells,
        load_nasa_cells,
        load_oxford_cells,
        load_severson_cells,
        load_zhu2022_cells,
    )

    loaders = {
        "nasa": load_nasa_cells,
        "severson": load_severson_cells,
        "zhu2022": load_zhu2022_cells,
        "oxford": load_oxford_cells,
        "calce": load_calce_cells,
    }
    print("Five public datasets, one standardized schema\n")
    for name, loader in loaders.items():
        first_doc = (loader.__doc__ or "").strip().splitlines()
        summary = first_doc[0] if first_doc else ""
        print(f"  {name:<9} {loader.__module__}:{loader.__name__}")
        if summary:
            print(f"  {'':<9} {summary}")
    if not args.load:
        print("\n(--load actually loads each fleet; that may download source files.)")
        return 0

    print()
    for name, loader in loaders.items():
        # A loader that cannot reach its data returns {} or raises; neither is
        # an error for this listing, so both are reported as "unavailable"
        # next to whatever it said rather than aborting the verb.
        try:
            cells = loader()
        except Exception as exc:  # noqa: BLE001 - a listing must not abort
            print(f"  {name:<9} unavailable: {type(exc).__name__}: {exc}")
            continue
        print(f"  {name:<9} {len(cells)} cells")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from batlab.validation import run_lco

    spec = args.loader or _DATASET_LOADERS.get(args.dataset or "nasa")
    if spec is None:
        raise SystemExit(
            f"unknown dataset {args.dataset!r}; choose one of "
            f"{', '.join(sorted(_DATASET_LOADERS))} or pass --loader module:function"
        )
    loader = _resolve_loader(spec)

    print(f"loading {spec} ...")
    cells = loader()
    if not cells:
        raise SystemExit(
            f"{spec} returned no cells — its raw files may not be present "
            f"(see `batlab datasets`)."
        )
    print(f"  {len(cells)} cells: {', '.join(sorted(cells))}")

    print(f"leave-cell-out validation (seed={args.seed}) ...")
    lco = run_lco(
        cells,
        seed=args.seed,
        include_predictions=args.predictions,
    )

    print("\nleave-cell-out (new-cell generalization)")
    print(f"  SOH R²   {float(lco['soh_r2']):.4f}   MAE {float(lco['soh_mae']):.4f}")
    if lco.get("rul_reliable") and lco.get("rul_r2") is not None:
        print(f"  RUL R²   {float(lco['rul_r2']):.4f}   MAE {float(lco['rul_mae']):.4f}")
    else:
        # Not a failure — the honest state of a fleet with too few observed
        # end-of-life rows. Say so, and say how many there were.
        print(
            "  RUL      not evaluable "
            f"(observed rows: {lco.get('n_rul_observed_rows', 0)} of "
            f"{lco.get('n_rul_observed_rows', 0) + lco.get('n_rul_extrapolated_rows', 0)})"
        )
    fc = lco.get("fold_cache") or {}
    if fc:
        print(
            f"  folds    {fc.get('hits', 0)} reused / {fc.get('fitted', 0)} fitted "
            f"(cache {fc.get('mode', '?')})"
        )

    if args.out:
        payload = {"loader": spec, "seed": args.seed, "lco": lco}
        Path(args.out).write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batlab",
        description="Battery degradation analysis: load a public dataset, "
        "validate a model, print the numbers.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s (run `batlab version`)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print the batlab version")
    p_version.set_defaults(func=_cmd_version)

    p_cite = sub.add_parser("cite", help="BibTeX for the library, or one dataset")
    p_cite.add_argument("--dataset", help="nasa / severson / zhu2022 / oxford / calce")
    p_cite.set_defaults(func=_cmd_cite)

    p_datasets = sub.add_parser("datasets", help="list the supported datasets")
    p_datasets.add_argument(
        "--load", action="store_true", help="also load each fleet and report its cell count"
    )
    p_datasets.set_defaults(func=_cmd_datasets)

    p_bench = sub.add_parser(
        "benchmark", help="leave-cell-out validation of batlab's own GBRT on one fleet"
    )
    p_bench.add_argument(
        "--dataset",
        choices=sorted(_DATASET_LOADERS),
        default=None,
        help="friendly alias (default: nasa)",
    )
    p_bench.add_argument("--loader", help="explicit 'module:function' loader to use instead")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--out", help="write the full result as JSON to this path")
    p_bench.add_argument(
        "--predictions",
        action="store_true",
        help="also compute out-of-fold predictions (slower, larger report)",
    )
    p_bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via python -m batlab
    sys.exit(main())
