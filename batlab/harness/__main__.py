"""
Command-line entry point for the validation harness.

    python -m batlab.harness --loader batlab.datasets.nasa:load_nasa_cells
    python -m batlab.harness --loader my_lab.data:load_cells \
        --model my_lab.models:make_forecaster --gate floors.json --seal out/bundle

The loader and the model are both 'module:function' import strings, so the
harness can grade code it has never heard of without a plugin system: your
loader returns {cell_id: cycles DataFrame}, your model factory returns a fresh
model per call.

Exit code is 0 unless the metric gate FAILED. A run with no --gate exits 0
while printing NOT CHECKED — nothing was enforced, which the report says out
loud rather than implying a pass.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _load_callable(spec: str, what: str):
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise SystemExit(
            f"--{what} must be 'module.path:function' (e.g. batlab.datasets.nasa:load_nasa_cells)"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"could not import {module_name!r} for --{what}: {exc}. Run this from a "
            "directory where your module is importable (PYTHONPATH=/path/to/pkg)."
        ) from exc
    return getattr(module, func_name, None) or _missing(module_name, func_name)


def _missing(module_name: str, func_name: str):
    raise SystemExit(f"{module_name!r} has no attribute {func_name!r}")


def run_cli(args) -> int:
    """Execute a parsed CLI invocation. Returns a process exit code."""
    from batlab.harness.harness import format_report, seal_bundle, validate_forecaster
    from batlab.validation.lco import unwrap_cell_data

    # The report uses R², ±, and em dashes. A legacy Windows console (cp1252
    # or worse) would raise UnicodeEncodeError mid-report — losing the part
    # of the output that matters — so degrade the glyph, never the report.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - non-reconfigurable stream
            pass

    loader = _load_callable(args.loader, "loader")
    loader_kwargs = json.loads(args.loader_kwargs or "{}")
    loaded = loader(**loader_kwargs)
    if not isinstance(loaded, dict):
        raise SystemExit(
            f"--loader returned {type(loaded).__name__}, expected {{cell_id: DataFrame}}"
        )
    cell_data = unwrap_cell_data(loaded)
    if not cell_data:
        raise SystemExit("--loader returned no cells with a recognizable cycles DataFrame")

    model = _load_callable(args.model, "model") if args.model else None

    splits = ("lco",) if args.no_prospective else ("lco", "prospective")
    dataset = loader_kwargs.get("dataset") or args.loader.split(":")[0].split(".")[-1]

    print(f"grading {len(cell_data)} cell(s) from {args.loader} ...", file=sys.stderr)
    report = validate_forecaster(
        cell_data,
        model=model,
        seed=args.seed,
        splits=splits,
        train_fraction=args.train_fraction,
        intervals=not args.no_intervals,
        gate_path=args.gate,
        dataset=dataset,
    )

    print(format_report(report))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nreport written to {out}")

    if args.seal:
        # loader_kwargs is recorded (not just used) so the verifier replicating
        # this bundle calls the loader with the same arguments rather than
        # guessing — see replication.main's --loader. For a loader that takes
        # no arguments this is an empty dict and changes nothing.
        replication = seal_bundle(
            report, cell_data, args.seal, loader=args.loader,
            loader_kwargs=loader_kwargs or None, dataset=dataset,
        )
        print(
            f"\nsealed bundle written to {args.seal} "
            f"({len(replication.get('files') or {})} file(s) hashed)\n"
            f"  verify: python -m batlab.validation.replication {args.seal} "
            f"--loader {args.loader}"
            + (f" --model {args.model}" if args.model else "")
            + " --recompute"
            + ("\n  (the bundle records --loader-kwargs, so the verifier "
               "supplies them automatically)" if args.loader_kwargs else "")
        )

    return 1 if (report.get("gate") or {}).get("verdict") == "fail" else 0


if __name__ == "__main__":  # pragma: no cover
    from batlab.harness.harness import main

    sys.exit(main())
