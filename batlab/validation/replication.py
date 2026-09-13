"""
Independent replication — a sealed harness a third party runs to verify a
published accuracy number without trusting the publisher's word, code, or
environment.

Why this exists
---------------
A manifest proves a number is REPRODUCIBLE by whoever holds the original
environment; replication asks the harder question: can someone with nothing
but the published bundle and the public data re-derive it? The harness is
deliberately paranoid in the order a skeptic would be:

  1. SEAL      — the bundle's files must match the digests recorded in
                 replication.json at publish time (tamper check; same
                 content-hash discipline as holdout.py's sealed manifest).
  2. IDENTITY  — the third party's own copy of the public dataset must
                 digest byte-identically to the publisher's (catches
                 "same dataset name, different bytes" — the Tier-4 trap).
  3. ENVIRONMENT — library/interpreter versions are compared and every
                 difference is LISTED (warn-level: fits usually survive
                 small version drift, but a reviewer deserves to know).
  4. RECOMPUTE — the LCO evaluation is re-run from the bundle's own seed,
                 feature version, and fold list, and the recomputed
                 headline metrics are compared to the published ones.

Verdict is pass only when every check passes. Every failure says exactly
which check failed and why — "the number doesn't replicate" is useful only
if it names the reason.

Usage (third party):
    python -m batlab.validation.replication <bundle-dir> \
        --loader batlab.datasets.nasa:load_nasa_cells [--recompute]

Publishing side: scripts/publish_replication_bundle.py.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

REPLICATION_SCHEMA = "batlab-replication-bundle"
REPLICATION_SCHEMA_VERSION = 1

# Recomputed-vs-reported tolerance. Same code + seed + byte-identical data
# should agree to near machine precision; anything larger means the
# environment (or the claim) differs.
RECOMPUTE_TOLERANCE = 1e-6

_HEADLINE_METRICS = ("soh_r2", "rul_r2")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(bundle_dir: "str | Path") -> dict:
    """Load and structurally validate a replication bundle directory."""
    bundle_dir = Path(bundle_dir)
    replication_path = bundle_dir / "replication.json"
    if not replication_path.exists():
        raise FileNotFoundError(f"{replication_path} not found — is this a replication bundle directory?")
    replication = json.loads(replication_path.read_text(encoding="utf-8"))
    if replication.get("schema") != REPLICATION_SCHEMA:
        raise ValueError(f"schema {replication.get('schema')!r} is not {REPLICATION_SCHEMA!r}")
    if replication.get("schema_version") != REPLICATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported bundle schema_version {replication.get('schema_version')!r}")
    benchmark_path = bundle_dir / "benchmark.json"
    if benchmark_path.exists():
        replication["_benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return replication


def verify_bundle(
    bundle: dict,
    bundle_dir: "str | Path | None" = None,
    cell_data: "dict | None" = None,
    recompute: bool = False,
) -> dict:
    """Run the four checks. Pure over its inputs (recompute runs run_lco).

    Returns {verdict, checks: [{name, status, detail}]} — status is
    pass / warn / fail; verdict is pass only with zero fails.
    """
    checks: list = []

    # 1. SEAL — files match their published digests.
    files = bundle.get("files") or {}
    if bundle_dir is not None and files:
        mismatches = []
        for fname, digest in sorted(files.items()):
            fpath = Path(bundle_dir) / fname
            if not fpath.exists():
                mismatches.append(f"{fname}: missing")
            else:
                actual = _sha256_file(fpath)
                if actual != digest:
                    mismatches.append(f"{fname}: digest mismatch")
        if mismatches:
            checks.append({"name": "seal", "status": "fail", "detail": "; ".join(mismatches)})
        else:
            checks.append({"name": "seal", "status": "pass",
                           "detail": f"{len(files)} file(s) byte-identical to the published digests"})
    else:
        checks.append({"name": "seal", "status": "warn",
                       "detail": "no files/digests to verify (bundle passed by value?)"})

    # 2. IDENTITY — third party's data digests to the publisher's.
    digests = bundle.get("cell_digests") or {}
    if cell_data is not None and digests:
        from batlab.validation.fingerprints import cell_digest

        changed, missing, extra = [], [], []
        for cid, published in sorted(digests.items()):
            if cid not in cell_data:
                missing.append(cid)
            elif cell_digest(cell_data[cid]) != published:
                changed.append(cid)
        extra = sorted(set(cell_data) - set(digests))  # informational only
        if missing or changed:
            checks.append({"name": "data-identity", "status": "fail",
                           "detail": (f"changed cells: {changed or 'none'}; missing: {missing or 'none'}")})
        else:
            detail = f"{len(digests)} cell(s) byte-identical"
            if extra:
                detail += f" (bundle did not use: {extra})"
            checks.append({"name": "data-identity", "status": "pass", "detail": detail})
    else:
        checks.append({"name": "data-identity", "status": "warn",
                       "detail": "no data supplied — identity unchecked"})

    # 3. ENVIRONMENT — every difference listed, none fatal by itself.
    from batlab.validation.fingerprints import environment_snapshot

    published_env = bundle.get("environment") or {}
    current_env = environment_snapshot()
    diffs = {k: (published_env.get(k), current_env.get(k))
             for k in sorted(set(published_env) | set(current_env))
             if published_env.get(k) != current_env.get(k)}
    if diffs:
        detail = "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diffs.items())
        checks.append({"name": "environment", "status": "warn", "detail": detail})
    else:
        checks.append({"name": "environment", "status": "pass", "detail": "identical"})

    # 4. RECOMPUTE — re-run the evaluation, compare headline metrics.
    if recompute:
        if cell_data is None:
            checks.append({"name": "recompute", "status": "warn",
                           "detail": "no data supplied — cannot recompute"})
        else:
            from batlab.features.engineering import FEATURE_VERSION
            from batlab.validation.lco import run_lco

            if bundle.get("feature_version") != FEATURE_VERSION:
                checks.append({"name": "recompute", "status": "fail",
                               "detail": (f"feature_version {bundle.get('feature_version')!r} != "
                                          f"installed {FEATURE_VERSION!r} — evaluation would not be "
                                          "comparable even if it ran")})
            else:
                cell_ids = bundle.get("cell_ids") or []
                restricted = {cid: cell_data[cid] for cid in cell_ids if cid in cell_data}
                result = run_lco(restricted, seed=int(bundle.get("seed", 42)))
                mismatches = []
                for metric in _HEADLINE_METRICS:
                    reported = (bundle.get("reported") or {}).get(metric)
                    fresh = result.get(metric)
                    # Not-evaluable equivalence: run_lco reports None for a
                    # metric it declined to score; a fresh fold run can
                    # produce float('nan') for the same underlying fact
                    # (no observed-EOL rows). Both mean "not evaluable" —
                    # treat them as matching rather than failing the seal.
                    reported_val: "float | None" = (
                        None
                        if reported is None or (isinstance(reported, float) and reported != reported)
                        else float(reported)  # pyright: ignore[reportArgumentType]
                    )
                    fresh_val: "float | None" = (
                        None
                        if fresh is None or (isinstance(fresh, float) and fresh != fresh)
                        else float(fresh)  # pyright: ignore[reportArgumentType]
                    )
                    if (reported_val is None) != (fresh_val is None):
                        mismatches.append(f"{metric}: reported {reported} vs recomputed {fresh}")
                        continue
                    if reported_val is not None and fresh_val is not None:
                        if abs(reported_val - fresh_val) > RECOMPUTE_TOLERANCE:
                            mismatches.append(
                                f"{metric}: reported {reported_val:.6g} vs recomputed {fresh_val:.6g}")
                if mismatches:
                    checks.append({"name": "recompute", "status": "fail",
                                   "detail": "; ".join(mismatches)})
                else:
                    checks.append({"name": "recompute", "status": "pass",
                                   "detail": "headline metrics reproduce within "
                                             f"{RECOMPUTE_TOLERANCE:g}"})
    else:
        checks.append({"name": "recompute", "status": "warn",
                       "detail": "not requested (pass --recompute to re-run the evaluation)"})

    verdict = "pass" if all(c["status"] != "fail" for c in checks) else "fail"
    return {"verdict": verdict, "checks": checks}


def format_verification(result: dict) -> str:
    """Human-readable verification report for the CLI."""
    lines = [f"replication verify: {result['verdict'].upper()}"]
    for c in result["checks"]:
        lines.append(f"  [{c['status'].upper():4}] {c['name']}: {c['detail']}")
    return "\n".join(lines)


def _load_cell_data(loader_spec: str, loader_kwargs: "dict | None") -> dict:
    """Import 'module:function' and call it — the third party's data path."""
    module_name, _, func_name = loader_spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("--loader must be 'module.path:function' (e.g. batlab.datasets.nasa:load_nasa_cells)")
    loader = getattr(importlib.import_module(module_name), func_name)
    return loader(**(loader_kwargs or {}))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m batlab.validation.replication",
        description="Verify a published replication bundle against your own copy of the public data.",
    )
    parser.add_argument("bundle_dir", help="directory containing replication.json (+ benchmark.json)")
    parser.add_argument("--loader", default=None,
                        help="your data path: 'module:function' returning {cell_id: DataFrame} "
                             "(e.g. batlab.datasets.nasa:load_nasa_cells)")
    parser.add_argument("--recompute", action="store_true",
                        help="re-run the LCO evaluation and compare headline metrics")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle_dir)
    cell_data = None
    if args.loader:
        cell_data = _load_cell_data(args.loader, bundle.get("loader_kwargs"))
    result = verify_bundle(bundle, bundle_dir=args.bundle_dir, cell_data=cell_data,
                           recompute=args.recompute)
    print(format_verification(result))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
