"""Write the committed 3D-scene artifacts: a real sample scene and the manifest.

Two things come out of this script, and both are committed on purpose:

* ``app/static/cell_scene/sample_scene.json`` — a real ``CellSceneSpec`` for one
  real cell, so the static harness next to it works in a fresh checkout with no
  data loaded, no API running and no Node toolchain. It is also the artifact the
  renderer's own tests read: `frontend/src/scene/spec.test.ts` mounts it, so the
  Python producer and the JavaScript renderer are checked against the *same*
  document rather than against two hand-written ideas of it.
* ``app/static/cell_scene/manifest.json`` — the hashes that make a stale or
  hand-edited bundle impossible to commit. ``tests/test_cell_scene_bundle.py``
  recomputes every one of them: the bundle's own digest (hand-edits), the digest
  of every source file the bundle is built from (a source change that was never
  rebuilt), and the sample's digest. The bundle is built by
  ``npm run build:scene`` in ``frontend/``; this script records what came out.

Run order (this is also what ``--check`` verifies, so a fresh clone can tell):

    cd frontend && npm ci && npm run build:scene
    python scripts/export_scene_sample.py --cell B0005

``--check`` recomputes the hashes and exits non-zero on any mismatch without
writing anything, which is what CI-adjacent workflows want.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))  # _paths.py lives at the root
for _p in ("src", "app", "scripts"):
    sys.path.insert(0, str(_root / _p))
import _paths  # noqa: F401  (side-effect: canonical sys.path bootstrap)

SCENE_DIR = _root / "app" / "static" / "cell_scene"
BUNDLE = "cell_scene.js"
MANIFEST = SCENE_DIR / "manifest.json"
SAMPLE = SCENE_DIR / "sample_scene.json"
SOURCE_DIR = _root / "frontend" / "src" / "scene"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Every file the bundle is built from, relative to frontend/.

    Test-only files are excluded: they are never bundled, so editing one must
    not demand a rebuild. Everything that *is* bundled is included, so editing
    one must.
    """
    if not SOURCE_DIR.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(SOURCE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if "__" in path.name or path.suffix not in {".ts", ".tsx", ".css", ".json"}:
            continue
        if path.name.endswith(".test.ts") or path.name == "fixture.ts":
            continue
        out[str(path.relative_to(_root / "frontend")).replace("\\", "/")] = sha256(path)
    for extra in ("vite.scene.config.ts", "package.json"):
        path = _root / "frontend" / extra
        if path.is_file():
            out[extra] = sha256(path)
    return out


def _scene_schema_version() -> int:
    """The one version number, read from the producer that owns it."""
    source = (_root / "src" / "cell_scene.py").read_text(encoding="utf-8")
    match = re.search(r"^SCENE_SCHEMA_VERSION\s*=\s*(\d+)", source, flags=re.MULTILINE)
    if not match:
        raise SystemExit("could not find SCENE_SCHEMA_VERSION in src/cell_scene.py")
    return int(match.group(1))


def _three_version() -> str | None:
    package = _root / "frontend" / "package.json"
    if not package.is_file():
        return None
    for section in ("dependencies", "devDependencies"):
        deps = json.loads(package.read_text(encoding="utf-8")).get(section, {})
        if "three" in deps:
            return str(deps["three"])
    return None


def build_sample(cell_id: str) -> dict:
    """One real cell's scene, built exactly as a host builds it."""
    from cell_scene import build_cell_scene

    import cell_store

    columns = cell_store.available_columns(cell_id)
    if columns is None:
        raise SystemExit(
            f"{cell_id!r} is not in the cell store on this machine, so no sample scene can be "
            "exported from real data. Load a fleet first (app/main.py's loader populates the "
            "store), or pass --cell with a cell id that is present."
        )
    wanted = [
        "cycle_number", "soh_pct", "capacity_ah", "resistance_ohm",
        "resistance_normalized", "temperature_c", "sop_pct", "fade_rate_30cy",
    ]
    df = cell_store.get_cell_df(cell_id, columns=[c for c in wanted if c in columns])
    if df is None or len(df) == 0:
        raise SystemExit(f"the cell store returned no rows for {cell_id!r}")
    return build_cell_scene(cell_id, df)


def write_manifest(extra_sample_cell: str | None = None) -> dict:
    manifest: dict = {
        "bundle": BUNDLE,
        "bundleSha256": sha256(SCENE_DIR / BUNDLE),
        "bundleBytes": (SCENE_DIR / BUNDLE).stat().st_size,
        "schemaVersion": _scene_schema_version(),
        "threeVersion": _three_version(),
        "builtBy": "cd frontend && npm run build:scene (frontend/vite.scene.config.ts)",
        "sourceSha256": source_hashes(),
        "sampleScene": SAMPLE.name,
        "sampleSceneSha256": sha256(SAMPLE) if SAMPLE.is_file() else None,
        "sampleSceneCell": extra_sample_cell,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def check() -> int:
    """Recompute every recorded hash without writing anything."""
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST.relative_to(_root)} — run the export script to create it")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []

    bundle = SCENE_DIR / manifest.get("bundle", BUNDLE)
    if not bundle.is_file():
        problems.append(f"{bundle.name} is missing")
    elif sha256(bundle) != manifest.get("bundleSha256"):
        problems.append(f"{bundle.name} does not match its recorded digest (stale or hand-edited)")

    recorded = manifest.get("sourceSha256") or {}
    current = source_hashes()
    for name, digest in current.items():
        if name not in recorded:
            problems.append(f"{name} is new and is not in the manifest — rebuild the bundle")
        elif recorded[name] != digest:
            problems.append(f"{name} changed since the bundle was built — rebuild the bundle")
    for name in recorded:
        if name not in current:
            problems.append(f"{name} is in the manifest but no longer exists — rebuild the bundle")

    if SAMPLE.is_file() and manifest.get("sampleSceneSha256") != sha256(SAMPLE):
        problems.append("sample_scene.json does not match its recorded digest")

    version = _scene_schema_version()
    if manifest.get("schemaVersion") != version:
        problems.append(
            f"the manifest records schema version {manifest.get('schemaVersion')} but "
            f"src/cell_scene.py declares {version}"
        )

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("cell scene artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", default="B0005", help="cell id to export the sample scene for")
    parser.add_argument("--check", action="store_true", help="verify the recorded hashes instead of writing")
    args = parser.parse_args()

    if args.check:
        return check()

    if not (SCENE_DIR / BUNDLE).is_file():
        raise SystemExit(
            f"{BUNDLE} is not built yet. Run:\n  cd frontend && npm ci && npm run build:scene"
        )

    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    spec = build_sample(args.cell)
    SAMPLE.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    manifest = write_manifest(args.cell)
    print(
        f"wrote {SAMPLE.relative_to(_root)} ({args.cell}, "
        f"{spec['record']['nCycles']} cycles, schema v{spec['schemaVersion']})\n"
        f"wrote {MANIFEST.relative_to(_root)} "
        f"(bundle {manifest['bundleBytes']} bytes, three {manifest['threeVersion']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
