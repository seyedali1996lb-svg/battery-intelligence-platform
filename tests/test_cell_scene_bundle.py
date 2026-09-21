"""The committed 3D-scene artifacts must be current, complete and self-describing.

The scene's renderer is a build artifact: a 578 kB minified bundle in
``app/static/cell_scene/`` that Streamlit serves straight to the browser
(``server.enableStaticServing``, ``.streamlit/config.toml``) and that CI's Node
job is the only thing able to regenerate. A committed build artifact has three
failure modes, and each one is invisible in review:

* **stale** — someone changed ``frontend/src/scene/geometry.ts``, forgot
  ``npm run build:scene``, and the bundle still draws the old cell;
* **hand-edited** — a "quick fix" applied to the minified file, which is the one
  artifact in this repository that no reviewer can read;
* **orphaned** — the harness page, the sample scene or the manifest renamed, so
  the page 404s at boot while every Python test still passes.

``scripts/export_scene_sample.py --check`` owns the hashing (one implementation,
also runnable by hand), and this file runs it, then checks the properties a hash
cannot see: that the page points at the bundle that exists, that the sample is
the schema version the producer declares, and that the bundle is really the
scene renderer rather than some other file with the right name.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_DIR = REPO_ROOT / "app" / "static" / "cell_scene"
HARNESS = SCENE_DIR / "index.html"
MANIFEST = SCENE_DIR / "manifest.json"
SAMPLE = SCENE_DIR / "sample_scene.json"
BUNDLE = SCENE_DIR / "cell_scene.js"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
import export_scene_sample  # noqa: E402  (the single implementation of the hashing)
from cell_scene import MESH_PART_IDS, SCENE_SCHEMA_VERSION  # noqa: E402


def test_every_committed_artifact_hash_matches_its_sources():
    """The whole freshness argument, in one call."""
    assert MANIFEST.is_file(), "manifest.json is missing — run scripts/export_scene_sample.py"
    assert export_scene_sample.check() == 0


def test_the_bundle_is_present_and_is_the_scene_renderer():
    assert BUNDLE.is_file(), "cell_scene.js has not been built — run `npm run build:scene`"
    text = BUNDLE.read_text(encoding="utf-8")
    # The global a host calls into. If this is absent the bundle is some other
    # file that happens to have the right name.
    assert "CellScene" in text
    # A sourcemap comment would mean the artifact shipped a debugging path that
    # the manifest does not cover.
    assert "sourceMappingURL" not in text
    size = BUNDLE.stat().st_size
    assert 50_000 < size < 2_000_000, f"the bundle is {size} bytes — did the build change shape?"


def test_the_manifest_describes_the_bundle_it_ships():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["bundle"] == BUNDLE.name
    assert manifest["bundleBytes"] == BUNDLE.stat().st_size
    assert manifest["schemaVersion"] == SCENE_SCHEMA_VERSION
    assert manifest["builtBy"].startswith("cd frontend")
    # The library the geometry is drawn with, pinned where a reader can find it.
    assert manifest["threeVersion"]
    sources = manifest["sourceSha256"]
    assert any(name.endswith("src/scene/geometry.ts") for name in sources)
    assert any(name.endswith("src/scene/engine.ts") for name in sources)
    assert any(name.endswith("vite.scene.config.ts") for name in sources)


def test_the_bundle_is_rebuilt_from_every_source_file_that_exists():
    """New files must be recorded, not silently left out of the freshness check."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = set(manifest["sourceSha256"])
    actual = set(export_scene_sample.source_hashes())
    assert recorded == actual, (
        "the manifest and frontend/src/scene disagree about which files the bundle is "
        f"built from: only in manifest {sorted(recorded - actual)}, only on disk {sorted(actual - recorded)}"
    )


def test_the_harness_page_loads_the_bundle_and_the_sample_that_exist():
    page = HARNESS.read_text(encoding="utf-8")
    assert '<script src="cell_scene.js">' in page, "the harness no longer loads the bundle by name"
    assert "sample_scene.json" in page, "the harness no longer falls back to the committed sample"
    assert "/cells/" in page and "/scene" in page, "the harness no longer knows the REST surface"
    # Every path it mentions must be a sibling file that is actually committed.
    for referenced in ("cell_scene.js", "sample_scene.json"):
        assert (SCENE_DIR / referenced).is_file(), f"the harness references missing {referenced}"


def test_the_sample_is_the_current_schema_version_and_carries_its_sources():
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert sample["schemaVersion"] == SCENE_SCHEMA_VERSION
    assert sample["cell"]["id"] == json.loads(MANIFEST.read_text(encoding="utf-8"))["sampleSceneCell"]
    assert sample["record"]["nCycles"] == len(sample["series"]["cycles"])
    assert len(sample["parts"]) == len(MESH_PART_IDS)
    assert sample["disclosures"]


def test_the_sample_validates_against_the_published_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO_ROOT / "docs" / "cell_scene.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(SAMPLE.read_text(encoding="utf-8")), schema)


def test_streamlit_is_configured_to_serve_the_static_assets_this_view_needs():
    """The Streamlit page links `app/static/cell_scene/cell_scene.js` by URL.

    That only resolves because `server.enableStaticServing` is on (the same
    switch `app/static/theme.css` already depends on). If it is ever turned
    off, the 3D view becomes an empty box with no error anywhere.
    """
    config = (REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r"enableStaticServing\s*=\s*true", config), (
        "app/static/ is no longer served, so the scene bundle cannot be fetched"
    )


def test_the_docs_that_explain_this_artifact_exist():
    """A committed binary nobody can regenerate is a liability; the recipe is
    part of the artifact."""
    readme = (SCENE_DIR / "README.md").read_text(encoding="utf-8")
    assert "npm run build:scene" in readme
    assert "npm run test:scene" in readme
    assert "export_scene_sample.py" in readme
