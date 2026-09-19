"""The React SPA's delivery path: src/api.py mounts frontend/dist at /app.

The mount is deliberately conditional on the build existing (the built assets
are generated, not committed), which makes the *interesting* branch — the one
that would break in a deployment — the one a test has to construct. So this
mounts a fake two-file dist into its own FastAPI instance rather than
depending on `npm run build` having been run in whatever checkout the suite
happens to be running in, and it asserts the negative case too (no dist -> no
mount, rather than a half-registered route).

What it does NOT prove: that the real `frontend/dist` exists, is current, or
resolves its own asset paths. That is the frontend CI job's build step plus
its dist-layout check (.github/workflows/ci.yml).
"""

import os
import pathlib
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api as api_mod


def _fake_dist(tmp_path) -> str:
    """A minimal stand-in for `vite build`'s output."""
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><div id=\"root\">Battery Intelligence</div>"
        "<script type=\"module\" src=\"/app/assets/index-abc123.js\"></script></body></html>",
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index-abc123.js").write_text("console.log('spa');", encoding="utf-8")
    return str(tmp_path)


def test_frontend_dist_dir_is_the_vite_output_directory():
    """The path src/api.py mounts has to be the path vite writes to."""
    parts = pathlib.Path(api_mod.frontend_dist_dir()).parts
    assert parts[-2:] == ("frontend", "dist"), parts


def test_mount_spa_serves_index_html_at_app_root(tmp_path):
    app = FastAPI()
    assert api_mod.mount_spa(app, _fake_dist(tmp_path)) is True
    resp = TestClient(app).get("/app/")
    assert resp.status_code == 200
    assert "Battery Intelligence" in resp.text


def test_mount_spa_serves_built_assets(tmp_path):
    """Same-origin delivery only works if the hashed asset paths resolve — the
    build emits /app/assets/... absolute URLs, so the mount point and the
    index.html's own script src have to agree."""
    app = FastAPI()
    api_mod.mount_spa(app, _fake_dist(tmp_path))
    resp = TestClient(app).get("/app/assets/index-abc123.js")
    assert resp.status_code == 200
    assert "spa" in resp.text


def test_mount_spa_is_a_no_op_when_the_build_is_absent(tmp_path):
    """No build -> the API still starts and answers, /app simply 404s. A
    failure to boot the whole REST layer because an optional asset directory
    is missing would be a much worse outcome than a missing SPA."""
    app = FastAPI()
    assert api_mod.mount_spa(app, str(tmp_path / "does-not-exist")) is False
    assert not [r for r in app.routes if getattr(r, "path", "").startswith("/app")]


def test_real_app_mounts_the_spa_when_the_build_exists():
    """The module-level mount is wired to the same helper, not to a second
    hand-rolled copy of it. Skips when frontend/dist has not been built, which
    is the normal state of a fresh Python-only checkout."""
    if not os.path.isdir(api_mod.frontend_dist_dir()):
        return
    mounted = [r for r in api_mod.app.routes if getattr(r, "path", "") == "/app"]
    assert mounted, "frontend/dist exists but the API did not mount it at /app"
