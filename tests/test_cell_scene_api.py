"""The REST surface of the 3D cell scene.

`GET /cells/{id}/scene` is the endpoint a renderer that is not this repository's
frontend talks to, so what is tested here is what such a consumer depends on:
that the document validates against the published schema, that it is the *same*
document the Streamlit page builds in-process (one builder, two hosts), that the
static host at `/scene` serves the committed bundle, and that the honesty rules
survive the trip over HTTP — a refused forecast and an unavailable part must
still arrive as refusals, not as nulls a client would render as zero.

The frame map is monkeypatched rather than loaded, on purpose: these tests are
about the *contract*, and a test that has to train a fleet to check a route is a
test that stops being run. The builder's own data behaviour is covered by
tests/test_cell_scene.py.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from api import _create_access_token  # noqa: E402
from cell_scene import MESH_PART_IDS, SCENE_SCHEMA_VERSION, build_cell_scene  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_DIR = REPO_ROOT / "app" / "static" / "cell_scene"
CELL = "SCENE-TEST-CELL"


def _frame(n: int = 160) -> pd.DataFrame:
    """A featured-shaped frame: the columns the scene builder reads."""
    cycles = np.arange(1, n + 1, dtype=float)
    soh = 100.0 - 0.11 * (cycles - 1)
    frame = pd.DataFrame({
        "cycle_number": cycles,
        "soh_pct": soh,
        "capacity_ah": 2.0 * soh / 100.0,
        "resistance_ohm": 0.045 + 0.0002 * cycles,
        "temperature_c": np.full(n, 31.0),
        "sop_pct": 100.0 * 0.045 / (0.045 + 0.0002 * cycles),
        "fade_rate_30cy": np.full(n, 0.0022),
    })
    frame["resistance_normalized"] = frame["resistance_ohm"] / frame["resistance_ohm"].iloc[0]
    return frame


@pytest.fixture()
def client(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(api, "_get_featured_dfs", lambda org_id=None: {CELL: frame})
    monkeypatch.setattr(api, "_get_bundles", lambda org_id=None: {})
    return TestClient(api.app)


@pytest.fixture()
def auth_headers():
    token = _create_access_token({"username": "engineer", "org_id": 1, "role": "engineer"})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def test_the_scene_endpoint_serves_a_spec_any_renderer_can_draw(client, auth_headers):
    response = client.get(f"/cells/{CELL}/scene", headers=auth_headers)
    assert response.status_code == 200
    spec = response.json()

    assert spec["schemaVersion"] == SCENE_SCHEMA_VERSION
    assert spec["cell"]["id"] == CELL
    assert len(spec["parts"]) == len(MESH_PART_IDS)
    assert spec["disclosures"]
    assert spec["record"]["nCycles"] == len(spec["series"]["cycles"]) == 160


def test_the_document_over_http_is_the_one_the_streamlit_page_builds(client, auth_headers):
    """One builder, two hosts. Two code paths would eventually disagree, and the
    disagreement would be a number nobody could reconcile."""
    over_http = client.get(f"/cells/{CELL}/scene", headers=auth_headers).json()
    in_process = build_cell_scene(CELL, _frame())
    assert over_http == in_process


def test_the_response_validates_against_the_published_schema(client, auth_headers):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO_ROOT / "docs" / "cell_scene.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(client.get(f"/cells/{CELL}/scene", headers=auth_headers).json(), schema)


def test_an_unknown_cell_is_a_404_like_every_other_cell_endpoint(client, auth_headers):
    assert client.get("/cells/not-a-cell/scene", headers=auth_headers).status_code == 404


def test_a_horizon_of_zero_returns_the_measured_record_and_says_why(client, auth_headers):
    """The knob a client uses when it wants the past only. It must not quietly
    still serve a projection — or quietly serve an empty future with no reason."""
    spec = client.get(f"/cells/{CELL}/scene", params={"horizon_cycles": 0}, headers=auth_headers).json()
    assert spec["projection"]["available"] is False
    assert spec["projection"]["reason"]
    assert "horizon of zero" in spec["projection"]["reason"]
    assert any("stops at the last measured cycle" in text for text in spec["disclosures"])
    assert spec["record"]["nCycles"] == 160


@pytest.mark.parametrize("horizon", [-1, 5000])
def test_an_out_of_range_horizon_is_rejected_rather_than_clamped(client, auth_headers, horizon):
    """A caller that asked for 5,000 cycles and silently got 2,000 would draw a
    timeline it thinks is longer than it is."""
    response = client.get(
        f"/cells/{CELL}/scene", params={"horizon_cycles": horizon}, headers=auth_headers,
    )
    assert response.status_code == 422


def test_the_scene_requires_authentication_like_every_other_read(client):
    assert client.get(f"/cells/{CELL}/scene").status_code in (401, 403)


def test_refusals_survive_the_trip_over_http(client, auth_headers):
    """A part with no measurement arrives as `available: false` with a reason —
    a client that rendered a null as zero would be drawing a claim nobody made,
    so the document must not make that mistake available to it."""
    spec = client.get(f"/cells/{CELL}/scene", headers=auth_headers).json()
    for part in spec["parts"]:
        if part["available"]:
            continue
        assert part["unavailableReason"], f"{part['id']} arrived unavailable with no reason"
        assert part["value"] is None
        assert part["series"] is None
    if not spec["physics"]["splitIdentified"]:
        assert spec["physics"]["splitReason"]
    if not spec["projection"]["available"]:
        assert spec["projection"]["reason"]


# ---------------------------------------------------------------------------
# The static host
# ---------------------------------------------------------------------------

def test_the_scene_static_host_mounts_and_serves_the_committed_bundle(tmp_path):
    """Same shape as tests/test_api_serves_spa.py: a fake directory, mounted
    into its own app, so the route is proved without depending on a build."""
    (tmp_path / "index.html").write_text("<html>scene host</html>", encoding="utf-8")
    (tmp_path / "cell_scene.js").write_text("window.CellScene={};", encoding="utf-8")

    app = FastAPI()
    assert api.mount_cell_scene(app, directory=str(tmp_path)) is True
    with TestClient(app) as test_client:
        assert test_client.get("/scene/index.html").status_code == 200
        assert test_client.get("/scene/").status_code == 200
        assert test_client.get("/scene/cell_scene.js").status_code == 200


def test_a_missing_directory_is_not_a_boot_failure(tmp_path):
    app = FastAPI()
    assert api.mount_cell_scene(app, directory=str(tmp_path / "nope")) is False


def test_the_real_scene_directory_has_everything_the_host_needs():
    for name in ("index.html", "cell_scene.js", "sample_scene.json", "manifest.json", "README.md"):
        assert (SCENE_DIR / name).is_file(), f"app/static/cell_scene/{name} is missing"


def test_the_api_mounts_the_scene_directory_this_repo_ships():
    """The module-level mount is what a deployment actually gets."""
    assert api.cell_scene_dir() == str(SCENE_DIR)
    assert api._CELL_SCENE_MOUNTED is True
