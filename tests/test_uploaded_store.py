"""
Tests for src/uploaded_store.py — the raw cycles of a tenant's own upload.

This is the module that decides whether an uploaded fleet can be graded by the
validation harness and sealed into a bundle at all, so the failures that matter
are not "the file is missing" but:

1. A key becomes a directory name. It must never escape the store: `../`,
   separators, absolute paths and empty keys are all refused, on write AND on
   read, so no caller — a form field, a URL, a future page — can reach another
   tenant's data or the filesystem around it.
2. The store is content-addressed and fingerprinted. Re-saving the same upload
   must be idempotent and must not change the fingerprint; a store that drifted
   on disk (a hand-edit, a partial write, a file copied from another upload)
   must be caught as a named identity failure rather than silently graded.
3. org_id is a tenant boundary, not a label: a manifest belonging to another
   org is refused, and another org's uploads never appear in a picker.
4. The manifest is what the app offers fleets from, so a corrupt one must read
   as "not available" and must not take the other fleets down with it.
"""

import json

import pytest
from conftest import make_cycles_df

import uploaded_store as us


def _fleet(n_cells: int = 3, n_cycles: int = 90) -> dict:
    return {
        f"Cell{i + 1}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=0.0006 * (1.0 + 0.1 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Never touch the deployment's real store (or a developer's own uploads)."""
    monkeypatch.setattr(us, "UPLOADED_STORE_DIR", tmp_path / "uploaded_fleets")
    return tmp_path / "uploaded_fleets"


# ---------------------------------------------------------------------------
# 1. Keys are directory names, and are treated as such
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key",
    ["../../etc/passwd", "a/b", "..", "", "   ", ".", "with space", "C:\\windows",
     "/absolute/path", "x" * 200, "null\x00byte"],
)
def test_a_key_that_is_not_a_plain_identifier_is_refused_everywhere(key, isolated_store):
    fleet = _fleet(2)
    with pytest.raises(us.UploadedFleetError):
        us.save_uploaded_cell_data(1, key, fleet)
    with pytest.raises(us.UploadedFleetError):
        us.load_uploaded_cell_data(key)
    with pytest.raises(us.UploadedFleetError):
        us.store_dir(key)
    # A refused key must not have created anything, anywhere.
    assert not isolated_store.exists() or not list(isolated_store.iterdir())


def test_a_valid_key_writes_only_inside_the_store(isolated_store):
    key = "upload-a1b2c3d4e5f6a7b8c9d0"
    us.save_uploaded_cell_data(1, key, _fleet(2))
    assert us.store_dir(key).parent == isolated_store
    assert (us.store_dir(key) / "cells" / "index.json").is_file()
    assert (us.store_dir(key) / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# 2. Content-addressed, fingerprinted, idempotent, tamper-evident
# ---------------------------------------------------------------------------

def test_save_then_load_round_trips_the_exact_frames(isolated_store):
    fleet = _fleet(3)
    manifest = us.save_uploaded_cell_data(1, "upload-deadbeefdeadbeef0000", fleet,
                                          meta={"n_cells": 3})

    reloaded = us.load_uploaded_cell_data("upload-deadbeefdeadbeef0000", expected_org_id=1)
    assert set(reloaded) == set(fleet)
    for cell_id, df in fleet.items():
        assert reloaded[cell_id]["capacity_ah"].equals(df["capacity_ah"])

    assert manifest["schema"] == us.STORE_SCHEMA
    assert manifest["org_id"] == 1
    assert manifest["n_cells"] == 3
    assert manifest["n_rows"] == 270
    assert manifest["cell_ids"] == sorted(fleet)
    assert manifest["fingerprint"]["dataset_sha256"]
    assert manifest["context"] == {"n_cells": 3}


def test_the_manifest_fingerprint_describes_the_files_on_disk(isolated_store):
    """cell_digests in every report and bundle come from this value."""
    from batlab.validation.fingerprints import cell_digest, dataset_fingerprint

    fleet = _fleet(2)
    key = "upload-11112222333344445555"
    manifest = us.save_uploaded_cell_data(1, key, fleet)

    assert manifest["fingerprint"] == dataset_fingerprint(fleet)
    for entry in json.loads((us.store_dir(key) / "cells" / "index.json").read_text())["cells"]:
        assert entry["cell_digest"] == cell_digest(fleet[entry["cell_id"]])


def test_resaving_the_same_upload_is_idempotent(isolated_store):
    key = "upload-aaaabbbbccccddddeeee"
    fleet = _fleet(2)
    first = us.save_uploaded_cell_data(1, key, fleet)
    second = us.save_uploaded_cell_data(1, key, fleet)

    assert first["fingerprint"] == second["fingerprint"]
    assert len([p for p in isolated_store.iterdir() if p.is_dir()]) == 1


def test_a_cell_file_edited_on_disk_is_refused_by_name(isolated_store):
    key = "upload-99998888777766665555"
    us.save_uploaded_cell_data(1, key, _fleet(2))
    target = us.store_dir(key) / "cells" / "Cell1.csv"

    lines = target.read_text().splitlines()
    target.write_bytes(("\n".join([lines[0]] + [lines[1] + "9"] + lines[2:]) + "\n").encode())

    with pytest.raises(us.UploadedFleetError) as exc:
        us.load_uploaded_cell_data(key, expected_org_id=1)
    assert "unusable" in str(exc.value)
    assert "does not match the digest its own index records" in str(exc.value)


def test_a_store_that_drifted_from_its_manifest_fingerprint_is_refused(isolated_store):
    """Even when the index agrees with the file, the manifest may not.

    This is the case a digest-per-file check alone would miss: a whole cell
    table replaced and its index entry updated to match it.
    """
    key = "upload-abcabcabcabcabcabcab"
    us.save_uploaded_cell_data(1, key, _fleet(2))
    directory = us.store_dir(key)

    from batlab.validation.bundle_data import write_bundle_cells

    replaced = {"Cell1": make_cycles_df(n_cycles=90, fade_per_cycle=0.05), "Cell2": _fleet(2)["Cell2"]}
    write_bundle_cells(replaced, directory / "cells")  # index rewritten to match

    with pytest.raises(us.UploadedFleetError) as exc:
        us.load_uploaded_cell_data(key, expected_org_id=1)
    assert "no longer match their manifest fingerprint" in str(exc.value)


def test_saving_nothing_or_a_non_dataframe_is_refused(isolated_store):
    with pytest.raises(us.UploadedFleetError):
        us.save_uploaded_cell_data(1, "upload-00000000000000000000", {})
    with pytest.raises(us.UploadedFleetError):
        us.save_uploaded_cell_data(1, "upload-00000000000000000000", {"C1": [1, 2, 3]})


def test_loading_an_upload_that_was_never_persisted_explains_itself(isolated_store):
    with pytest.raises(us.UploadedFleetError) as exc:
        us.load_uploaded_cell_data("upload-ffffffffffffffffffff")
    assert "No persisted raw cycles" in str(exc.value)
    assert us.read_manifest("upload-ffffffffffffffffffff") is None


# ---------------------------------------------------------------------------
# 3. org_id is a boundary
# ---------------------------------------------------------------------------

def test_another_orgs_upload_is_refused_and_never_offered(isolated_store):
    key = "upload-12341234123412341234"
    us.save_uploaded_cell_data(7, key, _fleet(2))

    assert us.list_uploaded_fleets(1) == []
    assert us.latest_uploaded_fleet(1) is None
    assert [m["upload_key"] for m in us.list_uploaded_fleets(7)] == [key]

    with pytest.raises(us.UploadedFleetError) as exc:
        us.load_uploaded_cell_data(key, expected_org_id=1)
    assert "belongs to another organization" in str(exc.value)
    # ... and with no session to check against (scripts/tests) it still loads.
    assert set(us.load_uploaded_cell_data(key)) == {"Cell1", "Cell2"}


# ---------------------------------------------------------------------------
# 4. The catalogue the app's picker is built from
# ---------------------------------------------------------------------------

def test_fleets_are_listed_newest_first_and_survive_a_corrupt_entry(isolated_store):
    import datetime

    us.save_uploaded_cell_data(1, "upload-00000000000000000001", _fleet(2))
    us.save_uploaded_cell_data(1, "upload-00000000000000000002", _fleet(3))
    newest = us.store_dir("upload-00000000000000000002") / "manifest.json"
    manifest = json.loads(newest.read_text())
    manifest["created_utc"] = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    ).isoformat()
    newest.write_text(json.dumps(manifest))

    # A directory that is not a valid store entry at all.
    junk = isolated_store / "upload-00000000000000000003"
    junk.mkdir(parents=True)
    (junk / "manifest.json").write_text("{not json")

    keys = [m["upload_key"] for m in us.list_uploaded_fleets(1)]
    assert keys == ["upload-00000000000000000002", "upload-00000000000000000001"]
    assert us.latest_uploaded_fleet(1)["upload_key"] == keys[0]


def test_a_manifest_declaring_the_wrong_schema_is_ignored(isolated_store):
    key = "upload-5a5a5a5a5a5a5a5a5a5a"
    us.save_uploaded_cell_data(1, key, _fleet(2))
    path = us.store_dir(key) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "some-other-format"
    path.write_text(json.dumps(manifest))

    assert us.read_manifest(key) is None
    assert us.list_uploaded_fleets(1) == []


def test_clearing_removes_the_cycles_and_nothing_else(isolated_store):
    keep = "upload-77777777777777777777"
    drop = "upload-88888888888888888888"
    us.save_uploaded_cell_data(1, keep, _fleet(2))
    us.save_uploaded_cell_data(1, drop, _fleet(2))

    assert us.clear_uploaded_cell_data(drop) is True
    assert us.clear_uploaded_cell_data(drop) is False   # already gone
    assert us.read_manifest(drop) is None
    assert us.read_manifest(keep) is not None
    with pytest.raises(us.UploadedFleetError):
        us.clear_uploaded_cell_data("../nope")
