"""
Tests for batlab/validation/bundle_data.py + sealing a bundle that CARRIES its
own raw cycles (batlab.harness.seal_bundle(embed_data=True)).

What these tests defend, in order of how much damage their absence would do:

1. A bundle that carries its data must REPRODUCE its own number, not
   approximate it. If the embedded table loses precision on the way out, the
   recompute check fails on a bundle that is correct — and the whole point of
   embedding is that a third party re-derives the metric with nothing else in
   hand. Proven here at abs=0.0 against the original fit, not at a tolerance.
2. The data-identity check must still pass: reloaded frames must digest to the
   same cell_digests the bundle records.
3. Tampering must be caught, and caught by NAME — a cell file edited after
   sealing fails the seal; an index pointing outside its own directory is
   refused rather than read; a file whose bytes disagree with the index is
   refused before it is scored.
4. Sealing without embed_data must not change what the existing (public
   reference fleet) bundles look like — two files, no cells/ — so this
   addition cannot move the platform's published artifacts.
"""

import hashlib
import io
import json
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_cycles_df

from batlab.harness import seal_bundle, validate_forecaster
from batlab.validation.bundle_data import (
    CELLS_INDEX_NAME,
    cell_file_name,
    load_bundle_cells,
    read_bundle_cell_file,
    table_csv_text,
    write_bundle_cells,
)
from batlab.validation.fingerprints import cell_csv_text, cell_digest, dataset_fingerprint
from batlab.validation.lco import run_lco
from batlab.validation.replication import (
    _load_cell_data,
    format_verification,
    load_bundle,
    verify_bundle,
)


def _fleet(n_cells: int = 3, n_cycles: int = 180) -> dict:
    """Cells that fade differently, so folds and features are not symmetric."""
    return {
        f"C{i}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=0.005 * (1.0 + 0.08 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


@pytest.fixture(scope="module")
def fleet() -> dict:
    return _fleet()


@pytest.fixture(scope="module")
def report(fleet) -> dict:
    """A real harness report (leave-cell-out only — nothing else is needed)."""
    return validate_forecaster(fleet, dataset="synth-eol", splits=("lco",), intervals=False)


# ---------------------------------------------------------------------------
# 1. The round trip is exact, so the recompute check can be exact
# ---------------------------------------------------------------------------

def test_embedded_round_trip_reproduces_the_fit_bit_for_bit(fleet, tmp_path):
    """The load-bearing property: same bytes in, same number out."""
    write_bundle_cells(fleet, tmp_path / "cells")
    reloaded = load_bundle_cells("cells", bundle_dir=tmp_path)

    original = run_lco(fleet, seed=42)
    again = run_lco(reloaded, seed=42)
    for metric in ("soh_r2", "rul_r2", "soh_mae", "rul_mae"):
        assert again[metric] == pytest.approx(original[metric], abs=0.0), metric
    assert again["rul_reliable"] == original["rul_reliable"]


def test_embedded_round_trip_preserves_the_dataset_fingerprint(fleet, tmp_path):
    """cell_digests are what the data-identity check compares — they must hold."""
    write_bundle_cells(fleet, tmp_path / "cells")
    reloaded = load_bundle_cells("cells", bundle_dir=tmp_path)

    assert set(reloaded) == set(fleet)
    for cell_id, df in fleet.items():
        assert cell_digest(reloaded[cell_id]) == cell_digest(df), cell_id
    assert (
        dataset_fingerprint(reloaded)["dataset_sha256"]
        == dataset_fingerprint(fleet)["dataset_sha256"]
    )


def test_the_index_records_both_the_file_hash_and_the_frame_digest(fleet, tmp_path):
    """A reviewer must not have to guess which digest they are comparing.

    `sha256` is the file's own bytes (checkable with sha256sum); `cell_digest`
    is the canonical fingerprint — the same value every report, registry row,
    and sealed bundle records for that cell.
    """
    index = write_bundle_cells(fleet, tmp_path / "cells")
    written = {entry["cell_id"]: entry for entry in index["cells"]}
    assert set(written) == set(fleet)

    for cell_id, entry in written.items():
        raw = (tmp_path / "cells" / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert entry["cell_digest"] == cell_digest(fleet[cell_id])
        assert entry["n_rows"] == len(fleet[cell_id])
    # Two different digests, and saying so here is the point.
    assert written["C0"]["sha256"] != written["C0"]["cell_digest"]


def test_written_tables_are_lossless_where_the_fingerprint_form_is_not(fleet):
    """The two serializations differ deliberately; this pins WHY.

    `cell_csv_text` (the fingerprint's %.10g form) is lossy enough that
    re-scoring a table read back from it lands outside the replication
    tolerance. If storage is ever "simplified" to reuse it, this test states
    what breaks.
    """
    df = fleet["C0"]
    canonical = cell_csv_text(df)
    stored = table_csv_text(df)
    assert canonical != stored

    from_fingerprint_form = pd.read_csv(io.StringIO(canonical), float_precision="round_trip")
    from_stored_form = pd.read_csv(io.StringIO(stored), float_precision="round_trip")
    assert not from_fingerprint_form["resistance_ohm"].equals(df["resistance_ohm"])
    assert from_stored_form["resistance_ohm"].equals(df["resistance_ohm"])


def test_the_reader_pins_round_trip_float_parsing(fleet, tmp_path):
    """pandas' default float converter is lossy; the loader must not use it.

    Measured on pandas 3.0.3: reading these files with the default converter
    moves values by 1 ULP and a recompute of an embedded bundle lands ~6e-5
    from its own reported R². That is sixty times the replication tolerance, on
    data that is byte-identical — a failure with no visible cause.
    """
    write_bundle_cells(fleet, tmp_path / "cells")
    cell = load_bundle_cells("cells", bundle_dir=tmp_path)["C0"]

    raw = (tmp_path / "cells" / "C0.csv").read_text()
    lossy_parser = pd.read_csv(io.StringIO(raw))
    assert not lossy_parser["resistance_ohm"].equals(cell["resistance_ohm"])
    assert cell["resistance_ohm"].equals(fleet["C0"]["resistance_ohm"])


# ---------------------------------------------------------------------------
# 2 + 3. Sealing with the data inside, and how that gets tampered with
# ---------------------------------------------------------------------------

def test_sealed_bundle_with_embedded_data_verifies_from_another_directory(report, fleet, tmp_path):
    """bundle-relative means exactly that: unzip it anywhere and verify.

    This is the whole claim of embed_data=True.
    """
    sealed = seal_bundle(report, fleet, tmp_path / "published", dataset="synth-eol",
                         embed_data=True)
    assert sealed["loader"]["module"] == "batlab.validation.bundle_data"
    assert sealed["loader"]["function"] == "load_bundle_cells"
    assert sealed["loader"]["kwargs"] == {"cells_dir": "cells"}
    assert sealed["embedded_cells"]["n_cells"] == len(fleet)
    assert f"cells/{CELLS_INDEX_NAME}" in sealed["files"]

    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(tmp_path / "published", moved)

    loader_spec = f"{sealed['loader']['module']}:{sealed['loader']['function']}"
    cell_data = _load_cell_data(loader_spec, sealed["loader"]["kwargs"], bundle_dir=moved)
    assert set(cell_data) == set(fleet)

    result = verify_bundle(load_bundle(moved), bundle_dir=moved, cell_data=cell_data,
                           recompute=True)
    assert result["verdict"] == "pass", format_verification(result)


def test_real_enriched_cycles_round_trip_and_verify(tmp_path):
    """The synthetic fixtures above are tidy; real loader output is not.

    An enriched reference frame carries bool (`is_eol`), str (`chemistry`),
    sentinel-bearing and rolling-mean columns — the shapes most likely to
    break a CSV round trip or flip a digest. This is the same path a tenant's
    uploaded cycles take, since import_adapter calls the same enrich_cycles().
    """
    import experiment_registry as reg

    loaded = reg.reload_reference_cell_data("synth")
    cells = {cid: loaded[cid] for cid in sorted(loaded)[:3]}
    assert cells, "the synthetic fleet must be available for this test"

    report = validate_forecaster(cells, dataset="synth", splits=("lco",), intervals=False)
    sealed = seal_bundle(report, cells, tmp_path, dataset="synth", embed_data=True)

    loader_spec = f"{sealed['loader']['module']}:{sealed['loader']['function']}"
    reloaded = _load_cell_data(loader_spec, sealed["loader"]["kwargs"], bundle_dir=tmp_path)

    for cell_id, df in cells.items():
        assert cell_digest(reloaded[cell_id]) == cell_digest(df), cell_id
    original = run_lco(cells, seed=42)
    again = run_lco(reloaded, seed=42)
    assert again["soh_r2"] == pytest.approx(original["soh_r2"], abs=0.0)
    assert again["rul_r2"] == pytest.approx(original["rul_r2"], abs=0.0)

    result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path, cell_data=reloaded,
                           recompute=True)
    assert result["verdict"] == "pass", format_verification(result)


def test_a_cell_edited_after_sealing_fails_both_gates(report, fleet, tmp_path):
    """Two independent refusals, each naming what moved."""
    seal_bundle(report, fleet, tmp_path, dataset="synth-eol", embed_data=True)
    target = tmp_path / "cells" / "C1.csv"

    lines = target.read_text().splitlines()
    tampered = "\n".join([lines[0], lines[1] + "9"] + lines[2:]) + "\n"
    target.write_bytes(tampered.encode("utf-8"))

    # (a) the loader refuses to score data that disagrees with its own index
    with pytest.raises(ValueError) as exc:
        load_bundle_cells("cells", bundle_dir=tmp_path)
    assert "does not match the digest its own index records" in str(exc.value)

    # (b) the seal check fails too — the cell files are sealed as well
    result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path,
                           cell_data=fleet, recompute=True)
    assert result["verdict"] == "fail"
    seal_check = next(c for c in result["checks"] if c["name"] == "seal")
    assert seal_check["status"] == "fail"
    assert "C1.csv" in seal_check["detail"]


def test_an_index_pointing_outside_its_directory_is_refused(fleet, tmp_path):
    """A bundle is attacker-controlled input at verification time."""
    index = write_bundle_cells(fleet, tmp_path / "cells")
    (tmp_path / "secret.csv").write_bytes(b"cycle_number,capacity_ah\n1,2.0\n")

    index["cells"][0]["file"] = "../secret.csv"
    (tmp_path / "cells" / CELLS_INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_bundle_cells("cells", bundle_dir=tmp_path)
    assert "not a plain file name inside the bundle" in str(exc.value)


@pytest.mark.parametrize(
    ("edit", "expected"),
    [("missing_file", "but that file is missing"), ("bad_schema", "declares schema")],
)
def test_a_broken_or_mislabelled_index_says_what_is_wrong(fleet, tmp_path, edit, expected):
    index = write_bundle_cells(fleet, tmp_path / "cells")
    if edit == "missing_file":
        index["cells"][0]["file"] = "not-here.csv"
    else:
        index["schema"] = "something-else"
    (tmp_path / "cells" / CELLS_INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_bundle_cells("cells", bundle_dir=tmp_path)
    assert expected in str(exc.value)


def test_reading_a_bundle_without_embedded_cells_says_how_it_was_sealed(report, fleet, tmp_path):
    """The failure a verifier will actually hit, with the reason it happened."""
    seal_bundle(report, fleet, tmp_path, dataset="synth-eol")  # embed_data=False
    with pytest.raises(ValueError) as exc:
        load_bundle_cells("cells", bundle_dir=tmp_path)
    assert "embed_data=False" in str(exc.value)


def test_reading_a_single_file_verifies_its_bytes_first(fleet, tmp_path):
    """Hashing after scoring would score the wrong data. Hash first."""
    write_bundle_cells(fleet, tmp_path / "cells")
    path = tmp_path / "cells" / "C0.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # check_dtype=False is deliberate: dtypes are re-inferred by the reader (a
    # whole-number float column comes back int64), and that is the behaviour we
    # WANT — cast them back and a bool column that gained a NaN turns True.
    # Values, digests and fits are what have to survive, and they do.
    pd.testing.assert_frame_equal(
        read_bundle_cell_file(path, expected_sha256=digest), fleet["C0"], check_dtype=False
    )
    with pytest.raises(ValueError) as exc:
        read_bundle_cell_file(path, expected_sha256="0" * 64)
    assert "was changed after this bundle was sealed" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. The default path is untouched
# ---------------------------------------------------------------------------

def test_sealing_without_embed_data_writes_exactly_the_old_files(report, fleet, tmp_path):
    """No cells/, no embedded_cells: existing bundles are unchanged by this."""
    sealed = seal_bundle(report, fleet, tmp_path, dataset="synth-eol",
                         loader="batlab.datasets.nasa:load_nasa_cells",
                         loader_kwargs={"dataset": "nasa"})
    assert sorted(sealed["files"]) == ["benchmark.json", "harness_report.json"]
    assert "embedded_cells" not in sealed
    assert not (tmp_path / "cells").exists()
    assert sealed["loader"]["module"] == "batlab.datasets.nasa"
    assert sealed["loader"]["kwargs"] == {"dataset": "nasa"}


def test_an_explicit_loader_wins_over_the_embedded_default(report, fleet, tmp_path):
    """Sealing the data AND naming another data path must stay possible."""
    sealed = seal_bundle(report, fleet, tmp_path, embed_data=True,
                         loader="my.pkg.data:load_cells", loader_kwargs={"org": 7})
    assert sealed["loader"]["module"] == "my.pkg.data"
    assert sealed["loader"]["kwargs"] == {"org": 7}
    assert (tmp_path / "cells").is_dir()
    assert f"cells/{CELLS_INDEX_NAME}" in sealed["files"]


# ---------------------------------------------------------------------------
# bundle_dir injection: only for loaders that ask for it
# ---------------------------------------------------------------------------

def test_bundle_dir_is_injected_only_into_a_loader_that_declares_it(fleet, tmp_path):
    write_bundle_cells(fleet, tmp_path / "cells")
    seen: dict = {}

    def declaring(cells_dir="cells", bundle_dir=None):
        seen["bundle_dir"] = bundle_dir
        return load_bundle_cells(cells_dir, bundle_dir=bundle_dir)

    def not_declaring(cells_dir="cells"):
        seen["plain"] = True
        return load_bundle_cells(cells_dir, bundle_dir=tmp_path)

    module = types.ModuleType("bundle_data_test_loader")
    module.declaring = declaring  # type: ignore[attr-defined]
    module.not_declaring = not_declaring  # type: ignore[attr-defined]
    sys.modules["bundle_data_test_loader"] = module
    try:
        data = _load_cell_data("bundle_data_test_loader:declaring", {"cells_dir": "cells"},
                               bundle_dir=tmp_path)
        assert set(data) == set(fleet)
        assert Path(seen["bundle_dir"]) == tmp_path.resolve()

        # A loader that does not declare it gets exactly what the bundle
        # recorded — nothing injected, nothing silently added.
        _load_cell_data("bundle_data_test_loader:not_declaring", {"cells_dir": "cells"},
                        bundle_dir=tmp_path)
        assert seen["plain"] is True
    finally:
        sys.modules.pop("bundle_data_test_loader", None)


def test_a_recorded_bundle_dir_is_not_overridden(report, fleet, tmp_path):
    """An explicit kwarg is the bundle's own statement; the verifier defers."""
    sealed = seal_bundle(report, fleet, tmp_path, embed_data=True,
                         loader_kwargs={"cells_dir": "cells", "bundle_dir": "recorded"})
    assert sealed["loader"]["kwargs"]["bundle_dir"] == "recorded"


def test_a_data_path_that_fails_to_load_is_a_failure_not_a_pass(report, fleet, tmp_path):
    """The non-embedded seal's weak spot, made loud.

    A bundle that points at a store the verifier cannot reach must not report
    PASS: the seal can still be checked, but the identity check did not happen,
    and "unchecked" is not "verified".
    """
    sealed = seal_bundle(report, fleet, tmp_path, dataset="synth-eol", embed_data=False,
                         loader="src.uploaded_store:load_uploaded_cell_data",
                         loader_kwargs={"upload_key": "upload-gone"})

    attempted, error = None, None
    try:
        attempted = _load_cell_data(
            "src.uploaded_store:load_uploaded_cell_data", sealed["loader"]["kwargs"]
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    assert attempted is None and error and "No persisted raw cycles" in error

    result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path, cell_data=None,
                           recompute=True, data_error=error)
    assert result["verdict"] == "fail"
    identity = next(c for c in result["checks"] if c["name"] == "data-identity")
    assert identity["status"] == "fail"
    assert "could not be loaded" in identity["detail"]
    assert next(c for c in result["checks"] if c["name"] == "seal")["status"] == "pass"

    # And the same bundle with NO data attempted is a warn, not a fail — the
    # distinction the check exists to draw.
    un_attempted = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path,
                                 cell_data=None, recompute=False)
    assert un_attempted["verdict"] == "pass"
    assert next(c for c in un_attempted["checks"]
                if c["name"] == "data-identity")["status"] == "warn"


# ---------------------------------------------------------------------------
# File names: readable when possible, never a path
# ---------------------------------------------------------------------------

def test_cell_ids_that_are_not_safe_file_names_get_a_safe_one(fleet, tmp_path):
    """Cell ids come from a user's CSV — the index carries the real id."""
    hostile = dict(fleet)
    hostile["../../etc/passwd"] = fleet["C0"]

    index = write_bundle_cells(hostile, tmp_path / "cells")
    names = {entry["cell_id"]: entry["file"] for entry in index["cells"]}
    assert names["C1"] == "C1.csv"
    assert Path(names["../../etc/passwd"]).name == names["../../etc/passwd"]
    assert ".." not in names["../../etc/passwd"]

    reloaded = load_bundle_cells("cells", bundle_dir=tmp_path)
    assert set(reloaded) == set(hostile)
    assert cell_digest(reloaded["../../etc/passwd"]) == cell_digest(fleet["C0"])


def test_two_ids_that_sanitize_onto_one_name_stay_apart(fleet, tmp_path):
    """A name collision must not silently drop a cell."""
    colliding = dict(fleet)
    colliding["bad/id"] = fleet["C0"]
    colliding["bad_id"] = fleet["C1"]

    index = write_bundle_cells(colliding, tmp_path / "cells")
    files = [entry["file"] for entry in index["cells"]]
    assert len(files) == len(set(files)) == len(colliding)
    assert set(load_bundle_cells("cells", bundle_dir=tmp_path)) == set(colliding)


def test_value_equality_holds_even_where_dtypes_are_re_inferred(fleet, tmp_path):
    """The dtype question, stated as a property rather than left implicit."""
    write_bundle_cells(fleet, tmp_path / "cells")
    reloaded = load_bundle_cells("cells", bundle_dir=tmp_path)

    for cell_id, df in fleet.items():
        pd.testing.assert_frame_equal(reloaded[cell_id], df, check_dtype=False)
    # ... and the digest does not care, because both serialize a whole-number
    # float and an int to the same text.
    assert cell_digest(reloaded["C0"]) == cell_digest(fleet["C0"])


def test_cell_file_name_only_keeps_plain_names_readable():
    assert cell_file_name("B0005") == "B0005.csv"
    assert cell_file_name("weird/../id") == "cell_0000.csv"
    assert cell_file_name("") == "cell_0000.csv"
    assert cell_file_name("Cell1.csv", 3) == "cell_0003.csv"
