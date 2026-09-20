"""app/_data.py: the validation / forecast / calibration layers are separable.

Why this guard exists: on 2026-09-19 a profile of the boot path
(app/_probe_timing.py) showed that a cold reference fleet's training is
80-95% leave-cell-out refitting — run_lco() plus the quantile calibration in
run_lco_quantiles() — and only 0.1-0.2 s hierarchical/survival/routing:

    fleet (cells/rows)      train_models   run_lco   quantile LCO   hier+surv+routing
    nasa    (4 /    580)          2.7 s      7.4 s         5.5 s             0.17 s
    zhu2022 (9 /  8 726)          6.3 s     23.6 s        30.2 s             0.17 s
    severson(46 / 36 971)        43.5 s    > 400 s     (> 400 s)              0.1 s

Nothing renders until that finished, so every AppTest that missed the bundle
cache retrained every fleet for minutes. load_everything() now serves the core
bundle (features + GBRT + predictions) and finishes the three layers on a
background thread that re-serves the frames, logs the run and rewrites the
bundle cache (app/_data.py's "Boot layers" section).

These tests pin the contract with the layer bodies stubbed out (the layers
themselves are covered by test_lco_eval.py / test_calibration.py /
test_tier4_modeling.py):
  - default mode returns immediately and runs on a daemon thread
  - eager mode runs inline (scripts that need a complete bundle + registry row)
  - off mode computes nothing and says so
  - a deferred bundle carries EXPLICIT None placeholders for every layer
    metric (a missing key would read as "not evaluated", or raise)
  - a cached bundle whose layers never landed is retried on the next boot
  - the expensive calls no longer live in train_and_predict()'s body
"""

import ast
import inspect
import os as _os
import sys as _sys
import threading
import time

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401

import pytest

import _data


_LAYER_FUNCS = ("_layer_validation", "_layer_forecast", "_layer_calibration")


@pytest.fixture(autouse=True)
def _reset_status(monkeypatch):
    monkeypatch.delenv("BATLAB_BOOT_LAYERS", raising=False)
    _data.BOOT_LAYERS.update(
        state="idle", started_at=None, finished_at=None,
        pending=[], completed=[], failed=[], skipped={}, seconds={},
    )
    yield
    _data.BOOT_LAYERS.update(
        state="idle", started_at=None, finished_at=None,
        pending=[], completed=[], failed=[], skipped={}, seconds={},
    )


def _stub_layers(monkeypatch, record: dict, *, block: threading.Event | None = None, state: str = "done"):
    """Replace the layer body / re-serve / persist / log / cache writes."""

    def run_layers(bndl, cell_cycles, raw_fdfs, key=None, guarded=False):
        record["thread"] = threading.current_thread().name
        record["daemon"] = threading.current_thread().daemon
        record["key"] = key
        record["guarded"] = guarded
        bndl.setdefault("metrics", {})["boot_layers"] = state
        if block is not None:
            block.wait(timeout=5)
        return {"lco": True}

    monkeypatch.setattr(_data, "run_boot_layers", run_layers)
    monkeypatch.setattr(_data, "_serve_frames", lambda b, r, m: ({"cell": "served"}, {"cell": 1}))
    monkeypatch.setattr(_data, "_persist_cell_data", lambda fdfs: record.setdefault("persisted", 0) or record.__setitem__("persisted", record.get("persisted", 0) + 1))
    monkeypatch.setattr(_data, "_log_bundle_run", lambda b, lco, ds, org: record.__setitem__("logged", ds))
    monkeypatch.setattr(_data, "save_cached", lambda key, cell_dict, data: record.__setitem__("cached", key))


def _launch(key="nasa"):
    return _data._launch_boot_layers(
        key, {"c1": {"cycles": None}}, {"metrics": {"boot_layers": "pending"}},
        {"c1": None}, {"c1": None}, {"c1": None},
    )


# ── Modes ──────────────────────────────────────────────────────────────────


def test_default_mode_is_background():
    assert _data._boot_layers_mode() == "background"


def test_unknown_mode_falls_back_to_background(monkeypatch):
    monkeypatch.setenv("BATLAB_BOOT_LAYERS", "turbo")
    assert _data._boot_layers_mode() == "background"


def test_launch_returns_immediately_and_runs_on_a_daemon_thread(monkeypatch):
    record: dict = {}
    gate = threading.Event()
    _stub_layers(monkeypatch, record, block=gate)

    t0 = time.perf_counter()
    _launch()
    assert time.perf_counter() - t0 < 1.0, "launch must not wait for the layers"

    status = _data.boot_layers_status()
    assert status["state"] == "running"
    assert status["pending"] == ["nasa"]

    gate.set()
    deadline = time.time() + 5
    while _data.BOOT_LAYERS["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)

    assert _data.BOOT_LAYERS["state"] == "done"
    assert _data.boot_layers_status()["completed"] == ["nasa"]
    assert record["thread"] == "boot-layers-nasa"
    assert record["daemon"] is True, "layers nobody waits for must not block interpreter exit"
    assert record["guarded"] is True, "the runner contains a layer failure to that layer"
    assert record["persisted"] == 1 and record["cached"] == "nasa"
    assert record["logged"] == "nasa"


def test_off_mode_launches_nothing(monkeypatch):
    monkeypatch.setenv("BATLAB_BOOT_LAYERS", "off")
    called = []
    monkeypatch.setattr(_data, "run_boot_layers", lambda *a, **k: called.append(1))

    _launch()

    assert not called
    assert _data.BOOT_LAYERS["state"] == "off"
    assert _data.BOOT_LAYERS["pending"] == []


def test_second_launch_while_pending_is_a_no_op(monkeypatch):
    record: dict = {}
    gate = threading.Event()
    _stub_layers(monkeypatch, record, block=gate)

    _launch()
    _launch()
    gate.set()
    deadline = time.time() + 5
    while _data.BOOT_LAYERS["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)
    assert _data.boot_layers_status()["completed"] == ["nasa"]


def test_a_failed_layer_run_leaves_no_run_row_and_no_completed_cache(monkeypatch):
    record: dict = {}
    _stub_layers(monkeypatch, record, state="failed")

    _launch()
    deadline = time.time() + 5
    while _data.BOOT_LAYERS["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)

    assert "logged" not in record, "a half-validated fleet must not log a registry row"
    assert "cached" not in record, "the core-only cache entry stays so the next boot retries"
    assert _data.boot_layers_status()["completed"] == []
    assert _data.BOOT_LAYERS["pending"] == []


def test_a_raising_runner_is_recorded_and_marks_the_bundle_failed(monkeypatch):
    record: dict = {}

    def boom(bndl, cell_cycles, raw_fdfs, key=None, guarded=False):
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(_data, "run_boot_layers", boom)
    bndl = {"metrics": {"boot_layers": "pending"}}

    _data._launch_boot_layers("nasa", {}, bndl, {}, {}, {})
    deadline = time.time() + 5
    while _data.BOOT_LAYERS["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)

    assert _data.boot_layers_status()["failed"] == ["nasa:runner"]
    assert bndl["metrics"]["boot_layers"] == "failed"
    assert record == {}


def test_status_snapshot_is_a_copy():
    snap = _data.boot_layers_status()
    snap["pending"].append("tampered")
    snap["skipped"]["x"] = "tampered"
    snap["state"] = "tampered"
    assert _data.BOOT_LAYERS["pending"] == []
    assert _data.BOOT_LAYERS["skipped"] == {}
    assert _data.BOOT_LAYERS["state"] == "idle"


# ── Placeholders + layer wiring ────────────────────────────────────────────


def test_baseline_absence_is_explained_not_left_blank():
    """A trivial baseline can come back without a number WITHOUT raising (every
    fold unscorable returns NaN), and NaN -> None at the metrics seam is
    indistinguishable from "not computed yet". The reason must travel with the
    absence: the fold notes are lifted into one string."""
    assert _data._baseline_absence_reason({"baseline_soh_r2": 0.603, "per_cell": {}}) is None

    reason = _data._baseline_absence_reason({
        "baseline_soh_r2": float("nan"),
        "n_cells": 2,
        "per_cell": {
            "A": {"baseline_soh_r2": None, "note": "not scored: no soh_pct column"},
            "B": {"baseline_soh_r2": None, "note": "not scored: one row only"},
        },
    })
    assert reason and "no soh_pct column" in reason and "one row only" in reason

    single = _data._baseline_absence_reason({"baseline_soh_r2": float("nan"), "n_cells": 1})
    assert single and "fewer than two cells" in single


def test_mark_layers_pending_sets_every_key_explicitly():
    bndl = {"metrics": {}}
    _data._mark_layers_pending(bndl)

    for key, placeholder in _data._LAYER_METRIC_PLACEHOLDERS.items():
        assert key in bndl["metrics"], f"{key} missing -- a missing key is not 'pending'"
        if isinstance(placeholder, dict):
            assert bndl["metrics"][key] == {}
        elif isinstance(placeholder, list):
            assert bndl["metrics"][key] == []
        else:
            assert bndl["metrics"][key] is None
    for key in _data._LAYER_BUNDLE_KEYS:
        assert bndl[key] is None
    assert bndl["metrics"]["boot_layers"] == "pending"


def test_pending_placeholders_are_fresh_and_container_safe():
    """A consumer that indexes a container metric (knowledge_graph's
    per_cell_ok.get(cell_id, ...), src/api.py, battery_copilot) must not be
    handed a None by a deferred bundle — and two bundles must not share one
    mutable placeholder."""
    first, second = {"metrics": {}}, {"metrics": {}}
    _data._mark_layers_pending(first)
    _data._mark_layers_pending(second)

    for key in ("per_cell_rul_reliable", "lco_per_cell", "censored_rul"):
        assert isinstance(first["metrics"][key], dict)
        assert first["metrics"][key].get("c1") is None  # .get() must not raise
        assert first["metrics"][key] is not second["metrics"][key]
    assert first["metrics"]["forecast_routing"] == []
    assert not first["metrics"]["rul_reliable"]  # falsy: RUL stays withheld


def test_every_metric_a_layer_writes_has_a_placeholder():
    """A layer that writes a metric _mark_layers_pending() doesn't know about
    would leave that key MISSING on a deferred bundle."""
    tree = ast.parse(inspect.getsource(_data))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    written = set()
    for name in _LAYER_FUNCS:
        for node in ast.walk(funcs[name]):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Subscript)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "bndl"
                    and getattr(target.value.slice, "value", None) == "metrics"
                    and isinstance(target.slice, ast.Constant)
                ):
                    written.add(str(target.slice.value))
    missing = written - set(_data._LAYER_METRIC_KEYS)
    assert not missing, f"layer metrics with no placeholder: {sorted(missing)}"


def test_run_boot_layers_runs_the_three_layers_in_order(monkeypatch):
    order = []
    for short, attr in zip(_data.BOOT_LAYER_NAMES, _LAYER_FUNCS):
        monkeypatch.setattr(
            _data, attr,
            (lambda n: lambda bndl, cc, rf, lco: (order.append(n), lco or {})[1])(short),
        )

    bndl = {"metrics": {}}
    _data.run_boot_layers(bndl, {}, {}, key="nasa")

    assert order == list(_data.BOOT_LAYER_NAMES)
    assert bndl["metrics"]["boot_layers"] == "done"
    assert set(_data.boot_layers_status()["seconds"]["nasa"]) == set(_data.BOOT_LAYER_NAMES)


def test_run_boot_layers_contains_failures_only_when_guarded(monkeypatch):
    def boom(bndl, cc, rf, lco):
        raise RuntimeError("layer failed")

    monkeypatch.setattr(_data, "_layer_validation", boom)
    monkeypatch.setattr(_data, "_layer_forecast", lambda b, cc, rf, lco: lco or {})
    monkeypatch.setattr(_data, "_layer_calibration", lambda b, cc, rf, lco: lco or {})

    bndl = {"metrics": {}}
    with pytest.raises(RuntimeError):
        _data.run_boot_layers(bndl, {}, {}, key="nasa")  # an eager boot fails as it always has

    bndl = {"metrics": {}}
    assert _data.run_boot_layers(bndl, {}, {}, key="nasa", guarded=True) == {}
    assert bndl["metrics"]["boot_layers"] == "failed"
    assert _data.boot_layers_status()["failed"] == ["nasa:validation"]


def test_train_and_predict_defers_only_when_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(_data, "train_models", lambda *a, **k: {"metrics": {}})
    monkeypatch.setattr(
        _data, "run_boot_layers",
        lambda bndl, cc, rf, key=None, guarded=False: calls.append(key) or {"lco": 1},
    )
    monkeypatch.setattr(
        _data, "_serve_frames",
        lambda bndl, rf, mi: ({"c1": "df"}, {"c1": 1}),
    )
    monkeypatch.setattr(_data, "_log_bundle_run", lambda *a, **k: None)

    import pandas as pd

    X = pd.DataFrame({"cycle_number": [1, 2]})
    y = pd.Series([1.0, 2.0])
    inputs = {"c1": (X, y, y)}
    cells = {"c1": {"cycles": pd.DataFrame({"cycle_number": [1, 2]})}}

    bndl, served, splits = _data.train_and_predict(
        cells, {"c1": X}, inputs, dataset="nasa", org_id=1, defer_layers=False,
    )
    assert calls == [None] and "boot_layers" not in bndl["metrics"]

    calls.clear()
    bndl, served, splits = _data.train_and_predict(
        cells, {"c1": X}, inputs, dataset="nasa", org_id=1, defer_layers=True,
    )
    assert calls == [] and bndl["metrics"]["boot_layers"] == "pending"


# ── Resume on a warm cache ─────────────────────────────────────────────────


def test_resume_relaunches_a_pending_cached_bundle(monkeypatch):
    launched = []
    monkeypatch.setattr(
        _data, "_launch_boot_layers",
        lambda *a, **k: launched.append(a[0]),
    )
    monkeypatch.setattr(_data, "load_features_cached", lambda key, cells: ({"c1": "df"}, {"c1": "X"}))

    _data._resume_deferred_layers(
        "nasa", ({"metrics": {"boot_layers": "pending"}}, {}), {"c1": {"cycles": None}},
    )

    assert launched == ["nasa"]


def test_resume_leaves_complete_and_legacy_bundles_alone(monkeypatch):
    launched = []
    monkeypatch.setattr(_data, "_launch_boot_layers", lambda *a, **k: launched.append(a[0]))
    monkeypatch.setattr(_data, "load_features_cached", lambda key, cells: ({"c1": "df"}, {"c1": "X"}))

    # layers already done
    _data._resume_deferred_layers("nasa", ({"metrics": {"boot_layers": "done"}}, {}), {"c1": {"cycles": None}})
    # a bundle trained before the split carries no boot_layers key at all
    _data._resume_deferred_layers("zhu2022", ({"metrics": {"lco_soh_r2": 0.9}}, {}), {"c1": {"cycles": None}})

    assert launched == []


def test_resume_records_why_it_could_not_run(monkeypatch):
    monkeypatch.setattr(_data, "load_features_cached", lambda key, cells: None)
    monkeypatch.setattr(_data, "_launch_boot_layers", lambda *a, **k: pytest.fail("must not launch"))

    _data._resume_deferred_layers(
        "severson", ({"metrics": {"boot_layers": "failed"}}, {}), {"c1": {"cycles": None}},
    )

    assert "severson" in _data.boot_layers_status()["skipped"]


def test_resume_is_inert_in_eager_mode(monkeypatch):
    monkeypatch.setenv("BATLAB_BOOT_LAYERS", "eager")
    monkeypatch.setattr(_data, "_launch_boot_layers", lambda *a, **k: pytest.fail("must not launch"))

    _data._resume_deferred_layers(
        "nasa", ({"metrics": {"boot_layers": "pending"}}, {}), {"c1": {"cycles": None}},
    )


# ── Structural guard ───────────────────────────────────────────────────────


def test_the_expensive_layers_no_longer_run_inside_train_and_predict():
    """Structural guard: the LCO harness and the quantile calibration must not
    appear in train_and_predict()'s own body — they belong to the layers, which
    the caller decides how to schedule."""
    tree = ast.parse(inspect.getsource(_data))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    body_calls = {
        node.func.id
        for node in ast.walk(funcs["train_and_predict"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    expensive = {"run_lco", "run_lco_quantiles"}
    assert not (body_calls & expensive), (
        f"train_and_predict() calls {body_calls & expensive} inline again"
    )
    for name in _LAYER_FUNCS:
        layer_calls = {
            node.func.id
            for node in ast.walk(funcs[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if name == "_layer_validation":
            assert "run_lco" in layer_calls
        if name == "_layer_calibration":
            assert "run_lco_quantiles" in layer_calls


def test_validation_layer_does_not_rebuild_features_it_already_has():
    """run_lco() rebuilds every cell's features — including the PyBaMM-backed
    physics calibration — unless it is handed the frames the caller already
    built. The boot path holds them (raw_fdfs), so it must pass them."""
    tree = ast.parse(inspect.getsource(_data))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    calls = [
        node for node in ast.walk(funcs["_layer_validation"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "run_lco"
    ]
    assert len(calls) == 1
    assert {kw.arg for kw in calls[0].keywords} >= {"featured"}
