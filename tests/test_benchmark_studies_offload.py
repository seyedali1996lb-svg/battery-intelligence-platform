"""app/_data.py: the five benchmark studies run AFTER load_everything()
returns, not inside it.

Why this guard exists: on 2026-09-13 a genuinely cold boot took the
better part of an hour, because run_cross_chemistry_study /
run_pinn_benchmark_study / run_prospective_benchmark_study /
run_modeling_benchmark_study / run_robustness_study all ran inline in
load_everything() the moment the registry held its first real run. Every
other test hit a warm bundle cache (no log_run -> empty registry -> the
studies never ran), so only tests/test_cold_start_smoke.py ever paid for
them -- and it timed out. The studies are Benchmark-page content; nothing
else needs them to render.

These tests pin the offload contract with the study runner stubbed out
(the studies themselves are covered by test_experiment_registry.py and
test_tier4_modeling.py):
  - default mode runs on a daemon thread and returns immediately
  - eager mode runs inline (scripts that want a complete registry)
  - off mode skips and says so
  - the status snapshot the Benchmark page reads reflects all of that
"""

import sys as _sys
import os as _os
import threading
import time

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401

import pytest

import _data


@pytest.fixture(autouse=True)
def _reset_status(monkeypatch):
    monkeypatch.delenv("BATLAB_BOOT_STUDIES", raising=False)
    _data.BENCHMARK_STUDIES.update(
        state="idle", started_at=None, finished_at=None, completed=[], current=None,
    )
    yield
    _data.BENCHMARK_STUDIES.update(
        state="idle", started_at=None, finished_at=None, completed=[], current=None,
    )


def _stub_runner(record: dict, *, block: threading.Event | None = None):
    def run(study_datasets, bundles):
        record["thread"] = threading.current_thread().name
        record["daemon"] = threading.current_thread().daemon
        record["datasets"] = list(study_datasets)
        if block is not None:
            block.wait(timeout=5)
        _data.BENCHMARK_STUDIES["completed"].append("stub")
    return run


def test_default_mode_returns_immediately_and_runs_on_a_daemon_thread(monkeypatch):
    record: dict = {}
    gate = threading.Event()
    monkeypatch.setattr(_data, "_run_benchmark_studies", _stub_runner(record, block=gate))

    t0 = time.perf_counter()
    _data._launch_benchmark_studies({"nasa": {}}, {"nasa": None})
    assert time.perf_counter() - t0 < 1.0, "launch must not wait for the studies"

    status = _data.benchmark_studies_status()
    assert status["state"] == "running"
    assert status["started_at"] is not None and status["finished_at"] is None

    gate.set()
    deadline = time.time() + 5
    while _data.BENCHMARK_STUDIES["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)
    assert _data.BENCHMARK_STUDIES["state"] == "done"
    assert record["thread"] == "benchmark-studies"
    assert record["daemon"] is True, "a study nobody waits for must not block interpreter exit"
    assert record["datasets"] == ["nasa"]
    assert _data.benchmark_studies_status()["completed"] == ["stub"]


def test_eager_mode_runs_inline(monkeypatch):
    monkeypatch.setenv("BATLAB_BOOT_STUDIES", "eager")
    record: dict = {}
    monkeypatch.setattr(_data, "_run_benchmark_studies", _stub_runner(record))

    _data._launch_benchmark_studies({"synth": {}}, {})

    assert record["thread"] == threading.current_thread().name
    assert _data.benchmark_studies_status()["state"] == "done"
    assert _data.benchmark_studies_status()["completed"] == ["stub"]


def test_off_mode_skips_and_reports_off(monkeypatch):
    monkeypatch.setenv("BATLAB_BOOT_STUDIES", "off")
    called = []
    monkeypatch.setattr(_data, "_run_benchmark_studies", lambda *a, **k: called.append(1))

    _data._launch_benchmark_studies({"synth": {}}, {})

    assert not called
    assert _data.benchmark_studies_status()["state"] == "off"


def test_second_launch_while_running_is_a_no_op(monkeypatch):
    launches = []
    gate = threading.Event()

    def runner(study_datasets, bundles):
        launches.append(1)
        gate.wait(timeout=5)

    monkeypatch.setattr(_data, "_run_benchmark_studies", runner)
    _data._launch_benchmark_studies({}, {})
    _data._launch_benchmark_studies({}, {})
    gate.set()
    deadline = time.time() + 5
    while _data.BENCHMARK_STUDIES["state"] != "done" and time.time() < deadline:
        time.sleep(0.01)
    assert launches == [1]


def test_status_snapshot_is_a_copy():
    snap = _data.benchmark_studies_status()
    snap["completed"].append("tampered")
    snap["state"] = "tampered"
    assert _data.BENCHMARK_STUDIES["completed"] == []
    assert _data.BENCHMARK_STUDIES["state"] == "idle"


def test_load_everything_no_longer_calls_the_studies_inline():
    """Structural guard: the study entry points must not appear inside
    load_everything()'s own body any more -- only inside
    _run_benchmark_studies(), which _launch_benchmark_studies() schedules."""
    import ast
    import inspect

    src = inspect.getsource(_data)
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    body_calls = {
        node.func.attr
        for node in ast.walk(funcs["load_everything"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    studies = {
        "run_cross_chemistry_study", "run_pinn_benchmark_study",
        "run_prospective_benchmark_study", "run_modeling_benchmark_study",
        "run_robustness_study",
    }
    assert not (body_calls & studies), (
        f"load_everything() calls benchmark studies inline again: {body_calls & studies}"
    )
    runner_calls = {
        node.func.attr
        for node in ast.walk(funcs["_run_benchmark_studies"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert studies <= runner_calls, f"a study went missing from the runner: {studies - runner_calls}"
