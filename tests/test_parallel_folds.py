"""batlab._parallel — the fold-level thread pool behind run_lco(),
run_lco_quantiles() and train_models()'s four estimator fits.

The contract these tests pin: results come back in input order (every
mean, per_cell insertion order and bootstrap CI in the LCO evaluators
depends on it), a fold's exception reaches the caller, and the worker
count honours the BATLAB_FOLD_WORKERS override and never exceeds the
fold count. Numeric equivalence with the serial loop is covered by the
existing run_lco/calibration/gbrt tests, which run the same folds.
"""

import threading
import time

import pytest

from batlab._parallel import fold_workers, map_folds


def test_results_preserve_input_order_under_concurrency():
    # Later items finish first (shorter sleeps), so any implementation
    # that collected results by completion order would fail this.
    delays = [0.05, 0.04, 0.03, 0.02, 0.01]
    seen_threads: set[int] = set()

    def fold(i):
        seen_threads.add(threading.get_ident())
        time.sleep(delays[i])
        return f"fold-{i}"

    assert map_folds(fold, range(5)) == [f"fold-{i}" for i in range(5)]


def test_fold_exception_propagates():
    def fold(i):
        if i == 2:
            raise ValueError("fold 2 blew up")
        return i

    with pytest.raises(ValueError, match="fold 2 blew up"):
        map_folds(fold, range(4))


def test_single_fold_runs_serially_without_a_pool():
    calls: list[int] = []
    assert map_folds(lambda i: calls.append(threading.get_ident()) or i, [7]) == [7]
    assert calls == [threading.get_ident()]  # ran on the calling thread


def test_fold_workers_bounds(monkeypatch):
    monkeypatch.delenv("BATLAB_FOLD_WORKERS", raising=False)
    assert fold_workers(0) == 1
    assert fold_workers(1) == 1
    assert 1 <= fold_workers(4) <= 4
    assert fold_workers(100) <= 8

    monkeypatch.setenv("BATLAB_FOLD_WORKERS", "1")
    assert fold_workers(8) == 1
    monkeypatch.setenv("BATLAB_FOLD_WORKERS", "3")
    assert fold_workers(8) == 3
    assert fold_workers(2) == 2  # never more workers than folds
    monkeypatch.setenv("BATLAB_FOLD_WORKERS", "not-a-number")
    assert 1 <= fold_workers(8) <= 8  # malformed override falls back to auto


def test_serial_override_still_preserves_order(monkeypatch):
    monkeypatch.setenv("BATLAB_FOLD_WORKERS", "1")
    assert map_folds(lambda i: i * i, range(6)) == [0, 1, 4, 9, 16, 25]


def test_nested_map_folds_runs_serially_on_the_worker(monkeypatch):
    """A fold that itself calls map_folds() must not open a second pool:
    the inner call runs on the same thread that holds the outer permit,
    which is what makes the process-wide budget deadlock-free."""
    monkeypatch.delenv("BATLAB_FOLD_WORKERS", raising=False)
    inner_threads: dict[int, set[int]] = {}

    def inner(j):
        return threading.get_ident()

    def outer(i):
        inner_threads[i] = set(map_folds(inner, range(3)))
        return i

    assert map_folds(outer, range(4)) == [0, 1, 2, 3]
    for i, threads in inner_threads.items():
        assert len(threads) == 1, f"outer fold {i} fanned its inner folds out to {threads}"


def test_process_wide_budget_caps_concurrent_fits(monkeypatch):
    """Two independent map_folds() calls running at the same time (the
    app's per-dataset pipelines) never have more fits in flight than the
    budget, even though each pool alone could run that many."""
    monkeypatch.setenv("BATLAB_FOLD_WORKERS", "2")
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def fold(i):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return i

    pipelines = [threading.Thread(target=lambda: map_folds(fold, range(6))) for _ in range(3)]
    for t in pipelines:
        t.start()
    for t in pipelines:
        t.join()
    assert peak <= 2, f"budget of 2 was exceeded: peak {peak} concurrent fits"
    assert peak == 2, "expected the pools to actually use the budget"
