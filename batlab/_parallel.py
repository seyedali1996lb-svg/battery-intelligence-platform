"""
Thread-level parallelism for independent model fits (LCO folds, the four
estimators inside train_models).

Every LCO fold is independent: it fits its own GradientBoostingRegressor(s)
on the other cells and scores the held-out one. scikit-learn's tree builder
releases the GIL inside the Cython fit loop, so running folds on a thread
pool gives a real wall-clock speedup (measured ~2.8x for 4 folds on a
12-core machine) with no extra process memory -- which matters on a
1 GB Streamlit Community Cloud instance where a process pool would not fit.

Numbers are unchanged by construction: each fold is seeded independently
(`random_state=seed` inside the fold), the fold function is pure, and
results are re-assembled in the caller's original cell order, so every
mean, per-cell entry, and bootstrap CI sees exactly the sequence it saw
when the loop was serial.

Process-wide fit budget
-----------------------
The app trains several reference fleets concurrently (app/_data.py's
load_everything() runs one pipeline per dataset on its own thread), and
each pipeline calls run_lco / run_lco_quantiles / train_models. Without a
shared limit that nests into 4 x 8 fold threads plus the estimator fits --
dozens of runnable GBRT threads contending for the GIL between their
nogil sections, which is *slower* than running serially (a 300 s cold
boot became 17 minutes). So every fit submitted through map_folds()
acquires a process-wide semaphore sized to the CPU budget: nesting can
never put more fits in flight than the machine can run.

A map_folds() call made from inside another map_folds() worker runs its
items serially on that worker (no second pool, no second permit), which
makes the semaphore deadlock-free by construction.

Workers are daemon threads, deliberately: concurrent.futures' pool threads
are joined at interpreter exit, which made a test process (and would make
a Streamlit shutdown) wait for every in-flight benchmark fit to finish
before it could exit. A fit that nobody is waiting for should die with
the process.

Set BATLAB_FOLD_WORKERS=1 to force the serial path (useful when
bisecting a fold failure, or on a memory-starved host).
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Callable, Iterable, Sequence, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")

_MAX_DEFAULT_WORKERS = 8


def cpu_budget() -> int:
    """Total fits this process runs concurrently, across every nested pool.

    Bounded by the CPU count, a fixed cap (beyond ~8 threads the
    GIL-holding parts of a fit dominate), and the BATLAB_FOLD_WORKERS
    override. Never less than 1.
    """
    override = os.environ.get("BATLAB_FOLD_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, _MAX_DEFAULT_WORKERS))


def fold_workers(n_folds: int) -> int:
    """How many folds one evaluation may run concurrently: the CPU budget,
    never more than the fold count, never less than 1."""
    if n_folds <= 1:
        return 1
    return max(1, min(n_folds, cpu_budget()))


# One permit per concurrently-running fit, process-wide. Sized lazily so an
# environment override set before the first fit is honoured, and re-sized
# only if that budget changes (tests toggle BATLAB_FOLD_WORKERS).
_budget_lock = threading.Lock()
_budget_size: int | None = None
_budget_sem: threading.BoundedSemaphore | None = None

# Set on a thread while it is executing a fold for map_folds(), so a nested
# map_folds() call from inside a fold runs serially instead of stacking a
# second pool (and a second permit) on top of the one it already holds.
_state = threading.local()


def _semaphore() -> threading.BoundedSemaphore:
    global _budget_size, _budget_sem
    size = cpu_budget()
    with _budget_lock:
        if _budget_sem is None or _budget_size != size:
            _budget_size = size
            _budget_sem = threading.BoundedSemaphore(size)
        return _budget_sem


def map_folds(fn: Callable[[_T], _R], items: Sequence[_T] | Iterable[_T]) -> list[_R]:
    """Apply `fn` to every fold key in `items`, returning results in order.

    Serial when only one worker is available or when called from inside a
    running fold (keeps tracebacks simple, prevents nested pools);
    otherwise a thread pool whose fits share the process-wide budget.
    Exceptions from any fold propagate to the caller exactly as they would
    from a plain loop.
    """
    items = list(items)
    workers = fold_workers(len(items))
    if workers <= 1 or getattr(_state, "in_fold", False):
        return [fn(item) for item in items]

    sem = _semaphore()
    results: list = [None] * len(items)
    errors: list[BaseException] = []
    pending: "queue.SimpleQueue[int | None]" = queue.SimpleQueue()
    for idx in range(len(items)):
        pending.put(idx)
    for _ in range(workers):
        pending.put(None)  # one stop token per worker

    def worker() -> None:
        while True:
            idx = pending.get()
            if idx is None:
                return
            if errors:
                continue  # a fold already failed: drain, don't start more work
            with sem:
                _state.in_fold = True
                try:
                    results[idx] = fn(items[idx])
                except BaseException as exc:  # re-raised on the caller's thread
                    errors.append(exc)
                finally:
                    _state.in_fold = False

    threads = [threading.Thread(target=worker, daemon=True, name="batlab-fold") for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results
