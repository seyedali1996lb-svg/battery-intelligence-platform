"""
Parallel download support for the auto-downloaded reference datasets.

Each loader (nasa.py, severson.py, oxford.py, zhu2022.py) owns its own
download entry point — URL, destination naming, checksum pinning, and
skip-if-cached logic all live with the loader that verifies them, and
this module deliberately does NOT re-implement any of that. What it adds
is the one thing the loaders didn't share: a way to run several of those
entry points at once.

The four reference archives total ~1.5 GB (NASA ~200 MB, Severson ~100 MB,
Oxford ~750 MB across its per-group zips, Zhu 2022 ~356 MB). Downloaded
serially — which is how a fresh clone's first `load_*` call has always
behaved — that is several minutes of mostly idle network time; a
`ThreadPoolExecutor` over the independent downloads turns it into one
bounded burst. Threads (not processes) are the right tool: the work is
I/O-bound, and each loader is independent (separate dest dirs, separate
locks), so no shared state needs protecting.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

# Default worker count for download_parallel(). Kept deliberately small —
# the real bottleneck is the source hosts' own per-connection throughput,
# and 4 concurrent downloads is already a large speedup over serial while
# staying polite to the (academic, ungated) hosts these archives live on.
DEFAULT_MAX_WORKERS = 4


def download_parallel(
    jobs: "list[tuple[str, Callable[[], Any]]]",
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> "list[dict[str, Any]]":
    """Run several dataset-download callables concurrently.

    Parameters
    ----------
    jobs : list[(label, callable)]
        Each entry is ``(label, fn)`` where ``fn`` is one of the loaders'
        own download/prepare entry points (each already skips its own
        download when the archive is cached, verifies its own checksum,
        and raises on failure).
    max_workers : int
        Thread-pool size. Defaults to ``DEFAULT_MAX_WORKERS``.

    Returns
    -------
    list[dict]
        One result per job, in input order, each::

            {"label": str, "path": str | None, "error": str | None}

        A job that raised leaves ``error`` set and ``path`` None; other
        jobs continue regardless, so one flaky host can't abort the whole
        fan-out. Callers decide whether any error is fatal.
    """
    results: "list[dict[str, Any]]" = []

    def _run(label: str, fn: "Callable[[], Any]") -> dict[str, Any]:
        try:
            path = fn()
            return {"label": label, "path": path, "error": None}
        except Exception as exc:  # noqa: BLE001 — aggregated per-job, never fatal here
            return {"label": label, "path": None, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run, label, fn) for label, fn in jobs]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    # Deterministic order for the caller/tests: input order, not completion order.
    results.sort(key=lambda r: [label for label, _ in jobs].index(r["label"]))
    return results


def download_all_reference_data(
    datasets: "tuple[str, ...] | None" = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    status_fn: "Callable[[str], Any] | None" = None,
) -> "list[dict[str, Any]]":
    """Fan out the auto-downloadable reference datasets' downloads.

    Concurrent version of what a fresh clone does today when each dataset
    is first loaded serially. Each loader runs its own `download_and_prepare`
    entry point (which also caches derived summaries where applicable), so
    calling this repeatedly is cheap once everything is cached.

    Parameters
    ----------
    datasets : tuple[str, ...] | None
        Subset of ``{"nasa", "severson", "oxford", "zhu2022"}`` to fetch;
        defaults to all four.
    max_workers : int
        Passed through to :func:`download_parallel`.
    status_fn : callable | None
        Optional ``(label) -> None`` hook invoked when each job starts, for
        callers (e.g. the app) that want to render per-dataset progress.

    Returns
    -------
    list[dict]
        One ``{"label", "path", "error"}`` entry per dataset, in input order.
    """
    known = {"nasa", "severson", "oxford", "zhu2022"}
    chosen = known if datasets is None else set(datasets)
    unknown = chosen - known
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)} (known: {sorted(known)})")

    def _make(label: str) -> "Callable[[], Any]":
        if label == "nasa":
            from . import nasa
            return nasa.load_nasa_cells  # downloads + parses when not cached
        if label == "severson":
            from . import severson
            return severson.download_and_prepare
        if label == "oxford":
            from . import oxford
            return oxford.download_and_prepare
        from . import zhu2022  # label == "zhu2022"
        return zhu2022.download_and_prepare

    jobs: "list[tuple[str, Callable[[], Any]]]" = []
    for label in sorted(chosen):
        if status_fn is not None:
            status_fn(label)
        jobs.append((label, _make(label)))

    return download_parallel(jobs, max_workers=max_workers)
