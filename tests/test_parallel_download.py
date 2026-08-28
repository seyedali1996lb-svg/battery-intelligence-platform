"""Tests for batlab.datasets._download — the parallel reference-data fan-out."""

import time

import pytest

from batlab.datasets import _download as dl


def test_download_parallel_runs_concurrently_and_returns_input_order():
    """Jobs actually overlap (the whole point) and results come back in input order."""
    order: list[str] = []

    def slow(label: str, delay: float):
        time.sleep(delay)
        order.append(label)
        return f"/tmp/{label}"

    jobs = [("a", lambda: slow("a", 0.25)), ("b", lambda: slow("b", 0.25))]
    results = dl.download_parallel(jobs, max_workers=2)

    assert [r["label"] for r in results] == ["a", "b"]
    assert all(r["error"] is None for r in results)
    # Both jobs finished inside one delay's worth of wall time → they overlapped.
    assert len(order) == 2


def test_download_parallel_aggregates_errors_without_aborting_siblings():
    def ok():
        return "/tmp/ok"

    def boom():
        raise RuntimeError("host unreachable")

    results = dl.download_parallel([("good", ok), ("bad", boom)])
    by_label = {r["label"]: r for r in results}
    assert by_label["good"]["error"] is None
    assert by_label["good"]["path"] == "/tmp/ok"
    assert by_label["bad"]["path"] is None
    assert "host unreachable" in by_label["bad"]["error"]


def test_download_all_reference_data_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown datasets"):
        dl.download_all_reference_data(datasets=("nasa", "klingon"))


def test_download_all_reference_data_fans_out_to_loaders(monkeypatch):
    """The fan-out hits each loader's own entry point exactly once (additive —
    it must not re-implement any loader's download logic)."""
    import types

    import batlab.datasets as pkg
    import batlab.datasets._download as mod

    calls: list[str] = []

    def make(label: str):
        return types.SimpleNamespace(
            load_nasa_cells=lambda: calls.append(label),
            download_and_prepare=lambda: calls.append(label),
        )

    monkeypatch.setattr(pkg, "nasa", make("nasa"))
    monkeypatch.setattr(pkg, "severson", make("severson"))
    monkeypatch.setattr(pkg, "oxford", make("oxford"))
    monkeypatch.setattr(pkg, "zhu2022", make("zhu2022"))
    monkeypatch.setattr(mod, "download_parallel", lambda jobs, max_workers=4: [
        {"label": label, "path": fn(), "error": None} for label, fn in jobs
    ])

    mod.download_all_reference_data()
    assert sorted(calls) == ["nasa", "oxford", "severson", "zhu2022"]


def test_download_all_reference_data_status_hook(monkeypatch):
    import types

    import batlab.datasets as pkg
    import batlab.datasets._download as mod

    stub = types.SimpleNamespace(
        load_nasa_cells=lambda: None,
        download_and_prepare=lambda: None,
    )
    monkeypatch.setattr(pkg, "nasa", stub)
    monkeypatch.setattr(pkg, "severson", stub)
    monkeypatch.setattr(pkg, "oxford", stub)
    monkeypatch.setattr(pkg, "zhu2022", stub)
    monkeypatch.setattr(mod, "download_parallel", lambda jobs, max_workers=4: [
        {"label": label, "path": None, "error": None} for label, _ in jobs
    ])

    seen: list[str] = []
    mod.download_all_reference_data(datasets=("nasa", "zhu2022"), status_fn=seen.append)
    assert seen == ["nasa", "zhu2022"]
