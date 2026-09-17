"""
Tests for the model sandbox (batlab/harness/sandbox.py + sandbox_worker.py).

The feature: the app's "Bring your own model" uploader used to import an
uploaded module in the Streamlit process, disclosed on the page in as many
words. It now runs in a child process under an audit-hook policy and resource
caps. These tests defend, in order of how much damage their absence would do:

1. **The number is unchanged.** A sandbox that quietly graded a different model
   would be a worse instrument than no sandbox. Predictions are compared
   bit-for-bit against an in-process fit, and a whole harness run is compared
   against the same run in-process.
2. **The model is really elsewhere.** Escapes are attempted for real — network,
   subprocess, ctypes, writes outside the scratch directory — and each has to
   be refused with a message naming what was attempted.
3. **The caps bite.** A model that hangs, or allocates past the memory cap, is
   stopped, and the failure says which cap.
4. **Nothing is left running.** The child (and anything under it — on Windows a
   venv's python.exe launches the real interpreter as a CHILD, which the first
   version of this module got wrong) is gone after close().
5. **The plumbing is honest.** A point-only model stays point-only through the
   proxy, a interval-capable one keeps its interval, the report's model identity
   names what the child holds rather than the proxy, and what the model printed
   is captured instead of corrupting the protocol.

The policy's own limits are documented in the worker's docstring rather than
asserted here: an audit hook is a Python-level control, not an OS sandbox, and
no test can turn that into a stronger claim.
"""

import json
import pathlib
import sys
import time

import pytest
from conftest import make_cycles_df

from batlab.harness import (
    CallableForecaster,
    SklearnForecaster,
    forecaster_identity,
    has_predict_interval,
    sandbox_forecaster,
    validate_forecaster,
)
from batlab.harness.sandbox import (
    SandboxError,
    SandboxLimits,
    SandboxedModelFactory,
    describe_enforcement,
)

# ---------------------------------------------------------------------------
# Model sources used across the tests
# ---------------------------------------------------------------------------

_RIDGE = """\
from sklearn.linear_model import Ridge
from batlab.harness import SklearnForecaster


def make_model():
    return SklearnForecaster(Ridge(alpha=1.0), scale=True)
"""

# A model whose predictions reveal WHICH data it was fitted on, so per-fold
# isolation can be observed rather than assumed.
_MEAN_MODEL = """\
from batlab.harness import CallableForecaster


def _fit(X, y):
    return float(y.mean())


def _predict(state, X):
    return [state] * len(X)


def make_model():
    print("make_model called")
    return CallableForecaster(_fit, _predict, label="training-mean")
"""

_INTERVAL_MODEL = """\
from batlab.harness import CallableForecaster


def _fit(X, y):
    return float(y.mean())


def _predict(state, X):
    return [state] * len(X)


def _interval(state, X):
    return ([state - 1.0] * len(X), [state + 1.0] * len(X))


def make_model():
    return CallableForecaster(_fit, _predict, interval_fn=_interval, label="mean-with-interval")
"""

_READER = """\
from sklearn.linear_model import Ridge
from batlab.harness import SklearnForecaster
import numpy as np


def make_model():
    # Reading outside the scratch directory is ALLOWED on purpose: sklearn and
    # numpy read their own data and libraries, and reading is not how a graded
    # model damages anything. A sandbox that broke this would break every model.
    _ = np.load.__module__
    with open(__file__, "r", encoding="utf-8") as handle:
        _ = handle.read()
    return SklearnForecaster(Ridge(alpha=1.0), scale=True)
"""


def _fleet(n_cells: int = 3, n_cycles: int = 140) -> dict:
    return {
        f"C{i}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=0.005 * (1.0 + 0.08 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


def _refusal(source: str, **kwargs) -> SandboxError:
    """Load `source` in a sandbox and return the refusal it must produce."""
    factory = SandboxedModelFactory(source, limits=kwargs.pop("limits", None), **kwargs)
    try:
        factory.start()
    except SandboxError as exc:
        return exc
    finally:
        factory.close()
    pytest.fail("the sandbox accepted a module it was supposed to refuse")


# ---------------------------------------------------------------------------
# 1. The number is unchanged
# ---------------------------------------------------------------------------

def test_a_sandboxed_fit_is_bit_identical_to_an_in_process_fit():
    """Raw float64 over the wire, so the graded number cannot drift."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X["f0"] * 1.5 - X["f2"] * 0.25)

    with sandbox_forecaster(_RIDGE) as factory:
        sandboxed = factory()
        sandboxed.fit(X, y)
        got = sandboxed.predict(X)

    from sklearn.linear_model import Ridge

    reference = SklearnForecaster(Ridge(alpha=1.0), scale=True).fit(X, y).predict(X)
    assert np.array_equal(got, reference)


def test_a_whole_harness_run_matches_the_in_process_run():
    """The end-to-end claim, not just one predict call.

    Same fleet, same folds, same seed; the only difference is that one model was
    fitted in another process. The LCO metrics have to agree exactly.
    """
    cells = _fleet()

    # The same configuration the module declares, byte for byte (scale=True).
    in_process = validate_forecaster(
        cells, model=SklearnForecaster(_ridge(), scale=True), splits=("lco",),
        intervals=False, dataset="sandbox-parity",
    )
    with sandbox_forecaster(_RIDGE) as factory:
        sandboxed = validate_forecaster(
            cells, model=factory, splits=("lco",), intervals=False,
            dataset="sandbox-parity",
        )

    assert sandboxed["lco"]["soh_r2"] == in_process["lco"]["soh_r2"]
    assert sandboxed["lco"]["rul_r2"] == in_process["lco"]["rul_r2"]
    assert sandboxed["lco"]["per_cell"] == in_process["lco"]["per_cell"]
    assert sandboxed["verdict"]["summary"] == in_process["verdict"]["summary"]


def _ridge():
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def test_two_live_handles_never_share_a_fit():
    """The bug this test was written for: one child, one fitted model, and
    handle B's fit silently answering handle A's predict."""
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    with sandbox_forecaster(_MEAN_MODEL) as factory:
        handles = [factory() for _ in range(12)]
        for index, model in enumerate(handles):
            model.fit(X, pd.Series([float(index)] * 3))
        # Every handle still answers with ITS OWN fit, in any order.
        assert list(np.asarray(handles[0].predict(X))) == [0.0, 0.0, 0.0]
        assert list(np.asarray(handles[-1].predict(X))) == [11.0, 11.0, 11.0]
        assert list(np.asarray(handles[5].predict(X))) == [5.0, 5.0, 5.0]
        assert handles[0].handle != handles[-1].handle


def test_every_fit_starts_from_a_fresh_model():
    """One child, many folds — and the folds still cannot see each other.

    The module returns a model that predicts the mean of whatever it was fitted
    on, so a handle that inherited the previous fold's fit would answer with the
    previous fold's number instead of its own.
    """
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    with sandbox_forecaster(_MEAN_MODEL) as factory:
        first, second = factory(), factory()
        first.fit(X, pd.Series([0.0, 0.0, 0.0]))
        second.fit(X, pd.Series([10.0, 10.0, 10.0]))
        assert list(np.asarray(first.predict(X))) == [0.0, 0.0, 0.0]
        assert list(np.asarray(second.predict(X))) == [10.0, 10.0, 10.0]
        # ... and refitting a handle is a fresh model too, not a warm start.
        first.fit(X, pd.Series([3.0, 3.0, 3.0]))
        assert list(np.asarray(first.predict(X))) == [3.0, 3.0, 3.0]


# ---------------------------------------------------------------------------
# 2. The model is really elsewhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,expected", [
    ("import socket\n\ndef make_model():\n    return None\n", "socket"),
    ("import urllib.request\n\ndef make_model():\n    return None\n", "urllib"),
    ("import subprocess\n\ndef make_model():\n    return None\n", "subprocess"),
    ("import ctypes\n\ndef make_model():\n    return None\n", "ctypes"),
    ("import multiprocessing\n\ndef make_model():\n    return None\n", "multiprocessing"),
    ("import asyncio\n\ndef make_model():\n    return None\n", "asyncio"),
])
def test_reaching_outside_the_process_is_refused_by_name(source, expected):
    exc = _refusal(source)
    assert exc.kind == "policy"
    assert expected in str(exc)
    assert "reaches outside this process" in str(exc)


def test_a_connection_attempt_is_refused_even_where_the_import_is_cached():
    """The purge, stated as behaviour.

    The platform's own preload pulls socket/ssl/subprocess/ctypes in as
    transitive dependencies, and an already-imported module is not re-loaded —
    so the audited import event never fires for it. Dropping those entries
    before the policy goes in is what makes a model's own `import socket` the
    refusal it should be (and the audited `socket.connect` catches the case
    where something reached the module by another route).
    """
    exc = _refusal("import socket\n\ndef make_model():\n    socket.socket()\n    return None\n")
    assert exc.kind == "policy"
    assert "socket" in str(exc)

    connect = _refusal(
        "def make_model():\n"
        "    import socket as _s\n"
        "    _s.create_connection(('example.com', 80), timeout=2)\n"
        "    return None\n"
    )
    assert connect.kind == "policy"
    assert "socket" in str(connect)


def test_writing_outside_the_scratch_directory_is_refused(tmp_path):
    target = tmp_path / "escaped.txt"
    exc = _refusal(
        "def make_model():\n"
        f"    open({str(target)!r}, 'w').write('escaped')\n"
        "    return None\n"
    )
    assert exc.kind == "policy"
    assert "writes are confined" in str(exc)
    assert not target.exists(), "the sandbox let a model write outside its scratch directory"


def test_writing_INSIDE_the_scratch_directory_is_allowed():
    """The refusal is about where, not about writing at all.

    A model that caches a fitted artefact next to its own module is doing
    something ordinary; the policy's job is to keep it inside its own
    directory, not to forbid the operation.
    """
    module = sandbox_forecaster(
        "import os\n"
        "\n"
        "def make_model():\n"
        "    with open(os.path.join(os.path.dirname(__file__), 'note.txt'), 'w') as fh:\n"
        "        fh.write('mine')\n"
        "    from sklearn.linear_model import Ridge\n"
        "    from batlab.harness import SklearnForecaster\n"
        "    return SklearnForecaster(Ridge(alpha=1.0), scale=True)\n"
    )
    try:
        module.start()
        assert pathlib.Path(module.module_path).parent.joinpath("note.txt").exists()
    finally:
        module.close()


def test_reading_outside_the_scratch_directory_is_allowed():
    """sklearn and numpy read their own libraries and data. A sandbox that broke
    that would refuse every model the platform was built to grade."""
    with sandbox_forecaster(_READER) as factory:
        assert factory.probe_class == "SklearnForecaster"


def test_system_calls_are_refused():
    for source, expected in (
        ("import os\n\ndef make_model():\n    os.system('echo hi')\n    return None\n", "os.system"),
        (
            "import subprocess\n\ndef make_model():\n"
            "    subprocess.Popen(['echo', 'hi'])\n    return None\n",
            "subprocess",
        ),
    ):
        exc = _refusal(source)
        assert exc.kind == "policy"
        assert expected in str(exc)


# ---------------------------------------------------------------------------
# 3. The caps bite
# ---------------------------------------------------------------------------

def test_a_hanging_model_is_killed_by_the_wall_clock():
    limits = SandboxLimits(wall_seconds=4.0, total_seconds=30.0)
    started = time.monotonic()
    exc = _refusal("import time\n\ndef make_model():\n    time.sleep(600)\n", limits=limits)
    elapsed = time.monotonic() - started
    assert exc.kind == "timeout"
    assert "wall-clock" in str(exc) or "did not answer" in str(exc)
    assert elapsed < 30, f"the wall-clock cap did not stop the model (took {elapsed:.1f}s)"


def test_a_memory_hungry_model_is_killed_by_the_watchdog():
    """Virtual size, not just RSS: an untouched np.zeros is virtual memory only,
    and the first version of this watchdog measured nothing but RSS."""
    limits = SandboxLimits(wall_seconds=30.0, total_seconds=60.0, memory_mb=700)
    exc = _refusal(
        "import numpy as np\n\ndef make_model():\n"
        "    hog = [np.zeros(40_000_000) for _ in range(20)]\n"
        "    return None\n",
        limits=limits,
    )
    assert exc.kind == "memory", exc
    assert "700 MiB" in str(exc)


def test_a_cpu_hungry_model_is_killed_by_the_cpu_cap():
    limits = SandboxLimits(wall_seconds=60.0, total_seconds=120.0, cpu_seconds=1)
    exc = _refusal(
        "def make_model():\n    total = 0\n    while True:\n        total += 1\n",
        limits=limits,
    )
    assert exc.kind == "cpu", exc


def test_the_run_budget_stops_a_model_that_answers_too_slowly_to_be_useful():
    limits = SandboxLimits(wall_seconds=60.0, total_seconds=3.0)
    exc = _refusal("import time\n\ndef make_model():\n    time.sleep(600)\n", limits=limits)
    assert exc.kind == "timeout"
    assert "budget" in str(exc)


def test_an_oversized_request_is_refused_rather_than_buffered():
    import numpy as np
    import pandas as pd

    with sandbox_forecaster(_MEAN_MODEL,
                            limits=SandboxLimits(max_message_bytes=1024)) as factory:
        model = factory()
        X = pd.DataFrame(np.zeros((5000, 10)), columns=[f"f{i}" for i in range(10)])
        with pytest.raises(SandboxError) as excinfo:
            model.fit(X, pd.Series(np.zeros(5000)))
    assert excinfo.value.kind == "too_large"


# ---------------------------------------------------------------------------
# 4. Nothing is left running
# ---------------------------------------------------------------------------

def test_closing_the_sandbox_leaves_no_process_behind():
    """The launcher trap, pinned.

    On Windows a venv's python.exe starts the real interpreter as a CHILD, so
    killing the pid we spawned left the model running. This asserts the whole
    tree is gone.
    """
    psutil = pytest.importorskip("psutil")
    factory = sandbox_forecaster(_RIDGE)
    factory.start()
    tree = [proc.pid for proc in factory._channel._tree()]
    assert tree, "expected the sandbox process tree to exist while it runs"
    factory.close()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        alive = [pid for pid in tree if psutil.pid_exists(pid)]
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, f"processes survived close(): {alive}"


def test_a_model_that_kills_its_own_process_is_reported_not_retried():
    exc = _refusal("import os\n\ndef make_model():\n    os._exit(7)\n")
    # os._exit during the probe takes the worker down mid-answer: the parent has
    # to say that, rather than time out or (worse) hang.
    assert exc.kind in ("child_gone", "module:import", "import")
    assert "exited" in str(exc) or "os._exit" in str(exc) or "SystemExit" in str(exc)


def test_the_scratch_directory_is_removed_on_close():
    factory = sandbox_forecaster(_RIDGE)
    factory.start()
    scratch = pathlib.Path(factory._channel.scratch)
    module_file = pathlib.Path(factory.module_path)
    assert scratch.is_dir() and module_file.is_file()
    factory.close()
    assert not scratch.exists()


def test_keep_scratch_leaves_the_module_for_inspection(tmp_path):
    factory = sandbox_forecaster(_RIDGE, keep_scratch=True)
    factory.start()
    scratch = pathlib.Path(factory._channel.scratch)
    factory.close()
    try:
        assert scratch.is_dir()
        assert (scratch / "uploaded_model.py").read_text(encoding="utf-8") == _RIDGE
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. The plumbing is honest
# ---------------------------------------------------------------------------

def test_a_point_model_stays_point_only_through_the_proxy():
    """If the proxy always exposed predict_interval, the harness would calibrate
    an interval the model never claimed."""
    with sandbox_forecaster(_RIDGE) as factory:
        assert factory.interval_capable is False
        assert has_predict_interval(factory()) is False


def test_an_interval_model_keeps_its_interval():
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"f0": [1.0, 2.0]})
    with sandbox_forecaster(_INTERVAL_MODEL) as factory:
        assert factory.interval_capable is True
        model = factory()
        assert has_predict_interval(model) is True
        model.fit(X, pd.Series([5.0, 5.0]))
        lower, upper = model.predict_interval(X)
        assert np.array_equal(np.asarray(lower), np.array([4.0, 4.0]))
        assert np.array_equal(np.asarray(upper), np.array([6.0, 6.0]))


def test_the_report_describes_the_model_that_ran_not_the_proxy():
    cells = _fleet(n_cells=3, n_cycles=120)
    with sandbox_forecaster(_RIDGE) as factory:
        identity = forecaster_identity(factory)
        report = validate_forecaster(
            cells, model=factory, splits=("lco",), intervals=False, dataset="sandbox-identity"
        )
    assert identity["class"] == "SklearnForecaster"
    assert identity["sandboxed"] is True
    assert identity["entry_point"] == "make_model"
    assert identity["pid"]
    assert identity["limits"]["memory_mb"] > 0
    assert report["model"]["identity"]["sandboxed"] is True
    assert report["model"]["identity"]["class"] == "SklearnForecaster"


def test_what_the_model_printed_is_captured_and_returned():
    """A print() in model code must not be read as a protocol message, and must
    not be thrown away either."""
    import numpy as np
    import pandas as pd

    with sandbox_forecaster(_MEAN_MODEL) as factory:
        model = factory()
        model.fit(pd.DataFrame({"f0": [1.0]}), pd.Series([1.0]))
        predicted = np.asarray(model.predict(pd.DataFrame({"f0": [1.0]})))
        output = factory.output
        # ... and using the closed sandbox is refused rather than silently
        # starting a second child with none of the state.
        factory.close()
        with pytest.raises(SandboxError) as excinfo:
            model.predict(pd.DataFrame({"f0": [1.0]}))
        assert excinfo.value.kind == "closed"
    assert "make_model called" in output
    assert predicted.shape == (1,)


def test_enforcement_is_described_not_claimed():
    """The report says which controls exist HERE, including the ones that do not.

    A "sandboxed" claim that hides the platform-specific half would be the exact
    overstatement this package exists to catch, so the wording itself is pinned.
    """
    enforced = describe_enforcement()
    for key in ("process", "wall_clock", "memory", "cpu", "imports", "operations",
                "file_writes", "not_enforced", "platform"):
        assert enforced.get(key), key
    assert "not" in enforced["not_enforced"] and "OS-level" in enforced["not_enforced"]
    assert "audit hook" in enforced["imports"]
    assert "separate child process" in enforced["process"]


def test_the_sandbox_reports_its_limits_to_the_caller():
    limits = SandboxLimits(wall_seconds=5, total_seconds=15, memory_mb=2048, cpu_seconds=10)
    with sandbox_forecaster(_RIDGE, limits=limits) as factory:
        assert factory.enforced["platform"] == sys.platform
        assert "5 s" in factory.enforced["wall_clock"]
        assert "2048 MiB" in factory.enforced["memory"]
    assert limits.to_dict()["memory_mb"] == 2048


def test_the_memory_cap_counts_the_platform_stack_it_must_load_first():
    """A cap below the stack's own footprint refuses every model, not big ones.

    Pinned because the number is not obvious: importing numpy/pandas/sklearn in
    the child costs hundreds of megabytes before any model exists, so the
    default is 4 GiB and the documented floor is above the stack. A deployment
    that sets a cap under it gets a refusal that says "memory", which is
    correct but would look like a bug without this note.
    """
    assert SandboxLimits().memory_mb >= 1024
    exc = _refusal(_RIDGE, limits=SandboxLimits(memory_mb=120, wall_seconds=30))
    assert exc.kind == "memory"
    assert "120 MiB" in str(exc)


def test_empty_source_is_refused_before_any_process_starts():
    """`kind` is the shared vocabulary with batlab.harness.model_source, so a
    caller that maps kinds to wording works across both paths."""
    for bad in ("", "   \n"):
        with pytest.raises(SandboxError) as excinfo:
            SandboxedModelFactory(bad)
        assert excinfo.value.kind == "empty"


def test_module_failure_kinds_match_the_in_process_loader():
    """Same names for the same failures, in the sandbox and out of it.

    The app translates these kinds into one set of sentences. A sandbox that
    invented its own names would leave the page with two vocabularies and half
    the failures unworded.
    """
    from batlab.harness.model_source import import_module_source, ModuleSourceError

    cases = (
        ("def make_model(:\n", "syntax"),
        ("x = 1\n", "missing_entry_point"),
        ("def make_model():\n    raise RuntimeError('boom')\n", "entry_raised"),
        ("def make_model():\n    return None\n", "returned_none"),
    )
    for source, expected in cases:
        with pytest.raises(ModuleSourceError) as direct:
            import_module_source(source)
        assert direct.value.kind == expected, source
        exc = _refusal(source)
        assert exc.kind == expected, (source, exc.kind)
