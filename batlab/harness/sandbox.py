"""
Grade a model you did not write without giving it this process.

Why this exists
---------------
The harness's contract with a supplied model is a factory that returns a fresh,
unfitted forecaster (see `batlab.harness.forecaster`). An uploaded `.py` module
satisfies that contract by being *imported*, and importing a module executes it
— with this process's privileges, its environment, its open files and its
network. The app disclosed that in as many words ("there is no sandbox"), which
is honest but is not something to offer on a shared deployment: the page's whole
premise is that a stranger can upload a model and be graded, and the honest
version of that promise has to include "and it cannot read your database".

So the model runs somewhere else. `SandboxedModelFactory` starts one child
process, loads the module there under a policy, and answers every `fit`,
`predict` and `predict_interval` by sending arrays over a pipe and receiving
predictions back. The parent never imports the model. What the parent holds is
a proxy, and the proxy reports what the child actually graded (see
`sandboxed_identity`) rather than claiming to be the model itself.

Two halves, and both are reported
---------------------------------
*Containment (the child).* `batlab.harness.sandbox_worker` installs an audit
hook before the module is imported: denied imports (network, subprocess, ctypes,
signal/resource, asyncio), denied audited operations (`socket.*`,
`subprocess.Popen`, `os.system`/`exec`/`fork`, `ctypes.dlopen`), and writes
confined to a scratch directory. Its docstring states exactly what that does and
does not stop — it is a Python-level control, not an OS sandbox.

*Limits (the parent).* Wall-clock per call and for the whole run, enforced with
a hard kill; CPU seconds and address-space limits as POSIX rlimits where the
platform has them; and, when `psutil` is importable, a watchdog that samples the
child's RSS and CPU time while it works and kills it if either cap is passed.
What is in force on the machine that ran a given model is not a guess: it is
`sandbox.enforced`.

The consequence for the numbers
-------------------------------
Arrays cross as raw float64 buffers (base64), never as formatted text, so a
sandboxed fit sees the same bits the in-process fit would see and the graded
metrics agree exactly rather than approximately. `tests/test_sandbox.py` pins
that equality, because a sandbox that quietly changed the number it was
measuring would be a worse instrument than no sandbox at all.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ALLOWED_DENIED_IMPORTS",
    "SandboxError",
    "SandboxLimits",
    "SandboxedModelFactory",
    "describe_enforcement",
    "sandbox_forecaster",
]

# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxLimits:
    """What the parent is willing to spend on a model nobody vouched for.

    Every field has a default that a published validation run fits inside; a
    model that needs more is not refused, it just has to say so, and the honest
    place to say so is a parameter rather than a silent exemption.
    """

    wall_seconds: float = 180.0
    """Hard limit for ONE fit/predict round trip (the parent kills the child)."""

    total_seconds: float = 1800.0
    """Hard limit for the whole run, all folds and targets together."""

    memory_mb: int = 4096
    """Address space (POSIX rlimit) and/or observed virtual size + RSS (watchdog).

    Counted against the whole child tree, INCLUDING the platform stack it loads
    before the model does: numpy/pandas/sklearn cost roughly 200 MiB of RSS and
    400 MiB of virtual size before a single model exists, so a cap below that
    refuses every model rather than oversized ones.
    """

    cpu_seconds: int = 600
    """CPU time, not wall time: a busy model is killed even if it yields."""

    max_message_bytes: int = 256 * 1024 * 1024
    """Refuse a single request/response larger than this instead of buffering it."""

    def to_dict(self) -> dict:
        return {
            "wall_seconds": self.wall_seconds,
            "total_seconds": self.total_seconds,
            "memory_mb": self.memory_mb,
            "cpu_seconds": self.cpu_seconds,
            "max_message_bytes": self.max_message_bytes,
        }


def _load_psutil() -> Any:
    """psutil if it happens to be installed — never a dependency of batlab.

    It is imported by the app's own stack (Streamlit ships it), so on a
    deployment that runs the page the watchdog is normally available; the
    sandbox works without it and says so in `enforced`.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return psutil


def describe_enforcement(limits: "SandboxLimits | None" = None) -> dict:
    """What is actually enforced on THIS machine, in words a reviewer can check.

    Not a capability list — a report. POSIX gets hard rlimits; Windows has no
    equivalent, so it gets the wall clock plus (psutil permitting) a watchdog
    that kills on RSS/CPU. A claim of "sandboxed" that does not distinguish those
    two would be exactly the kind of overstatement this package exists to catch.
    """
    limits = limits or SandboxLimits()
    posix = sys.platform != "win32"
    watchdog = _load_psutil() is not None
    if posix:
        memory = f"RLIMIT_AS {limits.memory_mb} MiB (hard, kernel-enforced)"
        cpu = f"RLIMIT_CPU {limits.cpu_seconds} s (hard, kernel-enforced)"
    elif watchdog:
        memory = (
            f"observed virtual size and RSS capped at {limits.memory_mb} MiB by the "
            "parent, sampled every 100 ms (psutil) — a sampled cap bounds damage, "
            "it does not prevent it"
        )
        cpu = f"observed CPU time capped at {limits.cpu_seconds} s by the parent (psutil)"
    else:
        memory = "not enforced on this platform (no rlimits, psutil unavailable)"
        cpu = "not enforced on this platform (no rlimits, psutil unavailable)"
    return {
        "process": "the model runs in a separate child process; this process never imports it",
        "wall_clock": (
            f"{limits.wall_seconds:g} s per fit/predict call, "
            f"{limits.total_seconds:g} s for the whole run — the parent kills the child"
        ),
        "memory": memory,
        "cpu": cpu,
        "imports": (
            "denied in the child by an audit hook: socket/ssl/http/urllib/ftplib/"
            "smtplib/telnetlib/xmlrpc, subprocess/multiprocessing, ctypes, "
            "signal/resource/fcntl/mmap/asyncio — everything else installed stays importable"
        ),
        "operations": (
            "denied in the child by an audit hook: socket.connect/bind/getaddrinfo, "
            "subprocess.Popen, os.system/exec/fork/posix_spawn, ctypes.dlopen"
        ),
        "file_writes": "confined in the child to its own scratch directory (reads are untouched)",
        "not_enforced": (
            "an OS-level boundary: the child runs as this user with read access to what "
            "this user can read, and a compiled extension could bypass the audit hook. "
            "This is containment, not a container."
        ),
        "platform": sys.platform,
    }


# Imports the child refuses beyond its own defaults. Kept here (not in the app)
# so a deployment can tighten the policy without editing the child's code.
DEFAULT_ALLOWED_DENIED_IMPORTS: tuple = ()


class SandboxError(RuntimeError):
    """The sandbox could not answer — a refusal, a crash, or a limit.

    `kind` is the stable part; branch on it, never on the message:
      policy | module:empty | module:syntax | module:import | module:missing_entry_point
      | module:entry_raised | module:returned_none | module:not_gradable
      | fit | predict | interval | timeout | memory | cpu | protocol
      | spawn | child_gone | too_large | closed
    """

    def __init__(self, kind: str, message: str, *, detail: str = "", extra: "dict | None" = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail
        # Structured fields the child reported (lineno, entry_point, path), so a
        # caller can re-render a module failure in its own wording instead of
        # pattern-matching a sentence.
        self.extra = dict(extra or {})


def _encode(array: Any) -> dict:
    """Raw float64 bytes, base64 — bit-exact, unlike printed decimals."""
    import numpy as np

    values = np.ascontiguousarray(np.asarray(array, dtype="float64"))
    return {
        "shape": list(values.shape),
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def _decode(message: dict) -> Any:
    import numpy as np

    raw = base64.b64decode(message["data"])
    return np.frombuffer(raw, dtype="float64").reshape(tuple(message["shape"]))


def _features(X: Any) -> dict:
    import numpy as np

    values = np.asarray(X, dtype="float64")
    if values.ndim != 2:
        raise SandboxError("protocol", f"expected a 2-D feature matrix, got shape {values.shape}")
    return _encode(values)


class _Channel:
    """One child process and the pipe protocol to it.

    Owns the process lifetime, the reader thread, the deadline and the watchdog,
    so the proxy above it can be written as if calls were local.
    """

    def __init__(
        self,
        *,
        limits: SandboxLimits,
        python: "str | None" = None,
        env: "dict | None" = None,
        denied_imports: "tuple[str, ...]" = DEFAULT_ALLOWED_DENIED_IMPORTS,
        keep_scratch: bool = False,
    ) -> None:
        self.limits = limits
        self.denied_imports = tuple(denied_imports)
        self.python = python or sys.executable
        self._env_extra = dict(env or {})
        self.keep_scratch = keep_scratch
        self._proc: "subprocess.Popen | None" = None
        self._queue: "queue.Queue" = queue.Queue()
        self._reader: "threading.Thread | None" = None
        self._scratch: "Path | None" = None
        self._stderr_tail: list[str] = []
        self._stderr_thread: "threading.Thread | None" = None
        # What the model printed, captured in the child so it cannot corrupt the
        # protocol and returned with each answer. Kept (bounded) rather than
        # dropped: a model's own warnings are part of reading its result.
        self.output: list[str] = []
        self._started_at = 0.0
        self._psutil_proc: Any = None
        self._closed = False
        # The harness runs folds in parallel (batlab/_parallel.map_folds), so two
        # threads can ask this channel for a fit at the same time. One pipe means
        # one command in flight: without this lock the answers interleave and a
        # predict reads the fit's answer — which is a wrong number, not an error.
        self._request_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def scratch(self) -> Path:
        if self._scratch is None:
            self._scratch = Path(tempfile.mkdtemp(prefix="batlab_sandbox_"))
        return self._scratch

    def _environment(self) -> dict:
        env = dict(os.environ)
        # The child resolves batlab the same way this process did (the app runs
        # from a checkout as often as from an installed package).
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(self.scratch)
        env["BATLAB_SANDBOX"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Nothing here is a secret the model needs, and the fewer handles it
        # inherits the better.
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "BATLAB_JWT_SECRET"):
            env.pop(key, None)
        env.update(self._env_extra)
        return env

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._closed:
            # Never silently respawn: a caller that kept using a closed sandbox
            # would get a FRESH child with no fitted state, and the orphaned
            # process it thought it had stopped would still be running.
            raise SandboxError(
                "closed",
                "this sandbox was closed; build a new one (a closed sandbox never "
                "starts a second child process).",
            )
        self.scratch.mkdir(parents=True, exist_ok=True)
        popen_kwargs: dict = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            env=self._environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        preexec = self._preexec()
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec
        try:
            self._proc = subprocess.Popen(
                [self.python, "-u", "-m", "batlab.harness.sandbox_worker"], **popen_kwargs
            )
        except OSError as exc:
            raise SandboxError(
                "spawn",
                f"could not start the sandbox process ({self.python}): {exc}",
            ) from exc
        self._started_at = time.monotonic()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        psutil = _load_psutil()
        self._psutil_proc = psutil.Process(self._proc.pid) if psutil is not None else None

    def _preexec(self):
        """POSIX-only hard caps, applied in the child before it execs.

        Both soft and hard limits are set, so the child cannot raise its own.
        """
        if sys.platform == "win32":
            return None
        try:
            import resource
        except ImportError:  # pragma: no cover - POSIX always has it
            return None
        limits = self.limits

        def _apply() -> None:  # pragma: no cover - runs inside the child process
            try:
                os.setsid()  # its own process group: one kill takes the whole tree
            except OSError:
                pass
            for which, value in (
                (getattr(resource, "RLIMIT_CPU", None), limits.cpu_seconds),
                (getattr(resource, "RLIMIT_AS", None), limits.memory_mb * 1024 * 1024),
                (getattr(resource, "RLIMIT_CORE", None), 0),
            ):
                if which is None or not value:
                    continue
                try:
                    resource.setrlimit(which, (value, value))
                except (ValueError, OSError):
                    pass

        return _apply

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._queue.put(line)
        self._queue.put(None)  # EOF: the child is gone

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line)
            del self._stderr_tail[:-40]

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail)[-2000:]

    def request(self, message: dict, *, what: str) -> dict:
        """Send one command and wait for its answer — one at a time, ever.

        Serialized deliberately. The alternative (request ids and a response
        router) would keep the harness's fold parallelism, but the child is one
        interpreter running the model's own code, so the parallelism would be
        inside a process this layer exists to keep simple. The cost is real and
        stated where it is paid: a sandboxed run fits folds one after another,
        so it takes longer than the same model graded in-process.
        """
        with self._request_lock:
            return self._request(message, what=what)

    def _request(self, message: dict, *, what: str) -> dict:
        """The round trip itself; every exit is a named refusal.

        A limit that was passed, a child that died, a protocol that broke — each
        has a kind. The one thing this must never do is wait forever, which is
        why the deadline is recomputed from the clock rather than assumed.
        """
        self.start()
        assert self._proc is not None and self._proc.stdin is not None
        if self._proc.poll() is not None:
            raise self._child_gone(what)

        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(payload) > self.limits.max_message_bytes:
            raise SandboxError(
                "too_large",
                f"{what} would send {len(payload) / 1e6:.0f} MB, above the "
                f"{self.limits.max_message_bytes / 1e6:.0f} MB sandbox limit.",
            )

        deadline = time.monotonic() + self.limits.wall_seconds
        total_deadline = self._started_at + self.limits.total_seconds
        try:
            self._proc.stdin.write(payload.decode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._child_gone(what) from None

        while True:
            now = time.monotonic()
            if now >= deadline:
                self.kill("timeout", f"{what} did not answer within {self.limits.wall_seconds:g} s")
            if now >= total_deadline:
                self.kill(
                    "timeout",
                    f"the run passed its {self.limits.total_seconds:g} s sandbox budget during {what}",
                )
            self._check_watchdog(what)
            try:
                line = self._queue.get(timeout=min(0.1, max(0.0, deadline - now)))
            except queue.Empty:
                continue
            if line is None:
                raise self._child_gone(what)
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SandboxError(
                    "protocol",
                    f"the sandbox sent something that is not a command answer: {exc}",
                    detail=self.stderr_tail,
                ) from exc
            for stream in ("stdout", "stderr"):
                text = answer.pop(stream, None)
                if text:
                    self.output.append(f"[{stream}] {text.strip()}")
                    del self.output[:-40]
            return answer

    def _tree(self) -> list:
        """The sandbox process AND everything it started.

        Not a detail: on Windows a venv's `python.exe` is a launcher that runs
        the real interpreter as a CHILD process. Sampling or killing only the
        pid we spawned would measure a 4 MB stub, miss every byte the model
        allocated, and — worse — leave the model running after we thought we had
        stopped it. The first version of this file did exactly that, which is
        why this method exists and why its behaviour is pinned by a test.
        """
        if self._psutil_proc is None:
            return []
        try:
            return [self._psutil_proc, *self._psutil_proc.children(recursive=True)]
        except Exception:  # process already gone
            return []

    def _check_watchdog(self, what: str) -> None:
        """Parent-side memory/CPU kill, for platforms without rlimits.

        Sampling, not interception: the child can exceed a cap by whatever it
        allocates between two samples (every 0.1 s here). It bounds damage; a
        POSIX rlimit prevents it.
        """
        if self._proc is None:
            return
        rss_mb = vms_mb = cpu = 0.0
        sampled = 0
        for proc in self._tree():
            try:
                info = proc.memory_info()
                # BOTH, deliberately: an untouched np.zeros(2 GB) is virtual
                # memory only — RSS stays flat, so an RSS-only watchdog misses
                # it, and on Windows the virtual size is what the page file
                # actually backs.
                rss_mb += info.rss / (1024 * 1024)
                vms_mb += info.vms / (1024 * 1024)
                cpu += sum(proc.cpu_times()[:2])
                sampled += 1
            except Exception:  # a process that exited between the two calls
                continue
        if not sampled:
            return
        if max(rss_mb, vms_mb) > self.limits.memory_mb:
            self.kill(
                "memory",
                f"{what} pushed the sandbox past {self.limits.memory_mb} MiB "
                f"(RSS {rss_mb:.0f} MiB, virtual {vms_mb:.0f} MiB across "
                f"{sampled} process{'es' if sampled > 1 else ''})",
            )
        if cpu > self.limits.cpu_seconds:
            self.kill("cpu", f"{what} used more than {self.limits.cpu_seconds} s of CPU")

    def _child_gone(self, what: str) -> SandboxError:
        code = self._proc.returncode if self._proc is not None else None
        self._proc = None
        return SandboxError(
            "child_gone",
            f"the sandbox process exited (code {code}) during {what}. The model may have "
            "called os._exit(), crashed the interpreter, or been killed by a limit.",
            detail=self.stderr_tail,
        )

    def _terminate_tree(self, proc: "subprocess.Popen") -> None:
        """Stop the sandbox and every process under it.

        POSIX: the child got its own session via `setsid`, so one signal to the
        group reaches the whole tree. Windows: walk the tree with psutil and
        kill the leaves first — killing the launcher alone leaves the
        interpreter (and the model) running.
        """
        tree = self._tree()
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                for entry in reversed(tree):
                    try:
                        entry.kill()
                    except Exception:  # pragma: no cover - already gone
                        pass
                proc.kill()
        except Exception:  # pragma: no cover - best effort by design
            try:
                proc.kill()
            except Exception:
                pass

    def kill(self, kind: str, message: str) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            self._terminate_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:  # pragma: no cover
                pass
        raise SandboxError(kind, message, detail=self.stderr_tail)

    def close(self) -> None:
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()  # the worker returns on EOF
                proc.wait(timeout=5)
            except Exception:
                pass
            if proc.poll() is None:
                self._terminate_tree(proc)
                try:
                    proc.wait(timeout=5)
                except Exception:  # pragma: no cover
                    pass
        if self._scratch is not None and not self.keep_scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The proxy the harness grades
# ---------------------------------------------------------------------------

class _RemoteModel:
    """A fresh model, on the other side of the pipe.

    `fit` asks the child to build a NEW model from the module's own factory and
    fit it — the per-fold isolation the harness contract requires happens there,
    not here, so a handle is cheap and one fold can never inherit another's fit.
    """

    # itertools.count: next() is atomic under the GIL, and folds really do call
    # the factory from several threads at once.
    _handles = itertools.count(1)

    def __init__(self, channel: _Channel, *, sandboxed_identity: dict) -> None:
        self._channel = channel
        self._identity = sandboxed_identity
        # This handle's own model on the other side: two live handles must never
        # share a fit, which is the whole point of the harness factory contract.
        self.handle = f"handle-{next(_RemoteModel._handles)}"

    # -- what the report records ------------------------------------------
    @property
    def sandboxed_identity(self) -> dict:
        return dict(self._identity)

    @property
    def sandbox(self) -> dict:
        return dict(self._identity.get("sandbox") or {})

    # -- Forecaster --------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "_RemoteModel":
        answer = self._channel.request(
            {"cmd": "fit", "handle": self.handle, "X": _features(X), "y": _encode(y)}, what="fit"
        )
        _raise_for_answer(answer, "fit")
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        answer = self._channel.request(
            {"cmd": "predict", "handle": self.handle, "X": _features(X)}, what="predict"
        )
        _raise_for_answer(answer, "predict")
        return np.asarray(_decode(answer["y"]), dtype=float)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self._identity.get('class', 'model')} "
            f"in sandbox pid={self._identity.get('pid')}>"
        )


class _RemoteIntervalModel(_RemoteModel):
    """Same proxy, plus the interval call — only for models that really have it.

    The class exists so `has_predict_interval()` stays truthful: a point-only
    model on the other side must look point-only here, or the harness would
    calibrate an interval the model never claimed.
    """

    def predict_interval(self, X: Any) -> tuple:
        import numpy as np

        answer = self._channel.request(
            {"cmd": "interval", "handle": self.handle, "X": _features(X)}, what="predict_interval"
        )
        _raise_for_answer(answer, "predict_interval")
        return (
            np.asarray(_decode(answer["lower"]), dtype=float),
            np.asarray(_decode(answer["upper"]), dtype=float),
        )


_WHAT_CONTEXT = {
    "fit": "the model raised while being fitted inside the sandbox",
    "predict": "the model raised while predicting inside the sandbox",
    "interval": "the model raised while producing an interval inside the sandbox",
}


def _raise_for_answer(answer: dict, what: str) -> None:
    if answer.get("ok"):
        return
    kind = str(answer.get("kind") or "protocol")
    error = str(answer.get("error") or f"the sandbox refused {what}")
    prefix = _WHAT_CONTEXT.get(kind)
    raise SandboxError(
        kind,
        f"{prefix}: {error}" if prefix else error,
        detail=str(answer.get("traceback") or ""),
        extra={k: answer.get(k) for k in ("lineno", "entry_point", "path") if answer.get(k) is not None},
    )


class SandboxedModelFactory:
    """Load a model module in a child process and grade it through a proxy.

    Call it like any other harness factory:

        factory = SandboxedModelFactory(open("my_model.py").read())
        report = validate_forecaster(cells, model=factory)
        factory.close()          # or use it as a context manager

    It is a FACTORY and a metadata object at once: `probe_class`,
    `interval_capable`, `callables`, `identity` and `enforced` describe what the
    child holds, and become available when the child is started (lazily, on the
    first factory call, identity request or fit).
    """

    def __init__(
        self,
        source: str,
        *,
        entry_point: str = "make_model",
        limits: "SandboxLimits | None" = None,
        python: "str | None" = None,
        denied_imports: "tuple[str, ...]" = DEFAULT_ALLOWED_DENIED_IMPORTS,
        env: "dict | None" = None,
        keep_scratch: bool = False,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            # Same `kind` vocabulary as batlab.harness.model_source, so a caller
            # that maps kinds to wording keeps working across the two paths.
            raise SandboxError("empty", "the model module source is empty")
        self.source = source
        self.entry_point = str(entry_point or "make_model")
        self.limits = limits or SandboxLimits()
        self._channel = _Channel(
            limits=self.limits,
            python=python,
            env=env,
            denied_imports=denied_imports,
            keep_scratch=keep_scratch,
        )
        self._info: dict = {}
        self._enforced = describe_enforcement(self.limits)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "SandboxedModelFactory":
        """Start the child and load the module into it (idempotent).

        Raises SandboxError with the module's own failure `kind` when the module
        cannot be graded — the same kinds `batlab.harness.model_source` uses, so
        the app's wording for each one stays in one place.
        """
        if self._info:
            return self
        ready = self._channel.request({"cmd": "ping"}, what="startup")
        _raise_for_answer(ready, "startup")
        answer = self._channel.request(
            {
                "cmd": "load",
                "source": self.source,
                "entry_point": self.entry_point,
                "scratch": str(self._channel.scratch),
                "denied_imports": list(self._channel.denied_imports),
            },
            what="loading the model module",
        )
        if not answer.get("ok"):
            kind = str(answer.get("kind") or "module:import")
            kind = kind[len("module:"):] if kind.startswith("module:") else kind
            raise SandboxError(
                kind,
                str(answer.get("error") or "the module could not be loaded"),
                extra={
                    k: answer.get(k)
                    for k in ("lineno", "entry_point", "path")
                    if answer.get(k) is not None
                },
            )
        self._info = answer
        return self

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "SandboxedModelFactory":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- factory contract --------------------------------------------------

    def __call__(self) -> _RemoteModel:
        """A fresh model handle — cheap; the child builds it at fit time."""
        self.start()
        cls = _RemoteIntervalModel if self.interval_capable else _RemoteModel
        return cls(self._channel, sandboxed_identity=self.sandboxed_identity)

    # -- what the child holds ---------------------------------------------

    @property
    def loaded(self) -> bool:
        return bool(self._info)

    @property
    def probe_class(self) -> "str | None":
        return (self._info or {}).get("probe_class")

    @property
    def interval_capable(self) -> bool:
        return bool((self._info or {}).get("interval_capable"))

    @property
    def callables(self) -> list:
        return list((self._info or {}).get("callables") or [])

    @property
    def module_path(self) -> "str | None":
        return (self._info or {}).get("path")

    @property
    def pid(self) -> "int | None":
        proc = self._channel._proc
        return proc.pid if proc is not None else None

    @property
    def enforced(self) -> dict:
        return dict(self._enforced)

    @property
    def output(self) -> str:
        """Everything the model printed, captured on the other side.

        Model code writing to stdout would otherwise be indistinguishable from a
        protocol message, so the child buffers it and sends it back with each
        answer. Empty string when the module was quiet.
        """
        return "\n".join(self._channel.output)[-4000:]

    @property
    def sandboxed_identity(self) -> dict:
        """What `forecaster_identity()` reports for a model that runs elsewhere.

        Starting the child here is deliberate: a report that says "sandboxed
        SklearnForecaster" must be describing a module that actually loaded, not
        a plan to load one.
        """
        self.start()
        identity = dict((self._info or {}).get("identity") or {})
        identity.update({
            "sandboxed": True,
            "entry_point": self.entry_point,
            "pid": self.pid,
            "limits": self.limits.to_dict(),
        })
        return identity


def sandbox_forecaster(
    source: str,
    *,
    entry_point: str = "make_model",
    limits: "SandboxLimits | None" = None,
    **kwargs: Any,
) -> SandboxedModelFactory:
    """`SandboxedModelFactory` in the calling style of `torch_forecaster`."""
    return SandboxedModelFactory(source, entry_point=entry_point, limits=limits, **kwargs)
