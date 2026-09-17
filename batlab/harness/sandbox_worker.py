"""
The sandboxed side: import one model module and fit/predict for the parent.

Started and driven by `batlab.harness.sandbox` (`python -m
batlab.harness.sandbox_worker`), which owns the caps, the watchdog and the
protocol. This process is the part that holds the untrusted code, so it is
deliberately small, single-purpose and disposable: it reads one module's
source, imports it under a policy, and answers fit/predict commands with
arrays. Nothing it does is trusted by the parent except the arrays it returns.

What this side enforces
-----------------------
An audit hook (`sys.addaudithook`, which cannot be removed once installed)
checks every audited operation the model performs:

* **Imports** — a denylist of modules whose only purpose here is to reach
  outside the process: `socket`/`ssl`/`http`/`urllib`/`ftplib`/`smtplib`/
  `telnetlib`/`xmlrpc` (network), `subprocess`/`multiprocessing` (processes),
  `ctypes` (foreign function calls), `signal`/`resource` (its own limits),
  `asyncio`, `pty`/`fcntl`/`tty`/`termios`/`mmap` (OS interfaces). Everything
  else already installed stays importable, so a normal model's dependencies
  keep working — a model that imports `requests` fails when `requests` imports
  `socket`, which is the same refusal stated from further away.
* **Audited escape hatches** — `socket.connect`/`bind`/`getaddrinfo`/`sendto`,
  `subprocess.Popen`, `os.system`/`exec`/`posix_spawn`/`spawn`/`fork`,
  `ctypes.dlopen`/`dlsym`/`call_function`. Each raises before performing the
  operation, so a denial names what was attempted.
* **Writes** — `open` for writing, and `os.remove`/`rename`/`mkdir`/`rmdir`/
  `chmod`/`utime`/`truncate`/`symlink`/`link`, are allowed only inside the
  scratch directory the parent created for this process (which is also what
  `TMPDIR`/`TEMP`/`TMP` point at). Reads are untouched: sklearn and numpy
  load their own data and libraries, and reading is not how a graded model
  damages anything.

The policy goes in AFTER the platform's own stack is loaded
------------------------------------------------------------
The first thing this process does is import numpy/pandas/sklearn (the libraries
any model here is written against) and batlab's model machinery, and only then
installs the hook. That ordering is not an optimisation, it is a finding from
testing: numpy's own init opens a native library through `ctypes.CDLL`, which
raises the audited `ctypes.dlopen` event — so a policy installed before it
denied *every* model the platform was built to grade. The hook governs what the
MODEL does, not what the platform loads for it, and the two are the same only if
the platform's half happens first.

What this side does NOT enforce, said plainly
---------------------------------------------
An audit hook is a **Python-level** control. It cannot bind a native extension
that does its own I/O (a compiled `.so` calling `connect(2)` directly), it does
not stop a `SIGKILL`ed process from having already written a file, and the
process still runs as the same OS user in the same filesystem namespace — so it
can *read* anything that user can read. It is containment for accidents and
ordinary malicious code, not an OS-level sandbox (a container, a seccomp
filter, or a separate uid is). The parent's rlimits/watchdog bound the damage
in time, memory and CPU; that pairing is the honest claim, and
`SandboxedModelFactory.enforced` reports which half is actually in force on the
machine that ran it.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# First, before anything the model can influence: the protocol channel is the
# REAL stdout, kept private here, and sys.stdout/sys.stderr become buffers.
_CONTROL = sys.stdout

_MAX_OUTPUT_CHARS = 4000


class _Captured:
    """A stdout/stderr sink that keeps the first N characters and nothing else.

    Model code prints — a progress line, a warning, a debug dump. None of it may
    reach the protocol channel, and an unbounded buffer is its own (small) denial
    of service, so this keeps a head and a count.
    """

    def __init__(self, name: str, limit: int = _MAX_OUTPUT_CHARS) -> None:
        self._name = name
        self._limit = limit
        self._chunks: list[str] = []
        self._size = 0
        self.dropped = 0

    def write(self, text: Any) -> int:
        text = str(text)
        self._size += len(text)
        if self._size <= self._limit:
            self._chunks.append(text)
        elif not self._chunks or self._size - len(text) < self._limit:
            self._chunks.append(text[: self._limit - sum(map(len, self._chunks))])
        else:
            self.dropped += len(text)
        return len(text)

    def flush(self) -> None:  # io protocol
        pass

    def isatty(self) -> bool:
        return False

    def drain(self) -> str:
        text = "".join(self._chunks)
        self._chunks = []
        self._size = 0
        dropped, self.dropped = self.dropped, 0
        if dropped:
            text += f"\n... [{self._name}: {dropped} more characters suppressed]"
        return text


_MODEL_STDOUT = _Captured("stdout")
_MODEL_STDERR = _Captured("stderr")


class SandboxPolicyError(PermissionError):
    """The sandbox refused an operation the model attempted."""


# Modules whose purpose is to reach outside this process. Importing any of them
# (or something that imports them) is refused, and the refusal names it.
DENIED_IMPORT_ROOTS = frozenset({
    # network
    "socket", "_socket", "ssl", "_ssl", "http", "urllib", "ftplib", "smtplib",
    "poplib", "imaplib", "telnetlib", "xmlrpc", "nntplib", "webbrowser",
    # processes and threads-out-of-process
    "subprocess", "multiprocessing", "concurrent.futures.process", "pty",
    # foreign function interface
    "ctypes", "cffi", "_ctypes",
    # the OS itself, or our own limits
    "signal", "resource", "fcntl", "termios", "tty", "mmap", "pwd", "grp",
    "spwd", "crypt", "syslog",
    # asyncio brings an event loop (and its own socket path) with it
    "asyncio",
})

# Audited operations that are refused outright, whoever performs them.
DENIED_AUDIT_EVENTS = frozenset({
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.sendto", "socket.sendmsg",
    "subprocess.Popen",
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function", "ctypes.addressof",
    "ctypes.create_string_buffer", "ctypes.set_errno",
})

# Audited operations allowed only inside the scratch directory.
_PATH_AUDIT_EVENTS = frozenset({
    "open", "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod",
    "os.chown", "os.utime", "os.truncate", "os.symlink", "os.link",
})

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _inside(path: Any, roots: "list[Path]") -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return True  # a file descriptor, not a path we can police
    try:
        resolved = Path(os.fsdecode(path)).resolve()
    except (OSError, ValueError):  # pragma: no cover - pathological paths
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _writes(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
        return True
    return bool(isinstance(flags, int) and flags & _WRITE_FLAGS)


def _path_args(event: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """The path arguments an audited event carries — at most two.

    Slices rather than indexes on purpose: for a variadic tuple, a
    `len(args) > 1` test narrows the false branch to the empty tuple, so an
    `args[0]` there reads as provably out of range to the checker even though
    it cannot be. An event with no argument at all is refused (fail closed)
    rather than silently treated as having nothing to police.
    """
    if not args:
        raise SandboxPolicyError(
            f"the sandbox cannot police {event}: the audited event carried no path."
        )
    if event == "open":
        return args[:1]
    if event in ("os.rename", "os.link", "os.symlink"):
        return args[:2]
    return args[:1]


# Imported before the policy exists, so the model inherits a loaded stack
# rather than paying for one through a hook that watches every import.
_PRELOAD = (
    "numpy", "pandas", "sklearn", "sklearn.base", "sklearn.preprocessing",
    "sklearn.ensemble", "sklearn.linear_model", "sklearn.pipeline",
    "scipy", "joblib", "threadpoolctl",
    "batlab.harness", "batlab.harness.forecaster", "batlab.harness.model_source",
)


def preload_platform_stack() -> "list[str]":
    """Import what a model written for this platform expects to already exist.

    Best-effort by design: a deployment without scipy still grades models that
    do not need it. Returns the names that loaded, for the child's own report.
    """
    loaded = []
    for name in _PRELOAD:
        try:
            __import__(name)
            loaded.append(name)
        except Exception:
            continue
    return loaded


def purge_denied_modules() -> "list[str]":
    """Forget denied modules the platform's own preload brought in.

    The audited `import` event fires when the import system LOADS a module, not
    when a name is already in `sys.modules` — and the preload above drags in
    socket, ssl, subprocess, multiprocessing, ctypes, asyncio, signal and mmap
    as transitive dependencies. Without this purge, `import socket` in a model
    would succeed silently on an already-cached module while the same import in
    a fresh process was refused: the policy would depend on what the platform
    happened to import first. Dropping the entries makes the model's own import
    hit the hook, which is the refusal it should have had.

    Nothing here depends on any of them (the worker talks over pipes), so the
    only thing a purge can break is code that reaches for those modules — which
    is exactly the code the policy exists to stop.
    """
    purged = []
    for name in list(sys.modules):
        root = name.partition(".")[0]
        if root in DENIED_IMPORT_ROOTS:
            sys.modules.pop(name, None)
            purged.append(name)
    return purged


def install_policy(scratch_dirs: "list[str]", *, extra_denied_imports: "tuple[str, ...]" = ()) -> None:
    """Install the audit hook. Cannot be undone — which is the point.

    Raises SandboxPolicyError from inside the audited operation, so the caller
    that attempted the import/write/connection sees a normal exception naming
    the refusal, and this process can report it as a policy failure rather than
    as a model bug.
    """
    roots = [Path(p).resolve() for p in scratch_dirs]
    denied_imports = DENIED_IMPORT_ROOTS | {str(m) for m in extra_denied_imports}

    def hook(event: str, args: tuple) -> None:
        if event == "import":
            name = str(args[0]) if args else ""
            root = name.partition(".")[0]
            if root in denied_imports or name in denied_imports:
                raise SandboxPolicyError(
                    f"the sandbox refuses to import {name!r}: it reaches outside this "
                    "process (network, subprocesses, foreign function calls, or the "
                    "OS/limits themselves). Model modules are graded without those."
                )
            return
        if event in DENIED_AUDIT_EVENTS:
            raise SandboxPolicyError(
                f"the sandbox refuses {event}: a graded model does not need to reach "
                "outside its own process."
            )
        if event in _PATH_AUDIT_EVENTS:
            if event == "open" and not _writes(args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else None):
                return  # reads are allowed
            for path in _path_args(event, args):
                if not _inside(path, roots):
                    raise SandboxPolicyError(
                        f"the sandbox refuses {event} on {os.fsdecode(path)!r}: writes "
                        "are confined to the scratch directory this process was given."
                    )

    sys.addaudithook(hook)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

class _ProtocolError(Exception):
    """The parent sent something this side cannot execute."""


def _respond(payload: dict) -> None:
    output = _MODEL_STDOUT.drain()
    errors = _MODEL_STDERR.drain()
    if output:
        payload.setdefault("stdout", output)
    if errors:
        payload.setdefault("stderr", errors)
    _CONTROL.write(json.dumps(payload, separators=(",", ":")) + "\n")
    _CONTROL.flush()


def _fail(kind: str, message: str, **extra: Any) -> None:
    _respond({"ok": False, "kind": kind, "error": message, **extra})


def _encode(array: Any) -> dict:
    """A float64 buffer as JSON: shape + base64 of the raw bytes.

    Raw bytes, not formatted text: a run that grades your model must compare
    against the in-process result exactly, and "0.30000000000000004" printed
    and re-parsed is not the same double.
    """
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


def _decode_features(message: dict) -> Any:
    import pandas as pd

    values = _decode(message)
    if len(values.shape) != 2:
        raise _ProtocolError("feature matrix must be two-dimensional")
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(values.shape[1])])


# How many fitted models the child keeps alive at once. The harness's pattern is
# fit-then-predict-immediately (run_lco calls the factory per fold and target and
# predicts straight away), so a handful is plenty; the cap exists so a run that
# keeps handles around cannot grow the child's memory without bound.
MAX_LIVE_HANDLES = 16


class _HandlePool:
    """Fitted models, one per handle the parent asked for.

    Sharing ONE fitted model between handles was the first version, and it is
    exactly the failure this package exists to prevent: fitting handle B and
    then predicting on handle A answered with B's fit. The harness contract is a
    fresh model per factory call, so a fresh model per handle is what the child
    holds — the pool is only about which of them are still alive, and an
    evicted handle says so by name instead of answering with someone else's fit.
    """

    def __init__(self, factory: Any, *, max_live: int = MAX_LIVE_HANDLES) -> None:
        self._factory = factory
        self._max_live = max(1, int(max_live))
        self._models: "dict[str, Any]" = {}

    def fit(self, handle: str, X: Any, y: Any) -> None:
        from batlab.harness.forecaster import fit_forecaster

        # A NEW model per fit, built by the module's own make_model().
        self._models[handle] = fit_forecaster(self._factory(), X, y)
        while len(self._models) > self._max_live:
            oldest = next(iter(self._models))
            del self._models[oldest]

    def get(self, handle: str) -> Any:
        try:
            return self._models[handle]
        except KeyError:
            raise _ProtocolError(
                f"no fitted model for this handle, and the sandbox will not guess: "
                f"it keeps the last {self._max_live} handles alive, so either this "
                "handle was never fitted or it was evicted by newer fits"
            ) from None


def _handle_load(message: dict, state: dict) -> None:
    from batlab.harness.forecaster import forecaster_identity, has_predict_interval
    from batlab.harness.model_source import ModuleSourceError, import_module_source

    source = message.get("source")
    entry_point = str(message.get("entry_point") or "make_model")
    scratch = message.get("scratch") or state["scratch"]
    path = Path(scratch) / "uploaded_model.py"
    # Written BEFORE the policy goes in, by design: the policy is installed
    # immediately after, so the module's own import is the first thing it sees.
    path.write_bytes(str(source).encode("utf-8"))

    if not state["policy_installed"]:
        preload_platform_stack()
        state["purged"] = purge_denied_modules()
        install_policy([scratch], extra_denied_imports=tuple(message.get("denied_imports") or ()))
        state["policy_installed"] = True

    try:
        module = import_module_source(
            str(source),
            module_name="batlab_sandboxed_upload",
            entry_point=entry_point,
            path=path,
        )
    except SandboxPolicyError as exc:
        return _fail("policy", str(exc), entry_point=entry_point)
    except ModuleSourceError as exc:
        # A policy refusal that surfaced through the module's import or its
        # entry-point call is reported as a POLICY failure, not as the module
        # raising an exception: the model is refused because of what it tried
        # to do, and saying "your make_model() raised SandboxPolicyError" would
        # bury the only sentence that matters.
        if isinstance(exc.exc, SandboxPolicyError):
            return _fail("policy", str(exc.exc), entry_point=exc.entry_point, lineno=exc.lineno)
        return _fail(
            f"module:{exc.kind}",
            str(exc),
            entry_point=exc.entry_point,
            lineno=exc.lineno,
        )
    except BaseException as exc:  # incl. SystemExit from the module itself
        return _fail(
            "module:import",
            f"importing the module raised {type(exc).__name__}: {exc}",
            entry_point=entry_point,
        )

    probe = module["probe"]
    state["models"] = _HandlePool(
        module["factory"], max_live=int(message.get("max_live_handles") or MAX_LIVE_HANDLES)
    )
    _respond({
        "ok": True,
        "entry_point": entry_point,
        "path": str(path),
        "probe_class": module["probe_class"],
        "interval_capable": has_predict_interval(probe),
        "callables": module["callables"],
        "identity": forecaster_identity(probe),
        "python": sys.version.split()[0],
        "purged_modules": state.get("purged") or [],
    })


def _handle_fit(message: dict, state: dict) -> None:
    state["models"].fit(
        str(message.get("handle") or ""),
        _decode_features(message["X"]),
        _decode(message["y"]).reshape(-1),
    )
    _respond({"ok": True, "live_handles": len(state["models"]._models)})


def _handle_predict(message: dict, state: dict) -> None:
    model = state["models"].get(str(message.get("handle") or ""))
    _respond({"ok": True, "y": _encode(model.predict(_decode_features(message["X"])))})


def _handle_interval(message: dict, state: dict) -> None:
    model = state["models"].get(str(message.get("handle") or ""))
    lower, upper = model.predict_interval(_decode_features(message["X"]))
    _respond({"ok": True, "lower": _encode(lower), "upper": _encode(upper)})


_HANDLERS = {
    "load": _handle_load,
    "fit": _handle_fit,
    "predict": _handle_predict,
    "interval": _handle_interval,
}


def main() -> int:
    """Read commands on stdin, answer on the private control channel, exit on stop.

    Every command is answered exactly once, including failures: the parent waits
    with a deadline, and a silent death would be reported as a timeout rather
    than as the refusal it actually was.
    """
    sys.stdout = _MODEL_STDOUT  # type: ignore[assignment]
    sys.stderr = _MODEL_STDERR  # type: ignore[assignment]

    import tempfile

    state: dict = {
        "scratch": tempfile.gettempdir(),
        "models": _HandlePool(lambda: None),
        "policy_installed": False,
    }
    # No unsolicited line here: the parent answers a `ping` before it sends
    # anything else, and a stray greeting would be read as that answer — which
    # is exactly how the first version of this file desynchronised the protocol.

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail("protocol", f"unreadable command: {exc}")
            continue

        command = str(message.get("cmd") or "")
        if command == "stop":
            _respond({"ok": True, "stopping": True})
            return 0
        if command == "ping":
            _respond({"ok": True, "pong": True})
            continue
        handler = _HANDLERS.get(command)
        if handler is None:
            _fail("protocol", f"unknown command {command!r}")
            continue

        try:
            handler(message, state)
        except SandboxPolicyError as exc:
            _fail("policy", str(exc))
        except _ProtocolError as exc:
            # The parent asked for something impossible (an unknown handle, a
            # predict before fit). That is a protocol failure, not a model bug,
            # and saying so keeps the model's own failures unambiguous.
            _fail("protocol", str(exc))
        except BaseException as exc:  # noqa: BLE001 - the parent decides what matters
            kind = {"load": "module:import", "fit": "fit", "predict": "predict",
                    "interval": "interval"}.get(command, "internal")
            trace = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _fail(kind, trace, traceback="".join(traceback.format_tb(exc.__traceback__))[-2000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
