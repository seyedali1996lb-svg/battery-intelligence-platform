"""The secrets → os.environ bridge that makes Cloud-set knobs reachable.

Why this guard exists: every deployment knob this app reads —
`BATLAB_BOOT_STUDIES`, `BATLAB_BOOT_LAYERS`, `BATLAB_FOLD_WORKERS`,
`SETTINGS_ENCRYPTION_KEY`, `ANTHROPIC_API_KEY` — comes from `os.environ`, but
Streamlit Community Cloud's only configuration surface is
`.streamlit/secrets.toml` (it has no environment-variable setting). So on a
CPU-throttled instance there was no way to say `BATLAB_BOOT_STUDIES=off`, and
the app's own docs — "Set SETTINGS_ENCRYPTION_KEY via Streamlit Cloud
secrets", "Anthropic API key in .streamlit/secrets.toml" — had no code behind
them. `app/main.py` now applies `app/_secrets_env.py`'s mirror at module top
level.

Contracts pinned here:

  - scalars are mirrored (bools as `true`/`false`); TOML tables are not —
    they are `st.secrets`' structured data, not environment variables
  - a variable already present in the environment is never overwritten: the
    machine outranks the file
  - no secrets file is the normal local case (`StreamlitSecretNotFoundError`
    subclasses `FileNotFoundError`) and must be a quiet no-op
  - a malformed file still raises — a typo'd deployment config should fail
    loudly at boot, not be silently swallowed
  - `main.py` applies the bridge BEFORE `from _data import ...`, so any
    import-time reader of a knob already sees it (AST order, the same idea
    as tests/test_db_init_ordering_guard.py for init_db/load_everything)
"""

from __future__ import annotations

import ast
import os
import pathlib

import sys as _sys
import os as _os

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401

import pytest

from _secrets_env import apply_secrets_to_environ

MAIN_PY = pathlib.Path(__file__).resolve().parent.parent / "app" / "main.py"


def test_scalars_are_mirrored_and_the_environment_outranks_the_file(monkeypatch):
    """An int TOML value arrives as its string form; a variable the machine
    already set is never taken over by the file — local exports and platform
    env stay authoritative. Keys the bridge did NOT write come back excluded,
    so the return value means "what this deployment actually configured"."""
    monkeypatch.setenv("BATLAB_BRIDGE_PRESET", "from-machine")
    monkeypatch.delenv("BATLAB_BRIDGE_STUDIES", raising=False)
    monkeypatch.delenv("BATLAB_BRIDGE_WORKERS", raising=False)

    applied = apply_secrets_to_environ({
        "BATLAB_BRIDGE_STUDIES": "off",
        "BATLAB_BRIDGE_WORKERS": 2,
        "BATLAB_BRIDGE_PRESET": "from-file",
    })

    assert os.environ["BATLAB_BRIDGE_STUDIES"] == "off"
    assert os.environ["BATLAB_BRIDGE_WORKERS"] == "2"
    assert os.environ["BATLAB_BRIDGE_PRESET"] == "from-machine"
    assert applied == ["BATLAB_BRIDGE_STUDIES", "BATLAB_BRIDGE_WORKERS"]


def test_bools_become_lowercase_and_tables_are_left_to_st_secrets(monkeypatch):
    """A TOML table is structured config for `st.secrets` readers — dumped
    into the environment it would be garbage. `=` in a key would corrupt an
    environment entry's own parsing, so it is dropped too."""
    monkeypatch.delenv("BATLAB_BRIDGE_FLAG", raising=False)

    applied = apply_secrets_to_environ({
        "BATLAB_BRIDGE_FLAG": True,
        "db": {"host": "localhost"},
        "NOT=AN_ENV_NAME": "skipped",
    })

    assert os.environ["BATLAB_BRIDGE_FLAG"] == "true"
    assert "db" not in os.environ
    assert "NOT=AN_ENV_NAME" not in os.environ
    assert applied == ["BATLAB_BRIDGE_FLAG"]


def test_a_missing_secrets_file_is_the_normal_case_not_an_error(monkeypatch):
    """Every local run has no secrets file: Streamlit raises
    StreamlitSecretNotFoundError, a FileNotFoundError subclass. Quiet no-op."""
    import _secrets_env

    def _missing():
        raise FileNotFoundError("No secrets file found")

    monkeypatch.setattr(_secrets_env, "_load_secrets", _missing)
    assert apply_secrets_to_environ() == []


def test_a_malformed_secrets_file_still_raises(monkeypatch):
    """A typo'd deployment config must be loud at boot. TOML decode failures
    are ValueErrors and are deliberately NOT swallowed — an app silently
    running without its knobs is the exact bug this bridge exists to fix."""
    import _secrets_env

    def _broken():
        raise ValueError("Malformed TOML")

    monkeypatch.setattr(_secrets_env, "_load_secrets", _broken)
    with pytest.raises(ValueError):
        apply_secrets_to_environ()


def test_main_applies_the_bridge_before_importing_data():
    """Ordering is load-bearing: `_data` and everything it pulls in may read
    a knob at import time, so the environment must already be populated when
    the first reader is imported. AST, not text — a mention in a comment must
    not satisfy the guard."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))

    bridge_idx = None
    data_idx = None
    for i, stmt in enumerate(tree.body):  # module top level only
        if bridge_idx is None and (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "apply_secrets_to_environ"
        ):
            bridge_idx = i
        if data_idx is None and isinstance(stmt, ast.ImportFrom) and stmt.module == "_data":
            data_idx = i

    assert bridge_idx is not None, (
        "app/main.py must call apply_secrets_to_environ() at module top level: "
        "Streamlit Cloud can only be configured through .streamlit/secrets.toml "
        "while every deployment knob is read from os.environ "
        "(app/_secrets_env.py)"
    )
    assert data_idx is not None, (
        "main.py no longer does `from _data import ...` — update this guard"
    )
    assert bridge_idx < data_idx, (
        "apply_secrets_to_environ() must come before `from _data import ...`: "
        "import-time readers of BATLAB_* / SETTINGS_* must already see the "
        "deployment's secrets"
    )
