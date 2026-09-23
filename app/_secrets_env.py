"""Mirror Streamlit secrets into the process environment.

Why this exists: every deployment knob this app reads comes from the
environment — ``BATLAB_BOOT_STUDIES``, ``BATLAB_BOOT_LAYERS``,
``BATLAB_FOLD_WORKERS``, ``SETTINGS_ENCRYPTION_KEY``, ``ANTHROPIC_API_KEY`` —
but Streamlit Community Cloud has no environment-variable setting: its only
configuration surface is ``.streamlit/secrets.toml``. Without a bridge those
knobs are unreachable exactly where they matter most — a Cloud instance that
has been CPU-throttled and needs ``BATLAB_BOOT_STUDIES=off`` (README, "What a
cold deploy costs"). The app's own Settings page already told deployers to
"set it via Streamlit Cloud secrets"; this is the code that makes that
sentence true.

The rules, all deliberate:

- **The machine outranks the file.** ``os.environ.setdefault`` never
  overwrites a variable that is already set, so real environment variables
  (local exports, container config, platform env) always win over the file.
- **Top-level scalars only.** A TOML table is structured configuration for
  ``st.secrets`` readers; stringified into the environment it would be
  garbage. Bools mirror as the conventional ``true`` / ``false``.
- **No secrets file is normal; a broken one is an error.** Every local run
  has no ``secrets.toml`` (Streamlit raises ``StreamlitSecretNotFoundError``,
  a ``FileNotFoundError`` subclass) — that is a no-op, not a crash. A
  malformed file still raises at boot: a typo'd deployment config must be
  loud, not silently ignored.

``app/main.py`` calls :func:`apply_secrets_to_environ` at module top level —
before ``from _data import ...`` — so even an import-time reader of a knob
sees the deployment's secrets (that ordering is AST-pinned by
tests/test_secrets_env.py). Scope note: the FastAPI service and the scripts
read the same environment variables without this bridge; self-hosters
configure a real environment, which outranks the file anyway.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

__all__ = ["apply_secrets_to_environ"]

#: Only these TOML scalar types become environment variables — a table would
#: stringify to nonsense (see the module docstring).
_SCALARS = (str, int, float, bool)


def _load_secrets() -> Mapping[str, Any]:
    """The deployment's Streamlit secrets, read lazily.

    Streamlit is imported here rather than at module level so this file stays
    importable — and its behaviour testable with an injected mapping — without
    touching ``st.secrets`` at all.
    """
    import streamlit as st

    return dict(st.secrets)


def apply_secrets_to_environ(secrets: Mapping[str, Any] | None = None) -> list[str]:
    """Mirror ``secrets`` (default: the deployment's ``st.secrets``) into
    ``os.environ`` without overwriting anything already set.

    Returns the keys actually written, so callers and tests can say what the
    deployment configured. Precedence and scope rules live in the module
    docstring; they are the contract, not incidental behaviour.
    """
    if secrets is None:
        try:
            secrets = _load_secrets()
        except FileNotFoundError:
            return []  # no secrets file anywhere: every local run, by design
    applied: list[str] = []
    for key, value in secrets.items():
        if not isinstance(key, str) or not isinstance(value, _SCALARS) or "=" in key:
            continue  # tables and odd keys are st.secrets' business, not env's
        if key in os.environ:
            continue  # the machine outranks the file
        if isinstance(value, bool):
            value = "true" if value else "false"
        os.environ[key] = str(value)
        applied.append(key)
    return applied
