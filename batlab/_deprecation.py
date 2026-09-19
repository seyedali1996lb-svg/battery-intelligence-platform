"""Deprecation mechanics for batlab's public API.

The policy is written down in ``docs/api_stability.md``; this module is its
mechanism, so a deprecation is a *tested* code path rather than a promise in
prose. A deprecated public name is expected to:

1. keep working, and return exactly what it returned before;
2. emit a :class:`DeprecationWarning` that names what to use instead, since when
   it deprecated, and the version that removes it;
3. be listed in ``CHANGELOG.md`` with the same three facts;
4. be removable in one commit on the announced version — no behavioural
   back-compat shims hiding behind a warning past that point.

``tests/test_public_api.py`` asserts the mechanism does all of (2), so the
policy cannot rot into documentation.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

__all__ = ["warn_deprecated", "deprecated"]

_F = TypeVar("_F", bound=Callable[..., Any])


def _message(what: str, *, since: str, removed_in: str, alternative: str | None) -> str:
    parts = [f"{what} is deprecated since batlab {since} and will be removed in batlab {removed_in}."]
    if alternative:
        parts.append(f"Use {alternative} instead.")
    return " ".join(parts)


def warn_deprecated(
    what: str,
    *,
    since: str,
    removed_in: str,
    alternative: str | None = None,
) -> None:
    """Emit the standard deprecation warning.

    ``stacklevel=3`` so the warning points at the *caller* of the deprecated
    name rather than at this helper or at the deprecated function's own body —
    a warning that points into batlab's internals is a warning nobody can act
    on.
    """
    warnings.warn(
        _message(what, since=since, removed_in=removed_in, alternative=alternative),
        DeprecationWarning,
        stacklevel=3,
    )


def deprecated(
    what: str,
    *,
    since: str,
    removed_in: str,
    alternative: str | None = None,
) -> Callable[[_F], _F]:
    """Decorator form of :func:`warn_deprecated` (keeps the wrapped docstring).

    The return annotation is preserved with ``_F`` so a type checker still sees
    the wrapped signature — a deprecated alias is still *typed* like the thing
    it forwards to.
    """

    def _decorate(func: _F) -> _F:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(what, since=since, removed_in=removed_in, alternative=alternative)
            return func(*args, **kwargs)

        return _wrapper  # type: ignore[return-value]

    return _decorate
