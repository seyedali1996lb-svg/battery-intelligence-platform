"""
Per-org rate limiting for the REST layer (Production Readiness Roadmap:
"per-org rate limiting" under multi-tenancy).

A fixed-window token bucket keyed by ``(org_id, endpoint)`` so one
organization's burst (or abuse) can never starve another org's API
consumers — the natural complement to the org scoping already enforced in
src/db.py. In-process and per-process by design: this is a coarse abuse
guard for the single-process demo/API deployment, not a distributed
rate limiter (that belongs behind a real gateway/Redis once the platform
is multi-instance, and is documented as such).

Disabled by default (``RATE_LIMIT_PER_ORG_PER_MINUTE=0``) so existing
tests and the demo deployment behave exactly as before. Enable with:

    RATE_LIMIT_PER_ORG_PER_MINUTE=120   # 120 requests / org / minute / endpoint
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

ENV_RATE = "RATE_LIMIT_PER_ORG_PER_MINUTE"
DEFAULT_RATE_PER_MINUTE = int(os.environ.get(ENV_RATE, "0"))  # 0 = disabled


class TokenBucketLimiter:
    """Fixed-window token bucket. ``check(key)`` consumes one token and
    returns ``(allowed, retry_after_s)``. Thread-safe (FastAPI serves
    concurrent requests)."""

    def __init__(self, rate_per_minute: int):
        if rate_per_minute < 0:
            raise ValueError("rate_per_minute must be >= 0 (0 disables limiting)")
        self.rate_per_minute = rate_per_minute
        self._tokens: dict[str, list[float]] = {}  # key -> [timestamps in window]
        self._lock = threading.Lock()

    def check(self, key: str) -> "tuple[bool, float]":
        """Consume one token for `key`.

        Returns (True, 0.0) when allowed; (False, retry_after_s) when the
        bucket is empty. A disabled limiter (rate 0) always allows.
        """
        if self.rate_per_minute <= 0:
            return True, 0.0
        now = time.monotonic()
        window = 60.0
        with self._lock:
            stamps = self._tokens.setdefault(key, [])
            cutoff = now - window
            stamps[:] = [s for s in stamps if s > cutoff]
            if len(stamps) >= self.rate_per_minute:
                retry_after = max(0.0, window - (now - stamps[0]))
                return False, retry_after
            stamps.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()


_limiter = TokenBucketLimiter(DEFAULT_RATE_PER_MINUTE)


def check_rate_limit(org_id: "int | str | None", path: str) -> "tuple[bool, float]":
    """Rate-limit one request for an org on one endpoint. Returns
    (allowed, retry_after_s)."""
    if org_id is None:
        return True, 0.0
    return _limiter.check(f"{org_id}:{path}")


def set_rate_limit(rate_per_minute: int) -> None:
    """Test/ops hook — reconfigure the shared limiter at runtime."""
    global _limiter
    _limiter = TokenBucketLimiter(rate_per_minute)


def current_rate_limit() -> int:
    return _limiter.rate_per_minute
