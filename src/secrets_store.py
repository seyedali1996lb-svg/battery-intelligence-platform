"""
Secrets-store abstraction + JWT signing-key rotation (Production Readiness
Roadmap: "rotate the demo-grade JWT secret via a real secrets manager").

Three pieces:

1. ``SecretsStore`` protocol — one ``get(name) -> str | None`` shape so the
   rest of the platform never cares where a secret came from. Built-ins:
   - ``EnvSecretsStore`` (os.environ) — the zero-setup default.
   - ``FileSecretsStore`` (JSON file, default ``data/secrets.json``,
     gitignored) — for local/self-hosted deployments that keep secrets out
     of the environment.
   - ``AwsSecretsManagerStore`` / ``GcpSecretManagerStore`` — built against
     each cloud's documented SDK shape (boto3 get_secret_value /
     google-cloud-secret-manager access_secret_version), honestly NOT
     validated against a live cloud account (none exists in this repo, same
     discipline as the EIA/ENTSO-E market adapters). They raise a clear
     error until the SDK is installed and a client is supplied.

2. ``resolve_jwt_secrets()`` — reads the current signing secret plus any
   previous (retired) secrets from the store, so a deployment can rotate
   the key without invalidating tokens issued under the old one.

3. Pure PyJWT helpers ``sign_jwt`` / ``verify_jwt`` implementing the actual
   rotation mechanics: every token carries a ``kid`` naming which key
   signed it; verification tries the current key first, then each previous
   key in order, and accepts legacy tokens that carry no ``kid`` (issued
   before rotation existed) against the current key.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Protocol, runtime_checkable

DEFAULT_SECRETS_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "secrets.json"

# Env keys this module reads for JWT rotation (documented, single source of
# truth for operators: JWT_SECRET is the current signing key, and any number
# of comma-separated previous keys in JWT_PREVIOUS_SECRETS stay accepted for
# verification until their tokens expire).
JWT_SECRET_ENV = "JWT_SECRET"
JWT_PREVIOUS_SECRETS_ENV = "JWT_PREVIOUS_SECRETS"


@runtime_checkable
class SecretsStore(Protocol):
    """Any source that resolves a named secret to a string."""

    def get(self, name: str) -> "str | None":
        """Return the secret's value, or None when it is not configured."""
        ...


class EnvSecretsStore:
    """Reads secrets from the process environment."""

    name = "env"

    def get(self, name: str) -> "str | None":
        return os.environ.get(name)


class FileSecretsStore:
    """Reads secrets from a JSON file like ``{"MY_KEY": "value"}``.

    A missing file or key returns None (never raises) — the caller decides
    what "not configured" means for its own secret.
    """

    name = "file"

    def __init__(self, path: "str | os.PathLike | None" = None):
        self.path = pathlib.Path(path) if path is not None else DEFAULT_SECRETS_FILE
        self._cache: "dict | None" = None

    def _load(self) -> dict:
        if self._cache is None:
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        return self._cache

    def get(self, name: str) -> "str | None":
        return self._load().get(name)


class AwsSecretsManagerStore:
    """AWS Secrets Manager, built against boto3's documented get_secret_value
    shape — NOT validated against a live account (none exists in this repo).
    Supply a boto3 client (or session) so no AWS SDK import happens at module
    import time."""

    name = "aws-secrets-manager"

    def __init__(self, client, prefix: str = ""):
        self._client = client
        self._prefix = prefix

    def get(self, name: str) -> "str | None":
        try:
            resp = self._client.get_secret_value(SecretId=f"{self._prefix}{name}")
        except Exception as exc:  # noqa: BLE001 — secret missing is "not configured"
            return None if _is_missing(exc) else _reraise(exc)
        return resp.get("SecretString")


class GcpSecretManagerStore:
    """Google Cloud Secret Manager, built against the documented
    google-cloud-secret-manager access_secret_version shape — NOT validated
    against a live project. Supply a SecretManagerServiceClient."""

    name = "gcp-secret-manager"

    def __init__(self, client, project_id: str):
        self._client = client
        self._project = project_id

    def get(self, name: str) -> "str | None":
        try:
            resp = self._client.access_secret_version(
                request={"name": f"projects/{self._project}/secrets/{name}/versions/latest"}
            )
        except Exception as exc:  # noqa: BLE001
            return None if _is_missing(exc) else _reraise(exc)
        return resp.payload.data.decode("utf-8")


def _is_missing(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("not found", "notfound", "resource not found", "404"))


def _reraise(exc: Exception):
    raise exc


# ── JWT rotation ─────────────────────────────────────────────────────────────

def resolve_jwt_secrets(store: "SecretsStore | None" = None) -> dict:
    """Resolve (current, previous) signing secrets from a store.

    Returns ``{"current": str, "previous": [str]}``. Raises RuntimeError when
    no current secret is configured — a deployment must never silently fall
    back to a shared/demo key (same stance as src/api.py's existing refusal
    to start without JWT_SECRET).
    """
    store = store or EnvSecretsStore()
    current = store.get(JWT_SECRET_ENV)
    if not current:
        raise RuntimeError(
            f"{JWT_SECRET_ENV} is not set. Refusing to operate with no signing "
            "secret — set it in your secrets store before starting."
        )
    raw_prev = store.get(JWT_PREVIOUS_SECRETS_ENV) or ""
    previous = [s.strip() for s in raw_prev.split(",") if s.strip()]
    return {"current": current, "previous": previous}


def sign_jwt(payload: dict, secret: str, algorithm: str = "HS256", kid: str = "current") -> str:
    """Sign a payload with an explicit key id (default 'current')."""
    import jwt as _jwt
    return _jwt.encode(payload, secret, algorithm=algorithm, headers={"kid": kid})


def verify_jwt(token: str, current: str, previous: "list[str]", algorithm: str = "HS256") -> dict:
    """Verify a token against the current key, then each previous key.

    Returns the decoded payload. Raises the underlying PyJWT error
    (ExpiredSignatureError / InvalidTokenError) on failure so callers keep
    their existing error mapping. Legacy tokens without a ``kid`` are
    accepted against the current key (they predate rotation).
    """
    import jwt as _jwt

    candidates = [current] + list(previous)
    last_error: "Exception | None" = None
    for secret in candidates:
        try:
            return _jwt.decode(token, secret, algorithms=[algorithm])
        except _jwt.ExpiredSignatureError:
            raise  # expired is expired regardless of which key signed it
        except _jwt.InvalidTokenError as exc:
            last_error = exc
            continue
    raise last_error  # type: ignore[misc]  # never reached when candidates non-empty


def rotate_jwt_secret(
    new_secret: str,
    store: "SecretsStore | None" = None,
    writer=None,
) -> dict:
    """Perform a documented rotation: promote the current secret into
    previous, install the new one as current. `writer` (a callable taking
    the store's key name and value) is how the new secret actually reaches
    the store (env/file/cloud); the returned dict shows what to set."""
    store = store or EnvSecretsStore()
    current = store.get(JWT_SECRET_ENV) or ""
    prev = [s for s in (store.get(JWT_PREVIOUS_SECRETS_ENV) or "").split(",") if s.strip()]
    if current:
        prev = [current] + prev
    prev = prev[:5]  # bound the verification chain
    new_prev = ",".join(prev)
    if writer is not None:
        writer(JWT_PREVIOUS_SECRETS_ENV, new_prev)
        writer(JWT_SECRET_ENV, new_secret)
    return {"current": new_secret, "previous": prev}


def _now() -> int:
    return int(time.time())
