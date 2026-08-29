"""
OAuth2/OIDC enterprise SSO (Production Readiness Roadmap: "OAuth2 via
Okta/LDAP for enterprise SSO").

The platform's demo auth is real (bcrypt users, org scoping) but username/
password only. This module is the documented path to enterprise SSO: an
OIDC authorization-code flow against any provider that publishes an
OpenID Connect discovery document (Okta, Entra ID, Keycloak, Auth0, ...).

Honesty boundary — same as the market-data adapters: this is built against
the OIDC Discovery spec (RFC 8414) and the token/UserInfo endpoints'
documented shapes, and there is NO live IdP tenant in this repo to validate
it against. The provider is "configured" only when the environment says so;
until then every function reports not-configured rather than pretending.

Deployment wiring (documented here, applied by the login layer):
  SSO_OIDC_ISSUER        — e.g. https://<tenant>.okta.com/oauth2/default
  SSO_CLIENT_ID          — the app's OIDC client id
  SSO_CLIENT_SECRET      — from the secrets store, never the repo
  SSO_REDIRECT_URI       — e.g. https://app.example.com/auth/callback
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Protocol, runtime_checkable

import requests

ENV_ISSUER = "SSO_OIDC_ISSUER"
ENV_CLIENT_ID = "SSO_CLIENT_ID"
ENV_CLIENT_SECRET = "SSO_CLIENT_SECRET"
ENV_REDIRECT_URI = "SSO_REDIRECT_URI"


def sso_configured() -> bool:
    """True only when every required OIDC setting is present — the honest
    "not configured" gate the rest of the platform can check."""
    return all(os.environ.get(k) for k in (ENV_ISSUER, ENV_CLIENT_ID, ENV_CLIENT_SECRET))


@runtime_checkable
class SSOProvider(Protocol):
    name: str

    def auth_url(self, state: str, redirect_uri: str) -> str: ...
    def exchange_code(self, code: str, redirect_uri: str) -> "dict[str, Any]": ...


class OIDCProvider:
    """OIDC authorization-code provider.

    Discovers ``/.well-known/openid-configuration`` at init (cached), then:
    - ``auth_url()`` builds the authorization endpoint URL with state +
      nonce + openid email scope;
    - ``exchange_code()`` posts the code to the token endpoint and returns
      the verified UserInfo (from the ``userinfo_endpoint`` with the access
      token, or the ID-token claims when UserInfo is unavailable).

    Built against the spec and honestly untested against a live tenant.
    """

    name = "oidc"

    def __init__(
        self,
        issuer: str,
        client_id: str,
        client_secret: str,
        timeout: float = 10.0,
        session: "requests.Session | None" = None,
    ):
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._session = session or requests.Session()
        self._discovery: "dict[str, Any] | None" = None

    # ── discovery ────────────────────────────────────────────────────────────

    def _config(self) -> dict:
        if self._discovery is None:
            url = f"{self.issuer}/.well-known/openid-configuration"
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            self._discovery = resp.json()
        return self._discovery  # pyright: ignore[reportReturnType]

    # ── flow ─────────────────────────────────────────────────────────────────

    def auth_url(self, state: str, redirect_uri: str, nonce: "str | None" = None) -> str:
        """Build the authorization-endpoint URL. A caller-supplied nonce is
        used when given (the login flow generates it so it can be stored and
        verified at the callback); otherwise one is generated here."""
        cfg = self._config()
        nonce = nonce or secrets.token_urlsafe(16)
        from urllib.parse import urlencode
        params = urlencode({
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
        })
        return f"{cfg['authorization_endpoint']}?{params}"

    def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        expected_nonce: "str | None" = None,
    ) -> "dict[str, Any]":
        """Exchange an authorization code for a verified identity.

        ``expected_nonce``, when supplied, must match the id_token's ``nonce``
        claim (the OIDC replay-protection binding this login flow sends in
        auth_url); a mismatch or missing nonce raises ValueError. The primary
        identity source is the UserInfo endpoint; the ID-token claims are the
        fallback only when UserInfo is unavailable."""
        cfg = self._config()
        resp = self._session.post(
            cfg["token_endpoint"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        tokens = resp.json()
        if "access_token" not in tokens:
            raise ValueError("Token response contained no access_token")

        # UserInfo is the standard, verified identity source.
        try:
            ui = self._session.get(
                cfg["userinfo_endpoint"],
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                timeout=self.timeout,
            )
            ui.raise_for_status()
            userinfo = ui.json()
        except Exception:  # noqa: BLE001 — fall back to ID-token claims
            import jwt as _jwt
            userinfo = _jwt.decode(
                tokens["id_token"], options={"verify_signature": False}
            )

        if expected_nonce is not None:
            # openid scope was requested, so an id_token is mandatory when
            # nonce verification is on — fail closed rather than skip.
            id_token = tokens.get("id_token")
            if not id_token:
                raise ValueError("No id_token returned — cannot verify nonce")
            import base64
            import json as _json
            payload_b64 = id_token.split(".")[1]
            pad = "=" * (-len(payload_b64) % 4)
            try:
                claims = _json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
            except Exception:  # noqa: BLE001
                raise ValueError("id_token payload is not valid JSON") from None
            if claims.get("nonce") != expected_nonce:
                raise ValueError("id_token nonce does not match the login request")

        email = userinfo.get("email")
        if not email:
            raise ValueError("IdP returned no verified email for this user")
        return {
            "email": email,
            "name": userinfo.get("name") or userinfo.get("preferred_username"),
            "sub": userinfo.get("sub"),
            "idp": self.issuer,
            "email_verified": bool(userinfo.get("email_verified", False)),
        }


# The login page renders on every interaction, and rendering the SSO button
# builds an authorization URL, which needs the discovery document. Without a
# cache every keystroke on the login page would hit the IdP's discovery
# endpoint; with it, discovery happens once per process. Keyed by the full
# (issuer, client_id, client_secret) triple so a settings change can't serve
# a stale provider.
_configured_provider_cache: "dict[tuple[str, str, str], OIDCProvider]" = {}


def configured_provider(session: "requests.Session | None" = None) -> "OIDCProvider | None":
    """Return a provider when SSO is configured, else None (the honest
    not-configured state callers should surface rather than guess). Cached
    per configuration triple so the login page's per-rerun URL building
    doesn't re-fetch OIDC discovery on every render."""
    if not sso_configured():
        return None
    key = (
        os.environ[ENV_ISSUER],
        os.environ[ENV_CLIENT_ID],
        os.environ[ENV_CLIENT_SECRET],
    )
    if key not in _configured_provider_cache:
        _configured_provider_cache[key] = OIDCProvider(
            issuer=key[0], client_id=key[1], client_secret=key[2], session=session,
        )
    return _configured_provider_cache[key]


def default_redirect_uri() -> str:
    # The IdP redirects back to the app itself; Streamlit reads the code and
    # state from its query params on the next load.
    return os.environ.get(ENV_REDIRECT_URI, "http://localhost:8501/")


# ── login-flow helpers (used by app/_pages/login.py) ────────────────────────

class SSOLoginError(ValueError):
    """A login-flow failure with a human-readable message (state mismatch,
    exchange failure, unverified email, ...) — the login page surfaces the
    message instead of a raw exception."""


def begin_sso_login(
    redirect_uri: "str | None" = None,
    provider: "OIDCProvider | None" = None,
) -> "tuple[str, str, str]":
    """Start an SSO login: generate the anti-CSRF state and the replay
    nonce, return (auth_url, state, nonce). The caller MUST store state and
    nonce (e.g. in the session) and pass them to :func:`complete_sso_login`
    when the IdP redirects back.

    Raises SSOLoginError when SSO is not configured (the button should not
    be shown in that case — this is the fail-closed guard)."""
    provider = provider or configured_provider()
    if provider is None:
        raise SSOLoginError("Enterprise SSO is not configured on this deployment.")
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(16)
    auth_url = provider.auth_url(state, redirect_uri or default_redirect_uri(), nonce=nonce)
    return auth_url, state, nonce


def complete_sso_login(
    code: str,
    state: str,
    expected_state: str,
    redirect_uri: "str | None" = None,
    nonce: "str | None" = None,
    provider: "OIDCProvider | None" = None,
) -> "dict[str, Any]":
    """Complete an SSO login at the callback: verify the returned state
    against the one this flow generated (CSRF protection), exchange the
    code, and return the verified userinfo dict (email, name, sub, idp,
    email_verified). Raises SSOLoginError on state mismatch or any flow
    failure — never returns an unauthenticated identity."""
    if not state or state != expected_state:
        raise SSOLoginError(
            "SSO state mismatch — the login request may have been tampered "
            "with or expired. Please try signing in again."
        )
    provider = provider or configured_provider()
    if provider is None:
        raise SSOLoginError("Enterprise SSO is not configured on this deployment.")
    try:
        userinfo = provider.exchange_code(
            code, redirect_uri or default_redirect_uri(), expected_nonce=nonce
        )
    except Exception as exc:  # noqa: BLE001 — wrap into the flow error type
        raise SSOLoginError(f"SSO login failed: {exc}") from exc
    if not userinfo.get("email_verified"):
        raise SSOLoginError("Your identity provider did not verify your email address.")
    if not userinfo.get("email"):
        raise SSOLoginError("Your identity provider returned no email address.")
    return userinfo
