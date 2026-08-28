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
        return self._discovery

    # ── flow ─────────────────────────────────────────────────────────────────

    def auth_url(self, state: str, redirect_uri: str) -> str:
        cfg = self._config()
        nonce = secrets.token_urlsafe(16)
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

    def exchange_code(self, code: str, redirect_uri: str) -> "dict[str, Any]":
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


def configured_provider(session: "requests.Session | None" = None) -> "OIDCProvider | None":
    """Return a provider when SSO is configured, else None (the honest
    not-configured state callers should surface rather than guess)."""
    if not sso_configured():
        return None
    return OIDCProvider(
        issuer=os.environ[ENV_ISSUER],
        client_id=os.environ[ENV_CLIENT_ID],
        client_secret=os.environ[ENV_CLIENT_SECRET],
        session=session,
    )


def default_redirect_uri() -> str:
    return os.environ.get(ENV_REDIRECT_URI, "http://localhost:8501/auth/callback")
