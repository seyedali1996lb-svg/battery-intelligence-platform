"""Tests for the production-readiness batch: secrets store + JWT rotation,
per-org rate limiting, and the OIDC SSO adapter."""

import json
import time

import pytest

import secrets_store as ss
import rate_limit as rl


# ── secrets_store ────────────────────────────────────────────────────────────

def test_file_secrets_store_missing_file_returns_none(tmp_path):
    store = ss.FileSecretsStore(tmp_path / "nope.json")
    assert store.get("ANYTHING") is None


def test_file_secrets_store_reads_json(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"JWT_SECRET": "s3cret"}), encoding="utf-8")
    store = ss.FileSecretsStore(path)
    assert store.get("JWT_SECRET") == "s3cret"
    assert store.get("MISSING") is None


def test_resolve_jwt_secrets_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "current-key")
    monkeypatch.setenv("JWT_PREVIOUS_SECRETS", "old-1,old-2")
    resolved = ss.resolve_jwt_secrets()
    assert resolved["current"] == "current-key"
    assert resolved["previous"] == ["old-1", "old-2"]


def test_resolve_jwt_secrets_requires_current(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        ss.resolve_jwt_secrets()


def test_sign_verify_roundtrip_and_rotation():
    resolved = {"current": "key-v2", "previous": ["key-v1"]}
    # A token signed under the OLD key still verifies after rotation…
    old_token = ss.sign_jwt({"sub": "u1", "org_id": 1}, "key-v1", kid="prev-1")
    payload = ss.verify_jwt(old_token, resolved["current"], resolved["previous"])
    assert payload["sub"] == "u1"
    # …and the current key verifies its own tokens.
    new_token = ss.sign_jwt({"sub": "u1", "org_id": 1}, "key-v1" if False else "key-v2")
    assert ss.verify_jwt(new_token, resolved["current"], resolved["previous"])["org_id"] == 1
    # A token signed with an unknown key is rejected.
    forged = ss.sign_jwt({"sub": "x"}, "attacker-key")
    with pytest.raises(Exception):
        ss.verify_jwt(forged, resolved["current"], resolved["previous"])


def test_rotate_jwt_secret_promotes_current(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "old-current")
    monkeypatch.setenv("JWT_PREVIOUS_SECRETS", "ancient")
    written = {}
    result = ss.rotate_jwt_secret("brand-new", writer=lambda k, v: written.update({k: v}))
    assert result["current"] == "brand-new"
    assert result["previous"] == ["old-current", "ancient"]
    assert written["JWT_PREVIOUS_SECRETS"] == "old-current,ancient"
    assert written["JWT_SECRET"] == "brand-new"


# ── rate_limit ───────────────────────────────────────────────────────────────

def test_rate_limiter_allows_under_budget_and_blocks_over():
    limiter = rl.TokenBucketLimiter(3)
    assert limiter.check("org1:/cells") == (True, 0.0)
    assert limiter.check("org1:/cells") == (True, 0.0)
    assert limiter.check("org1:/cells") == (True, 0.0)
    allowed, retry_after = limiter.check("org1:/cells")
    assert allowed is False
    assert retry_after > 0


def test_rate_limiter_is_per_key():
    limiter = rl.TokenBucketLimiter(2)
    limiter.check("org1:/cells")
    limiter.check("org1:/cells")
    # A different org/endpoint is unaffected
    assert limiter.check("org2:/cells")[0] is True
    assert limiter.check("org1:/fleet")[0] is True


def test_rate_limiter_zero_disables():
    limiter = rl.TokenBucketLimiter(0)
    for _ in range(100):
        assert limiter.check("any")[0] is True


def test_shared_limiter_env_default():
    assert rl.current_rate_limit() == 0  # disabled by default (CI-safe)


# ── SSO ──────────────────────────────────────────────────────────────────────

def test_sso_not_configured_by_default(monkeypatch):
    for k in ("SSO_OIDC_ISSUER", "SSO_CLIENT_ID", "SSO_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    from sso import sso_configured, configured_provider
    assert sso_configured() is False
    assert configured_provider() is None


def test_sso_configured_when_all_env_set(monkeypatch):
    monkeypatch.setenv("SSO_OIDC_ISSUER", "https://idp.example.com")
    monkeypatch.setenv("SSO_CLIENT_ID", "client-1")
    monkeypatch.setenv("SSO_CLIENT_SECRET", "hunter2")
    from sso import sso_configured
    assert sso_configured() is True


def test_oidc_discovery_and_exchange(monkeypatch):
    """The OIDC flow: discovery -> auth URL -> code exchange -> verified email."""
    from sso import OIDCProvider

    class FakeResp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")
            return None

        def json(self):
            return self._payload

    calls = []

    class FakeSession:
        def get(self, url, **kw):
            calls.append(("get", url))
            if "openid-configuration" in url:
                return FakeResp({
                    "authorization_endpoint": "https://idp.example.com/authorize",
                    "token_endpoint": "https://idp.example.com/token",
                    "userinfo_endpoint": "https://idp.example.com/userinfo",
                })
            return FakeResp({"email": "engineer@example.com", "email_verified": True,
                             "name": "Eng One", "sub": "sub-123"})

        def post(self, url, **kw):
            calls.append(("post", url))
            return FakeResp({"access_token": "at-1", "id_token": "eyJ.eyJ9.sig"})

    provider = OIDCProvider("https://idp.example.com", "client-1", "hunter2", session=FakeSession())
    auth_url = provider.auth_url("state-1", "https://app/cb")
    assert auth_url.startswith("https://idp.example.com/authorize")
    assert "state=state-1" in auth_url
    assert "nonce=" in auth_url

    user = provider.exchange_code("code-1", "https://app/cb")
    assert user["email"] == "engineer@example.com"
    assert user["email_verified"] is True
    assert user["idp"] == "https://idp.example.com"
    assert any(c[0] == "post" for c in calls)  # token endpoint hit


def test_oidc_rejects_missing_email(monkeypatch):
    from sso import OIDCProvider

    discovery = {
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "userinfo_endpoint": "https://idp.example.com/userinfo",
    }

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, **kw):
            if "openid-configuration" in url:
                return FakeResp(discovery)
            return FakeResp({"name": "No Email Here"})

        def post(self, url, **kw):
            return FakeResp({"access_token": "at-1"})

    provider = OIDCProvider("https://idp.example.com", "c", "s", session=FakeSession())
    with pytest.raises(ValueError, match="no verified email"):
        provider.exchange_code("code-1", "https://app/cb")
