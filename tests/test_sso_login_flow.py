"""Tests for the OIDC SSO login flow (src/sso.py begin/complete helpers +
src/db.py provision_or_link_sso_user) — the pieces the login page's SSO
button and callback route are built on.

The OIDCProvider itself is covered in tests/test_prod_readiness.py (spec
conformance, discovery, code exchange, email requirements). These tests
cover the login-flow layer: state/nonce handling, fail-closed guards, and
the provisioning/linking semantics against the existing User model.
"""

import base64
import json

import pytest
import db as db_module
import sso

_TEST_ENCRYPTION_KEY = "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY="  # test-only


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point src/db.py at a throwaway SQLite file for the duration of one
    test, same pattern as tests/test_db.py's db fixture (including the
    real SETTINGS_ENCRYPTION_KEY and the _fernet cache reset)."""
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(
            f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
        ),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", _TEST_ENCRYPTION_KEY)
    monkeypatch.setattr(db_module, "_fernet", None)
    db_module.init_db()
    return db_module


# ── begin / complete flow helpers ───────────────────────────────────────────

class FakeProvider:
    """Minimal stand-in for OIDCProvider implementing the SSOProvider
    protocol surface begin/complete actually use."""

    name = "fake"

    def __init__(self, userinfo=None):
        self.userinfo = userinfo or {
            "email": "engineer@example.com", "name": "Eng One",
            "sub": "sub-1", "idp": "https://idp.example.com",
            "email_verified": True,
        }
        self.last_nonce = None
        self.last_expected_nonce = None

    def auth_url(self, state, redirect_uri, nonce=None):
        self.last_nonce = nonce
        return f"https://idp.example.com/authorize?state={state}&nonce={nonce or ''}"

    def exchange_code(self, code, redirect_uri, expected_nonce=None):
        self.last_expected_nonce = expected_nonce
        if expected_nonce is not None and expected_nonce != "good-nonce":
            raise ValueError("nonce mismatch")
        return dict(self.userinfo)


def test_begin_sso_login_returns_url_state_and_nonce():
    provider = FakeProvider()
    auth_url, state, nonce = sso.begin_sso_login(provider=provider)
    assert state in auth_url
    assert provider.last_nonce == nonce  # the nonce the URL carries is the one the caller must store
    assert len(state) >= 16
    assert state != nonce


def test_begin_sso_login_fails_closed_when_not_configured(monkeypatch):
    for k in ("SSO_OIDC_ISSUER", "SSO_CLIENT_ID", "SSO_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(sso.SSOLoginError, match="not configured"):
        sso.begin_sso_login()


def test_complete_sso_login_success_returns_userinfo():
    provider = FakeProvider()
    userinfo = sso.complete_sso_login(
        code="code-1", state="state-1", expected_state="state-1",
        nonce="good-nonce", provider=provider,
    )
    assert userinfo["email"] == "engineer@example.com"
    assert userinfo["email_verified"] is True
    assert provider.last_expected_nonce == "good-nonce"


def test_complete_sso_login_rejects_state_mismatch():
    provider = FakeProvider()
    with pytest.raises(sso.SSOLoginError, match="state mismatch"):
        sso.complete_sso_login(
            code="code-1", state="forged", expected_state="state-1", provider=provider
        )


def test_complete_sso_login_rejects_empty_state():
    provider = FakeProvider()
    with pytest.raises(sso.SSOLoginError, match="state mismatch"):
        sso.complete_sso_login(
            code="code-1", state="", expected_state="state-1", provider=provider
        )


def test_complete_sso_login_rejects_unverified_email():
    provider = FakeProvider(userinfo={
        "email": "unverified@example.com", "email_verified": False,
        "sub": "s", "idp": "fake",
    })
    with pytest.raises(sso.SSOLoginError, match="did not verify"):
        sso.complete_sso_login(
            code="code-1", state="state-1", expected_state="state-1", provider=provider
        )


def test_complete_sso_login_fails_closed_when_not_configured(monkeypatch):
    for k in ("SSO_OIDC_ISSUER", "SSO_CLIENT_ID", "SSO_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(sso.SSOLoginError, match="not configured"):
        sso.complete_sso_login(
            code="code-1", state="state-1", expected_state="state-1"
        )


def test_complete_sso_login_wraps_nonce_mismatch_as_flow_error():
    """Nonce verification happens inside the real OIDCProvider (decoding the
    id_token); a mismatch must surface as SSOLoginError, never a raw error."""
    import requests as _requests

    def _b64(payload: dict) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")

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
                return FakeResp({
                    "authorization_endpoint": "https://idp.example.com/authorize",
                    "token_endpoint": "https://idp.example.com/token",
                    "userinfo_endpoint": "https://idp.example.com/userinfo",
                })
            return FakeResp({"email": "e@example.com", "email_verified": True, "sub": "s"})

        def post(self, url, **kw):
            return FakeResp({
                "access_token": "at-1",
                "id_token": f"header.{_b64({'nonce': 'WRONG-NONCE'})}.sig",
            })

    provider = sso.OIDCProvider(
        "https://idp.example.com", "client-1", "hunter2", session=FakeSession()
    )
    with pytest.raises(sso.SSOLoginError, match="nonce does not match"):
        sso.complete_sso_login(
            code="code-1", state="state-1", expected_state="state-1",
            nonce="good-nonce", provider=provider,
        )


# ── account provisioning / linking (src/db.py) ──────────────────────────────

def test_sso_provision_creates_new_org_with_admin(db):
    user = db.provision_or_link_sso_user(
        "carol@acme.com", "Carol Nguyen", "https://acme.okta.com", "sub-carol"
    )
    assert user["role"] == "admin"
    assert user["username"] == "carol"
    assert user["email"] == "carol@acme.com"
    assert user["sso_provider"] == "https://acme.okta.com"
    assert user["sso_subject"] == "sub-carol"
    assert user["org_name"] == "Carol Nguyen"
    # A brand-new org, not the demo org.
    assert user["org_id"] != 1
    # And it round-trips through the normal lookup.
    again = db.get_user_by_username("carol")
    assert again["user_id"] == user["user_id"]


def test_sso_provision_links_existing_email_account(db):
    result = db.create_organization_with_admin("Acme Corp", "alice", "pw123456")
    org_id = result["org_id"]
    # Give alice a real email the way a post-signup profile update would.
    with db.Session() as s:
        u = s.query(db_module.User).filter_by(username="alice").one()
        u.email = "alice@acme.com"
        s.commit()

    user = db.provision_or_link_sso_user(
        "alice@acme.com", "Alice", "https://acme.okta.com", "sub-alice"
    )
    assert user["user_id"] == result["user_id"]  # same account, not a duplicate
    assert user["org_id"] == org_id
    assert user["sso_provider"] == "https://acme.okta.com"
    with db.Session() as s:
        users = s.query(db_module.User).filter_by(username="alice").all()
    assert len(users) == 1  # no duplicate user row created


def test_sso_provision_links_account_whose_username_is_the_email(db):
    """Accounts created before the email column existed may have their email
    as the username — the link must still find them."""
    result = db.create_organization_with_admin("Mailer Co", "bob@mailer.com", "pw123456")
    user = db.provision_or_link_sso_user(
        "bob@mailer.com", "Bob", "https://mailer.okta.com", "sub-bob"
    )
    assert user["user_id"] == result["user_id"]


def test_sso_provision_is_idempotent_for_linked_identity(db):
    first = db.provision_or_link_sso_user(
        "dave@acme.com", "Dave", "https://acme.okta.com", "sub-dave"
    )
    second = db.provision_or_link_sso_user(
        "dave@acme.com", "Dave", "https://acme.okta.com", "sub-dave"
    )
    assert second["user_id"] == first["user_id"]
    with db.Session() as s:
        total = s.query(db_module.User).filter_by(username="dave").count()
    assert total == 1


def test_sso_provision_dedupes_username_across_domains(db):
    first = db.provision_or_link_sso_user(
        "alice@one.com", "Alice", "https://one.okta.com", "sub-a1"
    )
    second = db.provision_or_link_sso_user(
        "alice@two.com", "Alice", "https://two.okta.com", "sub-a2"
    )
    assert first["username"] == "alice"
    assert second["username"] == "alice2"


def test_sso_user_cannot_password_login(db):
    user = db.provision_or_link_sso_user(
        "eve@acme.com", "Eve", "https://acme.okta.com", "sub-eve"
    )
    # The stored hash is the unusable sentinel — verify_password must fail
    # for any password, so the password path can never sign in an SSO user.
    assert user["password_hash"] == db._SSO_NO_PASSWORD
    assert db.verify_password("anything", user["password_hash"]) is False
    assert db.verify_password("", user["password_hash"]) is False


def test_find_user_by_sso_and_link_user_sso(db):
    result = db.create_organization_with_admin("Link Co", "frank", "pw123456")
    db.link_user_sso(result["user_id"], "https://link.okta.com", "sub-frank")
    found = db.find_user_by_sso("https://link.okta.com", "sub-frank")
    assert found is not None
    assert found["user_id"] == result["user_id"]
    assert found["email"] is None  # password-account email stays untouched
    assert db.find_user_by_sso("https://link.okta.com", "someone-else") is None
