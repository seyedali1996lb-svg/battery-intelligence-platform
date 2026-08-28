"""AppTest page-level checks for the SSO login-button wiring on the login
wall (app/_pages/login.py): the button appears only when the deployment has
an OIDC provider configured, its state/nonce are stashed for the callback,
and the page degrades gracefully if the SSO flow itself fails.

The full callback route (code exchange -> provision/link -> login) is
covered at the unit level in tests/test_sso_login_flow.py + the db/sso
modules; AppTest here cannot inject ?code=&state= query params (not
exposed by this Streamlit version), so this file validates the button
rendering and the fail-closed guards around it.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
import db as db_module
import sso as sso_module
from streamlit.testing.v1 import AppTest

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Throwaway SQLite so the login wall's db.init_db() call never touches
    the real data/app.db (same pattern as tests/test_deployment_sizing_page.py)."""
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(
            f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
        ),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def test_login_page_shows_sso_button_when_configured(isolated_db, monkeypatch):
    monkeypatch.setattr(sso_module, "sso_configured", lambda: True)
    monkeypatch.setattr(
        sso_module, "begin_sso_login",
        lambda *a, **k: ("https://idp.example.com/authorize?state=s1&nonce=n1", "s1", "n1"),
    )

    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.run()
    assert not at.exception, f"Login page crashed with SSO configured: {at.exception}"

    labels = [lb.label for lb in at.get("link_button")]
    assert any("enterprise SSO" in label for label in labels), labels
    # The anti-CSRF state + replay nonce must be stashed before the browser
    # navigates away, so the callback can verify them.
    assert at.session_state["sso_state"] == "s1"
    assert at.session_state["sso_nonce"] == "n1"
    # Password path is still present alongside SSO.
    assert any(t.label == "Username" for t in at.text_input)


def test_login_page_has_no_sso_button_when_not_configured(isolated_db, monkeypatch):
    for k in ("SSO_OIDC_ISSUER", "SSO_CLIENT_ID", "SSO_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)

    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.run()
    assert not at.exception, f"Login page crashed: {at.exception}"

    labels = [lb.label for lb in at.get("link_button")]
    assert not any("enterprise SSO" in label for label in labels), labels
    assert any(t.label == "Username" for t in at.text_input)
    assert "sso_state" not in at.session_state


def test_login_page_survives_sso_flow_failure(isolated_db, monkeypatch):
    """If the SSO flow raises (e.g. discovery endpoint unreachable), the
    login wall must show an honest caption, never crash — password login
    must remain fully usable."""

    def _boom(*a, **k):
        raise sso_module.SSOLoginError("simulated discovery failure")

    monkeypatch.setattr(sso_module, "sso_configured", lambda: True)
    monkeypatch.setattr(sso_module, "begin_sso_login", _boom)

    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.run()
    assert not at.exception, f"Login page crashed on SSO failure: {at.exception}"
    captions = [c.value for c in at.caption]
    assert any("SSO is unavailable" in c for c in captions), captions
    assert any(t.label == "Username" for t in at.text_input)
