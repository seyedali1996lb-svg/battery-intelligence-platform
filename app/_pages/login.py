"""Login wall — real multi-tenant auth backed by src/db.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import db

_ROLE_DEFAULT_PAGE = {
    "fleet":      "fleet",
    "compliance": "passport",
    "engineer":   "health",
    "admin":      "overview",
}


def _log_in_user(user: dict) -> None:
    st.session_state["authenticated"]  = True
    st.session_state["auth_user"]      = user["username"]
    st.session_state["auth_role"]      = user["role"]
    st.session_state["auth_name"]      = user["display_name"]
    st.session_state["auth_org_id"]    = user["org_id"]
    st.session_state["auth_org_name"]  = user["org_name"]
    default = _ROLE_DEFAULT_PAGE.get(user["role"], "overview")
    if "page" not in st.session_state:
        st.session_state["page"] = default


def _handle_sso_callback() -> None:
    """The IdP redirected back to this app with ?code=...&state=.... Exchange
    the code, provision/link the account against the existing User model
    (src/db.provision_or_link_sso_user), and log the user in. Any failure is
    surfaced as a session message, never a crash."""
    import sso

    qp = st.query_params
    code = qp.get("code", "")
    state = qp.get("state", "")
    expected_state = st.session_state.pop("sso_state", "")
    nonce = st.session_state.pop("sso_nonce", None)
    try:
        userinfo = sso.complete_sso_login(
            code, state, expected_state, nonce=nonce
        )
        user = db.provision_or_link_sso_user(
            userinfo["email"],
            userinfo.get("name") or "",
            userinfo["idp"],
            userinfo["sub"],
        )
        _log_in_user(user)
        st.session_state["sso_notice"] = (
            f"Signed in via enterprise SSO ({userinfo['idp']})."
        )
    except sso.SSOLoginError as exc:
        st.session_state["sso_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — provisioning failure is a page message
        st.session_state["sso_error"] = f"SSO sign-in could not be completed: {exc}"
    finally:
        qp.clear()
        st.rerun()


def render_login() -> bool:
    """
    Show the login/signup form. Returns True if the user is already
    authenticated. Sets st.session_state['auth_user'/'auth_role'/'auth_org_id'/
    'auth_org_name'] etc. on success.
    """
    if st.session_state.get("authenticated"):
        return True

    db.init_db()

    # ── SSO callback: the IdP redirected back with ?code=&state= ─────────────
    # Must be handled before the tabs render — this run either completes the
    # login (st.rerun -> authenticated) or shows the error and clears the
    # query params so a refresh doesn't replay a stale code.
    _sso_error = st.session_state.pop("sso_error", None)
    if _sso_error:
        st.error(_sso_error)
    _sso_notice = st.session_state.pop("sso_notice", None)
    if _sso_notice:
        st.success(_sso_notice)
    if st.query_params.get("code") and st.query_params.get("state"):
        _handle_sso_callback()
        return False

    st.markdown("# Battery Intelligence Platform")

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["Sign in", "Create organization"])

        with tab_signin:
            st.markdown("### Sign in")
            # ── Enterprise SSO button (shown only when the deployment has an
            # OIDC provider configured — see src/sso.py). Generates the anti-
            # CSRF state + replay nonce, stashes them in the session so the
            # callback can verify them, and sends the user to the IdP.
            import sso
            if sso.sso_configured():
                try:
                    _auth_url, _state, _nonce = sso.begin_sso_login()
                except sso.SSOLoginError as exc:
                    st.caption(f"Enterprise SSO is unavailable: {exc}")
                else:
                    st.session_state["sso_state"] = _state
                    st.session_state["sso_nonce"] = _nonce
                    st.link_button(
                        "Continue with enterprise SSO",
                        _auth_url,
                        use_container_width=True,
                        type="secondary",
                    )
                    st.caption(
                        "Sign in with your organization's identity provider "
                        "(Okta / Entra ID / Keycloak / ...)"
                    )
                    st.markdown(
                        "<div style='display:flex;align-items:center;gap:12px;"
                        "margin:14px 0 6px;color:#5b6b7e;font-size:12px'>"
                        "<span style='flex:1;height:1px;background:#2d3748'></span>"
                        "or use a username and password"
                        "<span style='flex:1;height:1px;background:#2d3748'></span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            # st.form() captures every field's value atomically at submit time —
            # avoids a race between a fast Enter-key/click submit and a text_input's
            # value not yet having committed to session_state (a known Streamlit
            # gotcha with bare text_input + button, worse with browser autofill).
            with st.form("signin_form"):
                username = st.text_input("Username", key="login_user", placeholder="engineer")
                password = st.text_input("Password", key="login_pass", type="password", placeholder="••••••••")
                signin_submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

            if signin_submitted:
                _locked_until = db.is_login_locked_out(username)
                if _locked_until:
                    st.error(
                        f"Too many failed sign-in attempts. Locked until "
                        f"{_locked_until[:16].replace('T', ' ')} (server time)."
                    )
                else:
                    user = db.get_user_by_username(username)
                    if user and db.verify_password(password, user["password_hash"]):
                        db.reset_login_attempts(username)
                        _log_in_user(user)
                        st.rerun()
                    else:
                        db.record_failed_login(username)
                        st.error("Invalid username or password.")

            st.markdown(
                "<div style='margin-top:20px;font-size:11px;color:#a0aec0;text-align:center;line-height:2'>"
                "<strong style='color:#a0aec0'>Demo credentials (Demo Org)</strong><br>"
                + "<br>".join(
                    f"<span style='color:#a0aec0'>{u}</span> / {v[0]}"
                    f"<span style='color:#a0aec0;margin-left:8px'>({v[2]})</span>"
                    for u, v in db.DEMO_USERS.items()
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        with tab_signup:
            st.markdown("### Create your organization")
            st.markdown(
                "<div style='font-size:12px;color:#8896a8;margin-bottom:12px;line-height:1.6'>"
                "Creates a new organization with its own private fleet, decisions, and settings — "
                "isolated from every other organization on this platform. You become its first "
                "administrator."
                "</div>",
                unsafe_allow_html=True,
            )
            with st.form("signup_form"):
                org_name  = st.text_input("Organization name", key="signup_org", placeholder="Acme Batteries")
                su_user   = st.text_input("Your username", key="signup_user", placeholder="alice")
                su_name   = st.text_input("Display name (optional)", key="signup_display", placeholder="Alice Nguyen")
                su_pass   = st.text_input("Password", key="signup_pass", type="password")
                su_pass2  = st.text_input("Confirm password", key="signup_pass2", type="password")
                signup_submitted = st.form_submit_button("Create organization", use_container_width=True, type="primary")

            if signup_submitted:
                if not (org_name.strip() and su_user.strip() and su_pass):
                    st.error("Organization name, username, and password are required.")
                elif su_pass != su_pass2:
                    st.error("Passwords do not match.")
                elif len(su_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    result = db.create_organization_with_admin(org_name, su_user, su_pass, su_name)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        user = db.get_user_by_username(su_user)
                        _log_in_user(user)
                        st.success(f"Organization '{result['org_name']}' created.")
                        st.rerun()

    return False
