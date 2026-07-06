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


def render_login() -> bool:
    """
    Show the login/signup form. Returns True if the user is already
    authenticated. Sets st.session_state['auth_user'/'auth_role'/'auth_org_id'/
    'auth_org_name'] etc. on success.
    """
    if st.session_state.get("authenticated"):
        return True

    db.init_db()

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["Sign in", "Create organization"])

        with tab_signin:
            st.markdown("### Sign in")
            # st.form() captures every field's value atomically at submit time —
            # avoids a race between a fast Enter-key/click submit and a text_input's
            # value not yet having committed to session_state (a known Streamlit
            # gotcha with bare text_input + button, worse with browser autofill).
            with st.form("signin_form"):
                username = st.text_input("Username", key="login_user", placeholder="engineer")
                password = st.text_input("Password", key="login_pass", type="password", placeholder="••••••••")
                signin_submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

            if signin_submitted:
                user = db.get_user_by_username(username)
                if user and db.verify_password(password, user["password_hash"]):
                    _log_in_user(user)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.markdown(
                "<div style='margin-top:20px;font-size:11px;color:#4a5568;text-align:center;line-height:2'>"
                "<strong style='color:#718096'>Demo credentials (Demo Org)</strong><br>"
                + "<br>".join(
                    f"<span style='color:#718096'>{u}</span> / {v[0]}"
                    f"<span style='color:#4a5568;margin-left:8px'>({v[2]})</span>"
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
