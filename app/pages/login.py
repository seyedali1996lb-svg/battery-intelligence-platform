"""Login wall — demo credentials gate."""

import streamlit as st

# Demo credentials  (username → (password, role, display_name))
_USERS = {
    "engineer":   ("battery",   "engineer",   "Battery Engineer"),
    "fleet":      ("ops2024",   "fleet",      "Fleet Operations"),
    "compliance": ("eu2024",    "compliance", "Compliance Officer"),
    "admin":      ("admin",     "admin",      "Administrator"),
}

_ROLE_DEFAULT_PAGE = {
    "fleet":      "fleet",
    "compliance": "passport",
    "engineer":   "health",
    "admin":      "overview",
}

_ROLE_DESCRIPTIONS = {
    "engineer":   "Cell-level analysis · Health · Insights · Copilot",
    "fleet":      "Fleet operations · Recommendations · Grading",
    "compliance": "EU Passport · Sustainability · Reports",
    "admin":      "Full platform access",
}


def render_login() -> bool:
    """
    Show the login form. Returns True if the user is already authenticated.
    Sets st.session_state['auth_user'] and ['auth_role'] on success.
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        """
        <div style='display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:60vh;padding:40px'>
          <div style='font-size:32px;font-weight:800;color:#e2e8f0;margin-bottom:6px'>
            ⚡ Battery Intelligence Platform
          </div>
          <div style='font-size:13px;color:#4a5568;margin-bottom:40px'>
            v1.0 · Phase 1 · Demo environment
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown(
            "<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:14px;"
            "padding:32px 28px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:16px;font-weight:700;color:#e2e8f0;margin-bottom:20px'>"
            "Sign in</div>",
            unsafe_allow_html=True,
        )

        username = st.text_input("Username", key="login_user", placeholder="engineer")
        password = st.text_input("Password", key="login_pass", type="password", placeholder="••••••••")

        if st.button("Sign in", use_container_width=True, type="primary"):
            creds = _USERS.get(username.strip().lower())
            if creds and creds[0] == password:
                st.session_state["authenticated"]  = True
                st.session_state["auth_user"]      = username.strip().lower()
                st.session_state["auth_role"]      = creds[1]
                st.session_state["auth_name"]      = creds[2]
                # Set default page for this role
                default = _ROLE_DEFAULT_PAGE.get(creds[1], "overview")
                if "page" not in st.session_state:
                    st.session_state["page"] = default
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Demo credential hints
        st.markdown(
            "<div style='margin-top:20px;font-size:11px;color:#4a5568;text-align:center;line-height:2'>"
            "<strong style='color:#718096'>Demo credentials</strong><br>"
            + "<br>".join(
                f"<span style='color:#718096'>{u}</span> / {v[0]}"
                f"<span style='color:#4a5568;margin-left:8px'>({v[2]})</span>"
                for u, v in _USERS.items()
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    return False
