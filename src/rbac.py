"""
Single source of truth for role-based permissions — server write enforcement
AND Streamlit UI affordances read from the same registry.

This grew out of two separate fixes. First it was the server-side write gate
closing the "no server-side write-action gating yet" roadmap row: before it,
the REST layer trusted any authenticated role on the Decision/CMMS-ticket
write endpoints, and roles only filtered what the Streamlit UI *showed*. Then
it absorbed the UI side: the sidebar's per-role nav expansion and the admin
Settings section gate were hardcoded dicts/ad-hoc `auth_role == "admin"`
checks in app code, a separate copy of the same role knowledge that could
drift from server enforcement. Now both live here and both flow through the
same `role_capabilities()` object:

  - **Server write enforcement** (src/api.py `require_action`): the
    Decision/CMMS-ticket write actions you see below, checked with
    `can(role, action)`.
  - **UI affordances** (app/_pages/_settings_config.py, app/main.py):
    `settings.manage` gates the org-wide config sections (admin only, the
    same policy `src/db.py`'s `_require_admin()` enforces server-side), and
    the `ui.nav.*` / `ui.frontload.*` capabilities drive which nav groups a
    role can reach and which start expanded in the sidebar.

Because every affordance is a capability key in this one registry, adding or
loosening a permission updates the UI automatically and vice-versa — the app
never hardcodes a role check that the enforcement path doesn't know about.

Namespaces, made explicit because the app genuinely has two:
  - **Auth identities** (`engineer` / `fleet` / `compliance` / `admin`, the
    `User.role` values in src/db.py carried in the JWT `role` claim and stored
    in `st.session_state['auth_role']`). These are the security boundary and
    are what server enforcement checks.
  - **Personas** (`Engineer` / `Fleet Manager` / `Executive` / `Compliance
    Officer`, the role-onboarding display names in `st.session_state
    ['user_role']`). These drive *nav UX* only — they are a demo affordance a
    user freely switches, never a security identity.

Unknown roles and unknown capabilities resolve to the empty set (fail closed)
for anything security-relevant; `front_loaded_nav` returns None (expand every
nav group) for unknown personas so the sidebar never breaks.
"""

from __future__ import annotations

from typing import FrozenSet

# ── Role keys (auth identities → src/db.py's User.role) ─────────────────────
ROLE_ADMIN = "admin"
ROLE_ENGINEER = "engineer"
ROLE_FLEET = "fleet"
ROLE_COMPLIANCE = "compliance"

# Persona display names used by the role-onboarding nav selector. Not auth
# identities — never checked server-side, only read for sidebar expansion.
PERSONA_ENGINEER = "Engineer"
PERSONA_FLEET = "Fleet Manager"
PERSONA_EXECUTIVE = "Executive"
PERSONA_COMPLIANCE = "Compliance Officer"

# ── Capability keys ─────────────────────────────────────────────────────────

# Decision/CMMS-ticket write actions (server enforcement, src/api.py).
ACTION_CREATE_TICKET = "action.create"      # Author an action-center / decision ticket
ACTION_TRIAGE_TICKET = "action.triage"      # Change a ticket's status / assignment
ACTION_DISPATCH_TICKET = "action.dispatch"  # Commit a ticket to CMMS/Warranty/Circularity

# Org-wide configuration sections (Settings page). Mirrors src/db.py's
# `_require_admin`/`_ADMIN_ONLY_SETTING_KEYS` boundary: the UI hides these
# sections for anyone `can(role, CAP_SETTINGS_MANAGE)` returns False for, and
# the db layer independently refuses the write — one policy, two layers.
CAP_SETTINGS_MANAGE = "settings.manage"

# Nav groups the app brands (see app/main.py NAV_GROUPS — kept in lock-step by
# a drift test). `ui.nav.<group>` grants reachability; `ui.frontload.<group>`
# marks it as default-expanded ("front-loaded") for that role.
UI_NAV_GROUPS = ("Analyse", "EU Passport", "Operate", "Configure")


def _nav_caps(groups: tuple[str, ...]) -> FrozenSet[str]:
    return frozenset(f"ui.nav.{g}" for g in groups)


def _frontload(g: str) -> str:
    return f"ui.frontload.{g}"


_NAV_ACCESS = _nav_caps(UI_NAV_GROUPS)  # every role can reach every group
_ALL_WRITES = frozenset({ACTION_CREATE_TICKET, ACTION_TRIAGE_TICKET, ACTION_DISPATCH_TICKET})


# ── The registry ────────────────────────────────────────────────────────────
# role -> full capability set. Anything not listed for a role is denied.
_ROLE_CAPABILITIES: dict[str, FrozenSet[str]] = {
    # Auth identities (security boundary).
    ROLE_ADMIN: _ALL_WRITES | _NAV_ACCESS | frozenset({CAP_SETTINGS_MANAGE}),
    ROLE_ENGINEER: _ALL_WRITES | _NAV_ACCESS,
    ROLE_FLEET: frozenset({ACTION_CREATE_TICKET, ACTION_TRIAGE_TICKET}) | _NAV_ACCESS,
    ROLE_COMPLIANCE: _NAV_ACCESS,

    # Personas (nav UX only). Write actions are deliberately NOT granted here
    # — personas are not a security identity; server enforcement keys off the
    # auth identity above.
    PERSONA_ENGINEER: _NAV_ACCESS,
    PERSONA_FLEET: _NAV_ACCESS,
    PERSONA_EXECUTIVE: _NAV_ACCESS | frozenset({_frontload("Operate"), _frontload("EU Passport")}),
    PERSONA_COMPLIANCE: _NAV_ACCESS | frozenset({_frontload("EU Passport")}),
}


def role_capabilities(role: str | None) -> FrozenSet[str]:
    """Full capability set for `role`; None/unknown roles fail closed to empty."""
    if role is None:
        return frozenset()
    return _ROLE_CAPABILITIES.get(role, frozenset())


def can(role: str | None, capability: str) -> bool:
    """Pure check: may `role` perform `capability`? Unknown roles are denied."""
    return capability in role_capabilities(role)


def allowed_roles(capability: str) -> FrozenSet[str]:
    """Roles permitted to perform `capability` (empty set for unknown keys)."""
    return frozenset(r for r in _ROLE_CAPABILITIES if capability in _ROLE_CAPABILITIES[r])


def describe(capability: str) -> str:
    """Human-readable permission line for `capability`, used in 403 details."""
    roles = sorted(allowed_roles(capability))
    required = ", ".join(roles) if roles else "(no role)"
    return f"requires one of: {required}"


def front_loaded_nav(role: str | None) -> FrozenSet[str] | None:
    """Nav groups to start expanded in the sidebar for `role`.

    None -> expand EVERY group (the default for technical personas and for any
    personality we don't explicitly curate, so the sidebar never breaks for an
    unknown role). Otherwise returns the subset of UI_NAV_GROUPS to open by
    default; the rest stay reachable but collapsed (never hidden).
    """
    caps = role_capabilities(role)
    front = frozenset(g for g in UI_NAV_GROUPS if _frontload(g) in caps)
    return front or None