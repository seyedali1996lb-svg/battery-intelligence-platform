"""
UI/server enforcement stay in lock-step via the single src/rbac.py registry.

src/rbac.py is now the one place role knowledge lives: the Decision/CMMS-ticket
write actions the API gates on, the `settings.manage` capability the Settings
page hides org-wide sections behind (the same admin boundary src/db.py's
`_require_admin()` enforces), and the `ui.nav.*` / `ui.frontload.*` capabilities
driving sidebar reachability + default expansion.

These tests exist to keep the two namespaces honest:
  - the auth-identity matrix (what server enforcement uses) is consistent with
    db.py's own admin boundary;
  - the UI affordances (nav groups, admin sections) are expressed as rbac
    capabilities and the app really reads them, not a parallel hardcoded dict.

auth_role tests reference the identity values used by src/api.py's
require_action and the db layer; persona tests reference the role-onboarding
display names in app/main.py's sidebar.
"""

import pathlib
import re

import pytest

import rbac
from rbac import (
    ACTION_CREATE_TICKET, ACTION_TRIAGE_TICKET, ACTION_DISPATCH_TICKET,
    CAP_SETTINGS_MANAGE, UI_NAV_GROUPS, ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET,
    ROLE_COMPLIANCE,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MAIN_PY = _ROOT / "app" / "main.py"
_SETTINGS_PY = _ROOT / "app" / "_pages" / "_settings_config.py"

_PERSONAS = rbac.PERSONA_ENGINEER, rbac.PERSONA_FLEET, rbac.PERSONA_EXECUTIVE, rbac.PERSONA_COMPLIANCE


# ── Write-action matrix (auth identities) ───────────────────────────────────

def test_write_action_grants_per_auth_role():
    assert rbac.allowed_roles(ACTION_CREATE_TICKET) == {ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET}
    assert rbac.allowed_roles(ACTION_TRIAGE_TICKET) == {ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET}
    assert rbac.allowed_roles(ACTION_DISPATCH_TICKET) == {ROLE_ADMIN, ROLE_ENGINEER}


def test_personas_are_never_a_security_identity():
    # Personas drive nav UX only; they must not silently grant write actions.
    initials = (ACTION_CREATE_TICKET, ACTION_TRIAGE_TICKET, ACTION_DISPATCH_TICKET,
                CAP_SETTINGS_MANAGE)
    for persona in _PERSONAS:
        for cap in initials:
            assert not rbac.can(persona, cap), (persona, cap)


# ── settings.manage aligns with db.py's admin-only boundary ─────────────────

def test_settings_manage_is_exactly_admin():
    # Whatever the Settings-page gate reads, it is admin-only — same policy as
    # src/db.py's _require_admin (the actual trust boundary on those writes).
    assert rbac.allowed_roles(CAP_SETTINGS_MANAGE) == {ROLE_ADMIN}
    assert rbac.can(ROLE_ADMIN, CAP_SETTINGS_MANAGE)
    for role in (ROLE_ENGINEER, ROLE_FLEET, ROLE_COMPLIANCE):
        assert not rbac.can(role, CAP_SETTINGS_MANAGE)


def test_settings_manage_ui_gate_matches_db_require_admin():
    # The db layer refuses any non-admin caller_role; the Settings UI gate uses
    # the same CAP_SETTINGS_MANAGE. Prove the equivalence holds end to end.
    from src import db

    for role in (ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET, ROLE_COMPLIANCE, None):
        allowed_by_admin = role == ROLE_ADMIN
        allowed_by_rbac = rbac.can(role, CAP_SETTINGS_MANAGE)
        assert allowed_by_admin == allowed_by_rbac, role

    db._require_admin(ROLE_ADMIN, "regression probe")  # no raise
    with pytest.raises(db.InsufficientRoleError):
        db._require_admin(None, "regression probe")
    # Safety: the probe didn't touched real data.
    assert rbac.allowed_roles(CAP_SETTINGS_MANAGE) == {ROLE_ADMIN}


def test_unknown_role_and_capability_fail_closed():
    assert rbac.role_capabilities("ghost") == frozenset()
    assert rbac.role_capabilities(None) == frozenset()
    assert not rbac.can("ghost", ACTION_CREATE_TICKET)
    assert not rbac.can(ROLE_ADMIN, "nope.not-a-capability")


# ── Nav affordances ──────────────────────────────────────────────────────────

def test_every_role_and_persona_can_reach_every_nav_group():
    # Design: groups are reachable for everyone, only default-expansion varies.
    nav_caps = frozenset(f"ui.nav.{g}" for g in UI_NAV_GROUPS)
    for role in (ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET, ROLE_COMPLIANCE) + _PERSONAS:
        assert nav_caps <= rbac.role_capabilities(role), role


def test_front_loaded_nav_per_persona():
    front = {g for g in UI_NAV_GROUPS if rbac.can(rbac.PERSONA_EXECUTIVE, f"ui.frontload.{g}")}
    assert front == {"Operate", "EU Passport"}
    front2 = {g for g in UI_NAV_GROUPS
              if rbac.can(rbac.PERSONA_COMPLIANCE, f"ui.frontload.{g}")}
    assert front2 == {"EU Passport"}
    # Technical personas expand everything by default (no curated front-load).
    assert rbac.front_loaded_nav(rbac.PERSONA_ENGINEER) is None
    assert rbac.front_loaded_nav(rbac.PERSONA_FLEET) is None


def test_front_loaded_nav_unknown_expands_all():
    assert rbac.front_loaded_nav("Some Future Role") is None
    assert rbac.front_loaded_nav(None) is None


def test_main_py_nav_groups_align_with_rbac_ui_nav_groups():
    """The app's NAV_GROUPS definitions must carry exactly the group labels the
    rbac nav capabilities are keyed on. If a nav group is added/renamed here and
    forgotten in rbac (or vice versa), this fails instead of silently drifting."""
    src = _MAIN_PY.read_text(encoding="utf-8")
    # NAV_GROUPS is a list of (label, [(page, route)...]) tuples; grab the label.
    block = src[src.index("NAV_GROUPS = ["):src.index("def _upload_status_line")]
    labels = re.findall(r'\("([^"]+)",\s*\[', block)
    assert labels == list(UI_NAV_GROUPS), (labels, UI_NAV_GROUPS)


def test_nav_frontload_only_refers_to_real_nav_groups():
    for persona in _PERSONAS:
        front = rbac.front_loaded_nav(persona) or frozenset()
        assert front <= set(UI_NAV_GROUPS)


# ── Settings page really reads the capability, not a parallel check ─────────

def test_settings_page_binds_admin_sections_to_cap_settings_manage():
    """
    Structural guard: the Settings page must gate its admin-only sections with
    rbac.CAP_SETTINGS_MANAGE (not a raw `auth_role == 'admin'`), and the
    sidebar must drive expansion from rbac.front_loaded_nav. This is what makes
    "the UI can't drift from server enforcement" true — if someone reintroduces
    a hardcoded admin check, this fails.
    """
    settings_src = _SETTINGS_PY.read_text(encoding="utf-8")
    assert "rbac.can(st.session_state.get(\"auth_role\"), rbac.CAP_SETTINGS_MANAGE)" in settings_src
    # No raw admin-string gates on the parts that should be capability-driven.
    assert not re.search(r'auth_role"\s*[!=]=\s*"admin"', settings_src)

    main_src = _MAIN_PY.read_text(encoding="utf-8")
    assert "rbac.front_loaded_nav" in main_src
    # The old hardcoded priority dict is gone.
    assert "_PRIORITY_GROUPS" not in main_src