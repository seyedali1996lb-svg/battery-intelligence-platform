"""
Server-side role-based write gating for the REST layer.

This is the fix for the long-standing "no server-side write-action gating
yet" row in the Settings roadmap (docs/history.md + the Settings page). Before
this module, role awareness (Engineer / Fleet Manager / Executive / Compliance
Officer) lived only in the Streamlit UI: the sidebar/nav filtered what a role
could *see*, but the REST layer trusted any authenticated user on every write
endpoint. Nothing stopped a direct HTTP call from a read-only role -- e.g.
Compliance -- creating or triaging a decision ticket, or dispatching a work
order to CMMS.

This module is the enforcement boundary: a data-driven permission matrix plus a
pure `can()` check. The FastAPI dependency that raises 403 lives in src/api.py
(`require_action`), but the *policy* is here so it can be unit-tested without
FastAPI, and so a new write route has one obvious place to declare its roles.

Roles are the string values stored in `User.role` (src/db.py) and carried in
the JWT `role` claim (src/api.py's get_current_user): engineer, fleet,
compliance, admin. Unknown roles are denied (fail closed), and the matrix is
more restrictive than generous -- a role that has no listed write capability
is read-only.
"""

from __future__ import annotations

from typing import FrozenSet

# Authenticated role values (User.role in src/db.py; JWT "role" claim).
ROLE_ADMIN = "admin"
ROLE_ENGINEER = "engineer"
ROLE_FLEET = "fleet"
ROLE_COMPLIANCE = "compliance"

# The write actions the REST layer exposes. Keeping them in one registry makes
# the whole gated surface auditable at a glance and gives reviewers a single
# list to extend as new write routes are added.
ACTION_CREATE_TICKET = "action.create"      # Author an action-center / decision ticket
ACTION_TRIAGE_TICKET = "action.triage"      # Change a ticket's status / assignment
ACTION_DISPATCH_TICKET = "action.dispatch"  # Commit a ticket to CMMS/Warranty/Circularity

# Permission matrix: action -> set of roles allowed to perform it.
#
# Rationale:
#  - admin is privileged (and already gate-keeps settings/site/fleet/pack
#    management at the db.py trust boundary) -> allowed on every write.
#  - engineer and fleet are the operational roles that own tickets and are
#    responsible for the day-to-day disposition of a cell -> allowed to
#    create and triage tickets.
#  - dispatch commits a real, external work order (CMMS / warranty /
#    circularity) and is the highest-privilege write here -> restricted to
#    engineer + admin only. Fleet can assemble and triage, but the external
#    commit is gated above it.
#  - compliance is a review-only role -> denied every write (read-only).
_PERMISSIONS: dict[str, FrozenSet[str]] = {
    ACTION_CREATE_TICKET: frozenset({ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET}),
    ACTION_TRIAGE_TICKET: frozenset({ROLE_ADMIN, ROLE_ENGINEER, ROLE_FLEET}),
    ACTION_DISPATCH_TICKET: frozenset({ROLE_ADMIN, ROLE_ENGINEER}),
}


def can(role: str, action: str) -> bool:
    """Pure check: may `role` perform `action`? Unknown roles are denied."""
    return role in allowed_roles(action)


def allowed_roles(action: str) -> FrozenSet[str]:
    """Roles permitted to perform `action` (empty set for unknown actions)."""
    return _PERMISSIONS.get(action, frozenset())


def describe(action: str) -> str:
    """Human-readable permission line for `action`, used in 403 details."""
    roles = sorted(allowed_roles(action))
    required = ", ".join(roles) if roles else "(no role)"
    return f"requires one of: {required}"