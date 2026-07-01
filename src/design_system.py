"""
Phase 8 — Design System constants.

Single source of truth for badge rendering, color tokens, and page-level
metadata used across all dashboard pages. Import from here; do not
re-define locally inside page functions.

Badge exclusivity rules (enforced by using only these constants):
  BADGE_VALIDATED  Green   — ML pipeline outputs only (SOH, RUL, fade rate)
  BADGE_ESTIMATE   Amber   — literature / cited figures with a named source
  BADGE_ILLUST     Gray    — engineering judgment, no verifiable source
  BADGE_UNAVAIL    Muted   — honest gap; would require data this demo lacks

Three-state availability badge (used by Passport + Green Deal pages):
  make_state_badge("available")   Green   — pipeline-derived
  make_state_badge("estimated")   Amber   — cited/illustrative estimate
  make_state_badge("unavailable") Muted   — not available in this demo
"""

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

C_GREEN  = "#2f855a"   # validated output / available
C_AMBER  = "#b7791f"   # cited estimate / estimated
C_MUTED  = "#718096"   # illustrative / unavailable / secondary text
C_ORANGE = "#c05621"   # uncertain confidence / recycle action
C_BLUE   = "#63b3ed"   # info / continue action
C_YELLOW = "#d69e2e"   # inspect action / warning


# ---------------------------------------------------------------------------
# Badge renderer — do not call directly; use the pre-built instances below
# ---------------------------------------------------------------------------

def make_badge(label: str, colour: str = C_AMBER) -> str:
    """Return an inline HTML pill badge with the given label and colour."""
    return (
        f"<span style='background:{colour}22;border:1px solid {colour}55;"
        f"color:{colour};font-size:10px;font-weight:700;padding:1px 7px;"
        f"border-radius:10px;letter-spacing:0.06em'>{label}</span>"
    )


# ---------------------------------------------------------------------------
# Data-provenance badge instances
# ---------------------------------------------------------------------------

BADGE_VALIDATED = make_badge("Validated model output", C_GREEN)
BADGE_ESTIMATE  = make_badge("Cited estimate",           C_AMBER)
BADGE_ILLUST    = make_badge("Illustrative — not sourced", C_MUTED)
BADGE_UNAVAIL   = make_badge("Not available in demo",    C_MUTED)


# ---------------------------------------------------------------------------
# Three-state availability badge (Passport + EU Green Deal pages)
# ---------------------------------------------------------------------------

_STATE_CFG = {
    "available":   (C_GREEN,  "Available",             False),
    "estimated":   (C_AMBER,  "Estimated",             False),
    "unavailable": (C_MUTED,  "Not available in demo", True),
}


def make_state_badge(state: str) -> str:
    """Return an HTML availability badge for passport / green-deal fields."""
    c, label, italic = _STATE_CFG[state]
    extra = ";font-style:italic" if italic else ""
    return (
        f"<span style='background:{c}22;border:1px solid {c}55;color:{c};"
        f"font-size:10px;font-weight:700;padding:1px 7px;border-radius:10px;"
        f"letter-spacing:0.04em{extra}'>{label}</span>"
    )


# ---------------------------------------------------------------------------
# Section header HTML (shared across Sustainability, Consequences, etc.)
# ---------------------------------------------------------------------------

def section_header_html(title: str) -> str:
    """Return styled section-divider HTML for use with unsafe_allow_html."""
    return (
        f"<div style='font-size:11px;font-weight:600;color:#4a5568;"
        f"text-transform:uppercase;letter-spacing:0.08em;padding-bottom:8px;"
        f"border-bottom:1px solid #2d3748;margin:28px 0 16px'>{title}</div>"
    )


# ---------------------------------------------------------------------------
# Recommendations page metadata
# ---------------------------------------------------------------------------

ACTION_META = {
    "continue":    ("Continue Operation",   C_GREEN,  "#0d2016"),
    "inspect":     ("Schedule Inspection",  C_YELLOW, "#1f1a08"),
    "second_life": ("Route to Second-Life", C_BLUE,   "#0a1628"),
    "recycle":     ("Recycle",              C_ORANGE, "#1f0f06"),
}

CONF_META = {
    "high":      (C_GREEN,  "High confidence"),
    "medium":    (C_AMBER,  "Medium confidence — RUL not calibrated"),
    "lower":     (C_MUTED,  "Lower certainty — see notes below"),
    "uncertain": (C_ORANGE, "Lower certainty — multiple uncertainty factors"),
}
