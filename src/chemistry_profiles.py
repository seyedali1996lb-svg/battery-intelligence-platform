"""
Chemistry registry — pluggable profile system.

Adding a new chemistry: create a subclass of ChemistryProfile,
implement the three abstract methods, and call register().
The Health page and sidebar selector discover it automatically via REGISTRY.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ChemistryProfile:
    name: str
    short_name: str
    color: str            # hex color for UI badges
    eol_soh_pct: float    # default EOL threshold
    typical_cycles: int   # nominal cycle life
    notes: str = ""

    REGISTRY: ClassVar[dict[str, "ChemistryProfile"]] = {}

    def __post_init__(self):
        ChemistryProfile.REGISTRY[self.name] = self

    def get_health_notes(self) -> list[str]:
        """Return chemistry-specific interpretation notes for the Health page."""
        return []

    def get_eis_params(self) -> dict:
        """Return EIS model parameters specific to this chemistry."""
        return {}

    def get_degradation_notes(self) -> str:
        """Return a one-paragraph description of expected degradation mechanisms."""
        return self.notes


LI_ION = ChemistryProfile(
    name="Li-ion (LiCoO₂)",
    short_name="LCO",
    color="#63b3ed",
    eol_soh_pct=80.0,
    typical_cycles=500,
    notes=(
        "LiCoO₂ degrades primarily via SEI growth (calendar aging) and "
        "lithium plating at high C-rates. Resistance rise is the dominant "
        "early indicator; capacity fade accelerates after ~400 cycles."
    ),
)

LI_S = ChemistryProfile(
    name="Li-S (Lithium-Sulfur)",
    short_name="LiS",
    color="#f6ad55",
    eol_soh_pct=70.0,
    typical_cycles=200,
    notes=(
        "Li-S exhibits a dual discharge plateau (2.35 V / 2.1 V). "
        "Polysulfide shuttle drives CE losses (typical 95–99%). "
        "Fade is faster than Li-ion; EOL convention at 70% SOH."
    ),
)

SSB = ChemistryProfile(
    name="SSB (Solid-State)",
    short_name="SSB",
    color="#9f7aea",
    eol_soh_pct=85.0,
    typical_cycles=1000,
    notes=(
        "Solid-state batteries show no Warburg diffusion tail in EIS "
        "(no liquid electrolyte). Interface resistance is dominant. "
        "Higher Arrhenius Ea (~0.6–0.8 eV vs 0.3–0.5 eV for Li-ion). "
        "Longer cycle life but sensitive to stack pressure."
    ),
)


def get_profile(name: str) -> ChemistryProfile:
    """Return a profile by name, defaulting to Li-ion."""
    return ChemistryProfile.REGISTRY.get(name, LI_ION)


def all_profiles() -> list[ChemistryProfile]:
    return list(ChemistryProfile.REGISTRY.values())
