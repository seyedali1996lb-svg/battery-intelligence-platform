"""
Design tokens, constants, and configuration shared across the Streamlit app.

Extracted from utils.py to separate pure-data definitions from UI rendering logic.
Every module that needs a Plotly config, a feature label, or a cell-ID list
imports from here instead of duplicating the value.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Cell identifiers
# ---------------------------------------------------------------------------

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]
SEVERSON_CELL_PREFIX = "S-"   # all Severson cells start with "S-"
MEASURED_CELL_IDS = set(NASA_CELL_IDS)  # extended at load time


# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------

LEGEND_H = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#718096"),
)

# Publication-quality export config — Plotly toolbar SVG button.
PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "battery_intel_chart",
        "height": 500,
        "width": 900,
        "scale": 2,
    },
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


# ---------------------------------------------------------------------------
# Feature labels (human-readable names for ML feature columns)
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "cycle_number":        "Cycle age",
    "fade_rate_10cy":      "Fade rate (10-cy)",
    "fade_rate_30cy":      "Fade rate (30-cy)",
    "fade_rate_50cy":      "Fade rate (50-cy)",
    "fade_acceleration":   "Fade acceleration",
    "soh_velocity_50cy":   "SOH velocity",
    "resistance_ohm":      "Internal resistance",
    "resistance_normalized": "Resistance (norm.)",
    "resistance_trend_30cy": "Resistance trend",
    "temp_rolling_30cy":   "Temperature (30-cy avg)",
    "c_rate_rolling_10cy": "C-rate (10-cy avg)",
    "stress_index":        "Composite stress index",
    "dod_proxy":           "Depth of Discharge (proxy)",
}


# ---------------------------------------------------------------------------
# State-of-health: the ONE place the band thresholds and colours are written
# ---------------------------------------------------------------------------

#: At or above this SOH% a cell is called Healthy; at or below ``SOH_EOL_MIN``
#: it is End of Life; between the two, Degrading. These two numbers (and the
#: three colours below) used to be retyped wherever a cell was labelled — the
#: scene's bands, the pack chart's bands, the 3D scatter's ramp, a tile's
#: accent, and app/_ui_helpers.soh_status()'s own comparisons — so any drift
#: between them could paint a cell green here and amber there. Everything now
#: imports from here; ``tests/test_cell_scene.py`` pins this module against
#: app/static/theme.css, which is where the same hexes are painted for text.
SOH_HEALTHY_MIN = 90.0
SOH_EOL_MIN = 80.0

#: The three band colours, matching app/static/theme.css's
#: ``.hero-green`` / ``.hero-yellow`` / ``.hero-red`` exactly.
SOH_HEALTHY_COLOR = "#48bb78"
SOH_DEGRADING_COLOR = "#f6e05e"
SOH_EOL_COLOR = "#fc8181"


# ---------------------------------------------------------------------------
# Card / tile colours
# ---------------------------------------------------------------------------

CARD_BG     = "#1e2a38"
CARD_BORDER = "#2d3748"


# ---------------------------------------------------------------------------
# Pack builder source-key mapping
# ---------------------------------------------------------------------------

PACK_BUNDLE_KEY = {
    "nasa": "nasa", "severson": "severson", "synthetic": "synth", "uploaded": "upload",
    "zhu2022": "zhu2022", "calce": "calce",
}
