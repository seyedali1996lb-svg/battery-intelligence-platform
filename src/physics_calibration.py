"""
Compatibility shim — the implementation moved to
``batlab.features.physics_calibration`` (battery-lab 0.2.0).

Why it moved
------------
This module used to *be* the implementation, inside the demo application, and
``batlab.features.engineering.build_features()`` reached across the package
boundary to import it opportunistically. That made a library feature column
depend on whose ``src/`` happened to be on ``sys.path``: the same four NASA
cells scored SOH R² **0.9580 without** the physics features and **0.9471 with
them** (measured 2026-09-19) depending only on how the process was launched.
A library whose numbers depend on the caller's import path is not a library.

So the implementation now lives in ``batlab.features.physics_calibration``, and
every environment that can ``pip install battery-lab`` computes one set of
features. PyBaMM stays an optional extra (``pip install "battery-lab[physics]"``),
whose absence is a declared dependency being missing — not an accident of
``sys.path``.

What this file is for
---------------------
Two things only, both application concerns:

1. Re-export the library's public names, so the existing
   ``from physics_calibration import calibrate_cell`` call sites
   (src/recommendations.py, app/_pages/health.py,
   app/_pages/_health_diagnostics.py, app/_pages/benchmark.py) keep working
   unchanged and there is exactly one implementation.
2. Register the application's ML mechanism classifier
   (``recommendations.diagnose_mechanism``) for ``physics_ml_agreement()``'s
   physics-vs-ML comparison. That classifier is an *application* artifact, so
   the library takes it through ``register_mechanism_classifier()`` instead of
   importing it.

Importing this module is what performs the registration.
"""

from __future__ import annotations

from batlab.features import physics_calibration as _impl

# Re-export the public surface the library declares (see its __all__).
from batlab.features.physics_calibration import *  # noqa: F401,F403
from batlab.features.physics_calibration import (  # noqa: F401  explicit, for readers and type checkers
    ANCHOR_PARAM_SETS,
    DOMINANT_MODE_RATIO,
    MIN_CYCLES_FOR_CALIBRATION,
    MIN_FIT_R2_FOR_DOMINANT_MODE,
    PHYSICS_FEATURE_COLUMNS,
    REFIT_EVERY_CYCLES,
    calibrate_cell,
    calibrated_feature_series,
    dominant_mode,
    fit_resistance_growth,
    fit_two_term_fade,
    get_mechanism_classifier,
    physics_gbrt_divergence_report,
    physics_ml_agreement,
    register_anchor_param_set,
    register_mechanism_classifier,
    reset_nominal_capacity_cache,
)

# ── Application-only wiring ────────────────────────────────────────────────
# Registered at import time: recommendations imports this module, and the pages
# import recommendations, so by the time any UI calls physics_ml_agreement() the
# classifier is in place. `register_mechanism_classifier` itself is part of the
# re-exported surface above.
from recommendations import diagnose_mechanism as _diagnose_mechanism  # noqa: E402

_impl.register_mechanism_classifier(_diagnose_mechanism)

__all__ = [
    "ANCHOR_PARAM_SETS",
    "DOMINANT_MODE_RATIO",
    "MIN_CYCLES_FOR_CALIBRATION",
    "MIN_FIT_R2_FOR_DOMINANT_MODE",
    "PHYSICS_FEATURE_COLUMNS",
    "REFIT_EVERY_CYCLES",
    "calibrate_cell",
    "calibrated_feature_series",
    "dominant_mode",
    "fit_resistance_growth",
    "fit_two_term_fade",
    "get_mechanism_classifier",
    "physics_gbrt_divergence_report",
    "physics_ml_agreement",
    "register_anchor_param_set",
    "register_mechanism_classifier",
    "reset_nominal_capacity_cache",
]
