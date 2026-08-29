# src/ — Battery Intelligence Platform core modules.
#
# This package makes every module importable as `src.db`, `src.api`, etc.
# for code that wants explicit domain paths.  Bare imports (`import db`,
# `from data_loader import ...`) continue to work because src/ is on
# sys.path via _paths.py.

from __future__ import annotations

import importlib as _importlib
import os as _os
import sys as _sys

# ── Ensure src/ is importable as bare names (idempotent). ──────────────────
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in _sys.path:
    _sys.path.insert(0, _dir)

# ── Lazy-import all sibling modules so `from src import db` works. ──────────
# We intentionally do NOT eagerly import everything — that would create
# circular-import risk and slow startup.  Instead, we register the names
# so that `import src; src.db` resolves lazily via __getattr__.

_MODULE_NAMES = [
    "action_center", "adapter_contract", "api", "audit",
    "bankability_report", "battery_copilot", "battery_knowledge",
    "bms_connectors", "bundle_cache", "cell_store",
    "chemistry_profiles", "china_recycling_export",
    "circunomics_adapter", "cmms_adapter", "consequences",
    "copilot_agent", "copilot_retrieval", "copilot_templates",
    "contracts", "data_loader", "db", "deployment_sizing",
    "design_system", "digital_twin", "dynamic_circularity",
    "eis_model", "experiment_registry", "fleet_aggregation",
    "fleet_clustering", "grid_services", "health_aware_dispatch",
    "import_adapter", "import_validator", "knowledge_graph",
    "lis_model", "live_feed", "managed_charging",
    "manufacturing_connector", "marketplace_matching",
    "market_data", "ml_anomaly", "model_cards", "mqtt_stream",
    "notifications", "optimade_export", "pack_builder",
    "passport", "passport_export", "physics_calibration",
    "pinn_model", "plugin_registry", "protocols", "pvgis_client",
    "pybamm_rul", "rate_limit", "rbac", "recommendations",
    "recycler_directory", "report_pdf", "secrets_store",
    "spine_export", "spm_projection", "sso", "stakeholder_views",
    "streaming_analytics", "sustainability", "task_queue",
    "trajectory_memory", "us_ira_export", "warranty",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in _MODULE_NAMES:
        return _importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_MODULE_NAMES)
