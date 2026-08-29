"""
Logical domain groupings for Battery Intelligence Platform's ``src/`` modules.

This file does NOT move any files — the flat ``src/*.py`` layout is preserved
for backward compatibility (bare imports like ``from db import ...`` must keep
working).  Instead, this module re-exports the same modules under organized
namespaces for IDE navigation and new code that wants explicit domain paths.

Usage (new code, recommended)::

    from src._domain.core import db, data_loader, chemistry_profiles
    from src._domain.auth import sso, rbac, secrets_store
    from src._domain.analytics import ml_anomaly, pybamm_rul

Usage (existing code, unchanged)::

    import db                    # still works — src/ is on sys.path
    from data_loader import ...  # still works

Domain groups:
  core        — data infrastructure, persistence, schema
  auth        — authentication, authorization, secrets
  analytics   — ML models, physics, anomaly detection
  connectors  — external system adapters (BMS, market, CMMS)
  lifecycle   — business logic, recommendations, EU compliance
  exports     — report generation, passport export, PDF
  copilot     — knowledge retrieval, agent, templates
"""

from __future__ import annotations

# ── Core (data infrastructure) ────────────────────────────────────────────
# Modules that every other domain ultimately depends on.
CORE_MODULES = (
    "db", "data_loader", "chemistry_profiles", "cell_store", "bundle_cache",
    "design_system", "audit", "experiment_registry", "import_validator",
    "adapter_contract", "plugin_registry",
)

# ── Auth (authentication & authorization) ──────────────────────────────────
AUTH_MODULES = ("sso", "rbac", "secrets_store", "rate_limit")

# ── Analytics (ML + physics + detection) ──────────────────────────────────
ANALYTICS_MODULES = (
    "ml_anomaly", "streaming_analytics", "eis_model", "pinn_model",
    "physics_calibration", "spm_projection", "pybamm_rul",
    "fleet_clustering", "pvgis_client",
)

# ── Connectors (external system adapters) ─────────────────────────────────
CONNECTOR_MODULES = (
    "bms_connectors", "mqtt_stream", "live_feed", "market_data",
    "circunomics_adapter", "cmms_adapter", "manufacturing_connector",
    "optimade_export",
)

# ── Lifecycle (business logic & compliance) ────────────────────────────────
LIFECYCLE_MODULES = (
    "warranty", "consequences", "recommendations", "passport",
    "passport_export", "sustainability", "health_aware_dispatch",
    "managed_charging", "fleet_aggregation", "grid_services",
    "dynamic_circularity", "marketplace_matching", "deployment_sizing",
    "recycler_directory", "stakeholder_views",
)

# ── Exports (reports, PDF, compliance exports) ─────────────────────────────
EXPORT_MODULES = (
    "report_pdf", "model_cards", "china_recycling_export",
    "us_ira_export", "spine_export", "bankability_report",
)

# ── Copilot (knowledge & agent) ───────────────────────────────────────────
COPILOT_MODULES = (
    "battery_copilot", "battery_knowledge", "copilot_agent",
    "copilot_retrieval", "copilot_templates",
)

# ── Intelligence (graph, digital twin, memory) ────────────────────────────
INTELLIGENCE_MODULES = (
    "trajectory_memory", "knowledge_graph", "digital_twin",
    "task_queue", "notifications", "import_adapter", "lis_model",
)

# ── API & operations ──────────────────────────────────────────────────────
OPS_MODULES = ("api", "action_center", "pack_builder")

ALL_GROUPS = {
    "core": CORE_MODULES,
    "auth": AUTH_MODULES,
    "analytics": ANALYTICS_MODULES,
    "connectors": CONNECTOR_MODULES,
    "lifecycle": LIFECYCLE_MODULES,
    "exports": EXPORT_MODULES,
    "copilot": COPILOT_MODULES,
    "intelligence": INTELLIGENCE_MODULES,
    "ops": OPS_MODULES,
}
