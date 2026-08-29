"""
Shared API response contracts for the Battery Intelligence Platform.

Defines the canonical response shapes that both the Streamlit dashboard and
the React/TypeScript frontend consume.  Every REST endpoint in ``src/api.py``
returns one of these types; the React frontend's TypeScript interfaces should
mirror them exactly.

The goal: if the backend changes a field name, adding a ``TypedDict`` here
makes the drift visible at Python type-check time, and a CI rule can verify
the TypeScript ``interfaces/`` files match.

Usage::

    from contracts import CellSummary, FleetSummary, HealthResponse

    def get_cell_summary(cell_id: str) -> CellSummary:
        ...
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# Cell-level responses
# ---------------------------------------------------------------------------

class CellSummary(TypedDict, total=False):
    """Minimal per-cell summary for list views (Fleet, Grading, Explore)."""
    cell_id: str
    source: str                         # "nasa" | "severson" | "synth" | "uploaded"
    chemistry: str                      # "LiCoO2" | "LFP" | "NCA" | …
    soh_pct: float
    rul_cycles: int | None
    rul_reliable: bool
    fade_rate_30cy: float
    cycle_count: int
    eol_cycle: int | None
    status: str                         # "healthy" | "degraded" | "critical"


class SOHPrediction(TypedDict):
    """Point estimate + interval for a single cell's SOH."""
    soh_pct: float
    soh_lower: float | None             # Q10 bound (None if not calibrated)
    soh_upper: float | None             # Q90 bound
    reliable: bool


class RULPrediction(TypedDict):
    """Point estimate + interval for a single cell's RUL."""
    rul_cycles: int
    rul_lower: int | None               # Q10
    rul_upper: int | None               # Q90
    reliable: bool
    n_folds: int | None                 # LCO sample size (e.g. 4 for NASA)


class FeatureImportance(TypedDict):
    """One entry in the model's feature-importance breakdown."""
    feature: str
    importance: float
    label: str                          # human-readable label


class HealthResponse(TypedDict, total=False):
    """Full health-as-a-service response for GET /cells/{id}/health."""
    cell_id: str
    chemistry: str
    soh: SOHPrediction
    rul: RULPrediction
    fade_rate_30cy: float
    sop_pct: float                      # State of Power
    knee_cycle: int | None
    knee_confidence: float | None
    mechanism: str | None               # "LLI" | "LAM" | "Mixed" | None
    mechanism_confidence: str | None    # "High" | "Medium" | "Low" | None
    recommendation: str                 # "continue" | "inspect" | "second_life" | "recycle"
    recommendation_confidence: str
    model_card: dict | None             # auto-generated model card for the run behind the model
    eol_r_code: str | None              # EU Battery Passport End-of-Life R-code
    second_life_applications: list[str] | None


# ---------------------------------------------------------------------------
# Fleet-level responses
# ---------------------------------------------------------------------------

class FleetSummary(TypedDict):
    """Aggregate fleet metrics for the Fleet page executive bar."""
    total_cells: int
    healthy_count: int
    degraded_count: int
    critical_count: int
    avg_soh: float
    min_soh: float
    cells_at_risk: int                  # below EOL threshold


class FleetAlert(TypedDict):
    """One fleet-level alert (anomaly, trajectory match, passport gap, etc.)."""
    alert_type: str                     # "anomaly" | "trajectory_match" | "passport_gap" | …
    severity: str                       # "critical" | "high" | "medium" | "low"
    cell_id: str
    message: str
    timestamp: str                      # ISO 8601


# ---------------------------------------------------------------------------
# Decision / action responses
# ---------------------------------------------------------------------------

class DecisionRecord(TypedDict):
    """A logged decision from the Decide & Ask page."""
    id: str
    cell_id: str
    action: str                         # "continue" | "inspect" | "replace" | …
    confidence: str                     # "High" | "Medium" | "Low"
    soh_pct: float
    timestamp: str
    status: str                         # "Pending" | "Approved" | "Completed" | "Verified"


# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------

class TwinResponse(TypedDict, total=False):
    """GET /cells/{id}/twin — digital twin representation."""
    cell_id: str
    measured_history: list[dict]         # [{cycle, soh_pct, resistance_ohm, …}]
    soh_current: float
    fade_rate_30cy: float
    knee_cycle: int | None
    eol_projected: int | None
    physics_projection: list[dict]       # [{cycle, soh_projected, soh_lower, soh_upper}]
    last_updated: str


# ---------------------------------------------------------------------------
# Market / dispatch
# ---------------------------------------------------------------------------

class PricePoint(TypedDict):
    """One timestamped electricity price."""
    timestamp: str
    price_per_kwh: float
    currency: str                       # "EUR" | "USD" | …
    source: str                         # "synthetic" | "eia" | "entsoe"


class DispatchSchedule(TypedDict):
    """Health-aware dispatch schedule for a time window."""
    cell_id: str
    schedule: list[dict]                # [{hour, charge_kwh, discharge_kwh, soc_start, soc_end}]
    revenue_eur: float
    energy_throughput_kwh: float
    efc: float                          # Equivalent Full Cycles
    stress_delta: float                 # additional degradation vs. no-dispatch baseline


# ---------------------------------------------------------------------------
# Passport / compliance
# ---------------------------------------------------------------------------

class PassportData(TypedDict, total=False):
    """EU Battery Passport response shape."""
    cell_id: str
    document_id: str                    # deterministic hash
    available_fields: dict[str, Any]
    estimated_fields: dict[str, Any]
    unavailable_fields: list[str]
    completeness_pct: float
    qr_code_data_url: str | None
