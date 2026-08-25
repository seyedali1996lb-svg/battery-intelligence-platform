"""
Battery Intelligence Platform — FastAPI REST layer.

Exposes validated model outputs over HTTP so external systems (dashboards,
BMS firmware, ERP integrations) can consume cell data without running the
Streamlit UI.

All data endpoints are read-only. Model outputs come from the same
`load_everything()` bundle used by the Streamlit pages — no separate
training, no separate cache. Every data endpoint requires a bearer token
from POST /auth/login (see src/db.py's Organization/User model, added for
this app's multi-tenancy). The shared reference-cell bundle (NASA/
Severson/synthetic) is served to every organization; each org's own
uploaded ("My Data") fleet — persisted via bundle_cache.save_tenant_bundle()
the same way the Streamlit UI does — is additionally merged in for that
org's requests only, using the org_id already carried in its bearer token.

Usage
-----
Production (from repo root):
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Development (auto-reload):
    uvicorn src.api:app --reload --port 8000

Docker (see Dockerfile.api):
    docker build -f Dockerfile.api -t battery-api .
    docker run -p 8000:8000 battery-api

Endpoints
---------
GET  /                      — API info + version (unauthenticated)
POST /auth/login            — {username, password} -> bearer token (unauthenticated)
GET  /health                — liveness check
GET  /cells                 — list all loaded cell IDs
GET  /cells/{cell_id}       — latest cycle metrics for one cell
GET  /cells/{cell_id}/history?limit=100  — SOH / RUL history
GET  /fleet/summary         — fleet-level KPIs (SOH distribution, EOL count)
GET  /fleet/alerts          — active alert list (same logic as Alert Inbox UI)
GET  /cells/{cell_id}/rul   — RUL prediction with reliability flag + band
GET  /cells/{cell_id}/lineage — data lineage for a cell's metrics
GET  /cells/{cell_id}/view/{oem|operator|recycler} — same cell, sliced for one real stakeholder (see src/stakeholder_views.py)

All endpoints other than / and /auth/login require:
    Authorization: Bearer <token>
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path so src.* imports work
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_APP_DIR = os.path.join(_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import datetime
from typing import Any, Optional, Literal

try:
    from fastapi import FastAPI, HTTPException, Query, Depends, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as _e:
    raise ImportError(
        "FastAPI and pydantic are required for the REST layer. "
        "Install with: pip install 'fastapi[standard]' uvicorn"
    ) from _e

try:
    import jwt as _pyjwt
except ImportError as _e:
    raise ImportError("PyJWT is required for API auth. Install with: pip install PyJWT") from _e

import numpy as np

from src.db import get_user_by_username, verify_password, init_db
from src.chemistry_profiles import ChemistryProfile

# ── App metadata ──────────────────────────────────────────────────────────────

API_VERSION = "0.1.0"

app = FastAPI(
    title="Battery Intelligence Platform API",
    description=(
        "REST interface for validated SOH/RUL predictions, fleet summaries, "
        "and degradation diagnostics. All model outputs are from leave-cell-out "
        "cross-validated GBRT pipelines — the same bundle used by the Streamlit UI."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Vite's dev server default origin. No production origin configured yet —
# there's no deployed frontend target in this pass (see README).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth (JWT) ────────────────────────────────────────────────────────────────
# Same honesty pattern as this app's other secrets (see README's Production
# Readiness Roadmap "Secrets management" row). Unlike src/db.py's Fernet key,
# nothing depends on this API layer being reachable today (no deployed
# frontend target, no live traffic — see frontend/README.md) so it's safe to
# refuse to start rather than silently issue tokens signed with a secret
# that's public in this repo's source. Exempted under pytest so the test
# suite (which imports this module directly, not via a real server process)
# doesn't need its own secrets-manager setup.
if "pytest" in sys.modules:
    _JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-insecure-secret-change-in-production")
else:
    _JWT_SECRET = os.environ.get("JWT_SECRET")
    if not _JWT_SECRET:
        raise RuntimeError(
            "JWT_SECRET is not set. Refusing to start src/api.py with the "
            "insecure demo-grade fallback secret that's public in this repo's "
            "source — set JWT_SECRET in the environment (a real secrets "
            "manager for any real deployment) before running this API."
        )
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_HOURS = 24

# Idempotent — creates tables + seeds the Demo Org on first run, matching how
# app/_pages/login.py already calls this. The API may be the first process to
# touch data/app.db on a fresh clone, so this can't be lazy like _get_bundle().
init_db()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    org_id: int
    org_name: str
    role: str
    display_name: str


def _create_access_token(user: dict) -> str:
    payload = {
        "sub": user["username"],
        "org_id": user["org_id"],
        "role": user["role"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=_JWT_EXPIRY_HOURS),
    }
    return _pyjwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency — decodes the `Authorization: Bearer <token>` header
    issued by POST /auth/login. Raises 401 on any missing/invalid/expired
    token so every gated endpoint below fails closed, not open.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = _pyjwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except _pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except _pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return {"username": payload["sub"], "org_id": payload["org_id"], "role": payload["role"]}


# ── Bundle loading (lazy, singleton) ─────────────────────────────────────────

_bundle_cache: dict | None = None


def _get_bundle() -> dict:
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache

    try:
        # Import the Streamlit app's loader — works from CLI too
        from main import load_everything  # type: ignore
        _bundle_cache = load_everything()
    except ImportError:
        _bundle_cache = _load_bundle_from_disk_cache()
    return _bundle_cache


def _load_bundle_from_disk_cache() -> dict:
    """
    Fallback for _get_bundle() when `main` (the Streamlit app) can't be
    imported — e.g. running as a bare `uvicorn src.api:app` process with no
    Streamlit context. There's no live battery_dict available here to
    rebuild a cache signature against, so this reads whatever's already on
    disk in the app's own bundle cache via bundle_cache.load_cached_unchecked()
    (no signature check — see that function's docstring for why), rather than
    the nonexistent `load_bundle()` this previously called (a bare ImportError
    from the missing name, uncaught, since it fired from inside an
    `except ImportError:` block that had already stopped trying to catch it).

    Returns the same {"featured_dfs": ..., "bundles": ...} shape
    _get_featured_dfs()/_get_bundles() already expect from a non-tuple
    _get_bundle() result — previously this returned {"severson": ...,
    "nasa": ...}, which doesn't have either key, so even a successful
    fallback would have silently served zero cells to every endpoint.

    bundle_cache's "nasa"/"severson"/"synth" keys now cache (bundle,
    split_cycles) pairs, not (bundle, featured_dfs, split_cycles) triples
    (see src/cell_store.py / src/bundle_cache.py's module docstrings) —
    featured_dfs is reconstructed here as a lazy, LRU-bounded
    cell_store.LazyCellFrameMap over the cell IDs split_cycles' keys give
    us, same as app/main.py's load_everything() itself now returns, rather
    than a real dict of every cell's full DataFrame.
    """
    try:
        from src.bundle_cache import load_cached_unchecked  # type: ignore
    except ImportError:
        from bundle_cache import load_cached_unchecked  # type: ignore
    try:
        from src.cell_store import LazyCellFrameMap  # type: ignore
    except ImportError:
        from cell_store import LazyCellFrameMap  # type: ignore

    cell_ids: list = []
    bundles: dict = {}
    for key in ("nasa", "severson", "synth"):
        cached = load_cached_unchecked(key)
        if cached is None:
            continue
        bundle, split_cycles = cached
        bundles[key] = bundle
        cell_ids.extend(split_cycles.keys())

    return {"featured_dfs": LazyCellFrameMap(cell_ids), "bundles": bundles}


def _get_org_bundle(org_id: "int | None"):
    """Return the org's own persisted (featured_dfs, bundle, split_cycles)
    triple via bundle_cache.load_tenant_bundle() — the exact same function
    the Streamlit "My Data" mode uses — or None if org_id is absent or the
    org hasn't uploaded anything yet. Loaded fresh per call (no caching)
    since different requests may belong to different orgs."""
    if org_id is None:
        return None
    try:
        from src.bundle_cache import load_tenant_bundle  # type: ignore
    except ImportError:
        from bundle_cache import load_tenant_bundle  # type: ignore
    return load_tenant_bundle(org_id)


def _get_featured_dfs(org_id: "int | None" = None) -> dict:
    b = _get_bundle()
    # load_everything() returns (featured_dfs, bundles, split_cycles) — see
    # app/main.py's load_everything(). A dict-shaped fallback (from the
    # ImportError branch in _get_bundle()) has no featured_dfs of its own.
    fdfs = b[0] if isinstance(b, tuple) else b.get("featured_dfs", {})

    org_bundle = _get_org_bundle(org_id)
    if org_bundle is not None:
        org_fdfs = org_bundle[0]
        fdfs = {**fdfs, **org_fdfs}
    return fdfs


def _get_bundles(org_id: "int | None" = None) -> dict:
    b = _get_bundle()
    bundles = b[1] if isinstance(b, tuple) else b.get("bundles", {})

    org_bundle = _get_org_bundle(org_id)
    if org_bundle is not None:
        bundles = {**bundles, "upload": org_bundle[1]}
    return bundles


def _cell_stat_rows(org_id: "int | None") -> list[dict]:
    """Normalized per-cell stats for fleet_summary()/fleet_alerts(): the
    shared reference fleet reads precomputed CellSummary rows (src/db.py --
    see src/cell_store.py's module docstring) instead of every cell's full
    per-cycle DataFrame; an org's own uploaded cells (never persisted to
    CellSummary -- see cell_store.py's "explicitly out of scope" note) are
    computed the same way this module always has, from the small,
    session-bounded featured_dfs that load_tenant_bundle() returns. Both
    are merged into one list with the same field names so both endpoints
    below can iterate a single, uniform source."""
    try:
        import db as db_module  # type: ignore
    except ImportError:
        from src import db as db_module  # type: ignore
    from experiment_registry import PLATFORM_ORG_ID

    if org_id is not None:
        summaries = db_module.get_cell_summaries(org_id, include_platform=True)
    else:
        summaries = db_module.get_cell_summaries(PLATFORM_ORG_ID, include_platform=False)

    rows = [
        {
            "cell_id": r["cell_id"],
            "soh_pct": r["soh_pct"],
            "cycle_number": r["cycle_number"],
            "eol_at_cycle": r["eol_at_cycle"],
            "fade_trend": r["fade_trend"],
        }
        for r in summaries
        if r.get("soh_pct") is not None
    ]

    org_bundle = _get_org_bundle(org_id)
    if org_bundle is not None:
        org_fdfs = org_bundle[0]
        for cell_id, df in org_fdfs.items():
            if len(df) == 0 or "soh_pct" not in df.columns:
                continue
            latest = df.iloc[-1]
            eol_rows = df[df["is_eol"]] if "is_eol" in df.columns else df[df["soh_pct"] < 80.0]
            eol_at = int(eol_rows["cycle_number"].iloc[0]) if len(eol_rows) else None
            fade_trend = "Stable"
            if "fade_rate_30cy" in df.columns and len(df) >= 31:
                fade_now = df["fade_rate_30cy"].iloc[-1]
                fade_prev = df["fade_rate_30cy"].iloc[-31]
                delta_pct = (fade_now - fade_prev) / (abs(fade_prev) + 1e-9) * 100
                fade_trend = "Accelerating" if delta_pct > 20 else ("Decelerating" if delta_pct < -20 else "Stable")
            rows.append({
                "cell_id": cell_id,
                "soh_pct": float(latest["soh_pct"]),
                "cycle_number": int(latest["cycle_number"]),
                "eol_at_cycle": eol_at,
                "fade_trend": fade_trend,
            })

    return rows


# ── Response models ────────────────────────────────────────────────────────────

class APIInfo(BaseModel):
    name: str
    version: str
    docs: str
    endpoints: list[str]


class HealthCheck(BaseModel):
    status: str
    cells_loaded: int
    bundles_loaded: list[str]


class CellLatest(BaseModel):
    cell_id: str
    cycle_number: int
    soh_pct: float
    status: str
    rul_pred: Optional[float]
    rul_reliable: bool
    fade_rate_30cy: Optional[float]
    capacity_ah: Optional[float]
    resistance_normalized: Optional[float]
    coulombic_efficiency: Optional[float]


class HistoryPoint(BaseModel):
    cycle_number: int
    soh_pct: float
    rul_pred: Optional[float]
    capacity_ah: Optional[float]


class FleetSummary(BaseModel):
    total_cells: int
    n_healthy: int
    n_degrading: int
    n_eol: int
    fleet_soh_mean: float
    fleet_soh_min: float
    fleet_soh_max: float
    cells_near_eol_3m: int
    cells_near_eol_6m: int
    capex_estimate_12m_usd: float


class AlertItem(BaseModel):
    severity: str          # "critical" | "high" | "medium"
    cell_id: str
    title: str
    body: str


class RULResult(BaseModel):
    cell_id: str
    cycle_number: int
    soh_pct: float
    rul_pred: Optional[float]
    rul_q10: Optional[float]
    rul_q90: Optional[float]
    rul_reliable: bool
    eol_threshold_pct: float


class LineageRow(BaseModel):
    metric: str
    latest_value: str
    valid_cycles: Any
    formula: str
    source: str


# ── Helpers ───────────────────────────────────────────────────────────────────

_CYCLES_PER_MONTH = 200
_REPLACEMENT_COST_USD = 150


def _soh_status(soh: float) -> str:
    if soh >= 90:
        return "Healthy"
    if soh >= 80:
        return "Degrading"
    return "End of Life"


def _rul_reliable_for(cell_id: str, bundles: dict) -> bool:
    _kind = ChemistryProfile.for_cell(cell_id).source_kind
    if _kind == "nasa":
        bndl = bundles.get("nasa", {})
    elif _kind == "severson":
        bndl = bundles.get("severson", {})
    else:
        # "synth" and "upload" are both real keys once an org has uploaded
        # data, so a plain dict.get("synth", dict.get("upload", {})) fallback
        # would always pick "synth" first even for a cell that's only in the
        # org's own upload bundle — pick whichever one actually lists this
        # cell in its per-cell reliability map instead of guessing by key order.
        for _key in ("synth", "upload"):
            _candidate = bundles.get(_key, {})
            if cell_id in _candidate.get("metrics", {}).get("per_cell_rul_reliable", {}):
                bndl = _candidate
                break
        else:
            bndl = bundles.get("upload") or bundles.get("synth", {})
    if not bndl:
        return False
    per_cell = bndl.get("metrics", {}).get("per_cell_rul_reliable", {})
    return per_cell.get(cell_id, bndl.get("metrics", {}).get("rul_reliable", False))


def _cell_not_found(cell_id: str):
    raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found in loaded bundle.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=APIInfo, summary="API info")
def root():
    return APIInfo(
        name="Battery Intelligence Platform API",
        version=API_VERSION,
        docs="/docs",
        endpoints=[
            "POST /auth/login",
            "GET /health (auth required)",
            "GET /cells (auth required)",
            "GET /cells/{cell_id}",
            "GET /cells/{cell_id}/history",
            "GET /cells/{cell_id}/rul",
            "GET /cells/{cell_id}/lineage",
            "GET /cells/{cell_id}/view/{oem|operator|recycler}",
            "GET /fleet/summary",
            "GET /fleet/alerts",
        ],
    )


@app.post("/auth/login", response_model=LoginResponse, summary="Log in and receive a bearer token")
def login(body: LoginRequest):
    user = get_user_by_username(body.username)
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return LoginResponse(
        access_token=_create_access_token(user),
        org_id=user["org_id"],
        org_name=user["org_name"],
        role=user["role"],
        display_name=user["display_name"],
    )


@app.get("/health", response_model=HealthCheck, summary="Liveness check")
def health(current_user: dict = Depends(get_current_user)):
    try:
        fdfs = _get_featured_dfs(current_user["org_id"])
        bundles = _get_bundles(current_user["org_id"])
        return HealthCheck(
            status="ok",
            cells_loaded=len(fdfs),
            bundles_loaded=list(bundles.keys()),
        )
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/cells", summary="List all cell IDs")
def list_cells(current_user: dict = Depends(get_current_user)) -> dict:
    fdfs = _get_featured_dfs(current_user["org_id"])
    return {"cells": sorted(fdfs.keys()), "count": len(fdfs)}


@app.get("/cells/{cell_id}", response_model=CellLatest, summary="Latest metrics for one cell")
def cell_latest(cell_id: str, current_user: dict = Depends(get_current_user)):
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    bundles = _get_bundles(current_user["org_id"])
    latest = df.iloc[-1]
    rul_ok = _rul_reliable_for(cell_id, bundles)

    def _safe(col: str) -> Optional[float]:
        v = latest.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    return CellLatest(
        cell_id=cell_id,
        cycle_number=int(latest["cycle_number"]),
        soh_pct=float(latest["soh_pct"]),
        status=_soh_status(float(latest["soh_pct"])),
        rul_pred=_safe("rul_pred") if rul_ok else None,
        rul_reliable=rul_ok,
        fade_rate_30cy=_safe("fade_rate_30cy"),
        capacity_ah=_safe("capacity_ah"),
        resistance_normalized=_safe("resistance_normalized"),
        coulombic_efficiency=_safe("coulombic_efficiency"),
    )


@app.get("/cells/{cell_id}/history", summary="SOH/RUL history for one cell")
def cell_history(
    cell_id: str,
    limit: int = Query(200, ge=1, le=2000, description="Max rows returned (newest last)"),
    current_user: dict = Depends(get_current_user),
) -> dict:
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    bundles = _get_bundles(current_user["org_id"])
    rul_ok = _rul_reliable_for(cell_id, bundles)

    # Return last `limit` cycles
    sub = df.tail(limit)[["cycle_number", "soh_pct", "capacity_ah"]].copy()
    if "rul_pred" in df.columns and rul_ok:
        sub["rul_pred"] = df.tail(limit)["rul_pred"]
    else:
        sub["rul_pred"] = None

    points = []
    for _, row in sub.iterrows():
        def _sf(col):
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        points.append(HistoryPoint(
            cycle_number=int(row["cycle_number"]),
            soh_pct=float(row["soh_pct"]),
            rul_pred=_sf("rul_pred"),
            capacity_ah=_sf("capacity_ah"),
        ))

    return {
        "cell_id": cell_id,
        "total_cycles": len(df),
        "returned": len(points),
        "rul_reliable": rul_ok,
        "history": [p.model_dump() for p in points],
    }


@app.get("/cells/{cell_id}/rul", response_model=RULResult, summary="RUL prediction with uncertainty band")
def cell_rul(
    cell_id: str,
    eol_threshold: float = Query(80.0, ge=60.0, le=95.0, description="EOL SOH threshold %"),
    current_user: dict = Depends(get_current_user),
):
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    bundles = _get_bundles(current_user["org_id"])
    rul_ok = _rul_reliable_for(cell_id, bundles)
    latest = df.iloc[-1]

    def _sf(col):
        v = latest.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    return RULResult(
        cell_id=cell_id,
        cycle_number=int(latest["cycle_number"]),
        soh_pct=float(latest["soh_pct"]),
        rul_pred=_sf("rul_pred") if rul_ok else None,
        rul_q10=_sf("rul_q10") if rul_ok else None,
        rul_q90=_sf("rul_q90") if rul_ok else None,
        rul_reliable=rul_ok,
        eol_threshold_pct=eol_threshold,
    )


@app.get("/cells/{cell_id}/lineage", summary="Data lineage — source of every metric")
def cell_lineage(cell_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]

    _col_map = {
        "soh_pct":               ("SOH %",                 "measured capacity / initial capacity × 100",        "Derived from capacity_ah"),
        "capacity_ah":           ("Capacity (Ah)",          "raw discharge capacity per cycle",                  "Direct measurement from cycler"),
        "fade_rate_30cy":        ("Fade Rate (30cy)",       "ΔQ / Δcycle over rolling 30-cycle window",          "pandas rolling().apply()"),
        "coulombic_efficiency":  ("Coulombic Efficiency",   "Q_discharge / Q_charge per cycle",                  "Direct measurement from cycler"),
        "resistance_normalized": ("Resistance (norm.)",     "R(n) / R(1) — normalised internal resistance",      "Derived from voltage step"),
        "rul_pred":              ("RUL Prediction",         "GBRT GradientBoostingRegressor output",             "ML model, leave-cell-out validated"),
        "rul_q10":               ("RUL Q10",                "10th percentile quantile regression",               "ML model quantile regressor"),
        "rul_q90":               ("RUL Q90",                "90th percentile quantile regression",               "ML model quantile regressor"),
        "ce_rolling_30cy":       ("CE Rolling (30cy)",      "30-cycle rolling mean of coulombic_efficiency",     "pandas rolling().mean()"),
        "dqdv_sim_peak_value":   ("dQ/dV Peak (sim)",       "simulated via LiCoO₂ OCV polynomial model",         "simulate_vq_curve() — physics simulation"),
        "sop_pct":               ("SOP %",                  "State of Power — rate capability proxy",            "Derived from resistance_normalized"),
    }

    rows = []
    for col, (label, formula, source) in _col_map.items():
        avail = col in df.columns
        n_valid = int(df[col].notna().sum()) if avail else 0
        latest_val = None
        if avail and n_valid > 0:
            v = df[col].dropna().iloc[-1]
            latest_val = round(float(v), 6)
        rows.append(LineageRow(
            metric=label,
            latest_value=str(latest_val) if latest_val is not None else "—",
            valid_cycles=n_valid if avail else "N/A",
            formula=formula,
            source=source,
        ))

    return {
        "cell_id": cell_id,
        "total_cycles": len(df),
        "first_cycle": int(df["cycle_number"].iloc[0]),
        "last_cycle": int(df["cycle_number"].iloc[-1]),
        "lineage": [r.model_dump() for r in rows],
    }


@app.get(
    "/cells/{cell_id}/view/{stakeholder}",
    summary="Stakeholder-sliced view of one cell (OEM / operator / recycler)",
)
def cell_stakeholder_view(
    cell_id: str,
    stakeholder: Literal["oem", "operator", "recycler"],
    region: Optional[str] = Query(None, description="Recycler-only: preferred region (North America/Europe/Asia)"),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    The real, external-facing mechanism behind src/stakeholder_views.py's
    module docstring: the same cell's data, gated by this endpoint's own
    JWT auth like every other route here, sliced for whichever real party
    (manufacturer, current operator, or recycler) is asking. No new
    login system, no new role -- a real integration partner authenticates
    the same way any other API consumer does (POST /auth/login).
    """
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    bundles = _get_bundles(current_user["org_id"])
    rul_ok = _rul_reliable_for(cell_id, bundles)
    latest = df.iloc[-1]

    def _sf(col):
        v = latest.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    profile = ChemistryProfile.for_cell(cell_id)
    soh = float(latest["soh_pct"])
    fade_30 = _sf("fade_rate_30cy") or 0.0
    sop_pct = _sf("sop_pct")

    from stakeholder_views import build_oem_view, build_operator_view, build_recycler_view
    from recommendations import diagnose_mechanism

    if stakeholder == "oem":
        fields = build_oem_view(
            cell_id, profile.short_name, profile.source_kind, soh, int(latest["cycle_number"]), fade_30,
            mechanism=diagnose_mechanism(df), rul_reliable=rul_ok,
            rul_pred=_sf("rul_pred") if rul_ok else None,
            rul_q10=_sf("rul_q10") if rul_ok else None,
            rul_q90=_sf("rul_q90") if rul_ok else None,
        )
    elif stakeholder == "operator":
        fields = build_operator_view(
            cell_id, profile.short_name, profile.source_kind, soh, int(latest["cycle_number"]),
            fade_30, _sf("fade_rate_50cy") or fade_30, None,
            rul_reliable=rul_ok,
            rul_pred=_sf("rul_pred") if rul_ok else None,
            rul_q10=_sf("rul_q10") if rul_ok else None,
            rul_q90=_sf("rul_q90") if rul_ok else None,
            sop_pct=sop_pct,
        )
    else:
        fields = build_recycler_view(cell_id, profile.short_name, soh, fade_30, sop_pct=sop_pct, user_region=region)

    return {"cell_id": cell_id, "stakeholder": stakeholder, "fields": fields}


@app.get("/fleet/summary", response_model=FleetSummary, summary="Fleet-level KPIs")
def fleet_summary(current_user: dict = Depends(get_current_user)):
    rows = _cell_stat_rows(current_user["org_id"])
    if not rows:
        raise HTTPException(status_code=503, detail="No cells loaded.")

    sohs = []
    n_healthy = n_degrading = n_eol = 0
    eol_3m = eol_6m = eol_12m = 0

    for r in rows:
        soh = float(r["soh_pct"])
        sohs.append(soh)
        status = _soh_status(soh)
        if status == "Healthy":
            n_healthy += 1
        elif status == "Degrading":
            n_degrading += 1
        else:
            n_eol += 1

        # Cycles to EOL
        eol_at = r["eol_at_cycle"]
        if eol_at is not None:
            current_cy = int(r["cycle_number"])
            cycles_left = max(0, eol_at - current_cy)
            if cycles_left <= 3 * _CYCLES_PER_MONTH:
                eol_3m += 1
            if cycles_left <= 6 * _CYCLES_PER_MONTH:
                eol_6m += 1
            if cycles_left <= 12 * _CYCLES_PER_MONTH:
                eol_12m += 1

    return FleetSummary(
        total_cells=len(rows),
        n_healthy=n_healthy,
        n_degrading=n_degrading,
        n_eol=n_eol,
        fleet_soh_mean=round(float(np.mean(sohs)), 2),
        fleet_soh_min=round(float(np.min(sohs)), 2),
        fleet_soh_max=round(float(np.max(sohs)), 2),
        cells_near_eol_3m=eol_3m,
        cells_near_eol_6m=eol_6m,
        capex_estimate_12m_usd=eol_12m * _REPLACEMENT_COST_USD,
    )


@app.get("/fleet/alerts", summary="Active alert list — same logic as Alert Inbox UI")
def fleet_alerts(current_user: dict = Depends(get_current_user)) -> dict:
    rows = _cell_stat_rows(current_user["org_id"])
    if not rows:
        raise HTTPException(status_code=503, detail="No cells loaded.")

    alerts: list[AlertItem] = []

    for r in rows:
        cell_id = r["cell_id"]
        soh = float(r["soh_pct"])
        status = _soh_status(soh)

        if status == "End of Life":
            alerts.append(AlertItem(
                severity="critical", cell_id=cell_id,
                title="End of Life",
                body=f"SOH {soh:.1f}% — replace or reassign to second-life application immediately.",
            ))

        # Cycles to EOL
        eol_at = r["eol_at_cycle"]
        cycles_to_eol = max(0, eol_at - int(r["cycle_number"])) if eol_at is not None else None

        if cycles_to_eol is not None and cycles_to_eol <= 3 * _CYCLES_PER_MONTH and status != "End of Life":
            alerts.append(AlertItem(
                severity="high", cell_id=cell_id,
                title="Replacement Due Within 3 Months",
                body=f"{cycles_to_eol} cycles remaining — SOH {soh:.1f}%.",
            ))

        # Fade acceleration -- reads the same precomputed classification
        # Fleet's own ranking table does (src/cell_store.py's build_summary()),
        # rather than recomputing the >20%-vs-30-cycles-ago check here.
        if r["fade_trend"] == "Accelerating" and status != "End of Life":
            alerts.append(AlertItem(
                severity="medium", cell_id=cell_id,
                title="Fade Rate Accelerating",
                body=f"30-cycle fade rate has accelerated vs the prior window. SOH {soh:.1f}%.",
            ))

    _SEV_ORDER = {"critical": 0, "high": 1, "medium": 2}
    alerts.sort(key=lambda a: _SEV_ORDER[a.severity])

    return {
        "total_alerts": len(alerts),
        "critical": sum(1 for a in alerts if a.severity == "critical"),
        "high":     sum(1 for a in alerts if a.severity == "high"),
        "medium":   sum(1 for a in alerts if a.severity == "medium"),
        "alerts":   [a.model_dump() for a in alerts],
    }


# ── Action Center & Operations Triage Endpoints ──────────────────────────────

from src.action_center import action_center
from batlab.features.partial_cycles import rainflow_counting, reconstruct_ocv_relaxation, partial_ica_analysis
from batlab.datasets.cycler_mapper import detect_cycler_format, ingest_cycler_data
from src.pinn_model import BatteryPINNEstimator
from src.dynamic_circularity import calculate_dynamic_lca, generate_verifiable_credential_passport, match_second_life_bids
from src.task_queue import task_queue
from src.streaming_analytics import streaming_engine


class ActionCreateRequest(BaseModel):
    cell_id: str
    title: str
    category: str = "DEGRADATION"
    severity: str = "HIGH"
    description: str
    recommended_action: str
    soh_pct: float
    sla_hours: int = 24


class ActionTriageRequest(BaseModel):
    status: str
    assigned_to: Optional[str] = None


class ActionDispatchRequest(BaseModel):
    target_system: str
    payload: Optional[dict] = None


class RainflowRequest(BaseModel):
    soc_series: list[float]
    time_series: Optional[list[float]] = None


class OCVRequest(BaseModel):
    time_sec: list[float]
    voltage_v: list[float]
    current_a: list[float]


class PartialICARequest(BaseModel):
    voltage_v: list[float]
    capacity_ah: list[float]
    v_min: float = 3.2
    v_max: float = 4.1


class PINNEstimateRequest(BaseModel):
    cycles: list[float]
    soh_pct: list[float]
    future_cycles_count: int = 1000
    temperature_c: float = 25.0
    c_rate: float = 1.0


class StreamingReadingRequest(BaseModel):
    cell_id: str
    voltage_v: float
    current_a: float
    temperature_c: float
    timestamp_s: Optional[float] = None


@app.get("/actions", summary="List operational action center tickets")
def list_action_tickets(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    return action_center.list_actions(
        org_id=current_user["org_id"],
        severity=severity,
        status=status,
        category=category,
    )


@app.post("/actions", summary="Create an operational action ticket")
def create_action_ticket(
    req: ActionCreateRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return action_center.create_action(
        cell_id=req.cell_id,
        title=req.title,
        category=req.category,
        severity=req.severity,
        description=req.description,
        recommended_action=req.recommended_action,
        soh_pct=req.soh_pct,
        org_id=current_user["org_id"],
        sla_hours=req.sla_hours,
    )


@app.post("/actions/{action_id}/triage", summary="Triage an action ticket")
def triage_action_ticket(
    action_id: str,
    req: ActionTriageRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    try:
        return action_center.triage_action(
            action_id=action_id,
            new_status=req.status,
            assigned_to=req.assigned_to,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/actions/{action_id}/dispatch", summary="Dispatch an action workflow to CMMS/Warranty/Circularity")
def dispatch_action_ticket(
    action_id: str,
    req: ActionDispatchRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    try:
        return action_center.dispatch_workflow(
            action_id=action_id,
            target_system=req.target_system,
            payload=req.payload,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Partial Cycles & Field Telemetry Analytics ───────────────────────────────

@app.post("/analytics/rainflow", summary="ASTM E1049-85 Rainflow cycle counting")
def run_rainflow_analysis(
    req: RainflowRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return rainflow_counting(
        soc_series=np.array(req.soc_series),
        time_series=np.array(req.time_series) if req.time_series else None,
    )


@app.post("/analytics/ocv-reconstruct", summary="Reconstruct OCV and R0 from rest interval")
def run_ocv_reconstruction(
    req: OCVRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return reconstruct_ocv_relaxation(
        time_sec=np.array(req.time_sec),
        voltage_v=np.array(req.voltage_v),
        current_a=np.array(req.current_a),
    )


@app.post("/analytics/partial-ica", summary="Partial window Incremental Capacity Analysis (dQ/dV)")
def run_partial_ica(
    req: PartialICARequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return partial_ica_analysis(
        voltage_v=np.array(req.voltage_v),
        capacity_ah=np.array(req.capacity_ah),
        v_min=req.v_min,
        v_max=req.v_max,
    )


# ── Universal Cycler Ingestion Helpers ───────────────────────────────────────

@app.post("/ingest/cycler-detect", summary="Detect cycler format and map columns")
def detect_cycler(
    columns: list[str],
    current_user: dict = Depends(get_current_user),
) -> dict:
    return detect_cycler_format(columns)


# ── Physics & PINN Modeling ──────────────────────────────────────────────────

@app.post("/physics/pinn-estimate", summary="Hybrid PINN LLI/LAM degradation estimation")
def run_pinn_estimation(
    req: PINNEstimateRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    estimator = BatteryPINNEstimator()
    estimator.fit(np.array(req.cycles), np.array(req.soh_pct))
    
    current_cy = int(req.cycles[-1]) if req.cycles else 100
    future_cy = np.arange(1, current_cy + req.future_cycles_count, 1)
    preds = estimator.predict(future_cy, temperature_c=req.temperature_c, c_rate=req.c_rate)
    rul = estimator.estimate_rul(current_cycle=current_cy, temperature_c=req.temperature_c, c_rate=req.c_rate)
    
    return {
        "rul_estimation": rul,
        "parameters": estimator.params,
        "sample_points_count": len(future_cy),
        "latest_soh_pct": float(preds["soh_pct"][-1]),
        "latest_knee_risk_score": float(preds["knee_risk_score"][-1]),
    }


# ── Dynamic LCA, Circularity & Verifiable Credentials ─────────────────────────

@app.get("/cells/{cell_id}/dynamic-lca", summary="Calculate cradle-to-grave dynamic CO2e footprint")
def get_cell_dynamic_lca(
    cell_id: str,
    region: str = "EU_AVG",
    current_user: dict = Depends(get_current_user),
) -> dict:
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    latest = df.iloc[-1]
    profile = ChemistryProfile.for_cell(cell_id)
    chem = getattr(profile, "cathode_name", "LFP") if profile else "LFP"
    nominal_kwh = (float(latest.get("capacity_ah", 2.0)) * 3.3) / 1000.0
    throughput = float(len(df) * max(nominal_kwh, 0.005) * 0.8)
    return calculate_dynamic_lca(
        cell_id=cell_id,
        chemistry=chem,
        nominal_kwh=nominal_kwh,
        cumulative_throughput_kwh=throughput,
        region=region,
    )


@app.get("/cells/{cell_id}/verifiable-passport", summary="W3C Verifiable Credential EU Battery Passport")
def get_verifiable_passport(
    cell_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    latest = df.iloc[-1]
    profile = ChemistryProfile.for_cell(cell_id)
    chem = getattr(profile, "cathode_name", "LFP") if profile else "LFP"
    soh = float(latest.get("soh_pct", 80.0))
    rul = int(latest.get("rul_pred", 500)) if not np.isnan(latest.get("rul_pred", 500)) else 500
    res = float(latest.get("resistance_ohm", 0.025)) if not np.isnan(latest.get("resistance_ohm", 0.025)) else 0.025
    nominal_kwh = (float(latest.get("capacity_ah", 2.0)) * 3.3) / 1000.0
    carbon = calculate_dynamic_lca(cell_id, chem, nominal_kwh, 15.0)

    return generate_verifiable_credential_passport(
        cell_id=cell_id,
        org_id=str(current_user["org_id"]),
        chemistry=chem,
        soh_pct=soh,
        rul_cycles=rul,
        resistance_ohm=res,
        carbon_data=carbon,
    )


@app.get("/cells/{cell_id}/second-life-bids", summary="Matched second-life buyer bids and valuations")
def get_second_life_bids(
    cell_id: str,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    fdfs = _get_featured_dfs(current_user["org_id"])
    if cell_id not in fdfs:
        _cell_not_found(cell_id)
    df = fdfs[cell_id]
    latest = df.iloc[-1]
    profile = ChemistryProfile.for_cell(cell_id)
    chem = getattr(profile, "cathode_name", "LFP") if profile else "LFP"
    soh = float(latest.get("soh_pct", 80.0))
    res_norm = float(latest.get("resistance_normalized", 1.2)) if not np.isnan(latest.get("resistance_normalized", 1.2)) else 1.2
    res_growth = res_norm * 100.0
    nominal_kwh = (float(latest.get("capacity_ah", 2.0)) * 3.3) / 1000.0

    return match_second_life_bids(
        cell_id=cell_id,
        chemistry=chem,
        soh_pct=soh,
        resistance_growth_pct=res_growth,
        nominal_kwh=nominal_kwh,
    )



# ── Asynchronous Task Queue ──────────────────────────────────────────────────

@app.get("/tasks", summary="List async analytics tasks")
def list_tasks(current_user: dict = Depends(get_current_user)) -> list[dict]:
    return task_queue.list_tasks(org_id=current_user["org_id"])


@app.get("/tasks/{task_id}", summary="Get async task status")
def get_task_status(task_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    t = task_queue.get_task(task_id)
    if not t or t.get("org_id") != current_user["org_id"]:
        raise HTTPException(status_code=404, detail="Task not found.")
    return t


# ── Real-Time Streaming Telemetry Anomaly Detection ──────────────────────────

@app.post("/telemetry/process", summary="Process high-frequency streaming BMS sample")
def process_streaming_sample(
    req: StreamingReadingRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return streaming_engine.process_reading(
        cell_id=req.cell_id,
        voltage_v=req.voltage_v,
        current_a=req.current_a,
        temperature_c=req.temperature_c,
        timestamp_s=req.timestamp_s,
    )

