"""
Regime-based forecast routing — which model answers "what happens next?"

Why this exists
---------------
The platform currently always lets the GBRT answer, then discloses after the
fact (validity banner, per-cell reliability caption) when the number is
thin. But the platform already MEASURES per regime where the GBRT cannot be
trusted: `domain_validity.regime_reliability()` verdicts every chemistry ×
temperature × SOH-stage regime reliable / thin / unvalidated from the same
leave-cell-out folds that train it, and the prospective split shows the GBRT
collapsing below a straight line once denied the future.

This module makes that measurement DECISIONAL instead of decorative: per
cell, route the forecast to

  - "gbrt"         — the cell's regime is reliable/thin with a measured,
                     floor-passing fold RUL R²: the interpolator earns its
                     keep and its conformally calibrated interval.
  - "hierarchical" — the GBRT's RUL is not evaluable or below floor for
                     this cell's regime (the Severson situation: zero
                     observed-EOL rows), but the hierarchical partial-
                     pooling model produces a measured per-cell forecast
                     (its own leave-cell-out SOH R² exists and clears 0).
                     Serve its posterior, disclose the uncalibrated
                     interval.
  - "refuse"       — neither model can back a number in this regime. SOH
                     status continues from measured data; RUL is withheld
                     with the reason stated at the point of decision.

The routing verdict travels with `reason` — a human sentence naming the
regime row and the model that answered — so the decision surface shows WHY
this cell got this model, not just which one.
"""

from __future__ import annotations

ROUTE_GBRT = "gbrt"
ROUTE_HIERARCHICAL = "hierarchical"
ROUTE_REFUSE = "refuse"

from batlab.validation.lco import RUL_RELIABLE_FLOOR


def _latest_row(df):
    """The last row of a featured df, or None."""
    try:
        if df is not None and len(df):
            return df.iloc[-1]
    except Exception:
        pass
    return None


def _hier_per_cell(bundle) -> dict:
    """The hierarchical validation per-cell block on a bundle, or {}."""
    m = (bundle or {}).get("metrics") if isinstance(bundle, dict) else None
    hv = (m or {}).get("hierarchical_validation") if isinstance(m, dict) else None
    per_cell = (hv or {}).get("per_cell") if isinstance(hv, dict) else None
    return per_cell if isinstance(per_cell, dict) else {}


def route_forecast_for_cell(
    cell_id: str,
    bundles: dict,
    featured_dfs: "dict | None" = None,
    cell_df: "object | None" = None,
) -> dict:
    """Route one cell's "what happens next" forecast.

    Args:
        cell_id: the cell to answer for.
        bundles: the {key: bundle} map (model_selection's shape) — routing
            reads each bundle's regime_reliability, per-cell folds and
            hierarchical_validation metrics, never retrains.
        featured_dfs: optional {cell_id: featured df} — the served-column
            source for routed numbers (rul_pred / rul_q10_hier / ...).
        cell_df: optional raw per-cycle df for the cell (regime fallback).

    Returns
    -------
    {
      "served":      "gbrt" | "hierarchical" | "refuse",
      "reason":      str,   # human sentence, safe to show verbatim
      "model_label": str|None,
      "rul_pred":    float|None,
      "rul_q10":     float|None,
      "rul_q90":     float|None,
      "interval_convention": str|None,  # hierarchical: posterior, uncalibrated
      "regime":      dict|None,         # the regime row the verdict came from
      "selection":   dict,              # model_selection.select_model_for_cell record
    }
    """
    from model_selection import select_model_for_cell
    from domain_validity import cell_regime
    from chemistry_profiles import ChemistryProfile

    selection = select_model_for_cell(cell_id, bundles or {})
    bundle = selection.get("bundle") or {}
    metrics = (bundle.get("metrics") or {}) if isinstance(bundle, dict) else {}

    def _refuse(reason: str) -> dict:
        return {
            "served": ROUTE_REFUSE, "reason": reason, "model_label": None,
            "rul_pred": None, "rul_q10": None, "rul_q90": None,
            "interval_convention": None, "regime": None, "selection": selection,
        }

    if not selection.get("found"):
        return _refuse(selection.get("reason") or
                       "No chemistry-compatible model is available — predictions are withheld.")

    # ── The cell's regime and the GBRT's per-regime verdict there ──
    regime = cell_regime(cell_id, cell_data={cell_id: cell_df} if cell_df is not None else None)
    regime_rows = metrics.get("regime_reliability") or []
    reg_row = None
    for r in regime_rows:
        if (
            (r.get("chemistry") or "?") == (regime.get("chemistry") or "?")
            and (r.get("temp_band") or "unmeasured") == (regime.get("temp_band") or "unmeasured")
            and (r.get("soh_stage") or "unmeasured") == (regime.get("soh_stage") or "unmeasured")
        ):
            reg_row = r
            break

    per_cell = metrics.get("lco_per_cell") or {}
    fold = per_cell.get(cell_id) or {}
    fold_rul_r2 = fold.get("rul_r2")
    gbrt_fold_ok = fold_rul_r2 is not None and fold_rul_r2 >= RUL_RELIABLE_FLOOR

    # ── Route 1: GBRT — its own fold passes the floor ──
    if gbrt_fold_ok:
        latest = _latest_row((featured_dfs or {}).get(cell_id))
        rul_pred = float(latest["rul_pred"]) if latest is not None and latest.get("rul_pred") is not None else None
        return {
            "served": ROUTE_GBRT,
            "reason": (
                f"GBRT answers: this cell's leave-cell-out fold R²={fold_rul_r2:.2f} "
                f"clears the {RUL_RELIABLE_FLOOR:.2f} floor"
                + (f" (regime '{reg_row.get('soh_stage')}' verdict: {reg_row.get('verdict')})" if reg_row else "")
                + ", and its served interval is conformally calibrated on unseen cells."
            ),
            "model_label": selection.get("model_label"),
            "rul_pred": rul_pred,
            "rul_q10": float(latest["rul_q10"]) if latest is not None and latest.get("rul_q10") is not None else None,
            "rul_q90": float(latest["rul_q90"]) if latest is not None and latest.get("rul_q90") is not None else None,
            "interval_convention": "conformally calibrated Q10/Q90 (coverage measured leave-cell-out)",
            "regime": reg_row,
            "selection": selection,
        }

    # ── Route 2: hierarchical — GBRT RUL unusable here, hierarchical has a
    #    measured per-cell forecast ──
    hier_pc = _hier_per_cell(bundle).get(cell_id) or {}
    hier_soh_r2 = hier_pc.get("soh_r2")
    latest = _latest_row((featured_dfs or {}).get(cell_id))
    # When featured_dfs is None (fleet-level summary), the served columns
    # simply were not provided — the bundle's own hierarchical validation
    # decides. When a featured frame IS provided, this cell's own forecast
    # column must exist: a cell with no usable early window has no
    # hierarchical answer even when its fleet does.
    hier_forecast_present = (
        latest is None or latest.get("rul_forecast") is not None
    )
    hier_ok = (
        bool(metrics.get("hierarchical_available"))
        and hier_soh_r2 is not None
        and hier_soh_r2 >= 0.0
        and hier_forecast_present
    )
    if hier_ok:
        reg_txt = ""
        if fold_rul_r2 is None:
            reg_txt = (
                "its RUL labels carry no measured EOL rows in this cell's regime"
                + (f" ('{reg_row.get('soh_stage')}' verdict: {reg_row.get('verdict')})" if reg_row else "")
                + " so the GBRT's RUL is not evaluable here"
            )
        else:
            reg_txt = f"its fold R²={fold_rul_r2:.2f} is below the {RUL_RELIABLE_FLOOR:.2f} floor"
        return {
            "served": ROUTE_HIERARCHICAL,
            "reason": (
                f"Hierarchical partial-pooling model answers instead of the GBRT: "
                f"{reg_txt}, while the hierarchical model's own leave-cell-out "
                f"SOH R²={hier_soh_r2:.2f} on this fleet clears zero. Its fade law is "
                "linear — knee-type degradation is exactly where the GBRT stays "
                "on — and its interval is a posterior, NOT conformally calibrated."
            ),
            "model_label": "Hierarchical partial-pooling (linear fade, chemistry prior)",
            "rul_pred": (float(latest["rul_forecast"]) if latest is not None and latest.get("rul_forecast") is not None else None),
            "rul_q10": (float(latest["rul_q10_hier"]) if latest is not None and latest.get("rul_q10_hier") is not None else None),
            "rul_q90": (float(latest["rul_q90_hier"]) if latest is not None and latest.get("rul_q90_hier") is not None else None),
            "interval_convention": _interval_convention(),
            "regime": reg_row,
            "selection": selection,
        }

    # ── Route 3: refuse — no model can back an RUL number here ──
    why_gbrt = (
        "its RUL is not evaluable in this cell's regime (no measured EOL labels)"
        if fold_rul_r2 is None
        else f"its fold R²={fold_rul_r2}" if fold_rul_r2 is not None
        else "this cell has no fold in its population"
    )
    why_hier = (
        f"hierarchical SOH R²={hier_soh_r2:.2f} fails on this fleet (knee-type fade)"
        if hier_soh_r2 is not None
        else "no hierarchical fit is available"
    )
    return _refuse(
        f"RUL withheld for this cell: the GBRT cannot answer ({why_gbrt}) and the "
        f"hierarchical model cannot either ({why_hier}). Decisions continue on "
        "measured SOH alone; any date-based schedule would be fabricated."
    )


def _interval_convention() -> str:
    from batlab.models.hierarchical import RUL_INTERVAL_CONVENTION
    return RUL_INTERVAL_CONVENTION


def route_fleet(
    cell_ids: "list[str]",
    bundles: dict,
    featured_dfs: "dict | None" = None,
) -> "list[dict]":
    """Routing verdicts for a list of cells — the fleet-level summary the
    Benchmark/Fleet surfaces can show ("N cells on GBRT, M on the
    hierarchical model, K refused")."""
    rows = []
    for cid in cell_ids or []:
        r = route_forecast_for_cell(cid, bundles, featured_dfs=featured_dfs)
        rows.append({
            "cell_id": cid,
            "served": r["served"],
            "model_label": r["model_label"],
            "rul_pred": r["rul_pred"],
            "reason": r["reason"],
        })
    return rows
