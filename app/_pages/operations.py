"""Page: Grid Services — the P1/P2 analytics surfaces in the app.

Everything this page shows was already built and REST-exposed in the
Lifecycle Intelligence layer (P1) and platform batch (P2): health-aware
dispatch, the grid-services revenue stack, managed charging, fleet
aggregation, and the ML anomaly scan. This page is the Streamlit surface
for them — no new analytics, just honest presentation of the library
outputs, with the same \"estimate, not control\" labels the modules carry.

The page is deliberately read-only in the same sense as the API: it
computes plans and offers, it does not push commands to any charger,
inverter, or market.
"""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _action_bar, _md_html, render_card, metric_tile_html
from chemistry_profiles import ChemistryProfile

_PRICE_HOURS = 48  # representative two-day window for the dispatch/revenue tabs


def _synthetic_prices() -> list:
    from market_data import SyntheticMarketAdapter
    # fetch_hourly_prices() with no start/end resolves to a 48-hour window
    data = SyntheticMarketAdapter(seed=7).fetch_hourly_prices()
    return data["prices"]


def _cell_signals(df: pd.DataFrame) -> dict:
    """The health signals the P1 modules consume, from one cell's featured df."""
    latest = df.iloc[-1]
    return {
        "soh_pct": float(latest["soh_pct"]),
        "sop_pct": (float(latest["sop_pct"]) if "sop_pct" in df.columns
                    and latest.get("sop_pct") is not None else None),
        "rul_cycles": (float(latest["rul_pred"]) if "rul_pred" in df.columns
                       and latest.get("rul_pred") is not None else None),
        "rul_reliable": bool(latest.get("rul_pred") is not None),
    }


def _nominal_kwh(cell_id: str) -> float:
    from consequences import CELL_NOMINAL_KWH
    src = ChemistryProfile.for_cell(cell_id).source_kind
    return CELL_NOMINAL_KWH.get(src, CELL_NOMINAL_KWH["synth"])


def _fmt(v, decimals=2, suffix=""):
    if v is None:
        return "—"
    try:
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_operations(cell_ids: list, active_fdfs: dict) -> None:
    _action_bar("operations")
    st.markdown("# Grid Services & Energy Operations")
    st.markdown(
        "#### The Lifecycle Intelligence layer, surfaced: health-aware dispatch, "
        "grid revenue, managed charging, fleet offers, and ML anomaly scans "
        "— estimates and plans, not control signals"
    )
    st.caption(
        "All figures use the same documented assumptions as the rest of the platform "
        "(module docstrings in src/health_aware_dispatch.py, src/grid_services.py, "
        "src/managed_charging.py, src/fleet_aggregation.py, src/ml_anomaly.py). "
        "No commands are sent to any charger, inverter, or market."
    )

    if not cell_ids:
        st.info("No cells loaded — nothing to analyse yet.")
        return

    _cell_id = st.selectbox("Cell", options=cell_ids, key="ops_cell")
    df = active_fdfs[_cell_id]
    signals = _cell_signals(df)
    nominal_kwh = _nominal_kwh(_cell_id)

    tab_dispatch, tab_revenue, tab_charge, tab_fleet, tab_anomaly = st.tabs(
        ["Dispatch", "Grid revenue", "Managed charging", "Fleet offers", "ML anomaly"]
    )

    # ── Dispatch ─────────────────────────────────────────────────────────────
    with tab_dispatch:
        from health_aware_dispatch import arbitrage_schedule, schedule_comparison

        prices = _synthetic_prices()
        result = arbitrage_schedule(
            prices, battery_kwh=nominal_kwh, c_rate=0.5,
            soh_pct=signals["soh_pct"], sop_pct=signals["sop_pct"],
            rul_cycles=signals["rul_cycles"], rul_reliable=signals["rul_reliable"],
        )
        cmp = schedule_comparison(
            prices, battery_kwh=nominal_kwh, c_rate=0.5,
            soh_pct=signals["soh_pct"], sop_pct=signals["sop_pct"],
            rul_cycles=signals["rul_cycles"], rul_reliable=signals["rul_reliable"],
        )

        _delta = cmp["delta"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health-aware revenue", f"{result['revenue_eur']:.2f} €", "2-day window")
        c2.metric("vs. assume-healthy", f"{_delta['revenue_eur_delta']:+.2f} €",
                  "signed delta (healthy − health-aware)")
        c3.metric("Cycles consumed (EFC)", f"{result['efc']:.2f}")
        c4.metric("Mean cycle DoD", f"{result['mean_cycle_dod_pct']:.0f} %")
        st.caption(
            "The same price window dispatched under the cohort's implicit 'assume healthy' "
            "assumptions vs. this cell's real health signals (SoP power cap, RUL/SOH-narrowed "
            "SOC band). A negative delta means health-aware dispatch sacrifices revenue to "
            "protect the cell — the differentiator the battery-software cohort doesn't price in."
        )

        sched = pd.DataFrame(result["schedule"])
        st.plotly_chart(
            _schedule_figure(sched, result["band"]), use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            f"Band: [{result['band']['min_soc_pct']:.0f}%, {result['band']['max_soc_pct']:.0f}%] SOC, "
            f"power cap {result['band']['power_cap_factor']*100:.0f}% of nominal C-rate. "
            f"{' — '.join(result['limitations'])}"
        )

    # ── Grid revenue ─────────────────────────────────────────────────────────
    with tab_revenue:
        from grid_services import grid_services_revenue

        rev = grid_services_revenue(
            prices, battery_kwh=nominal_kwh, c_rate=0.5,
            soh_pct=signals["soh_pct"], sop_pct=signals["sop_pct"],
            rul_cycles=signals["rul_cycles"], rul_reliable=signals["rul_reliable"],
        )
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Arbitrage (annualised)", f"{rev['arbitrage_eur']:,.0f} €/yr")
        rc2.metric("Frequency regulation", f"{rev['frequency_regulation_eur']:,.0f} €/yr")
        rc3.metric("Capacity payment", f"{rev['capacity_eur']:,.0f} €/yr")
        st.caption(rev.get("exclusivity_note", ""))
        if rev.get("assumptions_used"):
            st.caption("Rates used: " + "; ".join(
                f"{k.replace('_', ' ')} = {v}" for k, v in rev["assumptions_used"].items()
            ))
        st.caption(
            "Annualised from a 2-day representative window unless marked otherwise — "
            "an estimate of the estimate, labelled as such in src/grid_services.py."
        )

    # ── Managed charging ─────────────────────────────────────────────────────
    with tab_charge:
        from managed_charging import managed_charge_plan

        plan = managed_charge_plan(
            prices, battery_kwh=10.0, initial_soc_pct=30.0, target_soc_pct=85.0,
            max_charge_kw=3.7,
        )
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Managed cost", f"{plan['cost_eur']:.2f} €")
        cc2.metric("Unmanaged cost", f"{plan['unmanaged_cost_eur']:.2f} €")
        cc3.metric("Savings", f"{plan['savings_eur']:.2f} € ({plan['savings_pct']:.0f}%)")
        st.caption(f"Session flexibility: {plan['flexibility_efc']:.2f} EFC (rainflow) · "
                   f"{' — '.join(plan['limitations'])}")

        plan_df = pd.DataFrame(plan["plan"])
        fig = _charge_figure(plan_df, plan["unmanaged_cost_eur"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Cheapest-hour plan for a 10 kWh EV battery (30% → 85%) over the same 48-hour price "
            "window. A recommendation, not a control signal — the OCPP connector stays read-only."
        )

    # ── Fleet offers ─────────────────────────────────────────────────────────
    with tab_fleet:
        from fleet_aggregation import fleet_dispatchable_offer

        _cells = []
        for _cid in cell_ids:
            _cdf = active_fdfs[_cid]
            _sig = _cell_signals(_cdf)
            _cells.append({
                "cell_id": _cid,
                "nominal_kwh": _nominal_kwh(_cid),
                "soh_pct": _sig["soh_pct"],
                "soc_pct": 50.0,  # no per-cell SOC telemetry in reference data
                "sop_pct": _sig["sop_pct"],
                "rul_cycles": _sig["rul_cycles"],
                "rul_reliable": _sig["rul_reliable"],
            })
        offer = fleet_dispatchable_offer(_cells, window_hours=2)

        _offer = offer["offer"]
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Dispatchable energy", f"{_offer['energy_kwh']:.3f} kWh")
        fc2.metric("Dispatchable power", f"{_offer['power_kw']:.3f} kW")
        fc3.metric("Cells included", f"{len(offer.get('per_cell', []))} / {len(_cells)}")
        st.caption(offer.get("caveat", ""))
        st.caption(
            "SOC assumed 50% for every reference cell (no live SOC telemetry in public datasets) — "
            "a capability statement for this fleet's health state, not a live dispatchable offer."
        )
        if offer.get("excluded"):
            st.caption(
                "Excluded at band top: "
                + ", ".join(e.get("cell_id", "?") for e in offer["excluded"])
            )

    # ── ML anomaly ───────────────────────────────────────────────────────────
    with tab_anomaly:
        from ml_anomaly import detect_anomalous_cycles

        report = detect_anomalous_cycles(df)
        a1, a2 = st.columns(2)
        a1.metric("Cycles scored", report["n_scored"])
        a2.metric("Anomalous cycles", f"{report['n_flagged']} ({report.get('flagged_pct', 0):.1f}%)")

        per_cycle = pd.DataFrame(report["per_cycle"])
        if len(per_cycle) and per_cycle["anomaly_score"].notna().any():
            fig = _anomaly_figure(per_cycle, report["flagged_cycles"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        for c in report.get("caveats", []):
            st.caption(f"• {c}")
        if report.get("notes"):
            for n in report["notes"]:
                st.caption(f"• {n}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _schedule_figure(sched: pd.DataFrame, band: dict) -> object:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=sched["hour"], y=sched["soc_pct"], name="SOC %",
                             line=dict(color="#38bdf8", width=2)), row=1, col=1)
    fig.add_hline(y=band["max_soc_pct"], line_dash="dot", line_color="#64748b", row=1, col=1)
    fig.add_hline(y=band["min_soc_pct"], line_dash="dot", line_color="#64748b", row=1, col=1)
    fig.add_trace(go.Bar(x=sched["hour"], y=sched["charge_kw"], name="Charge kW",
                         marker_color="#4ade80"), row=2, col=1)
    fig.add_trace(go.Bar(x=sched["hour"], y=[-d for d in sched["discharge_kw"]], name="Discharge kW",
                         marker_color="#f87171"), row=2, col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), template="plotly_dark")
    fig.update_yaxes(title_text="SOC %", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    return fig


def _charge_figure(plan_df: pd.DataFrame, unmanaged_cost: float) -> object:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=plan_df["hour"], y=plan_df["soc_pct"], name="SOC %",
                             line=dict(color="#38bdf8", width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=plan_df["hour"], y=plan_df["charge_kw"], name="Charge kW",
                         marker_color="#4ade80"), row=2, col=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), template="plotly_dark")
    return fig


def _anomaly_figure(per_cycle: pd.DataFrame, flagged: list) -> object:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=per_cycle["cycle_number"], y=per_cycle["anomaly_score"],
        name="Anomaly score", mode="lines+markers", line=dict(color="#38bdf8", width=1.5),
        marker=dict(size=5),
    ))
    if flagged:
        fl = per_cycle[per_cycle["cycle_number"].isin(flagged)]
        fig.add_trace(go.Scatter(
            x=fl["cycle_number"], y=fl["anomaly_score"], name="Flagged",
            mode="markers", marker=dict(color="#f87171", size=9, symbol="x"),
        ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), template="plotly_dark")
    fig.update_yaxes(title_text="score (higher = more anomalous)")
    return fig
