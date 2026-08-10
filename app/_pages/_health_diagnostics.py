"""Health page — Engineering Details tier.

Extracted from health.py's page_health() (was 1,706 lines — the largest
file in app/_pages/). Everything here renders only when the "Engineering
details" checkbox is on; page_health() still owns the always-visible hero
section (provenance banner, SOH cards, mechanism/action cards, 12-month
forecast) and calls render_engineering_diagnostics() once that checkbox
is checked.

Split boundary rationale: hero-section state that's expensive or must not
be recomputed twice (_mech, _agreement, _physics_agreement_note -- see the
comment in page_health() about the mechanism-classifier drift bug this
guards against) is threaded in as parameters. Everything else (latest,
current_soh, cap/res anomalies, rul_q10/q90) is cheap to recompute from df
directly, so each function below does that locally instead of taking more
parameters -- keeps every function's signature minimal and self-contained.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    _md_html, _empty_state, base_layout, LEGEND_H,
    _cell_provenance, _analysis_provenance, _resample_df,
    PLOTLY_CONFIG, render_card, metric_tile_html,
)
from design_system import provenance_banner
from batlab.features.knee_detection import degradation_phases
from chemistry_profiles import ChemistryProfile


# ---------------------------------------------------------------------------
# Fade rate + degradation phase (always shown under Engineering details,
# not behind its own expander)
# ---------------------------------------------------------------------------

def render_fade_rate_and_phase_analysis(df: pd.DataFrame, _rdf: pd.DataFrame,
                                         cell_id: str, knee: dict) -> None:
    cap_anomalies = df[df.get("capacity_anomaly", pd.Series(False, index=df.index))] if "capacity_anomaly" in df.columns else df.iloc[0:0]
    res_anomalies = df[df.get("resistance_anomaly", pd.Series(False, index=df.index))] if "resistance_anomaly" in df.columns else df.iloc[0:0]

    # ── Fade acceleration detection (compute before fig3 so we can annotate it) ──
    # Heuristic: first cycle after cycle 30 where the 30-cycle rolling fade rate
    # exceeds 2× the mean rate from cycles 1–30. Not triggered in first 30 cycles
    # because early fade rates are noisy while rolling features stabilise.
    INFLECTION_MIN_CYCLE = 30
    ACCEL_FACTOR         = 2.0
    first_accel: int | None = None

    if "fade_rate_30cy" in df.columns:
        df_baseline = df[df["cycle_number"] <= INFLECTION_MIN_CYCLE]
        df_post30   = df[df["cycle_number"] >  INFLECTION_MIN_CYCLE]
        if len(df_baseline) > 0:
            baseline_rate = df_baseline["fade_rate_30cy"].mean()
            if baseline_rate > 0 and not pd.isna(baseline_rate):
                accel_threshold = baseline_rate * ACCEL_FACTOR
                accel_rows = df_post30[df_post30["fade_rate_30cy"] > accel_threshold]
                if len(accel_rows) > 0:
                    first_accel = int(accel_rows["cycle_number"].iloc[0])

    st.markdown(
        "<h4 class='section-header'>Fade Rate — Is degradation accelerating?</h4>",
        unsafe_allow_html=True,
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=_rdf["cycle_number"], y=_rdf["fade_rate_10cy"] * 1000,
        line=dict(color="#4a5568", width=1), name="10-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=_rdf["cycle_number"], y=_rdf["fade_rate_30cy"] * 1000,
        line=dict(color="#63b3ed", width=2), name="30-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=_rdf["cycle_number"], y=_rdf["fade_rate_50cy"] * 1000,
        line=dict(color="#48bb78", width=2), name="50-cycle window",
    ))
    if first_accel is not None:
        fig3.add_vline(
            x=first_accel,
            line=dict(color="#fc8181", width=1.5, dash="dash"),
            annotation_text=f"Fade acceleration threshold: {ACCEL_FACTOR:.0f}× {INFLECTION_MIN_CYCLE}-cycle baseline (heuristic)",
            annotation_position="top right",
            annotation=dict(font=dict(size=10, color="#fc8181"), bgcolor="rgba(26,32,44,0.85)"),
        )
    if knee["detected"]:
        fig3.add_vline(
            x=knee["cycle"],
            line=dict(color="#9f7aea", width=1.5, dash="dot"),
            annotation_text=f"Knee-point ({knee['cycle']})",
            annotation_position="top left",
            annotation=dict(font=dict(size=10, color="#9f7aea"), bgcolor="rgba(26,32,44,0.85)"),
        )
    fig3.update_layout(
        **base_layout(
            height=280, legend=LEGEND_H,
            xaxis=dict(title="Cycle", zeroline=False),
            yaxis=dict(title="mAh lost per cycle", zeroline=False),
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)

    if first_accel is not None:
        st.markdown(
            f"<div style='background:rgba(252,129,129,0.08);border:1px solid rgba(252,129,129,0.3);"
            f"border-radius:8px;padding:12px 16px;font-size:12px;color:#fed7d7;margin-top:-8px'>"
            f"⚠ <strong>Fade acceleration detected at cycle {first_accel}</strong> — "
            f"30-cycle rolling rate exceeded {ACCEL_FACTOR:.0f}× the cycle-1–{INFLECTION_MIN_CYCLE} baseline. "
            f"Schedule inspection or monitor closely."
            f"</div>",
            unsafe_allow_html=True,
        )

    early = df[df["confidence_tag"] == "Calibrating"]
    if len(early) > 0:
        st.info(
            f"**Calibrating** — the first {len(early)} cycles show higher variability "
            f"while rolling-window features stabilise. Readings from cycle 50 onward are reliable."
        )

    # ── Anomaly summary ──

    n_cap_anom = len(cap_anomalies)
    n_res_anom = len(res_anomalies)
    if n_cap_anom > 0 or n_res_anom > 0:
        st.markdown(
            f"<div style='background:rgba(246,173,85,0.08);border:1px solid rgba(246,173,85,0.3);"
            f"border-radius:8px;padding:12px 16px;font-size:12px;color:#fefcbf;margin-top:8px'>"
            f"⚠ <strong>Anomalous cycles detected</strong> — "
            f"{n_cap_anom} capacity outlier(s) · {n_res_anom} resistance spike(s) "
            f"(>2.5σ from 30-cycle rolling baseline). "
            f"Isolated anomalies are often measurement noise. Clustered anomalies may indicate "
            f"lithium plating, electrolyte leakage, or tab resistance changes."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Degradation phase summary ──
    st.markdown("<h4 class='section-header'>Degradation Phase Analysis</h4>", unsafe_allow_html=True)
    phase_labels = degradation_phases(df["soh_pct"], df["cycle_number"])
    phase_counts = phase_labels.value_counts()
    n_early = int(phase_counts.get("Early", 0))
    n_plateau = int(phase_counts.get("Plateau", 0))
    n_accel = int(phase_counts.get("Accelerating", 0))
    current_phase = knee["phase"]
    PHASE_COLOUR = {"Early": "#63b3ed", "Plateau": "#48bb78", "Accelerating": "#fc8181", "Unknown": "#4a5568"}
    pc = PHASE_COLOUR.get(current_phase, "#4a5568")

    render_card(
        f"""
        <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start">
            <div>
                <div style="font-size:11px;color:#a0aec0;text-transform:uppercase;letter-spacing:0.07em">Current phase</div>
                <div style="font-size:26px;font-weight:700;color:{pc}">{current_phase}</div>
                <div style="font-size:11px;color:#a0aec0;margin-top:2px">
                    {"Knee at cycle " + str(knee["cycle"]) + f" ({knee['soh_at_knee']}% SOH)" if knee["detected"] else "No knee detected yet"}
                </div>
            </div>
            <div style="border-left:1px solid #2d3748;padding-left:24px">
                <div style="font-size:11px;color:#a0aec0;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.07em">Cycle breakdown</div>
                <div style="font-size:12px;color:#a0aec0;line-height:2">
                    <span style="color:#63b3ed">Early</span> — {n_early} cycles (rolling features stabilising)<br>
                    <span style="color:#48bb78">Plateau</span> — {n_plateau} cycles (linear degradation regime)<br>
                    <span style="color:#fc8181">Accelerating</span> — {n_accel} cycles (post-knee, rapid fade)
                </div>
            </div>
            <div style="border-left:1px solid #2d3748;padding-left:24px;max-width:280px">
                <div style="font-size:11px;color:#a0aec0;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.07em">What this means</div>
                <div style="font-size:11px;color:#8896a8;line-height:1.7">
                    {"The cell is past its knee point — degradation has entered the rapid-fade regime. "
                     "Remaining life estimates are shorter and less predictable than in the plateau phase. "
                     "Prioritise replacement planning."
                     if current_phase == "Accelerating" else
                     "The cell is in the stable plateau phase — degradation is approximately linear "
                     "and predictable. RUL estimates are most reliable here."
                     if current_phase == "Plateau" else
                     "Early cycles — rolling-window features are still stabilising. "
                     "Fade rate estimates will become reliable from cycle 50 onward."}
                </div>
            </div>
        </div>
        """,
        padding="18px 22px",
        extra_style="margin-bottom:16px",
    )


# ---------------------------------------------------------------------------
# H4: Data Lineage
# ---------------------------------------------------------------------------

def render_data_lineage_expander(df: pd.DataFrame, cell_id: str) -> None:
    with st.expander("🔍 Data Lineage — Trace Any Value to Source Cycles", expanded=False):
        _lineage_rows = []
        _col_map = {
            "soh_pct":              ("SOH %",                "measured capacity / initial capacity × 100",        "Derived from measured capacity_ah"),
            "capacity_ah":          ("Capacity (Ah)",         "raw discharge capacity per cycle",                  "Direct measurement from cycler"),
            "fade_rate_30cy":       ("Fade Rate (30cy)",      "ΔQ / Δcycle over rolling 30-cycle window",          "Derived: pandas rolling().apply()"),
            "coulombic_efficiency": ("Coulombic Efficiency",  "Q_discharge / Q_charge per cycle",                  "Direct measurement from cycler"),
            "resistance_normalized":("Resistance (norm.)",    "R(n) / R(1) — normalised internal resistance",      "Derived from voltage step at cycle start"),
            "rul_pred":             ("RUL Prediction",        "GBRT model output (leave-cell-out validated)",       "ML model: GradientBoostingRegressor"),
            "ce_rolling_30cy":      ("CE Rolling (30cy)",     "30-cycle rolling mean of coulombic_efficiency",      "Derived: pandas rolling().mean()"),
            "dqdv_sim_peak_value":  ("dQ/dV Peak Value (sim)", "simulated via LiCoO₂ OCV polynomial model",        "Simulated: simulate_vq_curve()"),
            "sop_pct":              ("SOP %",                 "State of Power — rate capability proxy",            "Derived from R_normalized"),
            "c_rate_rolling_10cy":  ("C-rate (10cy avg)",     "charge/discharge rate (A / nominal Ah), 10-cy mean", "Protocol constant (NASA) or synthetic profile"),
            "stress_index":         ("Stress Index",          "Arrhenius(T) × C-rate^0.7 — composite aging driver", "Derived: features.build_features()"),
            "dod_proxy":            ("DoD proxy",             "capacity(n) / capacity(1) — depth of discharge",    "Derived from measured capacity_ah"),
            "usage_profile_code":   ("Usage Profile",         "0=Stationary-like, 1=Mixed, 2=EV-like — duty-cycle regime", "Derived from rolling C-rate/DoD variability"),
        }
        for _col, (_label, _formula, _source) in _col_map.items():
            _avail = _col in df.columns
            _latest_val = f"{df[_col].iloc[-1]:.4f}" if _avail and df[_col].notna().any() else "—"
            _n_valid = int(df[_col].notna().sum()) if _avail else 0
            _lineage_rows.append({
                "Metric": _label,
                "Latest value": _latest_val,
                "Valid cycles": _n_valid if _avail else "—",
                "Formula / derivation": _formula,
                "Source": _source,
            })
        _lin_df = pd.DataFrame(_lineage_rows)
        st.dataframe(_lin_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Cell: {cell_id} · {len(df)} total cycles · "
            f"First cycle: {int(df['cycle_number'].iloc[0])} · "
            f"Last cycle: {int(df['cycle_number'].iloc[-1])} · "
            f"Data provenance: {_cell_provenance(cell_id).upper()}"
        )


# ---------------------------------------------------------------------------
# H4.5: Why the model predicts this SOH — top drivers + citations
# ---------------------------------------------------------------------------

def render_prediction_drivers_expander(df: pd.DataFrame, cell_id: str, bundle: dict) -> None:
    # Same feature-importance data Copilot's answer_prediction_drivers()
    # narrates in chat form, surfaced natively here as a hover-tooltip
    # table so it's visible without asking the Copilot a question.
    if bundle is None:
        return
    with st.expander("🔎 Why the model predicts this SOH — top drivers", expanded=False):
        from battery_copilot import FEATURE_LABELS as _drv_labels, FEATURE_PHYSICS as _drv_physics
        from battery_knowledge import get_feature_citation
        from batlab.models.gbrt import top_drivers as _top_drivers

        _drivers = _top_drivers(bundle, model="soh", top_n=5)
        _driver_rows = []
        for _d in _drivers:
            _fname     = _d["feature"]
            _citation  = get_feature_citation(_fname)
            _tooltip   = _drv_physics.get(_fname, "")
            if _citation:
                _tooltip += (
                    f" (Source: {_citation['title']}, doi:{_citation['doi']})"
                    if _tooltip else
                    f"Source: {_citation['title']}, doi:{_citation['doi']}"
                )
            _driver_rows.append({
                "name":    _drv_labels.get(_fname, _fname),
                "pct":     _d["importance_pct"],
                "tooltip": _tooltip or "No physics explanation on file for this feature.",
            })
        # st.popover(), not a raw HTML title= attribute -- a real,
        # keyboard-focusable, screen-reader-exposed Streamlit widget, so the
        # physical explanation + citation is reachable without a mouse. See
        # the accessibility audit that replaced the old hover-only div here.
        for _row in _driver_rows:
            with st.popover(f"{_row['name']} — {_row['pct']:.1f}%", use_container_width=True):
                st.caption(_row["tooltip"])
        st.caption("Select a driver above for its physical explanation and literature citation (where one is on file).")


# ---------------------------------------------------------------------------
# T4: Degradation Mechanism Classifier (LLI vs LAM)
# ---------------------------------------------------------------------------

def render_mechanism_classifier_expander(_mech: dict) -> None:
    # _mech was already computed once, near the top of page_health(), from
    # the always-visible compact card above -- reused here rather than
    # recomputed, so there is exactly one call to diagnose_mechanism() per
    # page render, not two independent ones that could theoretically drift
    # if this expander were ever edited without touching the card above.
    with st.expander("⚗️ Degradation Mechanism Classifier — LLI vs LAM", expanded=False):
        try:
            # Render verdict card
            render_card(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>"
                f"<span style='font-size:22px'>{_mech['verdict_icon']}</span>"
                f"<div>"
                f"<div style='font-size:14px;font-weight:700;color:{_mech['verdict_color']}'>{_mech['verdict']}</div>"
                f"<div style='font-size:11px;color:#a0aec0;margin-top:2px'>"
                f"Confidence: <span style='color:{_mech['confidence_color']};font-weight:600'>{_mech['confidence_label']}</span>"
                f" · Signals used: {', '.join(_mech['confidence_notes']) if _mech['confidence_notes'] else 'none'}"
                f"</div></div></div>"
                f"<div style='font-size:12px;color:#a0aec0;line-height:1.65'>{_mech['verdict_body']}</div>",
                padding="16px 20px",
                extra_style="margin:8px 0 12px",
            )

            # Signal breakdown metrics
            _mech_signals = _mech["signals"]
            if _mech_signals:
                _mc1, _mc2, _mc3 = st.columns(3)
                with _mc1:
                    if "lli_ce_deficit" in _mech_signals:
                        st.metric("LLI Score", f"{_mech['lli_score']}",
                                  help="CE decline + linear fade pattern")
                with _mc2:
                    if "lam_nonlinearity" in _mech_signals:
                        st.metric("LAM Score", f"{_mech['lam_score']}",
                                  help="Accelerating fade + resistance rise")
                with _mc3:
                    if "resistance_slope" in _mech_signals:
                        st.metric("R Rise Rate", f"{_mech_signals['resistance_slope']:.3f} Ω/kcy",
                                  help="Resistance increase per 1000 cycles")
        except Exception as _e_mech:
            st.info(f"Mechanism classifier unavailable: {_e_mech}")


# ---------------------------------------------------------------------------
# dQ/dV Degradation Analysis
# ---------------------------------------------------------------------------

def render_dqdv_expander(df: pd.DataFrame, _rdf: pd.DataFrame, cell_id: str) -> None:
    # LFP cells (Severson) have a flat discharge plateau — the LiCoO₂ OCV
    # polynomial used in simulate_vq_curve() does not apply to LFP chemistry.
    # Showing simulated LiCoO₂ dQ/dV peaks for an LFP cell is physically wrong.
    _is_lfp = not ChemistryProfile.for_cell(cell_id).dqdv_applicable   # LFP (Severson) and NCA (Oxford) both lack the LiCoO2 dqdv model
    _dqdv_prov = _analysis_provenance(cell_id, "derived")
    _dqdv_badge = {"measured": "◐ SIMULATED", "simulated": "◐ SIMULATED", "synthetic": "○ SYNTHETIC"}[_dqdv_prov]
    _dqdv_title = (
        "dQ/dV Proxy — LiCoO₂ model only (not applicable to LFP)"
        if _is_lfp
        else f"dQ/dV Simulated Proxy (LiCoO₂ model) — {_dqdv_badge}"
    )
    with st.expander(f"🔬 {_dqdv_title}", expanded=False):
        if _is_lfp:
            st.warning(
                "**dQ/dV analysis is not available for LFP cells (Severson 2019).**\n\n"
                "The simulation uses a LiCoO₂ OCV polynomial with a well-defined voltage peak. "
                "LFP has a flat ≈3.2 V plateau — no peak exists, so the simulated features carry "
                "no physical meaning for this chemistry. "
                "Real dQ/dV for LFP requires per-cycle measured V(Q) curves from the cycler."
            )
        else:
          try:
            from batlab.features.dqdv import simulate_vq_curve
            _prov_dqdv = _analysis_provenance(cell_id, "derived")
            _md_html(provenance_banner(_prov_dqdv,
                "<strong>Simulated proxy — not measured data.</strong> "
                "V(Q) and dQ/dV curves are computed by <code>simulate_vq_curve()</code>, a "
                "LiCoO&#x2082; OCV polynomial model parameterised from the cell's measured "
                "capacity and resistance. No voltage traces were logged during cycling. "
                "These features are physically meaningful only for LiCoO&#x2082;-family chemistries."
            ))

            n_cycles = len(df)
            pct_indices = [int(n_cycles * p) for p in [0.05, 0.25, 0.50, 0.75, 0.95]]
            pct_indices = [min(max(i, 0), n_cycles - 1) for i in pct_indices]
            rep_rows = [df.iloc[i] for i in pct_indices]

            # ── A) V(Q) Curve Overlay ──
            _COLORS = ["#63b3ed", "#7fbfea", "#f6ad55", "#ed8936", "#fc8181"]
            fig_vq = go.Figure()
            for i, (row, color) in enumerate(zip(rep_rows, _COLORS)):
                try:
                    cap = float(row["capacity_ah"])
                    res = float(row.get("resistance_ohm", 0.02))
                    Q, V = simulate_vq_curve(cap, res)
                    fig_vq.add_trace(go.Scatter(
                        x=Q.tolist(), y=V.tolist(),
                        mode="lines",
                        name=f"Cycle {int(row['cycle_number'])}",
                        line=dict(color=color, width=1.8),
                        hovertemplate="Q=%{x:.3f} Ah  V=%{y:.3f} V<extra>Cycle " + str(int(row["cycle_number"])) + "</extra>",
                    ))
                except Exception:
                    pass
            fig_vq.update_layout(
                **base_layout(
                    height=300, legend=LEGEND_H,
                    xaxis=dict(title="Discharge Capacity (Ah)", zeroline=False),
                    yaxis=dict(title="Voltage (V)", zeroline=False),
                ),
            )
            fig_vq.update_layout(title=dict(text="Simulated Discharge Voltage Curves (V vs Q)", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(fig_vq, use_container_width=True)

            # ── B) dQ/dV Peak Trend ──
            dqdv_cols = ["dqdv_sim_peak_soc", "dqdv_sim_peak_value"]
            if all(c in df.columns for c in dqdv_cols):
                fig_peak = go.Figure()
                fig_peak.add_trace(go.Scatter(
                    x=_rdf["cycle_number"], y=_rdf["dqdv_sim_peak_soc"],
                    name="Peak SOC Position", line=dict(color="#63b3ed", width=2),
                    hovertemplate="Cycle %{x}: %{y:.3f}<extra>Peak SOC</extra>",
                ))
                fig_peak.add_trace(go.Scatter(
                    x=_rdf["cycle_number"], y=_rdf["dqdv_sim_peak_value"],
                    name="Peak Amplitude (Ah/V)", line=dict(color="#f6ad55", width=2),
                    yaxis="y2",
                    hovertemplate="Cycle %{x}: %{y:.2f} Ah/V<extra>Peak Amplitude</extra>",
                ))
                fig_peak.update_layout(
                    **base_layout(
                        height=300, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="Peak SOC Position", zeroline=False, color="#63b3ed"),
                        yaxis2=dict(title="Peak Amplitude (Ah/V)", overlaying="y", side="right",
                                    zeroline=False, color="#f6ad55"),
                    ),
                )
                fig_peak.update_layout(title=dict(text="dQ/dV Peak Shift Over Aging", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(fig_peak, use_container_width=True)
                st.caption("Peak SOC shifts left and amplitude drops as active material is lost — a direct electrochemical signature of degradation.")

            # ── C) FWHM Trend ──
            if "dqdv_sim_fwhm" in df.columns:
                fig_fwhm = go.Figure()
                fig_fwhm.add_trace(go.Scatter(
                    x=_rdf["cycle_number"], y=_rdf["dqdv_sim_fwhm"],
                    line=dict(color="#48bb78", width=2),
                    hovertemplate="Cycle %{x}: %{y:.4f} Ah<extra>FWHM</extra>",
                    showlegend=False,
                ))
                fig_fwhm.update_layout(
                    **base_layout(
                        height=260, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="FWHM (Ah)", zeroline=False),
                    ),
                )
                fig_fwhm.update_layout(title=dict(text="dQ/dV Peak Width (FWHM) — Broadening = increased heterogeneity", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(fig_fwhm, use_container_width=True)

            # ── ⚗️ Degradation Mechanism — single classification card ──
            _dom_mech, _dom_explain = "Inconclusive", "Insufficient dQ/dV data for classification."
            if all(c in df.columns for c in ["dqdv_sim_peak_soc", "dqdv_sim_peak_value", "dqdv_sim_fwhm"]):
                import numpy as _np2
                _em = df[df.cycle_number <= df.cycle_number.quantile(0.10)]
                _lm = df[df.cycle_number >= df.cycle_number.quantile(0.90)]
                if len(_em) > 0 and len(_lm) > 0 and _em.dqdv_sim_peak_value.mean() > 1e-9 and _em.dqdv_sim_fwhm.mean() > 1e-9:
                    _ps  = _em.dqdv_sim_peak_soc.mean() - _lm.dqdv_sim_peak_soc.mean()
                    _ad  = (_em.dqdv_sim_peak_value.mean() - _lm.dqdv_sim_peak_value.mean()) / _em.dqdv_sim_peak_value.mean() * 100
                    _fw  = (_lm.dqdv_sim_fwhm.mean() - _em.dqdv_sim_fwhm.mean()) / _em.dqdv_sim_fwhm.mean() * 100
                    if _ps > 0.05 and _ad > 10:
                        _dom_mech = "LLI + LAM"
                        _dom_explain = "Both SEI lithium loss and electrode material loss are active. Typical of combined calendar + high-rate cycling stress."
                    elif _ps > 0.05:
                        _dom_mech = "LLI — Loss of Lithium Inventory"
                        _dom_explain = "SEI growth is consuming cyclable lithium. Peak shift to lower SOC values. Common in calendar-aged or high-temperature cells."
                    elif _ad > 15:
                        _dom_mech = "LAM — Loss of Active Material"
                        _dom_explain = "Electrode active sites are being lost. Peak amplitude decreasing proportionally to capacity loss. Common in high C-rate cycling."
                    elif _fw > 20:
                        _dom_mech = "LAM-NE — Graphite heterogeneity"
                        _dom_explain = "Peak broadening without large position shift suggests graphite particle cracking or lithium plating creating heterogeneous intercalation."
                    else:
                        _dom_mech = "Early stage"
                        _dom_explain = "Degradation has not yet reached a classifiable signature. Monitor after further cycling."
            _mech_col = "#f6ad55" if "LAM" in _dom_mech or "LLI" in _dom_mech else "#8896a8"
            render_card(
                metric_tile_html("Dominant mechanism", _dom_mech, value_color=_mech_col, value_size="15px")
                + f"<div style='font-size:12px;color:#a0aec0;line-height:1.5;margin-top:6px'>{_dom_explain}</div>",
                padding="12px 16px",
                extra_style="margin-top:8px",
            )
          except Exception as _e:
            st.info(f"dQ/dV analysis unavailable: {_e}")


# ---------------------------------------------------------------------------
# Coulombic Efficiency
# ---------------------------------------------------------------------------

def render_ce_expander(df: pd.DataFrame, cell_id: str) -> None:
    if "coulombic_efficiency" not in df.columns:
        return
    _ce_prov = _analysis_provenance(cell_id, "cycle")
    _ce_badge = "● MEASURED" if _ce_prov == "measured" else "○ SYNTHETIC"
    with st.expander(f"⚡ Coulombic Efficiency (CE) Analysis — {_ce_badge}", expanded=False):
        try:
            ce_latest  = float(df["coulombic_efficiency"].iloc[-1])
            ce_initial = float(df["coulombic_efficiency"].iloc[:10].mean())
            ce_drop    = ce_initial - ce_latest

            _ce1, _ce2, _ce3 = st.columns(3)
            with _ce1:
                st.metric("Current CE", f"{ce_latest*100:.3f}%",
                          help="Q_discharge / Q_charge — lithium consumed by SEI each cycle")
            with _ce2:
                st.metric("CE Drop from Cycle 1", f"{ce_drop*100:.3f}%",
                          delta=f"{-ce_drop*100:.3f}%",
                          help="Cumulative lithium inventory loss signature")
            with _ce3:
                ce_risk = "High" if ce_latest < 0.993 else ("Moderate" if ce_latest < 0.997 else "Low")
                st.metric("Lithium Loss Risk", ce_risk,
                          help="< 99.7%: moderate SEI growth; < 99.3%: accelerated LLI")

            _md_html("""<div style="font-size:12px;color:#8896a8;margin:8px 0 12px;line-height:1.6"><strong style="color:#a0aec0">Why CE matters:</strong> Every cycle where CE &lt; 100% permanently loses cyclable lithium to the SEI layer. A CE drop from 99.95% → 99.30% may seem small but represents ~0.65% lithium lost per cycle — the dominant degradation mechanism in calendar-aged cells. CATL and BYD track cumulative CE deficit as the primary warranty signal for LLI-dominated failures.</div>""")

            # CE trend chart
            _rdf_ce = _resample_df(df[df["coulombic_efficiency"].notna()])
            fig_ce = go.Figure()
            fig_ce.add_trace(go.Scatter(
                x=_rdf_ce["cycle_number"].tolist(),
                y=(_rdf_ce["coulombic_efficiency"] * 100).tolist(),
                name="CE per cycle", mode="lines",
                line=dict(color="#4a5568", width=1),
                hovertemplate="Cycle %{x}: %{y:.4f}%<extra>CE</extra>",
            ))
            if "ce_rolling_30cy" in df.columns:
                fig_ce.add_trace(go.Scatter(
                    x=_rdf_ce["cycle_number"].tolist(),
                    y=(_rdf_ce["ce_rolling_30cy"] * 100).tolist(),
                    name="30-cycle rolling avg", mode="lines",
                    line=dict(color="#63b3ed", width=2),
                    hovertemplate="Cycle %{x}: %{y:.4f}%<extra>30cy avg</extra>",
                ))
            fig_ce.add_hline(y=99.70, line=dict(color="#f6ad55", width=1, dash="dot"),
                             annotation_text="99.70% — moderate risk threshold",
                             annotation=dict(font=dict(size=9, color="#f6ad55")))
            fig_ce.add_hline(y=99.30, line=dict(color="#fc8181", width=1, dash="dot"),
                             annotation_text="99.30% — high LLI risk",
                             annotation=dict(font=dict(size=9, color="#fc8181")))
            fig_ce.update_layout(
                **base_layout(
                    height=280, legend=LEGEND_H,
                    xaxis=dict(title="Cycle", zeroline=False),
                    yaxis=dict(title="Coulombic Efficiency (%)", zeroline=False),
                ),
            )
            fig_ce.update_layout(title=dict(text="Coulombic Efficiency — Q_discharge / Q_charge", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(fig_ce, use_container_width=True)

            # Cumulative lithium loss
            ce_deficit = (1.0 - _rdf_ce["coulombic_efficiency"]).cumsum()
            fig_lii = go.Figure()
            fig_lii.add_trace(go.Scatter(
                x=_rdf_ce["cycle_number"].tolist(),
                y=(ce_deficit * float(df["capacity_ah"].iloc[0]) * 1000).tolist(),
                fill="tozeroy", fillcolor="rgba(252,129,129,0.08)",
                line=dict(color="#fc8181", width=2),
                hovertemplate="Cycle %{x}: %{y:.2f} mAh cumulative LLI<extra></extra>",
                showlegend=False,
            ))
            fig_lii.update_layout(
                **base_layout(
                    height=220, legend=LEGEND_H,
                    xaxis=dict(title="Cycle", zeroline=False),
                    yaxis=dict(title="Cumulative Li loss (mAh)", zeroline=False),
                ),
            )
            fig_lii.update_layout(title=dict(text="Cumulative Lithium Inventory Loss (LLI) from CE Deficit", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(fig_lii, use_container_width=True)
        except Exception as _ce_e:
            st.info(f"CE analysis unavailable: {_ce_e}")


# ---------------------------------------------------------------------------
# Energy & Capacity Throughput
# ---------------------------------------------------------------------------

def render_energy_throughput_expander(df: pd.DataFrame, _rdf: pd.DataFrame, cell_id: str) -> None:
    if "cumulative_kwh" not in df.columns:
        return
    _thru_badge = "● MEASURED" if _cell_provenance(cell_id) == "measured" else "○ SYNTHETIC"
    with st.expander(f"📈 Energy & Capacity Throughput — {_thru_badge}", expanded=False):
        try:
            total_kwh = float(df["cumulative_kwh"].iloc[-1])
            total_ah  = float(df["cumulative_ah"].iloc[-1])
            eq_cy     = float(df["equivalent_cycles"].iloc[-1]) if "equivalent_cycles" in df.columns and not pd.isna(df["equivalent_cycles"].iloc[-1]) else None

            _t1, _t2, _t3 = st.columns(3)
            with _t1:
                st.metric("Total Energy Delivered", f"{total_kwh:.2f} kWh",
                          help="Cumulative energy throughput — BYD/CATL warranty metric")
            with _t2:
                st.metric("Total Capacity Delivered", f"{total_ah:.0f} Ah",
                          help="Cumulative Ah throughput across all cycles")
            with _t3:
                if eq_cy is not None:
                    raw_cy = int(df["cycle_number"].iloc[-1])
                    st.metric("Equivalent Cycles", f"{eq_cy:,.0f}",
                              delta=f"{eq_cy - raw_cy:+,.0f} vs raw",
                              help="Stress-normalized cycle count: raw cycles × stress factor")
                else:
                    st.metric("Equivalent Cycles", "—", help="Not available for this data source")

            fig_kwh = go.Figure()
            fig_kwh.add_trace(go.Scatter(
                x=_rdf["cycle_number"].tolist(),
                y=_rdf["cumulative_kwh"].tolist(),
                fill="tozeroy", fillcolor="rgba(99,179,237,0.07)",
                line=dict(color="#63b3ed", width=2),
                hovertemplate="Cycle %{x}: %{y:.3f} kWh delivered<extra></extra>",
                showlegend=False,
            ))
            fig_kwh.update_layout(
                **base_layout(
                    height=250, legend=LEGEND_H,
                    xaxis=dict(title="Cycle", zeroline=False),
                    yaxis=dict(title="Cumulative kWh", zeroline=False),
                ),
            )
            fig_kwh.update_layout(title=dict(text="Cumulative Energy Throughput (kWh) — Warranty Accounting Metric", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(fig_kwh, use_container_width=True)
        except Exception as _te:
            st.info(f"Throughput analysis unavailable: {_te}")


# ---------------------------------------------------------------------------
# Lithium Plating Risk Score
# ---------------------------------------------------------------------------

def render_lithium_plating_expander(df: pd.DataFrame) -> None:
    with st.expander("⚠ Lithium Plating Risk", expanded=False):
        _md_html(
            "<div style='font-size:12px;color:#8896a8;margin-bottom:10px'>"
            "Lithium plating (Li deposition on the anode) is the primary cause of sudden capacity loss "
            "and thermal runaway risk. Detected via Coulombic Efficiency dips below 99.5% — each dip "
            "indicates a cycle where more lithium was lost than consumed by normal SEI growth. "
            "Correlation with high C-rate or low temperature (if available) increases confidence."
            "</div>"
        )
        try:
            _ce_col = "coulombic_efficiency" if "coulombic_efficiency" in df.columns else None
            if _ce_col is None:
                _empty_state("CE data unavailable",
                             "Coulombic Efficiency data is required to compute plating risk. "
                             "This column was not found in the feature set for this cell.",
                             "→ Ensure the source data includes charge and discharge capacity per cycle.", "⚠")
            else:
                _ce_valid = df[_ce_col].dropna()
                # CE dips: readings below 99.5% threshold
                _PLATING_CE_THRESH = 0.995
                _dip_mask  = _ce_valid < _PLATING_CE_THRESH
                _n_dips    = int(_dip_mask.sum())
                _dip_cycles = df.loc[_ce_valid[_dip_mask].index, "cycle_number"].tolist() if _n_dips else []

                # Consecutive dip runs → sustained plating events
                _runs, _in_run, _run_len = 0, False, 0
                for _d in _dip_mask:
                    if _d:
                        _run_len += 1
                        if _run_len >= 3:
                            if not _in_run:
                                _runs += 1
                                _in_run = True
                    else:
                        _in_run, _run_len = False, 0

                # Risk score: 0-100
                _dip_rate   = _n_dips / max(len(_ce_valid), 1)
                _risk_score = min(100, int(_dip_rate * 400 + _runs * 15))
                _risk_label = ("Low" if _risk_score < 25 else
                               "Moderate" if _risk_score < 55 else
                               "High" if _risk_score < 80 else "Critical")
                _risk_colour = ("#48bb78" if _risk_score < 25 else
                                "#f6ad55" if _risk_score < 55 else
                                "#e53e3e" if _risk_score < 80 else "#c53030")

                # Score strip
                _pr1, _pr2, _pr3, _pr4 = st.columns(4)
                _pr1.metric("Plating Risk Score", f"{_risk_score}/100",
                            help="0=negligible · 25+=moderate · 55+=high · 80+=critical")
                _pr2.metric("Risk Level", _risk_label)
                _pr3.metric("CE Dip Events", f"{_n_dips}")
                _pr4.metric("Sustained Runs (≥3cy)", f"{_runs}")

                # Colour bar
                _md_html(
                    f"<div style='background:#1e2a38;border-radius:6px;height:8px;margin:8px 0 12px;overflow:hidden'>"
                    f"<div style='background:{_risk_colour};width:{_risk_score}%;height:100%;border-radius:6px;"
                    f"transition:width 0.3s'></div></div>"
                )

                # CE dip chart
                if _n_dips > 0:
                    _fig_plat = go.Figure()
                    _fig_plat.add_trace(go.Scatter(
                        x=df["cycle_number"].tolist(), y=(_ce_valid * 100).reindex(df.index).tolist(),
                        name="CE (%)", line=dict(color="#63b3ed", width=1.5),
                        hovertemplate="Cycle %{x}: CE %{y:.2f}%<extra></extra>",
                    ))
                    _fig_plat.add_hline(y=99.5, line_dash="dash", line_color="#fc8181", line_width=1,
                                        annotation_text="Plating threshold (99.5%)",
                                        annotation_font=dict(color="#fc8181", size=10))
                    if _dip_cycles:
                        _dip_ce = [float(_ce_valid.iloc[_ce_valid.index.get_loc(
                            df[df["cycle_number"] == c].index[0])] * 100)
                                   if len(df[df["cycle_number"] == c]) > 0 else None
                                   for c in _dip_cycles[:50]]
                        _fig_plat.add_trace(go.Scatter(
                            x=_dip_cycles[:50], y=_dip_ce,
                            mode="markers", name="CE dip (plating event)",
                            marker=dict(color="#fc8181", size=7, symbol="x"),
                        ))
                    _fig_plat.update_layout(
                        **base_layout(
                            height=220,
                            xaxis=dict(title="Cycle"),
                            yaxis=dict(title="CE (%)", range=[99.0, 100.05]),
                        ),
                    )
                    _fig_plat.update_layout(legend=dict(font=dict(size=10, color="#718096")))
                    st.plotly_chart(_fig_plat, use_container_width=True)

                # Interpretation
                _plat_text = {
                    "Low":      "CE stays consistently above 99.5%. No plating signals detected. "
                                "Current operating conditions appear safe.",
                    "Moderate": f"{_n_dips} CE dips detected below 99.5%. This may indicate isolated "
                                "plating events. Review charging conditions — reduce C-rate at low temperatures.",
                    "High":     f"{_n_dips} CE dips with {_runs} sustained run(s). Repeated plating is "
                                "occurring. Reduce maximum C-rate and avoid charging below 10°C. "
                                "Consider capacity check cycle to quantify lithium inventory loss.",
                    "Critical": f"CRITICAL: {_n_dips} CE dips, {_runs} sustained events. Active lithium "
                                "plating is likely. Risk of thermal runaway elevated. Immediate inspection "
                                "recommended. Do not fast-charge this cell.",
                }[_risk_label]
                _md_html(
                    f"<div style='background:{_risk_colour}11;border-left:3px solid {_risk_colour};"
                    f"border-radius:6px;padding:10px 14px;font-size:12px;color:#a0aec0'>"
                    f"<strong style='color:{_risk_colour}'>{_risk_label} risk:</strong> {_plat_text}"
                    f"</div>"
                )
        except Exception as _plat_err:
            st.info(f"Plating risk analysis unavailable: {_plat_err}")


# ---------------------------------------------------------------------------
# Resistance Component Proxy (simulated EIS)
# ---------------------------------------------------------------------------

def render_resistance_proxy_expander(df: pd.DataFrame, _rdf: pd.DataFrame, cell_id: str) -> None:
    _eis_badge = "◐ SIMULATED" if _analysis_provenance(cell_id) == "simulated" else "○ SYNTHETIC"
    with st.expander(f"🔬 Resistance Component Proxy (simulated) — {_eis_badge}", expanded=False):
        try:
            import numpy as _np_eis
            _md_html(provenance_banner(
                _analysis_provenance(cell_id),
                "Impedance components (R_ohm, R_SEI, R_ct, σ_w) are derived from DC resistance via "
                "fixed physics-attribution fractions, <strong>not from frequency-domain impedance "
                "spectroscopy measurements</strong>. The Nyquist plot is computed from a Randles "
                "circuit model parameterised by these attributed values. "
                "Connect a real EIS instrument file (Gamry / BioLogic .DTA/.mpt) to replace this "
                "with measured impedance data."
            ))
            _eis_cols = ["r_ohm_eis", "r_sei", "r_ct", "sigma_w"]
            _has_eis  = all(c in df.columns for c in _eis_cols)
            if not _has_eis:
                st.info("EIS columns not available in this dataset. Regenerate data to enable.")
            else:
                st.caption(
                    "Randles-circuit decomposition of the measured DC resistance. "
                    "R_ohm (bulk electrolyte, constant), R_SEI (SEI layer, grows with √time), "
                    "R_ct (charge transfer, grows with cycle^0.6), σ_w (Warburg / diffusion coefficient)."
                )
                # Pick 3 lifecycle snapshots: early / mid / late
                _n = len(df)
                _snap_idx = [max(0, int(_n * f) - 1) for f in [0.05, 0.50, 0.95]]
                _snap_labels = ["Early (5%)", "Mid (50%)", "Late (95%)"]

                # Nyquist simulation
                _FREQ = _np_eis.logspace(5, -2, 60)
                _OMEGA = 2 * _np_eis.pi * _FREQ
                _nyq_colors = ["#48bb78", "#63b3ed", "#fc8181"]
                _fig_nyq = go.Figure()
                for _si, (_idx, _lbl, _col) in enumerate(zip(_snap_idx, _snap_labels, _nyq_colors)):
                    _row = df.iloc[_idx]
                    _r_ohm = float(_row["r_ohm_eis"])
                    _r_sei = float(_row["r_sei"])
                    _r_ct  = float(_row["r_ct"])
                    _sw    = float(_row["sigma_w"])
                    # CPE-based Randles circuit
                    _alpha_sei, _alpha_dl = 0.82, 0.88
                    _Q_sei, _Q_dl = 8e-6, 3e-5
                    _Z_sei_cpe = 1.0 / (_Q_sei * (1j * _OMEGA) ** _alpha_sei)
                    _Z_dl_cpe  = 1.0 / (_Q_dl  * (1j * _OMEGA) ** _alpha_dl)
                    _Z_sei = (_r_sei * _Z_sei_cpe) / (_r_sei + _Z_sei_cpe)
                    _Z_w   = _sw * (1 - 1j) / _np_eis.sqrt(_np_eis.maximum(_OMEGA, 1e-9))
                    _Z_ct  = (_r_ct * (_Z_dl_cpe + _Z_w)) / (_r_ct + _Z_dl_cpe + _Z_w)
                    _Z     = _r_ohm + _Z_sei + _Z_ct
                    _fig_nyq.add_trace(go.Scatter(
                        x=_Z.real.tolist(), y=(-_Z.imag).tolist(),
                        mode="lines", name=_lbl,
                        line=dict(color=_col, width=2),
                        hovertemplate=f"<b>{_lbl}</b><br>Z' = %{{x:.4f}} Ω<br>-Z'' = %{{y:.4f}} Ω<extra></extra>",
                    ))
                _fig_nyq.update_layout(
                    **base_layout(
                        height=300, legend=LEGEND_H,
                        xaxis=dict(title="Z' (Ω) — Real", zeroline=False),
                        yaxis=dict(title="-Z'' (Ω) — Imaginary", zeroline=False),
                    ),
                )
                _fig_nyq.update_layout(title=dict(text="Nyquist Plot at 3 Lifecycle Stages (CPE Modified Randles)", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(_fig_nyq, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "nyquist_plot"}})

                # Component trend chart
                _fig_eis_trend = go.Figure()
                for _col_name, _label, _color in [
                    ("r_sei", "R_SEI (SEI layer)", "#f6ad55"),
                    ("r_ct",  "R_ct (charge transfer)", "#63b3ed"),
                    ("r_ohm_eis", "R_ohm (electrolyte)", "#48bb78"),
                ]:
                    _fig_eis_trend.add_trace(go.Scatter(
                        x=_rdf["cycle_number"].tolist(), y=_rdf[_col_name].tolist(),
                        mode="lines", name=_label,
                        line=dict(width=2),
                        hovertemplate=f"Cycle %{{x}}: %{{y:.4f}} Ω ({_label})<extra></extra>",
                    ))
                _fig_eis_trend.update_layout(
                    **base_layout(
                        height=270, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="Resistance (Ω)", zeroline=False),
                    ),
                )
                _fig_eis_trend.update_layout(title=dict(text="Resistance Component Trends — Simulated Proxy", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(_fig_eis_trend, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "eis_component_trends"}})

                # Mechanism interpretation
                _late = df.iloc[-1]
                _r0   = df.iloc[0]["r_sei"]
                _sei_growth = (_late["r_sei"] - _r0) / max(_r0, 1e-9) * 100
                _ct_growth  = (_late["r_ct"]  - df.iloc[0]["r_ct"]) / max(df.iloc[0]["r_ct"], 1e-9) * 100
                if _sei_growth > _ct_growth * 1.3:
                    _mech = "Calendar / SEI-dominated"
                    _mech_color = "#f6ad55"
                    _mech_explain = "R_SEI growth outpaces R_ct — primary degradation is SEI thickening from calendar storage and temperature exposure."
                elif _ct_growth > _sei_growth * 1.3:
                    _mech = "Cycle / Charge-transfer-dominated"
                    _mech_color = "#63b3ed"
                    _mech_explain = "R_ct growth outpaces R_SEI — primary degradation is charge-transfer impedance from active-material loss and lithium plating."
                else:
                    _mech = "Mixed SEI + charge-transfer"
                    _mech_color = "#48bb78"
                    _mech_explain = "Balanced SEI and charge-transfer growth — both calendar and cycling contribute comparably."
                _md_html(f"""<div style="background:rgba(26,32,44,0.7);border:1px solid #2d3748;border-radius:8px;padding:12px 16px;font-size:13px"><span style="color:{_mech_color};font-weight:700">{_mech}</span><span style="color:#a0aec0;margin-left:10px">{_mech_explain}</span><br><span style="color:#8896a8;font-size:12px;margin-top:4px;display:block">R_SEI growth: {_sei_growth:.1f}% · R_ct growth: {_ct_growth:.1f}%</span></div>""")
        except Exception as _eis_e:
            st.info(f"EIS analysis unavailable: {_eis_e}")


# ---------------------------------------------------------------------------
# Formation Protocol Analysis
# ---------------------------------------------------------------------------

def render_formation_protocol_expander(df: pd.DataFrame, cell_id: str) -> None:
    _form_badge = "◐ SIMULATED" if _analysis_provenance(cell_id) == "simulated" else "○ SYNTHETIC"
    with st.expander(f"🧪 Formation Protocol Analysis — {_form_badge}", expanded=False):
        try:
            import numpy as _np_form
            _md_html(provenance_banner(
                _analysis_provenance(cell_id),
                "Formation metrics (ICL, CE stabilisation) are read from the first cycles of "
                "this cell's cycle-summary data. For synthetic cells this data was generated, "
                "not measured. For NASA cells it is measured cycle capacity — "
                "but NASA cells were cycled at fixed conditions after formation, not logged during "
                "formation itself. CE values for NASA cells are <strong>not directly measured</strong>; "
                "they are model outputs from <code>generate_cell_data()</code> logic."
            ))
            _form_cycles = df[df["cycle_number"] <= 10].copy() if len(df) >= 10 else df.copy()
            if len(_form_cycles) < 3:
                st.info("Need at least 3 cycles to analyse formation.")
            else:
                st.caption(
                    "Formation = first 3–10 cycles. SEI forms on the anode, consuming cyclable lithium "
                    "(Irreversible Capacity Loss, ICL). CE rises steeply from ~90–95% toward >99.9% as the "
                    "SEI stabilises. Rate and magnitude predict long-term calendar ageing."
                )
                _q0   = float(_form_cycles["capacity_ah"].iloc[0])
                _q3   = float(_form_cycles["capacity_ah"].iloc[min(2, len(_form_cycles)-1)])
                _icl  = (_q0 - _q3) / max(_q0, 1e-9) * 100
                _ce0  = float(_form_cycles["coulombic_efficiency"].iloc[0]) * 100 if "coulombic_efficiency" in _form_cycles.columns else None
                _ce3  = float(_form_cycles["coulombic_efficiency"].iloc[min(2, len(_form_cycles)-1)]) * 100 if "coulombic_efficiency" in _form_cycles.columns else None

                _fc1, _fc2, _fc3 = st.columns(3)
                _fc1.metric("ICL (Cycles 1→3)", f"{_icl:.2f}%", help="Irreversible Capacity Loss during formation. Target <2% for LiCoO₂.")
                if _ce0 is not None:
                    _fc2.metric("CE Cycle 1", f"{_ce0:.2f}%", help="First-cycle CE; ~90–95% is normal — SEI consumes lithium.")
                    _fc3.metric("CE Cycle 3", f"{_ce3:.2f}%", delta=f"+{_ce3-_ce0:.2f}pp", help="CE should stabilise toward 99.9% by cycle 10.")

                # CE stabilisation curve
                if "coulombic_efficiency" in df.columns:
                    _form_plot = df[df["cycle_number"] <= 30].copy()
                    _fig_form = go.Figure()
                    _fig_form.add_trace(go.Scatter(
                        x=_form_plot["cycle_number"].tolist(),
                        y=(_form_plot["coulombic_efficiency"] * 100).tolist(),
                        mode="lines+markers",
                        marker=dict(size=5, color="#48bb78"),
                        line=dict(color="#48bb78", width=2),
                        name="CE",
                        hovertemplate="Cycle %{x}: CE = %{y:.3f}%<extra></extra>",
                    ))
                    _fig_form.add_hline(y=99.9, line_dash="dot", line_color="#f6ad55", annotation_text="99.9% — SEI stabilisation target", annotation_position="bottom right")
                    _fig_form.update_layout(
                        **base_layout(
                            height=260, legend=LEGEND_H,
                            xaxis=dict(title="Cycle", zeroline=False),
                            yaxis=dict(title="Coulombic Efficiency (%)", zeroline=False),
                        ),
                    )
                    _fig_form.update_layout(title=dict(text="CE Stabilisation — Formation Window (First 30 Cycles)", font=dict(size=12, color="#a0aec0"), x=0))
                    st.plotly_chart(_fig_form, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "ce_formation"}})

                # Lifetime prediction from formation metrics
                _icl_risk = "Low" if _icl < 1.5 else ("Moderate" if _icl < 3.0 else "High")
                _risk_color = {"Low": "#48bb78", "Moderate": "#f6ad55", "High": "#fc8181"}[_icl_risk]
                _md_html(f"""<div style="background:rgba(26,32,44,0.7);border:1px solid #2d3748;border-radius:8px;padding:12px 16px;font-size:13px"><span style="color:{_risk_color};font-weight:700">ICL Risk: {_icl_risk}</span><span style="color:#a0aec0;margin-left:10px">ICL = {_icl:.2f}% — literature correlation: cells with ICL &lt;1.5% retain &gt;90% SOH beyond 500 cycles (Severson et al. 2019).</span></div>""")
        except Exception as _form_e:
            st.info(f"Formation analysis unavailable: {_form_e}")


# ---------------------------------------------------------------------------
# Rate Capability Analysis
# ---------------------------------------------------------------------------

def render_rate_capability_expander(df: pd.DataFrame, cell_id: str) -> None:
    _rate_badge = "◐ SIMULATED" if _analysis_provenance(cell_id) == "simulated" else "○ SYNTHETIC"
    with st.expander(f"📊 Rate Capability — {_rate_badge}", expanded=False):
        try:
            import numpy as _np_rate
            _md_html(provenance_banner(
                _analysis_provenance(cell_id),
                "Capacity retention at each C-rate is computed from "
                "<code>Q(C) = Q_nom × SOH × exp(−k_rate × C^1.4)</code>, where k_rate scales "
                "with measured internal resistance. <strong>No multi-rate characterisation test was "
                "performed.</strong> Real rate capability requires dedicated CCCV cycling at each "
                "C-rate on a cell tester (Arbin, Maccor, Neware, etc.)."
            ))
            st.caption(
                "Rate capability shows how much capacity is retained at higher charge/discharge rates. "
                "Capacity drop at 2C vs C/5 quantifies power-fade — critical for EV and fast-charge applications."
            )
            _latest = df.iloc[-1]
            _soh_frac = float(_latest["soh_pct"]) / 100.0
            _r_now    = float(_latest.get("resistance_ohm", 0.150))
            _q_nom    = float(df["capacity_ah"].iloc[0])

            # Physics-based C-rate derating: Q(C) = Q_nom × SOH × exp(-k_rate × C^1.4)
            # k_rate scales with resistance: higher R → more rate-sensitive
            _k_rate    = 0.03 * (_r_now / 0.150)
            _c_rates   = [0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]
            _c_labels  = ["C/20", "C/10", "C/5", "C/2", "1C", "2C", "5C"]
            _capacities = [_q_nom * _soh_frac * _np_rate.exp(-_k_rate * c ** 1.4) for c in _c_rates]
            _retention  = [100.0 * cap / max(_capacities[0], 1e-9) for cap in _capacities]

            _fig_rate = go.Figure()
            _fig_rate.add_trace(go.Bar(
                x=_c_labels, y=_retention,
                marker_color=["#48bb78" if r >= 90 else "#f6ad55" if r >= 75 else "#fc8181" for r in _retention],
                text=[f"{r:.1f}%" for r in _retention],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}% retention<extra></extra>",
            ))
            _fig_rate.update_layout(
                **base_layout(
                    height=280, legend=LEGEND_H,
                    xaxis=dict(title="C-rate", zeroline=False),
                    yaxis=dict(title="Capacity Retention (%)", range=[0, 115], zeroline=False),
                ),
            )
            _fig_rate.update_layout(title=dict(text="Rate Capability: Capacity Retention at Current SOH", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(_fig_rate, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "rate_capability"}})

            _ret_2c = _retention[5]
            _ret_msg = f"At 2C (fast charge), this cell retains {_ret_2c:.1f}% of its C/5 capacity."
            _ret_color = "#48bb78" if _ret_2c >= 85 else "#f6ad55" if _ret_2c >= 70 else "#fc8181"
            _md_html(f"""<div style="background:rgba(26,32,44,0.7);border:1px solid #2d3748;border-radius:8px;padding:12px 16px;font-size:13px"><span style="color:{_ret_color};font-weight:700">{_ret_msg}</span><span style="color:#a0aec0;margin-left:10px">k_rate = {_k_rate:.3f} (resistance-scaled). >85% at 2C = good fast-charge fitness; &lt;70% = rate-limited application only.</span></div>""")
        except Exception as _rate_e:
            st.info(f"Rate capability analysis unavailable: {_rate_e}")


# ---------------------------------------------------------------------------
# Physics-Based RUL Projection (PyBaMM SPM) vs GBRT
# ---------------------------------------------------------------------------

def render_physics_rul_expander(df: pd.DataFrame, _rdf: pd.DataFrame, cell_id: str,
                                 rul_reliable: bool, knee: dict,
                                 _agreement, _physics_agreement_note) -> None:
    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    _e1_q10 = float(df["rul_q10"].iloc[-1]) if "rul_q10" in df.columns else None
    _e1_q90 = float(df["rul_q90"].iloc[-1]) if "rul_q90" in df.columns else None

    # C2: surface GBRT vs PyBaMM delta when cache is warm
    _pb2_key = f"pybamm_{cell_id}_{st.session_state.get('eol_threshold_pct', 80)}"
    _pb2_pre = st.session_state.get(_pb2_key, {})
    if (_pb2_pre and not _pb2_pre.get("error") and _pb2_pre.get("rul_physics")
            and rul_reliable and "rul_pred" in latest.index
            and latest["rul_pred"] is not None):
        _gbrt_rul2 = float(latest["rul_pred"])
        _phys_rul2 = int(_pb2_pre["rul_physics"])
        _gap_pct2 = abs(_phys_rul2 - _gbrt_rul2) / max(_gbrt_rul2, 1) * 100
        _gap_col2 = "#48bb78" if _gap_pct2 < 15 else "#f6ad55" if _gap_pct2 < 35 else "#fc8181"
        _conv_label = (
            "✓ Models converge — high confidence" if _gap_pct2 < 15 else
            f"⚡ Models diverge {_gap_pct2:.0f}% — expand for interpretation" if _gap_pct2 < 35 else
            f"⚠ Large divergence {_gap_pct2:.0f}% — review recommended"
        )
        _md_html(
            f"<div style='font-size:12px;margin:4px 0 6px;padding:6px 14px;"
            f"background:#1e2a38;border-radius:6px;border-left:3px solid {_gap_col2}'>"
            f"<span style='color:{_gap_col2};font-weight:600'>{_conv_label}</span>"
            f"<span style='color:#a0aec0;margin-left:10px'>"
            f"Physics: <strong style='color:#68d391'>{_phys_rul2:,} cy</strong> &nbsp;·&nbsp; "
            f"Data: <strong style='color:#f6ad55'>{int(_gbrt_rul2):,} cy</strong></span></div>"
        )
    with st.expander("Model Comparison — Physics (PyBaMM SPM) vs Data (GBRT)", expanded=False):
        _md_html(
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px'>"
            "<div style='background:#0e1117;border:1px solid #2d3748;border-radius:8px;padding:12px'>"
            "<div style='font-size:10px;font-weight:700;color:#68d391;text-transform:uppercase;"
            "letter-spacing:0.08em;margin-bottom:6px'>⚛ PyBaMM SPM — Physics model</div>"
            "<div style='font-size:12px;color:#a0aec0;line-height:1.6'>"
            "Derives nominal capacity from electrochemistry (Single Particle Model), then fits "
            "<code>SOH(n) = 1 − β√n</code> to the measured history. "
            "The √n term reflects diffusion-limited SEI growth slowing over time.<br>"
            "<span style='color:#a0aec0;font-size:11px'>Strong in: early life · new cells · unknown chemistry.<br>"
            "Weak in: post-knee acceleration · non-SEI failure modes.</span>"
            "</div></div>"
            "<div style='background:#0e1117;border:1px solid #2d3748;border-radius:8px;padding:12px'>"
            "<div style='font-size:10px;font-weight:700;color:#f6ad55;text-transform:uppercase;"
            "letter-spacing:0.08em;margin-bottom:6px'>📊 GBRT — Data-driven model</div>"
            "<div style='font-size:12px;color:#a0aec0;line-height:1.6'>"
            "Gradient-boosted trees trained across the full cell population under leave-cell-out "
            "validation. Extrapolates the current rolling fade rate forward linearly.<br>"
            "<span style='color:#a0aec0;font-size:11px'>Strong in: cells with rich history · mid–late life.<br>"
            "Weak in: early life (fade still decelerating) · sparse data.</span>"
            "</div></div>"
            "</div>"
        )
        _pybamm_cache_key = f"pybamm_{cell_id}_{st.session_state.get('eol_threshold_pct', 80)}"
        if _pybamm_cache_key not in st.session_state:
            with st.spinner("Running PyBaMM SPM — first run only, cached after…"):
                try:
                    import sys as _sys2
                    _src2 = os.path.join(os.path.dirname(__file__), "..", "src")
                    if _src2 not in _sys2.path:
                        _sys2.path.insert(0, _src2)
                    from pybamm_rul import project_rul as _project_rul
                    _eol_thr = float(st.session_state.get("eol_threshold_pct", 80.0))
                    _pb_res  = _project_rul(
                        cell_id, df,
                        data_mode=st.session_state.get("data_mode", "synthetic"),
                        eol_threshold=_eol_thr,
                        project_cycles=600,
                    )
                    st.session_state[_pybamm_cache_key] = _pb_res
                except Exception as _pb_install_err:
                    st.session_state[_pybamm_cache_key] = {"error": str(_pb_install_err)}

        _pb = st.session_state.get(_pybamm_cache_key, {})
        if _pb.get("error"):
            st.info(f"Physics projection unavailable: {_pb['error']}")
        elif _pb.get("proj_cycles"):
            # ── Parameter summary strip ──
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric("Parameter set", _pb["chem_label"].split("(")[0].strip())
            _pc2.metric("SPM nominal capacity", f"{_pb['spm_capacity_ah']:.3f} Ah")
            _pc3.metric("Fitted β (SEI)", f"{_pb['beta']:.5f}")
            _rul_ph = _pb.get("rul_physics")
            _pc4.metric("Physics RUL", f"{_rul_ph:,} cy" if _rul_ph is not None else "—")

            # ── SEI/LAM decomposition (src/physics_calibration.py) ──
            # Reuses the calibration already computed for the mechanism-
            # agreement note above _mc_left, rather than re-fitting here.
            _phys_cal = (_agreement or {}).get("physics")
            if _phys_cal is None:
                try:
                    from physics_calibration import calibrate_cell as _calibrate_cell_local
                    _phys_cal = _calibrate_cell_local(cell_id, df)
                except Exception:
                    _phys_cal = None
            if _phys_cal and _phys_cal.get("eligible") and _phys_cal.get("error") is None:
                st.caption(
                    "SEI/LAM decomposition — two-term fit splitting the single β above into "
                    "an SEI/LLI channel (√n) and an active-material-loss channel (linear-in-n):"
                )
                _pd1, _pd2, _pd3, _pd4 = st.columns(4)
                _pd1.metric("β SEI (√n)", f"{_phys_cal['beta_sei']:.5f}")
                _pd2.metric("β LAM (linear)", f"{_phys_cal['beta_lam']:.6f}")
                _pd3.metric(
                    "SEI resistance growth (k_r)",
                    f"{_phys_cal['k_r']:.4f}" if _phys_cal.get("k_r") is not None else "— (no IR data)",
                )
                _pd4.metric("Physics dominant mode", _phys_cal["dominant_mode_label"].split("(")[0].strip())
                if _physics_agreement_note:
                    st.caption(f"⚠ {_physics_agreement_note}")
                elif _agreement and _agreement.get("agree"):
                    st.caption(f"✓ {_agreement['note']}")

            # ── Projection chart ──
            _fig_pb = go.Figure()

            # Measured SOH history (resampled)
            _fig_pb.add_trace(go.Scatter(
                x=_rdf["cycle_number"].tolist(),
                y=_rdf["soh_pct"].tolist(),
                name="Measured SOH",
                line=dict(color="#63b3ed", width=2),
                hovertemplate="Cycle %{x}: %{y:.1f}% SOH<extra>Measured</extra>",
            ))

            # Physics projection — 95% uncertainty band (β ± 2σ)
            if _pb.get("proj_soh_lo") and _pb.get("proj_soh_hi"):
                _fig_pb.add_trace(go.Scatter(
                    x=_pb["proj_cycles"] + list(reversed(_pb["proj_cycles"])),
                    y=_pb["proj_soh_hi"] + list(reversed(_pb["proj_soh_lo"])),
                    fill="toself",
                    fillcolor="rgba(104,211,145,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Physics 95% CI (β ± 2σ)",
                    showlegend=True,
                    hoverinfo="skip",
                ))
            # Physics projection central line
            _fig_pb.add_trace(go.Scatter(
                x=_pb["proj_cycles"],
                y=_pb["proj_soh"],
                name="Physics projection (SPM · SEI)",
                line=dict(color="#68d391", width=2, dash="dot"),
                hovertemplate="Cycle %{x}: %{y:.1f}% SOH<extra>SPM central</extra>",
            ))

            # GBRT ML estimate as comparison (linear fade forward)
            _fade_50 = float(latest.get("fade_rate_50cy", 0)) if hasattr(latest, "get") else float(latest["fade_rate_50cy"]) if "fade_rate_50cy" in latest.index else 0
            if _fade_50 > 0 and _rul_ph:
                _ml_fut_cy  = _pb["proj_cycles"][:min(len(_pb["proj_cycles"]), _rul_ph + 50)]
                _ml_fut_soh = [current_soh - _fade_50 * 100 * (c - int(latest["cycle_number"])) for c in _ml_fut_cy]
                _ml_fut_soh = [max(0, s) for s in _ml_fut_soh]
                _fig_pb.add_trace(go.Scatter(
                    x=_ml_fut_cy,
                    y=_ml_fut_soh,
                    name="ML linear extrapolation (GBRT fade)",
                    line=dict(color="#f6ad55", width=1, dash="dash"),
                    hovertemplate="Cycle %{x}: %{y:.1f}% SOH<extra>GBRT</extra>",
                ))

            # GBRT 80% RUL interval (Q10/Q90), translated onto the SOH axis --
            # the real trained quantile models (batlab.models.gbrt's
            # rul_q10_model/rul_q90_model), not a heuristic. Q10/Q90 are
            # cycles-remaining-to-EOL estimates, so each traces a straight
            # line from (current cycle, current SOH) to (current cycle +
            # Q10 or Q90, EOL threshold) -- an honest, if approximate,
            # translation of a point interval into a trajectory cone, shown
            # alongside the physics β±2σ band above rather than replacing it
            # (Phase 6 physics-informed intelligence -- confidence intervals
            # from two independent sources, side by side).
            _gbrt_eol_soh = float(st.session_state.get("eol_threshold_pct", 80.0))
            if _e1_q10 is not None and _e1_q90 is not None and _e1_q10 >= 0 and _e1_q90 >= 0:
                _gbrt_cur_cy = int(latest["cycle_number"])
                _gbrt_lo_cy  = _gbrt_cur_cy + _e1_q10   # optimistic: EOL reached sooner (Q10)
                _gbrt_hi_cy  = _gbrt_cur_cy + _e1_q90   # pessimistic: EOL reached later (Q90)
                _fig_pb.add_trace(go.Scatter(
                    x=[_gbrt_cur_cy, _gbrt_hi_cy, _gbrt_lo_cy, _gbrt_cur_cy],
                    y=[current_soh, _gbrt_eol_soh, _gbrt_eol_soh, current_soh],
                    fill="toself",
                    fillcolor="rgba(246,173,85,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="GBRT 80% RUL interval (Q10–Q90)",
                    showlegend=True,
                    hoverinfo="skip",
                ))

            # Temperature sensitivity bands (H3) — β(T) = β₀·exp(−Ea/RT)
            _temp_colours = {"15°C": "#63b3ed", "25°C": "#68d391", "35°C": "#fc8181"}
            for _t_label, _t_soh in _pb.get("temp_scenarios", {}).items():
                if _t_label == "25°C":
                    continue   # 25°C is already the central line
                _fig_pb.add_trace(go.Scatter(
                    x=_pb["proj_cycles"], y=_t_soh,
                    name=f"SPM @ {_t_label}",
                    line=dict(color=_temp_colours.get(_t_label, "#a0aec0"), width=1, dash="longdash"),
                    opacity=0.6,
                    hovertemplate=f"Cycle %{{x}}: %{{y:.1f}}% SOH<extra>SPM {_t_label}</extra>",
                ))

            # EOL threshold line
            _eol_thr_val = float(st.session_state.get("eol_threshold_pct", 80.0))
            _fig_pb.add_hline(
                y=_eol_thr_val,
                line_dash="dash", line_color="#fc8181", line_width=1,
                annotation_text=f"EOL {_eol_thr_val:.0f}%",
                annotation_font=dict(color="#fc8181", size=10),
            )

            _fig_pb.update_layout(
                **base_layout(
                    height=340,
                    xaxis=dict(title="Cycle"),
                    yaxis=dict(title="SOH %",
                               range=[max(0, min(_pb["proj_soh"] or [60]) - 5), 102]),
                ),
            )
            _fig_pb.update_layout(
                legend=dict(font=dict(size=10, color="#718096")),
                title=dict(
                    text=f"PyBaMM SPM · {_pb['chem_label']} · SEI growth fade",
                    font=dict(size=12, color="#a0aec0"), x=0,
                ),
            )
            st.plotly_chart(_fig_pb, use_container_width=True)

            # ── Model comparison diagnostic ───────────────────────────────
            # Pull GBRT RUL prediction for this cell if available
            _rul_ml = None
            if "rul_pred" in latest.index:
                _v = latest["rul_pred"]
                if _v == _v and _v is not None:   # not NaN
                    _rul_ml = int(float(_v))

            _rul_ph_val = _rul_ph or 0
            _rul_diff   = _rul_ph_val - (_rul_ml or _rul_ph_val)
            _rul_diff_pct = abs(_rul_diff) / max(_rul_ph_val, 1) * 100

            # Determine cell life regime from fade trajectory
            _fade_30_val  = float(latest["fade_rate_30cy"]) if "fade_rate_30cy" in latest.index else 0
            _fade_50_val  = float(latest["fade_rate_50cy"]) if "fade_rate_50cy" in latest.index else 0
            _fade_accel   = (_fade_30_val - _fade_50_val) / max(_fade_50_val, 1e-9)
            _knee_detected = knee.get("detected", False) if isinstance(knee, dict) else False

            if current_soh >= 90:
                _regime = "early"
                _regime_label = "Early life (SOH ≥ 90%)"
            elif _knee_detected or _fade_accel > 0.25:
                _regime = "post_knee"
                _regime_label = "Post-knee / accelerating fade"
            elif current_soh >= 80:
                _regime = "mid"
                _regime_label = "Mid life (80–90% SOH)"
            else:
                _regime = "late"
                _regime_label = "Late life (SOH < 80%)"

            # Build physics model explanation
            _phys_explain = {
                "early": (
                    f"At {current_soh:.1f}% SOH the cell is still in early life. "
                    f"SEI growth is diffusion-limited — the rate of Li inventory loss slows as "
                    f"1/(2√n) because the SEI layer itself impedes further electrolyte reduction. "
                    f"With β = {_pb['beta']:.5f}, the model predicts the fade rate will "
                    f"<strong style='color:#68d391'>continue to decelerate</strong>. "
                    f"This is the regime where the √n model is most physically accurate."
                ),
                "mid": (
                    f"At {current_soh:.1f}% SOH the cell is in mid life — the √n model is a "
                    f"reasonable approximation but the SEI layer is now thick enough that "
                    f"secondary mechanisms (electrolyte depletion, active material loss) "
                    f"may be starting to contribute. β = {_pb['beta']:.5f} was fitted to the "
                    f"full measured history; if fade has been <em>accelerating recently</em>, "
                    f"this β underestimates future loss."
                ),
                "post_knee": (
                    f"The cell has passed its knee point (or 30-cycle fade rate exceeds "
                    f"50-cycle rate by >{_fade_accel*100:.0f}%). The √n SEI model "
                    f"<strong style='color:#fc8181'>does not capture this acceleration</strong> — "
                    f"SEI-only models have no mechanism for lithium plating, particle cracking, "
                    f"or electrolyte depletion, all of which cause superlinear fade. "
                    f"β = {_pb['beta']:.5f} was fitted to the full history including early slow fade, "
                    f"so the physics projection will be <strong style='color:#fc8181'>over-optimistic</strong>."
                ),
                "late": (
                    f"At {current_soh:.1f}% SOH the cell is in late life. The SEI model "
                    f"(β = {_pb['beta']:.5f}) was calibrated on the full cycle history, but "
                    f"late-life degradation is typically dominated by particle cracking and "
                    f"electrolyte depletion — neither appears in the SPM. "
                    f"The physics projection should be treated as an <em>optimistic bound</em> only."
                ),
            }[_regime]

            # Build ML model explanation
            _ml_explain = (
                f"The GBRT model extrapolates the <strong>current 50-cycle rolling fade rate "
                f"({_fade_50_val*1000:.3f} mSOH/cy)</strong> forward linearly. "
                f"It learned from {'{}'} cells under leave-cell-out validation, so it captures "
                f"population-level degradation patterns — but it has no electrochemical mechanism. "
                f"Linear extrapolation overestimates remaining life in early aging (fade is still "
                f"decelerating) and may underestimate it mid-life if the current window happens to "
                f"fall on a temporary rough patch."
            ).format("N")

            # Build divergence explanation
            if _rul_ml is None:
                _div_explain = (
                    "ML RUL was not available for this cell (reliability gate not met or "
                    "model not calibrated). Physics-only estimate shown."
                )
                _div_colour = "#718096"
            elif _rul_diff_pct < 10:
                _div_explain = (
                    f"The two estimates agree within {_rul_diff_pct:.0f}% "
                    f"({_rul_ph_val} vs {_rul_ml} cycles). "
                    f"This is expected in mid life when the √n curve is approximately linear "
                    f"over short windows. <strong style='color:#68d391'>High confidence "
                    f"in the {_rul_ph_val}-cycle estimate.</strong>"
                )
                _div_colour = "#68d391"
            elif _rul_diff > 0:
                _div_explain = (
                    f"Physics estimates <strong style='color:#68d391'>{_rul_ph_val} cycles</strong> "
                    f"vs ML <strong style='color:#f6ad55'>{_rul_ml} cycles</strong> "
                    f"— physics is more optimistic by {_rul_diff} cycles ({_rul_diff_pct:.0f}%). "
                    + (
                        "In early life this is expected: the √n model predicts ongoing deceleration "
                        "that the linear GBRT cannot see. <strong>Use the physics estimate with caution "
                        "until fade stabilises.</strong>"
                        if _regime == "early" else
                        "This divergence in mid-to-late life may indicate the GBRT is "
                        "over-weighting a recent high-fade window. "
                        "<strong>Conservative engineering decision: use the ML (lower) estimate.</strong>"
                    )
                )
                _div_colour = "#f6ad55"
            else:
                _div_explain = (
                    f"ML estimates <strong style='color:#68d391'>{_rul_ml} cycles</strong> "
                    f"vs physics <strong style='color:#fc8181'>{_rul_ph_val} cycles</strong> "
                    f"— ML is more optimistic by {-_rul_diff} cycles ({_rul_diff_pct:.0f}%). "
                    + (
                        "Post-knee, the √n model's fitted β was anchored to early slow fade "
                        "and is now over-projecting. The GBRT is responding to the recent "
                        "acceleration. <strong style='color:#fc8181'>Use the ML (lower) estimate "
                        "— the physics model is not capturing the acceleration mechanism.</strong>"
                        if _regime == "post_knee" else
                        "The SEI model predicts faster fade than the current GBRT window. "
                        "Worth investigating whether recent cycles show a change in operating "
                        "conditions (temperature, C-rate, depth of discharge)."
                    )
                )
                _div_colour = "#fc8181"

            # Render 3-section card
            _md_html(
                f"<div style='background:#111827;border:1px solid #2d3748;border-radius:10px;"
                f"padding:18px 20px;margin-top:8px;font-size:12px;line-height:1.7'>"

                # Section A — Physics model
                f"<div style='margin-bottom:14px'>"
                f"<div style='font-size:10px;font-weight:700;color:#68d391;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:5px'>⚛ PyBaMM SPM · SEI Growth Model</div>"
                f"<div style='color:#a0aec0'>{_phys_explain}</div>"
                f"<div style='color:#a0aec0;font-size:11px;margin-top:4px'>"
                f"Parameter set: {_pb['chem_label']} &nbsp;·&nbsp; "
                f"β = {_pb['beta']:.5f} &nbsp;·&nbsp; "
                f"Nominal capacity (SPM): {_pb['spm_capacity_ah']:.3f} Ah"
                f"</div></div>"

                # Section B — ML model
                f"<div style='border-top:1px solid #1e2a38;padding-top:12px;margin-bottom:14px'>"
                f"<div style='font-size:10px;font-weight:700;color:#f6ad55;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:5px'>📊 GBRT · Linear Extrapolation</div>"
                f"<div style='color:#a0aec0'>{_ml_explain}</div>"
                f"<div style='color:#a0aec0;font-size:11px;margin-top:4px'>"
                f"Current fade rate: {_fade_50_val*1000:.3f} mSOH/cy (50-cy rolling) &nbsp;·&nbsp; "
                f"Regime: {_regime_label}"
                f"</div></div>"

                # Section C — Divergence verdict
                f"<div style='border-top:1px solid #1e2a38;padding-top:12px;"
                f"background:{_div_colour}0d;border-radius:6px;padding:12px 14px;margin-top:2px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_div_colour};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:5px'>⚖ Model Verdict</div>"
                f"<div style='color:#a0aec0'>{_div_explain}</div>"
                f"</div>"

                f"</div>"
            )


# ---------------------------------------------------------------------------
# Orchestrator — called once from page_health() when Engineering details is on
# ---------------------------------------------------------------------------

def render_engineering_diagnostics(
    df: pd.DataFrame, _rdf: pd.DataFrame, cell_id: str,
    bundle: dict, rul_reliable: bool,
    knee: dict, _mech: dict, _agreement, _physics_agreement_note,
) -> None:
    render_fade_rate_and_phase_analysis(df, _rdf, cell_id, knee)
    render_data_lineage_expander(df, cell_id)
    render_prediction_drivers_expander(df, cell_id, bundle)

    _md_html(
        "<div style='font-size:10px;font-weight:700;color:#2d3748;text-transform:uppercase;"
        "letter-spacing:0.12em;margin:32px 0 4px;padding-bottom:4px;border-bottom:1px solid #1e2a38'>"
        "Evidence — supporting diagnostics</div>"
    )
    render_mechanism_classifier_expander(_mech)
    render_dqdv_expander(df, _rdf, cell_id)
    render_ce_expander(df, cell_id)

    _md_html(
        "<div style='font-size:10px;font-weight:700;color:#2d3748;text-transform:uppercase;"
        "letter-spacing:0.12em;margin:24px 0 4px;padding-bottom:4px;border-bottom:1px solid #1e2a38'>"
        "Advanced diagnostics</div>"
    )
    render_energy_throughput_expander(df, _rdf, cell_id)
    render_lithium_plating_expander(df)

    _md_html(
        "<div style='font-size:10px;font-weight:700;color:#2d3748;text-transform:uppercase;"
        "letter-spacing:0.12em;margin:32px 0 4px;padding-bottom:4px;border-bottom:1px solid #1e2a38'>"
        "Advanced analysis — physics models &amp; expert diagnostics</div>"
    )
    render_resistance_proxy_expander(df, _rdf, cell_id)
    render_formation_protocol_expander(df, cell_id)
    render_rate_capability_expander(df, cell_id)
    render_physics_rul_expander(df, _rdf, cell_id, rul_reliable, knee, _agreement, _physics_agreement_note)
