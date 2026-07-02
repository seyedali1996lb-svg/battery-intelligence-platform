import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # app/ dir for utils, sidebar
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import (
    base_layout, LEGEND_H, PLOTLY_CONFIG, _md_html, soh_status,
    friendly, _soh_sparkline_svg, _cell_provenance, _analysis_provenance,
    NASA_CELL_IDS,
)
from design_system import (
    make_badge, make_state_badge, provenance_banner,
    BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST, BADGE_UNAVAIL,
    BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC,
    ACTION_META, CONF_META,
)


def _passport_field_row(f: dict) -> str:
    muted = f["state"] == "unavailable"
    value_colour = "#4a5568" if muted else "#e2e8f0"
    note_html = (
        f"<div style='font-size:11px;color:#4a5568;margin-top:3px;line-height:1.5'>{f['note']}</div>"
        if f.get("note") else "<div style='height:0'></div>"
    )
    font_style = "italic" if muted else "normal"
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"gap:16px;padding:12px 0;border-bottom:1px solid #2d3748'>"
        f"<div style='flex:1;min-width:0'>"
        f"<div style='font-size:12px;color:#718096'>{f['label']}</div>"
        f"<div style='font-size:14px;font-weight:600;color:{value_colour};margin-top:2px;"
        f"font-style:{font_style}'>{f['value']}</div>"
        f"{note_html}"
        f"</div>"
        f"<div style='flex-shrink:0;padding-top:2px'>{make_state_badge(f['state'])}</div>"
        f"</div>"
    )


def page_passport(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport

    is_nasa = selected in NASA_CELL_IDS
    p = build_passport(selected, df, bundle, rul_reliable, is_nasa)
    summ = p["summary"]

    st.markdown("# Battery Passport")
    st.markdown(f"##### Battery Passport Interface · {selected}")

    _md_html(
        f"""
        <div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);
                    border-radius:10px;padding:14px 20px;margin-bottom:28px;
                    font-size:13px;color:#718096;line-height:1.7">
            <strong style="color:#63b3ed">Battery Passport Interface</strong> — demonstrating the
            EU Battery Regulation (EU) 2023/1542 data structure. This is <strong>not</strong> a
            compliance claim: every field below is marked {make_state_badge("available")},
            {make_state_badge("estimated")}, or {make_state_badge("unavailable")} based on what this
            demonstration actually has. Nothing is hidden or faked to look complete.
        </div>
        """
    )

    groups = [
        ("identity",  "1 · Battery Identity"),
        ("soh",       "2 · State of Health"),
        ("lifecycle", "3 · Lifecycle History"),
        ("carbon",    "4 · Carbon Footprint"),
    ]
    for key, title in groups:
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
            f"margin-bottom:4px;margin-top:20px'>{title}</div>",
            unsafe_allow_html=True,
        )
        rows_html = "".join(_passport_field_row(f) for f in p[key])
        st.markdown(f"<div>{rows_html}</div>", unsafe_allow_html=True)

    # ── 5: Critical Raw Materials (EU Battery Regulation Art. 13) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>5 · Critical Raw Materials (EU Reg. 2023/1542 Art. 13)</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Configure cell composition (optional)", expanded=False):
        _crm_col1, _crm_col2, _crm_col3 = st.columns(3)
        with _crm_col1:
            st.number_input("Cobalt content (wt%)", 0.0, 100.0, step=0.1, key="crm_co_pct",
                            value=st.session_state.get("crm_co_pct", 0.0))
        with _crm_col2:
            st.number_input("Nickel content (wt%)", 0.0, 100.0, step=0.1, key="crm_ni_pct",
                            value=st.session_state.get("crm_ni_pct", 0.0))
        with _crm_col3:
            st.number_input("Lithium content (wt%)", 0.0, 100.0, step=0.1, key="crm_li_pct",
                            value=st.session_state.get("crm_li_pct", 0.0))
        st.caption("Enter values to replace default stoichiometric estimates. Leave at 0.0 to keep defaults.")

    _co_pct = float(st.session_state.get("crm_co_pct", 0.0))
    _ni_pct = float(st.session_state.get("crm_ni_pct", 0.0))
    _li_pct = float(st.session_state.get("crm_li_pct", 0.0))

    _co_value  = f"{_co_pct:.1f} wt% (user-entered)" if _co_pct > 0 else "~14 wt% (LiCoO₂ cathode, est.)"
    _co_state  = "available" if _co_pct > 0 else "estimated"
    _ni_value  = f"{_ni_pct:.1f} wt% (user-entered)" if _ni_pct > 0 else "~0 wt% (pure LiCoO₂ baseline)"
    _ni_state  = "available" if _ni_pct > 0 else "estimated"
    _li_value  = f"{_li_pct:.1f} wt% (user-entered)" if _li_pct > 0 else "~7 wt% (cathode + anode combined)"
    _li_state  = "available" if _li_pct > 0 else "estimated"

    # EU recycled content targets — use user-entered values when available
    _rec_co_note = "Requires manufacturer supply-chain records. Threshold: 16% by 2031."
    _rec_ni_note = "Threshold: 6% by 2031 (Annex X, EU 2023/1542)."
    if _co_pct > 0:
        _rec_co_note = (f"User-entered Co content: {_co_pct:.1f} wt%. "
                        f"Recycled Co content record still requires supply-chain audit. Threshold: 16% by 2031.")
    if _ni_pct > 0:
        _rec_ni_note = (f"User-entered Ni content: {_ni_pct:.1f} wt%. "
                        f"Recycled Ni content record still requires supply-chain audit. Threshold: 6% by 2031.")

    _crm_fields = [
        {"label": "Cobalt (Co) content", "value": _co_value, "state": _co_state,
         "note": "Estimated from LiCoO₂ stoichiometry (or user-entered above). Art. 13 requires supply-chain due diligence from 2026."},
        {"label": "Nickel (Ni) content", "value": _ni_value, "state": _ni_state,
         "note": "NMC variants would show 15–33 wt%. Supply chain disclosure required."},
        {"label": "Lithium (Li) content", "value": _li_value, "state": _li_state,
         "note": "Calculated from stoichiometry (or user-entered above). Recycled Li content threshold: 4% by 2027, 10% by 2031 (Annex X)."},
        {"label": "Recycled Co content", "value": "Not available in demo", "state": "unavailable",
         "note": _rec_co_note},
        {"label": "Recycled Ni content", "value": "Not available in demo", "state": "unavailable",
         "note": _rec_ni_note},
        {"label": "Article 52 due diligence", "value": "Not assessed", "state": "unavailable",
         "note": "Third-party audit of Co/Ni/Li supply chain required for market access in EU."},
    ]
    _crm_html = "".join(_passport_field_row(f) for f in _crm_fields)
    st.markdown(f"<div>{_crm_html}</div>", unsafe_allow_html=True)

    # ── 6: End-of-Life R-code Taxonomy ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>6 · End-of-Life R-Code Taxonomy (IEC 62902 / EU Art. 70)</div>",
        unsafe_allow_html=True,
    )
    _soh_now = float(df.iloc[-1]["soh_pct"]) if "soh_pct" in df.columns else 85.0
    _eol_r_code = (
        "R3 — Remanufacture / Second-life application" if _soh_now >= 80
        else "R4 — Recycle (hydrometallurgical / direct)" if _soh_now >= 60
        else "R5 — Recover (energy or material)"
    )
    _eol_color  = "#48bb78" if _soh_now >= 80 else "#f6ad55" if _soh_now >= 60 else "#fc8181"
    _r_fields = [
        {"label": "Recommended R-code", "value": _eol_r_code, "state": "estimated",
         "note": f"Based on current SOH = {_soh_now:.1f}%. IEC 62902 R0–R9 taxonomy."},
        {"label": "R0 — Reuse", "value": "SOH ≥ 90% required", "state": "estimated" if _soh_now >= 90 else "unavailable"},
        {"label": "R3 — Second-life", "value": "SOH 80–90% (stationary storage)", "state": "available" if 80 <= _soh_now < 90 else "estimated"},
        {"label": "R4 — Recycle", "value": "Hydromet / direct recycling pathway", "state": "estimated"},
        {"label": "Recycled content declaration", "value": "IEC 63338 audit required", "state": "unavailable",
         "note": "Carbon footprint per kWh must be declared for market access (IEC 63338, from 2025)."},
    ]
    _r_html = "".join(_passport_field_row(f) for f in _r_fields)
    st.markdown(f"<div style='margin-bottom:4px'><span style='color:{_eol_color};font-weight:700;font-size:13px'>Recommended: {_eol_r_code}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div>{_r_html}</div>", unsafe_allow_html=True)

    # ── 7: Compliance Status (prose, no badge) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:20px'>7 · Compliance Status</div>",
        unsafe_allow_html=True,
    )
    _md_html(
        f"""
        <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px;
                    font-size:13px;color:#a0aec0;line-height:1.8">
            <strong style="color:#e2e8f0">This is a data-structure demonstration, not a regulatory
            submission.</strong><br><br>
            Of {summ['n_total']} fields modelled on the EU Battery Regulation's data requirements:
            <strong style="color:#48bb78">{summ['n_available']} are available</strong> from this
            platform's validated pipeline, <strong style="color:#d69e2e">{summ['n_estimated']} are
            cited estimates</strong> from the Consequences module, and
            <strong style="color:#718096">{summ['n_unavailable']} are not available</strong> in
            this demonstration.<br><br>
            An actual regulatory submission under (EU) 2023/1542 would additionally require:
            manufacturer-submitted identity and supply-chain records, a third-party accredited
            carbon footprint audit, repair/refurbishment history tracking, and notified-body
            sign-off — none of which a portfolio project can provide. No field on this page should
            be read as a compliance claim.
        </div>
        """
    )
