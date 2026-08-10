"""Compliance page — Stakeholder View tab.

Same slicing src/stakeholder_views.py's module docstring describes: the
same cell's data, sliced for an OEM / operator / recycler party. This
Streamlit tab is a preview/internal-tooling surface (what does each
party actually get) — the real "shared, external-facing" mechanism is
the matching REST endpoints in src/api.py (GET /cells/{id}/view/{type}),
gated by this platform's existing JWT auth, same as any other API
consumer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _md_html, render_card
from chemistry_profiles import ChemistryProfile


def page_stakeholder_view(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool) -> None:
    from stakeholder_views import build_oem_view, build_operator_view, build_recycler_view
    from recommendations import diagnose_mechanism

    st.markdown("# Stakeholder View")
    st.markdown(f"#### The same cell, sliced for who's actually asking · {selected}")
    _md_html(
        "<div style='background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);"
        "border-radius:10px;padding:14px 20px;margin-bottom:24px;"
        "font-size:13px;color:#8896a8;line-height:1.7'>"
        "<strong style='color:#63b3ed'>Preview only.</strong> This tab previews what each "
        "stakeholder's slice looks like. The real sharing mechanism is this platform's REST API "
        "(<code>GET /cells/{cell_id}/view/{oem|operator|recycler}</code>) — gated by the same JWT "
        "auth as every other endpoint, not a new external login system."
        "</div>"
    )

    _profile = ChemistryProfile.for_cell(selected)
    source = _profile.source_kind
    latest = df.iloc[-1]
    soh = float(latest["soh_pct"])
    cycle_count = int(latest["cycle_number"])
    fade_30 = float(latest.get("fade_rate_30cy", 0.0))
    fade_50 = float(latest.get("fade_rate_50cy", fade_30))
    sop_pct = float(latest["sop_pct"]) if "sop_pct" in latest.index and pd.notna(latest["sop_pct"]) else None
    rul_pred = float(latest["rul_pred"]) if rul_reliable and "rul_pred" in latest.index and pd.notna(latest["rul_pred"]) else None
    rul_q10 = float(latest["rul_q10"]) if rul_reliable and "rul_q10" in latest.index and pd.notna(latest["rul_q10"]) else None
    rul_q90 = float(latest["rul_q90"]) if rul_reliable and "rul_q90" in latest.index and pd.notna(latest["rul_q90"]) else None
    mechanism = diagnose_mechanism(df)

    stakeholder = st.radio(
        "View as", ["OEM (manufacturer)", "Operator (current owner)", "Recycler"],
        horizontal=True, key="stakeholder_view_radio",
    )

    if stakeholder == "OEM (manufacturer)":
        fields = build_oem_view(
            selected, _profile.short_name, source, soh, cycle_count, fade_30,
            mechanism=mechanism, rul_reliable=rul_reliable,
            rul_pred=rul_pred, rul_q10=rul_q10, rul_q90=rul_q90,
        )
    elif stakeholder == "Operator (current owner)":
        fields = build_operator_view(
            selected, _profile.short_name, source, soh, cycle_count, fade_30, fade_50, None,
            rul_reliable=rul_reliable, rul_pred=rul_pred, rul_q10=rul_q10, rul_q90=rul_q90,
            sop_pct=sop_pct,
        )
    else:
        _region = st.selectbox("Recycler region preference", ["North America", "Europe", "Asia"], key="stakeholder_recycler_region")
        fields = build_recycler_view(selected, _profile.short_name, soh, fade_30, sop_pct=sop_pct, user_region=_region)

    _STATE_COLOUR = {"available": "#48bb78", "estimated": "#f6ad55", "unavailable": "#718096"}
    for f in fields:
        _colour = _STATE_COLOUR[f["state"]]
        render_card(
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
            f"<div style='font-size:13px;color:#a0aec0'>{f['label']}</div>"
            f"<div style='font-size:10px;font-weight:700;color:{_colour};text-transform:uppercase'>{f['state']}</div>"
            f"</div>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0;margin-top:4px'>{f['value']}</div>"
            + (f"<div style='font-size:11px;color:#8896a8;margin-top:4px'>{f['note']}</div>" if f.get("note") else ""),
            padding="12px 16px",
            extra_style="margin-bottom:8px",
        )
