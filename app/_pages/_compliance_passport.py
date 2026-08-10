"""Compliance page — EU Battery Passport tab.

Extracted from compliance.py's page_passport() (was 1,059 lines -- the
4th largest file in app/_pages/). This function was already fully
self-contained (only needs selected/df/bundle/rul_reliable, no shared
state with the Reports/Sustainability/Regulatory-Alerts tabs also defined
in that file), so this is a clean move with no parameter re-threading.
page_compliance() imports it here for the "EU Battery Passport" tab.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _md_html, render_card
from design_system import make_state_badge


def _passport_field_row(f: dict) -> str:
    muted = f["state"] == "unavailable"
    value_colour = "#4a5568" if muted else "#e2e8f0"
    note_html = (
        f"<div style='font-size:11px;color:#a0aec0;margin-top:3px;line-height:1.5'>{f['note']}</div>"
        if f.get("note") else "<div style='height:0'></div>"
    )
    font_style = "italic" if muted else "normal"
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"gap:16px;padding:12px 0;border-bottom:1px solid #2d3748'>"
        f"<div style='flex:1;min-width:0'>"
        f"<div style='font-size:12px;color:#8896a8'>{f['label']}</div>"
        f"<div style='font-size:14px;font-weight:600;color:{value_colour};margin-top:2px;"
        f"font-style:{font_style}'>{f['value']}</div>"
        f"{note_html}"
        f"</div>"
        f"<div style='flex-shrink:0;padding-top:2px'>{make_state_badge(f['state'])}</div>"
        f"</div>"
    )


def page_passport(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport

    p = build_passport(selected, df, bundle, rul_reliable)
    summ = p["summary"]

    st.markdown("# Battery Passport")
    st.markdown(f"#### Battery Passport Interface · {selected}")

    # ── E8: Passport completeness score ───────────────────────────────────────
    _e8_n_avail = summ.get("n_available", 0)
    _e8_n_est   = summ.get("n_estimated", 0)
    _e8_n_unavail = summ.get("n_unavailable", 0)
    _e8_total   = max(1, summ.get("n_total", _e8_n_avail + _e8_n_est + _e8_n_unavail))
    _e8_score   = round((_e8_n_avail + _e8_n_est * 0.5) / _e8_total * 100)

    # Proactive webhook push — once per cell per session, only below threshold.
    _wh_url_pg  = st.session_state.get("webhook_url", "")
    _wh_evts_pg = st.session_state.get("webhook_events", [])
    _PASSPORT_GAP_THRESHOLD = 60
    if _wh_url_pg and "PASSPORT_GAP" in _wh_evts_pg and _e8_score < _PASSPORT_GAP_THRESHOLD:
        if "_alerted_passport_gap_cells" not in st.session_state:
            st.session_state["_alerted_passport_gap_cells"] = set()
        if selected not in st.session_state["_alerted_passport_gap_cells"]:
            from notifications import send_webhook
            send_webhook(
                "PASSPORT_GAP",
                {
                    "cell_id": selected, "completeness_pct": _e8_score,
                    "n_available": _e8_n_avail, "n_estimated": _e8_n_est,
                    "n_unavailable": _e8_n_unavail,
                },
                _wh_url_pg, st.session_state.get("webhook_secret", ""),
            )
            st.session_state["_alerted_passport_gap_cells"].add(selected)

    _e8_col1, _e8_col2 = st.columns([3, 1])
    with _e8_col1:
        _e8_bar_colour = "#48bb78" if _e8_score >= 70 else ("#f6ad55" if _e8_score >= 40 else "#fc8181")
        st.markdown(
            f"<div style='margin-bottom:4px;font-size:12px;color:#a0aec0'>"
            f"Passport completeness: <strong style='color:{_e8_bar_colour}'>{_e8_score}%</strong> "
            f"&nbsp;·&nbsp; {_e8_n_avail} available · {_e8_n_est} estimated · {_e8_n_unavail} unavailable</div>",
            unsafe_allow_html=True,
        )
        st.progress(_e8_score / 100)
    with _e8_col2:
        if _e8_n_unavail > 0:
            with st.expander(f"Fill {_e8_n_unavail} gaps", expanded=False):
                # Keyed by build_passport()'s real "label" text -- its field
                # dicts never carry a "field" key (only label/value/state/
                # note), so the previous version of this map (keyed by
                # field-id strings like "manufacturer"/"carbon_footprint_kg_co2"
                # that build_passport() never actually emits) always missed
                # and silently fell through to the generic default tip for
                # every single gap, regardless of which one it was.
                _E8_HOW_TO: dict[str, str] = {
                    "Manufacturer": "Set in sidebar → Cell Metadata → Manufacturer",
                    "Serial number / production batch": "Set in sidebar → Cell Metadata → Serial number",
                    "Chemistry type": "Detected automatically once enough cycle data is loaded",
                    "Nominal capacity": "Detected automatically once the first full cycle is loaded",
                    "Test date range": "Requires timestamped cycle data — not present in this dataset",
                    "Usage profile / test conditions": "Set on Import page → Advanced → Usage profile",
                    "Repair / refurbishment history": "Neither dataset models repair events — not fillable in this demonstration",
                    "Prior-owner / installation history": "Not tracked by this platform — not fillable in this demonstration",
                    "Full lifecycle carbon audit (mining → manufacture → transport → use → end-of-life)":
                        "Requires a real third-party accredited audit — not fillable in this demonstration",
                    "Verified carbon footprint declaration (Art. 7)":
                        "Requires a real third-party accredited audit — not fillable in this demonstration",
                }
                _all_fields: list[dict] = []
                for _grp_key in ("identity", "soh", "lifecycle", "carbon"):
                    _all_fields.extend(p.get(_grp_key, []))
                _missing = [f for f in _all_fields if f.get("state") == "unavailable"]
                for _mf in _missing:
                    _fname  = _mf.get("label", "Unknown field")
                    _tip    = _E8_HOW_TO.get(_fname, "Provide this data on the Import page")
                    st.markdown(
                        f"<div style='font-size:11px;padding:4px 0;border-bottom:1px solid #2d3748;'>"
                        f"<span style='color:#fc8181'>●</span> <strong style='color:#e2e8f0'>{_fname}</strong>"
                        f"<br><span style='color:#a0aec0;font-size:10px'>How to fill: {_tip}</span></div>",
                        unsafe_allow_html=True,
                    )
    st.markdown("")

    _md_html(
        f"""
        <div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);
                    border-radius:10px;padding:14px 20px;margin-bottom:28px;
                    font-size:13px;color:#8896a8;line-height:1.7">
            <strong style="color:#63b3ed">Battery Passport Interface</strong> — demonstrating the
            EU Battery Regulation (EU) 2023/1542 data structure. This is <strong>not</strong> a
            compliance claim: every field below is marked {make_state_badge("available")},
            {make_state_badge("estimated")}, or {make_state_badge("unavailable")} based on what this
            demonstration actually has. Nothing is hidden or faked to look complete.
        </div>
        """
    )

    _show_unavail = st.checkbox(
        "Show unavailable fields", key="passport_show_unavail", value=False
    )

    groups = [
        ("identity",  "1 · Battery Identity"),
        ("soh",       "2 · State of Health"),
        ("lifecycle", "3 · Lifecycle History"),
        ("carbon",    "4 · Carbon Footprint"),
    ]
    for key, title in groups:
        _fields = p[key] if _show_unavail else [f for f in p[key] if f.get("state") != "unavailable"]
        if not _fields:
            continue
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
            f"letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
            f"margin-bottom:4px;margin-top:20px'>{title}</div>",
            unsafe_allow_html=True,
        )
        rows_html = "".join(_passport_field_row(f) for f in _fields)
        st.markdown(f"<div>{rows_html}</div>", unsafe_allow_html=True)

    # ── 5: Critical Raw Materials (EU Battery Regulation Art. 13) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>5 · Critical Raw Materials (EU Reg. 2023/1542 Art. 13)</div>",
        unsafe_allow_html=True,
    )
    _crm_settings = {k: st.session_state.get(k) for k in [
        "crm_lfp_li_pct", "crm_nca_co_pct", "crm_nca_ni_pct", "crm_nca_li_pct",
        "crm_nca_recycled_co_pct", "crm_nca_recycled_ni_pct",
        "crm_synth_co_pct", "crm_synth_ni_pct", "crm_synth_li_pct",
        "crm_user_co_pct", "crm_user_ni_pct", "crm_user_li_pct",
        "crm_user_recycled_co_pct", "crm_user_recycled_ni_pct",
    ]}
    _crm_settings_clean = {k: v for k, v in _crm_settings.items() if v is not None}
    from chemistry_profiles import ChemistryProfile as _ChemProfile  # noqa: E402
    _crm_fields = _ChemProfile.for_cell(selected).get_crm_fields(_crm_settings_clean)
    _crm_visible = _crm_fields if _show_unavail else [f for f in _crm_fields if f.get("state") != "unavailable"]
    _crm_html = "".join(_passport_field_row(f) for f in _crm_visible)
    st.markdown(f"<div>{_crm_html}</div>", unsafe_allow_html=True)

    # ── 6: End-of-Life R-code Taxonomy ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>6 · End-of-Life R-Code Taxonomy (IEC 62902 / EU Art. 70)</div>",
        unsafe_allow_html=True,
    )
    # R-code comes from consequences.eol_r_code_recommendation(), not an
    # independent SOH ladder -- see that function's docstring for why
    # (Circular Economy Coverage finding: this page used to hardcode its
    # own 80/60 thresholds, unrelated to what application_fit()/classify()
    # would actually recommend for the same cell).
    from consequences import eol_r_code_recommendation
    _soh_now = float(df.iloc[-1]["soh_pct"]) if "soh_pct" in df.columns else 85.0
    _fade_30_now = float(df.iloc[-1].get("fade_rate_30cy", 0.0)) if "fade_rate_30cy" in df.columns else 0.0
    _sop_now = float(df.iloc[-1]["sop_pct"]) if "sop_pct" in df.columns and pd.notna(df.iloc[-1]["sop_pct"]) else None
    _r_rec = eol_r_code_recommendation(_soh_now, _fade_30_now, sop_pct=_sop_now)
    _eol_r_code = _r_rec["r_code"]
    _eol_color  = _r_rec["color"]
    # R3/R4 row states derive from the same _eol_r_code just computed above,
    # not a second independent SOH check -- two parallel derivations of the
    # same taxonomy is exactly the bug class this section was rewritten to
    # close.
    _r_fields = [
        {"label": "Recommended R-code", "value": _eol_r_code, "state": "estimated",
         "note": f"Based on current SOH = {_soh_now:.1f}% and second-life application fit. IEC 62902 R0–R9 taxonomy."},
        {"label": "R0 — Reuse", "value": "SOH ≥ 90% required", "state": "estimated" if _soh_now >= 90 else "unavailable"},
        {"label": "R3 — Second-life", "value": "Requires a fitting second-life application (see Consequences page)",
         "state": "available" if _eol_r_code.startswith("R3") else "estimated"},
        {"label": "R4 — Recycle", "value": "Hydromet / direct recycling pathway",
         "state": "available" if _eol_r_code.startswith("R4") else "estimated"},
        {"label": "Recycled content declaration", "value": "IEC 63338 audit required", "state": "unavailable",
         "note": "Carbon footprint per kWh must be declared for market access (IEC 63338, from 2025)."},
    ]
    _r_visible = _r_fields if _show_unavail else [f for f in _r_fields if f.get("state") != "unavailable"]
    _r_html = "".join(_passport_field_row(f) for f in _r_visible)
    st.markdown(f"<div style='margin-bottom:4px'><span style='color:{_eol_color};font-weight:700;font-size:13px'>Recommended: {_eol_r_code}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div>{_r_html}</div>", unsafe_allow_html=True)

    # ── 6b: Recycler recommendation (only when R4/R5 recycling is the actual
    # recommendation -- routing a cell that's actually second-life/reuse-fit
    # to a recycler directory would contradict the R-code just shown above) ──
    if _eol_r_code.startswith("R4") or _eol_r_code.startswith("R5"):
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        from recycler_directory import recommend_recyclers
        from chemistry_profiles import ChemistryProfile as _ChemProfileRec
        _rec_region = st.selectbox(
            "Your region (for logistics preference only — not a real shipping calculation)",
            ["North America", "Europe", "Asia"], key="passport_recycler_region",
        )
        _rec_chem = _ChemProfileRec.for_cell(selected).short_name
        _rec_matches = recommend_recyclers(_rec_chem, user_region=_rec_region)
        st.markdown(
            "<div style='font-size:12px;color:#8896a8;margin-bottom:10px'>"
            "Point-in-time research snapshot (2026-08) — verify current operating status "
            "before actually routing a cell. Chemistry compatibility is based on each "
            "company's publicly stated process focus, not a certified intake specification."
            "</div>",
            unsafe_allow_html=True,
        )
        for _rm in _rec_matches:
            _region_badge = "📍 Same region" if _rm["same_region"] else _rm["region"]
            render_card(
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-size:14px;font-weight:700;color:#e2e8f0'>{_rm['name']}</div>"
                f"<div style='font-size:11px;color:{'#48bb78' if _rm['same_region'] else '#a0aec0'}'>{_region_badge}</div>"
                f"</div>"
                f"<div style='font-size:12px;color:#a0aec0;margin-top:6px;line-height:1.6'>{_rm['process']}</div>"
                f"<div style='font-size:11px;color:#8896a8;margin-top:4px'>{_rm['recovery_note']}</div>",
                padding="14px 18px",
                extra_style="margin-bottom:8px",
            )

    # ── 7: Compliance Status (prose, no badge) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:20px'>7 · Compliance Status</div>",
        unsafe_allow_html=True,
    )
    render_card(
        f"""
        <strong style="color:#e2e8f0">This is a data-structure demonstration, not a regulatory
        submission.</strong><br><br>
        Of {summ['n_total']} fields modelled on the EU Battery Regulation's data requirements:
        <strong style="color:#48bb78">{summ['n_available']} are available</strong> from this
        platform's validated pipeline, <strong style="color:#d69e2e">{summ['n_estimated']} are
        cited estimates</strong> from the Consequences module, and
        <strong style="color:#8896a8">{summ['n_unavailable']} are not available</strong> in
        this demonstration.<br><br>
        An actual regulatory submission under (EU) 2023/1542 would additionally require:
        manufacturer-submitted identity and supply-chain records, a third-party accredited
        carbon footprint audit, repair/refurbishment history tracking, and notified-body
        sign-off — none of which a portfolio project can provide. No field on this page should
        be read as a compliance claim.
        """,
        padding="20px 24px",
        extra_style="font-size:13px;color:#a0aec0;line-height:1.8",
    )


    # ── E3: QR Code generator ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:24px'>8 · Physical Deployment — QR Code</div>",
        unsafe_allow_html=True,
    )
    _qr_api_base = st.text_input(
        "API base URL (for QR code)",
        value="http://localhost:8000",
        key="passport_qr_base",
        help="Points to your FastAPI deployment. The QR encodes /cells/{cell_id}.",
    )
    if st.button("Generate QR Code", key="passport_qr_btn"):
        try:
            import qrcode as _qrlib
            import io as _io
            _qr_url = f"{_qr_api_base.rstrip('/')}/cells/{selected}"
            _qr = _qrlib.QRCode(error_correction=_qrlib.constants.ERROR_CORRECT_M, box_size=6, border=3)
            _qr.add_data(_qr_url)
            _qr.make(fit=True)
            _qr_img = _qr.make_image(fill_color="white", back_color="#0e1117")
            _qr_buf = _io.BytesIO()
            _qr_img.save(_qr_buf, format="PNG")
            _qr_buf.seek(0)
            st.image(_qr_buf, caption=f"Scan to open live passport for {selected}", width=200)
            st.download_button(
                "Download QR PNG", data=_qr_buf.getvalue(),
                file_name=f"passport_qr_{selected}.png", mime="image/png",
                key="passport_qr_dl",
            )
            st.caption(f"Encoded URL: {_qr_url} · Print and affix to battery for on-site scanning.")
        except ImportError:
            st.warning(
                "qrcode library not installed. Add `qrcode[pil]` to requirements.txt "
                "and restart the app."
            )

    # ── Machine-readable export (JSON-LD) ─────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:24px'>9 · Machine-Readable Export</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "JSON-LD export of this passport's field data, preserving each field's "
        "available/estimated/unavailable provenance tag."
    )
    from passport_export import to_json_ld
    import json as _json_dpp
    _dpp_jsonld = to_json_ld(p, selected)
    st.download_button(
        "Download machine-readable passport (JSON-LD)",
        data=_json_dpp.dumps(_dpp_jsonld, indent=2).encode(),
        file_name=f"{selected}_passport.jsonld",
        mime="application/ld+json",
        key="passport_jsonld_dl",
    )
