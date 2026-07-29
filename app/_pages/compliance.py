"""Page: Compliance (EU Battery Passport + Reports + Sustainability + Regulatory Alerts)."""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    _action_bar, _md_html, base_layout, LEGEND_H, render_card,
    render_regenerate_report_button,
)
from design_system import (
    make_badge, make_state_badge, section_header_html,
    BADGE_ESTIMATE, BADGE_ILLUST,
)
from chemistry_profiles import ChemistryProfile


@st.cache_data(show_spinner=False)
def _cached_build_report_pdf(cache_key: str, _passport: dict, _second_life: dict | None, _assumptions: dict):
    """Cached wrapper around report_pdf.build_report_pdf().

    Without this, every widget interaction anywhere on the Reports page
    (a tab switch, a checkbox elsewhere on the page) rebuilt the full
    ReportLab document from scratch — multiple Table/Paragraph objects,
    styles, full layout — even when the selected cell and its data hadn't
    changed at all.

    cache_key is a content hash of (passport, second_life, assumptions)
    computed by the caller (see page_reports()). The leading-underscore
    params are excluded from Streamlit's own argument hashing (its
    documented convention) since these are plain dicts that don't need
    Streamlit's generic hasher when we already have a purpose-built key.

    This also fixes a real correctness bug, not just a performance one:
    document_id() (src/passport_export.py) embeds datetime.now() in the
    hash it returns, so the "Document ID" shown to the user — and the
    "Generated {time}" text inside the PDF body — changed on every rerun
    even for byte-identical passport data, contradicting document_id()'s
    own documented purpose ("a deterministic hash pairing a PDF export and
    its JSON-LD companion from the same export action"). Caching freezes
    both to the first-generation time for a given content hash, which is
    the behavior document_id() was already supposed to have.
    """
    from report_pdf import build_report_pdf
    return build_report_pdf(_passport, _second_life, _assumptions)


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
    st.markdown(f"##### Battery Passport Interface · {selected}")

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
                _E8_HOW_TO: dict[str, str] = {
                    "manufacturer": "Set in sidebar → Cell Metadata → Manufacturer",
                    "chemistry":    "Detected automatically once enough cycle data is loaded",
                    "carbon_footprint_kg_co2": "Upload manufacturer LCA document in Import page",
                    "recycled_content_pct":    "Enter on Import page → Advanced → Recycled content",
                    "supply_chain_due_diligence": "Link supplier declaration PDF in Import page",
                    "hazardous_substances":    "Set on Import page → Advanced → Hazardous substances",
                    "end_of_life_instructions": "Auto-populated once chemistry is confirmed",
                    "second_life_status":       "Set on Decision page after repurpose decision is logged",
                }
                _all_fields: list[dict] = []
                for _grp_key in ("identity", "soh", "lifecycle", "carbon"):
                    _all_fields.extend(p.get(_grp_key, []))
                _missing = [f for f in _all_fields if f.get("state") == "unavailable"]
                for _mf in _missing:
                    _fname  = _mf.get("label", _mf.get("field", "Unknown field"))
                    _tip    = _E8_HOW_TO.get(str(_mf.get("field", "")), "Provide this data on the Import page")
                    st.markdown(
                        f"<div style='font-size:11px;padding:4px 0;border-bottom:1px solid #2d3748;'>"
                        f"<span style='color:#fc8181'>●</span> <strong style='color:#e2e8f0'>{_fname}</strong>"
                        f"<br><span style='color:#718096;font-size:10px'>How to fill: {_tip}</span></div>",
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
            f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
            f"margin-bottom:4px;margin-top:20px'>{title}</div>",
            unsafe_allow_html=True,
        )
        rows_html = "".join(_passport_field_row(f) for f in _fields)
        st.markdown(f"<div>{rows_html}</div>", unsafe_allow_html=True)

    # ── 5: Critical Raw Materials (EU Battery Regulation Art. 13) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
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
    _r_visible = _r_fields if _show_unavail else [f for f in _r_fields if f.get("state") != "unavailable"]
    _r_html = "".join(_passport_field_row(f) for f in _r_visible)
    st.markdown(f"<div style='margin-bottom:4px'><span style='color:{_eol_color};font-weight:700;font-size:13px'>Recommended: {_eol_r_code}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div>{_r_html}</div>", unsafe_allow_html=True)

    # ── 7: Compliance Status (prose, no badge) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
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
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
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
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
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


def page_reports(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport
    from consequences import ASSUMPTIONS, application_fit, financial_comparison

    source  = ChemistryProfile.for_cell(selected).source_kind
    p       = build_passport(selected, df, bundle, rul_reliable)

    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])

    st.markdown("# Reports")
    st.markdown(f"##### Demonstration report export · {selected}")

    _md_html("""<div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);border-radius:10px;padding:14px 20px;margin-bottom:28px;font-size:13px;color:#8896a8;line-height:1.7"><strong style="color:#63b3ed">Demonstration report</strong> — not a regulatory document. Exports the current battery's identity, SOH/RUL with reliability flags, second-life recommendation (if applicable), and the assumption register, with the same Available / Estimate / Not-available-in-demo labelling used throughout this platform.</div>""")

    second_life = None
    if soh <= 85.0:
        fade_30 = float(latest.get("fade_rate_30cy", 0.0))
        fit     = application_fit(soh, fade_30, fleet_fade_median=None)
        best_key, best = max(fit.items(), key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]])

        a   = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        fc  = financial_comparison(
            soh=soh, source=source,
            recycling_value=a["recycling_value"], new_cell_cost=a["new_cell_cost"],
            sl_value_per_kwh=a["second_life_value_per_kwh"], repack_cost=a["repack_cost"],
        )
        second_life = {
            "best_app": best["name"],
            "best_fit": best["fit"],
            "financials": {
                "Reuse (second-life)": fc["sl_net"],
                "Recycle now":         fc["recycle_value"],
                "Buy new cell":        -fc["new_cell_cost"],
            },
        }

    st.markdown("##### Preview")
    st.markdown(f"**Cell:** {selected} &nbsp;·&nbsp; **SOH:** {soh:.1f}%")
    if second_life:
        st.markdown(
            f"**Second-life fit:** {second_life['best_app']} ({second_life['best_fit']}) — "
            f"figures are cited estimates, see Consequences page for full sliders."
        )
    else:
        st.markdown("**Second-life fit:** still in primary life — no recommendation yet.")
    summ = p["summary"]
    st.markdown(
        f"**Field coverage:** {summ['n_available']} available · {summ['n_estimated']} estimated · "
        f"{summ['n_unavailable']} not available in demo"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    from passport_export import to_json_ld
    import hashlib
    import json as _json_report

    _pdf_cache_key = hashlib.sha256(
        _json_report.dumps(
            {"passport": p, "second_life": second_life, "assumptions": ASSUMPTIONS},
            sort_keys=True, default=str,
        ).encode()
    ).hexdigest()[:20]
    pdf_bytes, _doc_id = _cached_build_report_pdf(_pdf_cache_key, p, second_life, ASSUMPTIONS)
    _jsonld = to_json_ld(p, selected, doc_id=_doc_id)

    st.caption(f"Document ID: {_doc_id} — both files below share this ID, since they're the same export.")
    _rep_col1, _rep_col2 = st.columns(2)
    with _rep_col1:
        st.download_button(
            label="Download demonstration report (PDF)",
            data=pdf_bytes,
            file_name=f"battery_passport_{selected}_{_doc_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with _rep_col2:
        st.download_button(
            label="Download machine-readable passport (JSON-LD)",
            data=_json_report.dumps(_jsonld, indent=2).encode(),
            file_name=f"battery_passport_{selected}_{_doc_id}.jsonld",
            mime="application/ld+json",
            use_container_width=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_regenerate_report_button(bundle, st.session_state.get("auth_org_id"), key_suffix=f"passport_{selected}")


def page_sustainability(selected: str, df: pd.DataFrame):
    _action_bar("sustainability")
    from consequences import ASSUMPTIONS, sustainability_snapshot, CELL_NOMINAL_KWH
    from sustainability import (
        CRITICAL_MATERIALS, EU_RECYCLED_TARGETS, EU_GREEN_DEAL_FIELDS,
        material_content_for_cell,
    )

    _profile = ChemistryProfile.for_cell(selected)
    source   = _profile.source_kind
    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])
    cycles  = int(latest["cycle_number"])
    # .get() with the same generic fallback decision.py uses — direct indexing
    # would KeyError for an uploaded cell, which has no dedicated entry.
    cell_kwh = CELL_NOMINAL_KWH.get(source, 0.0057)

    # ── Badge helpers ──
    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    # ── Page header ──
    st.markdown("# Sustainability")
    st.markdown(f"##### Lifecycle carbon + circularity · {selected}")

    st.markdown(
        f"<div style='background:rgba(183,121,31,0.07);border:1px solid rgba(183,121,31,0.25);"
        f"border-radius:10px;padding:14px 20px;margin-bottom:28px;"
        f"font-size:13px;color:#8896a8;line-height:1.7'>"
        f"<strong style='color:#d69e2e'>Figure transparency.</strong> "
        f"All CO₂ and material figures are estimates from literature sources — "
        f"not measurements from this specific cell. Each figure is labeled "
        f"{BADGE_ESTIMATE} or {BADGE_ILLUST} at the point of display. "
        f"No aggregated sustainability score is shown: individual labeled figures "
        f"are more honest than any index that mixes sources with different confidence levels."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Sliders ──
    with st.expander("Adjust assumptions", expanded=False):
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            co2_val = st.slider(
                f"CO₂ to make one new cell ({ASSUMPTIONS['co2_manufacture']['unit']})",
                min_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][1]),
                value=float(ASSUMPTIONS["co2_manufacture"]["value"]), step=0.05,
                key="sus7_co2_mfg",
                help=ASSUMPTIONS["co2_manufacture"]["source"],
            )
        with sl_col2:
            grid_val = st.slider(
                f"Grid carbon intensity ({ASSUMPTIONS['grid_carbon_intensity']['unit']})",
                min_value=float(ASSUMPTIONS["grid_carbon_intensity"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["grid_carbon_intensity"]["slider_range"][1]),
                value=float(ASSUMPTIONS["grid_carbon_intensity"]["value"]), step=0.01,
                key="sus7_grid",
                help=ASSUMPTIONS["grid_carbon_intensity"]["source"],
            )
        sl_col3, sl_col4 = st.columns(2)
        with sl_col3:
            mat_val = st.slider(
                f"Material recovery value ({ASSUMPTIONS['material_recovery']['unit']})",
                min_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][1]),
                value=float(ASSUMPTIONS["material_recovery"]["value"]), step=0.05,
                key="sus7_mat",
                help=ASSUMPTIONS["material_recovery"]["source"],
            )
        with sl_col4:
            extension_val = st.slider(
                f"Second-life extension ({ASSUMPTIONS['second_life_extension']['unit']})",
                min_value=float(ASSUMPTIONS["second_life_extension"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["second_life_extension"]["slider_range"][1]),
                value=float(ASSUMPTIONS["second_life_extension"]["value"]), step=0.1,
                key="sus7_extension",
                help=ASSUMPTIONS["second_life_extension"]["source"],
            )

    sus = sustainability_snapshot(source=source, co2_per_cell=co2_val, material_recovery=mat_val)
    use_phase_co2 = cell_kwh * cycles * grid_val

    # ────────────────────────────────────────────────────────────────────────
    # Section 1: Hero CO₂ comparison
    # ────────────────────────────────────────────────────────────────────────
    _section("CO₂ Impact — Reuse vs Recycle vs New Cell")

    h1, h2, h3, h4 = st.columns(4)
    hero_tiles = [
        (h1, "Manufacturing CO₂\n(one new cell)",   f"{co2_val:.2f} kg", "#f6ad55", BADGE_ESTIMATE,
         "IVL 2019 · range 0.30–1.00 kg/cell · adjust slider"),
        (h2, "Use phase CO₂\n(to date, this cell)", f"{use_phase_co2:.2f} kg", "#718096", BADGE_ILLUST,
         f"Grid: {grid_val:.2f} kg CO₂/kWh (IEA 2023) · {cycles} cycles"),
        (h3, "Reuse saves\n(vs making a new cell)", f"{sus['co2_avoided_by_reuse']:.2f} kg", "#48bb78", BADGE_ESTIMATE,
         "Harper et al. 2019 · Dunn 2015 · vs manufacturing a new cell"),
        (h4, "Recycle credit\n(15% cathode, Dunn 2015)", f"{sus['co2_recycling_credit']:.2f} kg", "#f6e05e", BADGE_ESTIMATE,
         "Dunn et al. 2015 · 15% cathode material recovery credit"),
    ]
    for col, label, val, colour, badge, src_note in hero_tiles:
        label_lines = label.split("\n")
        label_html = "<br>".join(label_lines)
        with col:
            render_card(
                f"<div style='font-size:11px;color:#4a5568;line-height:1.5'>{label_html}</div>"
                f"<div style='font-size:26px;font-weight:700;color:{colour};margin-top:6px'>{val}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>CO₂e</div>"
                f"<div style='margin-top:8px'>{badge}</div>"
                f"<div style='font-size:10px;color:#2d3748;margin-top:4px;line-height:1.4'>{src_note}</div>",
                padding="16px 20px",
                extra_style="height:100%",
            )

    # ────────────────────────────────────────────────────────────────────────
    # Section 2: Lifecycle carbon chart
    # ────────────────────────────────────────────────────────────────────────
    _section("Lifecycle Carbon Chart")

    st.markdown(
        f"<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        f"Manufacturing and EOL figures are {BADGE_ESTIMATE} from IVL 2019 and Dunn et al. 2015. "
        f"Use-phase CO₂ is {BADGE_ILLUST} — it depends entirely on your grid's carbon intensity "
        f"(set via slider above). Drag the grid slider to see how dominant the use phase becomes "
        f"on a coal grid vs a renewable grid."
        f"</div>",
        unsafe_allow_html=True,
    )

    lc_norm = st.radio(
        "Chart normalisation",
        ["Per cell", "Per kWh delivered"],
        index=0,
        horizontal=True,
        key="sus7_lc_norm",
        help="Per kWh delivered divides all bars by total energy throughput for each scenario, making scenarios with different lifetimes comparable.",
    )

    reuse_cycles       = cycles * extension_val
    new_cell_cycles    = cycles   # same history length as counterfactual baseline
    reuse_use_co2      = cell_kwh * reuse_cycles * grid_val
    new_cell_use_co2   = cell_kwh * new_cell_cycles * grid_val

    scenarios = ["Recycle now", f"Reuse (×{extension_val:.1f} cycles)", "New cell (counterfactual)"]
    mfg_bars       = [co2_val, co2_val, co2_val]
    use_bars        = [use_phase_co2, reuse_use_co2, new_cell_use_co2]
    eol_credit_bars = [
        -sus["co2_recycling_credit"],
        -(sus["co2_recycling_credit"] + sus["co2_avoided_by_reuse"]),
        -sus["co2_recycling_credit"],
    ]

    if lc_norm == "Per kWh delivered":
        kwh_denominators = [
            cell_kwh * cycles,
            cell_kwh * reuse_cycles,
            cell_kwh * new_cell_cycles,
        ]
        mfg_bars       = [v / d for v, d in zip(mfg_bars, kwh_denominators)]
        use_bars       = [v / d for v, d in zip(use_bars, kwh_denominators)]
        eol_credit_bars = [v / d for v, d in zip(eol_credit_bars, kwh_denominators)]
        yaxis_label    = "kg CO₂e per kWh delivered"
        bar_suffix     = " kg/kWh"
    else:
        yaxis_label = "kg CO₂e per cell"
        bar_suffix  = " kg"

    fig_lc = go.Figure()
    fig_lc.add_trace(go.Bar(
        name="Manufacturing CO₂",
        x=scenarios, y=mfg_bars,
        marker_color="#f6ad55",
        text=[f"{v:.3f}{bar_suffix}" for v in mfg_bars],
        textposition="inside", textfont=dict(size=10, color="#1a202c"),
    ))
    fig_lc.add_trace(go.Bar(
        name="Use phase CO₂ (illustrative)",
        x=scenarios, y=use_bars,
        marker_color="#718096",
        text=[f"{v:.3f}{bar_suffix}" for v in use_bars],
        textposition="inside", textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig_lc.add_trace(go.Bar(
        name="EOL credit (negative = saving)",
        x=scenarios, y=eol_credit_bars,
        marker_color="#48bb78",
        text=[f"{v:.3f}{bar_suffix}" for v in eol_credit_bars],
        textposition="inside", textfont=dict(size=10, color="#1a202c"),
    ))
    fig_lc.update_layout(
        **base_layout(
            barmode="relative",
            xaxis=dict(zeroline=False),
            yaxis=dict(
                zeroline=True,
                zerolinecolor="#4a5568",
                title=dict(text=yaxis_label, font=dict(size=11)),
            ),
            height=360,
        )
    )
    fig_lc.update_layout(
        legend=LEGEND_H,
        title=dict(
            text="Lifecycle CO₂ — three end-of-life scenarios",
            font=dict(size=13, color="#a0aec0"),
            x=0,
        ),
    )
    st.plotly_chart(fig_lc, use_container_width=True)
    st.markdown(
        f"<div style='font-size:11px;color:#4a5568;margin-top:-8px;margin-bottom:4px'>"
        f"'Reuse' use-phase uses {extension_val:.1f}× current cycles ({cycles} → {reuse_cycles:.0f} cycles). "
        f"Second-life extension slider is {BADGE_ILLUST}. "
        f"'New cell' use-phase uses the same {cycles}-cycle baseline at current grid intensity. "
        f"All use-phase figures are {BADGE_ILLUST}."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Section 2b: Per-cell carbon context (qualitative — no percentages)
    # ────────────────────────────────────────────────────────────────────────
    _section("Degradation Rate & Carbon — What the Trend Means")

    fade_30 = float(latest.get("fade_rate_30cy", float("nan")))
    if not (fade_30 != fade_30):  # not NaN
        if fade_30 * 1000 > 5.0:
            fade_signal = "accelerating"
            fade_implication = (
                "A fast fade rate shortens the useful life phase. "
                "Because manufacturing CO₂ is fixed at cell production and amortised across the "
                "full operating lifetime, a shorter useful life leaves more of that carbon "
                "unrecovered per unit of energy delivered. "
                "A cell degrading quickly reaches the recycling decision point sooner — "
                "meaning less time in which the reuse CO₂ saving accumulates."
            )
            fade_colour = "#fc8181"
        elif fade_30 * 1000 > 2.0:
            fade_signal = "moderate"
            fade_implication = (
                "A moderate fade rate means the manufacturing carbon is being amortised "
                "at a reasonable pace across the useful life. "
                "Extending service life through second-life deployment would increase that "
                "amortisation further, reducing the manufacturing carbon burden per kWh delivered."
            )
            fade_colour = "#f6e05e"
        else:
            fade_signal = "slow"
            fade_implication = (
                "A slow, stable fade rate maximises the amortisation of manufacturing CO₂ "
                "across a long useful life. "
                "This cell is recovering its embodied carbon effectively: a longer service "
                "life means the fixed manufacturing cost is spread across more kWh delivered."
            )
            fade_colour = "#48bb78"

        render_card(
            f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-bottom:8px'>Fade rate impact on carbon amortisation</div>"
            f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:12px'>"
            f"<div style='font-size:22px;font-weight:700;color:{fade_colour}'>"
            f"{fade_30*1000:.2f} mAh/cy</div>"
            f"<div style='font-size:13px;color:{fade_colour}99'>{fade_signal} degradation</div>"
            f"</div>"
            f"<div style='font-size:13px;color:#a0aec0;line-height:1.8'>"
            f"{fade_implication}"
            f"</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:12px'>"
            f"This is qualitative direction only — the carbon figures above are based on "
            f"literature estimates, not measurements from this cell. No percentage saving is "
            f"stated here because the absolute manufacturing CO₂ figure (above) already carries "
            f"wide uncertainty. "
            f"Tier thresholds (slow &lt;2 mAh/cy · moderate 2–5 mAh/cy · accelerating &gt;5 mAh/cy) "
            f"are illustrative — adjust for your cell chemistry."
            f"</div>",
            padding="18px 22px",
            extra_style="margin-bottom:4px",
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 3: Critical materials tracker
    # ────────────────────────────────────────────────────────────────────────
    _section("Critical Materials Tracker")

    if _profile.provenance != "measured":
        st.markdown(
            "<div style='background:#2d3748;border-radius:8px;padding:10px 16px;"
            "font-size:12px;color:#8896a8;margin-bottom:12px'>"
            "Synthetic cells model electrochemical behaviour only — material content "
            "figures below apply to the equivalent real LiCoO₂ 18650 chemistry, not the simulation."
            "</div>",
            unsafe_allow_html=True,
        )

    primary_materials = [m for m in CRITICAL_MATERIALS if m["name"] != "Nickel (Ni)"]
    mat_cols = st.columns(len(primary_materials))
    for col, mat in zip(mat_cols, primary_materials):
        scaled_g = material_content_for_cell(mat["g_per_2ah"], cell_kwh)
        badge_html = make_badge(mat["label"], "#b7791f" if "Cited" in mat["label"] else "#718096")
        rec_html = (
            f"<div style='font-size:12px;color:#48bb78;margin-top:4px'>"
            f"~{mat['recovery_pct']}% recovery<br>"
            f"<span style='font-size:11px;color:#4a5568'>{mat['recovery_note']}</span></div>"
            if mat["recovery_pct"] is not None else
            "<div style='font-size:12px;color:#4a5568;margin-top:4px'>Not recovered<br>"
            "<span style='font-size:11px'>Not primary material in LiCoO₂</span></div>"
        )
        eu_dot = (
            "<span style='color:#63b3ed;font-size:10px;margin-left:4px'>"
            "EU critical ●</span>"
            if mat["eu_critical"] else ""
        )
        with col:
            render_card(
                f"<div style='font-size:12px;color:#a0aec0;font-weight:600'>"
                f"{mat['name']}{eu_dot}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{mat['formula']}</div>"
                f"<div style='font-size:22px;font-weight:700;color:#e2e8f0;margin-top:8px'>"
                f"{scaled_g:.1f} g</div>"
                f"<div style='font-size:11px;color:#8896a8'>est. per cell ({mat['g_range']} @ 2 Ah)</div>"
                f"{rec_html}"
                f"<div style='margin-top:10px'>{badge_html}</div>",
                padding="14px 16px",
            )

    ni_mat = next((m for m in CRITICAL_MATERIALS if m["name"] == "Nickel (Ni)"), None)
    if ni_mat:
        ni_g = material_content_for_cell(ni_mat["g_per_2ah"], cell_kwh)
        st.markdown(
            f"<div style='font-size:11px;color:#4a5568;margin-top:10px;padding:8px 14px;"
            f"background:#1a202c;border-radius:6px;border-left:3px solid #2d3748'>"
            f"<strong style='color:#8896a8'>Nickel (Ni)</strong> — EU critical material, but trace-only in LiCoO₂ chemistry "
            f"(est. {ni_g:.2f} g per cell, {BADGE_ILLUST}). "
            f"EU 2023/1542 nickel recycled-content targets apply to NMC/NCA chemistries where nickel is a primary cathode material, "
            f"not to LiCoO₂. Shown here for completeness only."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 4: EU regulation recycled content targets
    # ────────────────────────────────────────────────────────────────────────
    _section("EU Battery Regulation — Recycled Content Targets (2023/1542, Annex XII)")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Targets apply to industrial batteries and EV batteries by mass of active materials. "
        "Estimated recycled content in <em>current</em> 18650 LiCoO₂ cells is not publicly "
        "certified — the figures below reflect industry-wide estimates, not this specific cell. "
        "This platform cannot make a compliance claim without manufacturer supply chain data."
        "</div>",
        unsafe_allow_html=True,
    )

    for target in EU_RECYCLED_TARGETS:
        est_recycled = target.get("current_industry_range", "—")
        current_note = target.get("current_note", "")
        bar_fill_31  = min(target["target_2031_pct"], 100)
        bar_fill_36  = min(target["target_2036_pct"], 100)
        render_card(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:8px'>"
            f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{target['material']}</div>"
            f"<div style='display:flex;gap:16px;font-size:12px;color:#a0aec0'>"
            f"<span>2031 target: <strong style='color:#63b3ed'>{target['target_2031_pct']}%</strong></span>"
            f"<span>2036 target: <strong style='color:#63b3ed'>{target['target_2036_pct']}%</strong></span>"
            f"<span>Est. current: <strong style='color:#f6ad55'>{est_recycled}</strong> "
            f"{make_badge('Illustrative', '#718096')}</span>"
            f"</div></div>"
            f"<div style='font-size:11px;color:#4a5568;margin-bottom:8px'>{current_note}</div>"
            f"<div style='background:#2d3748;border-radius:4px;height:8px;margin-bottom:4px'>"
            f"<div style='background:#63b3ed33;border-radius:4px;height:8px;width:{bar_fill_31}%;"
            f"position:relative'>"
            f"<div style='background:#63b3ed;border-radius:4px;height:8px;width:{bar_fill_36/bar_fill_31*100:.0f}%'>"
            f"</div></div></div>"
            f"<div style='font-size:10px;color:#4a5568'>Source: {target['source']}</div>",
            padding="14px 20px",
            extra_style="margin-bottom:10px",
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 5: EU Green Deal alignment (three-state)
    # ────────────────────────────────────────────────────────────────────────
    _section("EU Green Deal Alignment — Data Coverage")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px'>"
        "Same three-state system as the Battery Passport page — "
        "Available (pipeline output), Estimated (cited/illustrative), "
        "Not available in demo (genuine gap)."
        "</div>",
        unsafe_allow_html=True,
    )

    for field in EU_GREEN_DEAL_FIELDS:
        state = field["state"]
        badge = make_state_badge(state)
        muted = state == "unavailable"
        val_colour = "#4a5568" if muted else "#a0aec0"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
            f"gap:16px;padding:11px 0;border-bottom:1px solid #2d3748'>"
            f"<div style='flex:1'>"
            f"<div style='font-size:13px;color:{val_colour};font-style:{'italic' if muted else 'normal'}'>"
            f"{field['label']}</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:3px'>{field['note']}</div>"
            f"</div>"
            f"<div style='flex-shrink:0;padding-top:2px'>{badge}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Assumption register ──
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    with st.expander("Assumption sources — CO₂ and material figures", expanded=False):
        sus_keys = ["co2_manufacture", "grid_carbon_intensity", "material_recovery"]
        for key in sus_keys:
            a = ASSUMPTIONS[key]
            badge_colour = "#b7791f" if "Cited" in a["label"] else "#718096"
            st.markdown(
                f"<div style='padding:12px 0;border-bottom:1px solid #2d3748'>"
                f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:6px'>"
                f"{a['unit']} — default {a['value']} &nbsp; {make_badge(a['label'], badge_colour)}"
                f"</div>"
                f"<div style='font-size:12px;color:#8896a8;line-height:1.6'>{a['source']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )



COMING_SOON_META = {
    "recommendations": ("Recommendations", "Actionable maintenance recommendations driven by health trends and failure-mode modelling.", "Phase 2"),
    "economics":       ("Economics",       "Total cost of ownership analysis, replacement cost modelling, and second-life ROI.", "Phase 2"),
    "sustainability":  ("Sustainability",  "Carbon impact tracking, second-life suitability scoring, and recycling timeline.", "Phase 2"),
    "reports":         ("Reports",         "Exportable PDF/CSV health reports and audit trails for stakeholders.", "Phase 2"),
    "settings":        ("Settings",        "Data source configuration, alert thresholds, and user preferences.", "Phase 2"),
}

def page_coming_soon(key: str):
    label, description, phase = COMING_SOON_META[key]
    st.markdown(f"# {label}")
    _md_html(
        f"""
        <div style="border:1px dashed #2d3748;border-radius:12px;padding:64px 48px;
                    text-align:center;margin-top:32px">
            <div style="font-size:13px;font-weight:600;color:#2d3748;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:16px">{phase}</div>
            <div style="font-size:22px;font-weight:700;color:#4a5568;margin-bottom:16px">
                {label}
            </div>
            <div style="font-size:14px;color:#4a5568;max-width:520px;margin:0 auto;line-height:1.7;white-space:pre-line">
                {description}
            </div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Regulatory Alert Service
# ---------------------------------------------------------------------------

def _page_regulatory_alerts(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
):
    """EU Battery Regulation Article 14(4) compliance deadline tracker.

    Shows which cells will be non-compliant by key regulation dates,
    and generates a draft submission text.
    """
    import datetime as _dt_reg

    st.markdown("### Regulatory Alert Service")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:18px;line-height:1.6'>"
        "EU Battery Regulation 2023/1542 requires a battery state report under "
        "<strong style='color:#e2e8f0'>Article 14(4)</strong> when SOH drops below defined "
        "thresholds. This tracker identifies cells at risk of non-compliance before key deadlines."
        "</div>"
    )

    # Key regulatory deadlines
    _deadlines = [
        {
            "label":       "Art. 14(4) — SOH report obligation",
            "date":        _dt_reg.date(2026, 2, 18),
            "soh_floor":   80.0,
            "description": "Batteries placed on EU market must have SOH ≥ 80% or carry a non-compliance notice.",
        },
        {
            "label":       "Art. 70 — End-of-life transparency",
            "date":        _dt_reg.date(2027, 8, 18),
            "soh_floor":   None,
            "description": "Battery Passport must include full end-of-life R-code and recycled content declaration.",
        },
        {
            "label":       "Annex XII — Recycled content targets (Phase 1)",
            "date":        _dt_reg.date(2031, 1, 1),
            "soh_floor":   None,
            "description": "12% Co, 4% Li, 4% Ni recycled content required in active materials.",
        },
    ]

    _today = _dt_reg.date.today()
    _rows_reg = []
    for _cid, _fdf in featured_dfs.items():
        if _fdf.empty or "soh_pct" not in _fdf.columns:
            continue
        _soh_now = float(_fdf.iloc[-1]["soh_pct"])
        _fade    = float(_fdf.iloc[-1].get("fade_rate_50cy", 0))
        _eol_cy  = None
        if _fade > 1e-6:
            _eol_cy = int(max(0, (_soh_now - 80.0) / _fade * 50))
        _rows_reg.append({
            "cell_id": _cid,
            "soh_now": _soh_now,
            "fade":    _fade,
            "cycles_to_eol": _eol_cy,
        })

    # Alert summary
    _art14_floor = 80.0
    _art14_date  = _deadlines[0]["date"]
    _days_to_deadline = (_art14_date - _today).days

    _at_risk = [r for r in _rows_reg if r["soh_now"] < _art14_floor]
    _approaching = [
        r for r in _rows_reg
        if r["soh_now"] >= _art14_floor and r["cycles_to_eol"] is not None
        and r["cycles_to_eol"] < max(0, _days_to_deadline) * 1.5
    ]

    _al1, _al2, _al3 = st.columns(3)
    _al_col = "#fc8181" if _at_risk else ("#f6ad55" if _approaching else "#48bb78")
    _al1.metric("Cells non-compliant now", len(_at_risk), help="SOH already below 80%")
    _al2.metric("Cells approaching threshold", len(_approaching),
                help=f"Will breach 80% before {_art14_date.strftime('%B %Y')}")
    _al3.metric("Days to Art. 14(4) deadline", _days_to_deadline)

    # Per-deadline table
    for _dl in _deadlines:
        _dl_days = (_dl["date"] - _today).days
        _dl_col  = "#fc8181" if _dl_days < 180 else ("#f6ad55" if _dl_days < 365 else "#48bb78")
        _md_html(
            f"<div style='background:#1e2a38;border:1px solid {_dl_col}44;"
            f"border-radius:10px;padding:14px 20px;margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{_dl['label']}</div>"
            f"<div style='font-size:12px;font-weight:700;color:{_dl_col}'>"
            f"{_dl['date'].strftime('%d %b %Y')} · {_dl_days} days</div>"
            f"</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:6px'>{_dl['description']}</div>"
            f"</div>"
        )

    # Affected cells
    if _at_risk or _approaching:
        st.markdown("<div class='section-header'>Cells Requiring Action</div>", unsafe_allow_html=True)
        for _r in sorted(_at_risk + _approaching, key=lambda x: x["soh_now"]):
            _is_breach = _r["soh_now"] < _art14_floor
            _status_c  = "#fc8181" if _is_breach else "#f6ad55"
            _status_l  = "Non-compliant" if _is_breach else "Approaching threshold"
            _md_html(
                f"<div style='display:flex;justify-content:space-between;padding:10px 16px;"
                f"border-bottom:1px solid #1a202c;align-items:center'>"
                f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{_r['cell_id']}</div>"
                f"<div style='font-size:12px;color:#a0aec0'>SOH {_r['soh_now']:.1f}%</div>"
                f"<div style='font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;"
                f"background:{_status_c}22;color:{_status_c};border:1px solid {_status_c}44'>"
                f"{_status_l}</div>"
                f"</div>"
            )

        # Draft submission
        st.markdown("<div class='section-header'>Draft Compliance Notice</div>", unsafe_allow_html=True)
        _draft_cells = "\n".join(
            f"  - Cell {r['cell_id']}: SOH {r['soh_now']:.1f}% "
            f"({'below' if r['soh_now'] < _art14_floor else 'approaching'} 80% threshold)"
            for r in sorted(_at_risk + _approaching, key=lambda x: x["soh_now"])
        )
        _draft_text = (
            f"BATTERY STATE COMPLIANCE NOTICE\n"
            f"EU Battery Regulation (EU) 2023/1542 — Article 14(4)\n"
            f"Generated: {_today.strftime('%d %B %Y')}\n"
            f"Compliance deadline: {_art14_date.strftime('%d %B %Y')}\n\n"
            f"The following battery cells have been identified as non-compliant or approaching\n"
            f"the mandatory State of Health threshold under Art. 14(4):\n\n"
            f"{_draft_cells}\n\n"
            f"Recommended actions:\n"
            f"  1. Commission independent SOH verification for cells marked non-compliant.\n"
            f"  2. Update Battery Passport records with current SOH and R-code classification.\n"
            f"  3. Schedule replacement or repurposing before the compliance deadline.\n\n"
            f"This notice was generated automatically by the Battery Intelligence Platform\n"
            f"using GBRT-derived SOH estimates. Values should be confirmed by certified testing\n"
            f"before submission to a regulatory authority.\n\n"
            f"Platform: Battery Intelligence System\n"
            f"Data pipeline: GBRT with leave-cell-out cross-validation\n"
            f"Regulation reference: (EU) 2023/1542, OJ L 2023/1542\n"
        )
        st.text_area("Draft notice (edit before submitting)", value=_draft_text, height=280,
                     key="reg_draft_text")
        st.download_button(
            "Download notice as .txt",
            data=_draft_text,
            file_name=f"compliance_notice_{_today.isoformat()}.txt",
            mime="text/plain",
            key="reg_draft_download",
        )
    else:
        st.success(
            f"All {len(_rows_reg)} cells are compliant with the 80% SOH threshold. "
            f"Next scheduled review: {_art14_date.strftime('%B %Y')}."
        )


# ---------------------------------------------------------------------------
# Page: Compliance (dispatcher — tabs for Passport/Reports/Sustainability/Regulatory Alerts)
# ---------------------------------------------------------------------------

def page_compliance(
    selected: str,
    df: pd.DataFrame,
    bundle: dict,
    rul_reliable: bool,
    active_fdfs: dict,
    bundles: dict,
) -> None:
    _action_bar("compliance")
    st.markdown(f"# Is {selected} EU passport-ready?")
    st.markdown("##### EU Battery Regulation 2023/1542 · Passport · Reports · Sustainability")
    _tab_passport, _tab_reports, _tab_sus, _tab_reg = st.tabs(
        ["EU Battery Passport", "Reports & Export", "Sustainability", "Regulatory Alerts"]
    )
    with _tab_passport:
        page_passport(selected, df, bundle, rul_reliable)
    with _tab_reports:
        page_reports(selected, df, bundle, rul_reliable)
    with _tab_sus:
        page_sustainability(selected, df)
    with _tab_reg:
        _page_regulatory_alerts(selected, df, active_fdfs, bundles)
