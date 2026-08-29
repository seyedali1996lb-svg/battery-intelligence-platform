"""Page: Compliance (EU Battery Passport + Reports + Sustainability + Regulatory Alerts)."""


import _paths  # noqa: F401
import streamlit as st
import pandas as pd

from utils import (
    _action_bar, _md_html, render_regenerate_report_button,
)
from chemistry_profiles import ChemistryProfile
from _pages._compliance_passport import page_passport
from _pages._compliance_sustainability import page_sustainability
from _pages._compliance_stakeholder import page_stakeholder_view


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


def page_reports(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport
    from consequences import ASSUMPTIONS, application_fit, financial_comparison, best_fit_application

    source  = ChemistryProfile.for_cell(selected).source_kind
    p       = build_passport(selected, df, bundle, rul_reliable)

    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])

    st.markdown("# Reports")
    st.markdown(f"#### Demonstration report export · {selected}")

    _md_html("""<div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);border-radius:10px;padding:14px 20px;margin-bottom:28px;font-size:13px;color:#8896a8;line-height:1.7"><strong style="color:#63b3ed">Demonstration report</strong> — not a regulatory document. Exports the current battery's identity, SOH/RUL with reliability flags, second-life recommendation (if applicable), and the assumption register, with the same Available / Estimate / Not-available-in-demo labelling used throughout this platform.</div>""")

    second_life = None
    if soh <= 85.0:
        fade_30 = float(latest.get("fade_rate_30cy", 0.0))
        fit     = application_fit(soh, fade_30, fleet_fade_median=None)
        best_key, best = best_fit_application(fit)

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
            <div style="font-size:22px;font-weight:700;color:#a0aec0;margin-bottom:16px">
                {label}
            </div>
            <div style="font-size:14px;color:#a0aec0;max-width:520px;margin:0 auto;line-height:1.7;white-space:pre-line">
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

    # Reads precomputed CellSummary rows instead of every cell's full
    # per-cycle DataFrame -- see src/cell_store.py's module docstring.
    import db as _db_reg
    _active_ids_reg = set(featured_dfs.keys())

    _today = _dt_reg.date.today()
    _rows_reg = []
    for _r in _db_reg.get_cell_summaries(st.session_state["auth_org_id"]):
        if _r["cell_id"] not in _active_ids_reg or _r.get("soh_pct") is None:
            continue
        _soh_now = float(_r["soh_pct"])
        _fade    = float(_r["fade_rate_50cy"] or 0)
        _eol_cy  = None
        if _fade > 1e-6:
            _eol_cy = int(max(0, (_soh_now - 80.0) / _fade * 50))
        _rows_reg.append({
            "cell_id": _r["cell_id"],
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
        st.markdown("<h4 class='section-header'>Cells Requiring Action</h4>", unsafe_allow_html=True)
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
        st.markdown("<h4 class='section-header'>Draft Compliance Notice</h4>", unsafe_allow_html=True)
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
    st.markdown("#### EU Battery Regulation 2023/1542 · Passport · Reports · Sustainability")
    _tab_passport, _tab_reports, _tab_sus, _tab_stake, _tab_reg = st.tabs(
        ["EU Battery Passport", "Reports & Export", "Sustainability", "Stakeholder View", "Regulatory Alerts"]
    )
    with _tab_passport:
        page_passport(selected, df, bundle, rul_reliable)
    with _tab_reports:
        page_reports(selected, df, bundle, rul_reliable)
    with _tab_sus:
        page_sustainability(selected, df)
    with _tab_stake:
        page_stakeholder_view(selected, df, bundle, rul_reliable)
    with _tab_reg:
        _page_regulatory_alerts(selected, df, active_fdfs, bundles)
