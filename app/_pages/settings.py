"""Page: Settings.

The configuration tier (everything below Model Transparency -- Application
Profile, Alert Thresholds, RUL Reliability preview, Model Cache, CRM
Config, Cost-of-Delay, Webhook Notifications, BMS Connectors, Second-Life
Marketplace, Maintenance Write-Back, Team Members, Sites & Fleets, AI
Copilot key, Onboarding, Production Roadmap, About) lives in
_settings_config.py -- this file was 1,031 lines, the 5th largest in
app/_pages/. page_settings() keeps the sections most tied to loaded data
(My Data, Data Sources, Model Transparency) and delegates the rest.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from design_system import section_header_html
from utils import _action_bar, render_card
from chemistry_profiles import ChemistryProfile
from _pages._settings_config import render_settings_configuration


def page_settings(featured_dfs: dict, bundles: dict):
    _action_bar("settings")
    from batlab.validation.lco import RUL_RELIABLE_FLOOR
    from design_system import C_GREEN, C_ORANGE

    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    st.markdown("# Settings")
    st.markdown("#### Platform configuration · model transparency · reliability controls")

    # ────────────────────────────────────────────────────────────────────────
    # Section 0: Uploaded data (shown only when uploaded data is in session)
    # ────────────────────────────────────────────────────────────────────────
    # Gated on the lightweight metadata dict, not the full uploaded
    # DataFrames/bundle — those are never stored in session_state at all
    # (see utils.load_tenant_bundle_cached()), so "is there an upload"
    # only needs this small dict, not the heavy data itself.
    up_meta     = st.session_state.get("uploaded_mode_meta")
    cell_ids_up = up_meta.get("cell_ids", []) if up_meta else []
    if up_meta:
        _section("My Data")
        n_up         = up_meta.get("n_cells", 0)
        lco_lim      = up_meta.get("lco_limited", False)
        temp_assumed = up_meta.get("temperature_assumed_cells", [])
        calib_cnt    = up_meta.get("calibrating_count", 0)

        render_card(
            f"<div style='font-size:12px;font-weight:600;color:#63b3ed;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>My Data · this session only</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{n_up} cells</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.8'>"
            f"{'⚠ LCO limited — fewer than 3 cells<br>' if lco_lim else ''}"
            f"{calib_cnt} Calibrating · {n_up - calib_cnt} reliable<br>"
            f"{'Temperature assumed 25°C for: ' + ', '.join(temp_assumed) if temp_assumed else 'Temperature measured for all cells'}"
            f"</div>"
            f"<div style='font-size:11px;color:#a0aec0;margin-top:8px'>"
            f"{', '.join(cell_ids_up)}</div>",
            padding="18px 20px",
            extra_style="margin-bottom:12px",
        )

        if st.button("✕ Clear uploaded data", key="settings_clear", use_container_width=False):
            _clear_uploaded_data()
            st.rerun()

        st.markdown(
            "<div style='font-size:11px;color:#a0aec0;margin-top:8px'>"
            "Uploaded data is persisted per organization and survives a refresh or a new "
            "login — it is never visible to other organizations. "
            "Clearing uploaded data switches you back to NASA Research Mode for this session; "
            "re-selecting My Data mode later restores it, since it's still saved on disk."
            "</div>",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 1: Data sources
    # ────────────────────────────────────────────────────────────────────────
    _section("Data Sources")

    # featured_dfs always merges synth+NASA+Severson unconditionally (see
    # app/main.py's load_everything()) — this used to bucket every real
    # Severson cell into "Synthetic cells" below ("not in NASA_CELL_IDS" was
    # the only test), captioned "Not real measured data." Classified via
    # ChemistryProfile.for_cell() now so each real dataset gets its own count.
    synth_ids    = [c for c in featured_dfs if ChemistryProfile.for_cell(c).source_kind == "synth" and c not in cell_ids_up]
    nasa_ids     = [c for c in featured_dfs if ChemistryProfile.for_cell(c).source_kind == "nasa" and c not in cell_ids_up]
    severson_ids = [c for c in featured_dfs if ChemistryProfile.for_cell(c).source_kind == "severson" and c not in cell_ids_up]

    src_col1, src_col2, src_col3 = st.columns(3)
    with src_col1:
        render_card(
            f"<div style='font-size:12px;font-weight:600;color:#fc8181;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>Synthetic cells</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{len(synth_ids)}</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.6'>"
            f"Physics-informed simulation (Arrhenius SEI growth, empirical C-rate factor, "
            f"Rainflow DoD scaling). Resistance: 0.15–0.40 Ω internal. "
            f"<strong>Not real measured data.</strong></div>"
            f"<div style='font-size:11px;color:#a0aec0;margin-top:8px'>"
            f"{', '.join(synth_ids)}</div>",
            padding="18px 20px",
        )
    with src_col2:
        render_card(
            f"<div style='font-size:12px;font-weight:600;color:#48bb78;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>NASA PCoE real cells</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{len(nasa_ids)}</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.6'>"
            f"LiCoO₂ 18650 cells, ~2 Ah, 24°C, 2A constant discharge. "
            f"Re (electrolyte resistance) from EIS: 0.04–0.07 Ω. "
            f"Source: Saha &amp; Goebel (2007), NASA PCoE dataset.</div>"
            f"<div style='font-size:11px;color:#a0aec0;margin-top:8px'>"
            f"{', '.join(nasa_ids) if nasa_ids else 'Not loaded — run python -m batlab.datasets.nasa'}</div>",
            padding="18px 20px",
        )
    with src_col3:
        render_card(
            f"<div style='font-size:12px;font-weight:600;color:#63b3ed;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>Severson real cells</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{len(severson_ids)}</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.6'>"
            f"LFP (A123 APR18650M1A) cells, 1.1 Ah, 30°C chamber, fast-charging protocols. "
            f"Source: Severson et al. (2019), Nature Energy.</div>"
            f"<div style='font-size:11px;color:#a0aec0;margin-top:8px'>"
            f"{', '.join(severson_ids) if severson_ids else 'Not loaded — run python -m batlab.datasets.severson'}</div>",
            padding="18px 20px",
        )

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-top:12px;padding:10px 14px;"
        "background:#1a202c;border-radius:6px;border-left:3px solid #2d3748'>"
        "<strong style='color:#8896a8'>Why two separate models?</strong> "
        "Synthetic and NASA cells use incompatible resistance scales (0.15–0.40 Ω vs 0.04–0.07 Ω Re). "
        "A combined model produced R²=−0.49. Two separate GBRT models, each trained and validated "
        "on its own data source, keep the predictions honest. Fleet ranking uses SOH "
        "(scale-invariant) rather than RUL (model-dependent) for cross-type comparison.</div>",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Section 2: Model transparency
    # ────────────────────────────────────────────────────────────────────────
    _section("Model Transparency — Leave-Cell-Out Validation")

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-bottom:14px;line-height:1.6'>"
        "Leave-cell-out (LCO) cross-validation trains on N−1 cells and tests on the held-out "
        "cell entirely. This is the honest generalisation metric — a row-level split on "
        "a multi-cell dataset leaks cell identity into training. Per-cell fold R² below "
        "the reliability floor gates RUL display for that cell across all pages.</div>",
        unsafe_allow_html=True,
    )

    # Keyed by the same source_kind vocabulary as ChemistryProfile.for_cell()
    # — was a binary "nasa"/else-"Synthetic" check that mislabeled the
    # Severson and uploaded-data models as "Synthetic" (wrong colour too).
    _SOURCE_KEY_META = {
        "nasa":     ("NASA PCoE", "#48bb78"),
        "severson": ("Severson LFP", "#63b3ed"),
        "synth":    ("Synthetic", "#fc8181"),
        "upload":   ("Uploaded", "#f6ad55"),
    }
    for source_key, bundle in bundles.items():
        if bundle is None:
            continue
        m = bundle["metrics"]
        lco_per = m.get("lco_per_cell", {})
        per_cell_ok = m.get("per_cell_rul_reliable", {})
        label, colour = _SOURCE_KEY_META.get(source_key, (source_key.title(), "#8896a8"))

        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{colour};"
            f"margin:16px 0 8px'>{label} model</div>",
            unsafe_allow_html=True,
        )

        header_cols = st.columns([2, 1, 1, 1, 2])
        for col, hdr in zip(header_cols, ["Cell", "SOH fold R²", "RUL fold R²", "RUL status", "Note"]):
            col.markdown(
                f"<div style='font-size:10px;font-weight:600;color:#a0aec0;"
                f"text-transform:uppercase;letter-spacing:0.06em'>{hdr}</div>",
                unsafe_allow_html=True,
            )

        for cell_id, fold in lco_per.items():
            soh_r2  = fold.get("soh_r2", None)
            rul_r2  = fold.get("rul_r2", None)
            ok      = per_cell_ok.get(cell_id, True)
            status_c = C_GREEN if ok else C_ORANGE
            status_l = "Calibrated" if ok else "Not calibrated"
            note = "" if ok else f"fold R²={rul_r2:.2f} < {RUL_RELIABLE_FLOOR} floor — RUL withheld"
            row_cols = st.columns([2, 1, 1, 1, 2])
            row_cols[0].markdown(f"<div style='font-size:13px;color:#e2e8f0;padding:4px 0'>{cell_id}</div>", unsafe_allow_html=True)
            row_cols[1].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:4px 0'>{soh_r2:.2f}</div>", unsafe_allow_html=True)
            row_cols[2].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:4px 0'>{rul_r2:.2f}</div>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<div style='font-size:13px;color:{status_c};padding:4px 0'>{status_l}</div>", unsafe_allow_html=True)
            row_cols[4].markdown(f"<div style='font-size:11px;color:#a0aec0;padding:4px 0'>{note}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='display:flex;gap:24px;font-size:12px;color:#8896a8;"
            f"padding:8px 0;border-top:1px solid #2d3748;margin-top:4px'>"
            f"<span>Dataset SOH R²: <strong style='color:#e2e8f0'>{m.get('lco_soh_r2', 0):.3f}</strong></span>"
            f"<span>Dataset RUL R²: <strong style='color:#e2e8f0'>{m.get('lco_rul_r2', 0):.3f}</strong></span>"
            f"<span>Training cells: <strong style='color:#e2e8f0'>{m.get('n_cells', '—')}</strong></span>"
            f"<span>Training rows: <strong style='color:#e2e8f0'>{m.get('n_rows', 0):,}</strong></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # -- Sections extracted to _settings_config.py (Application Profile, EOL,
    # Alert Thresholds, RUL Reliability Threshold, Model Cache, CRM Config,
    # Cost-of-Delay, Webhook Notifications, BMS Connectors, Second-Life
    # Marketplace, Maintenance Write-Back, Team Members, Sites & Fleets,
    # AI Copilot key, Onboarding, Production Roadmap, About) --
    render_settings_configuration(featured_dfs, bundles)


def _clear_uploaded_data():
    """Revert this session to NASA mode. Does NOT delete the org's persisted
    upload from disk (bundle_cache.save_tenant_bundle) — re-selecting My Data
    mode later restores it, same as before this only ever cleared session
    state. "uploaded_featured_dfs"/"uploaded_bundle"/"uploaded_split_cycles"
    are no longer session_state keys at all (see utils.load_tenant_bundle_cached()),
    so there's nothing to pop for those — only the lightweight metadata dict."""
    st.session_state.pop("uploaded_mode_meta", None)
    st.session_state["data_mode"] = "nasa"
