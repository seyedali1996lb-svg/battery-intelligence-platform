"""Page: Settings."""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from design_system import make_badge, section_header_html
from utils import _md_html, _action_bar, _empty_state, NASA_CELL_IDS
import db


def page_settings(featured_dfs: dict, bundles: dict):
    _action_bar("settings")
    from lco_eval import RUL_RELIABLE_FLOOR
    from design_system import C_GREEN, C_AMBER, C_MUTED, C_ORANGE

    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    st.markdown("# Settings")
    st.markdown("##### Platform configuration · model transparency · reliability controls")

    # ────────────────────────────────────────────────────────────────────────
    # Section 0: Uploaded data (shown only when uploaded data is in session)
    # ────────────────────────────────────────────────────────────────────────
    up_fdfs = st.session_state.get("uploaded_featured_dfs", {})
    if up_fdfs:
        _section("My Data")
        up_meta      = st.session_state.get("uploaded_mode_meta") or {}
        n_up         = up_meta.get("n_cells", len(up_fdfs))
        lco_lim      = up_meta.get("lco_limited", False)
        temp_assumed = up_meta.get("temperature_assumed_cells", [])
        calib_cnt    = up_meta.get("calibrating_count", 0)
        cell_ids_up  = up_meta.get("cell_ids", list(up_fdfs.keys()))

        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 20px;margin-bottom:12px'>"
            f"<div style='font-size:12px;font-weight:600;color:#63b3ed;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>My Data · this session only</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{n_up} cells</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.8'>"
            f"{'⚠ LCO limited — fewer than 3 cells<br>' if lco_lim else ''}"
            f"{calib_cnt} Calibrating · {n_up - calib_cnt} reliable<br>"
            f"{'Temperature assumed 25°C for: ' + ', '.join(temp_assumed) if temp_assumed else 'Temperature measured for all cells'}"
            f"</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:8px'>"
            f"{', '.join(cell_ids_up)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("✕ Clear uploaded data", key="settings_clear", use_container_width=False):
            _clear_uploaded_data()
            st.rerun()

        st.markdown(
            "<div style='font-size:11px;color:#4a5568;margin-top:8px'>"
            "Uploaded data is stored in your browser session only — it never touches the "
            "filesystem and never persists between sessions or across users. "
            "Clearing uploaded data switches you back to NASA Research Mode."
            "</div>",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 1: Data sources
    # ────────────────────────────────────────────────────────────────────────
    _section("Data Sources")

    synth_ids = [c for c in featured_dfs if c not in NASA_CELL_IDS and c not in up_fdfs]
    nasa_ids  = [c for c in featured_dfs if c in NASA_CELL_IDS and c not in up_fdfs]

    src_col1, src_col2 = st.columns(2)
    with src_col1:
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 20px'>"
            f"<div style='font-size:12px;font-weight:600;color:#fc8181;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>Synthetic cells</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{len(synth_ids)}</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.6'>"
            f"Physics-informed simulation (Arrhenius SEI growth, empirical C-rate factor, "
            f"Rainflow DoD scaling). Resistance: 0.15–0.40 Ω internal. "
            f"<strong>Not real measured data.</strong></div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:8px'>"
            f"{', '.join(synth_ids)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with src_col2:
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 20px'>"
            f"<div style='font-size:12px;font-weight:600;color:#48bb78;text-transform:uppercase;"
            f"letter-spacing:0.07em;margin-bottom:8px'>NASA PCoE real cells</div>"
            f"<div style='font-size:26px;font-weight:700;color:#e2e8f0'>{len(nasa_ids)}</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:4px;line-height:1.6'>"
            f"LiCoO₂ 18650 cells, ~2 Ah, 24°C, 2A constant discharge. "
            f"Re (electrolyte resistance) from EIS: 0.04–0.07 Ω. "
            f"Source: Saha &amp; Goebel (2007), NASA PCoE dataset.</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:8px'>"
            f"{', '.join(nasa_ids) if nasa_ids else 'Not loaded — run src/nasa_loader.py'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-top:12px;padding:10px 14px;"
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
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Leave-cell-out (LCO) cross-validation trains on N−1 cells and tests on the held-out "
        "cell entirely. This is the honest generalisation metric — a row-level split on "
        "a multi-cell dataset leaks cell identity into training. Per-cell fold R² below "
        "the reliability floor gates RUL display for that cell across all pages.</div>",
        unsafe_allow_html=True,
    )

    for source_key, bundle in bundles.items():
        if bundle is None:
            continue
        m = bundle["metrics"]
        lco_per = m.get("lco_per_cell", {})
        per_cell_ok = m.get("per_cell_rul_reliable", {})
        label = "NASA PCoE" if source_key == "nasa" else "Synthetic"
        colour = "#48bb78" if source_key == "nasa" else "#fc8181"

        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{colour};"
            f"margin:16px 0 8px'>{label} model</div>",
            unsafe_allow_html=True,
        )

        header_cols = st.columns([2, 1, 1, 1, 2])
        for col, hdr in zip(header_cols, ["Cell", "SOH fold R²", "RUL fold R²", "RUL status", "Note"]):
            col.markdown(
                f"<div style='font-size:10px;font-weight:600;color:#4a5568;"
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
            row_cols[4].markdown(f"<div style='font-size:11px;color:#4a5568;padding:4px 0'>{note}</div>", unsafe_allow_html=True)

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

    # ────────────────────────────────────────────────────────────────────────
    # Section 2b: Application Profile + EOL threshold
    # ────────────────────────────────────────────────────────────────────────
    _section("Application Profile")
    _PROFILES = {
        "Custom (manual)":       None,
        "EV / Passenger Vehicle": 80.0,
        "Stationary Storage":    70.0,
        "Industrial UPS":        75.0,
        "Second-Life Reuse":     60.0,
    }
    _cur_profile = st.session_state.get("app_profile", "Custom (manual)")
    _new_profile = st.selectbox(
        "Application profile",
        options=list(_PROFILES.keys()),
        index=list(_PROFILES.keys()).index(_cur_profile) if _cur_profile in _PROFILES else 0,
        key="app_profile_select",
        help="Selecting a profile auto-sets the EOL threshold below. "
             "You can still override it manually after selecting a profile.",
    )
    if _new_profile != _cur_profile:
        st.session_state["app_profile"] = _new_profile
        db.set_setting("app_profile", _new_profile)
        if _PROFILES[_new_profile] is not None:
            st.session_state["eol_threshold_pct"] = _PROFILES[_new_profile]
            db.set_setting("eol_threshold_pct", _PROFILES[_new_profile])
        st.rerun()
    if _PROFILES.get(_new_profile) is not None:
        st.caption(
            f"Profile '{_new_profile}' sets EOL threshold to {_PROFILES[_new_profile]:.0f}% SOH. "
            f"Adjust the slider below to override."
        )

    _section("Application End-of-Life Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "The EOL threshold defines when a cell is 'retired' for your application. "
        "The standard industry convention is <strong style='color:#8896a8'>80% SOH</strong>, "
        "but this is not universal — a delivery van needing 90% range may retire at 88%, "
        "while stationary grid storage may run to 70%. "
        "Changing this threshold adjusts the displayed RUL on the Overview page "
        "using the current fade rate — <strong style='color:#8896a8'>it does not retrain the model</strong>. "
        "The model was trained on 80% EOL; the adjusted RUL is a fade-rate projection, not a new model prediction.</div>",
        unsafe_allow_html=True,
    )

    eol_col1, eol_col2 = st.columns([4, 1])
    with eol_col1:
        new_eol = st.slider(
            "Application EOL threshold (%)",
            min_value=70, max_value=95, step=1,
            value=int(st.session_state.get("eol_threshold_pct", 80)),
            key="settings_eol_threshold",
            help="RUL on Overview will reflect cycles remaining until SOH hits this value.",
        )
    with eol_col2:
        st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
        if st.button("Reset to 80%", key="settings_eol_reset"):
            st.session_state["eol_threshold_pct"] = 80.0
            db.set_setting("eol_threshold_pct", 80.0)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if new_eol != int(st.session_state.get("eol_threshold_pct", 80)):
        st.session_state["eol_threshold_pct"] = float(new_eol)
        db.set_setting("eol_threshold_pct", float(new_eol))
        st.rerun()

    if new_eol != 80:
        direction = "earlier" if new_eol > 80 else "later"
        st.markdown(
            f"<div style='font-size:12px;color:#d69e2e;margin:4px 0 8px;"
            f"padding:6px 12px;background:rgba(214,158,46,0.08);border-radius:6px;"
            f"border-left:3px solid #d69e2e'>"
            f"Active: {new_eol}% EOL threshold — RUL will show {direction} retirement than "
            f"the standard 80% convention. Model predictions are still anchored to 80%; "
            f"the Overview adjustment uses linear fade-rate extrapolation.</div>",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 2c: Alert thresholds
    # ────────────────────────────────────────────────────────────────────────
    _section("🔔 Alert Thresholds")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Configure alert thresholds that trigger warning banners across all pages. "
        "Alerts are evaluated on every page load against the currently active fleet.</div>",
        unsafe_allow_html=True,
    )

    _at_col1, _at_col2, _at_col3 = st.columns(3)
    with _at_col1:
        soh_alert = st.slider(
            "SOH Warning Threshold (%)", 70, 95,
            int(st.session_state.get("soh_alert_pct", 85)),
            key="soh_alert_pct",
        )
        st.caption("Show warning banner when any cell's SOH drops below this level.")
    with _at_col2:
        resistance_alert = st.slider(
            "Resistance Alert Multiplier (×initial)", 1.2, 3.0,
            float(st.session_state.get("resistance_alert_mult", 1.8)),
            step=0.1,
            key="resistance_alert_mult",
        )
        st.caption("Alert when resistance exceeds this multiple of the cell's initial resistance.")
    with _at_col3:
        spread_alert = st.slider(
            "Pack Spread Alert (%)", 1.0, 10.0,
            float(st.session_state.get("spread_alert_pct", 5.0)),
            step=0.5,
            key="spread_alert_pct",
        )
        st.caption("Alert when SOH spread across fleet cells exceeds this threshold.")

    # ────────────────────────────────────────────────────────────────────────
    # Section 3: RUL reliability threshold
    # ────────────────────────────────────────────────────────────────────────
    _section("RUL Reliability Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "The reliability floor gates whether RUL predictions are shown or suppressed. "
        "Cells whose held-out fold R² falls below this value have RUL withheld across "
        "all pages — shown as 'Not calibrated' instead of a cycle count. "
        "Adjusting the slider below shows which cells would flip at different thresholds. "
        "<strong style='color:#8896a8'>This is a read-only preview</strong> — "
        "the active floor is hardcoded at "
        f"<code style='color:#63b3ed'>{RUL_RELIABLE_FLOOR}</code> in "
        "<code style='color:#63b3ed'>src/lco_eval.py</code> and requires a code change to modify.</div>",
        unsafe_allow_html=True,
    )

    slider_col, reset_col = st.columns([5, 1])
    with slider_col:
        preview_floor = st.slider(
            "Preview threshold",
            min_value=0.0, max_value=0.5,
            value=float(st.session_state.get("settings_rul_floor_preview", RUL_RELIABLE_FLOOR)),
            step=0.05,
            key="settings_rul_floor_preview",
            help=f"Active floor in code: {RUL_RELIABLE_FLOOR}. Drag to see which cells would flip at different thresholds.",
        )
    with reset_col:
        st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
        if st.button("Reset", key="settings_rul_reset", help="Reset to default (0.30)"):
            st.session_state["settings_rul_floor_preview"] = float(RUL_RELIABLE_FLOOR)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    preview_rows = []
    for source_key, bundle in bundles.items():
        if bundle is None:
            continue
        lco_per = bundle["metrics"].get("lco_per_cell", {})
        for cell_id, fold in lco_per.items():
            rul_r2 = fold.get("rul_r2", None)
            active_ok  = rul_r2 >= RUL_RELIABLE_FLOOR if rul_r2 is not None else True
            preview_ok = rul_r2 >= preview_floor       if rul_r2 is not None else True
            changed = active_ok != preview_ok
            preview_rows.append((cell_id, rul_r2, active_ok, preview_ok, changed))

    th_cols = st.columns([2, 1, 1, 1, 2])
    for col, hdr in zip(th_cols, ["Cell", "RUL fold R²", f"At {RUL_RELIABLE_FLOOR} (active)", f"At {preview_floor:.2f} (preview)", "Change"]):
        col.markdown(
            f"<div style='font-size:10px;font-weight:600;color:#4a5568;"
            f"text-transform:uppercase;letter-spacing:0.06em'>{hdr}</div>",
            unsafe_allow_html=True,
        )

    # Example callout: at R²≥0.25 the B0018 NASA cell would cross the floor
    if 0.20 <= preview_floor <= 0.29:
        st.markdown(
            "<div style='font-size:12px;color:#d69e2e;margin:4px 0 8px;"
            "padding:6px 12px;background:rgba(214,158,46,0.08);border-radius:6px;"
            "border-left:3px solid #d69e2e'>"
            f"At R²≥{preview_floor:.2f}: B0018 becomes reliable — fold R²=0.22, "
            f"currently withheld (below the 0.30 active floor)."
            "</div>",
            unsafe_allow_html=True,
        )

    for cell_id, rul_r2, active_ok, preview_ok, changed in preview_rows:
        def _status(ok): return ("<span style='color:#2f855a'>✓ Shown</span>" if ok
                                 else "<span style='color:#c05621'>✗ Withheld</span>")
        change_html = (
            "<span style='color:#d69e2e;font-weight:600'>⚑ Would flip</span>" if changed
            else "<span style='color:#2d3748'>—</span>"
        )
        r2_str = f"{rul_r2:.2f}" if rul_r2 is not None else "—"
        row = st.columns([2, 1, 1, 1, 2])
        row[0].markdown(f"<div style='font-size:13px;color:#e2e8f0;padding:4px 0'>{cell_id}</div>", unsafe_allow_html=True)
        row[1].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:4px 0'>{r2_str}</div>", unsafe_allow_html=True)
        row[2].markdown(f"<div style='padding:4px 0'>{_status(active_ok)}</div>", unsafe_allow_html=True)
        row[3].markdown(f"<div style='padding:4px 0'>{_status(preview_ok)}</div>", unsafe_allow_html=True)
        row[4].markdown(f"<div style='padding:4px 0'>{change_html}</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # Section 3b: Model cache
    # ────────────────────────────────────────────────────────────────────────
    _section("Model Cache")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Training takes 20–60 s on first run. The trained model bundles are stored on disk "
        "in <code style='color:#63b3ed'>.cache/bundles/</code> and reloaded instantly on "
        "subsequent runs. The cache is automatically invalidated when cycle counts change "
        "(new data imported). Use the button below to force a full retrain on next app load.</div>",
        unsafe_allow_html=True,
    )

    from bundle_cache import clear_cache as _clear_bundle_cache, CACHE_DIR as _CACHE_DIR
    import pathlib as _pathlib
    cache_files = list(_CACHE_DIR.glob("*.joblib")) if _CACHE_DIR.exists() else []
    cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024) if cache_files else 0
    cache_info = f"{len(cache_files)} bundle(s) · {cache_size_mb:.1f} MB" if cache_files else "No cache on disk — will train fresh on next load"

    st.markdown(
        f"<div style='font-size:13px;color:#a0aec0;margin-bottom:12px'>"
        f"Current cache: <strong style='color:#e2e8f0'>{cache_info}</strong></div>",
        unsafe_allow_html=True,
    )

    _cache_col1, _cache_col2 = st.columns([2, 3])
    with _cache_col1:
        if st.button("Clear model cache", key="settings_clear_cache",
                     help="Deletes all cached .joblib bundles. Models will retrain on next app load."):
            _clear_bundle_cache()
            st.success("Model cache cleared — models will retrain on next app load.")
            st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Section 4: CRM Configuration (Critical Raw Materials)
    # ────────────────────────────────────────────────────────────────────────
    _section("CRM Configuration (EU Battery Regulation Art. 13)")
    st.caption(
        "Configure the critical raw material percentages for your cell chemistries. "
        "These values feed the Passport page CRM section and can be updated as supply-chain "
        "audit data becomes available. Leave blank to use built-in estimates."
    )

    with st.expander("LFP (Severson cells — LiFePO4)", expanded=False):
        st.number_input(
            "Lithium (Li) content — wt%", min_value=0.0, max_value=20.0, step=0.1,
            value=float(st.session_state.get("crm_lfp_li_pct", 4.4)),
            key="crm_lfp_li_pct",
            help="LFP cathode + graphite anode. Default 4.4 wt% from literature.",
        )

    with st.expander("LiCoO2 / NCA (NASA cells)", expanded=False):
        _c1, _c2 = st.columns(2)
        with _c1:
            st.number_input("Co content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_nca_co_pct", 14.0)), key="crm_nca_co_pct",
                help="LiCoO2 baseline ~14 wt%.")
            st.number_input("Ni content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_nca_ni_pct", 0.0)), key="crm_nca_ni_pct",
                help="Pure LiCoO2 has no Ni. NMC variants: 15–33 wt%.")
            st.number_input("Li content — wt%", min_value=0.0, max_value=20.0, step=0.1,
                value=float(st.session_state.get("crm_nca_li_pct", 7.0)), key="crm_nca_li_pct",
                help="Cathode + anode combined estimate.")
        with _c2:
            st.number_input("Recycled Co — %", min_value=0.0, max_value=100.0, step=0.5,
                value=float(st.session_state.get("crm_nca_recycled_co_pct", 0.0)),
                key="crm_nca_recycled_co_pct",
                help="EU 2030 target: 12%. Enter actual audit figure.")
            st.number_input("Recycled Ni — %", min_value=0.0, max_value=100.0, step=0.5,
                value=float(st.session_state.get("crm_nca_recycled_ni_pct", 0.0)),
                key="crm_nca_recycled_ni_pct",
                help="EU 2030 target: 4%. Enter actual audit figure.")

    with st.expander("Synthetic cells (LiCoO2 model)", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.number_input("Co content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_synth_co_pct", 14.0)), key="crm_synth_co_pct")
        with _c2:
            st.number_input("Ni content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_synth_ni_pct", 0.0)), key="crm_synth_ni_pct")
        with _c3:
            st.number_input("Li content — wt%", min_value=0.0, max_value=20.0, step=0.1,
                value=float(st.session_state.get("crm_synth_li_pct", 7.0)), key="crm_synth_li_pct")

    with st.expander("User-uploaded cells", expanded=False):
        _c1, _c2 = st.columns(2)
        with _c1:
            st.number_input("Co content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_user_co_pct", 0.0)), key="crm_user_co_pct")
            st.number_input("Ni content — wt%", min_value=0.0, max_value=40.0, step=0.1,
                value=float(st.session_state.get("crm_user_ni_pct", 0.0)), key="crm_user_ni_pct")
            st.number_input("Li content — wt%", min_value=0.0, max_value=20.0, step=0.1,
                value=float(st.session_state.get("crm_user_li_pct", 0.0)), key="crm_user_li_pct")
        with _c2:
            st.number_input("Recycled Co — %", min_value=0.0, max_value=100.0, step=0.5,
                value=float(st.session_state.get("crm_user_recycled_co_pct", 0.0)),
                key="crm_user_recycled_co_pct")
            st.number_input("Recycled Ni — %", min_value=0.0, max_value=100.0, step=0.5,
                value=float(st.session_state.get("crm_user_recycled_ni_pct", 0.0)),
                key="crm_user_recycled_ni_pct")

    # ────────────────────────────────────────────────────────────────────────
    # Section 5: Cost-of-Delay multiplier
    # ────────────────────────────────────────────────────────────────────────
    _section("Cost-of-Delay Multiplier")
    st.markdown(
        f"{make_badge('Illustrative — not sourced', '#718096')} &nbsp;"
        "The residual value penalty per % SOH below EOL threshold at replacement is "
        "a modelling assumption with no universal market data behind it. "
        "Adjust to match your fleet's observed resale or second-life market.",
        unsafe_allow_html=True,
    )
    _cod_mult = st.slider(
        "Value penalty per % SOH below EOL at replacement (%)",
        min_value=0.5, max_value=5.0, step=0.5,
        value=float(st.session_state.get("cost_of_delay_mult", 2.0)),
        key="settings_cod_mult",
        help="Default 2.0: each % SOH below EOL threshold at actual replacement = 2% additional value loss. Illustrative only.",
    )
    if _cod_mult != st.session_state.get("cost_of_delay_mult", 2.0):
        st.session_state["cost_of_delay_mult"] = _cod_mult
        db.set_setting("cost_of_delay_mult", _cod_mult)
        st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Section 6: About
    # ────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────
    # Webhook Notifications
    # ────────────────────────────────────────────────────────────────────────
    _section("Anomaly Webhook Notifications")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "POST a JSON payload to your endpoint whenever IEC 62619:2022 anomaly flags fire in the "
        "Live Monitor. Use this to trigger Slack alerts, PagerDuty incidents, or CMMS tickets."
        "</div>"
    )
    _wh_col1, _wh_col2 = st.columns([3, 1])
    _wh_url = _wh_col1.text_input(
        "Webhook URL", value=st.session_state.get("webhook_url", ""),
        placeholder="https://hooks.slack.com/services/...",
        key="webhook_url",
    )
    _wh_secret = _wh_col2.text_input(
        "HMAC secret (optional)", value=st.session_state.get("webhook_secret", ""),
        type="password", key="webhook_secret",
        help="If set, each request includes X-Signature-256: hmac-sha256 of the body.",
    )
    _wh_events = st.multiselect(
        "Fire on", key="webhook_events",
        options=["THERMAL_RUNAWAY_PRECURSOR", "UNDERTEMPERATURE", "CAPACITY_PLUNGE",
                 "VOLTAGE_HIGH", "VOLTAGE_LOW", "TEMPERATURE_HIGH", "SOC_ANOMALY",
                 "FLEET_DIGEST", "TRAJECTORY_MATCH", "PASSPORT_GAP"],
        default=st.session_state.get("webhook_events",
            ["THERMAL_RUNAWAY_PRECURSOR", "CAPACITY_PLUNGE", "VOLTAGE_HIGH"]),
        help="Only event types checked here will trigger a webhook POST. FLEET_DIGEST/"
             "TRAJECTORY_MATCH/PASSPORT_GAP are session/page-load-triggered best-effort "
             "alerts, not a real background cron.",
    )
    db.set_setting("webhook_url", _wh_url)
    db.set_setting("webhook_secret", _wh_secret)
    db.set_setting("webhook_events", _wh_events)
    if _wh_url:
        from notifications import send_webhook
        _wh_test_col, _wh_digest_col, _ = st.columns([1, 1, 3])
        if _wh_test_col.button("Send test ping", key="webhook_test_btn"):
            _ok = send_webhook(
                "TEST_PING",
                {"message": "Webhook connectivity test from Settings page."},
                _wh_url, _wh_secret,
            )
            if _ok:
                st.success("Test ping sent.")
            else:
                st.warning("Webhook did not return a success response — check the URL and endpoint.")
        if _wh_digest_col.button("Send digest now", key="webhook_digest_btn",
                                  help="Manually trigger the fleet digest webhook (also checked "
                                       "automatically, best-effort, on Fleet page load)."):
            if "FLEET_DIGEST" not in _wh_events:
                st.warning("Add FLEET_DIGEST to 'Fire on' above to enable this.")
            else:
                _n_cells = len(featured_dfs)
                _n_flagged = sum(
                    1 for _df in featured_dfs.values()
                    if len(_df) and "soh_pct" in _df.columns
                    and float(_df["soh_pct"].iloc[-1]) < st.session_state.get("eol_threshold_pct", 80.0)
                )
                _ok = send_webhook(
                    "FLEET_DIGEST",
                    {"n_cells": _n_cells, "n_flagged_below_eol": _n_flagged},
                    _wh_url, _wh_secret,
                )
                if _ok:
                    db.set_setting("last_digest_sent", datetime.date.today().isoformat())
                    st.success(f"Digest sent — {_n_cells} cells, {_n_flagged} below EOL threshold.")
                else:
                    st.warning("Webhook did not return a success response.")
    else:
        st.caption("Enter a webhook URL above to enable push notifications.")

    # ────────────────────────────────────────────────────────────────────────
    # BMS Connector (Victron VRM)
    # ────────────────────────────────────────────────────────────────────────
    _section("BMS Connector (Victron VRM)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Pull real cycle data directly from a Victron VRM installation instead of manual CSV "
        "upload. Paste your own VRM API token and installation ID below — credentials are "
        "never hardcoded and this platform never contacts VRM without both fields set."
        "</div>"
    )
    _bms_col1, _bms_col2 = st.columns(2)
    _vrm_token = _bms_col1.text_input(
        "VRM API token", value=st.session_state.get("vrm_api_token", ""),
        type="password", key="vrm_api_token",
        help="Generate under VRM Portal -> Settings -> Integrations -> API access tokens.",
    )
    _vrm_install_id = _bms_col2.text_input(
        "VRM installation ID", value=st.session_state.get("vrm_installation_id", ""),
        key="vrm_installation_id",
    )
    db.set_setting("vrm_api_token", _vrm_token)
    db.set_setting("vrm_installation_id", _vrm_install_id)
    if not (_vrm_token and _vrm_install_id):
        _empty_state(
            "Not yet connected",
            "Paste your VRM API token and installation ID above to enable pulling real cycle "
            "data directly from your Victron installation.",
            icon="🔌",
        )
    else:
        if st.button("Test VRM connection", key="vrm_test_btn"):
            from bms_connectors import fetch_victron_vrm
            try:
                _vrm_df = fetch_victron_vrm(
                    "https://vrmapi.victronenergy.com/v2", _vrm_token, _vrm_install_id,
                )
                if _vrm_df is None or len(_vrm_df) == 0:
                    st.warning("Connected, but no battery records were returned for this installation.")
                else:
                    st.success(f"Fetched {len(_vrm_df)} records from VRM installation {_vrm_install_id}.")
            except Exception as _vrm_e:
                st.error(f"VRM connection failed: {_vrm_e}")

    # ────────────────────────────────────────────────────────────────────────
    # LLM Copilot API Key
    # ────────────────────────────────────────────────────────────────────────
    _section("AI Copilot — Language Model")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "When an Anthropic API key is set, the Copilot answers in natural language using "
        "<strong style='color:#e2e8f0'>Claude Haiku</strong> — strictly constrained to the "
        "values in this platform's model bundle (no hallucination of numbers). "
        "Without a key, template answers are used as fallback."
        "</div>"
    )
    _llm_key = st.text_input(
        "Anthropic API key", type="password",
        value=st.session_state.get("anthropic_api_key", ""),
        placeholder="sk-ant-...",
        key="anthropic_api_key",
        help="Key is stored in session state only — never written to disk or transmitted except to Anthropic.",
    )
    if _llm_key:
        if _llm_key.startswith("sk-ant-"):
            st.success("Claude Haiku active — Copilot will use natural language responses.")
        else:
            st.warning("Key doesn't look like an Anthropic key (expected sk-ant-...). Check and re-enter.")
    else:
        st.caption("Without an API key the Copilot uses template answers grounded on bundle values.")

    _section("Onboarding")
    if st.button("↺ Replay guided tour", key="settings_replay_tour"):
        st.session_state["tour_seen"] = False
        st.session_state["tour_step"] = 0
        st.rerun()

    _section("About")

    phase_rows = [
        ("Phase 1", "Core Loop",       "SOH/RUL model, LCO validation, per-cell reliability gate, Overview/Health/Insights",       "Done"),
        ("Phase 2", "Fleet",           "Multi-cell fleet ranking by SOH + fade rate; cross-type RUL gate documented",               "Done"),
        ("Phase 3", "Copilot",         "Template-based narration grounded on bundle outputs — no LLM, no invented numbers",         "Done"),
        ("Phase 4", "Consequences",    "Second-life economics: fit scoring, break-even chart, full assumption register",             "Done"),
        ("Phase 5", "Passport",        "EU 2023/1542 Battery Passport field structure; PDF export via reportlab",                   "Done"),
        ("Phase 6", "Recommendations", "4-tier auditable confidence system; dual-signal SOH + fade acceleration routing",           "Done"),
        ("Phase 7", "Sustainability",  "Lifecycle CO₂ chart, critical materials tracker, EU recycled-content targets",              "Done"),
        ("Phase 8", "Design System",   "design_system.py: badge constants, state badges, color tokens; base_layout() documented",  "Done"),
        ("Phase 9", "Settings",        "Model transparency, per-cell LCO table, RUL floor preview, data source panel",             "Done"),
    ]

    for ph, name, desc, status in phase_rows:
        status_c = C_GREEN if status == "Done" else C_MUTED
        st.markdown(
            f"<div style='display:flex;gap:16px;padding:10px 0;border-bottom:1px solid #2d3748;align-items:flex-start'>"
            f"<div style='min-width:64px;font-size:11px;font-weight:600;color:#4a5568;padding-top:2px'>{ph}</div>"
            f"<div style='min-width:120px;font-size:13px;font-weight:600;color:#e2e8f0'>{name}</div>"
            f"<div style='flex:1;font-size:12px;color:#8896a8;line-height:1.5'>{desc}</div>"
            f"<div style='min-width:48px;font-size:12px;font-weight:600;color:{status_c};text-align:right'>{status}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#4a5568;line-height:1.8;padding:14px 18px;"
        "background:#1a202c;border-radius:8px'>"
        "<strong style='color:#8896a8'>Stack</strong> — "
        "scikit-learn GBRT · Streamlit · Plotly · reportlab<br>"
        "<strong style='color:#8896a8'>Model</strong> — "
        "Two separate GBRT instances (synthetic / NASA); leave-cell-out cross-validation<br>"
        "<strong style='color:#8896a8'>Data</strong> — "
        "8 synthetic cells (physics-informed) + 4 NASA PCoE cells (Saha &amp; Goebel, 2007) + "
        "12 Severson LFP cells (Severson et al., 2019, when cached)<br>"
        "<strong style='color:#8896a8'>Regulatory</strong> — "
        "EU Battery Regulation (EU) 2023/1542 — field structure demonstration only; "
        "not a compliance claim<br>"
        "<strong style='color:#8896a8'>Source</strong> — "
        "<a href='https://github.com/seyedali1996lb-svg/battery-intelligence-platform' "
        "style='color:#63b3ed'>github.com/seyedali1996lb-svg/battery-intelligence-platform</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def _clear_uploaded_data():
    """Remove all uploaded data from session state and revert to NASA mode."""
    for k in ["uploaded_featured_dfs", "uploaded_bundle", "uploaded_split_cycles", "uploaded_mode_meta"]:
        st.session_state.pop(k, None)
    st.session_state["data_mode"] = "nasa"
