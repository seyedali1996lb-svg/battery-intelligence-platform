"""Settings page — configuration tier (everything below Model Transparency).

Extracted from settings.py's page_settings() (was 1,031 lines -- the 5th
largest file in app/_pages/). page_settings() keeps the sections most
tied to loaded data (My Data, Data Sources, Model Transparency) and calls
render_settings_configuration() for the rest: Application Profile/EOL
threshold, Alert Thresholds, RUL Reliability Threshold preview, Model
Cache, CRM Configuration, Cost-of-Delay Multiplier, Webhook Notifications,
BMS Connectors (Victron/Orion), Second-Life Marketplace, Maintenance
Write-Back, Team Members, Sites & Fleets, AI Copilot key, Onboarding
replay, Production Readiness Roadmap, and About.

Each section already reads/writes its own db.set_setting()/session_state
keys independently -- there is no shared local state between sections in
the original function beyond featured_dfs/bundles (threaded through as
parameters where actually used), so this split is a low-risk, mostly
mechanical move: one function per _section() boundary, called in the
original order from render_settings_configuration().
"""

import sys
import os
import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from design_system import make_badge, section_header_html, C_GREEN, C_MUTED
from utils import _md_html, _empty_state
import db


def _section(title: str):
    st.markdown(section_header_html(title), unsafe_allow_html=True)


def _set_secret_setting(org_id: int, key: str, value, role: "str | None" = None) -> None:
    """Wraps db.set_setting() for the 5 credential-shaped keys, surfacing
    db.InsecureCredentialStorageError/db.InsufficientRoleError as a page
    error instead of an uncaught exception -- these calls fire on every
    Streamlit rerun (not behind a Save button), so either failure would
    otherwise crash the whole Settings page rather than just fail to save.
    role is forwarded to db.set_setting() -- every one of these 5 keys is
    also in db._ADMIN_ONLY_SETTING_KEYS, enforced there, not just by the
    orchestrator hiding this section's widgets from non-admins (defense in
    depth -- see db.InsufficientRoleError's docstring)."""
    try:
        db.set_setting(org_id, key, value, role=role)
    except db.InsecureCredentialStorageError:
        st.error(
            f"Not saved: SETTINGS_ENCRYPTION_KEY must be set before a new "
            f"{key.replace('_', ' ')} can be stored securely (see the "
            "warning banner above)."
        )
    except db.InsufficientRoleError:
        st.error(f"Not saved: {key.replace('_', ' ')} is org-wide configuration, restricted to the admin role.")


def render_encryption_key_warning() -> None:
    """Admin-only banner surfacing src/db.py's using_fallback_encryption_key()
    -- a server log line alone is easy to miss on a live deployment; an
    admin who actually opens Settings should see this directly. Not shown
    to non-admin roles since it names a real security gap in this org's
    deployment, not something every viewer needs surfaced."""
    if st.session_state.get("auth_role") != "admin":
        return
    if not db.using_fallback_encryption_key():
        return
    st.markdown(
        "<div style='background:rgba(252,129,129,0.08);border:1px solid rgba(252,129,129,0.3);"
        "border-radius:8px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#fed7d7;line-height:1.6'>"
        "⚠ <strong>SETTINGS_ENCRYPTION_KEY is not set.</strong> Stored credentials below "
        "(VRM/Orion/Circunomics/CMMS API keys, webhook secret) are being encrypted with a "
        "fallback key that is public in this repo's source — anyone with read access to the "
        "source can decrypt them. Set SETTINGS_ENCRYPTION_KEY via a real secrets manager "
        "before storing real credentials on this deployment."
        "</div>",
        unsafe_allow_html=True,
    )


def render_application_profile_and_eol() -> None:
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
        db.set_setting(st.session_state["auth_org_id"], "app_profile", _new_profile, role=st.session_state.get("auth_role"))
        if _PROFILES[_new_profile] is not None:
            st.session_state["eol_threshold_pct"] = _PROFILES[_new_profile]
            db.set_setting(st.session_state["auth_org_id"], "eol_threshold_pct", _PROFILES[_new_profile], role=st.session_state.get("auth_role"))
        st.rerun()
    if _PROFILES.get(_new_profile) is not None:
        st.caption(
            f"Profile '{_new_profile}' sets EOL threshold to {_PROFILES[_new_profile]:.0f}% SOH. "
            f"Adjust the slider below to override."
        )

    _section("Application End-of-Life Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-bottom:14px;line-height:1.6'>"
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
            db.set_setting(st.session_state["auth_org_id"], "eol_threshold_pct", 80.0, role=st.session_state.get("auth_role"))
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if new_eol != int(st.session_state.get("eol_threshold_pct", 80)):
        st.session_state["eol_threshold_pct"] = float(new_eol)
        db.set_setting(st.session_state["auth_org_id"], "eol_threshold_pct", float(new_eol), role=st.session_state.get("auth_role"))
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


def render_alert_thresholds() -> None:
    _section("🔔 Alert Thresholds")

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-bottom:14px;line-height:1.6'>"
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


def render_rul_reliability_threshold(bundles: dict) -> None:
    from batlab.validation.lco import RUL_RELIABLE_FLOOR

    _section("RUL Reliability Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-bottom:14px;line-height:1.6'>"
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
            f"<div style='font-size:10px;font-weight:600;color:#a0aec0;"
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


def render_model_cache() -> None:
    _section("Model Cache")

    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;margin-bottom:14px;line-height:1.6'>"
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


def render_crm_configuration() -> None:
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

    with st.expander("LiCoO2 (NASA cells)", expanded=False):
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


def render_cost_of_delay() -> None:
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
        db.set_setting(st.session_state["auth_org_id"], "cost_of_delay_mult", _cod_mult, role=st.session_state.get("auth_role"))
        st.rerun()


def render_webhook_notifications(featured_dfs: dict) -> None:
    _section("Anomaly Webhook Notifications")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "POST a JSON payload to your endpoint whenever IEC 62619:2022 anomaly flags fire in the "
        "Live Monitor. Use this to trigger Slack alerts, PagerDuty incidents, or CMMS tickets."
        "</div>"
    )
    # Streamlit warns ("...was created with a default value but also had its
    # value set via the Session State API") if a widget's key is pre-seeded
    # in session_state (as these are, by main.py's hydration block) AND the
    # widget also gets a value=/default= kwarg — it can't tell that both
    # sides agree. Fix is the pattern Streamlit itself recommends: seed
    # session_state once here (only relevant the very first time an org
    # configures webhooks, before hydration has anything to load), then let
    # each widget read/write purely through its key= binding.
    if "webhook_url" not in st.session_state:
        st.session_state["webhook_url"] = ""
    if "webhook_secret" not in st.session_state:
        st.session_state["webhook_secret"] = ""
    if "webhook_events" not in st.session_state:
        st.session_state["webhook_events"] = [
            "THERMAL_RUNAWAY_PRECURSOR", "CAPACITY_PLUNGE", "VOLTAGE_HIGH",
        ]

    _wh_col1, _wh_col2 = st.columns([3, 1])
    _wh_url = _wh_col1.text_input(
        "Webhook URL",
        placeholder="https://hooks.slack.com/services/...",
        key="webhook_url",
    )
    _wh_secret = _wh_col2.text_input(
        "HMAC secret (optional)", type="password", key="webhook_secret",
        help="If set, each request includes X-Signature-256: hmac-sha256 of the body.",
    )
    _wh_events = st.multiselect(
        "Fire on", key="webhook_events",
        options=["THERMAL_RUNAWAY_PRECURSOR", "UNDERTEMPERATURE", "CAPACITY_PLUNGE",
                 "VOLTAGE_HIGH", "VOLTAGE_LOW", "TEMPERATURE_HIGH", "SOC_ANOMALY",
                 "FLEET_DIGEST", "TRAJECTORY_MATCH", "PASSPORT_GAP", "INGESTION_FAULT"],
        help="Only event types checked here will trigger a webhook POST. FLEET_DIGEST/"
             "TRAJECTORY_MATCH/PASSPORT_GAP are session/page-load-triggered best-effort "
             "alerts, not a real background cron.",
    )
    db.set_setting(st.session_state["auth_org_id"], "webhook_url", _wh_url, role=st.session_state.get("auth_role"))
    _set_secret_setting(st.session_state["auth_org_id"], "webhook_secret", _wh_secret, role=st.session_state.get("auth_role"))
    db.set_setting(st.session_state["auth_org_id"], "webhook_events", _wh_events, role=st.session_state.get("auth_role"))
    if _wh_url:
        from notifications import send_webhook, notify_subscribers
        _wh_test_col, _wh_digest_col, _ = st.columns([1, 1, 3])
        if _wh_test_col.button("Send test ping", key="webhook_test_btn"):
            _ping_payload = {"message": "Webhook connectivity test from Settings page."}
            _ok = send_webhook("TEST_PING", _ping_payload, _wh_url, _wh_secret)
            notify_subscribers(st.session_state["auth_org_id"], "TEST_PING", _ping_payload)
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
                # Reads precomputed CellSummary rows instead of every cell's
                # full per-cycle DataFrame -- see src/cell_store.py's module
                # docstring. Same digest logic as Fleet's own auto-digest.
                _active_ids_wh = set(featured_dfs.keys())
                _n_cells = len(_active_ids_wh)
                _n_flagged = sum(
                    1 for r in db.get_cell_summaries(st.session_state["auth_org_id"])
                    if r["cell_id"] in _active_ids_wh and r.get("soh_pct") is not None
                    and float(r["soh_pct"]) < st.session_state.get("eol_threshold_pct", 80.0)
                )
                _digest_payload = {"n_cells": _n_cells, "n_flagged_below_eol": _n_flagged}
                _ok = send_webhook("FLEET_DIGEST", _digest_payload, _wh_url, _wh_secret)
                notify_subscribers(st.session_state["auth_org_id"], "FLEET_DIGEST", _digest_payload)
                if _ok:
                    db.set_setting(st.session_state["auth_org_id"], "last_digest_sent", datetime.date.today().isoformat())
                    st.success(f"Digest sent — {_n_cells} cells, {_n_flagged} below EOL threshold.")
                else:
                    st.warning("Webhook did not return a success response.")
    else:
        st.caption("Enter a webhook URL above to enable push notifications.")


def render_webhook_subscriptions() -> None:
    """Additional webhook destinations on top of the single URL above --
    src/db.py's WebhookSubscription table + src/notifications.py's
    notify_subscribers(), fanned out from the same 7 real event call
    sites the single webhook already fires from, additively (see
    notify_subscribers()'s own docstring)."""
    _section("Additional Webhook Destinations")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Fan the same events out to more than one destination — e.g. Slack AND PagerDuty AND a "
        "custom CRM webhook at once, instead of being limited to the single URL above."
        "</div>"
    )
    org_id = st.session_state["auth_org_id"]
    role = st.session_state.get("auth_role")

    with st.expander("Add a destination", expanded=False):
        _sub_name = st.text_input("Name", key="whsub_name", placeholder="PagerDuty")
        _sub_url = st.text_input("Webhook URL", key="whsub_url", placeholder="https://events.pagerduty.com/v2/enqueue")
        _sub_secret = st.text_input("HMAC secret (optional)", type="password", key="whsub_secret")
        _sub_events = st.multiselect(
            "Fire on (leave empty for all events)", key="whsub_events",
            options=["THERMAL_RUNAWAY_PRECURSOR", "UNDERTEMPERATURE", "CAPACITY_PLUNGE",
                     "VOLTAGE_HIGH", "VOLTAGE_LOW", "TEMPERATURE_HIGH", "SOC_ANOMALY",
                     "FLEET_DIGEST", "TRAJECTORY_MATCH", "PASSPORT_GAP", "INGESTION_FAULT", "TEST_PING"],
        )
        if st.button("Add destination", key="whsub_add_btn"):
            if _sub_name and _sub_url:
                import uuid as _uuid_whsub
                db.save_webhook_subscription(org_id, {
                    "id": _uuid_whsub.uuid4().hex, "name": _sub_name, "url": _sub_url,
                    "secret": _sub_secret, "event_types": _sub_events,
                    "created_at": datetime.datetime.now().isoformat(),
                }, caller_role=role)
                st.success(f"Added destination: {_sub_name}.")
                st.rerun()
            else:
                st.warning("Name and URL are required.")

    _subs = db.get_webhook_subscriptions(org_id)
    if not _subs:
        st.caption("No additional destinations configured yet.")
    for _sub in _subs:
        _sub_col1, _sub_col2, _sub_col3 = st.columns([4, 2, 1])
        _sub_col1.markdown(
            f"<div style='font-size:13px;color:#e2e8f0;padding:4px 0'>{_sub['name']}</div>"
            f"<div style='font-size:11px;color:#8896a8'>{_sub['url']}</div>",
            unsafe_allow_html=True,
        )
        _sub_col2.markdown(
            f"<div style='font-size:11px;color:#a0aec0;padding:4px 0'>"
            f"{', '.join(_sub['event_types']) if _sub['event_types'] else 'All events'}</div>",
            unsafe_allow_html=True,
        )
        if _sub_col3.button("Remove", key=f"whsub_del_{_sub['id']}"):
            db.delete_webhook_subscription(org_id, _sub["id"], caller_role=role)
            st.rerun()


def render_bms_connector_victron() -> None:
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
    _set_secret_setting(st.session_state["auth_org_id"], "vrm_api_token", _vrm_token, role=st.session_state.get("auth_role"))
    db.set_setting(st.session_state["auth_org_id"], "vrm_installation_id", _vrm_install_id, role=st.session_state.get("auth_role"))
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


def render_bms_connector_orion() -> None:
    _section("BMS Connector (Orion Jr2)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Pull real cycle data directly from an Orion Jr2 BMS's REST gateway instead of "
        "manual CSV upload. No live Orion account exists to test this integration against — "
        "it is built against Orion's publicly documented REST shape, not verified end-to-end. "
        "Paste your own API key and cell ID below; this platform never contacts Orion without "
        "a key set."
        "</div>"
    )
    _orion_key = st.text_input(
        "Orion BMS API key", value=st.session_state.get("orion_bms_api_key", ""),
        type="password", key="orion_bms_api_key",
    )
    _orion_cell_id = st.text_input(
        "Cell ID to monitor", value=st.session_state.get("orion_bms_cell_id", ""),
        key="orion_bms_cell_id",
    )
    _set_secret_setting(st.session_state["auth_org_id"], "orion_bms_api_key", _orion_key, role=st.session_state.get("auth_role"))
    db.set_setting(st.session_state["auth_org_id"], "orion_bms_cell_id", _orion_cell_id, role=st.session_state.get("auth_role"))
    if not (_orion_key and _orion_cell_id):
        _empty_state(
            "Not yet connected",
            "Paste your Orion BMS API key and a cell ID above to enable pulling real cycle "
            "data directly from your Orion Jr2 gateway.",
            icon="🔌",
        )
    else:
        if st.button("Test Orion BMS connection", key="orion_test_btn"):
            from bms_connectors import fetch_orion_bms
            _orion_result = fetch_orion_bms(_orion_cell_id, _orion_key)
            if _orion_result is None:
                st.warning("Connected, but no battery records were returned for this cell.")
            elif isinstance(_orion_result, dict) and "error" in _orion_result:
                st.error(f"Orion BMS connection failed: {_orion_result['error']}")
            else:
                st.success(f"Fetched {len(_orion_result)} records for cell {_orion_cell_id}.")


def render_bms_connector_modbus() -> None:
    _section("BMS Connector (Modbus TCP)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Pull a live snapshot from a Modbus TCP BMS/inverter (Pylontech, Growatt, SMA, "
        "Deye, ...). Modbus has no universal register layout across vendors — supply your "
        "device's own register map as JSON below rather than relying on a guessed default "
        "that could silently misread a real device. No live device exists to test this "
        "integration against; it is built against pymodbus's documented API shape, not "
        "verified end-to-end, and requires the optional <code>pymodbus</code> package."
        "</div>"
    )
    _mb_col1, _mb_col2, _mb_col3 = st.columns(3)
    _mb_host = _mb_col1.text_input("Modbus host/IP", value=st.session_state.get("modbus_host", ""), key="modbus_host")
    _mb_port = _mb_col2.text_input("Port", value=st.session_state.get("modbus_port", "502"), key="modbus_port")
    _mb_unit = _mb_col3.text_input("Unit/slave ID", value=st.session_state.get("modbus_unit_id", "1"), key="modbus_unit_id")
    _mb_cell_id = st.text_input("Cell/pack ID to record readings under", value=st.session_state.get("modbus_cell_id", ""), key="modbus_cell_id")
    _mb_map_raw = st.text_area(
        "Register map (JSON)", value=st.session_state.get("modbus_register_map", ""),
        key="modbus_register_map", height=90,
        placeholder='{"capacity_ah": [100, 0.01], "resistance_ohm": [102, 0.0001], "temperature_c": [104, 0.1]}',
        help="One entry per field: [register_address, scale_factor]. Omit a field your device doesn't expose.",
    )
    for _k, _v in [("modbus_host", _mb_host), ("modbus_port", _mb_port), ("modbus_unit_id", _mb_unit),
                   ("modbus_cell_id", _mb_cell_id), ("modbus_register_map", _mb_map_raw)]:
        db.set_setting(st.session_state["auth_org_id"], _k, _v, role=st.session_state.get("auth_role"))

    if not (_mb_host and _mb_map_raw and _mb_cell_id):
        _empty_state(
            "Not yet connected",
            "Enter a host, cell/pack ID, and your device's register map (JSON) above to enable "
            "pulling live readings from a Modbus TCP BMS.",
            icon="🔌",
        )
        return
    try:
        _mb_map = {k: tuple(v) for k, v in json.loads(_mb_map_raw).items()}
    except (json.JSONDecodeError, AttributeError) as _mb_json_err:
        st.error(f"Register map is not valid JSON: {_mb_json_err}")
        return
    if st.button("Test Modbus connection", key="modbus_test_btn"):
        from bms_connectors import ModbusBMSAdapter
        _mb_adapter = ModbusBMSAdapter(_mb_host, int(_mb_port or 502), int(_mb_unit or 1), _mb_map, _mb_cell_id)
        _mb_result = _mb_adapter.test_connection()
        (st.success if _mb_result["ok"] else st.error)(_mb_result["message"])


def render_bms_connector_can() -> None:
    _section("BMS Connector (CAN Bus)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Listen for battery telemetry on a CAN bus (common in EV packs) via a caller-supplied "
        "PGN/CAN-ID map — the most assumption-heavy of this platform's BMS connectors, since "
        "there is no universal frame layout across BMS vendors/DBC files and no real device "
        "to confirm one against. Requires the optional <code>python-can</code> package."
        "</div>"
    )
    _can_col1, _can_col2, _can_col3 = st.columns(3)
    _can_channel = _can_col1.text_input("CAN channel", value=st.session_state.get("can_channel", ""), key="can_channel", placeholder="can0")
    _can_bustype = _can_col2.text_input("python-can bustype", value=st.session_state.get("can_bustype", "socketcan"), key="can_bustype")
    _can_node = _can_col3.text_input("Node ID", value=st.session_state.get("can_node_id", "0"), key="can_node_id")
    _can_cell_id = st.text_input("Cell/pack ID to record readings under", value=st.session_state.get("can_cell_id", ""), key="can_cell_id")
    _can_map_raw = st.text_area(
        "PGN / CAN-ID map (JSON)", value=st.session_state.get("can_pgn_map", ""),
        key="can_pgn_map", height=90,
        placeholder='{"capacity_ah": [256, 0, 0.01], "resistance_ohm": [257, 2, 0.0001], "temperature_c": [258, 0, 0.1]}',
        help="One entry per field: [can_id, byte_offset, scale_factor]. Omit a field your device doesn't broadcast.",
    )
    for _k, _v in [("can_channel", _can_channel), ("can_bustype", _can_bustype), ("can_node_id", _can_node),
                   ("can_cell_id", _can_cell_id), ("can_pgn_map", _can_map_raw)]:
        db.set_setting(st.session_state["auth_org_id"], _k, _v, role=st.session_state.get("auth_role"))

    if not (_can_channel and _can_map_raw and _can_cell_id):
        _empty_state(
            "Not yet connected",
            "Enter a CAN channel, cell/pack ID, and your device's PGN/CAN-ID map (JSON) above "
            "to enable listening for live telemetry on a CAN bus.",
            icon="🔌",
        )
        return
    try:
        _can_map = {k: tuple(v) for k, v in json.loads(_can_map_raw).items()}
    except (json.JSONDecodeError, AttributeError) as _can_json_err:
        st.error(f"PGN/CAN-ID map is not valid JSON: {_can_json_err}")
        return
    if st.button("Test CAN bus connection", key="can_test_btn"):
        from bms_connectors import CANBusBMSAdapter
        _can_adapter = CANBusBMSAdapter(_can_channel, _can_bustype, int(_can_node or 0), _can_map, _can_cell_id)
        _can_result = _can_adapter.test_connection()
        (st.success if _can_result["ok"] else st.error)(_can_result["message"])


def render_bms_connector_ocpp() -> None:
    _section("EV Charging Connector (OCPP)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Pull completed charging-session records from an OCPP Central System's REST reporting "
        "API (e.g. SteVe) for one charge point. This is charging-SESSION data, not per-cell "
        "cycle data — capacity is approximated from energy delivered at an assumed nominal "
        "voltage, and internal resistance is never reported by OCPP at all. No live Central "
        "System exists to test this integration against."
        "</div>"
    )
    _ocpp_col1, _ocpp_col2 = st.columns(2)
    _ocpp_url = _ocpp_col1.text_input("Central System URL", value=st.session_state.get("ocpp_central_system_url", ""), key="ocpp_central_system_url")
    _ocpp_cp_id = _ocpp_col2.text_input("Charge point ID", value=st.session_state.get("ocpp_charge_point_id", ""), key="ocpp_charge_point_id")
    _ocpp_key = st.text_input("Central System API key", value=st.session_state.get("ocpp_api_key", ""), type="password", key="ocpp_api_key")
    _set_secret_setting(st.session_state["auth_org_id"], "ocpp_api_key", _ocpp_key, role=st.session_state.get("auth_role"))
    db.set_setting(st.session_state["auth_org_id"], "ocpp_central_system_url", _ocpp_url, role=st.session_state.get("auth_role"))
    db.set_setting(st.session_state["auth_org_id"], "ocpp_charge_point_id", _ocpp_cp_id, role=st.session_state.get("auth_role"))

    if not (_ocpp_url and _ocpp_key and _ocpp_cp_id):
        _empty_state(
            "Not yet connected",
            "Enter your OCPP Central System's URL, API key, and a charge point ID above to "
            "enable pulling completed charging sessions.",
            icon="🔌",
        )
    else:
        if st.button("Test OCPP connection", key="ocpp_test_btn"):
            from bms_connectors import OCPPAdapter
            _ocpp_result = OCPPAdapter(_ocpp_url, _ocpp_key, _ocpp_cp_id).test_connection()
            (st.success if _ocpp_result["ok"] else st.error)(_ocpp_result["message"])


def render_adapter_settings(registry_key: str) -> None:
    """
    Generic Settings UI for any src/plugin_registry.py adapter with
    `rendered_via_registry=True` — widgets, persistence, empty-state
    guard, and a test-connection button, all driven by the adapter's
    `AdapterField` specs instead of a new hand-written function per
    integration. See plugin_registry.py's module docstring for the full
    "why this exists" story.
    """
    from plugin_registry import ADAPTER_REGISTRY

    meta = ADAPTER_REGISTRY[registry_key]
    _section(f"BMS Connector ({meta.name})")
    _md_html(f"<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>{meta.description}</div>")

    org_id = st.session_state["auth_org_id"]
    role = st.session_state.get("auth_role")
    raw_values: dict = {}
    text_cols = st.columns(2)
    _col_i = 0
    for f in meta.fields:
        widget_key = f"plugin_{registry_key}_{f.key}"
        if f.kind == "textarea_json":
            val = st.text_area(
                f.label, value=st.session_state.get(f.key, f.default),
                key=widget_key, height=90, help=f.help, placeholder=f.placeholder,
            )
        else:
            target = text_cols[_col_i % 2]
            _col_i += 1
            val = target.text_input(
                f.label, value=st.session_state.get(f.key, f.default),
                type="password" if f.kind == "password" else "default",
                key=widget_key, help=f.help, placeholder=f.placeholder,
            )
        raw_values[f.key] = val
        if f.secret:
            _set_secret_setting(org_id, f.key, val, role=role)
        else:
            db.set_setting(org_id, f.key, val, role=role)

    missing_required = [f.label for f in meta.fields if f.required and not raw_values.get(f.key)]
    if missing_required:
        _empty_state(
            "Not yet connected",
            f"Fill in {', '.join(missing_required)} above to enable pulling live readings from {meta.name}.",
            icon="🔌",
        )
        return

    parsed_values = dict(raw_values)
    for f in meta.fields:
        if f.kind == "textarea_json":
            try:
                parsed_values[f.key] = json.loads(raw_values[f.key]) if raw_values[f.key] else {}
            except json.JSONDecodeError as e:
                st.error(f"{f.label} is not valid JSON: {e}")
                return

    if st.button(f"Test {meta.name} connection", key=f"plugin_{registry_key}_test_btn"):
        adapter = meta.adapter_class(**meta.build_kwargs(parsed_values))
        result = adapter.test_connection()
        (st.success if result["ok"] else st.error)(result["message"])


def render_second_life_marketplace() -> None:
    _section("Second-Life Marketplace (Circunomics)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Submit second-life listings directly to Circunomics instead of the local demo "
        "listing. Circunomics does not publish a public self-serve API — this requires a "
        "partner API key issued after onboarding. Without a key, the Decision page's "
        "\"List on Circunomics\" button keeps using the local demo listing behavior."
        "</div>"
    )
    _circ_key = st.text_input(
        "Circunomics API key", value=st.session_state.get("circunomics_api_key", ""),
        type="password", key="circunomics_api_key",
        help="Issued by Circunomics after partner onboarding — not a public self-serve key.",
    )
    _set_secret_setting(st.session_state["auth_org_id"], "circunomics_api_key", _circ_key, role=st.session_state.get("auth_role"))
    if not _circ_key:
        _empty_state(
            "Not yet connected",
            "Paste a Circunomics partner API key above to submit listings directly instead "
            "of the local demo listing.",
            icon="♻",
        )
    else:
        if st.button("Test Circunomics connection", key="circ_test_btn"):
            from circunomics_adapter import list_cell_on_circunomics
            from adapter_contract import classify_result
            _circ_result = list_cell_on_circunomics(
                "TEST-CELL", 80.0, "LiCoO2", 2.0, 50.0, _circ_key,
            )
            _circ_state = classify_result(_circ_result)
            if _circ_state == "unconfigured":
                st.warning("No API key configured.")
            elif _circ_state == "error":
                st.error(f"Circunomics connection failed: {_circ_result['error']}")
            else:
                st.success("Circunomics connection succeeded.")


def render_maintenance_writeback() -> None:
    _section("Maintenance Write-Back (CMMS/ERP)")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Create a maintenance ticket in your CMMS/ERP system directly from the Decide & Ask "
        "page's recommendation. No specific target system is wired up here — this is a "
        "documented adapter pattern built against the generic REST shape most ticketing "
        "systems expose (Maximo, SAP PM, Fiix, UpKeep, etc.). Point it at your own system's "
        "endpoint and API key below; without a key, the \"Create CMMS ticket\" button stays "
        "disabled."
        "</div>"
    )
    _cmms_base_url = st.text_input(
        "CMMS/ERP API base URL", value=st.session_state.get("cmms_api_base_url", ""),
        key="cmms_api_base_url",
        help="e.g. https://your-instance.example-cmms.com/v1",
    )
    db.set_setting(st.session_state["auth_org_id"], "cmms_api_base_url", _cmms_base_url, role=st.session_state.get("auth_role"))
    _cmms_key = st.text_input(
        "CMMS/ERP API key", value=st.session_state.get("cmms_api_key", ""),
        type="password", key="cmms_api_key",
        help="Issued by your CMMS/ERP provider — not a public self-serve key.",
    )
    _set_secret_setting(st.session_state["auth_org_id"], "cmms_api_key", _cmms_key, role=st.session_state.get("auth_role"))
    if not _cmms_key:
        _empty_state(
            "Not yet connected",
            "Paste a CMMS/ERP API key above to enable ticket creation directly from the "
            "Decide & Ask page instead of manual logging only.",
            icon="🛠",
        )
    else:
        if st.button("Test CMMS connection", key="cmms_test_btn"):
            from cmms_adapter import create_maintenance_ticket
            from adapter_contract import classify_result
            _cmms_result = create_maintenance_ticket(
                "TEST-CELL", "Connection test", "Test ticket from Battery Intelligence Platform",
                "low", _cmms_key,
                api_base_url=_cmms_base_url or "https://api.example-cmms.com/v1",
            )
            _cmms_state = classify_result(_cmms_result)
            if _cmms_state == "unconfigured":
                st.warning("No API key configured.")
            elif _cmms_state == "error":
                st.error(f"CMMS connection failed: {_cmms_result['error']}")
            else:
                st.success("CMMS connection succeeded.")


def render_team_members() -> None:
    if st.session_state.get("auth_role") != "admin":
        return
    _section(f"Team Members — {st.session_state.get('auth_org_name', '')}")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Add teammates to this organization. They sign in with their own username/password "
        "and see this organization's data — decisions, cohort tags, settings, and uploaded "
        "fleet — never another organization's."
        "</div>"
    )
    _tm_col1, _tm_col2 = st.columns(2)
    _tm_username = _tm_col1.text_input("Username", key="invite_username", placeholder="bob")
    _tm_display  = _tm_col2.text_input("Display name (optional)", key="invite_display", placeholder="Bob Chen")
    _tm_col3, _tm_col4 = st.columns(2)
    _tm_password = _tm_col3.text_input("Password", key="invite_password", type="password")
    _tm_role     = _tm_col4.selectbox(
        "Role", ["engineer", "fleet", "compliance", "admin"], key="invite_role",
    )
    if st.button("Add teammate", key="invite_submit_btn"):
        if not (_tm_username.strip() and _tm_password):
            st.error("Username and password are required.")
        elif len(_tm_password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            try:
                _invite_result = db.create_user(
                    st.session_state["auth_org_id"], _tm_username, _tm_password, _tm_role, _tm_display,
                    caller_role=st.session_state.get("auth_role"),
                )
            except db.InsufficientRoleError:
                st.error("Not added: inviting teammates is restricted to the admin role.")
            else:
                if "error" in _invite_result:
                    st.error(_invite_result["error"])
                else:
                    st.success(f"Added {_tm_username} ({_tm_role}) to {st.session_state.get('auth_org_name', 'this organization')}.")


def render_sites_and_fleets(featured_dfs: dict) -> None:
    if st.session_state.get("auth_role") != "admin":
        return
    _section("Sites & Fleets")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Group cells by physical deployment: a Site (e.g. a warehouse or depot) contains "
        "one or more Fleets, each of which contains Packs of cells. Every organization "
        "starts with one Default Site / Default Fleet — add more as your deployment grows. "
        "This is separate from the Explore page's Virtual Pack Builder, which simulates "
        "series/parallel packs for analysis and doesn't persist anything."
        "</div>"
    )
    _org_id_sf = st.session_state["auth_org_id"]
    _sf_sites  = db.list_sites(_org_id_sf)

    _sf_col1, _sf_col2 = st.columns(2)
    _new_site_name = _sf_col1.text_input("New site name", key="new_site_name", placeholder="Warehouse B")
    if _sf_col1.button("Add site", key="add_site_btn"):
        if _new_site_name.strip():
            try:
                db.create_site(_org_id_sf, _new_site_name, caller_role=st.session_state.get("auth_role"))
            except db.InsufficientRoleError:
                st.error("Not added: creating sites is restricted to the admin role.")
            else:
                st.rerun()
        else:
            st.error("Site name is required.")

    if _sf_sites:
        _site_options = {s["name"]: s["id"] for s in _sf_sites}
        _sel_site_name = _sf_col2.selectbox("Site", list(_site_options.keys()), key="sf_site_select")
        _sel_site_id   = _site_options[_sel_site_name]

        _sf_col3, _sf_col4 = st.columns(2)
        _new_fleet_name = _sf_col3.text_input("New fleet name", key="new_fleet_name", placeholder="Forklift Fleet")
        if _sf_col3.button("Add fleet", key="add_fleet_btn"):
            if _new_fleet_name.strip():
                try:
                    db.create_fleet(_org_id_sf, _sel_site_id, _new_fleet_name, caller_role=st.session_state.get("auth_role"))
                except db.InsufficientRoleError:
                    st.error("Not added: creating fleets is restricted to the admin role.")
                else:
                    st.rerun()
            else:
                st.error("Fleet name is required.")

        _sf_fleets = db.list_fleets(_org_id_sf, site_id=_sel_site_id)
        if _sf_fleets:
            _fleet_options = {f["name"]: f["id"] for f in _sf_fleets}
            _sel_fleet_name = _sf_col4.selectbox("Fleet", list(_fleet_options.keys()), key="sf_fleet_select")
            _sel_fleet_id   = _fleet_options[_sel_fleet_name]

            _new_pack_name = st.text_input("New pack name", key="new_pack_name", placeholder="Pack 1")
            if st.button("Add pack", key="add_pack_btn"):
                if _new_pack_name.strip():
                    try:
                        db.create_pack(_org_id_sf, _sel_fleet_id, _new_pack_name, caller_role=st.session_state.get("auth_role"))
                    except db.InsufficientRoleError:
                        st.error("Not added: creating packs is restricted to the admin role.")
                    else:
                        st.rerun()
                else:
                    st.error("Pack name is required.")

            _sf_packs = db.list_packs(_org_id_sf, fleet_id=_sel_fleet_id)
            for _pack in _sf_packs:
                _pack_cells = db.list_pack_cells(_org_id_sf, _pack["id"])
                with st.expander(f"📦 {_pack['name']} ({len(_pack_cells)} cell(s))"):
                    if _pack_cells:
                        st.write(", ".join(_pack_cells))
                    else:
                        st.caption("No cells assigned yet.")
                    _all_cell_ids = sorted(featured_dfs.keys())
                    _assignable = [c for c in _all_cell_ids if c not in _pack_cells]
                    if _assignable:
                        _sf_pick_cell = st.selectbox(
                            "Assign a cell", _assignable, key=f"sf_assign_cell_{_pack['id']}",
                        )
                        if st.button("Assign", key=f"sf_assign_btn_{_pack['id']}"):
                            try:
                                db.add_cell_to_pack(_org_id_sf, _pack["id"], _sf_pick_cell, position=len(_pack_cells),
                                                     caller_role=st.session_state.get("auth_role"))
                            except db.InsufficientRoleError:
                                st.error("Not assigned: managing pack cells is restricted to the admin role.")
                            else:
                                st.rerun()
        else:
            st.caption("No fleets in this site yet — add one above.")
    else:
        _empty_state(
            "No sites yet",
            "Add a site above to start building this organization's fleet hierarchy.",
            icon="🏭",
        )


def render_ai_copilot_key() -> None:
    _section("AI Copilot — Language Model")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "<strong style='color:#e2e8f0'>The Copilot explains decisions in plain language — it does not make "
        "decisions.</strong> The recommendation itself always comes from the deterministic model pipeline "
        "(leave-cell-out validated GBRT + the recommendation engine). Topic-button questions always use "
        "<strong style='color:#e2e8f0'>Claude Haiku</strong> to rephrase a fixed template, instructed to "
        "never add a fact or number the template doesn't already contain. When an Anthropic API key is set, "
        "typed free-text questions instead use <strong style='color:#e2e8f0'>Claude Sonnet 5 with real "
        "tool-calling</strong> — the model decides which of a handful of data-fetching tools to call (across "
        "multiple cells for compositional questions), but every tool returns only values already computed "
        "by the pipeline, never a number the model invents itself. Without a key, both paths show the "
        "underlying template text as-is."
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
            st.success("Claude Haiku + Sonnet 5 active — topic buttons use Haiku narration, typed questions use real tool-calling.")
        else:
            st.warning("Key doesn't look like an Anthropic key (expected sk-ant-...). Check and re-enter.")
    else:
        st.caption("Without an API key the Copilot uses template answers grounded on bundle values.")


def render_onboarding_replay() -> None:
    _section("Onboarding")
    if st.button("↺ Replay guided tour", key="settings_replay_tour"):
        st.session_state["tour_seen"] = False
        st.session_state["tour_step"] = 0
        st.rerun()


def render_production_roadmap() -> None:
    _section("Production Readiness Roadmap")
    st.caption("This platform's primary direction is a Research Platform for public battery-cycling data (see docs/product_direction.md) — the Streamlit app you're using is a real demo built on that library, run with intentional constraints, not a production fleet-operations deployment. The table below documents the credible path to production deployment for fleet-operator use cases.")

    roadmap_rows = [
        ("Authentication", "Real DB-backed auth (User/Organization models, bcrypt-hashed passwords) — self-service org signup, admin-invites-teammate. Session cookies/JWT are demo-grade", "OAuth2 via Okta/LDAP for enterprise SSO; rotate the demo-grade JWT secret via a real secrets manager"),
        ("Multi-tenancy", "Real per-org data isolation — every DB table scoped by org_id, uploaded fleets persisted per-org to disk. Shared reference-cell data is intentionally global, identical for every org", "PostgreSQL row-level security as an additional isolation layer; per-org rate limiting"),
        ("Upload persistence", "Decision logs, cohort tags, settings, trajectory signatures, and uploaded cell data/model bundles all persist per-org (SQLite + per-org joblib bundle) across restarts", "PostgreSQL in cloud for concurrent multi-user access at scale"),
        ("Real BMS integration", "Victron VRM and Orion Jr2 adapters target the real public/documented API shapes — untested against a live account, credential UI in Settings", "User supplies real VRM/Orion credentials to validate end-to-end"),
        ("Second-life marketplace", "Circunomics adapter targets the generic REST shape a B2B marketplace API would expose — untested against a live partner account, credential UI in Settings", "User supplies a real Circunomics partner API key to validate end-to-end and adjust the request shape to match their actual docs"),
        ("Solar + Storage Sizing", "PVGIS (EU Commission public solar API, no key required) provides real hourly PV yield. Default weather source is 'typical year' (PVGIS's TMY irradiance pattern shapes real PVcalc monthly totals — TMY has no PV-power calculation of its own, confirmed live), falling back automatically to a single-reference-year mode (year 2013, verified live valid across PVGIS's European/Americas/rest-of-world radiation databases — the previous 2019 default only worked for Europe) if TMY is unavailable; simulate_hourly_dispatch() runs a real 8760-hour/year dispatch simulation with mode-aware (charge/discharge) temperature power derating cited to Battery University/Cadex's BU-410 industry reference — a true LP/MILP optimizer was considered and explicitly declined (different order of engineering investment than this tool's honest-estimate premise warrants). Timezone is user-overridable (longitude-estimate default; a TMY 'irradiance_time_offset' field was investigated and found to be an unrelated sub-hour solar-time correction, not a usable timezone offset); supports weekday/weekend load shapes (a holiday calendar was considered and explicitly declined — new dependency for marginal gain), dated country-preset PV install costs, an NPV-vs-payback scatter chart plus full candidate-grid results table, and a real hourly/smart-meter consumption CSV upload (bare column or timestamp+kWh with or without a header, sub-hourly resampled) — see src/deployment_sizing.py's module docstring", "Install cost assumptions (PV €/kWp localized by country; BESS €/kWh remains EU-wide — no reliable per-country breakdown found) are cited market averages, not vendor quotes, and will drift out of date — replace with current quotes for any production use; a true LP/MILP dispatcher remains the main precision gap if ever needed beyond a sizing estimate"),
        ("REST API", "8 endpoints, Swagger UI, Dockerfile, JWT bearer-token auth gating every data endpoint — JWT_SECRET is now required outside tests (src/api.py refuses to start without it, rather than silently falling back to a public default), safe to enforce since no live deployment depends on this API layer", "Set JWT_SECRET before running src/api.py outside tests; deploy via Dockerfile.api to Cloud Run / ECS"),
        ("Credential storage", "Credential-shaped Settings (VRM/Orion/Circunomics/CMMS API keys, webhook secret) are envelope-encrypted at rest — SETTINGS_ENCRYPTION_KEY is not currently set on this deployment, so the fallback key (public in this repo's source) is in use; unlike JWT_SECRET this can't hard-fail without risking locking you out of this live app, so it logs a warning and shows the banner above instead", "Set SETTINGS_ENCRYPTION_KEY via Streamlit Cloud secrets; rotate it with a re-encryption migration if ever changed"),
        ("React frontend scope", "Deliberately frozen proof-of-concept (3 views, no routing/state library) demonstrating the REST API works end to end — Streamlit remains the primary product, not developed toward feature parity", "If ever commissioned as a real frontend, treat as a resourced initiative from scratch, not a continuation of these 3 views"),
        ("Scheduled reports", "UI configuration only (no-op in demo)", "Wire to APScheduler / Celery worker; SMTP for delivery; S3 for PDF storage"),
        ("MQTT production", "Connects to public test.mosquitto.org broker", "Point MQTT_HOST / MQTT_PORT at BMS broker; add TLS + broker auth credentials"),
        ("RBAC", "Role-aware navigation and onboarding (Engineer / Fleet Manager / Executive / Compliance Officer), no server-side write-action gating yet", "Enforce role-based write permissions on Decision/CMMS-ticket actions server-side, not just UI defaults"),
        ("Audit logging", "Page views logged to local CSV", "Forward to structured log store (Datadog / CloudWatch); immutable audit trail per EU 2023/1542"),
        ("Model retraining", "Blocking call on server start", "Background worker (Celery / APScheduler); re-train nightly on new uploads, push updated bundle"),
        ("Scalability", "All cells loaded into RAM as full DataFrames — realistic ceiling is the low thousands of cells, not a real fleet's tens of thousands+; a real data-layer migration, not started", "Per-cell lazy load from Parquet/SQLite; summary metrics pre-computed and cached"),
        ("Secrets management", "Anthropic API key in .streamlit/secrets.toml", "AWS Secrets Manager / GCP Secret Manager; never in source"),
        ("CI/CD", "GitHub Actions: syntax check + pytest suite on every push", "Add Docker build → deploy to Cloud Run or ECS"),
    ]

    for gap, demo, prod in roadmap_rows:
        st.markdown(
            f"<div style='padding:12px 0;border-bottom:1px solid #2d3748'>"
            f"<div style='font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:6px'>{gap}</div>"
            f"<div style='display:flex;gap:16px'>"
            f"<div style='flex:1'><div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px'>Demo behaviour</div>"
            f"<div style='font-size:12px;color:#8896a8;line-height:1.5'>{demo}</div></div>"
            f"<div style='flex:1'><div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px'>Production path</div>"
            f"<div style='font-size:12px;color:#8896a8;line-height:1.5'>{prod}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.caption("Estimated effort to reach internal-fleet MVP: 3–4 sprints (auth + persistence + MQTT production broker + Docker deployment pipeline).")


def render_about() -> None:
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
            f"<div style='min-width:64px;font-size:11px;font-weight:600;color:#a0aec0;padding-top:2px'>{ph}</div>"
            f"<div style='min-width:120px;font-size:13px;font-weight:600;color:#e2e8f0'>{name}</div>"
            f"<div style='flex:1;font-size:12px;color:#8896a8;line-height:1.5'>{desc}</div>"
            f"<div style='min-width:48px;font-size:12px;font-weight:600;color:{status_c};text-align:right'>{status}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#a0aec0;line-height:1.8;padding:14px 18px;"
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


# ---------------------------------------------------------------------------
# Orchestrator — called once from page_settings() after Model Transparency
# ---------------------------------------------------------------------------

def render_settings_configuration(featured_dfs: dict, bundles: dict) -> None:
    render_encryption_key_warning()
    render_alert_thresholds()
    render_rul_reliability_threshold(bundles)
    render_crm_configuration()
    # Org-wide configuration (integration credentials, alert/threshold
    # config) and the platform-wide model-cache clear -- restricted to
    # admin in the UI (matching Team Members/Sites & Fleets' existing
    # pattern below) *and* enforced underneath in src/db.py itself (see
    # db.InsufficientRoleError's docstring for why hiding these widgets
    # alone was never a real security boundary). Hiding them here as well
    # avoids a confusing "you can see the field but saving it always
    # errors" experience for non-admin roles.
    if st.session_state.get("auth_role") == "admin":
        render_application_profile_and_eol()
        render_model_cache()
        render_cost_of_delay()
        render_webhook_notifications(featured_dfs)
        render_webhook_subscriptions()
        render_bms_connector_victron()
        render_bms_connector_orion()
        render_bms_connector_modbus()
        render_bms_connector_can()
        render_bms_connector_ocpp()
        render_adapter_settings("generic_rest_bms")
        render_second_life_marketplace()
        render_maintenance_writeback()
    render_team_members()
    render_sites_and_fleets(featured_dfs)
    render_ai_copilot_key()
    render_onboarding_replay()
    render_production_roadmap()
    render_about()
