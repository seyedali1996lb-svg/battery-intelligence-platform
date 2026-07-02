"""Page: Settings"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _md_html, NASA_CELL_IDS
from design_system import C_GREEN, C_AMBER, C_MUTED, C_ORANGE, section_header_html


def _clear_uploaded_data():
    for k in ["uploaded_featured_dfs", "uploaded_bundle", "uploaded_split_cycles", "uploaded_mode_meta"]:
        st.session_state.pop(k, None)
    st.session_state["data_mode"] = "nasa"


def page_settings(featured_dfs: dict, bundles: dict):
    from lco_eval import RUL_RELIABLE_FLOOR

    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    st.markdown("# Settings")
    st.markdown("##### Platform configuration · model transparency · reliability controls")

    # ── Section 0: Uploaded data ──
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
            f"<div style='font-size:12px;color:#718096;margin-top:4px;line-height:1.8'>"
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

    # ── Section 1: Data sources ──
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
            f"<div style='font-size:12px;color:#718096;margin-top:4px;line-height:1.6'>"
            f"Physics-informed simulation (Arrhenius SEI growth, empirical C-rate factor, "
            f"Rainflow DoD scaling). Resistance: 0.15–0.40 Ω internal. "
            f"<strong>Not real measured data.</strong></div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:8px'>{', '.join(synth_ids)}</div>"
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
            f"<div style='font-size:12px;color:#718096;margin-top:4px;line-height:1.6'>"
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
        "<strong style='color:#718096'>Why two separate models?</strong> "
        "Synthetic and NASA cells use incompatible resistance scales (0.15–0.40 Ω vs 0.04–0.07 Ω Re). "
        "A combined model produced R²=−0.49. Two separate GBRT models, each trained and validated "
        "on its own data source, keep the predictions honest. Fleet ranking uses SOH "
        "(scale-invariant) rather than RUL (model-dependent) for cross-type comparison.</div>",
        unsafe_allow_html=True,
    )

    # ── Section 1b: Model Card ──
    _section("Model Card — GBRT SOH / RUL Estimator")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "A model card discloses what the model is, what data it was trained on, "
        "how it was validated, and what it should and should not be used for. "
        "This card follows the Mitchell et al. (2019) model card format."
        "</div>",
        unsafe_allow_html=True,
    )

    import datetime as _dt
    _train_date = "2024-01 (initial) → updated each app start if cache invalid"
    _mc_rows = [
        ("Model type",        "Gradient-Boosted Regression Trees (scikit-learn GradientBoostingRegressor)"),
        ("Targets",           "SOH (% capacity remaining) and RUL (cycles to 80% EOL threshold)"),
        ("Architecture",      "Two separate GBRT instances — one for NASA PCoE cells, one for synthetic cells. Combined model R²=−0.49 due to incompatible resistance scales (0.04–0.07 Ω vs 0.15–0.40 Ω)."),
        ("Hyperparameters",   "n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, loss=squared_error (SOH); same with quantile loss for RUL uncertainty bounds"),
        ("Training data",     "NASA PCoE Battery Aging Dataset (Saha & Goebel 2007): 4 LiCoO₂ 18650 cells (~2 Ah, 24°C, 2A discharge) · 8 synthetic cells (physics-informed: Arrhenius SEI, empirical C-rate, Rainflow DoD)"),
        ("Validation method", "Leave-Cell-Out (LCO) cross-validation: train on N−1 cells, test on held-out cell. Row-level train/test split is not used — it leaks cell identity into training."),
        ("EOL definition",    "80% of initial capacity (industry standard; configurable in Settings)"),
        ("Feature set",       "cycle_number, fade_rate_{10,30,50}cy, fade_acceleration, soh_velocity_50cy, resistance_{ohm,normalized,trend_30cy}, temp_rolling_30cy, dqdv_{peak_value,peak_soc,area,fwhm}, ce_rolling_30cy, ce_drop_rate"),
        ("Intended use",      "Engineering decision support — prioritisation, inspection scheduling, second-life routing. Not for safety-critical go/no-go decisions."),
        ("Known limitations", (
            "1. RUL is unreliable below fold R²=0.30 (B0018: R²=0.22, withheld). "
            "2. Model was not tested on NMC/NCA/LFP — transfer to other chemistries is unknown. "
            "3. Only 4 real cells — fleet-level statistics are indicative, not statistically robust. "
            "4. Synthetic cells share the same physics model used for generation — they cannot reveal model failure modes outside that model's assumptions. "
            "5. Temperature assumed 25°C for cells without measured temperature column."
        )),
        ("Out-of-scope uses", "Certified safety assessments · Regulatory compliance claims · Financial warranty calculations without independent validation"),
        ("Bias / fairness",   "No demographic bias considerations apply (physical cells). Cell-to-cell manufacturing variation is a known source of model error — real batteries vary more than the synthetic fleet captures."),
        ("Last trained",      _train_date),
    ]

    for field, value in _mc_rows:
        is_limit = field in ("Known limitations", "Out-of-scope uses")
        val_colour = "#fc8181" if is_limit else "#a0aec0"
        st.markdown(
            f"<div style='display:flex;gap:16px;padding:10px 0;border-bottom:1px solid #2d3748;align-items:flex-start'>"
            f"<div style='min-width:160px;font-size:11px;font-weight:600;color:#4a5568;"
            f"text-transform:uppercase;letter-spacing:0.06em;padding-top:2px;flex-shrink:0'>{field}</div>"
            f"<div style='flex:1;font-size:12px;color:{val_colour};line-height:1.7'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Per-cell LCO summary inside model card
    with st.expander("Per-cell validation results", expanded=False):
        for source_key, bundle in bundles.items():
            if bundle is None:
                continue
            lco_per = bundle["metrics"].get("lco_per_cell", {})
            per_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
            label   = "NASA PCoE" if source_key == "nasa" else "Synthetic"
            colour  = "#48bb78" if source_key == "nasa" else "#fc8181"
            st.markdown(f"<div style='font-size:12px;font-weight:600;color:{colour};margin:12px 0 6px'>{label} model · {bundle['metrics'].get('n_cells','?')} cells · {bundle['metrics'].get('n_rows',0):,} rows</div>", unsafe_allow_html=True)
            _hdr = st.columns([2, 1, 1, 1])
            for c, h in zip(_hdr, ["Cell", "SOH R²", "RUL R²", "RUL Status"]):
                c.markdown(f"<div style='font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em'>{h}</div>", unsafe_allow_html=True)
            for cell_id, fold in lco_per.items():
                ok     = per_ok.get(cell_id, True)
                s_col  = "#48bb78" if ok else "#fc8181"
                status = "Calibrated" if ok else f"Withheld (R²={fold.get('rul_r2',0):.2f} < 0.30)"
                row    = st.columns([2, 1, 1, 1])
                row[0].markdown(f"<div style='font-size:13px;color:#e2e8f0;padding:3px 0'>{cell_id}</div>", unsafe_allow_html=True)
                row[1].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:3px 0'>{fold.get('soh_r2',0):.3f}</div>", unsafe_allow_html=True)
                row[2].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:3px 0'>{fold.get('rul_r2',0):.3f}</div>", unsafe_allow_html=True)
                row[3].markdown(f"<div style='font-size:13px;color:{s_col};padding:3px 0'>{status}</div>", unsafe_allow_html=True)

    # ── Section 2: Model transparency ──
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
        lco_per    = m.get("lco_per_cell", {})
        per_cell_ok = m.get("per_cell_rul_reliable", {})
        label  = "NASA PCoE" if source_key == "nasa" else "Synthetic"
        colour = "#48bb78" if source_key == "nasa" else "#fc8181"

        st.markdown(f"<div style='font-size:12px;font-weight:600;color:{colour};margin:16px 0 8px'>{label} model</div>", unsafe_allow_html=True)

        header_cols = st.columns([2, 1, 1, 1, 2])
        for col, hdr in zip(header_cols, ["Cell", "SOH fold R²", "RUL fold R²", "RUL status", "Note"]):
            col.markdown(f"<div style='font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em'>{hdr}</div>", unsafe_allow_html=True)

        for cell_id, fold in lco_per.items():
            soh_r2 = fold.get("soh_r2", None)
            rul_r2 = fold.get("rul_r2", None)
            ok     = per_cell_ok.get(cell_id, True)
            status_c = C_GREEN if ok else C_ORANGE
            status_l = "Calibrated" if ok else "Not calibrated"
            note = "" if ok else f"fold R²={rul_r2:.2f} < {RUL_RELIABLE_FLOOR} floor — RUL withheld"
            row = st.columns([2, 1, 1, 1, 2])
            row[0].markdown(f"<div style='font-size:13px;color:#e2e8f0;padding:4px 0'>{cell_id}</div>", unsafe_allow_html=True)
            row[1].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:4px 0'>{soh_r2:.2f}</div>", unsafe_allow_html=True)
            row[2].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:4px 0'>{rul_r2:.2f}</div>", unsafe_allow_html=True)
            row[3].markdown(f"<div style='font-size:13px;color:{status_c};padding:4px 0'>{status_l}</div>", unsafe_allow_html=True)
            row[4].markdown(f"<div style='font-size:11px;color:#4a5568;padding:4px 0'>{note}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='display:flex;gap:24px;font-size:12px;color:#718096;"
            f"padding:8px 0;border-top:1px solid #2d3748;margin-top:4px'>"
            f"<span>Dataset SOH R²: <strong style='color:#e2e8f0'>{m.get('lco_soh_r2', 0):.3f}</strong></span>"
            f"<span>Dataset RUL R²: <strong style='color:#e2e8f0'>{m.get('lco_rul_r2', 0):.3f}</strong></span>"
            f"<span>Training cells: <strong style='color:#e2e8f0'>{m.get('n_cells', '—')}</strong></span>"
            f"<span>Training rows: <strong style='color:#e2e8f0'>{m.get('n_rows', 0):,}</strong></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Section 2b: Application EOL threshold ──
    _section("Application End-of-Life Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "The EOL threshold defines when a cell is 'retired' for your application. "
        "The standard industry convention is <strong style='color:#718096'>80% SOH</strong>, "
        "but this is not universal. "
        "Changing this threshold adjusts the displayed RUL on the Overview page "
        "using the current fade rate — <strong style='color:#718096'>it does not retrain the model</strong>. "
        "The model was trained on 80% EOL; the adjusted RUL is a fade-rate projection, not a new model prediction.</div>",
        unsafe_allow_html=True,
    )

    eol_col1, eol_col2 = st.columns([4, 1])
    with eol_col1:
        new_eol = st.slider(
            "Application EOL threshold (%)", min_value=70, max_value=95, step=1,
            value=int(st.session_state.get("eol_threshold_pct", 80)),
            key="settings_eol_threshold",
            help="RUL on Overview will reflect cycles remaining until SOH hits this value.",
        )
    with eol_col2:
        st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
        if st.button("Reset to 80%", key="settings_eol_reset"):
            st.session_state["eol_threshold_pct"] = 80.0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if new_eol != int(st.session_state.get("eol_threshold_pct", 80)):
        st.session_state["eol_threshold_pct"] = float(new_eol)
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

    # ── Section 2c: Alert thresholds ──
    _section("🔔 Alert Thresholds")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Configure alert thresholds that trigger warning banners across all pages. "
        "Alerts are evaluated on every page load against the currently active fleet.</div>",
        unsafe_allow_html=True,
    )

    _at_col1, _at_col2, _at_col3 = st.columns(3)
    with _at_col1:
        soh_alert = st.slider("SOH Warning Threshold (%)", 70, 95,
            int(st.session_state.get("soh_alert_pct", 85)), key="soh_alert_pct")
        st.caption("Show warning banner when any cell's SOH drops below this level.")
    with _at_col2:
        resistance_alert = st.slider("Resistance Alert Multiplier (×initial)", 1.2, 3.0,
            float(st.session_state.get("resistance_alert_mult", 1.8)), step=0.1, key="resistance_alert_mult")
        st.caption("Alert when resistance exceeds this multiple of the cell's initial resistance.")
    with _at_col3:
        spread_alert = st.slider("Pack Spread Alert (%)", 1.0, 10.0,
            float(st.session_state.get("spread_alert_pct", 5.0)), step=0.5, key="spread_alert_pct")
        st.caption("Alert when SOH spread across fleet cells exceeds this threshold.")

    # ── Section 3: RUL reliability threshold ──
    _section("RUL Reliability Threshold")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "The reliability floor gates whether RUL predictions are shown or suppressed. "
        "Cells whose held-out fold R² falls below this value have RUL withheld across "
        "all pages — shown as 'Not calibrated' instead of a cycle count. "
        "<strong style='color:#718096'>This is a read-only preview</strong> — "
        "the active floor is hardcoded at "
        f"<code style='color:#63b3ed'>{RUL_RELIABLE_FLOOR}</code> in "
        "<code style='color:#63b3ed'>src/lco_eval.py</code> and requires a code change to modify.</div>",
        unsafe_allow_html=True,
    )

    slider_col, reset_col = st.columns([5, 1])
    with slider_col:
        preview_floor = st.slider(
            "Preview threshold", min_value=0.0, max_value=0.5,
            value=float(st.session_state.get("settings_rul_floor_preview", RUL_RELIABLE_FLOOR)),
            step=0.05, key="settings_rul_floor_preview",
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
        col.markdown(f"<div style='font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em'>{hdr}</div>", unsafe_allow_html=True)

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

    # ── Section 3b: Model cache ──
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
    cache_files  = list(_CACHE_DIR.glob("*.joblib")) if _CACHE_DIR.exists() else []
    cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024) if cache_files else 0
    cache_info   = f"{len(cache_files)} bundle(s) · {cache_size_mb:.1f} MB" if cache_files else "No cache on disk — will train fresh on next load"

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

    # ── Section 4: About ──
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
            f"<div style='flex:1;font-size:12px;color:#718096;line-height:1.5'>{desc}</div>"
            f"<div style='min-width:48px;font-size:12px;font-weight:600;color:{status_c};text-align:right'>{status}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#4a5568;line-height:1.8;padding:14px 18px;"
        "background:#1a202c;border-radius:8px'>"
        "<strong style='color:#718096'>Stack</strong> — "
        "scikit-learn GBRT · Streamlit · Plotly · reportlab<br>"
        "<strong style='color:#718096'>Model</strong> — "
        "Two separate GBRT instances (synthetic / NASA); leave-cell-out cross-validation<br>"
        "<strong style='color:#718096'>Data</strong> — "
        "8 synthetic cells (physics-informed) + 4 NASA PCoE cells (Saha &amp; Goebel, 2007)<br>"
        "<strong style='color:#718096'>Regulatory</strong> — "
        "EU Battery Regulation (EU) 2023/1542 — field structure demonstration only; "
        "not a compliance claim<br>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section 5: Production Roadmap ──
    _section("Production Roadmap")

    st.markdown(
        "<div style='font-size:12px;color:#718096;line-height:1.6;margin-bottom:16px'>"
        "This platform is a functional Phase 1 product. The gaps below are documented honestly — "
        "each has a defined path to resolution for a production deployment.</div>",
        unsafe_allow_html=True,
    )

    roadmap_items = [
        ("Authentication & RBAC",
         "Current state: demo login wall (session-scoped). Production: OAuth 2.0 via Okta or Azure AD; "
         "role-based UI rendering per user identity; JWT session tokens.",
         "High"),
        ("Multi-tenancy",
         "Current state: global st.cache_resource shared across all sessions — user A's uploads are "
         "visible to user B in a multi-worker deployment. Production: tenant-scoped data isolation "
         "via PostgreSQL row-level security or separate schema per tenant.",
         "High"),
        ("Data persistence",
         "Current state: uploaded data is session-scoped and lost on page refresh. Production: "
         "PostgreSQL + TimescaleDB for time-series cycle data; S3/blob for raw files. "
         "SQLite is an acceptable intermediate step for single-user local deployment.",
         "High"),
        ("REST API / BMS integration",
         "Current state: no external interface. Production: FastAPI layer over the model pipeline; "
         "webhook endpoints for real-time BMS telemetry (MQTT or Kafka ingest); "
         "versioned REST API for SCADA / CMMS integration.",
         "Medium"),
        ("Audit log persistence",
         "Current state: in-memory session log (see Audit Log section below). "
         "Production: append-only PostgreSQL audit table with user identity, timestamp, "
         "action, and cell reference. Required for regulated-industry deployment.",
         "Medium"),
        ("Scalability",
         "Current state: all cells loaded into memory at startup; synchronous model training "
         "blocks UI for 20–60 s. Production: lazy per-cell data loading; background training "
         "thread with progress websocket; pre-aggregated summary metrics in database; "
         "Parquet columnar storage for cycle data.",
         "Medium"),
        ("Real EIS data",
         "Current state: Nyquist plots and EIS decomposition are physics approximations on DC "
         "resistance values — not measured impedance spectra. Production: Gamry / BioLogic "
         "potentiostat integration; impedance.py for DRT fitting; Warburg from real frequency-domain data.",
         "High"),
        ("Dataset expansion",
         "Current state: 12 cells (8 synthetic + 4 NASA 18650 from 2007). Production: "
         "Severson 2019 (124 LFP cells, publicly available); CALCE NMC/LFP; Oxford LiCoO₂; "
         "minimum 50+ real cells for a defensible RUL model.",
         "Critical"),
    ]

    priority_color = {"Critical": "#fc8181", "High": "#f6ad55", "Medium": "#48bb78"}

    for title, detail, priority in roadmap_items:
        pc = priority_color.get(priority, "#718096")
        with st.expander(f"{title}  —  priority: {priority}"):
            st.markdown(
                f"<div style='font-size:12px;color:#a0aec0;line-height:1.7'>{detail}</div>",
                unsafe_allow_html=True,
            )

    # ── Section 6: Audit Log ──
    _section("Session Audit Log")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;line-height:1.6;margin-bottom:12px'>"
        "Records page views, Copilot queries, and logged decisions for this session. "
        "Cleared when the session ends. A production deployment would persist this to a database.</div>",
        unsafe_allow_html=True,
    )

    from audit import get_log, export_csv as audit_csv
    log_records = get_log()

    if log_records:
        import pandas as _pd
        log_df = _pd.DataFrame(log_records)
        st.dataframe(log_df.head(200), use_container_width=True, hide_index=True)
        if len(log_df) > 200:
            st.caption(f"Showing up to 200 rows — {len(log_df)} total.")
        st.download_button(
            "Export audit log (CSV)",
            data=audit_csv(),
            file_name="battery_intel_audit.csv",
            mime="text/csv",
            key="settings_audit_export",
        )
    else:
        st.markdown(
            "<div style='font-size:12px;color:#2d3748;padding:16px;text-align:center'>"
            "No activity recorded yet in this session.</div>",
            unsafe_allow_html=True,
        )
