"""Page: Import"""

import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _md_html, LEGEND_H, base_layout
from design_system import section_header_html
from features import build_features, get_model_matrix
from model import train_models, predict
from lco_eval import run_lco, RUL_RELIABLE_FLOOR
from import_adapter import adapt_upload_to_pipeline


def _clear_uploaded_data():
    for k in ["uploaded_featured_dfs", "uploaded_bundle", "uploaded_split_cycles", "uploaded_mode_meta"]:
        st.session_state.pop(k, None)
    st.session_state["data_mode"] = "nasa"


def _run_analysis_button(df_raw: pd.DataFrame, summary: dict):
    import datetime

    step_labels = [
        ("parse",       "Parsing uploaded data"),
        ("features",    "Engineering features"),
        ("soh",         "Training SOH model"),
        ("rul",         "Training RUL model"),
        ("lco",         "Running leave-cell-out validation"),
        ("reliability", "Computing per-cell reliability"),
        ("load",        "Loading results into dashboard"),
    ]

    if st.button("⚡ Analyse this data", type="primary", use_container_width=True, key="import_run_analysis"):
        st.markdown("<div style='font-size:11px;color:#718096;margin-bottom:12px'>~60–90 seconds</div>", unsafe_allow_html=True)
        slots = {k: st.empty() for k, _ in step_labels}

        def _step(key: str, icon: str, text: str):
            c = "#48bb78" if icon == "✓" else "#f6ad55" if icon == "⚠" else "#63b3ed"
            slots[key].markdown(f"<div style='font-size:13px;color:{c};padding:3px 0'>{icon} {text}</div>", unsafe_allow_html=True)

        for key, label in step_labels:
            _step(key, "☐", label)

        try:
            _step("parse", "⏳", "Parsing uploaded data…")
            battery  = adapt_upload_to_pipeline(df_raw)
            n_up     = len(battery["cells"])
            total_cy = sum(len(c["cycles"]) for c in battery["cells"].values())
            _step("parse", "✓", f"Parsed — {n_up} cells, {total_cy:,} cycles")

            _step("features", "⏳", "Engineering features…")
            all_X, all_y_soh, all_y_rul = [], [], []
            cell_featured = {}
            for cid, cell in battery["cells"].items():
                df_feat = build_features(cell["cycles"])
                X, y_soh, y_rul = get_model_matrix(df_feat)
                all_X.append(X); all_y_soh.append(y_soh); all_y_rul.append(y_rul)
                cell_featured[cid] = (df_feat, X)
            X_all     = pd.concat(all_X)
            y_soh_all = pd.concat(all_y_soh)
            y_rul_all = pd.concat(all_y_rul)
            _step("features", "✓", f"Features built — {len(X_all):,} rows")

            _step("soh", "⏳", "Training SOH model…")
            _step("rul", "⏳", "Training RUL model…")
            up_bndl = train_models(X_all, y_soh_all, y_rul_all)
            up_bndl["metrics"]["n_cells"] = n_up
            up_bndl["metrics"]["n_rows"]  = len(X_all)
            _step("soh", "✓", "SOH model trained")
            _step("rul", "✓", "RUL model trained")

            _step("lco", "⏳", "Running leave-cell-out validation…")
            cell_cycles = {cid: cell["cycles"] for cid, cell in battery["cells"].items()}
            lco = run_lco(cell_cycles)
            up_bndl["metrics"]["lco_soh_r2"]  = lco["soh_r2"]
            up_bndl["metrics"]["lco_rul_r2"]  = lco["rul_r2"]
            up_bndl["metrics"]["rul_reliable"] = lco["rul_reliable"]
            up_bndl["metrics"]["lco_per_cell"] = lco["per_cell"]
            _step("lco", "✓", f"LCO complete — SOH R²={lco['soh_r2']:.2f}  RUL R²={lco['rul_r2']:.2f}")

            _step("reliability", "⏳", "Computing per-cell reliability…")
            per_cell_ok = {
                cid: (fold["rul_r2"] >= RUL_RELIABLE_FLOOR)
                for cid, fold in lco["per_cell"].items()
            }
            up_bndl["metrics"]["per_cell_rul_reliable"] = per_cell_ok
            lco_limited     = n_up < 3
            calibrating_cnt = sum(1 for ok in per_cell_ok.values() if not ok)
            up_bndl["metrics"]["lco_limited"] = lco_limited
            _step("reliability", "✓", "Per-cell reliability computed"
                  + (" — ⚠ LCO limited (< 3 cells)" if lco_limited else ""))

            _step("load", "⏳", "Loading results into dashboard…")
            up_fdfs, up_sc = {}, {}
            for cid, (df_feat, X) in cell_featured.items():
                preds  = predict(up_bndl, X)
                df_out = df_feat.loc[X.index].copy()
                df_out["soh_pred"]       = preds["soh_pred"]
                df_out["rul_pred"]       = preds["rul_pred"]
                df_out["confidence_tag"] = preds["confidence_tag"]
                up_fdfs[cid] = df_out
                split_idx    = int(len(X) * 0.8)
                up_sc[cid]   = int(X["cycle_number"].iloc[split_idx])

            st.session_state["uploaded_featured_dfs"] = up_fdfs
            st.session_state["uploaded_bundle"]       = up_bndl
            st.session_state["uploaded_split_cycles"] = up_sc
            st.session_state["uploaded_mode_meta"]    = {
                "n_cells":                   n_up,
                "cell_ids":                  list(up_fdfs.keys()),
                "upload_date":               datetime.date.today().isoformat(),
                "calibrating_count":         calibrating_cnt,
                "lco_limited":               lco_limited,
                "temperature_assumed_cells": battery["temperature_assumed_cells"],
            }
            st.session_state["data_mode"] = "uploaded"
            _step("load", "✓", "Done — results loaded into all pages")
            st.rerun()

        except Exception as exc:
            import traceback
            st.error(f"Pipeline error: {exc}")
            st.code(traceback.format_exc())


def _show_upload_summary():
    meta = st.session_state["uploaded_mode_meta"]
    n    = meta["n_cells"]
    k    = meta["calibrating_count"]
    lco  = st.session_state["uploaded_bundle"]["metrics"]
    lco_lim_note = (
        "\n⚠ LCO run on fewer than 3 cells — reliability estimates are less stable than usual."
        if meta.get("lco_limited") else ""
    )
    rul_reliable_count = n - k
    st.markdown(
        f"<div style='background:rgba(47,133,90,0.10);border:1px solid rgba(47,133,90,0.35);"
        f"border-radius:10px;padding:20px 24px;margin-bottom:16px'>"
        f"<div style='font-size:16px;font-weight:700;color:#48bb78;margin-bottom:12px'>✓ Analysis complete</div>"
        f"<div style='font-size:13px;color:#a0aec0;line-height:2'>"
        f"<strong style='color:#e2e8f0'>{n}</strong> cells loaded<br>"
        f"SOH model R²: <strong style='color:#e2e8f0'>{lco.get('lco_soh_r2', 0):.2f}</strong> "
        f"<span style='color:#4a5568'>(leave-cell-out)</span><br>"
        f"RUL reliable: <strong style='color:#e2e8f0'>{rul_reliable_count} of {n}</strong> cells"
        f"</div>"
        f"{('<div style=\"font-size:12px;color:#f6ad55;margin-top:10px\">' + lco_lim_note + '</div>') if meta.get('lco_limited') else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_view, col_clear = st.columns([2, 1])
    with col_view:
        if st.button("View results →", type="primary", use_container_width=True, key="import_view_results"):
            st.session_state["page"] = "overview"
            st.rerun()
    with col_clear:
        if st.button("✕ Clear uploaded data", use_container_width=True, key="import_clear"):
            _clear_uploaded_data()
            st.rerun()


def page_import():
    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    st.markdown("# Import Your Battery Data")
    st.markdown("##### Upload cycle data from your own cells to run the full analysis pipeline on your batteries")

    h1, h2 = st.columns(2)
    template_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "import_template.csv")
    with h1:
        try:
            with open(template_path, "rb") as f:
                template_bytes = f.read()
            st.download_button(
                label="⬇ Download Template CSV",
                data=template_bytes, file_name="battery_import_template.csv",
                mime="text/csv", use_container_width=True,
            )
        except FileNotFoundError:
            st.error("Template file not found at data/import_template.csv")
    with h2:
        if st.button("📖 View Format Guide", use_container_width=True, key="import_guide_toggle"):
            st.session_state["import_guide_open"] = not st.session_state.get("import_guide_open", False)

    if st.session_state.get("import_guide_open", False):
        guide_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "import_format_guide.md")
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_text = f.read()
            with st.expander("Format Guide", expanded=True):
                st.markdown(guide_text)
        except FileNotFoundError:
            st.warning("Format guide not found at docs/import_format_guide.md")

    with st.expander("📦 External Benchmark Datasets", expanded=False):
        st.markdown("Use the loaders in `src/severson_loader.py` and `src/calce_loader.py` to integrate published benchmark datasets into the pipeline.")
        st.markdown("**Severson 2019 (Nature Energy)**")
        st.markdown("Download `batch1.pkl`, `batch2.pkl`, `batch3.pkl` from data.matr.io (Severson et al., Nature Energy 2019). Upload them below — the importer auto-detects `.pkl` files and loads all cells.")
        st.link_button("Download Severson Dataset (data.matr.io)", "https://data.matr.io/1/")
        st.markdown("**CALCE Battery Research Group**")
        st.markdown("Download CSV or XLSX files from the CALCE battery data portal. The importer handles column-name variation across CALCE sub-datasets.")
        st.link_button("Download CALCE Dataset", "https://web.calce.umd.edu/batteries/data.htm")

    _section("Upload CSV / XLSX / PKL")

    uploaded = st.file_uploader(
        "Upload battery cycle data",
        type=["csv", "xlsx", "pkl"],
        accept_multiple_files=True,
        key="import_csv_upload",
        help="CSV/XLSX: Battery Intelligence Platform format. PKL: Severson batch file.",
        label_visibility="collapsed",
    )

    _pkl_files = [f for f in (uploaded or []) if f.name.endswith(".pkl")]
    _csv_files = [f for f in (uploaded or []) if not f.name.endswith(".pkl")]

    for _pkl_file in _pkl_files:
        try:
            import tempfile
            from severson_loader import load_severson_batch
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp.write(_pkl_file.read())
                tmp_path = tmp.name
            cell_dfs = load_severson_batch(tmp_path)
            st.success(f"Loaded {len(cell_dfs)} cells from Severson batch file")
            preview_rows = [
                {"Cell ID": cid, "Cycles": len(_df),
                 "Final SOH": f"{(_df.capacity_ah.iloc[-1] / max(_df.capacity_ah.iloc[0], 1e-9) * 100):.1f}%"}
                for cid, _df in cell_dfs.items()
            ]
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            st.session_state["severson_cells"] = cell_dfs
            st.info("Severson cells loaded. Switch to Overview to analyze individual cells.")
        except Exception as _e:
            st.error(f"Could not load Severson batch file '{_pkl_file.name}': {_e}")

    uploaded = _csv_files[0] if _csv_files else None

    if uploaded is None and not _pkl_files:
        st.markdown(
            "<div style='background:#1a202c;border:1px dashed #2d3748;border-radius:10px;"
            "padding:40px;text-align:center;color:#4a5568;font-size:13px'>"
            "Drag and drop a CSV, XLSX, or PKL file here, or click to browse.<br>"
            "<span style='font-size:11px;color:#2d3748;margin-top:6px;display:block'>"
            "CSV/XLSX: 7 columns required — download the template for the exact format. "
            "PKL: Severson batch file.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if uploaded is None:
        return

    try:
        _fname = getattr(uploaded, "name", "")
        if _fname.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded)
        else:
            df_raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        return

    _calce_tried = False
    _calce_df    = None
    try:
        from calce_loader import load_calce_file
        import tempfile as _tmpmod
        _fname = getattr(uploaded, "name", "")
        if _fname.endswith((".csv", ".xlsx", ".xls")):
            uploaded.seek(0)
            with _tmpmod.NamedTemporaryFile(suffix=os.path.splitext(_fname)[1] or ".csv", delete=False) as _tmp:
                _tmp.write(uploaded.read())
                _tmp_path = _tmp.name
            try:
                _calce_df   = load_calce_file(_tmp_path)
                _calce_tried = True
            except Exception:
                _calce_df = None
            uploaded.seek(0)
    except Exception:
        pass

    from import_validator import validate_upload, fuzzy_match_columns, apply_column_mapping

    col_mapping = fuzzy_match_columns(df_raw)
    renames = {orig: canon for orig, canon in col_mapping.items() if orig != canon}
    if renames:
        rename_items = "".join(
            f"<div style='font-size:12px;color:#a0aec0;padding:4px 0;border-bottom:1px solid #2d3748'>"
            f"<span style='color:#718096'>{orig}</span>"
            f"<span style='color:#4a5568;padding:0 8px'>→</span>"
            f"<span style='color:#48bb78'>{canon}</span>"
            f"</div>"
            for orig, canon in sorted(renames.items())
        )
        st.markdown(
            f"<div style='background:rgba(104,211,145,0.07);border:1px solid rgba(104,211,145,0.3);"
            f"border-radius:10px;padding:14px 18px;margin:12px 0'>"
            f"<div style='font-size:13px;font-weight:600;color:#48bb78;margin-bottom:8px'>"
            f"✓ Auto-matched {len(renames)} column{'s' if len(renames) != 1 else ''}</div>"
            f"{rename_items}"
            f"<div style='font-size:11px;color:#4a5568;margin-top:10px'>"
            f"Column names were automatically remapped to the pipeline's expected format. "
            f"If a mapping looks wrong, rename the column in your CSV and re-upload."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        df_raw = apply_column_mapping(df_raw, col_mapping)

    result   = validate_upload(df_raw)
    errors   = result["errors"]
    warnings = result["warnings"]
    summary  = result["summary"]

    if errors and _calce_tried and _calce_df is not None and len(_calce_df) > 50:
        st.info(f"Standard format not detected — auto-detected as CALCE format ({len(_calce_df)} cycles loaded). Using CALCE loader.")
        _fname_stem = os.path.splitext(getattr(uploaded, "name", "calce_cell"))[0]
        df_raw = _calce_df.copy()
        df_raw["cell_id"] = _fname_stem
        result   = validate_upload(df_raw)
        errors   = result["errors"]
        warnings = result["warnings"]
        summary  = result["summary"]

    if errors:
        st.markdown(
            f"<div style='background:rgba(197,48,48,0.12);border:1px solid rgba(197,48,48,0.4);"
            f"border-radius:10px;padding:16px 20px;margin:16px 0'>"
            f"<div style='font-size:14px;font-weight:700;color:#fc8181;margin-bottom:10px'>"
            f"Upload cannot be processed — {len(errors)} issue{'s' if len(errors) != 1 else ''} found</div>",
            unsafe_allow_html=True,
        )
        for err in errors:
            st.markdown(
                f"<div style='display:flex;gap:10px;padding:6px 0;border-top:1px solid rgba(197,48,48,0.2)'>"
                f"<span style='color:#fc8181;font-size:14px;flex-shrink:0'>✕</span>"
                f"<span style='font-size:13px;color:#fed7d7;line-height:1.5'>{err}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px;color:#718096;margin-top:8px'>Fix these issues in your CSV and re-upload.</div>", unsafe_allow_html=True)
        return

    if warnings:
        st.markdown(
            f"<div style='background:rgba(183,121,31,0.10);border:1px solid rgba(183,121,31,0.35);"
            f"border-radius:10px;padding:16px 20px;margin:16px 0'>"
            f"<div style='font-size:14px;font-weight:700;color:#f6ad55;margin-bottom:10px'>"
            f"Upload parsed successfully — {len(warnings)} note{'s' if len(warnings) != 1 else ''} to review</div>",
            unsafe_allow_html=True,
        )
        for w in warnings:
            st.markdown(
                f"<div style='display:flex;gap:10px;padding:6px 0;border-top:1px solid rgba(183,121,31,0.2)'>"
                f"<span style='color:#f6ad55;font-size:14px;flex-shrink:0'>⚠</span>"
                f"<span style='font-size:13px;color:#fefcbf;line-height:1.5'>{w}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px;color:#718096;margin:12px 0 8px'>Acknowledge each note to continue:</div>", unsafe_allow_html=True)
        all_acked = True
        for i, w in enumerate(warnings):
            short = w.split(".")[0][:90] + ("…" if len(w.split(".")[0]) > 90 else "")
            acked = st.checkbox(f"I understand: {short}", key=f"import_ack_{i}")
            if not acked:
                all_acked = False
        if not all_acked:
            st.markdown("<div style='font-size:12px;color:#718096;margin-top:12px'>Tick all boxes above to enable the Proceed button.</div>", unsafe_allow_html=True)
            return
    else:
        st.markdown(
            "<div style='background:rgba(47,133,90,0.10);border:1px solid rgba(47,133,90,0.35);"
            "border-radius:10px;padding:14px 20px;margin:16px 0'>"
            "<span style='font-size:14px;font-weight:700;color:#48bb78'>✓ Upload validated — ready to analyse</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    _section("Data Preview")

    st.markdown("<div style='font-size:12px;color:#718096;margin-bottom:10px'>One row per cell:</div>", unsafe_allow_html=True)

    df_raw_clean = df_raw.copy()
    df_raw_clean["capacity_ah"]    = pd.to_numeric(df_raw_clean["capacity_ah"],    errors="coerce")
    df_raw_clean["resistance_ohm"] = pd.to_numeric(df_raw_clean["resistance_ohm"], errors="coerce")

    tbl_cols = st.columns([2, 1, 2, 2, 1])
    for col, hdr in zip(tbl_cols, ["Cell ID", "Cycles", "Capacity range (Ah)", "Resistance range (Ω)", "Temp"]):
        col.markdown(f"<div style='font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em;padding-bottom:6px'>{hdr}</div>", unsafe_allow_html=True)

    has_temp = "temperature_c" in df_raw_clean.columns
    for cid, n_cy in summary["cycles_per_cell"].items():
        cell_df = df_raw_clean[df_raw_clean["cell_id"].astype(str).str.strip() == cid]
        cap_min = cell_df["capacity_ah"].min()
        cap_max = cell_df["capacity_ah"].max()
        res_min = cell_df["resistance_ohm"].min()
        res_max = cell_df["resistance_ohm"].max()
        temp_ok = has_temp and cell_df["temperature_c"].notna().any() if has_temp else False
        row = st.columns([2, 1, 2, 2, 1])
        row[0].markdown(f"<div style='font-size:13px;color:#e2e8f0;padding:3px 0'>{cid}</div>", unsafe_allow_html=True)
        row[1].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:3px 0'>{n_cy}</div>", unsafe_allow_html=True)
        row[2].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:3px 0'>{cap_min:.3f} – {cap_max:.3f}</div>", unsafe_allow_html=True)
        row[3].markdown(f"<div style='font-size:13px;color:#a0aec0;padding:3px 0'>{res_min:.4f} – {res_max:.4f}</div>", unsafe_allow_html=True)
        row[4].markdown(f"<div style='font-size:13px;padding:3px 0;color:{'#48bb78' if temp_ok else '#4a5568'}'>{'Yes' if temp_ok else 'No'}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    fig_prev = go.Figure()
    capacity_going_up = False
    for cid in summary["cycles_per_cell"]:
        cell_df = df_raw_clean[df_raw_clean["cell_id"].astype(str).str.strip() == cid].sort_values("cycle_number")
        cap = pd.to_numeric(cell_df["capacity_ah"], errors="coerce").dropna()
        if len(cap) >= 3:
            first_third = cap.iloc[:max(1, len(cap)//3)].mean()
            last_third  = cap.iloc[-max(1, len(cap)//3):].mean()
            if last_third > first_third * 1.05:
                capacity_going_up = True
        fig_prev.add_trace(go.Scatter(
            x=cell_df["cycle_number"].tolist(), y=cell_df["capacity_ah"].tolist(),
            mode="lines", name=cid, line=dict(width=1.5),
        ))
    fig_prev.update_layout(
        **base_layout(
            height=280,
            xaxis=dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
                       title=dict(text="Cycle number", font=dict(size=11))),
            yaxis=dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
                       title=dict(text="Capacity (Ah)", font=dict(size=11))),
        )
    )
    fig_prev.update_layout(legend=LEGEND_H, title=dict(text="Capacity fade — uploaded cells", font=dict(size=12, color="#a0aec0"), x=0))
    st.plotly_chart(fig_prev, use_container_width=True)

    if capacity_going_up:
        st.markdown(
            "<div style='background:rgba(183,121,31,0.08);border:1px solid rgba(183,121,31,0.3);"
            "border-radius:8px;padding:10px 16px;font-size:12px;color:#f6ad55;margin-top:-8px'>"
            "⚠ One or more cells show capacity <em>increasing</em> over cycles — this may indicate "
            "a unit error (mAh uploaded instead of Ah). If so, divide all capacity values by 1000 "
            "and re-upload. Fade in Ah should decrease or stay flat over cycle life."
            "</div>",
            unsafe_allow_html=True,
        )

    _section("Analyse This Data")

    st.markdown(
        "<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
        "padding:20px 24px;margin-bottom:16px'>"
        f"<div style='font-size:13px;color:#a0aec0;line-height:1.7'>"
        f"<strong style='color:#e2e8f0'>{summary['n_cells']} cells</strong> · "
        f"<strong style='color:#e2e8f0'>{sum(summary['cycles_per_cell'].values()):,} total cycles</strong> · "
        f"{'Temperature available' if summary['has_temperature'] else 'Temperature assumed 25°C'}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("uploaded_featured_dfs") and st.session_state.get("uploaded_mode_meta"):
        _show_upload_summary()
    else:
        _run_analysis_button(df_raw, summary)

    st.markdown(
        "<div style='font-size:11px;color:#4a5568;margin-top:12px;text-align:center'>"
        "Uploaded data is session-scoped only — refreshing the page reverts to NASA mode. "
        "Switch between modes at any time using the sidebar without losing any data."
        "</div>",
        unsafe_allow_html=True,
    )
