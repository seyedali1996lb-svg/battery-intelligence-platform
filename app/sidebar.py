"""
Sidebar rendering extracted from main.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from utils import NASA_CELL_IDS, _md_html, LEGEND_H
from data_loader import CELL_STRESS_PROFILES, _stress_factor
from design_system import make_badge, make_state_badge

NAV_ITEMS = [
    ("Overview",        "overview",        True),
    ("Fleet",           "fleet",           True),
    ("Health",          "health",          True),
    ("Insights",        "insights",        True),
    ("Copilot",         "copilot",         True),
    ("Compare",         "compare",         True),
    ("Recommendations", "recommendations", True),
    ("Consequences",    "consequences",    True),
    ("Sustainability",  "sustainability",  True),
    ("Passport",        "passport",        True),
    ("Grading",         "grading",         True),
    ("Reports",         "reports",         True),
    ("Import",          "import",          True),
    ("Settings",        "settings",        True),
]


def _upload_status_line(meta: dict) -> str:
    """One-line status string for the My Data mode row."""
    n = meta["n_cells"]
    k = meta.get("calibrating_count", 0)
    parts = [f"{n} cells"]
    if k > 0:
        parts.append(f"{k} Calibrating")
    parts.append("uploaded today")
    if meta.get("lco_limited"):
        parts.append("limited LCO")
    return " · ".join(parts)


def render_mode_switcher(nasa_n: int, synth_n: int, up_meta: dict | None):
    """Persistent three-mode data source selector rendered inside the sidebar."""
    current = st.session_state.get("data_mode", "nasa")

    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding:0 4px 6px'>Data Source</div>",
        unsafe_allow_html=True,
    )

    modes = [
        {
            "key":       "nasa",
            "label":     "NASA Research Mode",
            "status":    f"{nasa_n} cells · real measured data",
            "available": nasa_n > 0,
        },
        {
            "key":       "synthetic",
            "label":     "Synthetic Fleet Mode",
            "status":    f"{synth_n} cells · physics-informed synthetic",
            "available": True,
        },
        {
            "key":       "uploaded",
            "label":     "My Data",
            "status":    _upload_status_line(up_meta) if up_meta else "Not yet uploaded",
            "available": up_meta is not None,
        },
    ]

    for m in modes:
        is_active    = current == m["key"]
        is_available = m["available"]

        if is_active:
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:8px;"
                f"padding:9px 12px;margin-bottom:5px'>"
                f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>"
                f"● {m['label']}</div>"
                f"<div style='font-size:11px;color:#718096;margin-top:2px'>{m['status']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif is_available:
            if st.button(
                f"○  {m['label']}",
                key=f"mode_btn_{m['key']}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["data_mode"] = m["key"]
                st.rerun()
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;margin:-8px 0 5px 4px'>"
                f"{m['status']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:8px 12px;margin-bottom:5px;opacity:0.45'>"
                f"<div style='font-size:13px;color:#718096'>○  {m['label']}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{m['status']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def render_sidebar(cell_ids: list[str], mode: str, nasa_n: int, synth_n: int,
                   up_meta: dict | None) -> str:
    with st.sidebar:
        n_cells = len(cell_ids)
        if mode == "nasa":
            subtitle = f"{n_cells} NASA real cells · leave-cell-out model"
        elif mode == "synthetic":
            subtitle = f"{n_cells} synthetic cells · leave-cell-out model"
        elif mode == "uploaded":
            cell_label = up_meta.get("cell_ids", cell_ids) if up_meta else cell_ids
            subtitle = f"{n_cells} uploaded cell{'s' if n_cells != 1 else ''} · your data"
        else:
            subtitle = f"{nasa_n + synth_n} cells ({synth_n} synthetic + {nasa_n} NASA real) · multi-cell model"

        st.markdown(
            f"<div style='padding:0 4px 16px'>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0'>⚡ Battery Intel</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{subtitle}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        render_mode_switcher(nasa_n, synth_n, up_meta)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if "page" not in st.session_state:
            st.session_state.page = "overview"
        current_page = st.session_state.page
        _nav_key_counts: dict[str, int] = {}
        for label, key, enabled in NAV_ITEMS:
            _nav_key_counts[key] = _nav_key_counts.get(key, 0) + 1
            btn_key = f"nav_{key}" if _nav_key_counts[key] == 1 else f"nav_{key}_b"
            if enabled:
                if st.button(
                    label, key=btn_key, use_container_width=True,
                    type="primary" if current_page == key else "secondary",
                ):
                    st.session_state.page = key
                    st.rerun()
            else:
                st.markdown(
                    f"<div style='padding:7px 12px;color:#4a5568;font-size:14px;"
                    f"font-weight:500;display:flex;justify-content:space-between;align-items:center'>"
                    f"<span>{label}</span>"
                    f"<span style='font-size:10px;background:#1a202c;color:#4a5568;"
                    f"padding:1px 7px;border-radius:10px'>Soon</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;padding:0 4px 8px'>Cell</div>",
            unsafe_allow_html=True,
        )
        if "_nav_cell" in st.session_state:
            _nav_target = st.session_state.pop("_nav_cell")
            if _nav_target in cell_ids:
                st.session_state["selected_cell"] = _nav_target
        _cur_sel = st.session_state.get("selected_cell")
        _sel_idx = cell_ids.index(_cur_sel) if _cur_sel in cell_ids else 0
        selected = st.selectbox(
            "Select cell",
            options=cell_ids,
            index=_sel_idx,
            key="selected_cell",
            label_visibility="collapsed",
        )

        _cur_idx = cell_ids.index(selected)
        _nav_prev, _nav_next = st.columns(2)
        with _nav_prev:
            if st.button("← Prev", key="cell_prev", use_container_width=True,
                         disabled=(_cur_idx == 0)):
                st.session_state["_nav_cell"] = cell_ids[_cur_idx - 1]
                st.rerun()
        with _nav_next:
            if st.button("Next →", key="cell_next", use_container_width=True,
                         disabled=(_cur_idx == len(cell_ids) - 1)):
                st.session_state["_nav_cell"] = cell_ids[_cur_idx + 1]
                st.rerun()

        if mode == "uploaded":
            temp_assumed_cells = (up_meta or {}).get("temperature_assumed_cells", [])
            temp_assumed = selected in temp_assumed_cells
            temp_note    = "25°C (assumed)" if temp_assumed else "measured"
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;padding:4px 4px 0;line-height:1.7'>"
                f"Source: user upload · this session only<br>"
                f"Temperature: <span style='color:{'#718096' if temp_assumed else '#a0aec0'}'>"
                f"{temp_note}</span><br>"
                f"<span style='color:#63b3ed'>Uploaded cell</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif mode == "nasa":
            st.markdown(
                "<div style='font-size:11px;color:#4a5568;padding:4px 4px 0;line-height:1.7'>"
                "Source: NASA PCoE Battery Aging Dataset<br>"
                "T=24°C &nbsp; C-rate=2A &nbsp; DoD=100%<br>"
                "<span style='color:#68d391'>Real measured data</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:  # synthetic
            p  = CELL_STRESS_PROFILES.get(selected, {})
            sf = _stress_factor(p.get("temp_mean", 25), p.get("c_rate", 1), p.get("dod", 1))
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;padding:4px 4px 0;line-height:1.7'>"
                f"T={p.get('temp_mean',25):.0f}°C &nbsp; "
                f"C-rate={p.get('c_rate',1.0):.1f}C &nbsp; "
                f"DoD={p.get('dod',1.0)*100:.0f}%<br>"
                f"Stress: {sf:.2f}× baseline &nbsp; "
                f"<span style='color:#fc8181'>Synthetic</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;padding:0 4px 8px'>Chemistry</div>",
            unsafe_allow_html=True,
        )
        _chem_options = ["Li-ion (LiCoO₂)", "Li-S (Lithium-Sulfur)", "SSB (Solid-State)"]
        _chem_sel = st.selectbox(
            "Chemistry", options=_chem_options,
            index=0, key="chemistry_selector", label_visibility="collapsed",
        )
        st.session_state["active_chemistry"] = _chem_sel
        if "Li-S" in _chem_sel:
            st.markdown(
                "<div style='font-size:11px;color:#f6ad55;padding:2px 4px'>⚠ Li-S: dual plateau, "
                "shuttle-driven CE ~95–99%, faster fade</div>", unsafe_allow_html=True,
            )
        elif "SSB" in _chem_sel:
            st.markdown(
                "<div style='font-size:11px;color:#63b3ed;padding:2px 4px'>ℹ SSB: no Warburg "
                "diffusion tail, interface resistance dominant, higher Ea</div>", unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#2d3748;padding:0 4px;line-height:1.7'>"
            "Phase 1 · scikit-learn GBRT<br>"
            "8 synthetic + 4 NASA real cells<br>"
            "Cell-to-cell stress variation (T, C-rate, DoD)<br>"
            "<span style='color:#fc8181'>⚠ Synthetic cells: not real measured data</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.toggle("Light mode", key="light_mode", value=False)

    return selected
