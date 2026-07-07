"""Page: Live Monitor (MQTT BMS streaming + anomaly detection)."""

import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from utils import _md_html, _empty_state, _action_bar, base_layout


def page_live_monitor(cell_ids: list, active_fdfs: dict):
    _action_bar("live_monitor")
    st.markdown("# Live Monitor")

    import sys as _sys_lm
    _src_lm = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    if _src_lm not in _sys_lm.path:
        _sys_lm.path.insert(0, _src_lm)
    from mqtt_stream import (
        start_subscriber, stop_subscriber, is_subscriber_connected,
        start_publisher, stop_publisher, publisher_running,
        drain_telemetry, drain_anomalies,
        DEFAULT_HOST, DEFAULT_PORT, TOPIC_PREFIX,
    )

    # ── Connection settings ───────────────────────────────────────────────────
    _md_html(
        "<div style='font-size:12px;color:#8896a8;margin-bottom:12px'>"
        "Streams real-time BMS telemetry via MQTT. In <strong>Demo mode</strong> the replay "
        "publisher re-publishes the selected cell's historical data as a live stream — "
        "the same pipeline works with a real BMS broker in production."
        "</div>"
    )

    _replay_speed = st.selectbox(
        "Replay speed", options=[1, 5, 10, 20], index=2,
        format_func=lambda x: f"{x}×",
        key="mqtt_speed_input",
        help="How many telemetry readings per second the replay publisher sends.",
    )
    with st.expander("⚙ Broker settings", expanded=False):
        _cfg_col1, _cfg_col2 = st.columns([3, 1])
        _broker_host = _cfg_col1.text_input(
            "MQTT broker", value=st.session_state.get("mqtt_host", DEFAULT_HOST),
            key="mqtt_host_input",
            help="Hostname of your MQTT broker. Default: test.mosquitto.org (public, no auth).",
        )
        _broker_port = _cfg_col2.number_input(
            "Port", value=int(st.session_state.get("mqtt_port", DEFAULT_PORT)),
            min_value=1, max_value=65535, step=1, key="mqtt_port_input",
        )
        st.session_state["mqtt_host"] = _broker_host
        st.session_state["mqtt_port"] = _broker_port

    # ── Cell selector for replay ──────────────────────────────────────────────
    _replay_cell = st.selectbox(
        "Cell to replay", options=cell_ids, key="lm_replay_cell",
        help="Historical data for this cell is replayed through MQTT as BMS telemetry.",
    )

    # ── Start / Stop controls ─────────────────────────────────────────────────
    _btn_col1, _btn_col2, _btn_col3 = st.columns([1, 1, 4])
    _sub_connected = is_subscriber_connected()
    _pub_active    = publisher_running()

    def _clear_pybamm_cache(cell_id: str) -> None:
        st.session_state.pop(f"lm_pybamm_result_{cell_id}", None)
        st.session_state.pop(f"lm_pybamm_computed_at_{cell_id}", None)

    if _btn_col1.button(
        "▶ Start" if not (_sub_connected and _pub_active) else "⏹ Stop",
        key="lm_toggle",
        use_container_width=True,
        type="primary" if not (_sub_connected and _pub_active) else "secondary",
    ):
        if _sub_connected or _pub_active:
            stop_publisher()
            stop_subscriber()
            st.session_state["lm_telemetry"] = []
            st.session_state["lm_anomalies"] = []
            _clear_pybamm_cache(_replay_cell)
            st.rerun()
        else:
            # Clear old buffers
            st.session_state["lm_telemetry"] = []
            st.session_state["lm_anomalies"] = []
            _clear_pybamm_cache(_replay_cell)
            # Start subscriber first, then publisher
            _data_mode = st.session_state.get("data_mode", "synthetic")
            _chem_map  = {cid: ("LFP" if cid.startswith("S-") else "LiCoO2")
                          for cid in cell_ids}
            _ok_sub = start_subscriber(
                host=_broker_host, port=_broker_port,
                cell_ids=[_replay_cell], chemistry_map=_chem_map,
            )
            if _ok_sub:
                _df_replay = active_fdfs.get(_replay_cell)
                if _df_replay is not None:
                    start_publisher(
                        _df_replay, _replay_cell,
                        host=_broker_host, port=_broker_port,
                        speed=float(_replay_speed), loop=True,
                    )
                st.rerun()
            else:
                st.error(f"Could not connect to {_broker_host}:{_broker_port}. "
                         "Check broker address or network access.")

    if _btn_col2.button("🗑 Clear", key="lm_clear", use_container_width=True):
        st.session_state["lm_telemetry"] = []
        st.session_state["lm_anomalies"] = []
        _clear_pybamm_cache(_replay_cell)
        st.rerun()

    # ── Status strip ─────────────────────────────────────────────────────────
    _sub_connected = is_subscriber_connected()
    _pub_active    = publisher_running()
    _status_colour = "#48bb78" if (_sub_connected and _pub_active) else "#718096"
    _status_label  = (
        f"🟢 Streaming {_replay_cell} → {_broker_host}:{_broker_port} at {_replay_speed}×"
        if (_sub_connected and _pub_active) else
        "⚫ Stopped — press ▶ Start to begin replay"
    )
    st.markdown(
        f"<div style='background:#1e2a38;border:1px solid {_status_colour}44;"
        f"border-radius:8px;padding:8px 14px;font-size:12px;color:{_status_colour};"
        f"margin-bottom:12px'>{_status_label}</div>",
        unsafe_allow_html=True,
    )

    # ── Accumulate telemetry into session state ───────────────────────────────
    if "lm_telemetry" not in st.session_state:
        st.session_state["lm_telemetry"] = []
    if "lm_anomalies" not in st.session_state:
        st.session_state["lm_anomalies"] = []

    if _sub_connected:
        _new_telem   = drain_telemetry(200)
        _new_anomaly = drain_anomalies(100)
        if _new_telem:
            st.session_state["lm_telemetry"].extend(_new_telem)
            # Keep last 1000 readings in memory
            st.session_state["lm_telemetry"] = st.session_state["lm_telemetry"][-1000:]
        if _new_anomaly:
            st.session_state["lm_anomalies"].extend(_new_anomaly)
            st.session_state["lm_anomalies"] = st.session_state["lm_anomalies"][-200:]
            # ── Webhook push ────────────────────────────────────────────────
            _wh_url_lm  = st.session_state.get("webhook_url", "")
            _wh_evts_lm = st.session_state.get("webhook_events", [])
            _wh_sec_lm  = st.session_state.get("webhook_secret", "")
            if _wh_url_lm and _wh_evts_lm:
                from notifications import send_webhook
                for _an in _new_anomaly:
                    _evt_type = _an.get("anomaly_type", "")
                    if _evt_type not in _wh_evts_lm:
                        continue
                    send_webhook(
                        _evt_type,
                        {
                            "cell_id":    _an.get("cell_id", _replay_cell),
                            "severity":   _an.get("severity", "HIGH"),
                            "value":      _an.get("value"),
                            "threshold":  _an.get("threshold"),
                            "message":    _an.get("message", ""),
                            "standard":   "IEC 62619:2022",
                            "timestamp":  _an.get("timestamp", datetime.datetime.now().isoformat()),
                        },
                        _wh_url_lm, _wh_sec_lm,
                    )

    _telem = st.session_state["lm_telemetry"]
    _anom  = st.session_state["lm_anomalies"]

    if not _telem:
        _empty_state(
            "No telemetry received yet",
            "The subscriber is connected and listening. Readings will appear here as they arrive. "
            "If nothing appears within a few seconds, check the broker address and port.",
            "→ Press ▶ Start above to begin the replay stream.",
            "📡",
        )
        if _sub_connected:
            time.sleep(0.5)
            st.rerun()
        return

    # ── Convert to DataFrame ──────────────────────────────────────────────────
    import pandas as _pd_lm
    _df_telem = _pd_lm.DataFrame(_telem)
    _df_telem = _df_telem[_df_telem["cell_id"] == _replay_cell] if "cell_id" in _df_telem.columns else _df_telem

    # ── Live metrics strip ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Live Readings</div>", unsafe_allow_html=True)
    _latest_row = _telem[-1]
    _lm1, _lm2, _lm3, _lm4, _lm5 = st.columns(5)
    _lm1.metric("Voltage",     f"{_latest_row.get('voltage_v', '—'):.3f} V"     if _latest_row.get('voltage_v')     is not None else "—")
    _lm2.metric("Current",     f"{_latest_row.get('current_a', '—'):.2f} A"     if _latest_row.get('current_a')     is not None else "—")
    _lm3.metric("Temperature", f"{_latest_row.get('temperature_c', '—'):.1f} °C" if _latest_row.get('temperature_c') is not None else "—")
    _lm4.metric("SOH",         f"{_latest_row.get('soh_pct', '—'):.1f} %"       if _latest_row.get('soh_pct')       is not None else "—")
    _lm5.metric("Readings",    f"{len(_telem):,}")

    # ── Physics Twin Check (PyBaMM re-fit against streamed telemetry) ────────
    # Digital Twin Quality review finding: PyBaMM previously only ran once,
    # offline, against a cell's full historical data (Health page's Model
    # Comparison) — never against telemetry as it arrives. This re-fits the
    # same SEI-fade physics model (src/pybamm_rul.py) against only the
    # cycles received via the live stream so far, re-run periodically as
    # more data arrives. This is NOT the "continuously re-parameterized
    # per-cell physics twin" a real Siemens/ABB-grade twin would be — the
    # PyBaMM parameter set itself is still fixed per chemistry, not fitted
    # from telemetry — so it's labelled accordingly rather than oversold.
    st.markdown("<div class='section-header'>⚛ Physics Twin Check (PyBaMM)</div>", unsafe_allow_html=True)
    _PB_RECOMPUTE_EVERY = 15  # readings — a full SPM run takes ~2-3s; too slow to redo every 1s rerun
    _cycle_vals = [t.get("cycle") for t in _telem if t.get("cycle") is not None]
    _soh_vals   = [t.get("soh_pct") for t in _telem if t.get("soh_pct") is not None]
    _cap_vals   = [t.get("capacity_ah") for t in _telem]
    _n_usable   = min(len(_cycle_vals), len(_soh_vals))

    if _n_usable < 5:
        st.caption(f"⚛ Physics twin check needs ≥5 telemetry readings with cycle/SOH data (have {_n_usable}) — waiting for more of the stream.")
    else:
        _pb_cache_key      = f"lm_pybamm_result_{_replay_cell}"
        _pb_computed_at_key = f"lm_pybamm_computed_at_{_replay_cell}"
        _last_computed_at  = st.session_state.get(_pb_computed_at_key, 0)
        _should_recompute  = (
            _pb_cache_key not in st.session_state
            or (_n_usable - _last_computed_at) >= _PB_RECOMPUTE_EVERY
        )
        if _should_recompute:
            import pandas as _pd_pb
            _telem_df = _pd_pb.DataFrame({
                "cycle_number": _cycle_vals[:_n_usable],
                "soh_pct":      _soh_vals[:_n_usable],
                "capacity_ah":  _cap_vals[:_n_usable],
            })
            _lm_data_mode = st.session_state.get("data_mode", "synthetic")
            with st.spinner("Re-fitting physics model against streamed telemetry…"):
                from pybamm_rul import project_rul
                st.session_state[_pb_cache_key] = project_rul(_replay_cell, _telem_df, _lm_data_mode)
                st.session_state[_pb_computed_at_key] = _n_usable

        _pb_result = st.session_state.get(_pb_cache_key, {})
        if _pb_result.get("error"):
            st.caption(f"⚛ Physics twin check unavailable: {_pb_result['error']}")
        else:
            _pb1, _pb2, _pb3 = st.columns(3)
            _pb1.metric("Physics RUL estimate", f"{_pb_result.get('rul_physics', '—')} cy" if _pb_result.get("rul_physics") is not None else "—")
            _pb2.metric("Chemistry model", _pb_result.get("chem_label", "—"))
            _pb3.metric("Fit from telemetry", f"{st.session_state.get(_pb_computed_at_key, _n_usable)} cycles")
            st.caption(
                f"Re-fit against {st.session_state.get(_pb_computed_at_key, _n_usable)} cycles received via this "
                f"live stream so far (recomputed every {_PB_RECOMPUTE_EVERY} new readings, not on every tick — "
                f"a full physics simulation takes a few seconds). Still uses a fixed PyBaMM parameter set per "
                f"chemistry, not one re-parameterized from telemetry — this is a physics-consistency re-check "
                f"against streaming data, not a live-synced digital twin."
            )

    # ── Real-time charts ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Telemetry Stream</div>", unsafe_allow_html=True)

    _x_axis = _df_telem["seq"].tolist() if "seq" in _df_telem.columns else list(range(len(_df_telem)))

    _chart_pairs = [
        ("voltage_v",      "Voltage (V)",       "#63b3ed"),
        ("current_a",      "Current (A)",        "#f6ad55"),
        ("temperature_c",  "Temperature (°C)",   "#fc8181"),
        ("soh_pct",        "SOH (%)",            "#68d391"),
    ]
    _available = [(col, lbl, clr) for col, lbl, clr in _chart_pairs if col in _df_telem.columns and _df_telem[col].notna().any()]

    if _available:
        _n_charts = len(_available)
        _chart_cols = st.columns(_n_charts if _n_charts <= 2 else 2)
        for _ci, (col, lbl, clr) in enumerate(_available):
            _ccol = _chart_cols[_ci % 2]
            _fig_lm = go.Figure()
            _y_vals = _df_telem[col].tolist()

            # Shade anomaly points red
            _anom_seqs = {a["seq"] for a in _anom if f"_{col.split('_')[0].upper()}" in a.get("kind","").upper() or "VOLTAGE" in a.get("kind","").upper()}
            _anom_mask = [1 if x in _anom_seqs else 0 for x in _x_axis]
            _anom_y    = [y if m else None for y, m in zip(_y_vals, _anom_mask)]

            _fig_lm.add_trace(go.Scatter(
                x=_x_axis, y=_y_vals,
                name=lbl, line=dict(color=clr, width=1.5),
                hovertemplate=f"%{{y:.3f}}<extra>{lbl}</extra>",
            ))
            if any(v is not None for v in _anom_y):
                _fig_lm.add_trace(go.Scatter(
                    x=_x_axis, y=_anom_y,
                    name="Anomaly", mode="markers",
                    marker=dict(color="#fc8181", size=8, symbol="x"),
                    hovertemplate="⚠ Anomaly<extra></extra>",
                ))
            _fig_lm.update_layout(
                height=200,
                **base_layout(
                    margin=dict(l=10, r=10, t=28, b=10),
                    showlegend=False,
                    xaxis=dict(title="Reading #", zeroline=False),
                    yaxis=dict(zeroline=False),
                ),
                title=dict(text=lbl, font=dict(size=11, color="#a0aec0"), x=0),
            )
            _ccol.plotly_chart(_fig_lm, use_container_width=True)

    # ── Anomaly log ───────────────────────────────────────────────────────────
    # D2: Rule-based differential diagnosis per anomaly type
    def _anomaly_diagnosis(kind: str, detail: str, value=None, threshold=None) -> str:
        k = str(kind).upper()
        if "THERMAL_RUNAWAY" in k:
            return (
                "Rapid temperature rise detected. If during active charge/discharge, "
                "halt immediately — possible separator failure or internal short. "
                "If at replay start, may be measurement artifact; monitor next 5 readings."
            )
        if "UNDERTEMPERATURE" in k:
            return (
                "Cell below safe operating temperature. Lithium plating risk during "
                "charge is elevated below 0 °C. Suspend charging until temperature recovers. "
                "Discharge at reduced rate is acceptable."
            )
        if "TEMP_RATE" in k:
            return (
                "Temperature rising faster than expected. Not yet critical, but watch the "
                "next 3–5 readings. If rise continues, check for cooling system fault or "
                "abnormal load. CE trend will confirm if chemistry is involved."
            )
        if "CAPACITY_PLUNGE" in k:
            return (
                "SOH dropped sharply in one reading. Most likely a BMS communication glitch "
                "or measurement artifact — single-cycle drops rarely reflect true capacity loss. "
                "Check if the drop persists next cycle before escalating."
            )
        if "VOLTAGE_HIGH" in k:
            return (
                "Voltage above upper limit. Overcharge condition — stop charging immediately. "
                "Sustained overcharge accelerates electrolyte decomposition and SEI growth. "
                "Check charger cutoff threshold against cell specification."
            )
        if "VOLTAGE_LOW" in k:
            return (
                "Voltage below lower cutoff. Deep discharge causes copper dissolution and "
                "irreversible capacity loss. Discontinue discharge. If voltage does not "
                "recover at rest, the cell may need retirement evaluation."
            )
        return "Monitor subsequent readings. If pattern repeats, correlate with SOH trend."

    st.markdown("<div class='section-header'>Anomaly Log</div>", unsafe_allow_html=True)
    if _anom:
        _anom_recent = list(reversed(_anom[-50:]))
        for _a in _anom_recent:
            _sev   = _a.get("severity", "warning")
            _ac    = "#fc8181" if _sev == "critical" else "#f6ad55"
            _akind = _a.get("kind", "UNKNOWN")
            _adet  = _a.get("detail", "")
            _ats   = _a.get("ts", "")[:19].replace("T", " ")
            _diag  = _anomaly_diagnosis(_akind, _adet, _a.get("value"), _a.get("threshold"))
            st.markdown(
                f"<div style='background:{_ac}11;border-left:3px solid {_ac};"
                f"border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:12px'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                f"<span style='color:{_ac};font-weight:700'>{_akind}</span>"
                f"<span style='color:#4a5568;font-size:11px'>{_ats}</span>"
                f"</div>"
                f"<div style='color:#a0aec0;margin-bottom:4px'>{_adet}</div>"
                f"<div style='color:#718096;font-size:11px;border-top:1px solid {_ac}22;"
                f"padding-top:4px;margin-top:4px'>"
                f"<span style='color:{_ac};font-weight:600'>Diagnosis: </span>{_diag}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        # CSV export
        _df_anom_exp = _pd_lm.DataFrame(_anom)
        _anom_csv    = _df_anom_exp.to_csv(index=False).encode()
        st.download_button(
            "Export anomaly log CSV", data=_anom_csv,
            file_name=f"anomaly_log_{_replay_cell}.csv", mime="text/csv",
            key="lm_export_anom",
        )
    else:
        st.markdown(
            "<div style='color:#4a5568;font-size:13px;padding:12px 0'>No anomalies detected.</div>",
            unsafe_allow_html=True,
        )

    # ── Auto-refresh while streaming ──────────────────────────────────────────
    if _sub_connected and _pub_active:
        time.sleep(1.0)
        st.rerun()
