"""Page: Live Monitor (MQTT BMS streaming + anomaly detection)."""

import datetime

import os
import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path
import streamlit as st
import plotly.graph_objects as go

from utils import _md_html, _empty_state, _action_bar, base_layout
from chemistry_profiles import ChemistryProfile

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
        drain_telemetry, drain_anomalies, drain_faults,
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
            st.session_state["lm_faults"] = []
            _clear_pybamm_cache(_replay_cell)  # pyright: ignore[reportArgumentType]
            st.rerun()
        else:
            # Clear old buffers
            st.session_state["lm_telemetry"] = []
            st.session_state["lm_anomalies"] = []
            st.session_state["lm_faults"] = []
            _clear_pybamm_cache(_replay_cell)  # pyright: ignore[reportArgumentType]
            # Start subscriber first, then publisher
            _data_mode = st.session_state.get("data_mode", "synthetic")
            # ChemistryProfile.for_cell() also correctly wires up "NCA" for
            # Oxford cells — src/mqtt_stream.py's _VOLTAGE_LIMITS already had
            # an NCA entry that nothing ever fed before this.
            _chem_map  = {cid: ChemistryProfile.for_cell(cid).short_name
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
        st.session_state["lm_faults"] = []
        _clear_pybamm_cache(_replay_cell)  # pyright: ignore[reportArgumentType]
        st.rerun()

    # ── Status strip + streaming telemetry ────────────────────────────────────
    # Wrapped in a fragment so periodic refresh only re-renders this region
    # instead of rerunning the whole page (buttons, broker settings, cell
    # selector). Auto-refresh uses @st.fragment(run_every=...) rather than a
    # manual time.sleep()+st.rerun(scope="fragment") loop: that pattern
    # raises StreamlitAPIException the very first time the fragment executes
    # as part of a full script run, because scope="fragment" is only valid
    # from inside an already-in-progress fragment rerun -- which the first
    # execution never is (confirmed against Streamlit's source, and against
    # a real deployment where it broke exactly this way). run_every is
    # Streamlit's own timer-driven fragment-rerun mechanism and has no such
    # restriction -- no manual sleep/rerun needed at all.
    _lm_run_every = 0.5 if is_subscriber_connected() else None

    @st.fragment(run_every=_lm_run_every)
    def _telemetry_fragment():
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

        # ── Accumulate telemetry into session state ───────────────────────────
        if "lm_telemetry" not in st.session_state:
            st.session_state["lm_telemetry"] = []
        if "lm_anomalies" not in st.session_state:
            st.session_state["lm_anomalies"] = []
        if "lm_faults" not in st.session_state:
            st.session_state["lm_faults"] = []

        if _sub_connected:
            _new_telem   = drain_telemetry(200)
            _new_anomaly = drain_anomalies(100)
            _new_fault   = drain_faults(100)
            if _new_telem:
                st.session_state["lm_telemetry"].extend(_new_telem)
                # Keep last 1000 readings in memory
                st.session_state["lm_telemetry"] = st.session_state["lm_telemetry"][-1000:]
            _wh_url_lm  = st.session_state.get("webhook_url", "")
            _wh_evts_lm = st.session_state.get("webhook_events", [])
            _wh_sec_lm  = st.session_state.get("webhook_secret", "")
            if _new_anomaly:
                st.session_state["lm_anomalies"].extend(_new_anomaly)
                st.session_state["lm_anomalies"] = st.session_state["lm_anomalies"][-200:]
                # ── Webhook push ────────────────────────────────────────────
                if _wh_url_lm and _wh_evts_lm:
                    from notifications import send_webhook, notify_subscribers
                    for _an in _new_anomaly:
                        _evt_type = _an.get("anomaly_type", "")
                        if _evt_type not in _wh_evts_lm:
                            continue
                        _an_payload = {
                            "cell_id":    _an.get("cell_id", _replay_cell),
                            "severity":   _an.get("severity", "HIGH"),
                            "value":      _an.get("value"),
                            "threshold":  _an.get("threshold"),
                            "message":    _an.get("message", ""),
                            "standard":   "IEC 62619:2022",
                            "timestamp":  _an.get("timestamp", datetime.datetime.now().isoformat()),
                        }
                        send_webhook(_evt_type, _an_payload, _wh_url_lm, _wh_sec_lm)
                        notify_subscribers(st.session_state["auth_org_id"], _evt_type, _an_payload)
            if _new_fault:
                st.session_state["lm_faults"].extend(_new_fault)
                st.session_state["lm_faults"] = st.session_state["lm_faults"][-200:]
                # ── Webhook push (ingestion faults -- malformed/corrupted
                # data, distinct from the anomaly push above) ──────────────
                if _wh_url_lm and "INGESTION_FAULT" in _wh_evts_lm:
                    from notifications import send_webhook, notify_subscribers
                    for _flt in _new_fault:
                        _flt_payload = {
                            "cell_id":   _flt.get("cell_id", _replay_cell),
                            "kind":      _flt.get("kind"),
                            "severity":  _flt.get("severity", "warning"),
                            "detail":    _flt.get("detail", ""),
                            "timestamp": _flt.get("ts", datetime.datetime.now().isoformat()),
                        }
                        send_webhook("INGESTION_FAULT", _flt_payload, _wh_url_lm, _wh_sec_lm)
                        notify_subscribers(st.session_state["auth_org_id"], "INGESTION_FAULT", _flt_payload)

        _telem  = st.session_state["lm_telemetry"]
        _anom   = st.session_state["lm_anomalies"]
        _faults = st.session_state["lm_faults"]

        if not _telem:
            _empty_state(
                "No telemetry received yet",
                "The subscriber is connected and listening. Readings will appear here as they arrive. "
                "If nothing appears within a few seconds, check the broker address and port.",
                "→ Press ▶ Start above to begin the replay stream.",
                "📡",
            )
            return

        # ── Convert to DataFrame ────────────────────────────────────────────────
        import pandas as _pd_lm
        _df_telem = _pd_lm.DataFrame(_telem)
        _df_telem = _df_telem[_df_telem["cell_id"] == _replay_cell] if "cell_id" in _df_telem.columns else _df_telem

        # ── Live metrics strip ──────────────────────────────────────────────────
        st.markdown("<h4 class='section-header'>Live Readings</h4>", unsafe_allow_html=True)
        _latest_row = _telem[-1]
        _lm1, _lm2, _lm3, _lm4, _lm5 = st.columns(5)
        _lm1.metric("Voltage",     f"{_latest_row.get('voltage_v', '—'):.3f} V"     if _latest_row.get('voltage_v')     is not None else "—")
        _lm2.metric("Current",     f"{_latest_row.get('current_a', '—'):.2f} A"     if _latest_row.get('current_a')     is not None else "—")
        _lm3.metric("Temperature", f"{_latest_row.get('temperature_c', '—'):.1f} °C" if _latest_row.get('temperature_c') is not None else "—")
        _lm4.metric("SOH",         f"{_latest_row.get('soh_pct', '—'):.1f} %"       if _latest_row.get('soh_pct')       is not None else "—")
        _lm5.metric("Readings",    f"{len(_telem):,}")

        # ── Physics Twin Check (PyBaMM re-fit against streamed telemetry) ──────
        # Digital Twin Quality review finding: PyBaMM previously only ran once,
        # offline, against a cell's full historical data (Health page's Model
        # Comparison) — never against telemetry as it arrives. This re-fits the
        # same SEI-fade physics model (src/pybamm_rul.py) against only the
        # cycles received via the live stream so far, re-run periodically as
        # more data arrives. This is NOT the "continuously re-parameterized
        # per-cell physics twin" a real Siemens/ABB-grade twin would be — the
        # PyBaMM parameter set itself is still fixed per chemistry, not fitted
        # from telemetry — so it's labelled accordingly rather than oversold.
        st.markdown("<h4 class='section-header'>⚛ Physics Twin Check (PyBaMM)</h4>", unsafe_allow_html=True)
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

        # ── Digital Twin (Phase 3 architecture) ────────────────────────────────
        # The CellTwin (src/digital_twin.py) is the platform's Phase 3
        # architecture: one continuously-updated {history + indicators +
        # physics projection} representation, re-fit on every update. Here it
        # consumes the same streamed cycles as the Physics Twin Check above,
        # kept fast (no SPM anchor) so it can re-derive every rerun; the twin's
        # own labels carry the "not a live-synced digital twin" honesty note.
        st.markdown("<h4 class='section-header'>🔄 Digital Twin (Phase 3)</h4>", unsafe_allow_html=True)
        if _n_usable < 5:
            st.caption(f"Digital twin needs ≥5 streamed cycles (have {_n_usable}).")
        else:
            import pandas as _pd_twin
            _lm_data_mode = st.session_state.get("data_mode", "synthetic")
            _twin_cache_key = f"lm_twin_{_replay_cell}"
            _twin_df = _pd_twin.DataFrame({
                "cycle_number": _cycle_vals[:_n_usable],
                "soh_pct":      _soh_vals[:_n_usable],
                "capacity_ah":  _cap_vals[:_n_usable],
            })
            if _twin_cache_key not in st.session_state:
                from digital_twin import CellTwin
                st.session_state[_twin_cache_key] = CellTwin(_replay_cell, _lm_data_mode, anchor_spm=False)
            _twin = st.session_state[_twin_cache_key]
            _twin.update(_twin_df)
            _snap = _twin.snapshot()
            _proj = _snap.get("projection") or {}
            _tw1, _tw2, _tw3, _tw4 = st.columns(4)
            _tw1.metric("SOH (twin)", f"{_snap['indicators'].get('soh_pct', '—'):.1f} %" if _snap["indicators"].get("soh_pct") is not None else "—")
            _tw2.metric("Fade rate (30cy)", f"{_snap['indicators'].get('fade_rate_30cy', 0):.3f} %/cy" if _snap["indicators"].get("fade_rate_30cy") is not None else "—")
            _tw3.metric("Projected RUL", f"{_proj.get('rul_cycles_to_eol', '—')} cy" if _proj.get("rul_cycles_to_eol") is not None else "—")
            _tw4.metric("Knee", (_snap["indicators"].get("knee") or {}).get("phase", "—"))
            _knee = (_snap["indicators"].get("knee") or {})
            if _knee.get("detected"):
                st.caption(
                    f"Knee detected at cycle {_knee.get('cycle')} (SOH {_knee.get('soh_at_knee'):.1f}%) — "
                    f"projected EOL in {_proj.get('rul_cycles_to_eol', '—')} cycles at {_proj.get('eol_threshold', 80)}% SOH."
                )
            if _snap.get("last_error"):
                st.caption(f"Twin last error: {_snap['last_error']}")
            st.caption(
                "One self-consistent representation re-fit on each streamed cycle batch: measured history → "
                "derived health indicators → SEI sqrt-fade projection (same model as the Physics Twin Check above). "
                "Projection, not prediction; not a live-synced digital twin (fixed per-chemistry parameter set)."
            )

        # ── Real-time charts ────────────────────────────────────────────────────
        st.markdown("<h4 class='section-header'>Telemetry Stream</h4>", unsafe_allow_html=True)

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

        # ── Anomaly log ──────────────────────────────────────────────────────────
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
            if "MULTI_SIGNAL" in k:
                return (
                    "Multiple sensor channels are drifting together — individually moderate, but "
                    "correlated across channels, which is a stronger fault signal than any one reading "
                    "alone (e.g. voltage sag with a temperature rise together is far more diagnostic of "
                    "an internal short than either alone). Treat as higher-confidence than a single-"
                    "channel Z-score flag; correlate against the Telemetry Stream charts below."
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

        def _group_anomalies(anom_list):
            # Collapse consecutive readings of the same kind into one episode
            # -- otherwise a sustained condition (e.g. overvoltage held across
            # 40 readings) renders as 40 near-identical cards instead of one
            # card with a status and reading count.
            groups = []
            for a in anom_list:
                k = a.get("kind", "UNKNOWN")
                if groups and groups[-1]["kind"] == k:
                    groups[-1]["last"] = a
                    groups[-1]["count"] += 1
                else:
                    groups.append({"kind": k, "first": a, "last": a, "count": 1})
            return groups

        st.markdown("<h4 class='section-header'>Anomaly Log</h4>", unsafe_allow_html=True)
        if _anom:
            _groups_all    = _group_anomalies(_anom)
            _groups_recent = list(reversed(_groups_all[-30:]))
            # aria-live: a screen reader is otherwise never notified when a
            # new (possibly safety-relevant, e.g. THERMAL_RUNAWAY) anomaly
            # appears during the 0.5s auto-refresh -- see the accessibility
            # audit that added this. Scoped to this log only, not the
            # faster-changing metrics strip above, since making a
            # sub-second-refreshing region aria-live would spam-announce.
            _anom_html = "<div aria-live='polite' aria-relevant='additions'>"
            for _g in _groups_recent:
                _first, _last = _g["first"], _g["last"]
                _sev   = _first.get("severity", "warning")
                _ac    = "#fc8181" if _sev == "critical" else "#f6ad55"
                _akind = _g["kind"]
                _adet  = _last.get("detail", "")
                _ts_start = _first.get("ts", "")[:19].replace("T", " ")
                _ts_end   = _last.get("ts", "")[:19].replace("T", " ")
                _is_active = _sub_connected and (_g is _groups_all[-1])
                _cnt_txt = f"{_g['count']} reading{'s' if _g['count'] != 1 else ''}"
                if _is_active:
                    _status = f"<span style='color:#68d391;font-weight:700'>● ACTIVE</span> — since {_ts_start} · {_cnt_txt}"
                elif _g["count"] > 1:
                    _status = f"<span style='color:#a0aec0;font-weight:700'>RESOLVED</span> — {_ts_start} → {_ts_end} · {_cnt_txt}"
                else:
                    _status = f"<span style='color:#a0aec0;font-weight:700'>RESOLVED</span> — {_ts_start} · {_cnt_txt}"
                _diag  = _anomaly_diagnosis(_akind, _adet, _last.get("value"), _last.get("threshold"))
                _anom_html += (
                    f"<div style='background:{_ac}11;border-left:3px solid {_ac};"
                    f"border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:12px'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                    f"<span style='color:{_ac};font-weight:700'>{_akind}</span>"
                    f"<span style='color:#a0aec0;font-size:11px'>{_ts_end}</span>"
                    f"</div>"
                    f"<div style='font-size:11px;margin-bottom:4px'>{_status}</div>"
                    f"<div style='color:#a0aec0;margin-bottom:4px'>{_adet}</div>"
                    f"<div style='color:#a0aec0;font-size:11px;border-top:1px solid {_ac}22;"
                    f"padding-top:4px;margin-top:4px'>"
                    f"<span style='color:{_ac};font-weight:600'>Diagnosis: </span>{_diag}</div>"
                    f"</div>"
                )
            _anom_html += "</div>"
            st.markdown(_anom_html, unsafe_allow_html=True)
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
                "<div style='color:#a0aec0;font-size:13px;padding:12px 0'>No anomalies detected.</div>",
                unsafe_allow_html=True,
            )

        # ── Ingestion Faults ────────────────────────────────────────────────
        # Malformed/corrupted DATA (missing fields, bad timestamps, dropped
        # packets, wrong-unit-scale values) caught by mqtt_stream.
        # validate_telemetry() -- a structurally different concern from the
        # Anomaly Log above, which only ever sees well-formed readings and
        # judges whether the value itself is physically worrying.
        with st.expander(f"🚧 Ingestion Faults ({len(_faults)})", expanded=False):
            if _faults:
                _fault_recent = list(reversed(_faults[-50:]))
                for _f in _fault_recent:
                    _fsev = _f.get("severity", "warning")
                    _fc   = "#fc8181" if _fsev == "critical" else "#f6ad55"
                    _fts  = str(_f.get("ts", ""))[:19].replace("T", " ")
                    st.markdown(
                        f"<div style='background:{_fc}11;border-left:3px solid {_fc};"
                        f"border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:12px'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                        f"<span style='color:{_fc};font-weight:700'>{_f.get('kind', 'UNKNOWN')}</span>"
                        f"<span style='color:#a0aec0;font-size:11px'>{_fts}</span>"
                        f"</div>"
                        f"<div style='color:#a0aec0'>{_f.get('cell_id', '')} — {_f.get('detail', '')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                _df_fault_exp = _pd_lm.DataFrame(_faults)
                _fault_csv    = _df_fault_exp.to_csv(index=False).encode()
                st.download_button(
                    "Export ingestion fault log CSV", data=_fault_csv,
                    file_name=f"ingestion_faults_{_replay_cell}.csv", mime="text/csv",
                    key="lm_export_faults",
                )
            else:
                st.markdown(
                    "<div style='color:#a0aec0;font-size:13px;padding:8px 0'>"
                    "No ingestion faults detected. This panel exercises real fault-detection logic "
                    "(missing/out-of-order timestamps, dropped-packet gaps, implausible unit-scale "
                    "values) against whatever replay traffic is flowing — it will only show entries "
                    "when the underlying data actually trips one of those checks."
                    "</div>",
                    unsafe_allow_html=True,
                )

    _telemetry_fragment()
