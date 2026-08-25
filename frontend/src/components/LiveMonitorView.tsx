import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { processStreamingSample } from "../api";

interface TelemetryPoint {
  time: string;
  voltage: number;
  current: number;
  temperature: number;
  anomalyScore: number;
}

export default function LiveMonitorView() {
  const [streaming, setStreaming] = useState<boolean>(true);
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const [anomalyLogs, setAnomalyLogs] = useState<any[]>([]);

  useEffect(() => {
    if (!streaming) return;

    const interval = setInterval(async () => {
      const now = new Date();
      const timeStr = now.toTimeString().split(" ")[0];

      // Simulated continuous sensor values
      const baseV = 3.82 - 0.002 * (data.length % 50);
      const noiseV = (Math.random() - 0.5) * 0.01;
      const v = Number((baseV + noiseV).toFixed(3));
      const i = Number((-2.0 + (Math.random() - 0.5) * 0.1).toFixed(2));
      const temp = Number((26.5 + (Math.random() - 0.5) * 0.4).toFixed(1));

      try {
        const res = await processStreamingSample({
          cell_id: "B0005",
          voltage_v: v,
          current_a: i,
          temperature_c: temp,
        });

        const newPoint: TelemetryPoint = {
          time: timeStr,
          voltage: v,
          current: i,
          temperature: temp,
          anomalyScore: res.mahalanobis_score,
        };

        setData((prev) => [...prev.slice(-30), newPoint]);

        if (res.anomalies && res.anomalies.length > 0) {
          setAnomalyLogs((prev) => [
            { time: timeStr, cell_id: "B0005", ...res.anomalies[0], severity: res.severity },
            ...prev.slice(0, 15),
          ]);
        }
      } catch (err) {
        console.error(err);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [streaming, data.length]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: 20, margin: 0 }}>High-Frequency BMS Live Telemetry Monitor</h1>
          <p className="subtitle" style={{ margin: 0, marginTop: 4 }}>
            Sub-10ms continuous streaming anomaly detection (CUSUM, Mahalanobis Distance, IEC 62619 TRP)
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span className="badge badge-healthy">
            {streaming ? "● Streaming Active (1.2s)" : "Paused"}
          </span>
          <button
            onClick={() => setStreaming(!streaming)}
            style={{
              padding: "6px 14px",
              background: streaming ? "transparent" : "var(--c-accent)",
              border: "1px solid var(--c-border)",
              color: streaming ? "var(--c-text)" : "#0e1117",
            }}
          >
            {streaming ? "Pause Stream" : "Resume Stream"}
          </button>
        </div>
      </div>

      {/* KPI Stats */}
      <div className="kpi-grid">
        <div className="card">
          <div className="kpi-label">Cell Voltage (Latest)</div>
          <div className="kpi-value" style={{ color: "var(--c-healthy)" }}>
            {data.length > 0 ? `${data[data.length - 1].voltage} V` : "--"}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Pack Bus Monitored</div>
        </div>
        <div className="card">
          <div className="kpi-label">Cell Current</div>
          <div className="kpi-value" style={{ color: "var(--c-accent)" }}>
            {data.length > 0 ? `${data[data.length - 1].current} A` : "--"}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Discharge Rate</div>
        </div>
        <div className="card">
          <div className="kpi-label">Surface Temperature</div>
          <div className="kpi-value" style={{ color: "var(--c-high)" }}>
            {data.length > 0 ? `${data[data.length - 1].temperature} °C` : "--"}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Thermocouple T1</div>
        </div>
        <div className="card">
          <div className="kpi-label">Mahalanobis Distance</div>
          <div className="kpi-value">
            {data.length > 0 ? `${data[data.length - 1].anomalyScore}` : "0.0"}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Threshold: 3.0</div>
        </div>
      </div>

      {/* Real-time Streaming Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card">
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 10 }}>Live Terminal Voltage & Current Stream</div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="time" stroke="#8896a8" fontSize={10} />
                <YAxis yAxisId="left" domain={[3.6, 4.0]} stroke="#68d391" fontSize={10} unit="V" />
                <YAxis yAxisId="right" orientation="right" domain={[-3, 0]} stroke="#63b3ed" fontSize={10} unit="A" />
                <Tooltip contentStyle={{ backgroundColor: "#1e2a38", border: "1px solid #2d3748", borderRadius: 6 }} />
                <Line yAxisId="left" type="monotone" dataKey="voltage" stroke="#68d391" strokeWidth={2} dot={false} name="Voltage (V)" />
                <Line yAxisId="right" type="monotone" dataKey="current" stroke="#63b3ed" strokeWidth={2} dot={false} name="Current (A)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 10 }}>Continuous Mahalanobis Anomaly Z-Score</div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="time" stroke="#8896a8" fontSize={10} />
                <YAxis domain={[0, 4.0]} stroke="#8896a8" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: "#1e2a38", border: "1px solid #2d3748", borderRadius: 6 }} />
                <Line type="monotone" dataKey="anomalyScore" stroke="#f6ad55" strokeWidth={2} dot={false} name="Z-Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Streaming Anomaly Trigger Log */}
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "12px 16px", background: "#17202c", fontWeight: 700, fontSize: 13, borderBottom: "1px solid var(--c-border)" }}>
          Real-Time Streaming Anomaly Log (IEC 62619 & Statistical Change-Points)
        </div>
        {anomalyLogs.length === 0 ? (
          <div style={{ padding: "16px", textAlign: "center", color: "var(--c-muted)", fontSize: 13 }}>
            ✓ Continuous stream nominal — no threshold violations or CUSUM drift detected.
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <tbody>
              {anomalyLogs.map((log, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--c-border)" }}>
                  <td style={{ padding: "10px 16px", width: 90, color: "var(--c-muted)" }}>{log.time}</td>
                  <td style={{ padding: "10px 16px", width: 80, fontWeight: 700 }}>{log.cell_id}</td>
                  <td style={{ padding: "10px 16px" }}>
                    <span className={`badge ${log.severity === "CRITICAL" ? "badge-critical" : "badge-high"}`}>
                      {log.type}
                    </span>
                    <span style={{ marginLeft: 10, color: "var(--c-text)" }}>{log.message}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
