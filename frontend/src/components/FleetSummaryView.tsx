import { useEffect, useState } from "react";
import { getFleetSummary, getFleetAlerts, ApiError } from "../api";
import type { FleetSummary, FleetAlerts } from "../types";

export default function FleetSummaryView() {
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [alerts, setAlerts] = useState<FleetAlerts | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getFleetSummary(), getFleetAlerts()])
      .then(([s, a]) => {
        setSummary(s);
        setAlerts(a);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load fleet data."));
  }, []);

  if (error) return <div className="error-text">{error}</div>;
  if (!summary || !alerts) return <p className="subtitle">Loading fleet summary…</p>;

  return (
    <div>
      <h2>Fleet KPIs</h2>
      <div className="kpi-grid">
        <div className="card">
          <div className="kpi-value">{summary.total_cells}</div>
          <div className="kpi-label">Total cells</div>
        </div>
        <div className="card">
          <div className="kpi-value" style={{ color: "var(--c-healthy)" }}>{summary.n_healthy}</div>
          <div className="kpi-label">Healthy</div>
        </div>
        <div className="card">
          <div className="kpi-value" style={{ color: "var(--c-high)" }}>{summary.n_degrading}</div>
          <div className="kpi-label">Degrading</div>
        </div>
        <div className="card">
          <div className="kpi-value" style={{ color: "var(--c-critical)" }}>{summary.n_eol}</div>
          <div className="kpi-label">End of life</div>
        </div>
        <div className="card">
          <div className="kpi-value">{summary.fleet_soh_mean.toFixed(1)}%</div>
          <div className="kpi-label">Avg SOH ({summary.fleet_soh_min.toFixed(0)}–{summary.fleet_soh_max.toFixed(0)}%)</div>
        </div>
        <div className="card">
          <div className="kpi-value">${summary.capex_estimate_12m_usd.toLocaleString()}</div>
          <div className="kpi-label">12mo CAPEX estimate</div>
        </div>
      </div>

      <h2>Active Alerts ({alerts.total_alerts})</h2>
      <div className="card">
        {alerts.alerts.length === 0 && <p className="subtitle" style={{ margin: 0 }}>No active alerts.</p>}
        {alerts.alerts.map((a, i) => (
          <div className="alert-row" key={i}>
            <span className={`badge badge-${a.severity}`}>{a.severity}</span>
            <div>
              <strong>{a.cell_id}</strong> — {a.title}
              <div className="subtitle" style={{ margin: "2px 0 0" }}>{a.body}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
