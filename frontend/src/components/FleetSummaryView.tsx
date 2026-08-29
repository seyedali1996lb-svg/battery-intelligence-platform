import { useEffect, useState } from "react";
import { getFleetSummary, getFleetAlerts, ApiError } from "../api";
import type { FleetSummary, FleetAlerts } from "../types";

/** Map SOH % → a color token name derived from battery state-of-charge */
function sohColor(soh: number): string {
  if (soh >= 80) return "var(--c-healthy)";
  if (soh >= 60) return "var(--c-degraded)";
  return "var(--c-critical)";
}

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
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load fleet data right now."));
  }, []);

  if (error) return <div className="error-text">{error}</div>;
  if (!summary || !alerts) return <p className="subtitle" style={{ marginTop: 20 }}>Loading fleet data…</p>;

  return (
    <div>
      <h2>Fleet Overview</h2>
      <div className="kpi-grid">
        <div className="card">
          <div className="kpi-label">Total Cells</div>
          <div className="kpi-value">{summary.total_cells}</div>
          <div className="kpi-sub">In active monitoring</div>
        </div>
        <div className="card">
          <div className="kpi-label">Healthy</div>
          <div className="kpi-value" style={{ color: "var(--c-healthy)" }}>{summary.n_healthy}</div>
          <div className="kpi-sub">Above 80% SOH threshold</div>
        </div>
        <div className="card">
          <div className="kpi-label">Degrading</div>
          <div className="kpi-value" style={{ color: "var(--c-degraded)" }}>{summary.n_degrading}</div>
          <div className="kpi-sub">Needs closer attention</div>
        </div>
        <div className="card">
          <div className="kpi-label">End of Life</div>
          <div className="kpi-value" style={{ color: "var(--c-critical)" }}>{summary.n_eol}</div>
          <div className="kpi-sub">Below 80% SOH floor</div>
        </div>
        <div className="card">
          <div className="kpi-label">Average SOH</div>
          <div className="kpi-value" style={{ color: sohColor(summary.fleet_soh_mean) }}>
            {summary.fleet_soh_mean.toFixed(1)}%
          </div>
          <div className="kpi-sub">Range {summary.fleet_soh_min.toFixed(0)}–{summary.fleet_soh_max.toFixed(0)}%</div>
        </div>
        <div className="card">
          <div className="kpi-label">12-Month CAPEX</div>
          <div className="kpi-value">${summary.capex_estimate_12m_usd.toLocaleString()}</div>
          <div className="kpi-sub">Estimated replacement cost</div>
        </div>
      </div>

      <h2>Active Alerts ({alerts.total_alerts})</h2>
      <div className="card">
        {alerts.alerts.length === 0 && (
          <p className="subtitle" style={{ margin: 0 }}>
            All clear — no alerts right now. Your fleet is looking healthy.
          </p>
        )}
        {alerts.alerts.map((a, i) => (
          <div className="alert-row" key={i}>
            <span className={`badge badge-${a.severity}`}>{a.severity}</span>
            <div>
              <strong>{a.cell_id}</strong> — {a.title}
              <div className="subtitle" style={{ margin: "2px 0 0", fontSize: 12 }}>{a.body}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
