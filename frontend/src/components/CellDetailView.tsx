import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { listCells, getCellLatest, getCellHistory, getCellRul, ApiError } from "../api";
import type { CellLatest, CellHistory, RULResult } from "../types";

export default function CellDetailView() {
  const [cells, setCells] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [latest, setLatest] = useState<CellLatest | null>(null);
  const [history, setHistory] = useState<CellHistory | null>(null);
  const [rul, setRul] = useState<RULResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listCells()
      .then((r) => {
        setCells(r.cells);
        if (r.cells.length > 0) setSelected(r.cells[0]);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load cell list."));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setError(null);
    Promise.all([getCellLatest(selected), getCellHistory(selected, 300), getCellRul(selected)])
      .then(([l, h, r]) => {
        setLatest(l);
        setHistory(h);
        setRul(r);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load cell detail."));
  }, [selected]);

  return (
    <div>
      <h2>Cell Detail</h2>
      <select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ maxWidth: 240, marginBottom: 20 }}>
        {cells.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>

      {error && <div className="error-text">{error}</div>}

      {latest && (
        <div className="kpi-grid" style={{ marginBottom: 20 }}>
          <div className="card">
            <div className="kpi-value">{latest.soh_pct.toFixed(1)}%</div>
            <div className="kpi-label">SOH — {latest.status}</div>
          </div>
          <div className="card">
            <div className="kpi-value">{latest.cycle_number}</div>
            <div className="kpi-label">Cycle</div>
          </div>
          <div className="card">
            <div className="kpi-value">
              {rul?.rul_reliable && rul.rul_pred !== null ? Math.round(rul.rul_pred) : "—"}
            </div>
            <div className="kpi-label">
              RUL (cycles){rul && !rul.rul_reliable ? " · calibrating" : ""}
            </div>
          </div>
          <div className="card">
            <div className="kpi-value">
              {latest.capacity_ah !== null ? latest.capacity_ah.toFixed(2) : "—"}
            </div>
            <div className="kpi-label">Capacity (Ah)</div>
          </div>
        </div>
      )}

      {history && history.history.length > 0 && (
        <div className="card">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history.history}>
              <CartesianGrid stroke="#2d3748" />
              <XAxis dataKey="cycle_number" stroke="#8896a8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#8896a8" tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "#1e2a38", border: "1px solid #2d3748", fontSize: 12 }}
                labelStyle={{ color: "#e2e8f0" }}
              />
              <Line type="monotone" dataKey="soh_pct" stroke="#63b3ed" dot={false} name="SOH %" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
