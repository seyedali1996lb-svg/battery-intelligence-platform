import { useState, useEffect } from "react";
import { listActions, triageAction, dispatchAction } from "../api";
import type { ActionTicket } from "../types";

export default function ActionCenterView() {
  const [actions, setActions] = useState<ActionTicket[]>([]);
  const [filterSeverity, setFilterSeverity] = useState<string>("ALL");
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
  const [dispatchResult, setDispatchResult] = useState<string | null>(null);

  const fetchActions = () => {
    listActions()
      .then((data) => setActions(data))
      .catch(console.error);
  };


  useEffect(() => {
    fetchActions();
  }, []);

  const handleTriage = async (id: string, newStatus: string) => {
    try {
      await triageAction(id, newStatus, "Battery Engineer");
      fetchActions();
    } catch (e: any) {
      alert(`Triage failed: ${e.message}`);
    }
  };

  const handleDispatch = async (id: string, targetSystem: string) => {
    try {
      const res = await dispatchAction(id, targetSystem);
      setDispatchResult(`[${res.dispatch_reference}] ${res.summary}`);
      fetchActions();
      setTimeout(() => setDispatchResult(null), 6000);
    } catch (e: any) {
      alert(`Dispatch failed: ${e.message}`);
    }
  };

  const filtered = actions.filter((a) => {
    if (filterSeverity !== "ALL" && a.severity !== filterSeverity) return false;
    if (filterStatus !== "ALL" && a.status !== filterStatus) return false;
    return true;
  });

  const getSeverityBadge = (sev: string) => {
    switch (sev) {
      case "CRITICAL":
        return <span className="badge badge-critical">Critical Risk</span>;
      case "HIGH":
        return <span className="badge badge-high">High Priority</span>;
      case "MEDIUM":
        return <span className="badge badge-medium">Medium</span>;
      default:
        return <span className="badge" style={{ background: "#4a556844", color: "#cbd5e0" }}>Routine</span>;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header & Stats */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: 20, margin: 0 }}>Unified Operations Action Center</h1>
          <p className="subtitle" style={{ margin: 0, marginTop: 4 }}>
            Proactive triage, SLA resolution tracking, and one-click CMMS / warranty / circularity dispatch
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <select
            value={filterSeverity}
            onChange={(e) => setFilterSeverity(e.target.value)}
            style={{ width: 140 }}
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            style={{ width: 130 }}
          >
            <option value="ALL">All Statuses</option>
            <option value="NEW">New</option>
            <option value="TRIAGED">Triaged</option>
            <option value="IN_PROGRESS">In Progress</option>
            <option value="DISPATCHED">Dispatched</option>
          </select>
        </div>
      </div>

      {dispatchResult && (
        <div className="card" style={{ background: "rgba(104, 211, 145, 0.1)", borderColor: "var(--c-healthy)", color: "var(--c-healthy)", fontSize: 13, fontWeight: 600 }}>
          ✓ {dispatchResult}
        </div>
      )}

      {/* KPI Row */}
      <div className="kpi-grid">
        <div className="card">
          <div className="kpi-label">Open Critical SLA Tickets</div>
          <div className="kpi-value" style={{ color: "var(--c-critical)" }}>
            {actions.filter((a) => a.severity === "CRITICAL" && a.status !== "RESOLVED").length}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Requires 12h resolution</div>
        </div>
        <div className="card">
          <div className="kpi-label">Warranty Horizon Breaches</div>
          <div className="kpi-value" style={{ color: "var(--c-high)" }}>
            {actions.filter((a) => a.category === "WARRANTY").length}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Claims pending dispatch</div>
        </div>
        <div className="card">
          <div className="kpi-label">Dispatched Workflows</div>
          <div className="kpi-value" style={{ color: "var(--c-accent)" }}>
            {actions.filter((a) => a.status === "DISPATCHED").length}
          </div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>CMMS / ERP / Circularity</div>
        </div>
        <div className="card">
          <div className="kpi-label">Total Action Tickets</div>
          <div className="kpi-value">{actions.length}</div>
          <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Active Fleet Assets</div>
        </div>
      </div>

      {/* Action Ticket List */}
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#17202c", textAlign: "left", borderBottom: "1px solid var(--c-border)" }}>
              <th style={{ padding: "12px 16px" }}>Severity</th>
              <th style={{ padding: "12px 16px" }}>Cell ID</th>
              <th style={{ padding: "12px 16px" }}>Title & Diagnostic Description</th>
              <th style={{ padding: "12px 16px" }}>SOH</th>
              <th style={{ padding: "12px 16px" }}>Status</th>
              <th style={{ padding: "12px 16px", textAlign: "right" }}>Workflow Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((a) => (
              <tr key={a.id} style={{ borderBottom: "1px solid var(--c-border)" }}>
                <td style={{ padding: "12px 16px" }}>{getSeverityBadge(a.severity)}</td>
                <td style={{ padding: "12px 16px", fontWeight: 800 }}>{a.cell_id}</td>
                <td style={{ padding: "12px 16px" }}>
                  <div style={{ fontWeight: 700, color: "var(--c-text)" }}>{a.title}</div>
                  <div style={{ fontSize: 12, color: "var(--c-muted)", marginTop: 2 }}>{a.description}</div>
                </td>
                <td style={{ padding: "12px 16px", fontWeight: 700, color: a.soh_pct >= 80 ? "var(--c-healthy)" : "var(--c-critical)" }}>
                  {a.soh_pct}%
                </td>
                <td style={{ padding: "12px 16px" }}>
                  <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: a.status === "DISPATCHED" ? "var(--c-healthy)" : "var(--c-accent)" }}>
                    {a.status}
                  </span>
                  {a.dispatched_to && (
                    <div style={{ fontSize: 10, color: "var(--c-muted)" }}>→ {a.dispatched_to}</div>
                  )}
                </td>
                <td style={{ padding: "12px 16px", textAlign: "right" }}>
                  <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                    {a.status === "NEW" && (
                      <button
                        style={{ padding: "4px 10px", fontSize: 11, background: "transparent", border: "1px solid var(--c-accent)", color: "var(--c-accent)" }}
                        onClick={() => handleTriage(a.id, "IN_PROGRESS")}
                      >
                        Start Triage
                      </button>
                    )}
                    {a.category === "DEGRADATION" && (
                      <button
                        style={{ padding: "4px 10px", fontSize: 11 }}
                        onClick={() => handleDispatch(a.id, "CMMS")}
                      >
                        Create CMMS Ticket
                      </button>
                    )}
                    {a.category === "WARRANTY" && (
                      <button
                        style={{ padding: "4px 10px", fontSize: 11, background: "var(--c-high)" }}
                        onClick={() => handleDispatch(a.id, "WARRANTY")}
                      >
                        File Warranty Claim
                      </button>
                    )}
                    {a.category === "CIRCULARITY" && (
                      <button
                        style={{ padding: "4px 10px", fontSize: 11, background: "var(--c-healthy)" }}
                        onClick={() => handleDispatch(a.id, "CIRCULARITY")}
                      >
                        Match Buyer Bid
                      </button>
                    )}
                    {a.category === "COMPLIANCE" && (
                      <button
                        style={{ padding: "4px 10px", fontSize: 11, background: "var(--c-accent)" }}
                        onClick={() => handleDispatch(a.id, "PASSPORT")}
                      >
                        Generate Passport
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
