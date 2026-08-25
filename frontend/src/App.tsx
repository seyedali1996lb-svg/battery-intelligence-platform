import { useState } from "react";
import Login from "./components/Login";
import FleetSummaryView from "./components/FleetSummaryView";
import ElectrochemicalWorkbench from "./components/ElectrochemicalWorkbench";
import ActionCenterView from "./components/ActionCenterView";
import PassportCircularityView from "./components/PassportCircularityView";
import UniversalIngestionView from "./components/UniversalIngestionView";
import LiveMonitorView from "./components/LiveMonitorView";
import { clearToken, getToken } from "./api";
import type { LoginResponse } from "./types";

type Tab = "fleet" | "workbench" | "actions" | "passport" | "ingest" | "monitor";

function App() {
  const [user, setUser] = useState<LoginResponse | null>(null);
  const [tab, setTab] = useState<Tab>("workbench");
  const [loggedOut, setLoggedOut] = useState(!getToken());

  if (loggedOut || !user) {
    return (
      <Login
        onLoggedIn={(u) => {
          setUser(u);
          setLoggedOut(false);
        }}
      />
    );
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>
      {/* Top Header */}
      <div className="topbar">
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1>Battery Intelligence Platform</h1>
          <span style={{ fontSize: 12, color: "var(--c-accent)", fontWeight: 700 }}>v2.0 Enterprise</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="org">{user.org_name} · {user.display_name} ({user.role})</span>
          <button
            className="signout"
            onClick={() => {
              clearToken();
              setUser(null);
              setLoggedOut(true);
            }}
          >
            Sign out
          </button>
        </div>
      </div>
      <p className="subtitle">
        Validated electrochemical analytics, Leave-Cell-Out RUL, real-time BMS streaming, and EU 2023/1542 digital product passports.
      </p>

      {/* Navigation Tabs */}
      <nav className="tabs" style={{ flexWrap: "wrap", gap: 6, marginBottom: 24 }}>
        <button className={tab === "workbench" ? "active" : ""} onClick={() => setTab("workbench")}>
          ⚡ Cell Diagnostic Workbench
        </button>
        <button className={tab === "fleet" ? "active" : ""} onClick={() => setTab("fleet")}>
          📊 Fleet Analytics
        </button>
        <button className={tab === "actions" ? "active" : ""} onClick={() => setTab("actions")}>
          🎯 Operations Action Center
        </button>
        <button className={tab === "passport" ? "active" : ""} onClick={() => setTab("passport")}>
          🌱 Passport & Circularity
        </button>
        <button className={tab === "ingest" ? "active" : ""} onClick={() => setTab("ingest")}>
          📥 Universal Cycler Ingestion
        </button>
        <button className={tab === "monitor" ? "active" : ""} onClick={() => setTab("monitor")}>
          📡 Live Telemetry Monitor
        </button>
      </nav>

      {/* Active View Container */}
      {tab === "workbench" && <ElectrochemicalWorkbench />}
      {tab === "fleet" && <FleetSummaryView />}
      {tab === "actions" && <ActionCenterView />}
      {tab === "passport" && <PassportCircularityView />}
      {tab === "ingest" && <UniversalIngestionView />}
      {tab === "monitor" && <LiveMonitorView />}
    </div>
  );
}

export default App;
