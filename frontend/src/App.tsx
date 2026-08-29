import { useState } from "react";
import { FlaskConical, BarChart3, Target, Leaf, UploadCloud, Radio } from "lucide-react";
import Login from "./components/Login";
import FleetSummaryView from "./components/FleetSummaryView";
import ElectrochemicalWorkbench from "./components/ElectrochemicalWorkbench";
import ActionCenterView from "./components/ActionCenterView";
import PassportCircularityView from "./components/PassportCircularityView";
import UniversalIngestionView from "./components/UniversalIngestionView";
import LiveMonitorView from "./components/LiveMonitorView";
import { clearToken, getToken } from "./api";
import type { LoginResponse } from "./types";

type Tab = "workbench" | "fleet" | "actions" | "passport" | "ingest" | "monitor";

const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
  { key: "workbench", label: "Diagnostic Workbench", icon: <FlaskConical size={15} /> },
  { key: "fleet", label: "Fleet Analytics", icon: <BarChart3 size={15} /> },
  { key: "actions", label: "Action Center", icon: <Target size={15} /> },
  { key: "passport", label: "Passport & Circularity", icon: <Leaf size={15} /> },
  { key: "ingest", label: "Cycler Ingestion", icon: <UploadCloud size={15} /> },
  { key: "monitor", label: "Live Telemetry", icon: <Radio size={15} /> },
];

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
    <div>
      {/* Top Header */}
      <div className="topbar">
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <h1>Battery Intelligence</h1>
          <span className="version">v2.0</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
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
      <p className="subtitle" style={{ marginBottom: 20 }}>
        Track cell health, predict remaining useful life, stream live BMS data,
        and generate EU-compliant battery passports — all from one place.
      </p>

      {/* Navigation Tabs */}
      <nav className="tabs">
        {tabs.map((t) => (
          <button
            key={t.key}
            className={tab === t.key ? "active" : ""}
            onClick={() => setTab(t.key)}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
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
