import { useState } from "react";
import { FlaskConical, BarChart3, Target, Leaf, UploadCloud, Radio, Box } from "lucide-react";
import Login from "./components/Login";
import FleetSummaryView from "./components/FleetSummaryView";
import ElectrochemicalWorkbench from "./components/ElectrochemicalWorkbench";
import ActionCenterView from "./components/ActionCenterView";
import PassportCircularityView from "./components/PassportCircularityView";
import UniversalIngestionView from "./components/UniversalIngestionView";
import LiveMonitorView from "./components/LiveMonitorView";
import CellSceneView from "./components/CellSceneView";
import { clearToken, getToken } from "./api";
import type { LoginResponse } from "./types";

type Tab = "workbench" | "scene" | "fleet" | "actions" | "passport" | "ingest" | "monitor";

const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
  { key: "workbench", label: "Diagnostic Workbench", icon: <FlaskConical size={15} /> },
  { key: "scene", label: "Cell 3D", icon: <Box size={15} /> },
  { key: "fleet", label: "Fleet Analytics", icon: <BarChart3 size={15} /> },
  { key: "actions", label: "Action Center", icon: <Target size={15} /> },
  { key: "passport", label: "Passport & Circularity", icon: <Leaf size={15} /> },
  { key: "ingest", label: "Cycler Ingestion", icon: <UploadCloud size={15} /> },
  { key: "monitor", label: "Live Telemetry", icon: <Radio size={15} /> },
];

const TAB_KEYS = new Set<string>(tabs.map((tab) => tab.key));

/**
 * The tab a shared link names, or the default.
 *
 * The parameter is validated rather than cast: a hand-edited `?tab=banana`
 * must fall back to the workbench, not render an empty pane — and an unknown
 * value is dropped from the URL by `writeTab` on the next click, so the address
 * bar never advertises a tab this app does not have.
 */
function tabFromUrl(): Tab {
  const value = new URLSearchParams(window.location.search).get("tab");
  return value !== null && TAB_KEYS.has(value) ? (value as Tab) : "workbench";
}

/** Mirror the active tab into the URL without dropping anyone else's keys. */
function writeTab(tab: Tab): void {
  try {
    const params = new URLSearchParams(window.location.search);
    // `workbench` is the default tab; carrying it would make every plain link
    // look parameterised, so the default is written as its absence.
    if (tab === "workbench") params.delete("tab");
    else params.set("tab", tab);
    const query = params.toString();
    window.history.replaceState(
      null,
      "",
      window.location.pathname + (query ? `?${query}` : "") + window.location.hash,
    );
  } catch {
    // A sandboxed frame may refuse history writes; the tab still switches.
  }
}

function App() {
  const [user, setUser] = useState<LoginResponse | null>(null);
  const [tab, setTab] = useState<Tab>(tabFromUrl);
  const [loggedOut, setLoggedOut] = useState(!getToken());

  /** Switch tabs and keep the address bar in step, in one place. */
  const goToTab = (next: Tab): void => {
    setTab(next);
    writeTab(next);
  };

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
            onClick={() => goToTab(t.key)}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </nav>

      {/* Active View Container */}
      {tab === "workbench" && <ElectrochemicalWorkbench />}
      {tab === "scene" && <CellSceneView />}
      {tab === "fleet" && <FleetSummaryView />}
      {tab === "actions" && <ActionCenterView />}
      {tab === "passport" && <PassportCircularityView />}
      {tab === "ingest" && <UniversalIngestionView />}
      {tab === "monitor" && <LiveMonitorView />}
    </div>
  );
}

export default App;
