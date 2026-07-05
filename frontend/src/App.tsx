import { useState } from "react";
import Login from "./components/Login";
import FleetSummaryView from "./components/FleetSummaryView";
import CellDetailView from "./components/CellDetailView";
import { clearToken, getToken } from "./api";
import type { LoginResponse } from "./types";

type Tab = "fleet" | "cell";

function App() {
  const [user, setUser] = useState<LoginResponse | null>(null);
  const [tab, setTab] = useState<Tab>("fleet");
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
      <div className="topbar">
        <h1>Battery Intelligence</h1>
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
        Read-only view onto the shared reference-cell fleet — same data every organization
        sees in the Streamlit app today. Org-scoped uploaded fleets aren't exposed here yet.
      </p>

      <nav className="tabs">
        <button className={tab === "fleet" ? "active" : ""} onClick={() => setTab("fleet")}>
          Fleet Summary
        </button>
        <button className={tab === "cell" ? "active" : ""} onClick={() => setTab("cell")}>
          Cell Detail
        </button>
      </nav>

      {tab === "fleet" ? <FleetSummaryView /> : <CellDetailView />}
    </div>
  );
}

export default App;
