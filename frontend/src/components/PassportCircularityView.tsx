import { useState, useEffect } from "react";
import { listCells, getDynamicLCA, getVerifiablePassport, getSecondLifeBids } from "../api";
import type { DynamicLCA, SecondLifeBid } from "../types";

export default function PassportCircularityView() {
  const [cells, setCells] = useState<string[]>([]);
  const [selectedCell, setSelectedCell] = useState<string>("");
  const [region, setRegion] = useState<string>("EU_AVG");
  const [lca, setLca] = useState<DynamicLCA | null>(null);
  const [passport, setPassport] = useState<any>(null);
  const [bids, setBids] = useState<SecondLifeBid[]>([]);
  const [loading, setLoading] = useState(false);
  const [viewJson, setViewJson] = useState(false);

  useEffect(() => {
    listCells().then((res) => {
      setCells(res.cells);
      if (res.cells.length > 0) {
        setSelectedCell(res.cells[0]);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedCell) return;
    setLoading(true);
    Promise.all([
      getDynamicLCA(selectedCell, region),
      getVerifiablePassport(selectedCell),
      getSecondLifeBids(selectedCell),
    ])
      .then(([l, p, b]) => {
        setLca(l);
        setPassport(p);
        setBids(b);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedCell, region]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: 20, margin: 0 }}>EU Battery Passport & Circularity Marketplace</h1>
          <p className="subtitle" style={{ margin: 0, marginTop: 4 }}>
            Regulation (EU) 2023/1542 digital product passports, dynamic use-phase LCA carbon footprint, and second-life buyer matching
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {loading && <span className="badge badge-accent">Loading...</span>}
          <select
            value={selectedCell}
            onChange={(e) => setSelectedCell(e.target.value)}
            style={{ width: 130 }}
          >
            {cells.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <select
            value={region}
            onChange={(e) => setRegion(e.target.value)}
            style={{ width: 140 }}
          >
            <option value="EU_AVG">Grid: EU Average</option>
            <option value="GERMANY">Grid: Germany</option>
            <option value="FRANCE">Grid: France (Nuclear)</option>
            <option value="NORWAY">Grid: Norway (Hydro)</option>
            <option value="US_AVG">Grid: US Average</option>
          </select>
        </div>
      </div>

      {/* LCA Carbon Breakdown */}
      {lca && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontWeight: 700, fontSize: 15 }}>Cradle-to-Grave Dynamic Carbon Footprint (LCA)</span>
            <span className="badge badge-healthy">Net: {lca.net_lifecycle_co2_kg} kg CO₂e</span>
          </div>

          <div className="kpi-grid">
            <div className="card" style={{ background: "#141c24" }}>
              <div className="kpi-label">Manufacturing Emissions</div>
              <div className="kpi-value" style={{ color: "var(--c-text)" }}>{lca.mfg_co2_kg} kg</div>
              <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Cathode & Cell Assembly</div>
            </div>
            <div className="card" style={{ background: "#141c24" }}>
              <div className="kpi-label">Use-Phase Charging Losses</div>
              <div className="kpi-value" style={{ color: "var(--c-high)" }}>{lca.use_phase_co2_kg} kg</div>
              <div style={{ fontSize: 11, color: "var(--c-muted)" }}>At {lca.grid_intensity_g_kwh} g CO₂e/kWh</div>
            </div>
            <div className="card" style={{ background: "#141c24" }}>
              <div className="kpi-label">Recycling Avoided Credit</div>
              <div className="kpi-value" style={{ color: "var(--c-healthy)" }}>{lca.eol_recycling_credit_kg} kg</div>
              <div style={{ fontSize: 11, color: "var(--c-muted)" }}>Closed-Loop Recovery</div>
            </div>
            <div className="card" style={{ background: "#141c24" }}>
              <div className="kpi-label">Intensity Per kWh Delivered</div>
              <div className="kpi-value" style={{ color: "var(--c-accent)" }}>{lca.carbon_intensity_kg_per_kwh_delivered}</div>
              <div style={{ fontSize: 11, color: "var(--c-muted)" }}>kg CO₂e / kWh throughput</div>
            </div>
          </div>
        </div>
      )}

      {/* Grid: Second-Life Buyer Marketplace & Verifiable Credential */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 16 }}>
        {/* Second-Life Bids */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontWeight: 700, fontSize: 14 }}>Automated Second-Life Marketplace Bids</span>
            <span className="badge badge-accent">Live Valuation</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {bids.map((b) => (
              <div
                key={b.buyer_id}
                style={{
                  background: "#141c24",
                  border: "1px solid var(--c-border)",
                  borderRadius: 6,
                  padding: 12,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}>{b.buyer_name}</div>
                  <div style={{ fontSize: 11, color: "var(--c-muted)", marginTop: 2 }}>{b.application} ({b.region})</div>
                  <div style={{ fontSize: 11, marginTop: 4 }}>
                    Fit Score: <span style={{ fontWeight: 700, color: "var(--c-healthy)" }}>{b.fit_score}%</span>
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 18, fontWeight: 800, color: "var(--c-healthy)" }}>${b.total_bid_usd}</div>
                  <div style={{ fontSize: 11, color: "var(--c-muted)" }}>${b.offer_per_kwh_usd}/kWh</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* W3C Verifiable Credential */}
        <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontWeight: 700, fontSize: 14 }}>W3C Verifiable Credential Passport</span>
              <button
                style={{ padding: "4px 8px", fontSize: 11, background: "transparent", border: "1px solid var(--c-border)", color: "var(--c-muted)" }}
                onClick={() => setViewJson(!viewJson)}
              >
                {viewJson ? "Show Summary" : "View JSON-LD"}
              </button>
            </div>

            {passport && !viewJson && (
              <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                <div><span style={{ color: "var(--c-muted)" }}>Subject ID:</span> <span style={{ fontWeight: 600 }}>{passport.credentialSubject?.id}</span></div>
                <div><span style={{ color: "var(--c-muted)" }}>Standard:</span> <span style={{ fontWeight: 600 }}>{passport.credentialSubject?.complianceStandard}</span></div>
                <div><span style={{ color: "var(--c-muted)" }}>Issuer DID:</span> <span style={{ fontWeight: 600 }}>{passport.issuer}</span></div>
                <div><span style={{ color: "var(--c-muted)" }}>End of Life Status:</span> <span style={{ fontWeight: 700, color: "var(--c-healthy)" }}>{passport.credentialSubject?.endOfLifeStatus}</span></div>
                <div><span style={{ color: "var(--c-muted)" }}>Ed25519 Proof:</span> <code style={{ fontSize: 10, color: "var(--c-accent)" }}>{passport.proof?.verificationMethod}</code></div>
              </div>
            )}

            {passport && viewJson && (
              <pre
                style={{
                  background: "#0e1117",
                  padding: 10,
                  borderRadius: 6,
                  fontSize: 10,
                  overflowX: "auto",
                  maxHeight: 220,
                  color: "var(--c-muted)",
                }}
              >
                {JSON.stringify(passport, null, 2)}
              </pre>
            )}
          </div>

          <div style={{ marginTop: 14, paddingTop: 10, borderTop: "1px solid var(--c-border)", fontSize: 11, color: "var(--c-muted)" }}>
            Cryptographically anchored to EU Battery Passport Registry standards.
          </div>
        </div>
      </div>
    </div>
  );
}
