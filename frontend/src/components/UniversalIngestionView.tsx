import { useState } from "react";
import { detectCyclerFormat } from "../api";
import type { CyclerDetection } from "../types";

export default function UniversalIngestionView() {
  const [detection, setDetection] = useState<CyclerDetection | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("Arbin");
  const [loading, setLoading] = useState(false);

  const presets: Record<string, string[]> = {
    Arbin: [
      "Test_Time(s)", "Step_Time(s)", "Cycle_Index", "Current(A)", "Voltage(V)",
      "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)", "Temperature(C)"
    ],
    BioLogic: [
      "time/s", "control/V/mA", "Ecell/V", "<I>/mA", "Capacity/mA.h", "Energy/W.h", "half cycle"
    ],
    Maccor: [
      "Cycle", "Step", "TestTime(Sec)", "Current", "Voltage", "Cap(Ah)", "Watt-hr", "Temp 1"
    ],
    Neware: [
      "Time", "Step Name", "Current(mA)", "Voltage(V)", "Cap(mAh)", "Energy(mWh)", "Temp(℃)"
    ],
    Novonix: [
      "Step Number", "Time (h)", "Current (A)", "Voltage (V)", "Capacity (Ah)", "Temperature (°C)"
    ],
  };

  const handleTestPreset = async (presetName: string) => {
    setSelectedPreset(presetName);
    setLoading(true);
    try {
      const res = await detectCyclerFormat(presets[presetName]);
      setDetection(res);
    } catch (e: any) {
      alert(`Detection failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: 20, margin: 0 }}>Cycler Ingestion</h1>
          <p className="subtitle" style={{ margin: 0, marginTop: 4 }}>
            Upload data from any test bench — we detect the format, normalize units, and prepare it for degradation modeling
          </p>
        </div>
        {loading && <span className="badge badge-energy">Analyzing…</span>}
      </div>


      {/* Preset Cycler Format Buttons */}
      <div className="card">
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>Test Bench Hardware Presets</div>
        <p className="subtitle" style={{ fontSize: 12, margin: "0 0 12px 0" }}>
          Pick a cycler format to preview how we auto-map column headers and scale units:
        </p>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Object.keys(presets).map((p) => (
            <button
              key={p}
              onClick={() => handleTestPreset(p)}
              style={{
                background: selectedPreset === p ? "var(--c-accent)" : "transparent",
                color: selectedPreset === p ? "#080B09" : "var(--c-text)",
                border: "1px solid var(--c-border)",
                padding: "8px 16px",
                fontSize: 13,
              }}
            >
              {p} Cycler
            </button>
          ))}
        </div>
      </div>

      {/* Detection Results */}
      {detection && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <span style={{ fontWeight: 800, fontSize: 16 }}>Hardware Detected: </span>
              <span style={{ fontWeight: 800, color: "var(--c-healthy)", fontSize: 16 }}>{detection.hardware}</span>
            </div>
            <span className="badge badge-healthy">{(detection.confidence * 100).toFixed(0)}% Mapping Match</span>
          </div>

          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8 }}>Mapped Schema Columns & Unit Conversions</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, marginBottom: 16 }}>
            <thead>
              <tr style={{ background: "#17202c", textAlign: "left", borderBottom: "1px solid var(--c-border)" }}>
                <th style={{ padding: "8px 12px" }}>Target Schema Field</th>
                <th style={{ padding: "8px 12px" }}>Raw Source Column</th>
                <th style={{ padding: "8px 12px" }}>Auto-Scale Multiplier</th>
                <th style={{ padding: "8px 12px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(detection.mapped_columns).map(([target, src]) => (
                <tr key={target} style={{ borderBottom: "1px solid var(--c-border)" }}>
                  <td style={{ padding: "8px 12px", fontWeight: 700, color: "var(--c-accent)" }}>{target}</td>
                  <td style={{ padding: "8px 12px" }}><code>{src}</code></td>
                  <td style={{ padding: "8px 12px" }}>
                    {detection.unit_scales[target] ? `${detection.unit_scales[target]}x` : "1.0x"}
                  </td>
                  <td style={{ padding: "8px 12px", color: "var(--c-healthy)", fontWeight: 700 }}>✓ Mapped</td>
                </tr>
              ))}
            </tbody>
          </table>

          <button style={{ width: "100%", background: "var(--c-healthy)", color: "#080B09", fontWeight: 700 }}>
            Ingest & Start Training
          </button>
        </div>
      )}
    </div>
  );
}
