import { useEffect, useRef, useState } from "react";
import { getCellScene, listCells, ApiError } from "../api";
import type { CellSceneHandle, CellSceneSpec, FrameState, PartReading } from "../scene";

/**
 * The scene engine, loaded on demand.
 *
 * It carries three.js (~578 kB minified), which is most of this SPA's weight
 * and is needed by exactly one tab. A static import would make every visitor pay
 * for it; a dynamic one loads it when this tab is opened and lets the bundler
 * split it out. The type imports above stay static — they cost nothing at
 * runtime and keep the document's shape visible in this file.
 */
type SceneModule = typeof import("../scene");

/**
 * The React host of the 3D cell scene.
 *
 * The SPA is the third host of the same engine (the Streamlit page and the
 * standalone page are the others), and this component is deliberately the
 * thinnest of the three: it fetches a `CellSceneSpec` from the API, hands it to
 * `mountCellScene`, and re-renders the panels around it from the frame state the
 * engine reports. It contains no geometry, no colour and no battery vocabulary
 * of its own — if this file and the Streamlit page disagree about a cell, one of
 * them is reading a different document, which is exactly the failure the shared
 * spec exists to make impossible.
 */
const PART_DOSSIERS: Record<string, { latinTitle: string; subsystem: string; material: string; degradation: string }> = {
  can: {
    latinTitle: "THORAX METALLICUS",
    subsystem: "Deep-Drawn Can & Structural Pressure Shell",
    material: "Nickel-plated cold-rolled steel (0.25 mm wall)",
    degradation: "Mechanical deformation, internal pressure bulging, atmospheric corrosion.",
  },
  wrap: {
    latinTitle: "TUNICA CONTRACTA",
    subsystem: "Heat-Shrink Electrical Isolation Jacket",
    material: "Polyethylene terephthalate (PET) film (0.15 mm)",
    degradation: "Thermal abrasion, chemical puncture, dielectric breakdown.",
  },
  cap: {
    latinTitle: "GALEA TERMINALIS",
    subsystem: "Lathed Cap Plate, Boss & Crimp Groove",
    material: "Aluminium / Nickel-plated steel formed assembly",
    degradation: "Mechanical stress relaxation of crimp seal, micro-fissuring under thermal cycling.",
  },
  vent: {
    latinTitle: "VALVULA SALUTIS",
    subsystem: "Laser-Scored Overpressure Safety Vent Disc",
    material: "Embossed aluminum rupture foil (4.5 mm diameter)",
    degradation: "Fatigue from cyclic gas accumulation; engineered rupture at 1.5–2.0 MPa.",
  },
  crimp: {
    latinTitle: "CORONA COMPRESSA",
    subsystem: "Mechanical Crimp Seal & Radial Compression Bead",
    material: "Rolled steel rim over polypropylene (PP) gasket",
    degradation: "Polymer creep under thermal loads, micro-leakage of volatile carbonate solvent.",
  },
  terminal_pos: {
    latinTitle: "POLUS POSITIVUS",
    subsystem: "Positive Current Collector Stud & Button",
    material: "Cold-forged nickel-plated copper/steel stud",
    degradation: "Surface oxidation, contact resistance rise, ultrasonic weld degradation.",
  },
  terminal_neg: {
    latinTitle: "POLUS NEGATIVUS",
    subsystem: "Negative Cell Floor Current Collector Contact",
    material: "Direct steel can base (18.4 mm OD)",
    degradation: "Fretting wear, interfacial contact oxidation and impedance rise.",
  },
  tab_pos: {
    latinTitle: "LIGAMENTUM ALUMINII",
    subsystem: "Positive Electrode Current Lead & Ultrasonic Weld",
    material: "High-purity aluminium foil ribbon (0.1 mm)",
    degradation: "Ultrasonic weld fatigue, localized Joule heating, vibration detachment.",
  },
  tab_neg: {
    latinTitle: "LIGAMENTUM CUPRI",
    subsystem: "Negative Electrode Current Lead & Bottom Spot Weld",
    material: "High-conductivity annealed copper ribbon",
    degradation: "Localized overcurrent stress, micro-cracking at sharp bend radii.",
  },
  mandrel: {
    latinTitle: "AXIS WINDING",
    subsystem: "Removable Steel Winding Core & Jelly-Roll Datum",
    material: "Hardened steel mandrel (4.0 mm diameter, withdrawn after winding)",
    degradation: "None — it is not electrochemically active; concentricity loss shows up as uneven electrode tension.",
  },
  cathode_sheet: {
    latinTitle: "STRATUM CATHODICUM",
    subsystem: "Lithiated Transition Metal Intercalation Matrix",
    material: "Active oxide (e.g. LiCoO2 / NMC) on 15 µm aluminium foil",
    degradation: "Transition metal dissolution, micro-cracking, lattice distortion, impedance rise.",
  },
  anode_sheet: {
    latinTitle: "STRATUM ANODICUM",
    subsystem: "Graphite / Silicon Intercalation Host & Current Collector",
    material: "MCMB graphite / Si blend on 10 µm copper foil",
    degradation: "Lithium plating under low temp/fast charge, particle pulverization, exfoliation.",
  },
  separator: {
    latinTitle: "SEPTUM SEPARANS",
    subsystem: "Microporous Polymeric Electronic Barrier",
    material: "Trilayer PE/PP ceramic-coated microporous film (20 µm)",
    degradation: "Pore clogging by decomposed species, dendrite penetration, thermal shrinkage.",
  },
  electrolyte: {
    latinTitle: "LIQUIDUM CONDUCTOR",
    subsystem: "Non-Aqueous Lithium Salt & Alkyl Carbonate Solution",
    material: "1.0–1.2 M LiPF6 in EC/DMC/EMC organic solvent",
    degradation: "Parasitic solvent oxidation, salt consumption, HF formation, gassing.",
  },
  particles: {
    latinTitle: "PARTICULAE MOBILES",
    subsystem: "Active Insertion Material Volume & Cycling Kinetics",
    material: "Intercalation micro-crystallites (2–15 µm particles)",
    degradation: "Loss of active material (LAM) via particle isolation, lattice strain, crack networks.",
  },
  sei_film: {
    latinTitle: "MEMBRANA SEI",
    subsystem: "Solid Electrolyte Interphase Passivation Layer",
    material: "Li2CO3, LiF, lithium alkyl carbonates (compact inner + porous outer)",
    degradation: "Continuous parasitic reduction consumes cyclable lithium inventory (LLI).",
  },
};

export default function CellSceneView() {
  const [cells, setCells] = useState<string[]>([]);
  const [cellId, setCellId] = useState<string>("");
  const [spec, setSpec] = useState<CellSceneSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<FrameState | null>(null);
  const [exploded, setExploded] = useState(0);
  const [dataScaled, setDataScaled] = useState(false);
  const [annotations, setAnnotations] = useState(true);
  const [stripCasing, setStripCasing] = useState(false);
  const [playing, setPlaying] = useState(false);
  /**
   * The part the scene is showing a dossier for.
   *
   * Set from the engine's `onInspect`, which reports every change to the
   * effective selection — a hover in the canvas, a click on a badge, or a pick
   * in the quick list — so the panel and the scene cannot disagree about what
   * is being inspected.
   */
  const [inspected, setInspected] = useState<string | null>(null);
  /** The engine's part readings at the cursor; null until the scene is up. */
  const [readings, setReadings] = useState<PartReading[] | null>(null);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);
  const handleRef = useRef<CellSceneHandle | null>(null);
  const moduleRef = useRef<SceneModule | null>(null);

  useEffect(() => {
    listCells()
      .then((response) => {
        setCells(response.cells);
        setCellId((current) => current || response.cells[0] || "");
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't list cells."));
  }, []);

  useEffect(() => {
    if (!cellId) return;
    setError(null);
    getCellScene(cellId)
      .then(setSpec)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load this cell's scene."));
  }, [cellId]);

  // Mount on the document, and re-mount only when the *document* changes: the
  // view controls below go through handle.update, which never reloads the
  // canvas (a Streamlit-style full re-render would throw away the camera).
  useEffect(() => {
    const stage = stageRef.current;
    if (!spec || !stage) return;
    setReadings(null); // the previous cell's numbers must not survive a switch
    setInspected(null); // …and neither must the previous cell's inspection
    let disposed = false;
    let handle: CellSceneHandle | null = null;

    void (async () => {
      const sceneModule = moduleRef.current ?? (await import("../scene"));
      if (disposed) return;
      moduleRef.current = sceneModule;
      handleRef.current?.dispose();
      // `mount`, not `mountCellScene`: the wrapper refuses a document produced by
      // another schema version with a sentence, and turns a missing WebGL
      // context into an explanation, where the raw call would return null.
      const mounted = sceneModule.mount(stage, spec, {
        cursor: Math.max(0, spec.series.cycles.length - 1),
        onFrame: (state) => {
          setFrame(state);
          // The cards move with the scene: a card that kept showing today's
          // number while the cursor scrubbed back would be a second, quieter
          // reading of the same cell. `handleRef` is still empty during the
          // first frame, which the fallback below covers.
          setReadings(handleRef.current?.parts() ?? null);
        },
        // Playback stops at the end of the timeline on its own, so the button
        // has to be told — otherwise it sits on "Pause" over a still scene.
        onPlaybackEnd: () => setPlaying(false),
        // The dossier follows the scene, not the other way round: the engine
        // owns which part is selected (hover or pin), and this is it handing
        // that choice over.
        onInspect: setInspected,
      });
      if (!mounted.handle) {
        setError(mounted.error ?? "The 3D view could not start.");
        return;
      }
      handle = mounted.handle;
      handleRef.current = handle;
      setReadings(handle.parts());
      if (legendRef.current) sceneModule.renderLegend(legendRef.current, spec.theme);
    })();

    return () => {
      disposed = true;
      handle?.dispose();
      handleRef.current = null;
    };
  }, [spec]);

  useEffect(() => {
    if (!spec || !handleRef.current) return;
    handleRef.current.update(spec, { exploded, dataScaled, casing: stripCasing ? "hidden" : "translucent" });
  }, [spec, exploded, dataScaled, stripCasing]);

  useEffect(() => {
    handleRef.current?.setAnnotations(annotations);
  }, [annotations]);

  const cards: PartReading[] =
    readings ??
    (spec?.parts ?? []).map((part) => ({
      id: part.id,
      label: part.label,
      value: part.value,
      unit: part.unit,
      provenance: part.provenance,
      available: part.available,
      carried: false,
      meaning: part.meaning,
      reason: part.unavailableReason,
    }));

  const partValue = (part: PartReading): string => {
    if (part.value === null) return "—";
    const magnitude = Math.abs(part.value);
    const decimals = magnitude >= 100 ? 1 : magnitude >= 10 ? 2 : 3;
    return `${part.value.toFixed(decimals)}${part.unit ? ` ${part.unit}` : ""}` +
      (part.carried ? " (last measured)" : "");
  };

  const activePartId = inspected;
  const inspectedPart = cards.find((c) => c.id === activePartId) ?? null;
  const dossier = activePartId ? PART_DOSSIERS[activePartId] : null;

  if (error) return <div className="error-text">{error}</div>;

  return (
    <div>
      <h2>Cell 3D</h2>
      <p className="subtitle">
        One cell, its anatomy, and the measurement behind every part. Orbit with a drag, zoom with the
        wheel, hover a part to read what it is.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <select value={cellId} onChange={(event) => setCellId(event.target.value)}>
          {cells.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <span style={{ color: "var(--c-muted, #94a3b8)", fontSize: 12 }}>
          {frame?.soh !== null && frame?.soh !== undefined ? `${frame.soh.toFixed(1)}%` : "—"}{" "}
          {frame?.gaugeLabel}
          {frame?.cycle !== null && frame?.cycle !== undefined ? ` · cycle ${Math.round(frame.cycle)}` : ""}
          {frame?.projected ? ` · ${frame.projectionLabel ?? "projected"}` : " · measured"}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "220px minmax(340px, 1fr) 280px", gap: 14, alignItems: "start" }}>
        <div className="card">
          <div className="kpi-label">Controls</div>
          <label style={{ display: "block", fontSize: 12, marginBottom: 10 }}>
            Exploded view <span style={{ float: "right" }}>{Math.round(exploded * 100)}%</span>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(exploded * 100)}
              onChange={(event) => setExploded(Number(event.target.value) / 100)}
              style={{ width: "100%" }}
            />
          </label>
          <label style={{ display: "block", fontSize: 12, marginBottom: 10 }}>
            Life cursor{" "}
            <span style={{ float: "right" }}>
              {(frame?.cursor ?? 0) + 1} / {(frame?.measuredCount ?? 0) + (frame?.projectionLength ?? 0)}
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, (frame?.measuredCount ?? 1) + (frame?.projectionLength ?? 0) - 1)}
              value={frame?.cursor ?? 0}
              onChange={(event) => handleRef.current?.setCursor(Number(event.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
            <button
              className="btn-outline"
              onClick={() => {
                if (!handleRef.current) return;
                if (playing) {
                  handleRef.current.pause();
                  setPlaying(false);
                  return;
                }
                // play() answers false when there is no life to play, and the
                // label must not claim a playback that never started.
                setPlaying(handleRef.current.play() === true);
              }}
            >
              {playing ? "⏸ Pause" : "▶ Play life"}
            </button>
            <button
              className="btn-outline"
              onClick={() => {
                if (!spec) return;
                handleRef.current?.setCursor(Math.max(0, spec.series.cycles.length - 1));
              }}
            >
              Today
            </button>
            <button className="btn-outline" onClick={() => handleRef.current?.resetView()}>
              Reset view
            </button>
          </div>
          <label style={{ display: "flex", gap: 6, fontSize: 12, marginBottom: 6 }}>
            <input type="checkbox" checked={dataScaled} onChange={(event) => setDataScaled(event.target.checked)} />
            Data-scaled geometry
          </label>
          <label style={{ display: "flex", gap: 6, fontSize: 12, marginBottom: 6 }}>
            <input
              type="checkbox"
              checked={annotations}
              onChange={(event) => setAnnotations(event.target.checked)}
            />
            Part annotations
          </label>
          <label style={{ display: "flex", gap: 6, fontSize: 12 }}>
            <input
              type="checkbox"
              checked={stripCasing}
              onChange={(event) => setStripCasing(event.target.checked)}
            />
            Strip the casing
          </label>
          <div ref={legendRef} style={{ marginTop: 10 }} />
        </div>

        <div>
          <div
            ref={stageRef}
            style={{
              position: "relative",
              height: 560,
              border: "1px solid var(--c-border, #1f2937)",
              borderRadius: 10,
              overflow: "hidden",
            }}
          />
          <p style={{ fontSize: 11.5, color: "var(--c-muted, #94a3b8)", marginTop: 8 }}>
            {frame?.scaleNote ||
              "Anatomical proportions. Data-scaled geometry is opt-in, and prints the mapping it uses when on."}
          </p>
        </div>

        <div className="card" style={{ borderLeft: inspectedPart ? `3px solid ${spec?.theme.provenanceColors[inspectedPart.provenance] ?? "var(--c-accent, #63b3ed)"}` : "1px solid var(--c-border, #1f2937)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
            <span style={{ fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--c-muted, #94a3b8)", fontWeight: 700 }}>
              Inspected Component
            </span>
            {inspectedPart && (
              <button
                className="btn-outline"
                style={{ padding: "1px 6px", fontSize: 11, lineHeight: "1.2" }}
                onClick={() => handleRef.current?.inspect(null)}
                title="Clear inspection"
              >
                ✕
              </button>
            )}
          </div>

          {inspectedPart && dossier ? (
            <div>
              <div style={{ fontFamily: "serif", fontSize: 16, fontWeight: 700, color: "#f8fafc", letterSpacing: "0.02em" }}>
                {dossier.latinTitle}
              </div>
              <div style={{ fontSize: 11, color: "var(--c-accent, #63b3ed)", fontWeight: 600, marginBottom: 10 }}>
                // {inspectedPart.label.toUpperCase()}
              </div>

              <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "8px 10px", borderRadius: 6, marginBottom: 12, border: "1px solid rgba(51, 65, 85, 0.5)" }}>
                <div style={{ fontSize: 10, color: "var(--c-muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Current Reading
                </div>
                <div style={{ fontSize: 20, fontWeight: 700, fontVariantNumeric: "tabular-nums", color: "#f1f5f9" }}>
                  {partValue(inspectedPart)}
                </div>
                <div style={{ fontSize: 10.5, color: spec?.theme.provenanceColors[inspectedPart.provenance] ?? "#94a3b8", fontWeight: 600, marginTop: 2 }}>
                  {inspectedPart.provenance.toUpperCase()}
                  {inspectedPart.carried ? " · LAST MEASURED" : ""}
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 11.5 }}>
                <div>
                  <span style={{ color: "var(--c-muted, #94a3b8)", fontWeight: 600, fontSize: 10, display: "block", textTransform: "uppercase" }}>Subsystem</span>
                  <span style={{ color: "#e2e8f0" }}>{dossier.subsystem}</span>
                </div>
                <div>
                  <span style={{ color: "var(--c-muted, #94a3b8)", fontWeight: 600, fontSize: 10, display: "block", textTransform: "uppercase" }}>Material Spec</span>
                  <span style={{ color: "#cbd5e0" }}>{dossier.material}</span>
                </div>
                <div>
                  <span style={{ color: "var(--c-muted, #94a3b8)", fontWeight: 600, fontSize: 10, display: "block", textTransform: "uppercase" }}>Degradation Mode</span>
                  <span style={{ color: "#fca5a5" }}>{dossier.degradation}</span>
                </div>
                <div>
                  <span style={{ color: "var(--c-muted, #94a3b8)", fontWeight: 600, fontSize: 10, display: "block", textTransform: "uppercase" }}>Diagnostic Significance</span>
                  <span style={{ color: "var(--c-muted, #94a3b8)" }}>{inspectedPart.meaning || inspectedPart.reason || "No specific note."}</span>
                </div>
              </div>

              <div style={{ marginTop: 14 }}>
                <button
                  className="btn-outline"
                  style={{ width: "100%", fontSize: 11, padding: "5px 8px" }}
                  onClick={() => handleRef.current?.inspect(inspectedPart.id)}
                >
                  🎯 Frame Camera
                </button>
              </div>
            </div>
          ) : (
            <div style={{ color: "var(--c-muted, #94a3b8)", fontSize: 12 }}>
              <p style={{ marginTop: 4, marginBottom: 10 }}>
                Hover or click any component in 3D or in the annotations to inspect its technical codex specifications.
              </p>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 6, color: "var(--c-muted, #94a3b8)" }}>
                Quick Inspect
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {cards.slice(0, 7).map((p) => (
                  <button
                    key={p.id}
                    className="btn-outline"
                    style={{ textAlign: "left", fontSize: 11, padding: "4px 8px", display: "flex", justifyContent: "space-between" }}
                    onClick={() => handleRef.current?.inspect(p.id)}
                  >
                    <span>{p.label}</span>
                    <span style={{ color: spec?.theme.provenanceColors[p.provenance] ?? "inherit", fontSize: 10 }}>{p.provenance}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {spec && (
        <details style={{ marginTop: 16 }}>
          <summary style={{ cursor: "pointer", color: "var(--c-muted, #94a3b8)", fontSize: 12 }}>
            What this view does and does not claim
          </summary>
          <ul style={{ color: "var(--c-muted, #94a3b8)", fontSize: 12 }}>
            {spec.disclosures.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
