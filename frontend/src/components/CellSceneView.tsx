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

      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 12, alignItems: "start" }}>
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
      </div>

      <div className="kpi-grid" style={{ marginTop: 18 }}>
        {spec && cards.map((part) => (
          <div className="card" key={part.id} style={{ borderLeft: `2px solid ${spec.theme.provenanceColors[part.provenance] ?? "#718096"}` }}>
            <div className="kpi-label">{part.label}</div>
            <div className="kpi-value" style={{ fontSize: 17 }}>
              {partValue(part)}
            </div>
            <div className="kpi-sub">
              {(part.provenance || "not measured").toUpperCase()}
              {!part.available && part.reason ? ` — ${part.reason}` : ""}
            </div>
          </div>
        ))}
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
