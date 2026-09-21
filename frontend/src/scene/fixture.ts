/**
 * Synthetic specs for the renderer's headless tests.
 *
 * Not a test file (Node only picks up `*.test.ts`), and deliberately *not* a
 * copy of the Python fixture in tests/test_cell_scene.py: it is the renderer's
 * own idea of a spec, so the two halves can disagree in a test rather than only
 * in a browser. The real cross-language check is `spec.test.ts`, which reads a
 * committed spec produced by the Python builder.
 */

import { ANATOMY_PART_IDS } from "./types.ts";
import type { CellSceneSpec, PartId, ScenePart, Series } from "./types.ts";

const LABELS: Record<PartId, string> = {
  can: "Cell casing",
  cap: "Top cap assembly",
  vent: "Vent disc",
  terminal_pos: "Positive terminal",
  terminal_neg: "Negative terminal",
  tab_pos: "Positive tab",
  tab_neg: "Negative tab",
  cathode_sheet: "Cathode coating",
  anode_sheet: "Anode coating",
  separator: "Separator",
  electrolyte: "Electrolyte",
  particles: "Active material particles",
  sei_film: "SEI film on the anode",
};

export const THEME = {
  background: "#0b1120",
  panel: "#111827",
  text: "#e2e8f0",
  muted: "#94a3b8",
  accent: "#63b3ed",
  grid: "#1f2937",
  metal: "#cbd5e0",
  anodeColor: "#b08d57",
  cathodeColor: "#7b6ba8",
  separatorColor: "#d9e2ec",
  electrolyteColor: "#63b3ed",
  seiColor: "#b794f4",
  sohBands: [
    { min: 90, max: null, color: "#48bb78", label: "Healthy" },
    { min: 80, max: 90, color: "#f6e05e", label: "Degrading" },
    { min: 0, max: 80, color: "#fc8181", label: "End of Life" },
  ],
  temperatureBands: [
    { min: 0, max: 30, color: "#4299e1", label: "cool" },
    { min: 30, max: 45, color: "#ecc94b", label: "warm" },
    { min: 45, max: null, color: "#e53e3e", label: "hot" },
  ],
  provenanceColors: { measured: "#48bb78", derived: "#63b3ed", fitted: "#b794f4", projected: "#f6ad55", "": "#718096" },
  sopFloorPct: 70,
  powerGaugeMinPct: 40,
  powerGaugeMaxPct: 110,
};

export interface FixtureOptions {
  n?: number;
  splitIdentified?: boolean;
  fadePerCycle?: number;
  seiAtStart?: number;
  seiAtEnd?: number;
  lamAtStart?: number;
  lamAtEnd?: number;
  formFactor?: "cylindrical" | "prismatic" | "unknown";
  unavailable?: PartId[];
  projection?: "none" | "refused" | "available";
  emphasis?: "sei" | "particles" | "neutral";
}

export function makeSpec(options: FixtureOptions = {}): CellSceneSpec {
  const n = options.n ?? 120;
  const fadePerCycle = options.fadePerCycle ?? 0.1;
  const series = (fn: (i: number) => number): Series =>
    Array.from({ length: n }, (_ignored, i) => fn(i));

  const cycles = series((i) => i + 1);
  const sohPct = series((i) => 100 - fadePerCycle * i);
  const capacityAh = series((i) => 2 * (100 - fadePerCycle * i) / 100);
  const resistanceOhm = series((i) => 0.045 + 0.0002 * i);
  const resistanceNormalized = series((i) => 1 + 0.012 * i);
  const temperatureC = series((i) => 28 + 12 * Math.sin(i / 20));
  const sopPct = series((i) => 100 - 0.35 * i);
  const seiPct = series(
    (i) => (options.seiAtStart ?? 2.5) + ((options.seiAtEnd ?? 8) - (options.seiAtStart ?? 2.5)) * (i / (n - 1)),
  );
  const lamPct = series(
    (i) => (options.lamAtStart ?? 0.4) + ((options.lamAtEnd ?? 3.0) - (options.lamAtStart ?? 0.4)) * (i / (n - 1)),
  );
  const fadeModelPct = seiPct.map((v, i) => (v ?? 0) + (lamPct[i] ?? 0));
  const seiSharePct = seiPct.map((v, i) => {
    const total = (v ?? 0) + (lamPct[i] ?? 0);
    return total > 0 ? ((v ?? 0) / total) * 100 : null;
  });

  const unavailable = new Set(options.unavailable ?? []);
  const seriesByPart: Partial<Record<PartId, Series>> = {
    can: temperatureC,
    terminal_pos: resistanceOhm,
    terminal_neg: resistanceOhm,
    tab_pos: resistanceNormalized,
    tab_neg: sopPct,
    cathode_sheet: series((i) => 140 + 0.5 * i),
    anode_sheet: series((i) => -4 + 0.01 * i),
  };

  const parts: ScenePart[] = ANATOMY_PART_IDS.map((id) => {
    const isUnavailable = unavailable.has(id);
    const values = seriesByPart[id] ?? null;
    const last = values ? values[values.length - 1] : null;
    return {
      id,
      label: LABELS[id],
      value: id === "sei_film" ? seiPct[n - 1] : id === "particles" ? lamPct[n - 1] : last,
      unit: id === "sei_film" || id === "particles" ? "% of initial capacity (fitted)" : "",
      provenance: id === "sei_film" || id === "particles" ? "fitted" : values ? "measured" : "",
      meaning: `${LABELS[id]} — what this scene claims, in words.`,
      law: "declared law",
      series: isUnavailable ? null : id === "sei_film" ? seiPct : id === "particles" ? lamPct : values,
      available: !isUnavailable,
      unavailableReason: isUnavailable
        ? "no measurement in the cycle-summary data separates this part"
        : null,
    };
  });

  const projection =
    options.projection === "available"
      ? {
          available: true,
          reason: null,
          cycles: series((i) => n + i + 1),
          sohPct: series((i) => 100 - fadePerCycle * (n + i) - 0.002 * i * i),
          sohQ10Pct: series((i) => 100 - fadePerCycle * (n + i) - 0.002 * i * i - 1.5),
          sohQ90Pct: series((i) => 100 - fadePerCycle * (n + i) - 0.002 * i * i + 1.5),
          kind: "projected" as const,
          modelLabel: "Hierarchical partial-pooling (shrunk log-fade line)",
        }
      : {
          available: false,
          reason:
            options.projection === "refused"
              ? "The platform's per-cell forecast routing refuses to forecast this cell."
              : "No hierarchical forecast fit is available for this cell.",
        };

  return {
    schemaVersion: 1,
    cell: {
      id: "B0005",
      source: "NASA",
      chemistry: "LiCoO₂ (lithium cobalt oxide), 18650 cylindrical",
      formFactor: options.formFactor ?? "cylindrical",
      formFactorNote: "18650 cylindrical",
      provenance: "measured",
      dqdvApplicable: true,
    },
    series: {
      cycles,
      sohPct,
      capacityAh,
      resistanceOhm,
      resistanceNormalized,
      temperatureC,
      sopPct,
      fadeRate30cy: series((i) => 0.004 + 0.0002 * i),
      seiPct,
      lamPct,
      seiSharePct,
      fadeModelPct,
    },
    record: {
      firstCycle: 1,
      lastCycle: n,
      nCycles: n,
      nRecordedRows: n,
      sampled: false,
      lastSohPct: sohPct[n - 1],
    },
    physics: {
      law: "SOH(n)/SOH₀ = 1 − β_sei·√n − β_lam·n",
      seiTerm: "β_sei·√n",
      lamTerm: "β_lam·n",
      available: true,
      splitIdentified: options.splitIdentified ?? true,
      splitReason: options.splitIdentified === false ? "This cell's data cannot separate the two channels." : null,
      betaSei: 0.0018,
      betaLam: 0.00012,
      fitR2: 0.97,
      reason: null,
      source: "batlab.features.physics_calibration.fit_two_term_fade",
      measuredFadePct: fadePerCycle * (n - 1),
      modelTotalFadePct: fadeModelPct[fadeModelPct.length - 1],
    },
    mechanism: {
      physics: { key: "lli", label: "LLI-dominated", gatePassed: true, gate: 0.3, fitR2: 0.97 },
      ml: { verdict: "LLI", confidenceLabel: "High", body: "coulombic-efficiency trend", source: "knowledge_graph (shared verdict)" },
      agree: true,
      agreementIsEvidence: true,
      note: "Both sides name the same mechanism.",
      emphasis: options.emphasis ?? "sei",
      emphasisSource: "physics",
    },
    knee: { detected: true, cycle: 90, sohAtKnee: 88.5, confidence: 0.7, phase: "knee" },
    projection,
    parts,
    geometryScales: {
      fade_model: {
        target: "capacityLoss",
        renderAs: "stateGauge",
        from: "series.fadeModelPct",
        unit: "% of initial capacity (fitted)",
        displayMin: 0,
        displayMax: null,
        note: "The drawn capacity loss is the fitted two-term model's TOTAL fade.",
      },
      sei_film: {
        target: "shellThickness",
        from: "series.seiPct",
        unit: "% of initial capacity (fitted)",
        displayMin: 0,
        displayMax: 20,
        note: "The film's drawn thickness is the fitted √n lithium-inventory loss.",
      },
      particles: {
        target: "lostFraction",
        from: "series.lamPct",
        unit: "% of initial capacity (fitted)",
        displayMin: 0,
        displayMax: null,
        note: "The fraction of particles drawn as lost tracks the fitted linear term.",
      },
      terminal_pos: {
        target: "crossSection",
        from: "series.resistanceOhm",
        unit: "Ω",
        displayMin: null,
        displayMax: null,
        note: "Drawn terminal cross-section narrows as measured DC resistance rises.",
      },
    },
    disclosures: [
      "This scene is a schematic of a cell's construction, not a metrology model or a CAD drawing.",
      "Anatomical proportions are shown by default. Data-scaled geometry is opt-in.",
      "Every part card is tagged measured, derived, fitted or projected.",
    ],
    theme: THEME,
  };
}
