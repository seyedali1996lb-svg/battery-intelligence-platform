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
import type { CellSceneSpec, FilmModel, PartId, PhysicalModel, ScenePart, Series } from "./types.ts";

const LABELS: Record<PartId, string> = {
  can: "Cell casing",
  wrap: "Heat-shrink jacket",
  cap: "Top cap assembly",
  vent: "Vent disc",
  crimp: "Crimp rim",
  terminal_pos: "Positive terminal",
  terminal_neg: "Negative terminal",
  tab_pos: "Positive tab",
  tab_neg: "Negative tab",
  mandrel: "Winding mandrel",
  cathode_sheet: "Cathode coating",
  anode_sheet: "Anode coating",
  separator: "Separator",
  electrolyte: "Electrolyte",
  particles: "Active material particles",
  sei_film: "SEI film on the anode",
};

/** The plain-language line a card leads with — the producer carries the same map. */
const TITLES: Record<PartId, string> = {
  can: "The steel case",
  wrap: "Heat-shrink jacket — the printed skin of the cell",
  cap: "The sealed top plate",
  vent: "The vent — the cell's safety valve",
  crimp: "Crimp rim — where the can was rolled shut",
  terminal_pos: "The positive button",
  terminal_neg: "The flat negative end",
  tab_pos: "Positive tab — the cathode's wire",
  tab_neg: "Negative tab — the anode's wire",
  mandrel: "Mandrel — the steel core the roll is wound around",
  cathode_sheet: "Cathode — the lithium's source",
  anode_sheet: "Anode — where the lithium waits",
  separator: "Separator — the plastic film keeping the electrodes apart",
  electrolyte: "Electrolyte — the liquid the lithium travels in",
  sei_film: "SEI film — the anode's skin, where fade first shows",
  particles: "Active material — the working powder on the sheets",
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
  /**
   * The document's declared dimensions: the 18650 block by default, omitted
   * entirely (`"absent"`), or overridden field by field.
   */
  physical?: "declared" | "absent" | Partial<PhysicalModel>;
}

/**
 * The same 18650 figures `src/cell_scene.py` declares, so the fixture's scene
 * is the scene a real document produces.
 */
export function makePhysical(formFactor: "cylindrical" | "prismatic" | "unknown" = "cylindrical"): PhysicalModel {
  const stackMm = {
    copperFoil: 0.010,
    anodeCoating: 0.070,
    separator: 0.020,
    cathodeCoating: 0.060,
    aluminiumFoil: 0.015,
  };
  const pitchMm = Object.values(stackMm).reduce((total, mm) => total + mm, 0);
  const unitMm = 65.0;
  const wallMm = 0.25;
  const clearanceMm = 0.2;
  const mandrelDiameterMm = 4.0;
  const envelopeRadiusMm = 18.4 / 2 - wallMm - clearanceMm;
  const turns = Math.floor((envelopeRadiusMm - mandrelDiameterMm / 2) / pitchMm);
  const drawnOuterRadiusMm = mandrelDiameterMm / 2 + turns * pitchMm;
  let lengthMm = 0;
  for (let i = 0; i < turns; i++) {
    lengthMm += 2 * Math.PI * (mandrelDiameterMm / 2 + (i + 0.5) * pitchMm);
  }
  return {
    formFactor,
    format: "18650 — 18.4 mm x 65.0 mm, the format's published envelope",
    unitsMmPerCellUnit: unitMm,
    // Same rule as the producer: a cylindrical cell declares its top
    // assembly, a prismatic one carries null and the renderer falls back.
    // Only the envelope this cell actually has is declared — the producer
    // nulls the other, and a fixture that kept both would stop matching the
    // real document's shape.
    prismatic:
      formFactor === "prismatic"
        ? { widthMm: 20.5, thicknessMm: 5.4, heightMm: 64.0, wallMm: 0.4 }
        : null,
    cylindrical:
      formFactor === "prismatic"
        ? null
        : {
            diameterMm: 18.4,
            heightMm: 65.0,
            wallMm,
            rollClearanceMm: clearanceMm,
            mandrelDiameterMm,
          },
    topAssembly:
      formFactor === "prismatic"
        ? null
        : {
            capThicknessMm: 0.8,
            capBossDiameterMm: 8.0,
            capBossHeightMm: 0.5,
            ventDiameterMm: 4.5,
            terminalDiameterMm: 5.5,
            terminalStudHeightMm: 0.8,
            crimpHeightMm: 0.6,
            wrapThicknessMm: 0.15,
            wrapTopSkipMm: 1.2,
            wrapBottomSkipMm: 1.2,
          },
    roll: {
      stackMm,
      pitchMm,
      turns,
      mandrelDiameterMm,
      envelopeDiameterMm: 2 * envelopeRadiusMm,
      drawnOuterDiameterMm: 2 * drawnOuterRadiusMm,
      drawnRibbonsMm: {
        anode: stackMm.copperFoil + stackMm.anodeCoating,
        separator: stackMm.separator,
        cathode: stackMm.aluminiumFoil + stackMm.cathodeCoating,
      },
      electrodeLengthM: lengthMm / 1000,
    } as PhysicalModel["roll"],
    provenance: {
      diameterMm: "format-standard",
      heightMm: "format-standard",
      topAssembly: "typical for the format, not a datasheet figure for this cell",
      stackMm: "typical for a 18650-class cell, not measured for this cell",
      turns: "derived",
      electrodeLengthM: "derived",
    },
    note: "The winding is drawn to these millimetres.",
    schematic: ["the particle cloud's count and size"],
    // The film, in nanometres, as the producer derives it — with the drawn band
    // and the magnification it implies. Kept on the fixture so the renderer's
    // band, its endpoints and the geometry are all exercised against the same
    // shape of document the app produces.
    film: makeFilm(formFactor),
  };
}

/** The fixture's film block: a stated chain, and a stated magnification. */
export function makeFilm(
  formFactor: "cylindrical" | "prismatic" | "unknown" = "cylindrical",
  initialNm = 5.0,
  nmPerPctLli = 80.0,
): FilmModel {
  const identified = formFactor !== "prismatic";
  return {
    modelledAs: "a compact Li₂CO₃ film on the anode",
    available: identified,
    identified,
    assumptions: {
      formulaUnit: "Li₂CO₃",
      lithiumPerFormulaUnit: 2.0,
      molarMassGPerMol: 73.89,
      densityGPerCm3: 2.11,
      coatedWidthMm: 58.0,
      anodeFaces: 2.0,
      initialNm,
    },
    provenance: { formulaUnit: "assumed — Li₂CO₃ is the phase most often reported" },
    derivation: identified
      ? {
          capacity0Ah: 2.0,
          electrodeLengthM: 1.2714,
          coatedWidthMm: 58.0,
          anodeAreaCm2: 1474.824,
          molarVolumeCm3PerMol: 35.019,
          nmPerPctLli,
          maxPctLli: 20.0,
          maxNm: initialNm + 20 * nmPerPctLli,
          // The top of the scale: a share of initial capacity, not this record's
          // own maximum, so two cells' films stay comparable by eye.
          displayMaxNm: initialNm + 30 * nmPerPctLli,
          displayMaxPctLli: 30.0,
          chain: "nm = (Q₀·(LLI%/100)·3600 / F) / (Li per Li₂CO₃) · (M/ρ) / coated area · 1e7",
          unitMm: 65.0,
        }
      : null,
    display: {
      drawnMinMm: 0.0975,
      drawnMaxMm: 0.312,
      drawnMinNm: 97500.0,
      drawnMaxNm: 312000.0,
      magnificationAtMaxX: identified ? 195.0 : null,
      growthMagnificationX: identified ? 198.1 : null,
      note: "The drawn SEI layer is a magnification, not a thickness.",
    },
    reason: identified
      ? null
      : "The thickness is withheld because this cell's data does not identify the lithium-inventory channel.",
    note: "Read the nanometres as an estimate whose assumptions bound it from above.",
  };
}

export function makeSpec(options: FixtureOptions = {}): CellSceneSpec {
  const physical =
    options.physical === "absent"
      ? null
      : options.physical && typeof options.physical === "object"
        ? { ...makePhysical(options.formFactor), ...options.physical }
        : makePhysical(options.formFactor);
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
  // The same fitted term as a thickness: the fixture applies the producer's
  // nm-per-percent scale, so the series the renderer reads is a nanometre one.
  const filmInitialNm = 5.0;
  const filmNmPerPct = 80.0;
  const seiThicknessNm = seiPct.map((v) =>
    v === null ? null : filmInitialNm + v * filmNmPerPct,
  );
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
      title: TITLES[id],
      value: id === "sei_film" ? seiThicknessNm[n - 1] : id === "particles" ? lamPct[n - 1] : last,
      unit:
        id === "sei_film"
          ? "nm (derived from the fitted lithium-inventory loss)"
          : id === "particles"
            ? "% of initial capacity (fitted)"
            : "",
      provenance: id === "sei_film" ? "derived" : id === "particles" ? "fitted" : values ? "measured" : "",
      meaning: `${LABELS[id]} — what this scene claims, in words.`,
      law: "declared law",
      series:
        isUnavailable ? null : id === "sei_film" ? seiThicknessNm : id === "particles" ? lamPct : values,
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
      seiThicknessNm,
    },
    physical,
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
        from: "series.seiThicknessNm",
        unit: "nm (derived from the fitted lithium-inventory loss)",
        displayMin: filmInitialNm,
        displayMax: filmInitialNm + 30 * filmNmPerPct,
        note: "The film's drawn thickness is the fitted √n lithium-inventory loss, in nanometres.",
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
