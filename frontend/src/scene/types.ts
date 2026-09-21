/**
 * The renderer's half of the CellSceneSpec contract.
 *
 * These types mirror `docs/cell_scene.schema.json`, which is validated on the
 * Python side by `tests/test_cell_scene.py`. Everything the renderer paints
 * comes from this document — it is the only thing a host has to produce, and
 * the renderer knows nothing else about any host (no Streamlit, no React, no
 * API). See `spec.test.ts` for the assertions that keep the two halves honest.
 */

/** Must equal `SCENE_SCHEMA_VERSION` in `src/cell_scene.py`. Checked at runtime
 *  by the renderer (a mismatched host gets a clear refusal, not a broken scene)
 *  and asserted by `spec.test.ts` against the Python constant read from source. */
export const SCENE_SCHEMA_VERSION = 1;

export type Provenance = "measured" | "derived" | "fitted" | "projected" | "";

/** The canonical anatomy parts: exactly one mesh and exactly one card each. */
export const ANATOMY_PART_IDS = [
  "can",
  "cap",
  "vent",
  "terminal_pos",
  "terminal_neg",
  "tab_pos",
  "tab_neg",
  "cathode_sheet",
  "anode_sheet",
  "separator",
  "electrolyte",
  "particles",
  "sei_film",
] as const;

export type PartId = (typeof ANATOMY_PART_IDS)[number];

/** A per-cycle series: a null entry means "no measurement at this cycle". */
export type Series = (number | null)[];

export interface ScenePart {
  id: PartId;
  label: string;
  value: number | null;
  unit: string;
  provenance: Provenance;
  meaning: string;
  law: string;
  series: Series | null;
  available: boolean;
  unavailableReason: string | null;
}

export interface SceneSeries {
  cycles: Series;
  sohPct: Series;
  capacityAh: Series;
  resistanceOhm: Series;
  resistanceNormalized: Series;
  temperatureC: Series;
  sopPct: Series;
  fadeRate30cy: Series;
  seiPct: Series;
  lamPct: Series;
  seiSharePct: Series;
  fadeModelPct: Series;
}

export interface SceneBand {
  min: number;
  max: number | null;
  color: string;
  label: string;
}

/**
 * Design tokens. The renderer reads every colour from here and hard-codes none
 * of its own — including the SOH band thresholds, so a host that reshapes the
 * bands reshapes what the scene calls healthy on both the mesh and the legend.
 */
export interface SceneTheme {
  background: string;
  panel: string;
  text: string;
  muted: string;
  accent: string;
  grid: string;
  metal: string;
  anodeColor: string;
  cathodeColor: string;
  separatorColor: string;
  electrolyteColor: string;
  seiColor: string;
  sohBands: SceneBand[];
  temperatureBands: SceneBand[];
  provenanceColors: Record<string, string>;
  sopFloorPct: number;
  powerGaugeMinPct: number;
  powerGaugeMaxPct: number;
}

export interface PhysicsVerdict {
  key: "lli" | "lam" | "mixed" | "insufficient_data";
  label: string;
  gatePassed: boolean;
  gate?: number | null;
  betaSei?: number | null;
  betaLam?: number | null;
  fitR2?: number | null;
  reason?: string | null;
  contributionSeiPct?: number | null;
  contributionLamPct?: number | null;
  atCycle?: number | null;
}

export interface MechanismVerdict {
  physics: PhysicsVerdict | null;
  ml: { verdict: string; confidenceLabel?: string | null; body?: string | null; source: string } | null;
  /** null whenever agreement would not be evidence. */
  agree: boolean | null;
  agreementIsEvidence: boolean;
  note: string;
  emphasis: "sei" | "particles" | "neutral";
  emphasisSource: "physics" | "ml" | "none";
}

export interface GeometryScale {
  target: "shellThickness" | "lostFraction" | "colour" | "crossSection" | "capacityLoss";
  from: string;
  unit: string;
  displayMin?: number | null;
  displayMax?: number | null;
  renderAs?: string;
  note: string;
}

export interface SceneProjection {
  available: boolean;
  reason: string | null;
  cycles?: Series;
  sohPct?: Series;
  sohQ10Pct?: Series;
  sohQ90Pct?: Series;
  kind?: "projected";
  modelLabel?: string;
  shrinkageFromPrior?: number | null;
  route?: Record<string, unknown> | null;
}

export interface CellSceneSpec {
  schemaVersion: number;
  cell: {
    id: string;
    source: string;
    chemistry: string;
    formFactor: "cylindrical" | "prismatic" | "unknown";
    formFactorNote: string | null;
    provenance: string;
    dqdvApplicable: boolean;
  };
  series: SceneSeries;
  record: {
    firstCycle: number | null;
    lastCycle: number | null;
    nCycles: number;
    nRecordedRows: number;
    sampled: boolean;
    lastSohPct: number | null;
  };
  physics: {
    law: string;
    seiTerm: string;
    lamTerm: string;
    available: boolean;
    splitIdentified: boolean;
    splitReason: string | null;
    betaSei: number | null;
    betaLam: number | null;
    fitR2: number | null;
    reason: string | null;
    source: string;
    modelTotalFadePct?: number | null;
    measuredFadePct?: number | null;
    kR?: number | null;
    fitR2Series?: Series;
  };
  mechanism: MechanismVerdict;
  knee: {
    detected: boolean;
    cycle: number | null;
    sohAtKnee: number | null;
    confidence: number | null;
    phase: string | null;
  };
  projection: SceneProjection;
  parts: ScenePart[];
  geometryScales: Record<string, GeometryScale>;
  disclosures: string[];
  theme: SceneTheme;
}
