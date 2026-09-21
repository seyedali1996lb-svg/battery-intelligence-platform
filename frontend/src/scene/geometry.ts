/**
 * The cell's geometry — DOM-free, three.js-free, and therefore testable.
 *
 * This module turns a `CellSceneSpec` plus a few display options into plain
 * typed arrays. It touches no canvas, no WebGL, no `document` and no three.js,
 * so the entire anatomy can be verified headlessly by Node's own test runner
 * (`geometry.test.ts`); `engine.ts` is the only file that knows a renderer
 * exists. Every number here is either a physical proportion of a real cell or a
 * mapping the spec itself declares (`geometryScales`) — nothing is decorative
 * invention presented as measurement.
 *
 * Two coordinate rules, held everywhere:
 *   * Units are cell heights. A cell is 1.0 tall and centred on the origin, so
 *     a 18650's radius is 9/65 and an 8 mm jelly-roll ribbon is 0.008. This is
 *     what keeps a prismatic cell from being drawn as a stretched cylinder.
 *   * `exploded` is a *view* control, never data. It moves parts apart; no
 *     measurement depends on it, so nothing can be misread by scrubbing it.
 */

import { ANATOMY_PART_IDS } from "./types.ts";
import type { CellSceneSpec, PartId, Series } from "./types.ts";
import { bandFor, normalize, powerColor, resistanceColor, sohColor, temperatureColor } from "./theme.ts";

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

/** Interleaved-free triangle mesh: parallel arrays, indexed, right-handed, Y up. */
export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
}

/** The 2D outline points used to build an extrusion (x, z), in cell units. */
export type Outline = Array<[number, number]>;

export function emptyMesh(): Mesh {
  return { positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint32Array(0) };
}

export function vertexCount(mesh: Mesh): number {
  return mesh.positions.length / 3;
}

export function triangleCount(mesh: Mesh): number {
  return mesh.indices.length / 3;
}

/**
 * Positions + indices → a mesh with area-weighted vertex normals.
 *
 * Vertex (not face) normals because the casing and the roll ribbons are curved:
 * flat shading turns every tessellation seam into a visible facet. Area
 * weighting is the standard accumulation and costs one extra multiply here.
 */
function finalize(positions: number[], indices: number[]): Mesh {
  const normals = new Array<number>(positions.length).fill(0);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const at of [a, b, c]) {
      normals[at] += nx;
      normals[at + 1] += ny;
      normals[at + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    if (len > 0) {
      normals[i] /= len;
      normals[i + 1] /= len;
      normals[i + 2] /= len;
    }
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  };
}

export function translateMesh(mesh: Mesh, dx: number, dy: number, dz: number): Mesh {
  const positions = Float32Array.from(mesh.positions);
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] += dx;
    positions[i + 1] += dy;
    positions[i + 2] += dz;
  }
  return { positions, normals: mesh.normals, indices: mesh.indices };
}

export function scaleMesh(mesh: Mesh, sx: number, sy: number, sz: number): Mesh {
  const positions = Float32Array.from(mesh.positions);
  const normals = Float32Array.from(mesh.normals);
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] *= sx;
    positions[i + 1] *= sy;
    positions[i + 2] *= sz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i] / sx;
    const ny = normals[i + 1] / sy;
    const nz = normals[i + 2] / sz;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      normals[i] = nx / len;
      normals[i + 1] = ny / len;
      normals[i + 2] = nz / len;
    }
  }
  return { positions, normals, indices: mesh.indices };
}

/** Merge meshes that already share one coordinate space (one draw call each). */
export function mergeMeshes(meshes: Mesh[]): Mesh {
  const positions: number[] = [];
  const indices: number[] = [];
  let offset = 0;
  for (const mesh of meshes) {
    if (mesh.positions.length === 0) continue;
    for (let i = 0; i < mesh.positions.length; i++) positions.push(mesh.positions[i]);
    for (let i = 0; i < mesh.indices.length; i++) indices.push(mesh.indices[i] + offset);
    offset += mesh.positions.length / 3;
  }
  return finalize(positions, indices);
}

// ---------------------------------------------------------------------------
// Outlines (x, z) — the 2D shapes every solid is extruded from
// ---------------------------------------------------------------------------

/** A circle as an outline, counter-clockwise in the (x, z) plane. */
export function ringPoints(radius: number, segments: number): Outline {
  const points: Outline = [];
  for (let i = 0; i < segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
  }
  return points;
}

/** An open arc — used for the cut-away casing, which is never a closed solid. */
export function arcPoints(radius: number, sweep: number, segments: number, start = 0): Outline {
  const points: Outline = [];
  for (let i = 0; i <= segments; i++) {
    const theta = start + (i / segments) * sweep;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
  }
  return points;
}

/** Signed area (shoelace). Positive means counter-clockwise in (x, z). */
export function signedArea(points: Outline): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const [x0, z0] = points[i];
    const [x1, z1] = points[(i + 1) % points.length];
    sum += x0 * z1 - x1 * z0;
  }
  return sum / 2;
}

export function ensureCounterClockwise(points: Outline): Outline {
  return signedArea(points) < 0 ? [...points].reverse() : points;
}

/**
 * A jelly-roll layer as a closed outline: the outer spiral forward, the inner
 * spiral back, joined at both ends.
 *
 * This is the shape that makes the roll read as a roll rather than as three
 * nested tubes. `thickness` is radial; `pitch` is how far the ribbon walks
 * outward per turn, so `turns * pitch` is the radial space the coil occupies —
 * the reason a real cell's roll counts its turns in tens and this schematic
 * counts them in twos and threes.
 */
export function spiralOutline(options: {
  innerRadius: number;
  turns: number;
  pitch: number;
  thickness: number;
  segmentsPerTurn?: number;
  phase?: number;
}): Outline {
  const { innerRadius, turns, pitch, thickness } = options;
  const segmentsPerTurn = options.segmentsPerTurn ?? 36;
  const phase = options.phase ?? 0;
  const segments = Math.max(8, Math.round(turns * segmentsPerTurn));
  const forward: Outline = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const theta = phase + t * turns * Math.PI * 2;
    const r = innerRadius + t * turns * pitch;
    forward.push([r * Math.cos(theta), r * Math.sin(theta)]);
  }
  const backward: Outline = [];
  for (let i = segments; i >= 0; i--) {
    const t = i / segments;
    const theta = phase + t * turns * Math.PI * 2;
    const r = innerRadius + thickness + t * turns * pitch;
    backward.push([r * Math.cos(theta), r * Math.sin(theta)]);
  }
  return ensureCounterClockwise([...forward, ...backward]);
}

/**
 * Extrude a closed outline along Y, centred on y = 0.
 *
 * Caps are triangulated as a fan from the first point. A fan is only correct
 * for a star-shaped polygon — every outline this module builds is one (a
 * circle, a spiral band, a rectangle), and a test asserts the fan's triangles
 * all point the right way, which is the property that would break first if a
 * non-star-shaped outline ever arrived here.
 */
export function extrudeClosed(points: Outline, height: number): Mesh {
  const outline = ensureCounterClockwise(points);
  const n = outline.length;
  if (n < 3 || height <= 0) return emptyMesh();
  const half = height / 2;
  const positions: number[] = [];
  const indices: number[] = [];

  for (const [x, z] of outline) positions.push(x, -half, z);
  for (const [x, z] of outline) positions.push(x, half, z);

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const b0 = i;
    const b1 = j;
    const t0 = n + i;
    const t1 = n + j;
    // Sides: (bottom_i, top_i, top_j) and (bottom_i, top_j, bottom_j) — the
    // winding that yields the outward `(dz, 0, -dx)` normal for a CCW outline.
    indices.push(b0, t0, t1, b0, t1, b1);
  }

  // Caps. A triangle wound counter-clockwise in the (x, z) plane has a -Y
  // normal (pinned by a test), so the *top* cap takes the reversed fan and the
  // bottom cap the natural one. Getting this backwards leaves a solid that
  // looks hollow from inside and invisible from outside — which is why it is
  // asserted on face normals rather than trusted.
  for (let i = 1; i < n - 1; i++) indices.push(n, n + i + 1, n + i);
  for (let i = 1; i < n - 1; i++) indices.push(0, i, i + 1);

  return finalize(positions, indices);
}

/** Extrude an open path (sides only) — the cut-away casing, drawn two-sided. */
export function extrudeOpen(path: Outline, height: number): Mesh {
  const n = path.length;
  if (n < 2 || height <= 0) return emptyMesh();
  const half = height / 2;
  const positions: number[] = [];
  const indices: number[] = [];
  for (const [x, z] of path) positions.push(x, -half, z);
  for (const [x, z] of path) positions.push(x, half, z);
  for (let i = 0; i < n - 1; i++) {
    const b0 = i;
    const b1 = i + 1;
    const t0 = n + i;
    const t1 = n + i + 1;
    // Emitted both ways: an open shell has no inside to hide, and a one-sided
    // shell simply vanishes when the camera orbits behind it.
    indices.push(b0, t0, t1, b0, t1, b1);
    indices.push(t1, t0, b0, b1, t1, b0);
  }
  return finalize(positions, indices);
}

export function boxMesh(width: number, height: number, depth: number): Mesh {
  const w = width / 2;
  const h = height / 2;
  const d = depth / 2;
  const corners: Array<[number, number, number]> = [
    [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
    [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],
  ];
  const positions: number[] = [];
  for (const [x, y, z] of corners) positions.push(x, y, z);
  const indices = [
    0, 2, 1, 0, 3, 2, // -Z
    4, 5, 6, 4, 6, 7, // +Z
    0, 1, 5, 0, 5, 4, // -Y
    3, 7, 6, 3, 6, 2, // +Y
    0, 4, 7, 0, 7, 3, // -X
    1, 2, 6, 1, 6, 5, // +X
  ];
  return finalize(positions, indices);
}

export function sphereMesh(radius: number, widthSegments = 6, heightSegments = 4): Mesh {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let iy = 0; iy <= heightSegments; iy++) {
    const v = iy / heightSegments;
    const phi = v * Math.PI;
    for (let ix = 0; ix <= widthSegments; ix++) {
      const u = ix / widthSegments;
      const theta = u * Math.PI * 2;
      positions.push(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta),
      );
    }
  }
  const rowLength = widthSegments + 1;
  for (let iy = 0; iy < heightSegments; iy++) {
    for (let ix = 0; ix < widthSegments; ix++) {
      const a = iy * rowLength + ix;
      const b = a + rowLength;
      indices.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }
  return finalize(positions, indices);
}

// ---------------------------------------------------------------------------
// Physical proportions (cell units: a cell is 1.0 tall)
// ---------------------------------------------------------------------------

/**
 * The proportions every mesh is built from. Real where it can be: an 18650's
 * 18 mm diameter over 65 mm height is 9/65, and the roll's 8 mm per turn is a
 * real jelly-roll pitch. Where it cannot be real it is a declared schematic —
 * the roll has tens of turns in a real cell, not three.
 */
export const CELL_GEOMETRY = {
  /** 18650-class cylindrical: 18 mm diameter, 65 mm tall. */
  canRadius: 9 / 65,
  /** Fraction of the casing's circumference drawn (the cut-away). */
  canSweep: (285 * Math.PI) / 180,
  canThickness: 0.004,
  /** A prismatic (CALCE 1.1 Ah pouch class) cell: 5.4 × 20.5 × 64 mm, schematically. */
  prismatic: { width: 0.34, depth: 0.125, thickness: 0.004 },
  roll: {
    /** Each coating is its own coil here; a real roll interleaves them. */
    anode: { innerRadius: 0.03, turns: 3, pitch: 0.008, thickness: 0.005 },
    separator: { innerRadius: 0.062, turns: 2, pitch: 0.008, thickness: 0.003 },
    cathode: { innerRadius: 0.084, turns: 2, pitch: 0.008, thickness: 0.005 },
    height: 0.84,
    segmentsPerTurn: 40,
  },
  /**
   * Explode displacements at `exploded = 1`, in cell units. `layerGap` is per
   * layer step, chosen so the fully exploded stack still fits inside the
   * casing (a test asserts exactly that): a cutaway that flings the cathode
   * through the can wall teaches the wrong thing about where it lives.
   */
  explode: { cap: 0.3, vent: 0.36, terminalPos: 0.44, terminalNeg: 0.3, layerGap: 0.014 },
  /** Draw-call economy: the whole particle cloud is one merged mesh. */
  particles: { count: 240, radius: 0.0055, anodeShare: 0.5 },
  /**
   * The drawn SEI film's thickness range, in cell units. Deliberately narrow:
   * it spans only the radial space the drawn roll leaves between the anode
   * ribbon and the separator, so a data-scaled film can never pierce a layer
   * it does not physically sit inside. Real SEI is nanometres thick — this is a
   * legibility-scaled metaphor, disclosed as one, and its opacity carries the
   * same fitted term so the two cannot disagree.
   */
  seiFilm: { anatomical: 0.0015, dataMax: 0.0048, minOpacity: 0.18, maxOpacity: 0.52 },
} as const;

// ---------------------------------------------------------------------------
// Timeline: measured replay, then the platform's own projection
// ---------------------------------------------------------------------------

export interface Timeline {
  cycles: number[];
  sohPct: (number | null)[];
  sohQ10Pct: (number | null)[];
  sohQ90Pct: (number | null)[];
  /** True for every cursor position past the last measured cycle. */
  projected: boolean[];
  measuredCount: number;
  lastMeasuredCycle: number | null;
  hasProjection: boolean;
  projectionLabel: string | null;
}

/**
 * The one index a host scrubs: measured cycles first, then the projection.
 *
 * Concatenating them here (rather than letting each host splice them) is what
 * makes "the scene stops at the last measured cycle" mechanical: with no
 * projection the timeline simply ends, and a refused forecast cannot leak into
 * the scrubber because it never entered it.
 */
export function buildTimeline(spec: CellSceneSpec): Timeline {
  const cycles: number[] = [];
  const soh: (number | null)[] = [];
  const q10: (number | null)[] = [];
  const q90: (number | null)[] = [];
  const projected: boolean[] = [];

  spec.series.cycles.forEach((cycle, i) => {
    cycles.push(cycle ?? Number.NaN);
    soh.push(spec.series.sohPct[i] ?? null);
    q10.push(null);
    q90.push(null);
    projected.push(false);
  });
  const measuredCount = cycles.length;

  const projection = spec.projection;
  if (projection.available && Array.isArray(projection.cycles)) {
    projection.cycles.forEach((cycle, i) => {
      cycles.push(cycle ?? Number.NaN);
      soh.push(projection.sohPct?.[i] ?? null);
      q10.push(projection.sohQ10Pct?.[i] ?? null);
      q90.push(projection.sohQ90Pct?.[i] ?? null);
      projected.push(true);
    });
  }

  return {
    cycles,
    sohPct: soh,
    sohQ10Pct: q10,
    sohQ90Pct: q90,
    projected,
    measuredCount,
    lastMeasuredCycle: measuredCount > 0 ? cycles[measuredCount - 1] : null,
    hasProjection: cycles.length > measuredCount,
    projectionLabel: projection.available ? projection.modelLabel ?? "Projection" : null,
  };
}

/** The default cursor: "today", i.e. the last measured cycle. */
export function todayCursor(timeline: Timeline): number {
  return Math.max(0, timeline.measuredCount - 1);
}

/**
 * The reading a part's series carries at the cursor.
 *
 * `carried` is the honesty flag: a gap in a series is a missing measurement,
 * and a missing measurement must not collapse a drawn layer to zero. So the
 * geometry holds the last known reading and says it did — the mesh keeps
 * showing the last thing that was actually measured, and the card says so.
 * Past the last measured cycle nothing is carried forward from a model either:
 * beyond that point every part simply holds its final measured state, because
 * no source here measures an SEI film 300 cycles into the future.
 */
export function readingAt(
  series: Series | null,
  cursor: number,
): { value: number | null; carried: boolean; inRange: boolean } {
  if (!series || series.length === 0) return { value: null, carried: false, inRange: false };
  const at = Math.min(cursor, series.length - 1);
  const direct = series[at];
  if (direct !== null && direct !== undefined && Number.isFinite(direct)) {
    return { value: direct, carried: false, inRange: cursor < series.length };
  }
  for (let i = Math.min(cursor, series.length - 1); i >= 0; i--) {
    const candidate = series[i];
    if (candidate !== null && candidate !== undefined && Number.isFinite(candidate)) {
      return { value: candidate, carried: true, inRange: cursor < series.length };
    }
  }
  return { value: null, carried: false, inRange: cursor < series.length };
}

// ---------------------------------------------------------------------------
// The anatomy
// ---------------------------------------------------------------------------

export interface BuildOptions {
  /** Index into the timeline; clamped into range. */
  cursor: number;
  /** 0 = assembled, 1 = fully separated. A view control, never data. */
  exploded: number;
  casing: "translucent" | "hidden";
  /** Data-scaled geometry instead of anatomical proportions. Opt-in. */
  dataScaled: boolean;
}

export const DEFAULT_BUILD_OPTIONS: BuildOptions = {
  cursor: 0,
  exploded: 0,
  casing: "translucent",
  dataScaled: false,
};

/**
 * Fold a host's partial request into the current view options.
 *
 * Written out field by field, deliberately: an option a host did not mention
 * keeps its current value (a caller that only moves the cursor must not reset
 * the explode position), and adding a field to `BuildOptions` forces this
 * function to be revisited rather than silently dropped.
 *
 * It exists because of a real bug it fixes: the engine used to apply a host's
 * options only in `update()`, never at `mount()`, so a host that passed
 * `cursor` — every host does, because a scene should open at *today's* reading
 * rather than at the cell's first cycle — was quietly ignored and the scene
 * disagreed with its own slider on first paint.
 */
export function mergeBuildOptions(build: BuildOptions, patch: Partial<BuildOptions>): BuildOptions {
  return {
    cursor: patch.cursor ?? build.cursor,
    exploded: patch.exploded ?? build.exploded,
    casing: patch.casing ?? build.casing,
    dataScaled: patch.dataScaled ?? build.dataScaled,
  };
}

export interface BuiltPart {
  id: PartId;
  mesh: Mesh;
  /** Whether the anatomy draws geometry for this part at all. */
  drawn: boolean;
  color: string;
  opacity: number;
  /** 0 = matte, 1 = fully self-lit (used for the parts carrying current). */
  emissive: number;
  /** Label anchor in cell space, including the explode displacement. */
  anchor: [number, number, number];
  label: string;
  /** Reading at the cursor, or null when this part has no measurement. */
  value: number | null;
  /** The reading is the last known one because this cycle has none. */
  carried: boolean;
  unit: string;
  provenance: string;
  available: boolean;
  reason: string | null;
}

/**
 * One part of the anatomy as a host can print it: the document's prose about the
 * part, resolved against the geometry's reading *at the current cursor*.
 *
 * Both halves are needed and neither alone is right. The document carries the
 * label, the unit and the sentence explaining what the part means; the built
 * scene carries the reading the scene is currently drawing, which is the whole
 * point — scrub the cursor and the number on the card has to move with the
 * cylinder it describes, or the host is showing two different cells.
 */
export interface PartReading {
  id: PartId;
  label: string;
  value: number | null;
  unit: string;
  provenance: string;
  available: boolean;
  /** True when this cycle has no measurement and the last known one is shown. */
  carried: boolean;
  /** One sentence, from the document: what this part is. */
  meaning: string;
  /** Why there is no reading, when there is none. */
  reason: string | null;
}

/**
 * The part table a host can render beside the scene, at the scene's own cursor.
 *
 * One row per part the scene drew, in the scene's order — the geometry builds
 * its parts *from* the document, so a document missing a part is a scene
 * missing that part, and the table can never list a card for a mesh that is not
 * there. The lookup is by id rather than by index so a reordered document still
 * matches, and the `?? ""` is defence against a part the two halves disagree
 * about, not a case that should ever be reachable.
 */
export function partReadings(spec: CellSceneSpec, scene: BuiltScene): PartReading[] {
  return scene.parts.map((part) => {
    const described = spec.parts.find((candidate) => candidate.id === part.id);
    return {
      id: part.id,
      label: part.label,
      value: part.value,
      unit: part.unit,
      provenance: part.provenance,
      available: part.available,
      carried: part.carried,
      meaning: described?.meaning ?? "",
      reason: part.reason ?? described?.unavailableReason ?? null,
    };
  });
}

export interface Gauge {
  /** 0–1: the fraction of a fresh cell's usable capacity this cell holds. */
  fraction: number;
  soh: number | null;
  color: string;
  label: string;
  /** Posterior band from the projection, when the cursor is in the future. */
  band: { low: number; high: number } | null;
  projected: boolean;
  cycle: number | null;
}

export interface BuiltScene {
  parts: BuiltPart[];
  timeline: Timeline;
  cursor: number;
  gauge: Gauge;
  /** Stage furniture: floor and contact shadow. Carries no data. */
  chrome: { floor: Mesh; shadow: Mesh };
  bounds: { radius: number; height: number };
  /** The data-scaling sentence in force, straight from the spec, or null. */
  scaleNote: string | null;
  formFactor: string;
}

interface Placement {
  mesh: Mesh;
  anchor: [number, number, number];
  drawn?: boolean;
}

function _rollRibbon(layer: { innerRadius: number; turns: number; pitch: number; thickness: number }, radialShift: number, height: number): Mesh {
  const outline = spiralOutline({
    innerRadius: layer.innerRadius + radialShift,
    turns: layer.turns,
    pitch: layer.pitch,
    thickness: layer.thickness,
    segmentsPerTurn: CELL_GEOMETRY.roll.segmentsPerTurn,
  });
  return extrudeClosed(outline, height);
}

/**
 * A deterministic sample of the coating volume — the same cloud every render.
 *
 * Seeded by a constant, not by the cursor: this is a fixed sample *of a cell*,
 * so scrubbing the timeline must not teleport the particles. What changes with
 * the data is how many of them are drawn as lost, not where they are.
 */
function _particlePositions(
  count: number,
  anodeShift = 0,
  cathodeShift = 0,
): Array<[number, number, number]> {
  let state = 0x2f6e2b1;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
  const { anode, cathode, height } = CELL_GEOMETRY.roll;
  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < count; i++) {
    const inner = rand() < CELL_GEOMETRY.particles.anodeShare;
    const band = inner ? anode : cathode;
    // Particles ride their own coating when the stack is exploded, so the
    // cloud never drifts away from the ribbons it belongs to.
    const shift = inner ? anodeShift : cathodeShift;
    const radius =
      shift + band.innerRadius + 0.002 + rand() * Math.max(0.004, band.turns * band.pitch - 0.004);
    const theta = rand() * Math.PI * 2;
    const y = (rand() - 0.5) * height * 0.94;
    out.push([radius * Math.cos(theta), y, radius * Math.sin(theta)]);
  }
  return out;
}

function _cylindricalPlacements(exploded: number, filmThickness: number): Record<string, Placement> {
  const { canRadius, canSweep, canThickness, roll, explode } = CELL_GEOMETRY;
  const casing = extrudeOpen(arcPoints(canRadius, canSweep, 64, Math.PI * 0.6), 1.0);
  const floorDisc = extrudeClosed(ringPoints(canRadius, 64), 0.012);
  const canMesh = mergeMeshes([casing, translateMesh(floorDisc, 0, -0.494, 0)]);

  const anodeOuter = roll.anode.innerRadius + roll.anode.turns * roll.anode.pitch + roll.anode.thickness;

  return {
    can: { mesh: canMesh, anchor: [canRadius * 0.98, 0.34, canRadius * 0.2] },
    cap: {
      mesh: translateMesh(extrudeClosed(ringPoints(canRadius, 64), 0.022), 0, 0.5 + explode.cap * exploded, 0),
      anchor: [canRadius * 0.8, 0.5 + explode.cap * exploded, 0],
    },
    vent: {
      mesh: translateMesh(extrudeClosed(ringPoints(0.03, 24), 0.008), 0, 0.511 + explode.vent * exploded, 0),
      anchor: [-canRadius * 0.85, 0.511 + explode.vent * exploded, 0],
    },
    terminal_pos: {
      mesh: translateMesh(extrudeClosed(ringPoints(0.038, 28), 0.05), 0, 0.53 + explode.terminalPos * exploded, 0),
      anchor: [canRadius * 0.9, 0.555 + explode.terminalPos * exploded, 0],
    },
    terminal_neg: {
      mesh: translateMesh(extrudeClosed(ringPoints(0.05, 28), 0.05), 0, -0.53 - explode.terminalNeg * exploded, 0),
      anchor: [-canRadius * 0.9, -0.555 - explode.terminalNeg * exploded, 0],
    },
    tab_pos: {
      mesh: translateMesh(boxMesh(0.05, 0.006, 0.02), 0.055, roll.height / 2 + 0.012, 0),
      anchor: [0.06, roll.height / 2 + 0.05, 0],
    },
    tab_neg: {
      mesh: translateMesh(boxMesh(0.05, 0.006, 0.02), 0.055, -roll.height / 2 - 0.012, 0),
      anchor: [-0.06, -roll.height / 2 - 0.05, 0],
    },
    anode_sheet: { mesh: _rollRibbon(roll.anode, 0, roll.height), anchor: [0, roll.height / 2 + 0.06, 0] },
    separator: {
      mesh: _rollRibbon(roll.separator, explode.layerGap * exploded, roll.height),
      anchor: [0, -roll.height / 2 - 0.06, 0],
    },
    cathode_sheet: {
      mesh: _rollRibbon(roll.cathode, explode.layerGap * exploded * 2, roll.height),
      anchor: [0, roll.height / 2 + 0.14, 0],
    },
    electrolyte: {
      mesh: extrudeClosed(ringPoints(canRadius - canThickness, 48), roll.height * 0.98),
      anchor: [0, -0.1, 0],
    },
    sei_film: {
      mesh: extrudeClosed(
        ringPoints(anodeOuter + explode.layerGap * exploded + filmThickness / 2, 48),
        roll.height * 0.98,
      ),
      anchor: [-0.05, 0.2, 0],
    },
    particles: { mesh: emptyMesh(), anchor: [0, -0.24, 0] },
  };
}

function _prismaticPlacements(exploded: number, filmThickness: number): Record<string, Placement> {
  const { prismatic, roll, explode } = CELL_GEOMETRY;
  const halfWidth = prismatic.width / 2;
  // The shell: four walls and a floor, drawn translucent so the stack is visible.
  const walls = mergeMeshes([
    translateMesh(boxMesh(prismatic.width, 1.0, prismatic.thickness), 0, 0, prismatic.depth / 2),
    translateMesh(boxMesh(prismatic.width, 1.0, prismatic.thickness), 0, 0, -prismatic.depth / 2),
    translateMesh(boxMesh(prismatic.thickness, 1.0, prismatic.depth), halfWidth, 0, 0),
    translateMesh(boxMesh(prismatic.thickness, 1.0, prismatic.depth), -halfWidth, 0, 0),
    translateMesh(boxMesh(prismatic.width, prismatic.thickness, prismatic.depth), 0, -0.5, 0),
  ]);
  const slab = (z: number, thickness: number, height: number) =>
    translateMesh(boxMesh(prismatic.width * 0.9, height, thickness), 0, 0, z);

  const anodeZ = -0.028;
  const cathodeZ = 0.028;
  return {
    can: { mesh: walls, anchor: [halfWidth, 0.3, prismatic.depth / 2] },
    cap: {
      mesh: translateMesh(boxMesh(prismatic.width, 0.02, prismatic.depth), 0, 0.5 + explode.cap * exploded, 0),
      anchor: [halfWidth, 0.5 + explode.cap * exploded, 0],
    },
    vent: {
      mesh: translateMesh(boxMesh(0.06, 0.008, 0.04), 0.08, 0.51 + explode.vent * exploded, 0),
      anchor: [-0.08, 0.511 + explode.vent * exploded, 2 * prismatic.depth],
    },
    terminal_pos: {
      mesh: translateMesh(extrudeClosed(ringPoints(0.03, 24), 0.04), 0.11, 0.53 + explode.terminalPos * exploded, 0),
      anchor: [halfWidth, 0.55 + explode.terminalPos * exploded, 0],
    },
    terminal_neg: {
      mesh: translateMesh(extrudeClosed(ringPoints(0.03, 24), 0.04), -0.11, -0.53 - explode.terminalNeg * exploded, 0),
      anchor: [-halfWidth, -0.55 - explode.terminalNeg * exploded, 0],
    },
    tab_pos: {
      mesh: translateMesh(boxMesh(0.02, 0.05, 0.006), -0.09, roll.height / 2 + 0.03, anodeZ),
      anchor: [-0.1, roll.height / 2 + 0.08, anodeZ],
    },
    tab_neg: {
      mesh: translateMesh(boxMesh(0.02, 0.05, 0.006), 0.09, roll.height / 2 + 0.03, cathodeZ),
      anchor: [0.1, roll.height / 2 + 0.08, cathodeZ],
    },
    anode_sheet: { mesh: slab(anodeZ, 0.012, roll.height), anchor: [0, roll.height / 2 + 0.12, anodeZ] },
    separator: {
      mesh: slab(0, 0.008, roll.height * 0.98),
      anchor: [0, -roll.height / 2 - 0.06, 0],
    },
    cathode_sheet: { mesh: slab(cathodeZ, 0.012, roll.height), anchor: [0, roll.height / 2 + 0.2, cathodeZ] },
    electrolyte: { mesh: slab(0, prismatic.depth - 0.01, roll.height * 0.99), anchor: [0, -0.14, 0] },
    sei_film: {
      mesh: slab(anodeZ + 0.008 + filmThickness / 2, filmThickness, roll.height * 0.98),
      anchor: [-halfWidth * 0.5, 0.22, anodeZ],
    },
    particles: { mesh: emptyMesh(), anchor: [0, -0.26, 0] },
  };
}

/**
 * Build every mesh and every drawn datum for one cursor position.
 *
 * Called on every host update (a scrub tick or a toggle). It rebuilds typed
 * arrays rather than mutating a scene graph, which is what keeps it testable —
 * and cheap enough that a slider drag stays interactive: the whole cell is
 * ~20k triangles, and only the layer that a control actually changes is
 * rebuilt with different numbers.
 */
export function buildScene(spec: CellSceneSpec, options: Partial<BuildOptions> = {}): BuiltScene {
  const opts: BuildOptions = { ...DEFAULT_BUILD_OPTIONS, ...options };
  const timeline = buildTimeline(spec);
  const cursor = Math.max(0, Math.min(Math.trunc(opts.cursor), Math.max(0, timeline.cycles.length - 1)));
  const exploded = Math.max(0, Math.min(1, Number.isFinite(opts.exploded) ? opts.exploded : 0));
  const projected = timeline.projected[cursor] === true;

  // Past the record every part holds its final measured state: see readingAt.
  const partCursor = projected ? Math.max(0, timeline.measuredCount - 1) : cursor;

  const soh = timeline.sohPct[cursor] ?? null;
  const seiAtCursor = readingAt(spec.series.seiPct, partCursor).value;
  const splitIdentified = spec.physics.splitIdentified === true;

  // The drawn film's thickness maps 0 → the anatomical minimum and the largest
  // fitted value in this cell's own record → the maximum. The top of the range
  // is read from the data, not fixed here, so a cell with a small film is not
  // drawn identically to a cell with a large one.
  const seiScale = spec.geometryScales.sei_film;
  const filmScaled = opts.dataScaled && splitIdentified;
  const filmFraction = filmScaled
    ? normalize(
        seiAtCursor,
        seiScale?.displayMin ?? 0,
        seiScale?.displayMax ?? seriesMax(spec.series.seiPct),
        0,
      )
    : 0;
  const filmThickness = filmScaled
    ? CELL_GEOMETRY.seiFilm.anatomical +
      filmFraction * (CELL_GEOMETRY.seiFilm.dataMax - CELL_GEOMETRY.seiFilm.anatomical)
    : CELL_GEOMETRY.seiFilm.anatomical;

  const placements =
    spec.cell.formFactor === "prismatic"
      ? _prismaticPlacements(exploded, filmThickness)
      : _cylindricalPlacements(exploded, filmThickness);

  // The particle cloud: a legibility-limited sample of the coating volume. The
  // *lost* fraction follows the fitted linear term only when the split is
  // identified — otherwise there is no measured particle-loss number to draw.
  const seiShare = readingAt(spec.series.seiSharePct, partCursor).value;
  // The share of the fitted fade that is NOT the film — i.e. the share of the
  // drawn cloud shown as no longer cycling. A share, not the raw LAM value:
  // "40% of the coating is dark" is a statement about this cell, whereas
  // "3.0 %/cycle" is not a fraction of anything.
  const lostFraction =
    splitIdentified && opts.dataScaled && seiShare !== null ? normalize(100 - seiShare, 0, 100, 0) : 0;
  const cloud = _particlePositions(
    CELL_GEOMETRY.particles.count,
    0,
    2 * CELL_GEOMETRY.explode.layerGap * exploded,
  );
  const liveCount = Math.round(cloud.length * (1 - lostFraction));
  if (placements.particles) {
    placements.particles.mesh = mergeMeshes(
      cloud.slice(0, liveCount).map(([x, y, z]) =>
        translateMesh(sphereMesh(CELL_GEOMETRY.particles.radius), x, y, z),
      ),
    );
  }

  const gauge: Gauge = {
    fraction: soh === null ? 0 : Math.max(0, Math.min(1, soh / 100)),
    soh,
    color: sohColor(soh, spec.theme),
    label: bandFor(soh, spec.theme.sohBands)?.label ?? "No reading",
    band:
      projected && timeline.sohQ10Pct[cursor] !== null && timeline.sohQ90Pct[cursor] !== null
        ? {
            low: Math.max(0, Math.min(1, (timeline.sohQ10Pct[cursor] as number) / 100)),
            high: Math.max(0, Math.min(1, (timeline.sohQ90Pct[cursor] as number) / 100)),
          }
        : null,
    projected,
    cycle: Number.isFinite(timeline.cycles[cursor]) ? timeline.cycles[cursor] : null,
  };

  const parts = spec.parts.map((part) => {
    const placement = placements[part.id];
    const reading = readingAt(part.series, partCursor);
    // A part with no series of its own still has one number — the latest
    // measured one — and that is what its card shows at every cursor position.
    const value = part.series ? reading.value : part.value;
    const temperature = readingAt(spec.series.temperatureC, partCursor).value;
    const sop = readingAt(spec.series.sopPct, partCursor).value;
    const rNorm = readingAt(spec.series.resistanceNormalized, partCursor).value;

    let color = spec.theme.metal;
    let emissive = 0;
    let opacity = 1;
    let mesh = placement?.mesh ?? emptyMesh();
    let drawn = placement !== undefined && vertexCount(mesh) > 0;
    if (part.id === "particles") drawn = vertexCount(mesh) > 0;

    switch (part.id) {
      case "can":
        color = temperatureColor(temperature, spec.theme);
        opacity = opts.casing === "hidden" ? 0 : 0.22;
        drawn = opts.casing !== "hidden" && vertexCount(mesh) > 0;
        if (opts.casing === "hidden") mesh = emptyMesh();
        break;
      case "cap":
      case "vent":
        color = spec.theme.metal;
        opacity = 0.9;
        break;
      case "terminal_pos":
      case "terminal_neg": {
        const load = readingAt(spec.series.resistanceNormalized, partCursor).value;
        color = spec.theme.accent;
        emissive = 0.25 + 0.5 * normalize(load, 1, 2.2, 0);
        opacity = 0.95;
        if (part.id === "terminal_pos" && opts.dataScaled) {
          // Narrowed current path as measured DC resistance rises. One number,
          // one mapping, printed by the scene from geometryScales.
          const shrink = 1 - 0.5 * normalize(load, 1, 2.2, 0);
          mesh = scaleMesh(mesh, shrink, 1, shrink);
        }
        break;
      }
      case "tab_pos":
        color = resistanceColor(rNorm, spec.theme);
        emissive = 0.3;
        break;
      case "tab_neg":
        color = powerColor(sop, spec.theme);
        emissive = 0.3;
        break;
      case "cathode_sheet":
        color = spec.theme.cathodeColor;
        opacity = 0.95;
        break;
      case "anode_sheet":
        color = spec.theme.anodeColor;
        opacity = 0.95;
        break;
      case "separator":
        color = spec.theme.separatorColor;
        opacity = 0.35;
        break;
      case "electrolyte":
        color = spec.theme.electrolyteColor;
        opacity = 0.09;
        break;
      case "particles":
        color = spec.theme.metal;
        opacity = Math.max(0.25, 0.9 - 0.5 * (opts.dataScaled ? lostFraction : 0));
        break;
      case "sei_film":
        color = spec.theme.seiColor;
        opacity = filmScaled
          ? CELL_GEOMETRY.seiFilm.minOpacity +
            filmFraction * (CELL_GEOMETRY.seiFilm.maxOpacity - CELL_GEOMETRY.seiFilm.minOpacity)
          : 0.28;
        emissive = filmScaled ? 0.18 + 0.3 * filmFraction : 0.12;
        break;
      default:
        break;
    }

    // Highlight: a part the mechanism verdict points at is lit, without moving.
    const emphasis = spec.mechanism.emphasis;
    if (
      (emphasis === "sei" && part.id === "sei_film") ||
      (emphasis === "particles" && part.id === "particles")
    ) {
      emissive = Math.max(emissive, 0.55);
      opacity = Math.max(opacity, 0.45);
    }

    return {
      id: part.id,
      mesh,
      drawn,
      color,
      opacity,
      emissive,
      anchor: (placement?.anchor ?? [0, 0, 0]) as [number, number, number],
      label: part.label,
      value,
      carried: reading.carried,
      unit: part.unit,
      provenance: part.provenance,
      available: part.available,
      reason: part.unavailableReason,
    };
  });

  const radius =
    spec.cell.formFactor === "prismatic"
      ? Math.hypot(CELL_GEOMETRY.prismatic.width / 2, CELL_GEOMETRY.prismatic.depth / 2)
      : CELL_GEOMETRY.canRadius;
  const scaleNote = opts.dataScaled ? spec.geometryScales[dataScaledKey(spec)]?.note ?? null : null;

  return {
    parts,
    timeline,
    cursor,
    gauge,
    chrome: {
      // The stage sits below the cell's own floor rather than through it: a
      // floor plane at y = 0 would slice the casing in half.
      floor: translateMesh(extrudeClosed(ringPoints(Math.max(0.55, radius * 3.6), 64), 0.006), 0, -0.575, 0),
      shadow: translateMesh(
        extrudeClosed(ringPoints(Math.max(0.17, radius * 1.6), 40), 0.001), 0, -0.5705, 0,
      ),
    },
    bounds: { radius, height: 1.0 },
    scaleNote,
    formFactor: spec.cell.formFactor,
  };
}

/** The largest finite value in a series — the top of a data-driven range. */
export function seriesMax(series: Series | null): number | null {
  if (!series) return null;
  let max: number | null = null;
  for (const value of series) {
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    max = max === null ? value : Math.max(max, value);
  }
  return max;
}

/**
 * The state gauge: a band on the casing that fills as the cell loses capacity.
 *
 * This is the one place the scene shows a *fraction of a fresh cell* rather
 * than a physical layer, and it is the reason the drawn capacity loss is the
 * fitted total fade rather than an apportionment: the band's height is how much
 * usable capacity is left, which is identified for every cell, and it is the
 * quantity that moves when the timeline is scrubbed into the projection.
 *
 * Returns geometry in cell space with the band's base on the cell's own floor.
 */
export function gaugeBandMesh(spec: CellSceneSpec, fraction: number): Mesh {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
  const height = Math.max(0.012, clamped);
  if (spec.cell.formFactor === "prismatic") {
    const plate = boxMesh(0.05, height, CELL_GEOMETRY.prismatic.depth * 0.9);
    return translateMesh(plate, CELL_GEOMETRY.prismatic.width * 0.62, -0.5 + height / 2, 0);
  }
  const radius = CELL_GEOMETRY.canRadius * 1.012;
  const band = extrudeOpen(arcPoints(radius, (42 * Math.PI) / 180, 12, (-21 * Math.PI) / 180), height);
  return translateMesh(band, 0, -0.5 + height / 2, 0);
}

/**
 * Which declared mapping a data-scaled build is honouring, for the caption.
 *
 * The scene prints the spec's own sentence for whichever mapping it used, so
 * the user can always read what a drawn size means — including on the cell
 * whose channel split is unidentified, where the film is architecture and the
 * drawn, scaling quantity is the *identified* total fade instead.
 */
export function dataScaledKey(spec: CellSceneSpec): string {
  return spec.physics.splitIdentified === true ? "sei_film" : "fade_model";
}

/** The anatomy part ids, in the order both the spec and the renderer use. */
export function anatomyPartIds(): readonly PartId[] {
  return ANATOMY_PART_IDS;
}
