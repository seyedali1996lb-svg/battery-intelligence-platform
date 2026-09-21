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

/**
 * Revolve a profile around the Y axis.
 *
 * A `profile` point is `(radius, y)` in cell units; segments sweep the full
 * circle. Every top-of-cell part is a surface of revolution — a cap plate is
 * a disc with a step in it, a crimp is a groove — and a lathe draws that
 * honestly: one profile, one revolution, no stacked discs pretending to be a
 * formed part. Profiles are traced counter-clockwise in (radius, y) so the
 * outward normal faces the camera, the same convention `extrudeClosed` pins.
 */
export function latheMesh(profile: Array<[number, number]>, segments = 48): Mesh {
  const n = profile.length;
  if (n < 2 || segments < 3) return emptyMesh();
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    const [r, y] = profile[i];
    for (let s = 0; s < segments; s++) {
      const theta = (s / segments) * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
    }
  }
  for (let i = 0; i < n - 1; i++) {
    for (let s = 0; s < segments; s++) {
      const s1 = (s + 1) % segments;
      const a = i * segments + s;
      const b = i * segments + s1;
      const c = (i + 1) * segments + s;
      const d = (i + 1) * segments + s1;
      indices.push(a, c, d, a, d, b);
    }
  }
  return finalize(positions, indices);
}

/**
 * A tab as it is really made: a foil lead peeled off the coil's edge.
 *
 * A vertical strip standing on the roll's outermost turn at one angular
 * position, bending radially outward over the cap's thickness to meet the
 * terminal it feeds — the alternative was a small box floating beside the
 * roll, attached to nothing.
 */
function tabStripMesh(
  rollRadius: number,
  yBottom: number,
  yTop: number,
  canRadius: number,
  theta: number,
  capThickness: number,
  /** ±1: the direction from the terminal end back toward the roll. */
  bendToward: number,
): Mesh {
  const inner: [number, number] = [rollRadius * Math.cos(theta), rollRadius * Math.sin(theta)];
  const outer: [number, number] = [canRadius * Math.cos(theta), canRadius * Math.sin(theta)];
  const halfW = 0.02;
  const nx = -Math.sin(theta) * halfW;
  const nz = Math.cos(theta) * halfW;
  const yBend = yTop + bendToward * capThickness * 0.8;
  const positions: number[] = [
    inner[0] - nx, yBottom, inner[1] - nz,
    inner[0] + nx, yBottom, inner[1] + nz,
    inner[0] - nx, yBend, inner[1] - nz,
    inner[0] + nx, yBend, inner[1] + nz,
    outer[0] - nx, yTop, outer[1] - nz,
    outer[0] + nx, yTop, outer[1] + nz,
  ];
  const indices: number[] = [
    0, 1, 2, 1, 3, 2,
    2, 3, 4, 3, 5, 4,
    2, 4, 0, 3, 5, 1,
  ];
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
 * The stage furniture and the parts of the anatomy that are *not* drawn to a
 * dimension.
 *
 * The winding is the exception, and it moved out of here on purpose: a roll's
 * pitch, its turn count and the radius it fills are facts about the cell, so
 * they come from the document's `physical` block (`rollModel` below) rather
 * than from a constant in the renderer. What remains here is either real by
 * derivation (the sweep of the cut-away) or declared schematic (cap, vent,
 * terminals, particle count) — and the spec's own `schematic` list names the
 * second group so nobody has to guess which is which.
 */
export const CELL_GEOMETRY = {
  /** Fallback 18650 figures, used only when a document carries no `physical`. */
  canRadius: 9.2 / 65,
  /** Fraction of the casing's circumference drawn (the cut-away). */
  canSweep: (285 * Math.PI) / 180,
  /** 0.25 mm of 304 stainless — the format's typical can wall, in cell units. */
  canThickness: 0.25 / 65,
  /** A prismatic (CALCE 1.1 Ah pouch class) cell: 20.5 × 5.4 × 64 mm. */
  prismatic: { width: 20.5 / 65, depth: 5.4 / 65, thickness: 0.4 / 65 },
  roll: {
    height: 0.84,
    /**
     * Arc segments per turn of the spiral. The roll's finest detail is a band a
     * fraction of a pixel wide, so this is set by the *silhouette*: a turn's
     * polygon must not read as a polygon.
     */
    segmentsPerTurn: 32,
  },
  /**
   * Explode displacements at `exploded = 1`, in cell units. `layerGap` is per
   * layer step, chosen so the fully exploded stack still fits inside the
   * casing (a test asserts exactly that): a cutaway that flings the cathode
   * through the can wall teaches the wrong thing about where it lives.
   */
  explode: {
    cap: 0.3,
    vent: 0.36,
    terminalPos: 0.44,
    terminalNeg: 0.3,
    layerGap: 0.014,
    /**
     * The radial offset of each of the winding's five concentric members at
     * `exploded = 1`, in millimetres — the casing, the mandrel, and the three
     * ribbons, each with a number of its own rather than sharing one.
     *
     *   * `casing` and `mandrel` are the two *datums*: the can already sits on
     *     the cell's declared diameter and the core already sits where the
     *     winding starts, so neither has room to move without leaving the
     *     envelope the document declares. They are listed anyway — a table that
     *     names three of five members would make the other two an omission
     *     rather than a decision — and both are still multiplied by `exploded`,
     *     so a host that widens either one gets the motion it asked for.
     *   * `anode` is the air opened between the core and the first ribbon.
     *   * `separator` and `cathode` are measured *cumulatively* from the core,
     *     so the lane gaps are the differences (separator − anode, cathode −
     *     separator). Taking them cumulatively is what keeps the three ribbons
     *     tiling one turn's advance exactly: the air is part of the advance,
     *     not a fudge beside it, and the turn count falls to pay for it — so
     *     the exploded roll still cannot leave its envelope.
     *
     * A view control, like every other field of `exploded`: at rest all five
     * are zero, so the assembled cell is exactly the winding the document
     * declares, and no measurement anywhere depends on these figures.
     */
    radial: { casing: 0, mandrel: 0, anode: 0.25, separator: 0.7, cathode: 0.95 },
  },
  /** Draw-call economy: the whole particle cloud is one merged mesh. */
  particles: { count: 240, radius: 0.0055, anodeShare: 0.5 },
  /**
   * The drawn SEI film's thickness range, in cell units. Deliberately narrow:
   * it spans only the radial space the drawn roll leaves between the anode
   * ribbon and the separator, so a data-scaled film can never pierce a layer
   * it does not physically sit inside. The film's real thickness is in
   * nanometres and is derived in the document (`physical.film`); this band is
   * the *magnification* that makes it visible, and the document states the
   * factor. These figures are the fallback for a document written before that
   * derivation existed.
   */
  seiFilm: { anatomical: 0.0015, dataMax: 0.0048, minOpacity: 0.18, maxOpacity: 0.52 },
} as const;

// ---------------------------------------------------------------------------
// The physical roll: the winding is drawn from the document's own millimetres
// ---------------------------------------------------------------------------

/**
 * The top assembly, in cell units, as the renderer draws it.
 *
 * Every field is derived from the document's declared millimetres — the
 * derivation lives in `topAssemblyModel()`, and the document's `schematic`
 * list no longer contains the cap, vent or terminal because of it.
 */
export interface TopAssembly {
  /** Total cell height in cell units (the document's own ruler). */
  height: number;
  /** The can's inner bore radius, the space the windings actually live in. */
  boreRadius: number;
  /** Cap plate: its thickness, and the boss raised around the vent. */
  capThickness: number;
  capBossRadius: number;
  capBossHeight: number;
  ventRadius: number;
  terminalRadius: number;
  terminalStudHeight: number;
  /** The crimped bead: where it starts, and how tall it is. */
  crimpBottom: number;
  crimpTop: number;
  /** The heat-shrink jacket: outer radius, and where it stops. */
  wrapOuterRadius: number;
  wrapTop: number;
  wrapBottom: number;
  /** True when these came from the document rather than the fallback. */
  declared: boolean;
}

/** A 18650-class stack, foil to foil, in millimetres. Fallback only. */
const DEFAULT_STACK_MM = {
  copperFoil: 0.010,
  anodeCoating: 0.070,
  separator: 0.020,
  cathodeCoating: 0.060,
  aluminiumFoil: 0.015,
} as const;

/**
 * How much thicker than reality the stack is drawn at full explode.
 *
 * The assembled roll is drawn at its real pitch and its real turn count, where
 * a layer is a fraction of a pixel wide — that *is* an 18 mm roll on a screen,
 * and it reads as the dense winding it is. The exploded view is the legibility
 * state: it magnifies the stack until the separator is a pixel wide, and pays
 * for the magnification by drawing fewer turns into the same envelope. It is a
 * view control, like the explode position itself, and never a data claim.
 */
export const EXPLODE_STACK_GAIN = 7;

/** The winding, in cell units, derived from the document's declared millimetres. */
export interface RollModel {
  /** Millimetres per cell unit — the cell's own declared height. */
  unitMm: number;
  canRadius: number;
  canThickness: number;
  mandrelRadius: number;
  /** The radius the can's bore allows the roll, clearance included. */
  envelopeRadius: number;
  pitch: number;
  /** Turns a roll of this envelope and pitch has. Derived, never chosen. */
  turns: number;
  anodeThickness: number;
  separatorThickness: number;
  cathodeThickness: number;
  pitchMm: number;
  /** What this winding implies for the electrode, in metres. */
  electrodeLengthM: number;
  /** The projected area of the wound layers, in cm² — the cross-check number. */
  woundAreaCm2: number;
  /**
   * The top of the cell, derived from the document's `topAssembly` millimetres
   * (or the fallback when the document predates the block) and converted to
   * cell units. Every size the renderer draws above or below the windings
   * comes from here — no part of the casing is picked by eye.
   */
  top: TopAssembly;
  /**
   * False when the document carried no `physical` block, so these are the
   * renderer's fallback 18650 figures rather than the cell's own.
   */
  declared: boolean;
}

/** One drawn ribbon: where it starts inside a turn, and how thick it is drawn. */
export interface DrawnRibbon {
  id: "anode_sheet" | "separator" | "cathode_sheet";
  offset: number;
  thickness: number;
}

export interface DrawnRoll {
  /** Laps in the roll. A ribbon advances one lap and ends: see `spiralTurns`. */
  turns: number;
  /**
   * The revolutions one ribbon's *advance* covers, which is one fewer than the
   * roll's lap count: the last lap is where the ribbon ends, not another place
   * its radius grows to. Getting this wrong draws a roll one stack thicker
   * than its own envelope — which is exactly the bug this field fixed.
   */
  spiralTurns: number;
  /** Radial advance per turn. The three ribbons tile it exactly. */
  advance: number;
  gain: number;
  ribbons: DrawnRibbon[];
  /**
   * The air each of the five concentric members has been given at this explode
   * position, in cell units — zero for all of them at rest.
   */
  offsets: RadialOffsets;
  /**
   * The radius the winding's first ribbon starts from: the drawn core surface
   * (the mandrel, wherever the explode has put it) plus the air opened between
   * that core and the anode. Every ribbon's `offset` is measured *from* here,
   * so the bundle's base, the lanes and the air between them stay one
   * arithmetic rather than three that have to be kept in step.
   */
  base: number;
  /** The outermost edge of the wound layers. Never past the envelope. */
  outerRadius: number;
  anodeOuterRadius: number;
  declared: boolean;
}

/**
 * The cell's declared dimensions, or the renderer's fallback 18650.
 *
 * Nothing here is picked for looks: the pitch is the declared stack, the turn
 * count is what that pitch needs to fill the declared envelope, and the
 * electrode length is what those two imply. The one thing the renderer adds is
 * the fallback, flagged as undeclared so a host can say so.
 */
export function rollModel(spec: CellSceneSpec): RollModel {
  const physical = spec.physical ?? null;
  const unitMm = positive(physical?.unitsMmPerCellUnit, 65.0);
  const cyl = physical?.cylindrical ?? null;
  const roll = physical?.roll ?? null;
  const toUnits = (mm: number) => mm / unitMm;

  const diameterMm = positive(cyl?.diameterMm, 18.4);
  const wallMm = positive(cyl?.wallMm, 0.25);
  const clearanceMm = positive(cyl?.rollClearanceMm, 0.2);
  const mandrelDiameterMm = positive(
    roll?.mandrelDiameterMm ?? cyl?.mandrelDiameterMm,
    4.0,
  );
  const stack = roll?.stackMm ?? DEFAULT_STACK_MM;
  const pitchMm = positive(
    roll?.pitchMm,
    stack.copperFoil + stack.anodeCoating + stack.separator +
      stack.cathodeCoating + stack.aluminiumFoil,
  );

  const mandrelRadius = toUnits(mandrelDiameterMm / 2);
  const envelopeRadius = toUnits(diameterMm / 2 - wallMm - clearanceMm);
  const pitch = toUnits(pitchMm);
  const turns = Math.max(1, Math.floor((envelopeRadius - mandrelRadius) / pitch + 1e-9));

  // The electrode length and the wound cross-section are the two numbers that
  // let a reader check the drawing without trusting it: a 0.175 mm stack wound
  // out to this envelope implies both, and both are reported.
  let lengthMm = 0;
  for (let i = 0; i < turns; i++) {
    lengthMm += 2 * Math.PI * (mandrelDiameterMm / 2 + (i + 0.5) * pitchMm);
  }
  const woundAreaCm2 =
    (Math.PI * (Math.pow(envelopeRadius * unitMm, 2) - Math.pow(mandrelRadius * unitMm, 2))) / 100;

  const top = topAssemblyModel(spec);

  return {
    unitMm,
    canRadius: toUnits(diameterMm / 2),
    canThickness: toUnits(wallMm),
    mandrelRadius,
    envelopeRadius,
    pitch,
    turns,
    anodeThickness: toUnits(stack.copperFoil + stack.anodeCoating),
    separatorThickness: toUnits(stack.separator),
    cathodeThickness: toUnits(stack.aluminiumFoil + stack.cathodeCoating),
    pitchMm,
    electrodeLengthM: lengthMm / 1000,
    woundAreaCm2,
    top,
    declared: physical !== null && physical !== undefined,
  };
}

/** A declared value, or the fallback when the document left it out or nulled it. */
function positive(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Derive the drawn top of the cell from the document's declared millimetres.
 *
 * The old renderer hard-picked cap, vent and terminal sizes by eye; this
 * function is the replacement. Each figure is the declared millimetre size
 * converted to cell units — cap plate and boss, vent disc, terminal, the
 * crimp bead's place and height, the wrap's outer radius and where it stops —
 * and when the document predates the block the same fallback figures are used
 * while `declared` says so.
 */
export function topAssemblyModel(spec: CellSceneSpec): TopAssembly {
  const physical = spec.physical ?? null;
  const unitMm = positive(physical?.unitsMmPerCellUnit, 65.0);
  const cyl = physical?.cylindrical ?? null;
  const toUnits = (mm: number) => mm / unitMm;
  const diameterMm = positive(cyl?.diameterMm, 18.4);
  const wallMm = positive(cyl?.wallMm, 0.25);
  const canRadius = toUnits(diameterMm / 2);
  const boreRadius = toUnits(diameterMm / 2 - wallMm);
  const height = toUnits(positive(cyl?.heightMm, 65.0));
  const top = physical?.topAssembly ?? null;
  const declared = top !== null && top !== undefined;
  const capThickness = toUnits(positive(top?.capThicknessMm, 0.8));
  const capBossRadius = toUnits(positive(top?.capBossDiameterMm, 8.0) / 2);
  const capBossHeight = toUnits(positive(top?.capBossHeightMm, 0.5));
  const ventRadius = toUnits(positive(top?.ventDiameterMm, 4.5) / 2);
  const terminalRadius = toUnits(positive(top?.terminalDiameterMm, 5.5) / 2);
  const terminalStudHeight = toUnits(positive(top?.terminalStudHeightMm, 0.8));
  const crimpHeight = toUnits(positive(top?.crimpHeightMm, 0.6));
  const wrapThickness = toUnits(positive(top?.wrapThicknessMm, 0.15));
  const wrapTopSkip = toUnits(positive(top?.wrapTopSkipMm, 1.2));
  const wrapBottomSkip = toUnits(positive(top?.wrapBottomSkipMm, 1.2));
  return {
    height,
    boreRadius,
    capThickness,
    capBossRadius,
    capBossHeight,
    ventRadius,
    terminalRadius,
    terminalStudHeight,
    crimpBottom: 0.5 - crimpHeight,
    crimpTop: 0.5,
    wrapOuterRadius: canRadius + wrapThickness,
    wrapTop: 0.5 - wrapTopSkip,
    wrapBottom: -0.5 + wrapBottomSkip,
    declared,
  };
}

/**
 * The radial offset of one concentric member of the winding, in cell units:
 * the air opened outwards from where the assembled cell has it.
 *
 * The five members the roll is drawn as, kept in one object because they are
 * meaningless apart — a casing that knows nothing about where the mandrel sits
 * cannot say whether the roll between them still fits.
 */
export interface RadialOffsets {
  casing: number;
  mandrel: number;
  anode: number;
  separator: number;
  cathode: number;
}

/**
 * The five members' offsets at one explode position.
 *
 * At rest every value is zero, which is the assembled cell drawn as the
 * document's own winding. As `exploded` rises each member moves by its own
 * declared amount (`CELL_GEOMETRY.explode.radial`), and that is what turns the
 * dense roll — a fraction of a pixel per layer — into five concentric surfaces
 * a viewer can count.
 */
export function radialOffsets(model: RollModel, exploded: number): RadialOffsets {
  const e = Math.max(0, Math.min(1, Number.isFinite(exploded) ? exploded : 0));
  const table = CELL_GEOMETRY.explode.radial;
  const at = (mm: number) => (mm / model.unitMm) * e;
  return {
    casing: at(table.casing),
    mandrel: at(table.mandrel),
    anode: at(table.anode),
    separator: at(table.separator),
    cathode: at(table.cathode),
  };
}

/**
 * The roll as drawn at an explode position.
 *
 * At rest this is the real winding: the declared stack as the pitch, the turn
 * count the envelope gives it, each ribbon including its current collector.
 * Exploding magnifies the stack, opens air between the core and the first
 * ribbon and lanes between the three, and the turn count falls to pay for it —
 * so the roll never grows past the bore, at any explode position, by
 * construction rather than by a clamp.
 */
export function drawnRoll(model: RollModel, exploded: number): DrawnRoll {
  const offsets = radialOffsets(model, exploded);
  const e = Math.max(0, Math.min(1, Number.isFinite(exploded) ? exploded : 0));
  const gain = 1 + EXPLODE_STACK_GAIN * e;
  // Where the winding begins: the drawn core, plus the air the explode opened
  // between that core and the first ribbon. Both datums in the radial table
  // (casing, mandrel) are zero at rest, so this is the declared mandrel radius
  // until a host deliberately moves it.
  const base = model.mandrelRadius + offsets.mandrel + offsets.anode;
  // The air between the ribbons is the *difference* of the two cumulative
  // offsets — separator − anode, cathode − separator — so the table is read
  // once here instead of being restated as three unrelated gaps.
  const gap1 = Math.max(0, offsets.separator - offsets.anode);
  const gap2 = Math.max(0, offsets.cathode - offsets.separator);
  const span = model.envelopeRadius - base;
  // The air is part of the advance, not a fudge beside it: it is added into
  // every turn's pitch, and the turn count below falls to pay for it. That is
  // what keeps the exploded roll inside the bore by construction rather than
  // by a clamp — the envelope test only guards what this arithmetic guarantees.
  const advance = model.pitch * gain + gap1 + gap2;
  const turns = Math.max(2, Math.min(model.turns, Math.floor(span / advance + 1e-9)));
  const spiralTurns = Math.max(1, turns - 1);
  const anode = model.anodeThickness * gain;
  const separator = model.separatorThickness * gain;
  const cathode = model.cathodeThickness * gain;
  // The three ribbons tile one turn's advance exactly, at every explode
  // position — which is what keeps the drawn roll a filled roll.
  const ribbons: DrawnRibbon[] = [
    { id: "anode_sheet", offset: 0, thickness: anode },
    { id: "separator", offset: anode + gap1, thickness: separator },
    { id: "cathode_sheet", offset: anode + gap1 + separator + gap2, thickness: cathode },
  ];
  const outerRadius = Math.min(base + turns * advance, model.envelopeRadius);
  return {
    turns,
    spiralTurns,
    advance,
    gain,
    ribbons,
    offsets,
    base,
    outerRadius,
    anodeOuterRadius: base + (turns - 1) * advance + anode,
    declared: model.declared,
  };
}

/**
 * The band the drawn SEI film occupies, in cell units.
 *
 * The endpoints come from the document's own `physical.film.display` when it
 * has one — they are fixed by the roll's radial clearance and by what is
 * visible at a cell's scale, not by taste — and from the fallback constants
 * otherwise. Whichever it is, the band is a magnification of a thickness
 * measured in nanometres, which is why the document prints the factor: the
 * card carries the thickness, the layer on screen is the magnification.
 */
export function filmBand(spec: CellSceneSpec): { min: number; max: number; declared: boolean } {
  const display = (spec.physical?.film ?? null)?.display ?? null;
  const unitMm = positive(spec.physical?.unitsMmPerCellUnit, 65.0);
  if (display && display.drawnMinMm > 0 && display.drawnMaxMm > display.drawnMinMm) {
    return {
      min: display.drawnMinMm / unitMm,
      max: display.drawnMaxMm / unitMm,
      declared: true,
    };
  }
  return {
    min: CELL_GEOMETRY.seiFilm.anatomical,
    max: CELL_GEOMETRY.seiFilm.dataMax,
    declared: false,
  };
}

/** The prismatic envelope, in cell units, from the same declared block. */
export function prismaticModel(spec: CellSceneSpec): {
  width: number;
  thickness: number;
  wallThickness: number;
  declared: boolean;
} {
  const physical = spec.physical ?? null;
  const unitMm = positive(physical?.unitsMmPerCellUnit, 65.0);
  const prism = physical?.prismatic ?? null;
  return {
    width: positive(prism?.widthMm, 20.5) / unitMm,
    thickness: positive(prism?.thicknessMm, 5.4) / unitMm,
    wallThickness: positive(prism?.wallMm, 0.4) / unitMm,
    declared: physical !== null && physical !== undefined,
  };
}

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
  /**
   * What the winding actually is, and what is drawn — the numbers a host or a
   * test needs to see that "drawn to the datasheet" is true of this build.
   */
  roll: {
    declared: boolean;
    turns: number;
    drawnTurns: number;
    pitchMm: number;
    drawnPitchMm: number;
    electrodeLengthM: number;
    woundAreaCm2: number;
    outerRadiusMm: number;
    envelopeRadiusMm: number;
    gain: number;
  };
}

interface Placement {
  mesh: Mesh;
  anchor: [number, number, number];
  drawn?: boolean;
}

/**
 * The wound ribbons, memoised.
 *
 * A real winding is 38 laps, and rebuilding three of those from scratch on
 * every scrub tick costs most of a frame — but the winding does not depend on
 * the cursor at all, only on the declared dimensions and the explode position,
 * so the same three calls arrive over and over while a user drags the timeline.
 * Meshes here are treated as immutable (every transform in this module copies),
 * which is what makes handing the same one back twice safe.
 */
const _ribbonCache = new Map<string, Mesh>();
const RIBBON_CACHE_LIMIT = 16;

function _rollRibbon(
  layer: { innerRadius: number; turns: number; pitch: number; thickness: number },
  height: number,
): Mesh {
  const key = [layer.innerRadius, layer.turns, layer.pitch, layer.thickness, height].join("|");
  const cached = _ribbonCache.get(key);
  if (cached) return cached;
  const outline = spiralOutline({
    innerRadius: layer.innerRadius,
    turns: layer.turns,
    pitch: layer.pitch,
    thickness: layer.thickness,
    segmentsPerTurn: CELL_GEOMETRY.roll.segmentsPerTurn,
  });
  const mesh = extrudeClosed(outline, height);
  if (_ribbonCache.size >= RIBBON_CACHE_LIMIT) _ribbonCache.clear();
  _ribbonCache.set(key, mesh);
  return mesh;
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
  drawn: DrawnRoll,
): Array<[number, number, number]> {
  let state = 0x2f6e2b1;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
  const { height } = CELL_GEOMETRY.roll;
  const anode = drawn.ribbons.find((ribbon) => ribbon.id === "anode_sheet") as DrawnRibbon;
  const cathode = drawn.ribbons.find((ribbon) => ribbon.id === "cathode_sheet") as DrawnRibbon;
  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < count; i++) {
    const inner = rand() < CELL_GEOMETRY.particles.anodeShare;
    const lane = inner ? anode : cathode;
    // A particle rides a *turn* of its own coating: the annulus that ribbon
    // actually occupies on that turn. So the cloud scales with the winding and
    // never drifts off the layers it belongs to as the stack is exploded.
    const turn = Math.min(drawn.turns - 1, Math.floor(rand() * drawn.turns));
    const radius =
      drawn.base +
      turn * drawn.advance +
      lane.offset +
      lane.thickness * (0.25 + 0.5 * rand());
    const theta = rand() * Math.PI * 2;
    const y = (rand() - 0.5) * height * 0.94;
    out.push([radius * Math.cos(theta), y, radius * Math.sin(theta)]);
  }
  return out;
}

function _cylindricalPlacements(
  exploded: number,
  filmThickness: number,
  spec: CellSceneSpec,
  model: RollModel,
  drawn: DrawnRoll,
): Record<string, Placement> {
  const { canSweep, roll, explode } = CELL_GEOMETRY;
  // The casing is a datum, so it sits on the declared diameter — but its offset
  // is still read from the radial table, so a host that gives the can room to
  // move gets the motion it asked for and everything measured *inside* the can
  // (the bore, the wrap, the cap that closes it) moves with it.
  const canRadius = model.canRadius + drawn.offsets.casing;
  const canThickness = model.canThickness;
  const casing = extrudeOpen(arcPoints(canRadius, canSweep, 64, Math.PI * 0.6), 1.0);
  const floorDisc = extrudeClosed(ringPoints(canRadius, 64), 0.012);
  const canMesh = mergeMeshes([casing, translateMesh(floorDisc, 0, -0.494, 0)]);

  // The winding's core: the mandrel, drawn where the document's stack starts
  // it and moved by its own declared offset — the datum the three ribbons are
  // wound around, which is why it is drawn as a part rather than implied by
  // the hole in the roll.
  const mandrelRadius = model.mandrelRadius + drawn.offsets.mandrel;
  const mandrel = extrudeClosed(ringPoints(mandrelRadius, 32), roll.height);

  // One ribbon per drawn layer, each starting at its own place inside the turn
  // and wound at the drawn advance — so the roll is the document's stack, not
  // the renderer's idea of one.
  const ribbon = (id: DrawnRibbon["id"], height: number): Mesh => {
    const lane = drawn.ribbons.find((candidate) => candidate.id === id) as DrawnRibbon;
    return _rollRibbon(
      {
        innerRadius: drawn.base + lane.offset,
        turns: drawn.spiralTurns,
        pitch: drawn.advance,
        thickness: lane.thickness,
      },
      height,
    );
  };
  // The film is the one drawn size that is not millimetres (see the spec's
  // `schematic` list), so it is magnified with the stack it sits in rather than
  // left at its assembled size inside a stack the explode just doubled.
  const film = filmThickness * drawn.gain;
  const anodeOuter = drawn.anodeOuterRadius;

  // The top of the cell, formed rather than stacked: every size below is a
  // declared millimetre from the document's `topAssembly`, lathed into the
  // surface of revolution the real part is. With no declared block the same
  // derivation runs on the 18650 fallbacks and `top.declared` says so.
  const top = topAssemblyModel(spec);
  const topCapY = 0.5 + explode.cap * exploded;
  // The stack above the rim is flush by construction: the boss tops the plate,
  // the vent sits in the boss, the button stands on the boss. `extrudeClosed`
  // centres its extrusion, so discs and buttons are translated to their *mid*
  // height and every span below is written out to keep them honest.
  const ventY = topCapY + top.capBossHeight + explode.vent * exploded;
  const terminalPosTopY = ventY + top.terminalStudHeight;
  const bottomTerminalY = -0.5 - explode.terminalNeg * exploded;
  const tabTopPos = topCapY - top.capThickness;
  const tabTopNeg = bottomTerminalY;
  const rollTopRadius = drawn.anodeOuterRadius;
  // The tabs leave the coil at two angular positions and rise to the
  // terminals they feed — the assembled cell's current path, drawn attached.
  const THETA_POS = Math.PI * 0.1;
  const THETA_NEG = Math.PI * 1.1;

  const anodeLane = drawn.ribbons.find((r) => r.id === "anode_sheet") ?? drawn.ribbons[0];
  const sepLane = drawn.ribbons.find((r) => r.id === "separator") ?? drawn.ribbons[1];
  const cathodeLane = drawn.ribbons.find((r) => r.id === "cathode_sheet") ?? drawn.ribbons[2];
  const midTurn = Math.max(0, Math.floor(drawn.turns / 2));
  const anodeR = drawn.base + midTurn * drawn.advance + anodeLane.offset + anodeLane.thickness * 0.5;
  const sepR = drawn.base + midTurn * drawn.advance + sepLane.offset + sepLane.thickness * 0.5;
  const cathodeR = drawn.base + midTurn * drawn.advance + cathodeLane.offset + cathodeLane.thickness * 0.5;
  const THETA_ANODE = Math.PI * 0.45;
  const THETA_SEP = Math.PI * 0.85;
  const THETA_CATHODE = Math.PI * 1.35;
  const THETA_SEI = Math.PI * 1.75;
  const THETA_MANDREL = Math.PI * 0.3;

  return {
    can: { mesh: canMesh, anchor: [canRadius * 0.98, 0.34, canRadius * 0.2] },
    mandrel: {
      // Anchored inside the cut-away sector, so "frame this part" turns the
      // camera to look *into* the cell at the core rather than at the can.
      mesh: mandrel,
      anchor: [
        mandrelRadius * Math.cos(THETA_MANDREL),
        -roll.height * 0.3,
        mandrelRadius * Math.sin(THETA_MANDREL),
      ],
    },
    wrap: {
      // The jacket is part of the casing, so the cut-away opens it too —
      // a full tube would hide the wound stack behind a printed skin.
      // Translated because `extrudeOpen` centres: the declared skips are
      // measured from the ends, not from the middle.
      mesh: translateMesh(
        extrudeOpen(arcPoints(top.wrapOuterRadius + drawn.offsets.casing, canSweep, 64, Math.PI * 0.6), top.wrapTop - top.wrapBottom),
        0,
        (top.wrapTop + top.wrapBottom) / 2,
        0,
      ),
      anchor: [
        (top.wrapOuterRadius + drawn.offsets.casing) * Math.cos(Math.PI * 1.5),
        0,
        (top.wrapOuterRadius + drawn.offsets.casing) * Math.sin(Math.PI * 1.5),
      ],
    },
    cap: {
      // Plate, boss and the groove the crimping die rolled the can wall into:
      // one profile, one revolution — a formed part, not stacked discs.
      mesh: translateMesh(
        latheMesh(
          [
            [canRadius, -top.capThickness],
            [canRadius, 0],
            [top.capBossRadius, 0],
            [top.capBossRadius, top.capBossHeight],
            [top.ventRadius, top.capBossHeight],
            [top.ventRadius, 0],
            [0, 0],
            [0, -top.capThickness],
          ],
          48,
        ),
        0,
        topCapY,
        0,
      ),
      anchor: [canRadius * 0.8, topCapY, 0],
    },
    vent: {
      mesh: translateMesh(extrudeClosed(ringPoints(top.ventRadius, 24), 0.008), 0, ventY, 0),
      anchor: [-canRadius * 0.85, ventY + 0.004, 0],
    },
    crimp: {
      mesh: translateMesh(
        latheMesh(
          [
            [canRadius, top.crimpBottom],
            [canRadius + canThickness, top.crimpBottom + canThickness],
            [canRadius + canThickness, top.crimpTop],
            [canRadius, top.crimpTop],
          ],
          48,
        ),
        0,
        0,
        0,
      ),
      anchor: [canRadius * 1.1, (top.crimpBottom + top.crimpTop) / 2, 0],
    },
    terminal_pos: {
      mesh: translateMesh(
        extrudeClosed(ringPoints(top.terminalRadius, 28), top.terminalStudHeight),
        0,
        ventY + top.terminalStudHeight / 2,
        0,
      ),
      anchor: [canRadius * 0.9, terminalPosTopY, 0],
    },
    terminal_neg: {
      mesh: translateMesh(
        extrudeClosed(ringPoints(top.terminalRadius, 28), top.terminalStudHeight),
        0,
        bottomTerminalY - top.terminalStudHeight / 2,
        0,
      ),
      anchor: [-canRadius * 0.9, bottomTerminalY - top.terminalStudHeight, 0],
    },
    tab_pos: {
      mesh: tabStripMesh(rollTopRadius, roll.height / 2, tabTopPos, canRadius, THETA_POS, top.capThickness, -1),
      anchor: [
        rollTopRadius * 0.7 * Math.cos(THETA_POS),
        roll.height / 2 + 0.05,
        rollTopRadius * 0.7 * Math.sin(THETA_POS),
      ],
    },
    tab_neg: {
      mesh: tabStripMesh(rollTopRadius, -roll.height / 2, tabTopNeg, canRadius, THETA_NEG, top.capThickness, 1),
      anchor: [
        rollTopRadius * 0.7 * Math.cos(THETA_NEG),
        -roll.height / 2 - 0.05,
        rollTopRadius * 0.7 * Math.sin(THETA_NEG),
      ],
    },
    anode_sheet: {
      mesh: ribbon("anode_sheet", roll.height),
      anchor: [anodeR * Math.cos(THETA_ANODE), roll.height * 0.15, anodeR * Math.sin(THETA_ANODE)],
    },
    separator: {
      mesh: ribbon("separator", roll.height),
      anchor: [sepR * Math.cos(THETA_SEP), -roll.height * 0.12, sepR * Math.sin(THETA_SEP)],
    },
    cathode_sheet: {
      mesh: ribbon("cathode_sheet", roll.height),
      anchor: [cathodeR * Math.cos(THETA_CATHODE), roll.height * 0.28, cathodeR * Math.sin(THETA_CATHODE)],
    },
    electrolyte: {
      mesh: extrudeClosed(ringPoints(canRadius - canThickness, 48), roll.height * 0.98),
      anchor: [(canRadius - canThickness) * 0.75 * Math.cos(Math.PI * 0.2), -0.15, (canRadius - canThickness) * 0.75 * Math.sin(Math.PI * 0.2)],
    },
    sei_film: {
      // The film is drawn where the SEI actually forms: on the anode's own
      // surface, at the anode–separator boundary.
      mesh: extrudeClosed(ringPoints(anodeOuter + film / 2, 48), roll.height * 0.98),
      anchor: [(anodeOuter + film / 2) * Math.cos(THETA_SEI), 0.18, (anodeOuter + film / 2) * Math.sin(THETA_SEI)],
    },
    particles: {
      mesh: emptyMesh(),
      anchor: [cathodeR * Math.cos(Math.PI * 0.6), -0.22, cathodeR * Math.sin(Math.PI * 0.6)],
    },
  };
}

function _prismaticPlacements(
  exploded: number,
  filmThickness: number,
  prism: { width: number; thickness: number; wallThickness: number },
  top: TopAssembly,
): Record<string, Placement> {
  const { roll, explode } = CELL_GEOMETRY;
  const width = prism.width;
  const depth = prism.thickness;
  const wall = Math.min(prism.wallThickness, depth / 4);
  const halfWidth = width / 2;
  const halfDepth = depth / 2;
  // The shell: four walls and a floor, drawn translucent so the stack is visible.
  const walls = mergeMeshes([
    translateMesh(boxMesh(width, 1.0, wall), 0, 0, halfDepth - wall / 2),
    translateMesh(boxMesh(width, 1.0, wall), 0, 0, -(halfDepth - wall / 2)),
    translateMesh(boxMesh(wall, 1.0, depth - 2 * wall), halfWidth - wall / 2, 0, 0),
    translateMesh(boxMesh(wall, 1.0, depth - 2 * wall), -(halfWidth - wall / 2), 0, 0),
    translateMesh(boxMesh(width, wall, depth), 0, -0.5 + wall / 2, 0),
  ]);
  const slab = (z: number, thickness: number, height: number) =>
    translateMesh(boxMesh(width * 0.9, height, thickness), 0, 0, z);

  // The stack is a diagram (a prismatic cell's layer count is a winding or
  // stacking choice this platform's data does not carry), but it is confined
  // *by construction*: the outermost slab the explode can reach is one wall
  // thickness inside the declared envelope, at any explode position.
  const slabThickness = Math.min(0.012, depth * 0.22);
  const zLimit = Math.max(slabThickness, halfDepth - wall - slabThickness / 2);
  const spread = (base: number): number => base + (Math.sign(base) * zLimit - base) * exploded;
  const anodeZ = spread(-Math.min(0.028, zLimit * 0.8));
  const cathodeZ = spread(Math.min(0.028, zLimit * 0.8));
  return {
    can: { mesh: walls, anchor: [halfWidth, 0.3, halfDepth] },
    wrap: {
      // A prismatic jacket is a printed sleeve on the flat, not a shrink tube:
      // two skinned faces, and skipped at the ends like the cylindrical one.
      mesh: mergeMeshes([
        translateMesh(boxMesh(width + 0.012, top.wrapTop - top.wrapBottom, 0.006), 0, (top.wrapTop + top.wrapBottom) / 2, halfDepth + 0.004),
        translateMesh(boxMesh(width + 0.012, top.wrapTop - top.wrapBottom, 0.006), 0, (top.wrapTop + top.wrapBottom) / 2, -halfDepth - 0.004),
      ]),
      anchor: [halfWidth, 0.3, halfDepth + 0.01],
    },
    crimp: {
      // The laser-welded rim: a lip around the open top, reading as a bead.
      mesh: translateMesh(boxMesh(width + 0.01, 0.014, depth + 0.01), 0, 0.493, 0),
      anchor: [halfWidth, 0.493, halfDepth],
    },
    cap: {
      mesh: translateMesh(boxMesh(width, 0.02, depth), 0, 0.5 + explode.cap * exploded, 0),
      anchor: [halfWidth, 0.5 + explode.cap * exploded, 0],
    },
    vent: {
      mesh: translateMesh(boxMesh(0.06, 0.008, 0.04), 0.08, 0.51 + explode.vent * exploded, 0),
      anchor: [-0.08, 0.511 + explode.vent * exploded, 2 * depth],
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
    anode_sheet: { mesh: slab(anodeZ, slabThickness, roll.height), anchor: [0, roll.height / 2 + 0.12, anodeZ] },
    separator: {
      mesh: slab(0, slabThickness * 0.67, roll.height * 0.98),
      anchor: [0, -roll.height / 2 - 0.06, 0],
    },
    cathode_sheet: { mesh: slab(cathodeZ, slabThickness, roll.height), anchor: [0, roll.height / 2 + 0.2, cathodeZ] },
    electrolyte: { mesh: slab(0, depth - 2 * wall - slabThickness, roll.height * 0.99), anchor: [0, -0.14, 0] },
    sei_film: {
      mesh: slab(anodeZ + slabThickness * 0.6 + filmThickness / 2, filmThickness, roll.height * 0.98),
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
  const splitIdentified = spec.physics.splitIdentified === true;
  // The drawn film's cursor reading comes from the series its own scale names,
  // so a document that states a thickness in nanometres drives the geometry
  // with it, and a document that only states a fitted share still renders.
  const seiScale = spec.geometryScales.sei_film;
  const filmSeries =
    seiScale?.from === "series.seiThicknessNm" ? (spec.series.seiThicknessNm ?? null) : spec.series.seiPct;
  const seiAtCursor = readingAt(filmSeries, partCursor).value;

  // The drawn film's thickness maps the cell's own derived nanometres — the
  // film it left formation with → the largest the fit puts on it — onto the
  // band `filmBand` returns. Both endpoints are read from the document, so a
  // cell with a small film is not drawn identically to a cell with a large one,
  // and the magnification the band implies is the document's own number rather
  // than something this file chose.
  const filmScaled = opts.dataScaled && splitIdentified;
  const band = filmBand(spec);
  const filmFraction = filmScaled
    ? normalize(
        seiAtCursor,
        seiScale?.displayMin ?? 0,
        // Older documents carry the fitted share rather than a thickness: fall
        // back to whatever series the scale actually points at.
        seiScale?.displayMax ?? seriesMax(spec.series.seiThicknessNm ?? spec.series.seiPct),
        0,
      )
    : 0;
  const filmThickness = filmScaled ? band.min + filmFraction * (band.max - band.min) : band.min;

  // The winding the scene draws: derived from the document's declared
  // millimetres, and rebuilt at each explode position (the explode is an
  // un-winding, and the assembled state is the real roll).
  const model = rollModel(spec);
  const drawn = drawnRoll(model, exploded);
  const prism = prismaticModel(spec);
  const placements =
    spec.cell.formFactor === "prismatic"
      ? _prismaticPlacements(exploded, filmThickness, prism, topAssemblyModel(spec))
      : _cylindricalPlacements(exploded, filmThickness, spec, model, drawn);

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
  const cloud = _particlePositions(CELL_GEOMETRY.particles.count, drawn);
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
      case "mandrel":
        // The steel winding core: opaque, because unlike the can it is not a
        // shell the viewer is meant to see through.
        color = spec.theme.metal;
        opacity = 0.95;
        break;
      case "wrap":
        // The heat-shrink jacket: deliberately not the can's metal colour, so
        // the skin and the steel read as two materials at a glance.
        color = spec.theme.accent;
        opacity = 0.55;
        break;
      case "crimp":
        color = spec.theme.metal;
        opacity = 0.95;
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
      ? Math.hypot(prism.width / 2, prism.thickness / 2)
      : model.canRadius;
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
    roll: {
      declared: drawn.declared,
      turns: model.turns,
      drawnTurns: drawn.turns,
      pitchMm: model.pitchMm,
      drawnPitchMm: drawn.advance * model.unitMm,
      electrodeLengthM: model.electrodeLengthM,
      woundAreaCm2: model.woundAreaCm2,
      outerRadiusMm: drawn.outerRadius * model.unitMm,
      envelopeRadiusMm: model.envelopeRadius * model.unitMm,
      gain: drawn.gain,
    },
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
