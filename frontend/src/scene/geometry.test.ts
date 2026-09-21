/**
 * Headless tests for the scene geometry (Node's own test runner — the frontend
 * has no other test framework, and this module needs none: it is pure arrays).
 *
 * `node --test` runs these, so they are cheap enough to be a gate rather than a
 * ritual. What they protect, in order of how much it matters:
 *
 * 1. **Orientation and closure of every solid.** A cap facing the wrong way is
 *    invisible from outside and looks like a hole from inside — a defect no
 *    numeric readout would ever catch, and one nobody can see without a
 *    browser. Pinned here by normal direction and vertex counts.
 * 2. **The draws/no-draw line.** Every anatomy part has geometry and exactly
 *    one card; a part whose source carries no measurement is still *drawn* while
 *    carrying no number. That is the difference between architecture and a
 *    claim, and it is asserted, not documented.
 * 3. **Refusals reaching the geometry.** An unidentifiable SEI/LAM split must
 *    turn off film growth and particle loss — the two places where a wrong
 *    number would be drawn as a plausible-looking object.
 * 4. **The scene is the same scene across hosts.** Colours come from the spec's
 *    theme, so a re-themed host repaints the cell, and two builds of one spec
 *    are byte-identical (no seeded randomness leaking into a render).
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { THEME, makePhysical, makeSpec } from "./fixture.ts";
import {
  CELL_GEOMETRY,
  DEFAULT_BUILD_OPTIONS,
  arcPoints,
  buildScene,
  filmBand,
  buildTimeline,
  ensureCounterClockwise,
  extrudeClosed,
  extrudeOpen,
  mergeBuildOptions,
  mergeMeshes,
  partReadings,
  readingAt,
  rollModel,
  ringPoints,
  scaleMesh,
  seriesMax,
  signedArea,
  sphereMesh,
  spiralOutline,
  todayCursor,
  translateMesh,
  triangleCount,
  vertexCount,
} from "./geometry.ts";
import type { Mesh } from "./geometry.ts";
import { bandFor, mixHex, normalize, powerColor, sohColor } from "./theme.ts";
import { ANATOMY_PART_IDS } from "./types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function bounds(mesh: Mesh) {
  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  let minZ = Infinity; let maxZ = -Infinity;
  for (let i = 0; i < mesh.positions.length; i += 3) {
    minX = Math.min(minX, mesh.positions[i]); maxX = Math.max(maxX, mesh.positions[i]);
    minY = Math.min(minY, mesh.positions[i + 1]); maxY = Math.max(maxY, mesh.positions[i + 1]);
    minZ = Math.min(minZ, mesh.positions[i + 2]); maxZ = Math.max(maxZ, mesh.positions[i + 2]);
  }
  return { minX, maxX, minY, maxY, minZ, maxZ };
}

function radialExtent(mesh: Mesh) {
  let min = Infinity; let max = 0;
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const r = Math.hypot(mesh.positions[i], mesh.positions[i + 2]);
    min = Math.min(min, r);
    max = Math.max(max, r);
  }
  return { min, max };
}

function partOf(scene: ReturnType<typeof buildScene>, id: string) {
  const part = scene.parts.find((p) => p.id === id);
  assert.ok(part, `no built part ${id}`);
  return part;
}

function assertMeshSane(mesh: Mesh, what: string) {
  const vertices = vertexCount(mesh);
  for (let i = 0; i < mesh.positions.length; i++) {
    assert.ok(Number.isFinite(mesh.positions[i]), `${what}: non-finite position`);
  }
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const len = Math.hypot(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]);
    assert.ok(Math.abs(len - 1) < 1e-3 || len === 0, `${what}: normal is not unit length`);
  }
  for (const index of mesh.indices) {
    assert.ok(index < vertices, `${what}: index ${index} exceeds ${vertices} vertices`);
  }
  assert.equal(mesh.positions.length % 3, 0, `${what}: ragged positions`);
}

// ---------------------------------------------------------------------------
// Primitives: orientation, closure, counts
// ---------------------------------------------------------------------------

test("a counter-clockwise outline in (x, z) is what the extruder expects", () => {
  const ring = ringPoints(1, 32);
  assert.ok(signedArea(ring) > 0, "ringPoints must be counter-clockwise in (x, z)");
  const reversed = [...ring].reverse();
  assert.ok(signedArea(reversed) < 0);
  assert.ok(signedArea(ensureCounterClockwise(reversed)) > 0, "ensureCounterClockwise did not fix the winding");
});

/**
 * Every face of a closed solid must point away from the solid's inside.
 *
 * Checked per triangle rather than per vertex on purpose: on a cylinder the
 * side and cap rings share y coordinates, so a vertex-level rule cannot tell
 * which faces a vertex belongs to — and the two seam vertices genuinely carry
 * a mixture. Per-face is the property that actually matters (a backwards face
 * is invisible from one side and a hole from the other).
 */
function assertOutwardFaces(mesh: Mesh, what: string): { sideFaces: number; capUp: number; capDown: number } {
  let sideFaces = 0;
  let capUp = 0;
  let capDown = 0;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
    const at = (v: number, o: number) => mesh.positions[v * 3 + o];
    const ux = at(b, 0) - at(a, 0);
    const uy = at(b, 1) - at(a, 1);
    const uz = at(b, 2) - at(a, 2);
    const vx = at(c, 0) - at(a, 0);
    const vy = at(c, 1) - at(a, 1);
    const vz = at(c, 2) - at(a, 2);
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    assert.ok(len > 0, `${what}: degenerate triangle`);
    const cx = (at(a, 0) + at(b, 0) + at(c, 0)) / 3;
    const cy = (at(a, 1) + at(b, 1) + at(c, 1)) / 3;
    const cz = (at(a, 2) + at(b, 2) + at(c, 2)) / 3;
    if (Math.abs(ny) / len > 0.5) {
      // A cap face must face away from the solid's own middle.
      if (ny > 0) {
        assert.ok(cy > 0, `${what}: an upward cap face sits below the centre`);
        capUp++;
      } else {
        assert.ok(cy < 0, `${what}: a downward cap face sits above the centre`);
        capDown++;
      }
    } else {
      const radial = Math.hypot(cx, cz);
      const dot = (nx * cx + nz * cz) / (len * (radial || 1));
      assert.ok(dot > 0.2, `${what}: a side face points inward or sideways (dot=${dot.toFixed(3)})`);
      sideFaces++;
    }
  }
  assert.ok(sideFaces > 0, `${what}: no side faces were generated`);
  assert.ok(capUp > 0 && capDown > 0, `${what}: a cap is missing`);
  return { sideFaces, capUp, capDown };
}

test("the top cap of an extruded solid faces up and the bottom faces down", () => {
  // A triangle whose winding is known by construction, extruded. It is centred
  // on the axis because the outward-face check reasons from the mesh's middle.
  const outline: Array<[number, number]> = [[-1, -1], [1, -1], [0, 1]];
  assert.ok(signedArea(outline) > 0);
  const mesh = extrudeClosed(outline, 2);
  // Three vertices → one fan triangle per cap, two triangles per side quad.
  const faces = assertOutwardFaces(mesh, "triangular prism");
  assert.deepEqual(faces, { sideFaces: 6, capUp: 1, capDown: 1 });
  // Flipping the outline's winding must not flip the solid inside out: the
  // extruder normalises the winding it is handed.
  const flipped = extrudeClosed([...outline].reverse(), 2);
  assert.deepEqual(assertOutwardFaces(flipped, "reversed winding"), faces);
});

test("every face of a cylinder points outwards", () => {
  assertOutwardFaces(extrudeClosed(ringPoints(0.5, 48), 1), "cylinder");
  assertOutwardFaces(extrudeClosed(ringPoints(0.2, 24), 0.05), "thin disc");
});

test("a closed extrusion has two vertices per outline point and is watertight by count", () => {
  const outline = ringPoints(0.3, 16);
  const mesh = extrudeClosed(outline, 0.5);
  assert.equal(vertexCount(mesh), outline.length * 2);
  // 2 triangles per side segment + (n - 2) per cap, two caps.
  assert.equal(triangleCount(mesh), outline.length * 2 + 2 * (outline.length - 2));
  assertMeshSane(mesh, "cylinder");
});

test("a jelly-roll ribbon is a closed outline spanning its own radial band", () => {
  const outline = spiralOutline({ innerRadius: 0.03, turns: 3, pitch: 0.008, thickness: 0.005 });
  assert.ok(signedArea(outline) > 0, "the ribbon outline must be counter-clockwise");
  const radii = outline.map(([x, z]) => Math.hypot(x, z));
  const inner = Math.min(...radii);
  const outer = Math.max(...radii);
  // The ribbon starts on the inner radius and ends on the outer one plus its
  // own thickness — that is the whole shape: two spirals, joined at both ends.
  assert.ok(Math.abs(inner - 0.03) < 1e-9, `inner radius drifted: ${inner}`);
  assert.ok(Math.abs(outer - (0.03 + 3 * 0.008 + 0.005)) < 1e-9, `outer radius drifted: ${outer}`);
  assert.equal(outline.length, 2 * (3 * 36 + 1));
  assertMeshSane(extrudeClosed(outline, 0.8), "ribbon");
});

test("an open extrusion emits both faces so the cut-away casing never disappears", () => {
  const path = arcPoints(0.5, Math.PI, 12);
  const mesh = extrudeOpen(path, 1);
  assert.equal(vertexCount(mesh), path.length * 2);
  assert.equal(triangleCount(mesh), (path.length - 1) * 2 * 2);
  const withoutBothWays = extrudeOpen(path, 1);
  assert.equal(triangleCount(withoutBothWays), (path.length - 1) * 4);
});

test("degenerate inputs return an empty mesh rather than throwing or emitting NaN", () => {
  assert.equal(vertexCount(extrudeClosed([[0, 0], [1, 0]], 1)), 0);
  assert.equal(vertexCount(extrudeClosed(ringPoints(1, 8), 0)), 0);
  assert.equal(vertexCount(extrudeOpen([[0, 0]], 1)), 0);
  assert.equal(vertexCount(sphereMesh(0.1, 2, 2)) > 0, true);
});

test("merging meshes reindexes rather than aliasing vertices", () => {
  const a = sphereMesh(0.05, 6, 4);
  const b = translateMesh(sphereMesh(0.05, 6, 4), 1, 0, 0);
  const merged = mergeMeshes([a, b]);
  assert.equal(vertexCount(merged), vertexCount(a) + vertexCount(b));
  assert.equal(triangleCount(merged), triangleCount(a) + triangleCount(b));
  assertMeshSane(merged, "merged cloud");
  assert.ok(bounds(merged).maxX > 0.9, "the translated copy did not move");
});

test("scaling a mesh keeps its normals unit length", () => {
  const scaled = scaleMesh(extrudeClosed(ringPoints(0.2, 24), 0.4), 0.5, 1, 0.5);
  assertMeshSane(scaled, "scaled cylinder");
});

// ---------------------------------------------------------------------------
// Reading a measurement at a cursor
// ---------------------------------------------------------------------------

test("a gap in a series holds the last measured value and says it did", () => {
  const series = [1, null, null, 4, null];
  const atZero = readingAt(series, 0);
  assert.deepEqual(atZero, { value: 1, carried: false, inRange: true });
  const inGap = readingAt(series, 2);
  assert.deepEqual(inGap, { value: 1, carried: true, inRange: true });
  const past = readingAt(series, 9);
  assert.equal(past.value, 4);
  assert.equal(past.inRange, false, "a cursor past the series must report out of range");
});

test("a series that never had a reading reports none rather than zero", () => {
  assert.deepEqual(readingAt([null, null], 1), { value: null, carried: false, inRange: true });
  assert.deepEqual(readingAt(null, 0), { value: null, carried: false, inRange: false });
});

test("seriesMax ignores gaps and returns null for an empty series", () => {
  assert.equal(seriesMax([null, 3, null, 7]), 7);
  assert.equal(seriesMax([null, null]), null);
  assert.equal(seriesMax(null), null);
});

// ---------------------------------------------------------------------------
// The timeline: measured replay, then the platform's projection
// ---------------------------------------------------------------------------

test("a refused projection adds no future to the timeline", () => {
  const spec = makeSpec({ projection: "refused" });
  const timeline = buildTimeline(spec);
  assert.equal(timeline.hasProjection, false);
  assert.equal(timeline.cycles.length, spec.record.nCycles);
  assert.ok(timeline.projected.every((flag) => flag === false));
  assert.equal(todayCursor(timeline), spec.record.nCycles - 1);
});

test("an available projection extends the timeline and is flagged as projected throughout", () => {
  const spec = makeSpec({ projection: "available" });
  const timeline = buildTimeline(spec);
  assert.equal(timeline.hasProjection, true);
  assert.equal(timeline.cycles.length, spec.record.nCycles + (spec.projection.cycles?.length ?? 0));
  const firstProjected = timeline.measuredCount;
  for (let i = 0; i < timeline.cycles.length; i++) {
    assert.equal(timeline.projected[i], i >= firstProjected, `projected flag wrong at ${i}`);
  }
  // The band exists only in the future half.
  for (let i = 0; i < firstProjected; i++) assert.equal(timeline.sohQ10Pct[i], null);
  assert.ok(timeline.sohQ10Pct[firstProjected] !== null);
});

// ---------------------------------------------------------------------------
// The anatomy: bijection, form factors, explode
// ---------------------------------------------------------------------------

test("the scene draws all thirteen parts and every part has sane geometry", () => {
  const scene = buildScene(makeSpec({ projection: "available" }), { cursor: 10 });
  assert.equal(scene.parts.length, 13);
  assert.deepEqual(
    [...new Set(scene.parts.map((p) => p.id))].sort(),
    [...ANATOMY_PART_IDS].sort(),
  );
  for (const part of scene.parts) {
    assert.equal(part.drawn, true, `${part.id} has no geometry`);
    assertMeshSane(part.mesh, part.id);
    assert.ok(part.anchor.every(Number.isFinite), `${part.id} has a non-finite label anchor`);
  }
});

test("an unavailable part is still drawn — architecture without a claim", () => {
  const spec = makeSpec({ unavailable: ["separator", "electrolyte", "cap"] });
  const scene = buildScene(spec, { cursor: 5 });
  for (const id of ["separator", "electrolyte", "cap"] as const) {
    const part = partOf(scene, id);
    assert.equal(part.available, false);
    assert.ok(part.reason, `${id} is unavailable with no reason`);
    assert.equal(part.drawn, true, `${id} vanished because it carries no number`);
    assert.equal(part.value, null, `${id} invented a number`);
  }
});

test("hiding the casing removes the shell but leaves the cell measurable", () => {
  const spec = makeSpec();
  const shown = buildScene(spec, { cursor: 0, casing: "translucent" });
  const hidden = buildScene(spec, { cursor: 0, casing: "hidden" });
  assert.equal(partOf(shown, "can").drawn, true);
  assert.ok(partOf(shown, "can").opacity > 0 && partOf(shown, "can").opacity < 1);
  assert.equal(partOf(hidden, "can").drawn, false);
  assert.equal(partOf(hidden, "can").mesh.positions.length, 0, "a hidden casing still has geometry");
  assert.equal(partOf(hidden, "sei_film").drawn, true);
});

test("the roll is wound, not stacked: each ribbon has its own lane in every turn", () => {
  // A real jelly roll interleaves its layers — anode, separator, cathode,
  // repeat — so the three ribbons share the same radial span and are ordered
  // *within* a turn. Asserting nested cylinders instead would be asserting the
  // schematic this replaced.
  const scene = buildScene(makeSpec(), { cursor: 12 });
  const anode = radialExtent(partOf(scene, "anode_sheet").mesh);
  const separator = radialExtent(partOf(scene, "separator").mesh);
  const cathode = radialExtent(partOf(scene, "cathode_sheet").mesh);
  assert.ok(anode.min < separator.min, "the separator's lane is inside the anode's");
  assert.ok(separator.min < cathode.min, "the cathode's lane is inside the separator's");
  assert.ok(anode.max < separator.max, "each ribbon reaches further out than the one inside it");
  assert.ok(separator.max < cathode.max, "each ribbon reaches further out than the one inside it");
  // …and the outermost lane is the roll's own outer edge, so the winding fills
  // the space it has rather than hanging in it. The mesh and the model have to
  // agree in millimetres, or one of them is lying about the other.
  const unitMm = rollModel(makeSpec()).unitMm;
  assert.ok(
    Math.abs(cathode.max * unitMm - scene.roll.outerRadiusMm) < 0.2,
    `the drawn cathode reaches ${(cathode.max * unitMm).toFixed(2)} mm, the model says ${scene.roll.outerRadiusMm.toFixed(2)} mm`,
  );
});

test("the winding follows the document's millimetres, not the renderer's taste", () => {
  const declared = buildScene(makeSpec(), { cursor: 12 });
  const physical = makePhysical();
  assert.equal(declared.roll.declared, true);
  assert.equal(declared.roll.turns, physical.roll?.turns);
  assert.equal(declared.roll.pitchMm, physical.roll?.pitchMm);
  assert.ok(
    Math.abs(declared.roll.electrodeLengthM - (physical.roll?.electrodeLengthM ?? 0)) < 1e-6,
    "the renderer and the producer must derive the same electrode length",
  );
  // The three ribbons tile one turn's advance exactly: that is what makes the
  // roll a filled roll instead of a spiral with gaps in it.
  const model = rollModel(makeSpec());
  assert.ok(
    Math.abs(
      model.anodeThickness + model.separatorThickness + model.cathodeThickness - model.pitch,
    ) < 1e-12,
    "the drawn ribbons do not tile the stack pitch",
  );
  // A thicker declared stack means a shorter electrode at the same diameter —
  // the trade-off is in the dimensions, not in the drawing.
  const thicker = buildScene(
    makeSpec({
      physical: {
        roll: {
          ...makePhysical().roll!,
          stackMm: {
            copperFoil: 0.02,
            anodeCoating: 0.14,
            separator: 0.04,
            cathodeCoating: 0.12,
            aluminiumFoil: 0.03,
          },
          pitchMm: 0.35,
        },
      },
    }),
    { cursor: 12 },
  );
  assert.ok(thicker.roll.turns < declared.roll.turns, "a thicker stack must need fewer turns");
  assert.ok(
    thicker.roll.outerRadiusMm <= thicker.roll.envelopeRadiusMm + 1e-9,
    "however thick the stack, the roll may not leave its envelope",
  );
});

test("a document with no declared dimensions is drawn with the fallback and says so", () => {
  const scene = buildScene(makeSpec({ physical: "absent" }), { cursor: 12 });
  assert.equal(scene.roll.declared, false);
  // The fallback is the same 18650 figures, so the drawing is unchanged — what
  // changes is that the scene can no longer claim they are this cell's.
  assert.equal(scene.roll.turns, makePhysical().roll?.turns);
  assert.equal(scene.roll.pitchMm, 0.175);
});

test("exploding un-winds the roll instead of pushing it through the can", () => {
  const spec = makeSpec();
  let previous = Infinity;
  for (const exploded of [0, 0.25, 0.5, 0.75, 1]) {
    const scene = buildScene(spec, { cursor: 12, exploded });
    const cathode = radialExtent(partOf(scene, "cathode_sheet").mesh);
    const bore = rollModel(spec).canRadius - rollModel(spec).canThickness;
    assert.ok(cathode.max < bore, `exploded=${exploded}: the cathode pierces the can wall`);
    assert.ok(
      scene.roll.outerRadiusMm <= scene.roll.envelopeRadiusMm + 1e-9,
      `exploded=${exploded}: the roll left its envelope`,
    );
    assert.ok(scene.roll.drawnTurns <= previous, "the un-winding must not add turns");
    previous = scene.roll.drawnTurns;
  }
  const assembled = buildScene(spec, { cursor: 12, exploded: 0 });
  const apart = buildScene(spec, { cursor: 12, exploded: 1 });
  assert.ok(apart.roll.gain > assembled.roll.gain, "exploding must magnify the stack");
  assert.ok(apart.roll.drawnPitchMm > assembled.roll.drawnPitchMm);
  assert.equal(assembled.roll.drawnTurns, assembled.roll.turns, "at rest the winding is the real one");
});

test("the whole cell stays inside a triangle budget a scrub can afford", () => {
  // The winding is 38 laps of three ribbons, and the rebuild is what a slider
  // drag pays for. The ribbons are memoised (they do not depend on the cursor),
  // but the budget is asserted anyway: if a future change multiplies the
  // geometry, this fails here rather than as a stutter on someone's machine.
  const spec = makeSpec();
  for (const exploded of [0, 1]) {
    const scene = buildScene(spec, { cursor: 12, exploded });
    const triangles = scene.parts.reduce((total, part) => total + triangleCount(part.mesh), 0);
    assert.ok(triangles < 80_000, `exploded=${exploded}: ${triangles} triangles in the cell`);
  }
});

test("the film sheathes the anode's own surface and stays inside the winding", () => {
  // With a physical stack the anode and the separator are *adjacent*, so the
  // film is a sheath on the anode's surface rather than a filler in a gap that
  // no longer exists. What must hold either way: it is outside the anode it
  // coats, and it never leaves the roll it is part of.
  const scene = buildScene(makeSpec(), { cursor: 12 });
  const film = radialExtent(partOf(scene, "sei_film").mesh);
  const anode = radialExtent(partOf(scene, "anode_sheet").mesh);
  const cathode = radialExtent(partOf(scene, "cathode_sheet").mesh);
  assert.ok(film.min > anode.max, "the film must sit outside the anode it coats");
  assert.ok(film.max < cathode.max, "the film must not reach past the roll's own layers");
  assert.ok(
    film.max < rollModel(makeSpec()).envelopeRadius,
    "the film must stay inside the roll's envelope",
  );
  // It grows along the anode's own surface: the drawn sheath follows the roll
  // as the stack is magnified.
  const apart = buildScene(makeSpec(), { cursor: 12, exploded: 1 });
  const filmApart = radialExtent(partOf(apart, "sei_film").mesh);
  const anodeApart = radialExtent(partOf(apart, "anode_sheet").mesh);
  assert.ok(filmApart.min > anodeApart.max);
  assert.ok(filmApart.max < radialExtent(partOf(apart, "cathode_sheet").mesh).max);
});

test("the explode control moves parts apart and changes nothing else", () => {
  const spec = makeSpec();
  const assembled = buildScene(spec, { cursor: 30, exploded: 0 });
  const exploded = buildScene(spec, { cursor: 30, exploded: 1 });
  assert.ok(
    partOf(exploded, "cap").anchor[1] > partOf(assembled, "cap").anchor[1],
    "the cap did not lift",
  );
  assert.ok(
    partOf(exploded, "terminal_pos").anchor[1] > partOf(assembled, "terminal_pos").anchor[1],
    "the positive terminal did not lift",
  );
  for (const part of assembled.parts) {
    const other = partOf(exploded, part.id);
    assert.equal(other.value, part.value, `${part.id} changed its number when exploded`);
    assert.equal(other.available, part.available, `${part.id} changed availability when exploded`);
  }
});

test("a prismatic source is drawn as a stack, not as a stretched cylinder", () => {
  const prismatic = buildScene(makeSpec({ formFactor: "prismatic" }), { cursor: 4 });
  const cylindrical = buildScene(makeSpec({ formFactor: "cylindrical" }), { cursor: 4 });
  assert.equal(prismatic.formFactor, "prismatic");
  const stack = bounds(partOf(prismatic, "cathode_sheet").mesh);
  const coil = bounds(partOf(cylindrical, "cathode_sheet").mesh);
  // The stack is a slab: wide in x, thin in z. The coil is round, so its two
  // horizontal extents agree — within the wobble a spiral's outer end causes.
  assert.ok(stack.maxX - stack.minX > 0.25, "the prismatic coating is not a plate");
  assert.ok(stack.maxZ - stack.minZ < 0.02, "the prismatic coating is not thin");
  const coilX = coil.maxX - coil.minX;
  const coilZ = coil.maxZ - coil.minZ;
  assert.ok(Math.abs(coilX - coilZ) / Math.max(coilX, coilZ) < 0.05, "the 18650 coil is not round");
  assert.ok(prismatic.bounds.radius > cylindrical.bounds.radius);
});

// ---------------------------------------------------------------------------
// Refusals reaching the geometry
// ---------------------------------------------------------------------------

test("an unidentifiable split means no film growth and no particle loss", () => {
  const identified = buildScene(makeSpec({ splitIdentified: true }), {
    cursor: 119, dataScaled: true,
  });
  const refused = buildScene(makeSpec({ splitIdentified: false }), {
    cursor: 119, dataScaled: true,
  });
  const filmIdentified = radialExtent(partOf(identified, "sei_film").mesh);
  const filmRefused = radialExtent(partOf(refused, "sei_film").mesh);
  assert.ok(
    filmIdentified.max > filmRefused.max + 1e-4,
    "a refused split still grew the drawn film",
  );
  assert.ok(
    vertexCount(partOf(identified, "particles").mesh) <
      vertexCount(partOf(refused, "particles").mesh),
    "a refused split still dimmed particles as lost",
  );
  assert.equal(partOf(identified, "sei_film").available, true);
});

test("the film thickens with the cell's own fitted film, and only when data-scaled", () => {
  const small = buildScene(makeSpec({ seiAtStart: 0.5, seiAtEnd: 1 }), {
    cursor: 119, dataScaled: true,
  });
  const large = buildScene(makeSpec({ seiAtStart: 5, seiAtEnd: 20 }), {
    cursor: 119, dataScaled: true,
  });
  assert.ok(
    radialExtent(partOf(large, "sei_film").mesh).max >
      radialExtent(partOf(small, "sei_film").mesh).max,
    "a cell with a larger fitted film was drawn identically",
  );
  const anatomical = buildScene(makeSpec({ seiAtStart: 5, seiAtEnd: 20 }), { cursor: 119 });
  assert.ok(
    radialExtent(partOf(anatomical, "sei_film").mesh).max <
      radialExtent(partOf(large, "sei_film").mesh).max,
    "anatomical proportions must not carry the data-sized film",
  );
  assert.equal(anatomical.scaleNote, null);
  assert.ok(large.scaleNote && large.scaleNote.includes("√n"));
});

test("an anatomical build prints no mapping and a data-scaled build prints the spec's own", () => {
  const spec = makeSpec();
  const scene = buildScene(spec, { cursor: 3, dataScaled: true });
  assert.equal(scene.scaleNote, spec.geometryScales.sei_film.note);
  const refused = buildScene(makeSpec({ splitIdentified: false }), { cursor: 3, dataScaled: true });
  assert.equal(refused.scaleNote, makeSpec({ splitIdentified: false }).geometryScales.fade_model.note);
});

// ---------------------------------------------------------------------------
// The cursor: measured replay, then the future
// ---------------------------------------------------------------------------

test("past the last measured cycle every part holds its final measured state", () => {
  const spec = makeSpec({ projection: "available" });
  const scene = buildScene(spec, { cursor: spec.record.nCycles + 20 });
  const today = buildScene(spec, { cursor: spec.record.nCycles - 1 });
  for (const part of scene.parts) {
    assert.equal(part.value, partOf(today, part.id).value, `${part.id} moved into the future`);
  }
  assert.equal(scene.gauge.projected, true);
  assert.equal(today.gauge.projected, false);
});

test("the gauge follows the projection's central line and shows its band", () => {
  const spec = makeSpec({ projection: "available" });
  const measured = buildScene(spec, { cursor: spec.record.nCycles - 1 });
  assert.equal(measured.gauge.soh, spec.series.sohPct[spec.record.nCycles - 1]);
  assert.equal(measured.gauge.band, null);
  assert.equal(measured.gauge.fraction, measured.gauge.soh! / 100);

  const future = buildScene(spec, { cursor: spec.record.nCycles + 5 });
  const projectedSoh = spec.projection.sohPct?.[5] ?? null;
  assert.equal(future.gauge.soh, projectedSoh);
  assert.ok(future.gauge.band, "the posterior band was dropped");
  assert.ok(future.gauge.band!.low <= future.gauge.band!.high);
  assert.equal(future.gauge.projected, true);
});

test("a cursor outside the timeline clamps instead of throwing", () => {
  const spec = makeSpec({ projection: "none" });
  assert.equal(buildScene(spec, { cursor: -5 }).cursor, 0);
  assert.equal(buildScene(spec, { cursor: 10_000 }).cursor, spec.record.nCycles - 1);
  assert.equal(buildScene(spec, {}).cursor, DEFAULT_BUILD_OPTIONS.cursor);
});

test("a cursor asked for at mount is the cursor the scene opens on", () => {
  // The regression this pins: the engine folded a host's options in `update()`
  // only, never at `mount()`, so every host's `cursor` was discarded and the
  // scene opened on the cell's first cycle while its own slider said "today".
  const spec = makeSpec();
  const today = todayCursor(buildTimeline(spec));
  assert.ok(today > 0, "fixture must have more than one cycle for this to mean anything");
  const opened = mergeBuildOptions(DEFAULT_BUILD_OPTIONS, { cursor: today });
  assert.equal(opened.cursor, today);
  assert.equal(buildScene(spec, opened).cursor, today);
});

test("an option a host does not mention keeps its current value", () => {
  const current = { cursor: 42, exploded: 1, casing: "hidden" as const, dataScaled: true };
  assert.deepEqual(mergeBuildOptions(current, { cursor: 7 }), { ...current, cursor: 7 });
  assert.deepEqual(mergeBuildOptions(current, { exploded: 0.25 }), { ...current, exploded: 0.25 });
});

test("false and zero are values, not absences", () => {
  const on = { ...DEFAULT_BUILD_OPTIONS, dataScaled: true, exploded: 1 };
  const off = mergeBuildOptions(on, { dataScaled: false, exploded: 0 });
  assert.equal(off.dataScaled, false);
  assert.equal(off.exploded, 0);
});

test("an explicit undefined does not erase a set option", () => {
  const current = { ...DEFAULT_BUILD_OPTIONS, casing: "hidden" as const, dataScaled: true };
  const merged = mergeBuildOptions(current, { cursor: undefined, casing: undefined, dataScaled: undefined });
  assert.deepEqual(merged, current);
});

test("folding options never mutates the options it was given", () => {
  const current = { ...DEFAULT_BUILD_OPTIONS };
  const snapshot = JSON.stringify(current);
  mergeBuildOptions(current, { cursor: 9, exploded: 1, casing: "hidden", dataScaled: true });
  assert.equal(JSON.stringify(current), snapshot);
});

// ---------------------------------------------------------------------------
// The part table a host prints beside the scene
// ---------------------------------------------------------------------------

test("the part table is the scene's own reading at the scene's own cursor", () => {
  const spec = makeSpec();
  const cursor = 30;
  const scene = buildScene(spec, { cursor });
  const readings = partReadings(spec, scene);
  assert.deepEqual(
    readings.map((reading) => reading.id),
    scene.parts.map((part) => part.id),
  );
  for (const reading of readings) {
    const built = scene.parts.find((part) => part.id === reading.id);
    assert.equal(reading.value, built?.value);
    assert.equal(reading.carried, built?.carried);
    assert.equal(reading.available, built?.available);
  }
  // And the reading really is the cursor's, not the document's last snapshot.
  const first = partReadings(spec, buildScene(spec, { cursor: 0 }));
  const last = partReadings(spec, buildScene(spec, { cursor: spec.record.nCycles - 1 }));
  const moved = first.filter((reading, i) => reading.value !== last[i].value);
  assert.ok(moved.length > 0, "the table must move when the cursor does");
});

test("the document supplies the prose and the scene supplies the number", () => {
  const spec = makeSpec({ unavailable: ["vent"] });
  const readings = partReadings(spec, buildScene(spec, { cursor: 5 }));
  const vent = readings.find((reading) => reading.id === "vent");
  const described = spec.parts.find((part) => part.id === "vent");
  assert.equal(vent?.meaning, described?.meaning);
  assert.equal(vent?.label, described?.label);
  assert.equal(vent?.available, false);
  assert.equal(vent?.reason, described?.unavailableReason);
  assert.ok(vent?.reason, "an unmeasured part must say why on the card too");
});

test("the table is the scene's own list of parts, never a longer one", () => {
  // The invariant that keeps a card and a mesh together: a document that drops a
  // part drops the row with it, because the geometry builds its parts from that
  // document. A table that outran its scene would be a card nobody can point at.
  const spec = makeSpec();
  const stripped = { ...spec, parts: spec.parts.filter((part) => part.id !== "separator") };
  const scene = buildScene(stripped, { cursor: 5 });
  const readings = partReadings(stripped, scene);
  assert.deepEqual(readings.map((reading) => reading.id), scene.parts.map((part) => part.id));
  assert.equal(readings.some((reading) => reading.id === "separator"), false);
  assert.equal(readings.length, ANATOMY_PART_IDS.length - 1);
});

// ---------------------------------------------------------------------------
// Theme: colours are data, not literals
// ---------------------------------------------------------------------------

test("the gauge takes its colour from the spec's own SOH bands", () => {
  const spec = makeSpec();
  const healthy = buildScene(makeSpec({ fadePerCycle: 0.01 }), { cursor: 10 });
  const eol = buildScene(makeSpec({ fadePerCycle: 0.5, n: 60 }), { cursor: 59 });
  assert.equal(healthy.gauge.color, THEME.sohBands[0].color);
  assert.equal(healthy.gauge.label, "Healthy");
  assert.equal(eol.gauge.color, THEME.sohBands[2].color);
  assert.equal(eol.gauge.label, "End of Life");

  const rebanded = makeSpec();
  rebanded.theme = { ...THEME, sohBands: [{ min: 0, max: null, color: "#123456", label: "Anything above zero" }] };
  assert.equal(buildScene(rebanded, { cursor: 10 }).gauge.color, "#123456");
  assert.equal(sohColor(spec.series.sohPct[10] ?? null, rebanded.theme), "#123456");
});

test("part colours come from the theme too, so a host can repaint the cell", () => {
  const spec = makeSpec();
  spec.theme = { ...THEME, anodeColor: "#010203", cathodeColor: "#040506", seiColor: "#070809" };
  const scene = buildScene(spec, { cursor: 3 });
  assert.equal(partOf(scene, "anode_sheet").color, "#010203");
  assert.equal(partOf(scene, "cathode_sheet").color, "#040506");
  assert.equal(partOf(scene, "sei_film").color, "#070809");
});

test("the mechanism verdict lights the part it points at and nothing else", () => {
  const sei = buildScene(makeSpec({ emphasis: "sei" }), { cursor: 3 });
  const particles = buildScene(makeSpec({ emphasis: "particles" }), { cursor: 3 });
  assert.ok(partOf(sei, "sei_film").emissive > partOf(particles, "sei_film").emissive);
  assert.ok(partOf(particles, "particles").emissive > partOf(sei, "particles").emissive);
});

test("band lookup refuses to guess outside the bands", () => {
  assert.equal(bandFor(95, THEME.sohBands)?.label, "Healthy");
  assert.equal(bandFor(85, THEME.sohBands)?.label, "Degrading");
  assert.equal(bandFor(79.9, THEME.sohBands)?.label, "End of Life");
  assert.equal(bandFor(null, THEME.sohBands), null);
  const narrow = [{ min: 50, max: 60, color: "#000000", label: "narrow" }];
  assert.equal(bandFor(70, narrow), null, "a value outside every band must not be assigned one");
});

test("power colour uses the host's own power-fade floor", () => {
  const spec = makeSpec();
  const below = powerColor(spec.theme.sopFloorPct - 1, spec.theme);
  const above = powerColor(110, spec.theme);
  assert.equal(below, THEME.sohBands[2].color);
  assert.equal(above, THEME.sohBands[0].color);
  assert.equal(powerColor(null, spec.theme), THEME.accent);
});

test("normalize clamps and refuses a degenerate window", () => {
  assert.equal(normalize(5, 0, 10), 0.5);
  assert.equal(normalize(-5, 0, 10), 0);
  assert.equal(normalize(50, 0, 10), 1);
  assert.equal(normalize(5, 10, 10), 0);
  assert.equal(normalize(null, 0, 10), 0);
  assert.equal(normalize(5, null, null), 0);
  assert.equal(mixHex("#000000", "#ffffff", 0.5), "#808080");
});

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

test("two builds of one spec are identical — no randomness leaks into a render", () => {
  const spec = makeSpec({ projection: "available" });
  const a = buildScene(spec, { cursor: 42, exploded: 0.3, dataScaled: true });
  const b = buildScene(spec, { cursor: 42, exploded: 0.3, dataScaled: true });
  for (const part of a.parts) {
    const other = partOf(b, part.id);
    assert.deepEqual(Array.from(part.mesh.positions), Array.from(other.mesh.positions), part.id);
    assert.deepEqual(Array.from(part.mesh.indices), Array.from(other.mesh.indices), part.id);
    assert.equal(part.color, other.color);
  }
});

test("scrubbing the timeline does not teleport the particle cloud", () => {
  const spec = makeSpec();
  const early = buildScene(spec, { cursor: 1 });
  const late = buildScene(spec, { cursor: 100 });
  const earlyCloud = Array.from(partOf(early, "particles").mesh.positions);
  const lateCloud = Array.from(partOf(late, "particles").mesh.positions);
  assert.equal(earlyCloud.length, lateCloud.length, "the cloud changed size without the data changing");
  assert.deepEqual(earlyCloud.slice(0, 30), lateCloud.slice(0, 30), "particles moved");
});

test("the scene carries no stage furniture into the anatomy", () => {
  const scene = buildScene(makeSpec(), { cursor: 2 });
  assert.equal(partOf(scene, "can").id, "can");
  assert.ok(bounds(scene.chrome.floor).maxY < bounds(partOf(scene, "terminal_neg").mesh).minY);
  assert.ok(bounds(scene.chrome.shadow).maxY < 0);
});

// ---------------------------------------------------------------------------
// The SEI film: a derived thickness, drawn at a declared magnification
// ---------------------------------------------------------------------------

test("the film band comes from the document, and falls back when it has none", () => {
  const spec = makeSpec();
  const band = filmBand(spec);
  assert.equal(band.declared, true, "the document declares its own drawn band");
  assert.equal(band.min, spec.physical!.film!.display!.drawnMinMm / spec.physical!.unitsMmPerCellUnit);
  assert.equal(band.max, spec.physical!.film!.display!.drawnMaxMm / spec.physical!.unitsMmPerCellUnit);
  // A document produced before the derivation existed still draws, on the
  // renderer's own band, and says the band is a fallback rather than this cell's.
  const stripped = { ...spec, physical: { ...spec.physical!, film: null } };
  assert.equal(filmBand(stripped).declared, false);
  assert.equal(filmBand(stripped).min, CELL_GEOMETRY.seiFilm.anatomical);
  assert.equal(filmBand(stripped).max, CELL_GEOMETRY.seiFilm.dataMax);
  // A degenerate band (no span) falls back instead of dividing a thickness by
  // zero: a zero-width film is not a scale, it is a divide-by-nothing.
  const film = spec.physical!.film!;
  const degenerate = {
    ...spec,
    physical: {
      ...spec.physical!,
      film: { ...film, display: { ...film.display!, drawnMaxMm: film.display!.drawnMinMm } },
    },
  };
  assert.equal(filmBand(degenerate).declared, false);
});

test("the drawn film follows the nanometre series its own scale names", () => {
  // The film's card carries a thickness, and the scale that draws it names
  // `series.seiThicknessNm`. So the *fitted share* must no longer be able to
  // move the geometry, and the thickness series must be the thing that does.
  const small = makeSpec({ splitIdentified: true, seiAtStart: 0.5, seiAtEnd: 1 });
  const large = makeSpec({ splitIdentified: true, seiAtStart: 5, seiAtEnd: 20 });
  const filmOf = (spec: typeof small) =>
    radialExtent(partOf(buildScene(spec, { cursor: 119, dataScaled: true }), "sei_film").mesh).max;
  assert.ok(filmOf(large) > filmOf(small), "a thicker fitted film was drawn identically");

  // Same document, a large fitted share, the thickness series left alone.
  const shareOnly = { ...small, series: { ...small.series, seiPct: small.series.seiPct.map(() => 20) } };
  assert.equal(
    filmOf(shareOnly),
    filmOf(small),
    "the fitted share drove a film the document states in nanometres",
  );

  // …and the reverse: the thickness series alone moves the drawing.
  const thickened = {
    ...small,
    series: { ...small.series, seiThicknessNm: large.series.seiThicknessNm },
  };
  assert.equal(filmOf(thickened), filmOf(large), "the thickness series did not drive the film");
});

test("a refused split draws no film growth, whatever the thickness series says", () => {
  // The values in the series come from a fit whose √n channel is not identified,
  // so the gate is `splitIdentified` and not the array being empty.
  const refused = makeSpec({ splitIdentified: false, seiAtStart: 5, seiAtEnd: 20 });
  const anatomical = buildScene(refused, { cursor: 119 });
  const scaled = buildScene(refused, { cursor: 119, dataScaled: true });
  assert.equal(
    radialExtent(partOf(scaled, "sei_film").mesh).max,
    radialExtent(partOf(anatomical, "sei_film").mesh).max,
    "a refused split drew a film growth",
  );
});
