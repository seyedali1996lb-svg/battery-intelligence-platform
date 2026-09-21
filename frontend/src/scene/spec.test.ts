/**
 * The contract test between the two halves of the scene.
 *
 * The other tests in this directory use synthetic specs, which are good at
 * pinning the renderer's own behaviour and useless for catching a producer and
 * a renderer that disagree about the *document*. This file closes that gap: it
 * reads the scene that `scripts/export_scene_sample.py` produced from a real
 * cell, reads the version constant out of `src/cell_scene.py`, and mounts the
 * real document. If the Python builder renames a field, renumbers the schema,
 * reorders the anatomy or drops a series, this fails — in CI, at the boundary,
 * rather than as a blank canvas in a browser nobody is watching.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { buildScene, buildTimeline, todayCursor } from "./geometry.ts";
import { checkSchemaVersion, SCENE_SCHEMA_VERSION } from "./index.ts";
import { ANATOMY_PART_IDS } from "./types.ts";
import type { CellSceneSpec } from "./types.ts";

const REPO = new URL("../../../", import.meta.url);
const SAMPLE_PATH = fileURLToPath(new URL("app/static/cell_scene/sample_scene.json", REPO));
const MANIFEST_PATH = fileURLToPath(new URL("app/static/cell_scene/manifest.json", REPO));
const PYTHON_SOURCE = fileURLToPath(new URL("src/cell_scene.py", REPO));
const SCHEMA_PATH = fileURLToPath(new URL("docs/cell_scene.schema.json", REPO));

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

const sample = readJson<CellSceneSpec>(SAMPLE_PATH);
const manifest = readJson<{
  schemaVersion: number;
  bundle: string;
  bundleBytes: number;
  sourceSha256: Record<string, string>;
}>(MANIFEST_PATH);
const schema = readJson<{ properties: { schemaVersion: { const: number }; parts: { items: { properties: { id: { enum: string[] } } } } } }>(
  SCHEMA_PATH,
);

test("the version in the Python producer, the JSON Schema and the renderer are one number", () => {
  const pythonConstant = /^SCENE_SCHEMA_VERSION\s*=\s*(\d+)/m.exec(readFileSync(PYTHON_SOURCE, "utf8"));
  assert.ok(pythonConstant, "src/cell_scene.py no longer declares SCENE_SCHEMA_VERSION");
  assert.equal(Number(pythonConstant[1]), SCENE_SCHEMA_VERSION);
  assert.equal(schema.properties.schemaVersion.const, SCENE_SCHEMA_VERSION);
  assert.equal(manifest.schemaVersion, SCENE_SCHEMA_VERSION);
  assert.equal(sample.schemaVersion, SCENE_SCHEMA_VERSION, "the committed sample is from another version");
});

test("the committed sample is the document the renderer expects", () => {
  assert.equal(sample.parts.length, 13);
  assert.deepEqual(
    sample.parts.map((part) => part.id),
    [...ANATOMY_PART_IDS],
    "the sample's anatomy order differs from the renderer's mesh order",
  );
  const enumIds = schema.properties.parts.items.properties.id.enum;
  assert.deepEqual([...enumIds].sort(), [...ANATOMY_PART_IDS].sort());
});

test("every series in the real sample is aligned to one cursor", () => {
  const n = sample.series.cycles.length;
  assert.ok(n > 10, "the sample has almost no cycles — is it a real export?");
  for (const [name, series] of Object.entries(sample.series)) {
    assert.equal(series.length, n, `${name} is ${series.length} long, not ${n}`);
  }
  for (const part of sample.parts) {
    if (part.series !== null) {
      assert.equal(part.series.length, n, `part ${part.id} is misaligned`);
    }
  }
  assert.equal(sample.record.nCycles, n);
});

test("the sample's own honesty fields are present and self-consistent", () => {
  for (const part of sample.parts) {
    if (!part.available) {
      assert.ok(part.unavailableReason, `${part.id} is unavailable with no reason`);
      assert.ok(part.meaning, `${part.id} is unavailable with no explanation`);
      assert.equal(part.value, null, `${part.id} is unavailable but carries a number`);
    }
    if (part.series === null) continue;
    if (!part.available) {
      // A refused layer must not ship a drawable series either: that is the
      // exact shape of the "unidentified split still drew a film" defect.
      assert.equal(part.series, null, `${part.id} is unavailable but ships a series`);
    }
  }
  if (!sample.physics.splitIdentified) {
    assert.ok(sample.physics.splitReason, "an unidentified split must say why");
    for (const id of ["sei_film", "particles"] as const) {
      const part = sample.parts.find((candidate) => candidate.id === id);
      assert.equal(part?.available, false, `${id} is available on an unidentified split`);
      assert.equal(part?.series, null, `${id} ships a series on an unidentified split`);
    }
  }
  assert.ok(sample.disclosures.length >= 3);
  assert.ok(sample.disclosures.some((text) => text.toLowerCase().includes("schematic")));
});

test("every geometry scale in the real sample points at a series that exists", () => {
  for (const [key, scale] of Object.entries(sample.geometryScales)) {
    assert.ok(scale.note, `${key} has no disclosed mapping`);
    const seriesName = scale.from.split(".")[1];
    assert.ok(seriesName in sample.series, `${key} reads ${scale.from}, which is not a series`);
  }
});

test("the theme the sample carries has everything the renderer paints with", () => {
  for (const key of [
    "background", "panel", "text", "muted", "accent", "grid", "metal",
    "anodeColor", "cathodeColor", "separatorColor", "electrolyteColor", "seiColor",
  ] as const) {
    assert.match(String(sample.theme[key]), /^#[0-9a-fA-F]{6}$/, `theme.${key} is not a hex colour`);
  }
  assert.ok(sample.theme.sohBands.length >= 2);
  for (const band of sample.theme.sohBands) {
    assert.match(band.color, /^#[0-9a-fA-F]{6}$/);
    assert.ok(band.label);
  }
  for (const part of sample.parts) {
    if (!part.provenance) continue;
    assert.ok(
      sample.theme.provenanceColors[part.provenance],
      `no colour for provenance ${part.provenance}`,
    );
  }
});

test("the renderer mounts the real sample and can scrub its whole timeline", () => {
  const timeline = buildTimeline(sample);
  assert.equal(timeline.measuredCount, sample.record.nCycles);
  if (!sample.projection.available) {
    assert.equal(timeline.hasProjection, false);
  }
  const scene = buildScene(sample, { cursor: todayCursor(timeline) });
  assert.equal(scene.parts.length, 13);
  assert.equal(scene.gauge.soh, sample.record.lastSohPct);
  for (const part of scene.parts) {
    assert.ok(Number.isFinite(part.anchor[0] + part.anchor[1] + part.anchor[2]));
  }
  for (let cursor = 0; cursor < timeline.cycles.length; cursor += 17) {
    const built = buildScene(sample, { cursor });
    assert.equal(built.cursor, cursor);
  }
});

test("a document from another version is refused with a sentence, not drawn", () => {
  const wrong = { ...sample, schemaVersion: SCENE_SCHEMA_VERSION + 1 };
  const check = checkSchemaVersion(wrong);
  assert.equal(check.ok, false);
  assert.match(check.message ?? "", /schema version/);
  assert.equal(check.received, SCENE_SCHEMA_VERSION + 1);
  const missing = checkSchemaVersion({ parts: [] });
  assert.equal(missing.ok, false);
});

test("the renderer's own fixture obeys the same contract as the real sample", async () => {
  // Guards against the inverse failure: fixtures that only the fixture agrees
  // with. Every field the sample carries must exist on a synthetic spec too,
  // and geometryScales keys must overlap (fixtures may carry a subset).
  const { makeSpec } = await import("./fixture.ts");
  const spec = makeSpec();
  assert.deepEqual(
    Object.keys(spec).sort(),
    Object.keys(sample).sort(),
    "the fixture and the committed sample disagree about the document's shape",
  );
  assert.deepEqual(
    spec.parts.map((part) => part.id),
    sample.parts.map((part) => part.id),
  );
});
