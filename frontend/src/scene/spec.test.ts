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

import { buildScene, buildTimeline, filmBand, rollModel, todayCursor } from "./geometry.ts";
import { DEFAULT_PART_MATERIAL, materialFor } from "./materials.ts";
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
  assert.equal(sample.parts.length, ANATOMY_PART_IDS.length);
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
  assert.equal(scene.parts.length, ANATOMY_PART_IDS.length);
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

test("the winding drawn is the winding the producer declared", () => {
  // The Python producer derives the turn count, the pitch and the electrode
  // length from the declared dimensions; the renderer derives them again from
  // the same inputs. If the two ever disagree, one of them is drawing a cell
  // the other never described — so this compares them on a real document.
  const declared = sample.physical;
  assert.ok(declared?.roll, "the committed sample carries no physical block");
  const model = rollModel(sample);
  assert.equal(model.declared, true);
  assert.equal(model.turns, declared.roll.turns, "turn count");
  assert.equal(model.pitchMm, declared.roll.pitchMm, "stack pitch");
  assert.ok(
    Math.abs(model.electrodeLengthM - declared.roll.electrodeLengthM) < 1e-3,
    `electrode length: renderer ${model.electrodeLengthM} m, producer ${declared.roll.electrodeLengthM} m`,
  );
  // And the drawn roll really fills the envelope the producer declared.
  const scene = buildScene(sample, { cursor: todayCursor(buildTimeline(sample)) });
  assert.ok(
    scene.roll.outerRadiusMm <= scene.roll.envelopeRadiusMm + 1e-9,
    "the assembled roll leaves its envelope",
  );
  assert.ok(
    scene.roll.outerRadiusMm >= 0.95 * scene.roll.envelopeRadiusMm,
    `the rolling fills only ${((scene.roll.outerRadiusMm / scene.roll.envelopeRadiusMm) * 100).toFixed(1)}% of its envelope`,
  );
  // Every declared field says where it came from; an unlabelled dimension is
  // one nobody can check.
  for (const key of Object.keys(declared.roll)) {
    if (key === "stackMm" || key === "turns" || key === "pitchMm") continue;
    assert.ok(
      declared.provenance[key] || key === "mandrelDiameterMm",
      `no provenance for physical.roll.${key}`,
    );
  }
  assert.ok(declared.schematic.length > 0, "the sample must name what is NOT to scale");
});

test("every part is given a material deliberately", async () => {
  await import("./fixture.ts");
  for (const id of ANATOMY_PART_IDS) {
    const material = materialFor(id);
    assert.notEqual(
      material,
      DEFAULT_PART_MATERIAL,
      `part ${id} falls through to the default material — give it a shading or say it is neutral`,
    );
    assert.ok(material.metalness >= 0 && material.metalness <= 1);
    assert.ok(material.roughness >= 0 && material.roughness <= 1);
    assert.ok(material.envMapIntensity > 0);
  }
  // The metals are the metals: a can is metal, a membrane is not, and the
  // electrolyte is the wettest thing in the cell.
  assert.ok(materialFor("can").metalness > 0.5);
  assert.equal(materialFor("separator").metalness, 0);
  assert.ok(materialFor("electrolyte").roughness < materialFor("separator").roughness);
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

test("the film's nanometres, the chain and the magnification all agree", () => {
  // The producer derives the thickness; the renderer draws the band the
  // document declares. This is the one place where both halves of that chain
  // can be checked against each other on a real cell rather than a fixture.
  const film = sample.physical?.film;
  assert.ok(film, "the committed sample carries no film block");
  const derivation = film.derivation;
  assert.ok(derivation, "the sample's film has no derivation to check");
  const assumptions = film.assumptions as {
    lithiumPerFormulaUnit: number;
    molarMassGPerMol: number;
    densityGPerCm3: number;
    coatedWidthMm: number;
    anodeFaces: number;
    initialNm: number;
  };
  // Every constant the chain uses states where it came from.
  for (const key of Object.keys(assumptions)) {
    assert.ok(film.provenance[key], `${key} is assumed without saying so`);
  }
  // The chain re-derives: capacity → coulombs → moles → volume → area → nm.
  const molarVolume = assumptions.molarMassGPerMol / assumptions.densityGPerCm3;
  const areaCm2 =
    assumptions.anodeFaces * (derivation.electrodeLengthM * 100) * (assumptions.coatedWidthMm / 10);
  const molFilm =
    (0.01 * derivation.capacity0Ah * 3600) / 96485.33212 / assumptions.lithiumPerFormulaUnit;
  assert.ok(
    Math.abs(derivation.nmPerPctLli - (molFilm * molarVolume) / areaCm2 * 1e7) < 1e-3,
    "the nm-per-percent factor does not fall out of the stated assumptions",
  );
  assert.equal(derivation.anodeAreaCm2, Number(areaCm2.toFixed(3)));
});

test("the film's thickness series is the fitted term, rescaled onto the document's own scale", () => {
  const film = sample.physical!.film!;
  const series = sample.series.seiThicknessNm;
  assert.ok(Array.isArray(series), "the sample carries no seiThicknessNm series");
  const sei = sample.series.seiPct;
  const perPct = film.derivation!.nmPerPctLli;
  const initial = film.assumptions.initialNm as number;
  assert.equal(series.length, sei.length);
  for (let i = 0; i < series.length; i++) {
    if (sei[i] === null) {
      assert.equal(series[i], null, "a gap in the fit must stay a gap in the thickness");
    } else {
      assert.ok(Math.abs((series[i] as number) - (initial + (sei[i] as number) * perPct)) < 1e-6);
    }
  }
  // The scale's endpoints are the document's, not the record's own maximum — so
  // two cells' films stay comparable, which normalising to each record's max
  // would destroy. Which *series* it names follows the document's own
  // identification: a cell whose √n channel is not identified has no thickness
  // to scale, and the mapping must then stay on the fitted share rather than on
  // a number the producer withheld.
  const scale = sample.geometryScales.sei_film;
  if (film.derivation!.maxNm === null) {
    assert.equal(film.identified, false);
    assert.equal(scale.from, "series.seiPct");
    assert.equal(film.display!.magnificationAtMaxX, null);
    assert.ok(film.reason && film.reason.includes("does not identify"));
    assert.equal(
      sample.parts.find((part) => part.id === "sei_film")!.value,
      null,
      "a withheld thickness must not reach the card as a number",
    );
  } else {
    assert.equal(film.identified, true);
    assert.equal(scale.from, "series.seiThicknessNm");
    assert.equal(scale.displayMin, initial);
    assert.equal(scale.displayMax, film.derivation!.displayMaxNm);
    assert.ok((scale.displayMax as number) > initial);
    assert.ok(
      Math.abs(
        film.display!.magnificationAtMaxX! - film.display!.drawnMaxNm / film.derivation!.maxNm,
      ) < 1,
      "the stated magnification is not the band over the thickness it draws",
    );
    const card = sample.parts.find((part) => part.id === "sei_film")!;
    assert.equal(card.provenance, "derived");
    assert.ok(card.unit.startsWith("nm"));
    assert.equal(card.value, Math.max(...series.filter((v): v is number => v !== null)));
  }
});

test("the renderer draws the magnified band the document declared, and says it is one", () => {
  const film = sample.physical!.film!;
  const display = film.display!;
  const unitMm = sample.physical!.unitsMmPerCellUnit;
  const band = filmBand(sample);
  assert.equal(band.declared, true);
  assert.equal(band.min, display.drawnMinMm / unitMm);
  assert.equal(band.max, display.drawnMaxMm / unitMm);
  assert.ok(band.max > band.min);
  // The two questions stay separate: the thickness has a number, the layer on
  // screen has a magnification, and the document states the second rather than
  // letting a viewer read the first off the drawing.
  assert.equal(display.note.includes("magnification"), true);
  assert.equal(sample.disclosures.some((d) => d.includes("drawn SEI layer")), true);
});

test("a document with no film block still draws, and says the band is a fallback", async () => {
  const { makeSpec } = await import("./fixture.ts");
  const stripped = makeSpec();
  stripped.physical = { ...stripped.physical!, film: null };
  const band = filmBand(stripped);
  assert.equal(band.declared, false);
  assert.ok(band.max > band.min);
  // The geometry still builds — an older document must not blank the canvas.
  const scene = buildScene(stripped, { cursor: 10, dataScaled: true });
  assert.ok(scene.bounds.radius > 0);
});

test("every part in the sample carries one of the ten declared categories", () => {
  const taxonomy = new Set([
    "SHELL", "INSULATION", "SEAL", "SAFETY", "TERMINAL",
    "WINDING", "ELECTRODE", "SEPARATOR", "ELECTROLYTE", "DEGRADATION",
  ]);
  for (const part of sample.parts) {
    assert.ok(taxonomy.has(part.category ?? ""), `${part.id}: ${part.category}`);
  }
});
