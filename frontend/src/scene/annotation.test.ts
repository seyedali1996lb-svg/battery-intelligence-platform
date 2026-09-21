/**
 * Headless tests for the annotation layer's pure half: the flank solver that
 * decides where callout cards sit, the colour mixing the dim state needs, the
 * drafting-frame chrome, and the editorial taxonomy.
 *
 * What they protect, in order of how much it matters:
 *
 * 1. **Cards never overlap and leader lines never cross.** Both are geometric
 *    facts a browser could hide for weeks (overlapping cards read as one
 *    corrupted card), so the solver's contract is asserted here as arithmetic:
 *    gaps hold, targets follow anchor order, everything stays on stage.
 * 2. **The degradation path.** A flank too crowded for two-row cards drops to
 *    single-line *as a whole* rather than half of it overlapping — a quiet
 *    failure mode nobody would report because it looks like a rendering bug.
 * 3. **Colour and chrome come from the document.** The frame may not spell a
 *    colour the theme did not hand it, same rule as the meshes.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { CATEGORY_TAXONOMY, chromeSvg, desaturateHex, layoutFlank } from "./annotation.ts";
import { makeSpec } from "./fixture.ts";

const stageH = 560;
const items = (ys: number[]) => ys.map((sy, i) => ({ id: `p${i}`, sy }));

test("cards on one flank never overlap vertically", () => {
  const ys = [30, 34, 40, 200, 205, 500, 505, 540];
  const { targetY, minGap } = layoutFlank(items(ys), stageH);
  for (let i = 1; i < targetY.length; i++) {
    assert.ok(targetY[i] - targetY[i - 1] >= minGap - 1e-9, `gap ${i} collapsed`);
  }
});

test("target order follows anchor order — leader lines cannot cross within a flank", () => {
  const ys = [500, 100, 300, 20];
  const { targetY } = layoutFlank(items(ys), stageH);
  const order = ys
    .map((sy, i) => ({ sy, ty: targetY[i] }))
    .sort((a, b) => a.sy - b.sy)
    .map((p) => p.ty);
  for (let i = 1; i < order.length; i++) assert.ok(order[i] >= order[i - 1]);
});

test("cards stay inside the stage no matter how the anchors pile up", () => {
  const ys = Array.from({ length: 10 }, (_v, i) => 40 + i); // ten anchors in 10px
  const { targetY } = layoutFlank(items(ys), stageH);
  const last = targetY[targetY.length - 1];
  assert.ok(last + 60 <= stageH, "two-row cards need their own height below the last card");
  assert.ok(targetY[0] >= 0);
});

test("a flank too crowded for two rows degrades to single-line cards", () => {
  const ys = Array.from({ length: 14 }, (_v, i) => 30 + i * 40);
  const twoRow = layoutFlank(items(ys.slice(0, 6)), stageH);
  assert.equal(twoRow.singleLine, false);
  const crowded = layoutFlank(items(ys), stageH);
  assert.equal(crowded.singleLine, true);
  for (let i = 1; i < crowded.targetY.length; i++) {
    assert.ok(crowded.targetY[i] - crowded.targetY[i - 1] >= crowded.minGap - 1e-9);
  }
});

test("an empty flank is an empty answer, not a NaN", () => {
  const { targetY, singleLine } = layoutFlank([], stageH);
  assert.deepEqual(targetY, []);
  assert.equal(singleLine, false);
});

test("desaturating mixes toward the palette's own neutral, and returns a hex", () => {
  const grey = desaturateHex("#ff0000", 1, "#808080");
  assert.match(grey, /^#[0-9a-f]{6}$/);
  assert.equal(desaturateHex("#ff0000", 0, "#808080"), "#ff0000");
});

test("chrome draws only from the theme it was handed", () => {
  const spec = makeSpec();
  const svg = chromeSvg({ theme: spec.theme, ornate: true, width: 800, height: 600 });
  assert.match(svg, /<svg/);
  assert.ok(
    !/#fff|white|black/i.test(svg.replace(spec.theme.text, "").replace(spec.theme.muted, "")),
    "no colour literal beyond what the theme itself spells",
  );
});

test("the category taxonomy is exactly the ten words the spec declares", () => {
  assert.equal(CATEGORY_TAXONOMY.length, 10);
});
