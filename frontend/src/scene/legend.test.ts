/**
 * Headless tests for the band and provenance key.
 *
 * Two defects this module exists to prevent, both invisible in a screenshot
 * until someone asks "what does amber mean?":
 *
 * 1. **Truthiness hiding a range.** The old renderer asked `band.min ? …`, so
 *    the End-of-Life band (`min: 0`) lost its numbers and the key became a row
 *    of colours. `bandRange(0, 80)` must print `< 80%`.
 * 2. **Drift from the painted scene.** The Python host mirrors this function
 *    line for line (`app/_scene_view.py`'s `legend_html`); these tests pin the
 *    TS half against the fixture theme, which itself mirrors the producer's
 *    table, so a threshold edit cannot quietly diverge.
 *
 * Also: temperature bands, which no renderer drew at all before, must appear
 * under their own heading with °C ranges — and tolerate the producer's
 * label-less default bands.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { bandRange, legendHtml } from "./legend.ts";
import { THEME } from "./fixture.ts";
import type { SceneTheme } from "./types.ts";

test("bandRange states every band's numbers, including a zero minimum", () => {
  assert.equal(bandRange(90, null), "≥ 90%");
  assert.equal(bandRange(80, 90), "80–90%");
  // The bug this replaces: 0 is falsy, so `min ? …` dropped this range.
  assert.equal(bandRange(0, 80), "< 80%");
  assert.equal(bandRange(null, null), "");
});

test("bandRange formats temperature ranges in °C", () => {
  assert.equal(bandRange(45, null, " °C"), "≥ 45 °C");
  assert.equal(bandRange(30, 45, " °C"), "30–45 °C");
  assert.equal(bandRange(0, 30, " °C"), "< 30 °C");
});

test("legend draws all three headings with explicit ranges", () => {
  const html = legendHtml(THEME as SceneTheme);
  assert.ok(html.includes("Health"));
  assert.ok(html.includes("Origin"));
  assert.ok(html.includes("Casing temperature"));
  // Every SOH band states its numbers — none hidden by a falsy check.
  assert.ok(html.includes("≥ 90%"), "healthy range missing");
  assert.ok(html.includes("80–90%"), "degrading range missing");
  assert.ok(html.includes("&lt; 80%"), "EOL range missing (the band.min=0 bug)");
});

test("legend renders label-less temperature bands as bare ranges", () => {
  // The producer's default temperatureBands carry no `label` key at all —
  // the formatter must survive that, not throw on undefined.
  const theme = {
    ...THEME,
    temperatureBands: [
      { min: 0, max: 30, color: "#4299e1" },
      { min: 30, max: 45, color: "#ecc94b" },
      { min: 45, max: null, color: "#e53e3e" },
    ],
  } as unknown as SceneTheme;
  const html = legendHtml(theme);
  assert.ok(html.includes("&lt; 30 °C"));
  assert.ok(html.includes("30–45 °C"));
  assert.ok(html.includes("≥ 45 °C"));
});

test("legend omits an empty section rather than a heading over nothing", () => {
  const theme = { ...THEME, temperatureBands: [] } as SceneTheme;
  const html = legendHtml(theme);
  assert.ok(!html.includes("Casing temperature"));
  assert.ok(html.includes("Health"));
});

test("legend lists every provenance line style under Origin", () => {
  const html = legendHtml(THEME as SceneTheme);
  for (const key of ["measured", "derived", "fitted", "projected"]) {
    assert.ok(html.includes(key), `provenance ${key} missing`);
  }
  // The empty-string key is the "no provenance" swatch — never labelled "".
  assert.ok(!html.includes("></i></span>"));
});
