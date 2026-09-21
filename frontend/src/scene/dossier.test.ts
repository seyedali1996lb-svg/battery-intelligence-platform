/**
 * Headless tests for the dossier composer: the function that merges the
 * document's prose and spec table with live, cursor-resolved readings.
 *
 * What they protect, in order of how much it matters:
 *
 * 1. **Live rows only print numbers the document actually carries.** The
 *    composer runs at every cursor position; a channel the fit could not
 *    identify must yield a sentence, not a plausible fabrication — and a
 *    cursor past the end of the timeline must clamp, not print NaN.
 * 2. **Tags travel with the rows.** Two-tier provenance is the whole design:
 *    a typical figure and a fitted one must never blur into each other on the
 *    way to the card.
 * 3. **A part with no dossier gets null, not an empty shell** — hosts render
 *    the old panel for documents from before dossiers existed.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { composeDossier } from "./dossier.ts";
import { makeSpec } from "./fixture.ts";

const TAGS = ["typical", "measured", "derived", "fitted", "refusal"];

test("a part with a document dossier composes with its spec rows intact", () => {
  const spec = makeSpec();
  const view = composeDossier(spec, "anode_sheet", 10)!;
  assert.ok(view, "dossier exists");
  assert.equal(view.partId, "anode_sheet");
  assert.ok(view.latinTitle && view.latinTitle === view.latinTitle.toUpperCase());
  assert.ok(view.live.length >= 4, "temperature, resistance, SOH and film rows at minimum");
  assert.ok(view.specs.every((r) => TAGS.includes(r.tag)));
  assert.ok(view.live.every((r) => TAGS.includes(r.tag)));
});

test("live rows only ever print numbers the document actually carries", () => {
  const spec = makeSpec({ splitIdentified: false });
  const view = composeDossier(spec, "sei_film", 50)!;
  const film = view.live.find((r) => r.label === "SEI thickness")!;
  assert.equal(film.tag, "refusal");
  assert.match(film.value, /not identified|refus/i);
  assert.ok(view.partial, "a withheld live row is what `partial` means");
});

test("an unidentified split never prints a thickness, a broken cursor never prints NaN", () => {
  const spec = makeSpec();
  const view = composeDossier(spec, "sei_film", 9999)!; // cursor clamps
  for (const row of [...view.live, ...view.specs]) {
    assert.ok(!/NaN|undefined/.test(`${row.value}${row.label}`), `${row.label}: ${row.value}`);
  }
  // The default fixture DOES identify the split, so the film row is a real
  // number here — the refusal case is pinned by the test above.
  const film = view.live.find((r) => r.label === "SEI thickness")!;
  assert.notEqual(film.tag, "refusal");
});

test("a part id with no dossier returns null rather than a hollow card", () => {
  assert.equal(composeDossier(makeSpec(), "no_such_part", 0), null);
});
