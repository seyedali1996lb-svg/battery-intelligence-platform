/**
 * Headless tests for the shareable view-state codec.
 *
 * A view that does not round-trip is a link that lies: the recipient opens a
 * different picture than the sender saw. So every field must survive
 * encode → decode, garbage in a URL must come back as a *safe default* rather
 * than an exception (hand-edited links are normal), and the precedence
 * three hosts compose with (`mergeViewState`) must be exactly URL > stored >
 * host default, field by field — a URL silent about the palette must not
 * erase the stored palette.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { decodeViewState, encodeViewState, mergeViewState } from "./viewstate.ts";
import type { ViewState } from "./viewstate.ts";

test("a full view round-trips through its query string", () => {
  const view: ViewState = {
    cursor: 42,
    exploded: 0.5,
    peel: 0.25,
    layout: "unrolled",
    part: "cathode_coating",
    annotations: false,
    theme: "codex",
  };
  assert.deepEqual(decodeViewState(encodeViewState(view)), view);
});

test("unset fields stay unset — encoding is idempotent", () => {
  const view: ViewState = { cursor: 7 };
  const once = encodeViewState(view);
  assert.equal(encodeViewState(decodeViewState(once)), once);
  assert.deepEqual(decodeViewState(once), { cursor: 7 });
});

test("garbage in a URL decodes to clamped, finite values", () => {
  assert.deepEqual(decodeViewState("cursor=abc"), {});
  assert.equal(decodeViewState("cursor=-5").cursor, 0);
  assert.equal(decodeViewState("exploded=9").exploded, 1);
  assert.equal(decodeViewState("peel=-2").peel, 0);
  // An unknown word reads as absent, not as an exception.
  assert.equal(decodeViewState("layout=wobble").layout, undefined);
  // Unknown keys are ignored whole.
  assert.deepEqual(decodeViewState("tracking=xyz&cursor=3"), { cursor: 3 });
  // annotations only accepts the two written forms.
  assert.equal(decodeViewState("annotations=yes").annotations, undefined);
  assert.equal(decodeViewState("annotations=1").annotations, true);
  assert.equal(decodeViewState("annotations=0").annotations, false);
});

test("a leading ? (a raw location.search) decodes the same", () => {
  assert.equal(encodeViewState(decodeViewState("?cursor=4&part=can")), "cursor=4&part=can");
});

test("merge precedence is URL > stored > host default, field by field", () => {
  const merged = mergeViewState(
    { cursor: 10 },
    { cursor: 20, theme: "codex", layout: "unrolled" },
    { cursor: 30, exploded: 0.75 },
  );
  assert.equal(merged.cursor, 10, "URL must beat stored");
  assert.equal(merged.theme, "codex", "stored must survive a URL silent about it");
  assert.equal(merged.layout, "unrolled");
  assert.equal(merged.exploded, 0.75, "host default fills what neither carrier set");
});

test("merge tolerates null carriers (no URL, no storage)", () => {
  assert.deepEqual(mergeViewState(null, undefined, { cursor: 5 }), { cursor: 5 });
  assert.deepEqual(mergeViewState(undefined, null, null), {});
});
