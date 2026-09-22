/**
 * The renderer's pins on three.js.
 *
 * three renames and removes APIs between minor versions, and this failure mode
 * is quiet: in r186 `PCFSoftShadowMap` is still *exported* as a constant while
 * its implementation is gone, so naming it produces a console warning and a
 * different shadow filter. Nothing in this repository would have noticed — there
 * is no browser in CI — and the only reason it was found at all is that someone
 * read the live page's console while looking at something else.
 *
 * So the constants the renderer depends on are asserted to exist in the
 * installed version, and the ones known to be hollow are asserted not to be
 * named in the renderer at all. It is a dependency pin, not a style rule.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { ACESFilmicToneMapping, PCFShadowMap, VSMShadowMap } from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

const ENGINE = new URL("./engine.ts", import.meta.url);
const MATERIALS = new URL("./materials.ts", import.meta.url);

test("the renderer's three.js constants exist in the installed version", () => {
  // If a future three drops one of these, this fails here rather than as a
  // silently different picture.
  assert.equal(typeof ACESFilmicToneMapping, "number");
  assert.equal(typeof PCFShadowMap, "number");
  assert.equal(typeof VSMShadowMap, "number");
});

test("the procedural environment still builds", () => {
  // `RoomEnvironment` is a scene of light-coloured boxes, no DOM and no canvas,
  // so it is constructible here — which is the part that would break if it were
  // rewritten to need a renderer or a document.
  const environment = new RoomEnvironment();
  assert.ok(environment.children.length > 0, "the room environment has no geometry");
});

/**
 * Comments stripped before the scan, because the renderer's *comment* names the
 * removed constant on purpose — the documentation of a trap is not the trap.
 */
function code(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

test("the renderer does not ask for an API three has hollowed out", () => {
  const source = code(readFileSync(ENGINE, "utf8"));
  const materials = code(readFileSync(MATERIALS, "utf8"));
  const hollowed = ["PCFSoftShadowMap"];
  for (const gone of hollowed) {
    assert.equal(
      source.includes(gone) || materials.includes(gone),
      false,
      `${gone} is still exported by three but no longer implemented — asking for it ` +
        "silently renders something else",
    );
  }
  // …and it *does* ask for the supported one.
  assert.ok(source.includes("PCFShadowMap"), "the renderer sets no shadow filter");
  assert.ok(source.includes("ACESFilmicToneMapping"), "the renderer does no tone mapping");
  assert.ok(source.includes("RoomEnvironment"), "the renderer sets no environment");
});

test("the composer and its passes exist in the installed version", () => {
  // The same pin as the constants above, one layer out: three moves addons
  // between paths and drops exports across minors, and a bloom chain that
  // quietly loses a pass is a picture rendered wrong, not a build error.
  for (const pass of [EffectComposer, RenderPass, UnrealBloomPass, OutputPass]) {
    assert.equal(typeof pass, "function");
  }
});

test("the chain tone-maps exactly once at the end, and counts whole frames", () => {
  const source = code(readFileSync(ENGINE, "utf8"));
  // three skips tone mapping for off-screen targets, so the OutputPass must
  // come last — drop it and the scene renders linear and dark; move it before
  // the bloom and the highlights get rolled off twice.
  const render = source.indexOf("new RenderPass");
  const bloom = source.indexOf("new UnrealBloomPass");
  const output = source.indexOf("new OutputPass");
  assert.ok(render >= 0 && bloom >= 0 && output >= 0, "the engine chains all three passes");
  assert.ok(
    render < bloom && bloom < output,
    "the order must be RenderPass → UnrealBloomPass → OutputPass (tone mapping happens once, on the way out)",
  );
  // The composer calls renderer.render() several times per frame and three
  // resets renderer.info at the start of each — with auto-reset left on, the
  // HUD's telemetry would report only the final fullscreen quad instead of
  // what the frame costs.
  assert.match(source, /renderer\.info\.autoReset\s*=\s*false/);
  assert.match(source, /renderer\.info\.reset\(\)/);
});
