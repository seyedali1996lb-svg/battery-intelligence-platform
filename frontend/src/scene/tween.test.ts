/**
 * Headless tests for the motion primitives: the spring the stage's hover and
 * reticle ride on, the easing every tween shares, and the camera poses a
 * "frame this part" tween interpolates between.
 *
 * What they protect: settling is arithmetic, not a thing to watch happen. A
 * spring that never settles freezes the reticle mid-pulse; one that overshoots
 * makes the camera ring like a fishing rod; one that diverges on a huge frame
 * step (a backgrounded tab waking up) throws the view into space. All three
 * are pinned here.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CRITICAL,
  easeInOut,
  poseFacing,
  poseLerp,
  poseToPosition,
  springSettled,
  springStep,
  type Spring,
} from "./tween.ts";

test("the spring settles at its target without ringing", () => {
  let s: Spring = { value: 0, velocity: 0 };
  let ms = 0;
  let peak = -Infinity;
  while (!springSettled(s, 1) && ms < 3000) {
    s = springStep(s, 1, 16);
    peak = Math.max(peak, s.value);
    ms += 16;
  }
  assert.ok(springSettled(s, 1), `never settled (t=${ms}ms, v=${s.value})`);
  // k=170 with ζ=1 settles to the 1e-3 band in ~830ms; the invariant that
  // matters for a camera is the other half — critically damped means it
  // approaches from below and never rings past the target.
  assert.ok(ms < 900, `settled in ${ms}ms — hard cap 900`);
  assert.ok(peak <= 1 + 1e-6, `overshot to ${peak} — not critically damped`);
  assert.ok(Math.abs(s.value - 1) < 1e-3);
});

test("one huge frame step cannot fling the spring", () => {
  const s = springStep({ value: 0, velocity: 0 }, 1, 500); // dt is clamped internally
  assert.ok(Number.isFinite(s.value) && Number.isFinite(s.velocity));
});

test("easeInOut is the identity at the ends and monotone between", () => {
  assert.equal(easeInOut(0), 0);
  assert.equal(easeInOut(1), 1);
  let prev = 0;
  for (let t = 0.05; t <= 0.95; t += 0.05) {
    const v = easeInOut(t);
    assert.ok(v >= prev);
    prev = v;
  }
});

test("poseLerp walks straight lines between poses", () => {
  const a = { azimuth: 0, elevation: 0.3, distance: 3, targetY: 0 };
  const b = { azimuth: Math.PI, elevation: 0.6, distance: 5, targetY: 0.2 };
  assert.deepEqual(poseLerp(a, b, 0), a);
  assert.equal(poseLerp(a, b, 0.5).distance, 4);
});

test("poseFacing puts the camera on the anchor's own side of the cell", () => {
  const pose = poseFacing([1, 0, 0], { azimuth: 9, elevation: 0.4, distance: 4, targetY: 0 });
  // sin(azimuth)≈1, cos(azimuth)≈0 → camera sits on +x
  assert.ok(Math.abs(Math.sin(pose.azimuth) - 1) < 1e-9);
  assert.equal(pose.elevation, 0.4); // elevation and distance are preserved
  assert.equal(pose.distance, 4);
});

test("a pose becomes a camera position on the sphere it describes", () => {
  // Due south: +z at the pose's own distance and height.
  assert.deepEqual(poseToPosition({ azimuth: 0, elevation: 0, distance: 4, targetY: 0 }), {
    x: 0,
    y: 0,
    z: 4,
  });
  // A quarter turn to +x, and elevation lifts the camera above the target.
  const raised = poseToPosition({ azimuth: Math.PI / 2, elevation: 0, distance: 4, targetY: 1 });
  assert.ok(Math.abs(raised.x - 4) < 1e-12 && Math.abs(raised.z) < 1e-12);
  assert.equal(raised.y, 1);
});

test("CRITICAL really is critically damped", () => {
  assert.ok(Math.abs(CRITICAL.damping - 2 * Math.sqrt(CRITICAL.stiffness)) < 1e-9);
});
