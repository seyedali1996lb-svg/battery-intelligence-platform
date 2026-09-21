/**
 * The motion primitives the stage's springs and camera tweens share.
 *
 * Pure numbers only — no three.js, no DOM — so settling behaviour is a thing a
 * test can assert rather than a thing a developer watches happen. The spring
 * is semi-implicit Euler with a clamped dt: a tab waking up after minutes must
 * take one bounded step, not integrate minutes of motion in one frame.
 */
export interface Spring { value: number; velocity: number; }
export interface SpringSpec { stiffness: number; damping: number; }
/** Critically damped: stiffness 170, damping 2√k → no overshoot, sub-second settle. */
export const CRITICAL: SpringSpec = { stiffness: 170, damping: 2 * Math.sqrt(170) };

/** One frame longer than this (a backgrounded tab waking up) clamps to this. */
const MAX_DT_MS = 50;

export function springStep(s: Spring, target: number, dtMs: number, spec: SpringSpec = CRITICAL): Spring {
  const dt = Math.min(Math.max(dtMs, 0), MAX_DT_MS) / 1000;
  const accel = spec.stiffness * (target - s.value) - spec.damping * s.velocity;
  return { value: s.value + s.velocity * dt, velocity: s.velocity + accel * dt };
}

export function springSettled(s: Spring, target: number, eps = 1e-3): boolean {
  return Math.abs(s.value - target) < eps && Math.abs(s.velocity) < eps;
}

/** Smoothstep: flat at both ends, monotone between — the house easing. */
export function easeInOut(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

/** Where the camera looks from: a sphere around (0, targetY, 0). */
export interface CameraPose { azimuth: number; elevation: number; distance: number; targetY: number; }

export function poseLerp(a: CameraPose, b: CameraPose, t: number): CameraPose {
  const e = easeInOut(t);
  return {
    azimuth: a.azimuth + (b.azimuth - a.azimuth) * e,
    elevation: a.elevation + (b.elevation - a.elevation) * e,
    distance: a.distance + (b.distance - a.distance) * e,
    targetY: a.targetY + (b.targetY - a.targetY) * e,
  };
}

/** Turn the camera to face an anchor, keeping elevation, distance and height. */
export function poseFacing(anchor: [number, number, number], base: CameraPose): CameraPose {
  return { ...base, azimuth: Math.atan2(anchor[0], anchor[2]) };
}

/** Camera position for a pose around (0, targetY, 0). */
export function poseToPosition(p: CameraPose): { x: number; y: number; z: number } {
  return {
    x: p.distance * Math.cos(p.elevation) * Math.sin(p.azimuth),
    y: p.targetY + p.distance * Math.sin(p.elevation),
    z: p.distance * Math.cos(p.elevation) * Math.cos(p.azimuth),
  };
}
