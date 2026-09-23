/**
 * Colour and band lookups — pure, DOM-free, and entirely theme-driven.
 *
 * Everything here reads `spec.theme` (see `src/cell_scene.default_theme()`).
 * Nothing in this file hard-codes a battery colour or a health threshold, so a
 * host can restyle or re-band the scene without touching the renderer, and the
 * scene can never call a cell healthy using bands the host would disagree with.
 */

import type { CellSceneSpec, SceneBand, SceneTheme } from "./types.ts";

/**
 * The palette a stage should paint with: the named one if the document
 * carries it, else the document's own theme. Unknown names and pre-palette
 * documents are not errors — they are the default, and a switcher that threw
 * on a stale document would be punishing its reader for the document's age.
 */
export function paletteFor(spec: CellSceneSpec, name: string | null | undefined): SceneTheme {
  if (name && spec.palettes && spec.palettes[name]) return spec.palettes[name];
  return spec.theme;
}

/**
 * Per-origin stage preferences, stored as one small JSON bag.
 *
 * A browser that refuses storage (private mode, disabled cookies) simply
 * keeps none: persistence is a courtesy here, never a requirement, and no
 * preference read or write may ever surface as an error in the scene.
 */
const PREF_KEY = "cell-scene.pref";

export function readPref<T>(key: string): T | null {
  try {
    const raw = window.localStorage.getItem(PREF_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return key in parsed ? (parsed[key] as T) : null;
  } catch {
    return null;
  }
}

export function writePref(key: string, value: unknown): void {
  try {
    const raw = window.localStorage.getItem(PREF_KEY);
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    parsed[key] = value;
    window.localStorage.setItem(PREF_KEY, JSON.stringify(parsed));
  } catch {
    /* no persistence, never an error */
  }
}

/** '#rrggbb' (or '#rgb') → [r, g, b] in 0–255. Throws on anything else. */
export function hexToRgb(hex: string): [number, number, number] {
  const value = hex.trim().replace(/^#/, "");
  const expanded =
    value.length === 3
      ? value
          .split("")
          .map((c) => c + c)
          .join("")
      : value;
  if (!/^[0-9a-fA-F]{6}$/.test(expanded)) {
    throw new Error(`not a hex colour: ${hex}`);
  }
  return [
    parseInt(expanded.slice(0, 2), 16),
    parseInt(expanded.slice(2, 4), 16),
    parseInt(expanded.slice(4, 6), 16),
  ];
}

/**
 * A theme colour at an opacity — `rgba(r, g, b, a)`, never a hand-typed literal.
 *
 * The badge chrome used to spell its own dark blues and greys, which meant a
 * light palette got dark badges painted over it. Every translucent surface in
 * the overlay now derives from `theme.panel`/`theme.grid`/`theme.accent`
 * through this one function, so the chrome repaints with the palette like the
 * meshes do. Falls back to the colour untouched when it is not a hex triple
 * (an `rgb()`/named token from a future schema still renders, just opaque).
 */
export function withAlpha(hex: string, alpha: number): string {
  try {
    const [r, g, b] = hexToRgb(hex);
    const a = Math.max(0, Math.min(1, alpha));
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  } catch {
    return hex;
  }
}

/**
 * The band a measurement falls in, or null when no band applies (a missing
 * reading, or a value outside every band's range).
 *
 * Bands are ordered and each carries an inclusive `min` and an exclusive `max`
 * (`null` = open-ended), which is exactly how the platform's own SOH ladder is
 * written (`soh >= 90` healthy, `soh >= 80` degrading, otherwise EOL). A value
 * matching no band returns null rather than the nearest band: guessing here is
 * how a view ends up disagreeing with the page beside it.
 */
export function bandFor(value: number | null, bands: SceneBand[]): SceneBand | null {
  if (value === null || !Number.isFinite(value)) return null;
  for (const band of bands) {
    const above = value >= band.min;
    const below = band.max === null || value < band.max;
    if (above && below) return band;
  }
  return null;
}

export function bandColor(value: number | null, bands: SceneBand[], fallback: string): string {
  return bandFor(value, bands)?.color ?? fallback;
}

export function sohColor(soh: number | null, theme: SceneTheme): string {
  return bandColor(soh, theme.sohBands, theme.muted);
}

export function sohLabel(soh: number | null, theme: SceneTheme): string {
  return bandFor(soh, theme.sohBands)?.label ?? "No reading";
}

export function temperatureColor(celsius: number | null, theme: SceneTheme): string {
  return bandColor(celsius, theme.temperatureBands, theme.metal);
}

export function provenanceColor(provenance: string, theme: SceneTheme): string {
  return theme.provenanceColors?.[provenance] ?? theme.muted;
}

/**
 * Position within a display window as 0–1, clamped.
 *
 * Used wherever a real measurement is mapped onto a drawn size. The window is
 * part of the spec's `geometryScales` (so the scene can print the mapping it
 * used); a missing or degenerate window returns 0 rather than a NaN that would
 * become a zero-sized mesh nobody notices.
 */
export function normalize(
  value: number | null,
  min: number | null | undefined,
  max: number | null | undefined,
  fallback = 0,
): number {
  if (value === null || !Number.isFinite(value)) return fallback;
  if (min === null || min === undefined || max === null || max === undefined) return fallback;
  if (!(max > min)) return fallback;
  const t = (value - min) / (max - min);
  return Math.min(1, Math.max(0, t));
}

/**
 * Power capability as colour, on the host's declared gauge window.
 *
 * Below `sopFloorPct` the colour is the host's lowest-band colour outright —
 * the platform's own "power-faded" floor, not a gradient that merely looks
 * warm. Above it the colour interpolates from that floor colour to the top
 * band's, so the tabs read as capability rather than as decoration.
 */
export function powerColor(sop: number | null, theme: SceneTheme): string {
  const eol = theme.sohBands[theme.sohBands.length - 1]?.color ?? theme.muted;
  const healthy = theme.sohBands[0]?.color ?? theme.accent;
  if (sop === null || !Number.isFinite(sop)) return theme.accent;
  if (sop <= theme.sopFloorPct) return eol;
  return mixHex(eol, healthy, normalize(sop, theme.sopFloorPct, theme.powerGaugeMaxPct, 1));
}

/** Resistance growth as colour: the accent, drifting toward the EOL colour. */
export function resistanceColor(rNorm: number | null, theme: SceneTheme): string {
  const eol = theme.sohBands[theme.sohBands.length - 1]?.color ?? theme.muted;
  return mixHex(theme.accent, eol, normalize(rNorm, 1, 2.2, 0));
}

/** Linear blend of two hex colours; `t` is clamped to 0–1. */
export function mixHex(from: string, to: string, t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const a = hexToRgb(from);
  const b = hexToRgb(to);
  const channel = (i: number) => Math.round(a[i] + (b[i] - a[i]) * clamped);
  const hex = (n: number) => n.toString(16).padStart(2, "0");
  return `#${hex(channel(0))}${hex(channel(1))}${hex(channel(2))}`;
}
