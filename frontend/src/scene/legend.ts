/**
 * The band and provenance key — the pure half, shared by every host that lets
 * the engine draw its legend.
 *
 * Extracted from `index.ts` because the old inline renderer had two defects a
 * test could not reach there: it asked `band.min ? …` — truthiness — so the
 * End-of-Life band (`min: 0`) silently lost its range, and it never drew the
 * temperature bands at all. A key that names colours without stating the
 * numbers they stand for is decoration, so this module always prints explicit
 * ranges (`≥ 90%`, `80–90%`, `< 80%`, `< 30 °C`, …) under three headings.
 *
 * Pure on purpose: no DOM, no engine. `app/_scene_view.py`'s `legend_html`
 * mirrors this function line for line and a test on each side pins both to the
 * same output, so the Streamlit iframe and the standalone harness cannot
 * describe different colours than the ones painted.
 */

import type { SceneBand, SceneTheme } from "./types.ts";

/**
 * One band's explicit numeric range, in the reader's own units.
 *
 * `min`/`max` are half-open in the schema (a value belongs to the band where
 * `min ≤ v < max`), and `max: null` means "no upper bound". The falsy check
 * the old code used is exactly what this replaces: `0` is a real minimum.
 */
export function bandRange(min: number | null | undefined, max: number | null | undefined, unit = "%"): string {
  if ((min === null || min === undefined) && (max === null || max === undefined)) return "";
  const fmt = (v: number): string => String(v);
  if (max === null || max === undefined) return `≥ ${fmt(min as number)}${unit}`;
  if (min === null || min === undefined || min === 0) return `< ${fmt(max)}${unit}`;
  return `${fmt(min as number)}–${fmt(max)}${unit}`;
}

const esc = (text: string): string =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

interface LegendRow {
  band: SceneBand;
  unit: string;
}

function bandRows(rows: LegendRow[], swatch: string): string {
  return rows
    .map(({ band, unit }) => {
      const span = bandRange(band.min ?? null, band.max ?? null, unit);
      const text = [band.label ?? "", span].filter(Boolean).join(" · ");
      return (
        `<span class="cell-scene-legend-band">` +
        `<i class="${swatch}" style="background:${band.color}"></i>${esc(text)}</span>`
      );
    })
    .join("");
}

/**
 * The whole key as an HTML fragment: Health (SOH bands), Origin (provenance
 * line styles) and Casing temperature (the temperature bands, in °C).
 *
 * Bands with no rows are omitted rather than drawn as an empty heading — old
 * documents carry no temperature bands, and a heading over nothing is noise.
 */
export function legendHtml(theme: SceneTheme): string {
  const health = bandRows(
    (theme.sohBands ?? []).map((band) => ({ band, unit: "%" })),
    "box",
  );
  const origin = Object.entries(theme.provenanceColors ?? {})
    .filter(([key]) => key !== "")
    .map(
      ([key, color]) =>
        `<span class="cell-scene-legend-band">` +
        `<i class="line" style="background:${color}"></i>${esc(key)}</span>`,
    )
    .join("");
  const tint = bandRows(
    (theme.temperatureBands ?? []).map((band) => ({ band, unit: " °C" })),
    "box",
  );

  const section = (heading: string, body: string): string =>
    body
      ? `<div class="cell-scene-legend-head">${heading}</div>` +
        `<div class="cell-scene-legend-body">${body}</div>`
      : "";

  return (
    section("Health", health) + section("Origin", origin) + section("Casing temperature", tint)
  );
}
