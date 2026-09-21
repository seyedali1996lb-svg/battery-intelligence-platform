/**
 * The dossier composer: document prose, document spec table and live
 * cursor-resolved readings, merged into the one view the card renders.
 *
 * Two rules are load-bearing here, and both are about honesty rather than
 * layout:
 *
 * 1. **A live row only ever prints a number the document actually carries.**
 *    Every channel goes through `readingAt` (the same function the geometry
 *    uses) and every value through `num`, which refuses rather than printing
 *    NaN. When this cell's channel split is not identified the SEI thickness
 *    row says so in words — the gate that hides the fitted fade from the
 *    geometry is the gate that hides the thickness from the card.
 * 2. **Tags travel with the rows.** `DossierTag` is a closed five-word
 *    vocabulary (schema-enforced on the document's rows); composing live rows
 *    must not widen it. The one vocabulary wrinkle: SOH past the last measured
 *    cycle is a *projection*, and there is no "projected" word — it is
 *    `derived` from the fitted model, and the row's own `atCycle` still says
 *    which cycle it describes.
 *
 * `partial` marks rows that *had* to refuse because data was missing (the
 * conditional refusals). The always-by-design refusal rows — impedance and
 * overpotential, which this platform never measures per part — are part of the
 * dossier's normal content, not a shortfall, so they do not set it.
 */

import { buildTimeline, readingAt } from "./geometry.ts";
import type { CellSceneSpec, DossierSpecRow, DossierTag } from "./types.ts";

export interface LiveRow extends DossierSpecRow {
  /** The cycle the number belongs to; null when the row states no cycle. */
  atCycle: number | null;
}

export interface DossierView {
  partId: string;
  label: string;
  latinTitle: string | null; // null when the document predates dossiers
  subsystem: string | null;
  material: string | null;
  degradation: string | null;
  insight: string | null;
  specs: DossierSpecRow[]; // document rows, tags intact
  live: LiveRow[]; // composed at the cursor, tags intact
  partial: boolean; // a live row had to refuse for want of data
}

/** Format a live number, or return null — never "NaN", never "undefined". */
function num(value: number | null | undefined, digits = 1): string | null {
  return value !== null && value !== undefined && Number.isFinite(value)
    ? value.toFixed(digits)
    : null;
}

export function composeDossier(
  spec: CellSceneSpec,
  partId: string,
  cursor: number,
): DossierView | null {
  const part = spec.parts.find((p) => p.id === partId);
  if (!part || !part.dossier) return null;
  const doc = part.dossier;

  const timeline = buildTimeline(spec);
  const safeCursor = Number.isFinite(cursor) ? Math.trunc(cursor) : 0;
  const c = Math.max(0, Math.min(safeCursor, timeline.cycles.length - 1));
  const cycle = timeline.cycles[c];
  const projected = timeline.projected[c] === true;

  const live: LiveRow[] = [];
  let partial = false;

  /** A row that could not print its number because data was missing. */
  const refuse = (label: string, value: string, unit: string): void => {
    live.push({ label, value, unit, tag: "refusal" satisfies DossierTag, atCycle: null });
    partial = true;
  };
  /** A row this platform never measures per part, by design. */
  const never = (label: string, unit: string): void => {
    live.push({
      label,
      value: "not measured per part on this platform",
      unit,
      tag: "refusal" satisfies DossierTag,
      atCycle: null,
    });
  };
  const measure = (label: string, raw: number | null, unit: string, digits = 1): void => {
    const value = num(raw, digits);
    if (value !== null) live.push({ label, value, unit, tag: "measured", atCycle: cycle });
    else refuse(label, `not carried by this document's cycle summaries`, "");
  };

  measure("Temperature", readingAt(spec.series.temperatureC, c).value, "°C");
  measure("Internal resistance", readingAt(spec.series.resistanceOhm, c).value, "Ω", 3);

  // Growth against the first recorded cycle: derived, and honest about it —
  // a missing baseline refuses instead of dividing by zero or by undefined.
  const nowR = readingAt(spec.series.resistanceOhm, c).value;
  const baseR = spec.series.resistanceOhm[0];
  if (nowR !== null && baseR !== null && baseR !== 0) {
    live.push({
      label: "Resistance growth",
      value: num(nowR / baseR, 3) ?? "not carried by this document",
      unit: "× initial",
      tag: "derived",
      atCycle: cycle,
    });
  } else {
    refuse("Resistance growth", "resistance series missing from this document", "");
  }

  const soh = timeline.sohPct[c];
  const sohValue = num(soh);
  if (sohValue !== null) {
    // Past the last measured cycle the number comes from the projection, not
    // the record: the vocabulary has no "projected", so it is `derived` —
    // never dressed up as a measurement, and `atCycle` names the cycle.
    live.push({
      label: "State of health",
      value: sohValue,
      unit: "%",
      tag: projected ? "derived" : "measured",
      atCycle: cycle,
    });
  } else {
    refuse("State of health", "not carried by this document", "");
  }

  // The same gate the geometry uses: no identified split, no thickness — a
  // sentence instead of a plausible fabrication.
  if (spec.physics.splitIdentified !== true) {
    refuse("SEI thickness", "withheld — this cell's channel split is not identified", "");
  } else {
    const film = readingAt(spec.series.seiThicknessNm ?? null, c).value;
    const filmValue = num(film);
    if (filmValue !== null) {
      live.push({ label: "SEI thickness", value: filmValue, unit: "nm", tag: "derived", atCycle: cycle });
    } else {
      refuse("SEI thickness", "not carried by this document", "");
    }
  }

  // The two questions no platform answer here is a per-part number: words,
  // every time, without setting `partial` (this is design, not a gap).
  never("Impedance contribution", "mΩ");
  never("Local overpotential", "mV");

  return {
    partId,
    label: part.label,
    latinTitle: doc.latinTitle ?? null,
    subsystem: doc.subsystem ?? null,
    material: doc.material ?? null,
    degradation: doc.degradation ?? null,
    insight: doc.insight ?? null,
    specs: doc.specs ?? [],
    live,
    partial,
  };
}
