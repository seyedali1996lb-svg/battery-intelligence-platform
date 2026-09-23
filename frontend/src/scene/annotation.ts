/**
 * The annotation layer's pure half: where callout cards sit, how a colour dims,
 * what the drafting frame draws, and the ten-word editorial taxonomy.
 *
 * Pure on purpose — no DOM, no three.js — for the same reason the geometry is:
 * the engine applies this, and a test can read it. The four things that make a
 * callout layer look *solved* rather than scattered (cards never overlap,
 * leader lines never cross, everything stays on stage, a crowded flank
 * degrades whole) are arithmetic, asserted in `annotation.test.ts`.
 */

import { hexToRgb } from "./theme.ts";
import type { SceneTheme } from "./types.ts";

/**
 * What subsystem a part belongs to. Colour- and claim-free by construction:
 * it says where a part lives in the cell's design, never what is true of this
 * cell — a category is editorial, and the schema keeps it optional so
 * documents from before it existed still render.
 */
export const CATEGORY_TAXONOMY = [
  "SHELL", "INSULATION", "SEAL", "SAFETY", "TERMINAL",
  "WINDING", "ELECTRODE", "SEPARATOR", "ELECTROLYTE", "DEGRADATION",
] as const;
export type Category = (typeof CATEGORY_TAXONOMY)[number];

/** Card heights and gaps, in pixels: the numbers `layoutFlank` reasons with. */
export const CARD_H_TWO = 40, CARD_H_ONE = 20, GAP_TWO = 46, GAP_ONE = 26, CARD_MARGIN = 12;

export interface FlankItem { id: string; sy: number; }
export interface FlankLayout { targetY: number[]; singleLine: boolean; minGap: number; }

/**
 * Where one flank's callout cards sit, vertically — card TOPS, one per input
 * item at the SAME index (`targetY[i]` belongs to `items[i]`, whatever order
 * they arrived in).
 *
 * Contract (all asserted in annotation.test.ts):
 *  (a) no two cards overlap vertically — consecutive targets in anchor order
 *      differ by minGap;
 *  (b) targets sorted by anchor are monotone in sy — two segments between two
 *      ordered pairs cannot cross, so the flank's leader lines never cross;
 *  (c) callers partition items left/right by anchor x before calling, so a
 *      solver run covers exactly one column and flanks never mix;
 *  (d) when the flank is too short for its cards, EVERYTHING drops to
 *      single-line mode instead of half of it overlapping.
 */
export function layoutFlank(items: FlankItem[], stageHeight: number): FlankLayout {
  const n = items.length;
  if (n === 0) return { targetY: [], singleLine: false, minGap: GAP_TWO };

  // Solve in anchor order (the order that makes (b) structural), then scatter
  // the answers back to the indexes the caller handed in.
  const order = items.map((_item, i) => i).sort((a, b) => items[a].sy - items[b].sy);
  const sorted = order.map((i) => items[i]);

  const space = stageHeight - 2 * CARD_MARGIN;
  let singleLine = n * GAP_TWO > space;
  const minGap = singleLine ? GAP_ONE : GAP_TWO;
  const cardH = singleLine ? CARD_H_ONE : CARD_H_TWO;

  // Forward pass: honour the gap, pulling each card as close to its anchor as
  // the card above allows, and never past the stage's bottom margin.
  const tops: number[] = [];
  for (let i = 0; i < n; i++) {
    const want = Math.max(sorted[i].sy - cardH / 2, i === 0 ? CARD_MARGIN : tops[i - 1] + minGap);
    tops.push(Math.min(want, stageHeight - CARD_MARGIN - cardH));
  }
  // Backward pass: the stack must also fit — pull the tail up while keeping
  // every gap and the order (a later card may only move up to its successor's
  // top minus the gap, never past its predecessor).
  for (let i = n - 1; i >= 0; i--) {
    const ceiling = i === n - 1 ? stageHeight - CARD_MARGIN - cardH : tops[i + 1] - minGap;
    tops[i] = Math.min(tops[i], ceiling);
    if (i > 0) tops[i] = Math.max(tops[i], tops[i - 1] + minGap);
  }
  // Pathologically short stage (or too many cards even single-line): re-space
  // the whole flank from the top, which preserves order by construction, and
  // declare single-line so the caller draws the shorter card.
  if (tops[n - 1] > stageHeight - CARD_MARGIN) {
    const step = Math.max(1, (stageHeight - 2 * CARD_MARGIN) / Math.max(1, n - 1));
    for (let i = 0; i < n; i++) tops[i] = Math.min(CARD_MARGIN + i * step, stageHeight - cardH);
    singleLine = true;
  }
  const targetY: number[] = new Array(n);
  order.forEach((originalIndex, k) => {
    targetY[originalIndex] = tops[k];
  });
  return { targetY, singleLine, minGap: singleLine ? GAP_ONE : minGap };
}

/** Mix `hex` toward `neutralHex` by `amount` (0 = unchanged, 1 = neutral). */
export function desaturateHex(hex: string, amount: number, neutralHex: string): string {
  const a = hexToRgb(hex), b = hexToRgb(neutralHex);
  const t = Math.max(0, Math.min(1, amount));
  const mix = (x: number, y: number) => Math.round(x + (y - x) * t);
  return "#" + [mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2])]
    .map((v) => v.toString(16).padStart(2, "0")).join("");
}

/** What one badge needs to know about the part it names, for grouping. */
export interface GroupableItem {
  id: string;
  /** False when the document carries no reading here — the "no reading" badge. */
  available: boolean;
}

export interface FlankGrouping<T extends GroupableItem> {
  /** Badges drawn individually, in input order. */
  shown: T[];
  /** Collapsed into one "N unmeasured" badge for this flank. */
  hidden: T[];
}

/**
 * Which badges stand alone and which collapse into a count.
 *
 * A cell early in life (or a document that measures few parts) can put a
 * dozen "no reading" badges on stage — clutter that hides the readings that
 * exist. So per flank: every *available* badge always shows, the **active**
 * part always shows even when unmeasured (you clicked it, it must stay
 * inspectable), and the rest collapse into one group badge the reader can
 * expand. `expanded` is the reader's toggle for that flank.
 *
 * Pure and order-preserving — `annotation.test.ts` asserts the invariants:
 * `shown ∪ hidden = input` (disjoint), available ⊆ shown, and expansion
 * empties `hidden` without reordering anything.
 */
export function groupUnmeasured<T extends GroupableItem>(
  items: T[],
  activeId: string | null,
  expanded: boolean,
): FlankGrouping<T> {
  const shown: T[] = [];
  const hidden: T[] = [];
  for (const item of items) {
    if (item.available || item.id === activeId || expanded) shown.push(item);
    else hidden.push(item);
  }
  return { shown, hidden };
}

/**
 * The drafting frame behind the stage, entirely from the theme's own tokens:
 * a ruled border with ticks, a compass rose (`ornate` selects codex's eight-
 * spoke rose over obsidian's plain reticle) and a centre ruler. The chrome
 * rule is the meshes' rule — it may not spell a colour the document did not
 * hand it, which a test asserts by scanning the output for literals.
 */
export function chromeSvg(opts: { theme: SceneTheme; ornate: boolean; width: number; height: number }): string {
  const { theme, ornate, width: w, height: h } = opts;
  const ink = theme.muted, rule = theme.grid, accent = theme.accent2 ?? theme.accent;
  const inset = 8;
  const ticks: string[] = [];
  for (let x = inset + 40; x < w - inset; x += 40) {
    const len = x % 200 === inset % 200 ? 10 : 5;
    ticks.push(`<line x1="${x}" y1="${inset}" x2="${x}" y2="${inset + len}" stroke="${rule}" stroke-width="1"/>`);
    ticks.push(`<line x1="${x}" y1="${h - inset}" x2="${x}" y2="${h - inset - len}" stroke="${rule}" stroke-width="1"/>`);
  }
  const roseR = ornate ? 34 : 18;
  const roseX = w - inset - roseR - 12, roseY = inset + roseR + 12;
  const spokes = ornate
    ? Array.from({ length: 8 }, (_v, i) => {
        const a = (i * Math.PI) / 4;
        return `<line x1="${roseX}" y1="${roseY}" x2="${roseX + roseR * Math.cos(a)}" y2="${roseY + roseR * Math.sin(a)}" stroke="${rule}" stroke-width="1"/>`;
      }).join("")
    : `<line x1="${roseX - roseR}" y1="${roseY}" x2="${roseX + roseR}" y2="${roseY}" stroke="${accent}" stroke-width="1"/>
       <line x1="${roseX}" y1="${roseY - roseR}" x2="${roseX}" y2="${roseY + roseR}" stroke="${accent}" stroke-width="1"/>`;
  const ruler = `<g><line x1="${w / 2 - 120}" y1="${h - inset - 14}" x2="${w / 2 + 120}" y2="${h - inset - 14}" stroke="${ink}" stroke-width="1"/>
    ${[0, 1, 2, 3, 4].map((i) => `<line x1="${w / 2 - 120 + i * 60}" y1="${h - inset - 18}" x2="${w / 2 - 120 + i * 60}" y2="${h - inset - 10}" stroke="${ink}" stroke-width="1"/>`).join("")}</g>`;
  return `<svg width="${w}" height="${h}" style="position:absolute;inset:0;pointer-events:none" aria-hidden="true">
    <rect x="${inset}" y="${inset}" width="${w - 2 * inset}" height="${h - 2 * inset}" fill="none" stroke="${rule}" stroke-width="${ornate ? 2 : 1}"/>
    ${ornate ? `<rect x="${inset + 4}" y="${inset + 4}" width="${w - 2 * inset - 8}" height="${h - 2 * inset - 8}" fill="none" stroke="${rule}" stroke-width="1"/>` : ""}
    ${ticks.join("")}
    <circle cx="${roseX}" cy="${roseY}" r="${roseR}" fill="none" stroke="${rule}" stroke-width="1"/>${spokes}
    <text x="${roseX}" y="${roseY + roseR + 12}" text-anchor="middle" font-size="9" fill="${ink}" font-family="${theme.fonts?.mono ?? "monospace"}">N</text>
    ${ruler}
  </svg>`;
}
