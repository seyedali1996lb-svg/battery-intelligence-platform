/**
 * Shareable view state — where the reader's angle on the cell is written down.
 *
 * Four hosts draw the same scene (Streamlit iframe, static harness, React SPA,
 * a third party's page), and before this module none of them could answer
 * "send me the view you are looking at". The state is deliberately small: the
 * things that change *how you look* (cursor, explode, peel, layout, the
 * inspected part, annotations, palette), never a number read from the data —
 * a shared link must reopen the same picture, not re-derive a different cell.
 *
 * Two carriers, one precedence (a host composes them with `mergeViewState`):
 *
 *  * **the URL** — explicit, travels in a message, always wins;
 *  * **storage** — the last view of this cell on this machine, per
 *    `readPref`/`writePref`'s one preference store (`theme.ts`), so a reader
 *    who reloads lands back where they were;
 *  * failing both, the **host default** (which the engine already has).
 *
 * Pure: no DOM globals touched except through the injected storage functions,
 * so `viewstate.test.ts` round-trips every field without a browser.
 */

import { readPref, writePref } from "./theme.ts";

/** The reader's view of one scene. Every field optional — absent means "default". */
export interface ViewState {
  /** Life cursor: index into measured + projected cycles. */
  cursor?: number;
  /** Explode position, 0–1. */
  exploded?: number;
  /** Peel position, 0–1. */
  peel?: number;
  /** Wound/unrolled geometry. */
  layout?: "wound" | "unrolled";
  /** The inspected part's id (`null`/absent = nothing pinned). */
  part?: string | null;
  /** Annotation layer visible. */
  annotations?: boolean;
  /** The palette name the reader chose (absent = document's own theme). */
  theme?: string;
}

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

/**
 * Encode a view as a query string (`cursor=42&exploded=0.5&part=can`).
 *
 * Unset fields are omitted rather than written as `undefined`, so two states
 * that differ only by "default" and "default" encode to the *same* string —
 * that equality is what the round-trip test asserts.
 */
export function encodeViewState(state: ViewState): string {
  const parts: string[] = [];
  if (state.cursor !== undefined) parts.push(`cursor=${Math.max(0, Math.round(state.cursor))}`);
  if (state.exploded !== undefined) parts.push(`exploded=${clamp01(state.exploded)}`);
  if (state.peel !== undefined) parts.push(`peel=${clamp01(state.peel)}`);
  if (state.layout !== undefined) parts.push(`layout=${state.layout === "unrolled" ? "unrolled" : "wound"}`);
  if (state.part !== undefined && state.part !== null) parts.push(`part=${encodeURIComponent(state.part)}`);
  if (state.annotations !== undefined) parts.push(`annotations=${state.annotations ? "1" : "0"}`);
  if (state.theme !== undefined && state.theme !== "") parts.push(`theme=${encodeURIComponent(state.theme)}`);
  return parts.join("&");
}

/**
 * Decode a query string back into a view — tolerantly.
 *
 * A hand-edited or truncated URL is not an error to surface: an out-of-range
 * number clamps, a wrong word falls back to the default (`layout=wobble` reads
 * as absent, not as an exception), and unknown keys are ignored. What comes
 * out is always a view the engine can apply directly.
 */
export function decodeViewState(query: string): ViewState {
  const params = new URLSearchParams(query.startsWith("?") ? query.slice(1) : query);
  const out: ViewState = {};

  const num = (key: string): number | undefined => {
    const raw = params.get(key);
    if (raw === null || raw.trim() === "") return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
  };

  const cursor = num("cursor");
  if (cursor !== undefined) out.cursor = Math.max(0, Math.round(cursor));
  const exploded = num("exploded");
  if (exploded !== undefined) out.exploded = clamp01(exploded);
  const peel = num("peel");
  if (peel !== undefined) out.peel = clamp01(peel);
  const layout = params.get("layout");
  if (layout === "wound" || layout === "unrolled") out.layout = layout;
  const part = params.get("part");
  if (part) out.part = part;
  const annotations = params.get("annotations");
  if (annotations === "1" || annotations === "0") out.annotations = annotations === "1";
  const theme = params.get("theme");
  if (theme) out.theme = theme;
  return out;
}

/**
 * Compose the three carriers with the stated precedence:
 * **URL beats stored beats host default**.
 *
 * Only fields *present* in the higher-precedence source override — a URL that
 * says nothing about the palette must not erase the palette the reader stored.
 */
export function mergeViewState(
  url: ViewState | null | undefined,
  stored: ViewState | null | undefined,
  hostDefault: ViewState | null | undefined,
): ViewState {
  return { ...(hostDefault ?? {}), ...(stored ?? {}), ...(url ?? {}) };
}

const viewKey = (cellId: string): string => `view:${cellId}`;

/**
 * The last stored view of this cell, or null.
 *
 * Storage failures (a sandboxed iframe, a disabled API) are not exceptional:
 * a viewer that cannot remember simply opens at the host default, which is the
 * behaviour it had before this module existed.
 */
export function readStoredView(cellId: string): ViewState | null {
  if (!cellId) return null;
  const value = readPref<Partial<ViewState>>(viewKey(cellId));
  if (!value || typeof value !== "object") return null;
  // Round-trip through the codec: whatever an older build or a hand-edit left
  // in storage comes back clamped, typed and complete, or not at all.
  return decodeViewState(encodeViewState(value as ViewState));
}

/** Remember this cell's view under the same key `readStoredView` reads. */
export function writeStoredView(cellId: string, state: ViewState): void {
  if (!cellId) return;
  writePref(viewKey(cellId), state);
}
