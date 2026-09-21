/**
 * The cockpit HUD: one rail along the stage's bottom edge carrying the scene's
 * own view controls — explode, peel, cutaway, breathe, annotations, lifecycle
 * scrub, layout, camera poses, the palette switcher — plus telemetry and a
 * collapse chevron.
 *
 * Two rules carry over from the rest of the engine:
 *
 * 1. **Every colour comes from `state.theme`**, the *active* palette, and
 *    `update()` rebuilds the rail when that object changes — the rail repaints
 *    with `setTheme` beside the stage, the badges and the dossier card, and
 *    hard-codes no colour of its own.
 * 2. **Preferences go through `./theme.ts`'s `readPref`/`writePref`** — one
 *    preference store for the whole stage, never a second copy beside it.
 *
 * The engine owns this rail: hosts render their own controls (or none), and
 * the callbacks below are the same handle methods a host could call itself —
 * there is exactly one implementation of "explode the cell" in the codebase.
 */
import { readPref, writePref } from "./theme.ts";
import type { SceneTheme } from "./types.ts";

export interface HudState {
  theme: SceneTheme;
  exploded: number;
  peel: number;
  layout: "wound" | "unrolled";
  cursor: number;
  measuredCount: number;
  cycle: number | null;
  soh: number | null;
  projected: boolean;
  playing: boolean;
  annotations: boolean;
  breathe: boolean;
  themeName: string | null;
  availablePalettes: string[];
  /** The full explode's travel in millimetres — axial lift, radial spread. */
  mmMaxAxial: number;
  mmMaxRadial: number;
}

export interface HudCallbacks {
  setExploded(v: number): void;
  setPeel(v: number): void;
  setLayout(l: "wound" | "unrolled"): void;
  toggleBreathe(): void;
  toggleAnnotations(): void;
  playPause(): boolean;
  scrub(cursor: number): void;
  preset(name: "iso" | "plan" | "section" | "unrolled"): void;
  setTheme(name: string): void;
  frameActive(): void;
}

export interface Telemetry {
  fps: number;
  calls: number;
  triangles: number;
  vertices: number;
}

export interface HudHandle {
  update(state: HudState): void;
  setTelemetry(t: Telemetry): void;
  collapsed(): boolean;
  toggleCollapsed(): void;
  dispose(): void;
}

/** Compact "184k" / "1.2M" for the telemetry readout. */
function count(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

export function createHud(
  container: HTMLElement,
  initialState: HudState,
  cb: HudCallbacks,
): HudHandle {
  let theme = initialState.theme;
  let last = initialState;
  let telemetry: Telemetry = { fps: 0, calls: 0, triangles: 0, vertices: 0 };
  // The rail's own persisted preference — read here (never in a second store)
  // so the first build already knows whether it starts as a pill.
  let collapsed = readPref<boolean>("hudCollapsed") === true;
  let disposed = false;

  /** Element refs — rebound whenever a palette swap rebuilds the rail. */
  interface RailRefs {
    rail: HTMLDivElement;
    pill: HTMLButtonElement;
    explode: HTMLInputElement;
    peel: HTMLInputElement;
    scrub: HTMLInputElement;
    mm: HTMLSpanElement;
    life: HTMLSpanElement;
    play: HTMLButtonElement;
    tel: HTMLSpanElement;
    breathe: HTMLButtonElement;
    annotations: HTMLButtonElement;
    layout: Record<string, HTMLButtonElement>;
    palettes: Record<string, HTMLButtonElement>;
  }

  let refs: RailRefs | null = null;

  /** The colour half of a button border: live, so the active state repaints. */
  const borderColorOf = (on: boolean): string => (on ? theme.accent : `${theme.muted}44`);

  const btn = (label: string, title: string, hotkey: string | null, onClick: () => void): HTMLButtonElement => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = hotkey ? `${title}  [${hotkey}]` : title;
    b.style.cssText =
      `background:${theme.grid};color:${theme.text};border:1px solid ${borderColorOf(false)};border-radius:4px;` +
      "padding:3px 8px;font:inherit;cursor:pointer;display:flex;gap:6px;align-items:center";
    if (hotkey) {
      const k = document.createElement("kbd");
      k.textContent = hotkey;
      k.style.cssText = `font-size:9px;background:${theme.background};border-radius:3px;padding:0 4px;color:${theme.muted}`;
      b.appendChild(k);
    }
    b.addEventListener("click", onClick);
    return b;
  };

  const slider = (min: number, max: number, step: number, value: number, onInput: (v: number) => void, width = "110px"): HTMLInputElement => {
    const s = document.createElement("input");
    s.type = "range";
    s.min = String(min);
    s.max = String(max);
    s.step = String(step);
    s.value = String(value);
    s.style.cssText = `width:${width};accent-color:${theme.accent}`;
    s.addEventListener("input", () => onInput(Number(s.value)));
    return s;
  };

  const group = (): HTMLDivElement => {
    const g = document.createElement("div");
    g.style.cssText = `display:flex;gap:6px;align-items:center;padding-right:8px;border-right:1px solid ${theme.grid}`;
    return g;
  };

  const caption = (text: string): HTMLSpanElement => {
    const s = document.createElement("span");
    s.textContent = text;
    s.style.cssText = `font-size:9px;letter-spacing:0.1em;color:${theme.muted}`;
    return s;
  };

  const readout = (): HTMLSpanElement => {
    const s = document.createElement("span");
    s.style.cssText = `font-variant-numeric:tabular-nums;color:${theme.text};white-space:nowrap`;
    return s;
  };

  /** Collapse / restore — one pill when the rail is away. */
  function toggle(): void {
    collapsed = !collapsed;
    if (!refs) return;
    refs.rail.style.display = collapsed ? "none" : "flex";
    refs.pill.style.display = collapsed ? "inline-flex" : "none";
    writePref("hudCollapsed", collapsed);
  }

  /**
   * (Re)build the rail's DOM from the current palette. State is never kept in
   * the elements: `render()` writes `last` into whatever `refs` points at, so
   * a rebuild mid-session loses nothing.
   */
  function build(): RailRefs {
    refs?.rail.remove();
    refs?.pill.remove();

    const rail = document.createElement("div");
    rail.className = "cell-scene-hud";
    rail.style.cssText =
      "position:absolute;left:12px;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:8px;" +
      `align-items:center;padding:8px 10px;border-radius:6px;background:${theme.panel}ee;` +
      `border:1px solid ${theme.grid};color:${theme.text};font-family:${theme.fonts?.mono ?? "ui-monospace,monospace"};font-size:11px;z-index:4;`;

    // 1 — Explode: sliders retarget, the engine's spring chases.
    const explode = slider(0, 1, 0.01, last.exploded, (v) => cb.setExploded(v));
    const mm = readout();
    const gExplode = group();
    gExplode.append(
      caption("EXPLODE"),
      explode,
      mm,
      btn("ASSEMBLED", "Snap back to the assembled cell", null, () => cb.setExploded(0)),
      btn("EXPLODED 100%", "Full disassembly", null, () => cb.setExploded(1)),
    );

    // 2 — Peel, with the ¼ detent drawn where the cutaway lands.
    const peel = slider(0, 1, 0.01, last.peel, (v) => cb.setPeel(v));
    const peelWrap = document.createElement("div");
    peelWrap.style.cssText = "position:relative;display:flex;align-items:center;width:110px";
    const detent = document.createElement("span");
    detent.textContent = "¼";
    detent.style.cssText = `position:absolute;left:25%;top:-9px;font-size:8px;color:${theme.muted};transform:translateX(-50%);pointer-events:none`;
    peelWrap.append(detent, peel);
    const gPeel = group();
    gPeel.append(
      caption("PEEL"),
      peelWrap,
      btn("CUTAWAY · ¼ PEEL", "Cut a quarter of the film away (again to restore)", "C", () => cb.setPeel(0.25)),
    );

    // 3 — The two motion toggles. Breathe carries no badge by design (the
    // spec's badge keys are Space/E/C/A/H — B is click-only).
    const breathe = btn("BREATHE", "A slow breath across the stack", null, () => cb.toggleBreathe());
    const annotations = btn("ANNOTATIONS", "Show or hide the leader lines", "A", () => cb.toggleAnnotations());
    const gToggles = group();
    gToggles.append(breathe, annotations);

    // 4 — Lifecycle: play/scrub the measured record (and its projection).
    const play = btn("▶", "Play the cell's life", "Space", () => void cb.playPause());
    const scrub = slider(0, Math.max(1, last.measuredCount), 1, last.cursor, (v) => cb.scrub(v), "150px");
    const life = readout();
    const gLife = group();
    gLife.append(caption("LIFE"), play, scrub, life);

    // 5 — Layout.
    const layout = {
      wound: btn("WOUND", "The assembled spiral", null, () => cb.setLayout("wound")),
      unrolled: btn("UNROLLED", "The three ribbons as one flat strip", null, () => cb.setLayout("unrolled")),
    };
    const gLayout = group();
    gLayout.append(layout.wound, layout.unrolled);

    // 6 — Camera poses; FRAME turns to whatever is pinned.
    const gCamera = group();
    gCamera.append(
      caption("POSE"),
      btn("ISO", "Isometric view", null, () => cb.preset("iso")),
      btn("PLAN", "Top-down view", null, () => cb.preset("plan")),
      btn("SECTION", "Cut open: partial explode + quarter peel", null, () => cb.preset("section")),
      btn("UNROLLED", "Flat strip, seen along its length", null, () => cb.preset("unrolled")),
      btn("FRAME", "Turn to the inspected part", null, () => cb.frameActive()),
    );

    // 7 — Palette switcher — only when the document actually carries a choice.
    const palettes: Record<string, HTMLButtonElement> = {};
    let gTheme: HTMLDivElement | null = null;
    if (last.availablePalettes.length > 1) {
      gTheme = group();
      gTheme.append(caption("PALETTE"));
      for (const name of last.availablePalettes) {
        palettes[name] = btn(name.toUpperCase(), `Switch to the ${name} palette`, null, () => cb.setTheme(name));
        gTheme.append(palettes[name]);
      }
    }

    // 8 — Telemetry, right-aligned into the rail's remaining width.
    const tel = document.createElement("span");
    tel.style.cssText = `flex:1;text-align:right;font-variant-numeric:tabular-nums;color:${theme.muted};font-size:10px`;

    // 9 — Collapse chevron.
    const collapse = btn("‹", "Collapse the HUD", "H", toggle);

    rail.append(
      gExplode, gPeel, gToggles, gLife, gLayout, gCamera,
      ...(gTheme ? [gTheme] : []),
      tel, collapse,
    );
    container.appendChild(rail);

    const pill = document.createElement("button");
    pill.textContent = "⋯";
    pill.title = "Show the HUD";
    pill.style.cssText =
      "position:absolute;left:12px;bottom:12px;z-index:4;padding:3px 10px;cursor:pointer;" +
      `background:${theme.panel}ee;color:${theme.text};border:1px solid ${theme.grid};border-radius:4px;` +
      `font-family:${theme.fonts?.mono ?? "ui-monospace,monospace"};display:none`;
    pill.addEventListener("click", toggle);
    container.appendChild(pill);

    const fresh: RailRefs = {
      rail, pill, explode, peel, scrub, mm, life, play, tel,
      breathe, annotations, layout, palettes,
    };
    refs = fresh;
    rail.style.display = collapsed ? "none" : "flex";
    pill.style.display = collapsed ? "inline-flex" : "none";
    return fresh;
  }

  /**
   * Write `last` (and the stored telemetry) into the current elements. Text
   * only when it actually changes — this runs at 4 Hz and the DOM should not
   * notice a scene standing still.
   */
  function render(): void {
    const r = refs;
    if (!r || disposed) return;
    const s = last;

    r.explode.value = String(s.exploded);
    r.peel.value = String(s.peel);

    const mmText = `+${(s.exploded * s.mmMaxAxial).toFixed(1)} mm · ${(s.exploded * s.mmMaxRadial).toFixed(2)} mm`;
    if (r.mm.textContent !== mmText) r.mm.textContent = mmText;

    const scrubMax = Math.max(s.measuredCount, s.cursor);
    if (r.scrub.max !== String(scrubMax)) r.scrub.max = String(scrubMax);
    r.scrub.value = String(s.cursor);

    const lifeText =
      `cyc ${s.cycle ?? "—"} · soh ${s.soh !== null ? s.soh.toFixed(1) : "—"}%` +
      (s.projected ? " PROJECTED" : "");
    if (r.life.textContent !== lifeText) r.life.textContent = lifeText;

    const playLabel = s.playing ? "❚❚" : "▶";
    if (r.play.textContent !== playLabel) r.play.textContent = playLabel;

    r.play.style.borderColor = borderColorOf(s.playing);
    r.breathe.style.borderColor = borderColorOf(s.breathe);
    r.annotations.style.borderColor = borderColorOf(s.annotations);
    r.layout.wound.style.borderColor = borderColorOf(s.layout === "wound");
    r.layout.unrolled.style.borderColor = borderColorOf(s.layout === "unrolled");
    for (const [name, b] of Object.entries(r.palettes)) {
      b.style.borderColor = borderColorOf(s.themeName === name);
    }

    const telText =
      `${telemetry.fps} fps · ${telemetry.calls} calls · ` +
      `${count(telemetry.triangles)} tris · ${count(telemetry.vertices)} verts`;
    if (r.tel.textContent !== telText) r.tel.textContent = telText;
  }

  build();
  render();

  return {
    update(next) {
      if (disposed) return;
      last = next;
      // The palette is the whole restyle trigger: same object, same rail.
      if (next.theme !== theme) {
        theme = next.theme;
        build();
      }
      render();
    },
    setTelemetry(t) {
      telemetry = { ...t };
    },
    collapsed: () => collapsed,
    toggleCollapsed: toggle,
    dispose() {
      disposed = true;
      refs?.rail.remove();
      refs?.pill.remove();
      refs = null;
    },
  };
}
