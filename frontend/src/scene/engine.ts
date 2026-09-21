/**
 * The renderer — the only file in this directory that knows three.js exists.
 *
 * Everything numeric lives in `geometry.ts` (DOM-free, tested by Node);
 * everything else about a host lives in the host. This module's whole job is to
 * take a built scene and paint it, then take a jitter click and/or another
 * built scene and repaint. That split is why the anatomy can be unit-tested at
 * all: nothing here decides what a cell looks like, it only decides where the
 * camera is.
 *
 * Behaviours a viewer would expect are all wired from controls the host owns:
 * orbit/zoom/pan, hover-to-inspect, the explode slider, the casing toggle, the
 * annotation layer, and the timeline cursor that replays the measured life and
 * then the platform's own projection.
 */

// Named imports, not `import * as THREE`: the bundle is committed to
// app/static/ and served to browsers that never built anything, so anything
// three ships that this scene does not draw is payload with no purpose. The
// named form is also what the module actually uses — one list, no wildcard.
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  Group,
  HemisphereLight,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  Raycaster,
  Scene,
  Vector2,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";

import {
  DEFAULT_BUILD_OPTIONS,
  buildScene,
  gaugeBandMesh,
  mergeBuildOptions,
  partReadings,
  vertexCount,
} from "./geometry.ts";
import type { BuildOptions, BuiltScene, PartReading } from "./geometry.ts";
import { provenanceColor } from "./theme.ts";
import type { CellSceneSpec } from "./types.ts";

/**
 * How far the labels of unlit parts fade when one part is lit.
 *
 * One number for both the hover and the host's own pick, so the same gesture
 * looks the same whichever way the viewer made it.
 */
const DIM_OPACITY = 0.3;

/** What the host is told on every frame the cursor or a hover changes. */
export interface FrameState {
  cursor: number;
  cycle: number | null;
  soh: number | null;
  gaugeLabel: string;
  gaugeColor: string;
  projected: boolean;
  projectionLabel: string | null;
  inspected: string | null;
  scaleNote: string | null;
  /** Cycle count of the measured record, so a host can mark "today" on a slider. */
  measuredCount: number;
  lastMeasuredCycle: number | null;
  partCount: number;
  drawnParts: number;
  /** How many future positions the timeline carries (0 when no forecast). */
  projectionLength: number;
}

export interface MountOptions extends Partial<BuildOptions> {
  onInspect?: (partId: string | null) => void;
  onFrame?: (state: FrameState) => void;
  /**
   * Playback stopped because the timeline ran out. The host owns the button, so
   * only the host can put its label back — without this a "Play life" button
   * stays stuck on "Pause" after the cell has finished aging.
   */
  onPlaybackEnd?: () => void;
}

export interface CellSceneHandle {
  update(spec: CellSceneSpec, options?: Partial<BuildOptions>): void;
  /** Scrub the timeline without rebuilding the spec. */
  setCursor(index: number): void;
  /** Highlight one part and turn the camera to its side (`null` clears it). */
  inspect(partId: string | null): void;
  /** Show or hide the floating part labels. */
  setAnnotations(visible: boolean): void;
  /**
   * Every part's reading at the current cursor — what a host needs to print a
   * table beside the scene that cannot disagree with it. Empty only before the
   * first build (the scene is mounted synchronously, so this is rare).
   */
  parts(): PartReading[];
  /** Walk the cell's life; the host keeps the button. */
  play(intervalMs?: number): boolean;
  pause(): void;
  isPlaying(): boolean;
  resetView(): void;
  /** The most recent state handed to `onFrame`. */
  state(): FrameState | null;
  dispose(): void;
}

interface PartObject {
  id: string;
  group: Group;
  mesh: Mesh;
  material: MeshStandardMaterial;
  label: CSS2DObject;
  /** Emissive intensity the current build asked for, before hover highlight. */
  baseEmissive: number;
}

const MAX_PIXEL_RATIO = 2;

function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext("webgl2") || canvas.getContext("webgl")),
    );
  } catch {
    return false;
  }
}

/**
 * Paint one cell in `container` and hand back its handle.
 *
 * Returns null when the browser cannot give us a WebGL context at all — the
 * caller shows its own empty state, because a scene that silently renders
 * nothing is worse than one that admits it could not start.
 */
export function mountCellScene(
  container: HTMLElement,
  spec: CellSceneSpec,
  options: MountOptions = {},
): CellSceneHandle | null {
  if (!isWebGLAvailable()) return null;

  const state: FrameState = {
    cursor: 0, cycle: null, soh: null, gaugeLabel: "", gaugeColor: "#000000",
    projected: false, projectionLabel: null, inspected: null, scaleNote: null,
    measuredCount: 0, lastMeasuredCycle: null, partCount: 0, drawnParts: 0,
    projectionLength: 0,
  };
  let build: BuildOptions = { ...DEFAULT_BUILD_OPTIONS };

  /** Fold a host's view options into the build state. Callbacks are not options. */
  function applyOptions(patch: Partial<BuildOptions>): void {
    build = mergeBuildOptions(build, patch);
  }
  let scene: BuiltScene | null = null;
  let currentSpec: CellSceneSpec = spec;
  let disposed = false;
  let playTimer: number | null = null;
  let hovered: string | null = null;
  /** A part the host pinned (a click in its own list), which outranks hover. */
  let pinned: string | null = null;
  let annotationsVisible = true;

  // ── Renderer, scene, camera ────────────────────────────────────────────
  const renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  renderer.setSize(container.clientWidth || 640, container.clientHeight || 480, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";
  renderer.domElement.style.borderRadius = "10px";
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth || 640, container.clientHeight || 480);
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.left = "0";
  labelRenderer.domElement.style.pointerEvents = "none";
  container.appendChild(labelRenderer.domElement);

  const scene3 = new Scene();
  const camera = new PerspectiveCamera(38, (container.clientWidth || 640) / (container.clientHeight || 480), 0.01, 100);
  camera.position.set(0.95, 0.55, 1.15);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.75;
  controls.maxDistance = 6;
  controls.target.set(0, 0, 0);
  controls.update();

  scene3.add(new HemisphereLight(0xffffff, 0x1a202c, 1.1));
  const key = new DirectionalLight(0xffffff, 1.6);
  key.position.set(1.4, 2.2, 1.6);
  scene3.add(key);
  const rim = new DirectionalLight(0x63b3ed, 0.7);
  rim.position.set(-1.6, -0.4, -1.2);
  scene3.add(rim);

  const stage = new Group();
  scene3.add(stage);

  const partsRoot = new Group();
  stage.add(partsRoot);
  const objects = new Map<string, PartObject>();

  let floorMesh: Mesh | null = null;
  let shadowMesh: Mesh | null = null;
  let gaugeMesh: Mesh | null = null;
  const gaugeMaterial = new MeshStandardMaterial({ transparent: true, opacity: 0.75, roughness: 0.4, metalness: 0.1 });

  function toBuffer(mesh: { positions: Float32Array; normals: Float32Array; indices: Uint32Array }): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(mesh.positions, 3));
    geometry.setAttribute("normal", new BufferAttribute(mesh.normals, 3));
    geometry.setIndex(new BufferAttribute(mesh.indices, 1));
    return geometry;
  }

  function labelElement(label: string, value: string, provenance: string, color: string): HTMLDivElement {
    const el = document.createElement("div");
    el.className = "cell-scene-label";
    el.style.cssText =
      "font:500 11px/1.25 ui-sans-serif,system-ui,'Segoe UI',sans-serif;color:#e2e8f0;" +
      `border-left:2px solid ${color};padding:2px 6px;background:rgba(11,17,32,0.72);` +
      "border-radius:4px;white-space:nowrap;backdrop-filter:blur(2px);";
    el.dataset.part = label;
    el.dataset.provenance = provenance;
    el.textContent = value ? `${label} · ${value}` : label;
    return el;
  }

  function formatValue(part: { value: number | null; unit: string; carried: boolean }): string {
    if (part.value === null || !Number.isFinite(part.value)) return "no reading";
    const decimals = Math.abs(part.value) >= 100 ? 1 : Math.abs(part.value) >= 10 ? 2 : 3;
    const unit = part.unit.length > 18 ? part.unit.slice(0, 16) + "…" : part.unit;
    return `${part.value.toFixed(decimals)}${unit ? " " + unit : ""}${part.carried ? " (carried)" : ""}`;
  }

  /** Paint (or repaint) everything a build carries. Called from update(). */
  function paint(built: BuiltScene, spec_: CellSceneSpec): void {
    scene = built;
    // Background and stage furniture first: they are host tokens, not data.
    scene3.background = new Color(spec_.theme.background);

    if (!floorMesh) {
      floorMesh = new Mesh(toBuffer(built.chrome.floor), new MeshStandardMaterial({ color: new Color(spec_.theme.grid), roughness: 0.95, metalness: 0.05 }));
      stage.add(floorMesh);
    }
    if (!shadowMesh) {
      shadowMesh = new Mesh(toBuffer(built.chrome.shadow), new MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
      stage.add(shadowMesh);
    }

    const drawn = built.parts.filter((part) => part.drawn && vertexCount(part.mesh) > 0);
    const live = new Set<string>(drawn.map((part) => part.id));

    for (const part of drawn) {
      let object = objects.get(part.id);
      if (!object) {
        const material = new MeshStandardMaterial({
          transparent: true, roughness: 0.55, metalness: 0.25, side: DoubleSide,
        });
        const mesh = new Mesh(toBuffer(part.mesh), material);
        mesh.userData.partId = part.id;
        const group = new Group();
        group.add(mesh);
        const label = new CSS2DObject(labelElement(part.label, "", part.provenance, spec_.theme.muted));
        group.add(label);
        partsRoot.add(group);
        object = { id: part.id, group, mesh, material, label, baseEmissive: part.emissive };
        objects.set(part.id, object);
      }
      object.mesh.geometry.dispose();
      object.mesh.geometry = toBuffer(part.mesh);
      object.material.color = new Color(part.color);
      object.material.opacity = part.opacity;
      object.material.emissive = new Color(part.color);
      object.baseEmissive = part.emissive;
      object.material.emissiveIntensity = part.emissive;
      object.material.roughness = part.opacity < 0.5 ? 0.25 : 0.55;
      object.material.metalness = part.id.startsWith("terminal") || part.id === "can" ? 0.55 : 0.2;
      object.group.position.set(0, 0, 0);
      object.group.visible = true;

      const el = object.label.element as HTMLDivElement;
      el.textContent = `${part.label} · ${formatValue(part)}`;
      const border = provenanceColor(part.provenance, spec_.theme);
      el.style.borderLeftColor = border;
      el.style.opacity = part.available ? "1" : "0.62";
      el.dataset.part = part.id;
      el.dataset.provenance = part.provenance;
      el.dataset.available = String(part.available);
      el.title = part.available
        ? `${part.label} — ${part.provenance || "unmeasured"}${part.unit ? ` (${part.unit})` : ""}`
        : `${part.label} — no measurement: ${part.reason ?? "not measured"}`;
      object.label.position.set(part.anchor[0], part.anchor[1], part.anchor[2]);
      object.label.visible = annotationsVisible;
    }

    for (const [id, object] of objects) {
      if (live.has(id)) continue;
      object.group.visible = false;
    }

    // The state gauge: a band on the casing, height = remaining capacity.
    const gaugeGeometry = toBuffer(gaugeBandMesh(spec_, built.gauge.fraction));
    if (!gaugeMesh) {
      gaugeMesh = new Mesh(gaugeGeometry, gaugeMaterial);
      stage.add(gaugeMesh);
    } else {
      gaugeMesh.geometry.dispose();
      gaugeMesh.geometry = gaugeGeometry;
    }
    gaugeMaterial.color = new Color(built.gauge.color);
    gaugeMaterial.emissive = new Color(built.gauge.color);
    gaugeMaterial.emissiveIntensity = built.gauge.projected ? 0.75 : 0.3;
  }

  // ── Hover inspection ──────────────────────────────────────────────────
  const raycaster = new Raycaster();
  const pointer = new Vector2();
  let pointerInside = false;

  function setHovered(id: string | null): void {
    if (hovered === id) return;
    hovered = id;
    if (pinned === null) {
      for (const [partId, object] of objects) {
        const isTarget = partId === id;
        // Highlight is an offset from the value the build asked for, never a
        // mutation of it — otherwise hovering would permanently brighten a part
        // until the next rebuild.
        object.material.emissiveIntensity = isTarget ? object.baseEmissive + 0.6 : object.baseEmissive;
        const el = object.label.element as HTMLDivElement;
        const dimmed = id !== null && !isTarget;
        el.style.opacity = dimmed && el.dataset.available === "true" ? String(DIM_OPACITY) : "1";
      }
      options.onInspect?.(id);
    }
    // The host paints from `onFrame`, so a hover that the host never hears
    // about is a hover it cannot show: the part list and the scene must agree
    // about which part is lit at all times, not only after a click.
    emit();
  }

  function onPointerMove(event: PointerEvent): void {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    pointerInside = true;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(
      [...objects.values()].filter((o) => o.group.visible).map((o) => o.mesh),
      false,
    );
    setHovered(hits.length > 0 ? (hits[0].object.userData.partId as string) : null);
  }

  function onPointerLeave(): void {
    pointerInside = false;
    setHovered(null);
  }

  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);

  // ── Resize ────────────────────────────────────────────────────────────
  function resize(): void {
    const width = container.clientWidth || 640;
    const height = container.clientHeight || 480;
    renderer.setSize(width, height, false);
    labelRenderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  let observer: ResizeObserver | null = null;
  if (typeof ResizeObserver !== "undefined") {
    observer = new ResizeObserver(resize);
    observer.observe(container);
  } else {
    window.addEventListener("resize", resize);
  }

  // ── Render loop ───────────────────────────────────────────────────────
  let raf = 0;
  function tick(): void {
    if (disposed) return;
    raf = window.requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene3, camera);
    labelRenderer.render(scene3, camera);
    if (pointerInside) {
      // A hover highlight decays once the pointer stops moving; refresh it so
      // the lit part stays lit while the camera orbits around it.
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(
        [...objects.values()].filter((o) => o.group.visible).map((o) => o.mesh),
        false,
      );
      setHovered(hits.length > 0 ? (hits[0].object.userData.partId as string) : null);
    }
  }

  function emit(): void {
    if (!scene) return;
    state.cursor = scene.cursor;
    state.cycle = scene.gauge.cycle;
    state.soh = scene.gauge.soh;
    state.gaugeLabel = scene.gauge.label;
    state.gaugeColor = scene.gauge.color;
    state.projected = scene.gauge.projected;
    state.projectionLabel = scene.timeline.projectionLabel;
    state.inspected = pinned ?? hovered;
    state.scaleNote = scene.scaleNote;
    state.measuredCount = scene.timeline.measuredCount;
    state.lastMeasuredCycle = scene.timeline.lastMeasuredCycle;
    state.partCount = scene.parts.length;
    state.drawnParts = scene.parts.filter((part) => part.drawn).length;
    state.projectionLength = Math.max(0, scene.timeline.cycles.length - scene.timeline.measuredCount);
    options.onFrame?.({ ...state });
  }

  function rebuild(spec_: CellSceneSpec): void {
    const built = buildScene(spec_, build);
    paint(built, spec_);
    emit();
  }

  // ── Public handle ─────────────────────────────────────────────────────
  const handle: CellSceneHandle = {
    update(spec_, opts = {}) {
      if (disposed) return;
      currentSpec = spec_;
      applyOptions(opts);
      rebuild(spec_);
    },
    setCursor(index) {
      if (disposed || !scene) return;
      build.cursor = index;
      rebuild(currentSpec);
    },
    inspect(partId) {
      if (disposed) return;
      pinned = partId;
      for (const [id, object] of objects) {
        const isTarget = id === partId;
        object.material.emissiveIntensity = isTarget ? object.baseEmissive + 0.6 : object.baseEmissive;
        const element = object.label.element as HTMLDivElement;
        element.style.opacity = partId === null || isTarget ? "1" : String(DIM_OPACITY);
      }
      // Turn the camera to the inspected part's own side, keeping the distance
      // and height the viewer chose — a click in a list should not throw away
      // their framing.
      if (partId && scene) {
        const part = scene.parts.find((candidate) => candidate.id === partId);
        if (part) {
          const azimuth = Math.atan2(part.anchor[2], part.anchor[0]);
          const distance = camera.position.length();
          camera.position.set(
            distance * Math.cos(azimuth),
            camera.position.y,
            distance * Math.sin(azimuth),
          );
          controls.update();
        }
      }
      options.onInspect?.(partId);
      emit();
    },
    setAnnotations(visible) {
      annotationsVisible = visible;
      for (const object of objects.values()) object.label.visible = visible;
    },
    parts: () => (scene ? partReadings(currentSpec, scene) : []),
    play(intervalMs = 120) {
      if (disposed) return false;
      handle.pause();
      // A timeline of one frame has nothing to play, and saying so is the only
      // honest answer.
      if (!scene || scene.timeline.cycles.length < 2) return false;
      // Pressing play on a scene already at its last frame replays from the
      // first: a button that silently does nothing is worse than a rewind, and
      // the default view now opens on the last measured cycle.
      if (build.cursor >= scene.timeline.cycles.length - 1) {
        build.cursor = 0;
        rebuild(currentSpec);
      }
      playTimer = window.setInterval(() => {
        if (!scene) return;
        const next = build.cursor + 1;
        if (next >= scene.timeline.cycles.length) {
          handle.pause();
          options.onPlaybackEnd?.();
          return;
        }
        build.cursor = next;
        rebuild(currentSpec);
      }, Math.max(30, intervalMs));
      return true;
    },
    pause() {
      if (playTimer !== null) {
        window.clearInterval(playTimer);
        playTimer = null;
      }
    },
    isPlaying: () => playTimer !== null,
    resetView() {
      camera.position.set(0.95, 0.55, 1.15);
      controls.target.set(0, 0, 0);
      controls.update();
    },
    state: () => ({ ...state }),
    dispose() {
      if (disposed) return;
      disposed = true;
      handle.pause();
      window.cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      observer?.disconnect();
      window.removeEventListener("resize", resize);
      for (const object of objects.values()) {
        object.mesh.geometry.dispose();
        object.material.dispose();
        object.label.element.remove();
      }
      objects.clear();
      floorMesh?.geometry.dispose();
      shadowMesh?.geometry.dispose();
      gaugeMesh?.geometry.dispose();
      gaugeMaterial.dispose();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
    },
  };

  // A host's options apply to the very first frame, not the second one: the
  // scene it asked for is the scene it gets.
  applyOptions(options);
  rebuild(spec);
  raf = window.requestAnimationFrame(tick);
  return handle;
}
