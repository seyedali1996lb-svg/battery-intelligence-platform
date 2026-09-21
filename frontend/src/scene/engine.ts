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
  ACESFilmicToneMapping,
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
  PCFShadowMap,
  PMREMGenerator,
  PerspectiveCamera,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { materialFor } from "./materials.ts";

import {
  DEFAULT_BUILD_OPTIONS,
  buildScene,
  gaugeBandMesh,
  mergeBuildOptions,
  partReadings,
  vertexCount,
} from "./geometry.ts";
import type { BuildOptions, BuiltScene, PartReading } from "./geometry.ts";
import { paletteFor, provenanceColor, readPref, writePref } from "./theme.ts";
import { CARD_H_ONE, CARD_H_TWO, chromeSvg, desaturateHex, layoutFlank } from "./annotation.ts";
import { poseFacing, poseLerp, poseToPosition, type CameraPose } from "./tween.ts";
import { composeDossier } from "./dossier.ts";
import type { CellSceneSpec, DossierSpecRow, SceneTheme } from "./types.ts";

/**
 * How far the labels of unlit parts fade when one part is lit.
 *
 * One number for both the hover and the host's own pick, so the same gesture
 * looks the same whichever way the viewer made it.
 */
const DIM_OPACITY = 0.3;

/**
 * The dark palettes' backdrop: a vignette in the palette's own background, so
 * the stage's edges deepen without the renderer painting a second scene.
 */
function vignette(theme: SceneTheme): string {
  return `background:radial-gradient(ellipse at center, transparent 55%, ${theme.background}cc 100%)`;
}

/**
 * The light palettes' backdrop: a procedural paper grain, ~64×64 pixels drawn
 * once onto a canvas. Deterministic (a fixed hash of the pixel index, never
 * `Math.random`) so the same document renders the same paper every time — and
 * a canvas that cannot be had (a bare DOM shim) falls back to no grain rather
 * than to a broken stage.
 */
function paperGrain(theme: SceneTheme): string {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d");
  if (!ctx) return vignette(theme);
  const img = ctx.createImageData(64, 64);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 200 + Math.floor((i * 2654435761) % 55);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 14;
  }
  ctx.putImageData(img, 0, 0);
  return `background-image:url(${c.toDataURL()});background-repeat:repeat;opacity:0.5;mix-blend-mode:multiply`;
}

/**
 * Where the annotation badges live, in pixels, relative to the stage's own
 * edges.
 *
 * The badges are pinned to these two margins and the leader lines are drawn
 * *to* them, so the annotation layer reads as a fixed legend beside the cell
 * rather than as labels floating over whatever part happens to be nearest. The
 * line's terminus is `margin + width` from each edge — the badge's inner
 * border — and it is one constant on both sides because a badge that stopped
 * at a different distance on the left than on the right would look mis-set.
 */
const BADGE_MARGIN = 12;
const BADGE_WIDTH = 158;

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
  /**
   * Which of the document's palettes to paint with on the first frame. Absent
   * means the reader's stored preference if there is one, else the document's
   * own `theme` — the default has to be the document's, not the host's taste.
   */
  theme?: string;
}

export interface CellSceneHandle {
  update(spec: CellSceneSpec, options?: Partial<BuildOptions>): void;
  /** Scrub the timeline without rebuilding the spec. */
  setCursor(index: number): void;
  /** Highlight one part and turn the camera to its side (`null` clears it). */
  inspect(partId: string | null): void;
  /** Show or hide the annotation layer (the badges and their leader lines). */
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
  /**
   * Paint the stage with one of the document's named palettes (`null`/unknown
   * returns to the document's own `theme`). Bands never move: a palette may
   * repaint the platform's colours, never re-band what it calls healthy.
   */
  setTheme(name: string | null): void;
  /** The most recent state handed to `onFrame`. */
  state(): FrameState | null;
  dispose(): void;
}

interface PartObject {
  id: string;
  group: Group;
  mesh: Mesh;
  material: MeshStandardMaterial;
  /** Emissive intensity the current build asked for, before hover highlight. */
  baseEmissive: number;
  /** The build's own colour, so clearing a selection restores exactly it. */
  baseColor: string;
  /** The build's own opacity — likewise, never a remembered dim. */
  baseOpacity: number;
  /** Uniform the Fresnel rim reads; 1 on the selected part, 0 everywhere else. */
  rim: { value: number };
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
  /** The document's chosen palette name: host option, else stored pref, else none. */
  let themeName: string | null = options.theme ?? readPref<string>("theme") ?? null;
  if (themeName === "") themeName = null;
  let themeVariant: "dark" | "light" = "dark";
  const activePalette = (): SceneTheme => paletteFor(currentSpec, themeName);
  /**
   * One uniform object shared by every part's rim shader, so a palette swap
   * recolours all nineteen rims with a single `.set()` instead of waiting for
   * a recompile that may never come.
   */
  const rimColorUniform = { value: new Color() };
  /** What the expensive stage styling last saw; "" forces the first pass. */
  let styleKey = "";
  /**
   * Themed copies of a host's documents, one per source. Caching is what keeps
   * `framedSpec !== spec_` honest: without it every view toggle while a palette
   * is active would hand `rebuild` a fresh object and re-frame the camera,
   * throwing away the orbit over a checkbox.
   */
  const themedSpecs = new WeakMap<CellSceneSpec, CellSceneSpec>();
  function themed(source: CellSceneSpec): CellSceneSpec {
    const palette = themeName ? paletteFor(source, themeName) : source.theme;
    if (palette === source.theme) return source;
    const cached = themedSpecs.get(source);
    if (cached && cached.theme === palette) return cached;
    const out = { ...source, theme: palette };
    themedSpecs.set(source, out);
    return out;
  }
  let disposed = false;
  let playTimer: number | null = null;
  let hovered: string | null = null;
  /** A part the host pinned (a click in its own list), which outranks hover. */
  let pinned: string | null = null;
  let annotationsVisible = true;
  /** The document the camera was framed for, so a scrub does not re-frame it. */
  let framedSpec: CellSceneSpec | null = null;
  let lastBounds = { radius: 0.5, height: 1 };

  // ── Renderer, scene, camera ────────────────────────────────────────────
  const renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  // Filmic tone mapping, so a specular highlight rolls off instead of clipping
  // to a white blob — the difference between "metal" and "shiny plastic" is
  // mostly what happens at the top of the range.
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  // PCFShadowMap, not PCFSoftShadowMap: three r186 still exports the latter as
  // a constant but has removed its implementation, so asking for it silently
  // falls back to PCF and warns to a console nobody reads in CI. The softness
  // comes from the filter radius instead.
  renderer.shadowMap.type = PCFShadowMap;
  renderer.setSize(container.clientWidth || 640, container.clientHeight || 480, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";
  renderer.domElement.style.borderRadius = "10px";
  container.appendChild(renderer.domElement);

  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }

  const overlayContainer = document.createElement("div");
  overlayContainer.className = "cell-scene-overlay";
  overlayContainer.style.position = "absolute";
  overlayContainer.style.top = "0";
  overlayContainer.style.left = "0";
  overlayContainer.style.width = "100%";
  overlayContainer.style.height = "100%";
  overlayContainer.style.pointerEvents = "none";
  overlayContainer.style.overflow = "hidden";
  container.appendChild(overlayContainer);

  // The palette's backdrop, painted over the canvas and under everything the
  // reader reads: a vignette for dark palettes, paper grain for light ones.
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:0";
  overlayContainer.appendChild(backdrop);

  const svgNamespace = "http://www.w3.org/2000/svg";
  const svgOverlay = document.createElementNS(svgNamespace, "svg");
  svgOverlay.setAttribute("class", "cell-scene-leader-svg");
  svgOverlay.style.position = "absolute";
  svgOverlay.style.top = "0";
  svgOverlay.style.left = "0";
  svgOverlay.style.width = "100%";
  svgOverlay.style.height = "100%";
  svgOverlay.style.pointerEvents = "none";
  overlayContainer.appendChild(svgOverlay);

  const badgeOverlay = document.createElement("div");
  badgeOverlay.setAttribute("class", "cell-scene-badge-overlay");
  badgeOverlay.style.position = "absolute";
  badgeOverlay.style.top = "0";
  badgeOverlay.style.left = "0";
  badgeOverlay.style.width = "100%";
  badgeOverlay.style.height = "100%";
  badgeOverlay.style.pointerEvents = "none";
  overlayContainer.appendChild(badgeOverlay);

  // The drafting frame: ruled border, ticks, compass rose. Above the badges
  // (it frames the whole plate, callouts included) and below the dossier card.
  const chromeLayer = document.createElement("div");
  chromeLayer.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:2";
  overlayContainer.appendChild(chromeLayer);

  // The floating dossier card: the stage's own copy of the document's prose,
  // shown while a part is pinned or hovered. Positioned below the HUD strip
  // (which sits at the top of the stage) and scrollable, because a dossier
  // with its full spec table must never be clipped into unreadability.
  const dossierCard = document.createElement("div");
  dossierCard.id = "cell-scene-dossier";
  dossierCard.style.cssText =
    "position:absolute;top:56px;left:16px;width:300px;max-height:60%;overflow:auto;" +
    "display:none;pointer-events:auto;padding:10px 12px;border-radius:6px;" +
    `background:${currentSpec.theme.panel};border:1px solid ${currentSpec.theme.grid};` +
    `color:${currentSpec.theme.text};font-family:${currentSpec.theme.fonts?.mono ?? "ui-monospace, monospace"};` +
    "font-size:11px;z-index:3;";
  overlayContainer.appendChild(dossierCard);

  /**
   * Paint the stage for the active palette: exposure, key/fill/rim lights,
   * environment strength, backdrop, drafting frame and the dossier card's
   * own surface. Called from every paint, so the cheap facts below always
   * hold; the expensive strings (grain canvas, chrome SVG) rebuild only when
   * the palette, its colours or the stage size actually changed.
   */
  function applyStageStyle(): void {
    const theme = activePalette();
    themeVariant = theme.variant ?? "dark";
    scene3.background = new Color(theme.background);
    renderer.toneMappingExposure = themeVariant === "light" ? 1.15 : 1.05;
    key.color.set(themeVariant === "light" ? "#fff6e0" : "#ffffff");
    fill.intensity = themeVariant === "light" ? 0.9 : 0.7;
    fill.groundColor.set(theme.panel); // bounce light is the palette's own panel
    rimLight.color.set(theme.accent);
    scene3.environmentIntensity = themeVariant === "light" ? 0.4 : 0.55;
    rimColorUniform.value.set(theme.accent2 ?? theme.accent);

    const nextKey =
      `${theme.background}|${theme.panel}|${theme.grid}|${theme.text}|${theme.accent}|${themeVariant}` +
      `|${container.clientWidth}x${container.clientHeight}`;
    if (nextKey === styleKey) return;
    styleKey = nextKey;

    backdrop.innerHTML = themeVariant === "light" ? paperGrain(theme) : vignette(theme);
    chromeLayer.innerHTML = chromeSvg({
      theme,
      ornate: themeVariant === "light",
      width: container.clientWidth || 640,
      height: container.clientHeight || 480,
    });
    dossierCard.style.background = theme.panel;
    dossierCard.style.borderColor = theme.grid;
    dossierCard.style.color = theme.text;
    writePref("theme", themeName ?? "");
  }

  const scene3 = new Scene();
  // An environment map, not extra lamps: metals need something to reflect, and
  // a room-sized procedural environment gives the can and the foils a believable
  // highlight without shipping an HDRI or reaching for the network.
  const pmrem = new PMREMGenerator(renderer);
  const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene3.environment = environment;
  scene3.environmentIntensity = 0.55;
  pmrem.dispose();
  const camera = new PerspectiveCamera(38, (container.clientWidth || 640) / (container.clientHeight || 480), 0.01, 100);
  camera.position.set(0.95, 0.55, 1.15);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.75;
  controls.maxDistance = 6;
  controls.target.set(0, 0, 0);
  controls.update();

  // A click in a list should *turn* the camera, not teleport it: the tween
  // carries the viewer's own framing (distance, elevation, height) to a pose
  // facing the part, and any grab of the controls cancels it mid-flight —
  // the hand always wins over an animation.
  let camTween: { from: CameraPose; to: CameraPose; start: number; dur: number } | null = null;

  function currentPose(): CameraPose {
    const offset = camera.position.clone().sub(controls.target);
    return {
      azimuth: Math.atan2(offset.x, offset.z),
      elevation: Math.asin(Math.max(-1, Math.min(1, offset.y / Math.max(offset.length(), 1e-9)))),
      distance: offset.length(),
      targetY: controls.target.y,
    };
  }

  function applyPose(p: CameraPose): void {
    const pos = poseToPosition(p);
    camera.position.set(pos.x, pos.y, pos.z);
    controls.target.set(0, p.targetY, 0);
    controls.update();
  }

  function framePose(to: CameraPose, durMs = 500): void {
    camTween = { from: currentPose(), to, start: performance.now(), dur: durMs };
  }

  controls.addEventListener("start", () => {
    camTween = null;
  });

  // A three-point rig over the environment: the hemisphere carries the fill (so
  // nothing is ever pitch black), the key does the modelling and casts the
  // shadow, and the rim separates the cell's far edge from the background.
  // All three are named because `applyStageStyle` re-tints them per palette.
  const fill = new HemisphereLight(0xffffff, 0x1a202c, 0.55);
  scene3.add(fill);
  const key = new DirectionalLight(0xffffff, 2.1);
  key.position.set(1.4, 2.2, 1.6);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.012;
  key.shadow.radius = 2.5;
  scene3.add(key);
  const rimLight = new DirectionalLight(0x63b3ed, 0.55);
  rimLight.position.set(-1.6, -0.4, -1.2);
  scene3.add(rimLight);

  /**
   * The framing a fresh view wants: the cell's bounding sphere seen from the
   * house direction. Shared by `fitView` (a new document, immediately) and
   * `resetView` (a deliberate return, tweened) so the two can never drift into
   * framing the same cell differently.
   *
   * The camera used to be a fixed position tuned for the 18650's tall aspect,
   * so a prismatic cell — 20.5 mm wide over 5.4 mm thick — was framed by luck.
   * Fitting the *distance* to the bounding sphere and keeping only the viewing
   * direction fixed gives every form factor the same framing.
   */
  function homePose(radius: number, height: number): CameraPose {
    const margin = height > radius * 2 ? 1.5 : 1.35;
    const extent = Math.max(radius, height / 2) * margin;
    const distance = extent / Math.sin(((camera.fov / 2) * Math.PI) / 180);
    const dir = new Vector3(0.62, 0.36, 0.72).normalize();
    return {
      azimuth: Math.atan2(dir.x, dir.z),
      elevation: Math.asin(Math.max(-1, Math.min(1, dir.y))),
      distance,
      targetY: 0,
    };
  }

  /** Frame the cell from its own bounds — a new document's first look. */
  function fitView(radius: number, height: number): void {
    const pose = homePose(radius, height);
    const pos = poseToPosition(pose);
    camera.position.set(pos.x, pos.y, pos.z);
    controls.minDistance = pose.distance * 0.5;
    controls.maxDistance = pose.distance * 3.5;
    controls.target.set(0, 0, 0);
    controls.update();
  }

  /** Keep the shadow camera just wide enough for whatever is being drawn. */
  function fitShadow(radius: number): void {
    const extent = Math.max(0.9, radius * 2.8);
    const shadowCamera = key.shadow.camera;
    shadowCamera.left = -extent;
    shadowCamera.right = extent;
    shadowCamera.top = extent;
    shadowCamera.bottom = -extent;
    shadowCamera.near = 0.4;
    shadowCamera.far = 8;
    shadowCamera.updateProjectionMatrix();
  }

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

  function formatValue(part: { value: number | null; unit: string; carried: boolean }): string {
    if (part.value === null || !Number.isFinite(part.value)) return "no reading";
    const decimals = Math.abs(part.value) >= 100 ? 1 : Math.abs(part.value) >= 10 ? 2 : 3;
    const unit = part.unit.length > 18 ? part.unit.slice(0, 16) + "…" : part.unit;
    return `${part.value.toFixed(decimals)}${unit ? " " + unit : ""}${part.carried ? " (carried)" : ""}`;
  }

  /**
   * Text into markup: the labels come from a producer's document, so a part
   * called `Foo "bar" <baz>` must not be able to close an attribute the scene
   * just opened for it.
   */
  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Paint (or repaint) everything a build carries. Called from update(). */
  function paint(built: BuiltScene): void {
    scene = built;
    // Background, lighting, backdrop and chrome come from the ACTIVE PALETTE
    // rather than the document's default theme, so they repaint when the
    // reader switches — and the cheap facts (scalars, one colour) run every
    // paint while the expensive strings (grain canvas, chrome SVG) only
    // rebuild when the palette, its colours or the stage size changed.
    applyStageStyle();

    lastBounds = built.bounds;
    if (!floorMesh) {
      floorMesh = new Mesh(toBuffer(built.chrome.floor), new MeshStandardMaterial({ color: new Color(activePalette().grid), roughness: 0.95, metalness: 0.05 }));
      // The stage catches the cell's shadow; the painted contact-shadow disc
      // under it stays, because at this scale it does the ambient-occlusion job
      // a shadow map cannot.
      floorMesh.receiveShadow = true;
      stage.add(floorMesh);
    }
    // The floor is stage furniture: it follows the palette's grid tone like
    // every other chrome colour, or a light palette would sit on a dark disc.
    (floorMesh.material as MeshStandardMaterial).color.set(activePalette().grid);
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
        // The Fresnel rim: a uniform-driven edge glow the selection treatment
        // turns up on exactly one part. Injection rather than a post-process
        // pass — the whole stage stays one render. Graceful by construction:
        // if three ever renames the chunk, `replace` matches nothing, the
        // shader compiles untouched, and the only cost is no rim.
        const rim = { value: 0 };
        material.onBeforeCompile = (shader) => {
          shader.uniforms.uRimStrength = rim;
          // The shared uniform: one object for all nineteen materials, so
          // `applyStageStyle` recolours every rim with a single `.set()`.
          shader.uniforms.uRimColor = rimColorUniform;
          shader.fragmentShader =
            "uniform float uRimStrength;\nuniform vec3 uRimColor;\n" +
            shader.fragmentShader.replace(
              "#include <emissivemap_fragment>",
              "#include <emissivemap_fragment>\n" +
                "float rimFactor = pow(1.0 - clamp(abs(dot(normalize(vViewPosition), normal)), 0.0, 1.0), 2.5);\n" +
                "totalEmissiveRadiance += uRimColor * rimFactor * uRimStrength;",
            );
        };
        material.customProgramCacheKey = () => "cell-rim";
        const mesh = new Mesh(toBuffer(part.mesh), material);
        mesh.userData.partId = part.id;
        const group = new Group();
        group.add(mesh);
        partsRoot.add(group);
        object = {
          id: part.id, group, mesh, material,
          baseEmissive: part.emissive,
          baseColor: part.color,
          baseOpacity: part.opacity,
          rim,
        };
        objects.set(part.id, object);
      }
      object.mesh.geometry.dispose();
      object.mesh.geometry = toBuffer(part.mesh);
      object.material.color = new Color(part.color);
      object.material.opacity = part.opacity;
      object.material.emissive = new Color(part.color);
      object.baseEmissive = part.emissive;
      object.material.emissiveIntensity = part.emissive;
      object.baseColor = part.color;
      object.baseOpacity = part.opacity;
      const shading = materialFor(part.id, themeVariant);
      object.material.roughness = shading.roughness;
      object.material.metalness = shading.metalness;
      object.material.envMapIntensity = shading.envMapIntensity;
      // Everything the anatomy draws can cast and receive, except the
      // electrolyte, which fills the can and would simply darken it.
      object.mesh.castShadow = part.id !== "electrolyte";
      object.mesh.receiveShadow = true;
      object.group.position.set(0, 0, 0);
      object.group.visible = true;
    }

    for (const [id, object] of objects) {
      if (live.has(id)) continue;
      object.group.visible = false;
    }

    // The state gauge: a band on the casing, height = remaining capacity.
    const gaugeGeometry = toBuffer(gaugeBandMesh(currentSpec, built.gauge.fraction));
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

    // A repaint restored the build's colours and opacities above; the active
    // selection must survive a scrub, so the treatment is re-applied from the
    // (unchanged) selection state rather than being remembered in a material.
    applyActiveTreatment();
  }

  // ── Hover inspection ──────────────────────────────────────────────────
  const raycaster = new Raycaster();
  const pointer = new Vector2();
  let pointerInside = false;

  /**
   * Paint the current selection onto every mesh.
   *
   * The active part keeps the build's colour, gains emissive and the Fresnel
   * rim; every other part falls back toward the theme's muted tone at 15%
   * opacity, so the eye lands where the callout points. `pinned` beats
   * `hovered` — the exact precedence the badges already compute at line one
   * of `updateLeaderLines` — so meshes and callouts cannot disagree about
   * what is selected. Everything is an offset from what the build asked for:
   * clearing the selection restores the build's own colour and opacity, never
   * a remembered highlight.
   */
  function applyActiveTreatment(): void {
    const active = pinned ?? hovered;
    for (const object of objects.values()) {
      const isActive = active !== null && object.id === active;
      // Highlight is an offset from the value the build asked for, never a
      // mutation of it — otherwise hovering would permanently brighten a part
      // until the next rebuild. The annotation badges dim the same way, from
      // the same precedence, in `updateLeaderLines`.
      object.material.emissiveIntensity = isActive ? object.baseEmissive + 0.6 : object.baseEmissive;
      object.rim.value = isActive ? 1 : 0;
      if (active !== null && !isActive) {
        object.material.opacity = Math.min(object.baseOpacity, 0.15);
        object.material.color.set(desaturateHex(object.baseColor, 0.7, currentSpec.theme.muted));
      } else {
        object.material.opacity = object.baseOpacity;
        object.material.color.set(object.baseColor);
      }
    }
  }

  function setHovered(id: string | null): void {
    if (hovered === id) return;
    hovered = id;
    if (pinned === null) {
      options.onInspect?.(id);
    }
    applyActiveTreatment();
    // The host paints from `onFrame`, so a hover that the host never hears
    // about is a hover it cannot show: the part list and the scene must agree
    // about which part is lit at all times, not only after a click.
    emit();
    updateLeaderLines();
    syncDossier();
  }

  // ── Dossier card ──────────────────────────────────────────────────────
  /** Tag dot: the same provenance-colour mapping the cards and React use. */
  const tagColor = (tag: string, theme: SceneTheme): string =>
    tag === "refusal"
      ? theme.muted
      : tag === "typical"
        ? theme.grid
        : theme.provenanceColors[tag] ?? theme.accent;

  function renderDossier(partId: string): void {
    const view = composeDossier(currentSpec, partId, build.cursor);
    if (!view) {
      // A document from before dossiers existed gets no card at all: an
      // empty shell would be a reading the document never wrote.
      dossierCard.style.display = "none";
      dossierCard.innerHTML = "";
      return;
    }
    const theme = activePalette(); // the reader's palette, not the document's default
    const row = (r: DossierSpecRow): string => `
      <div style="display:flex;justify-content:space-between;gap:8px;padding:2px 0">
        <span style="color:${theme.muted}">${escapeHtml(r.label)}</span>
        <span style="text-align:right">
          <span style="color:${r.tag === "refusal" ? theme.muted : theme.text}">${escapeHtml(r.value)}</span>
          <span style="color:${theme.muted};font-size:9px">${r.unit ? ` ${escapeHtml(r.unit)}` : ""}</span>
          <span title="${r.tag}" style="display:inline-block;width:6px;height:6px;border-radius:50%;margin-left:4px;background:${tagColor(r.tag, theme)}"></span>
        </span>
      </div>`;
    dossierCard.innerHTML =
      `<div style="font-weight:600;letter-spacing:0.08em">${escapeHtml(view.latinTitle ?? view.label)}</div>` +
      `<div style="color:${theme.muted};margin-bottom:6px">${escapeHtml(view.subsystem ?? "")}</div>` +
      `<div style="border-top:1px solid ${theme.grid};margin:6px 0;padding-top:6px">` +
      `<div style="color:${theme.accent};font-size:9px;letter-spacing:0.1em">LIVE AT CURSOR</div>${view.live.map(row).join("")}</div>` +
      (view.specs.length
        ? `<div style="border-top:1px solid ${theme.grid};margin:6px 0;padding-top:6px">` +
          `<div style="color:${theme.accent};font-size:9px;letter-spacing:0.1em">PHYSICAL SPEC</div>${view.specs.map(row).join("")}</div>`
        : "") +
      (view.insight
        ? `<div style="border-top:1px solid ${theme.grid};margin-top:6px;padding-top:6px;color:${theme.muted}">${escapeHtml(view.insight)}</div>`
        : "");
    dossierCard.style.display = "block";
  }

  /**
   * Show the dossier for the effective selection, or none.
   *
   * Called every frame (cheap: one string compare on the no-op path) because
   * the card moves with *both* axes of change — which part is active and
   * where the cursor is — and must re-render when the theme swaps. Called
   * explicitly from hover and inspect too, so the card lands with the click
   * rather than a frame later.
   */
  let dossierKey = "|none";
  function syncDossier(): void {
    const active = pinned ?? hovered;
    const theme = activePalette();
    const key =
      active === null
        ? "|none"
        : `${active}|${build.cursor}|${theme.panel}|${theme.accent}|${theme.muted}|${theme.grid}`;
    if (key === dossierKey) return;
    dossierKey = key;
    if (active === null) {
      dossierCard.style.display = "none";
      dossierCard.innerHTML = "";
      return;
    }
    renderDossier(active);
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

  function onPointerClick(): void {
    if (hovered) {
      handle.inspect(pinned === hovered ? null : hovered);
    }
  }

  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);
  renderer.domElement.addEventListener("click", onPointerClick);

  badgeOverlay.addEventListener("pointerover", (event) => {
    const target = (event.target as HTMLElement).closest("[data-part]") as HTMLElement | null;
    if (target?.dataset.part) {
      setHovered(target.dataset.part);
    }
  });

  badgeOverlay.addEventListener("pointerout", (event) => {
    const target = (event.target as HTMLElement).closest("[data-part]") as HTMLElement | null;
    if (target?.dataset.part) {
      setHovered(null);
    }
  });

  badgeOverlay.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest("[data-part]") as HTMLElement | null;
    if (target?.dataset.part) {
      handle.inspect(pinned === target.dataset.part ? null : target.dataset.part);
    }
  });

  const projVec = new Vector3();

  /**
   * Text put into the badge: the value string, and what the old floating
   * labels said in their tooltips (provenance, unit, or why there is no
   * number). Kept on the item so the badge can be rebuilt from the scene
   * alone, every frame.
   */
  interface LeaderItem {
    id: string;
    label: string;
    valueStr: string;
    title: string;
    color: string;
    sx: number;
    sy: number;
    targetY: number;
    available: boolean;
    /** Editorial grouping for the badge's second row; "" for old documents. */
    category: string;
    /** What the provenance dot says on hover, spelled for someone who has none. */
    provenanceWord: string;
    /** Set per flank by the solver: this badge dropped to single-line mode. */
    singleLine: boolean;
  }

  function updateLeaderLines(): void {
    if (!annotationsVisible || !scene) {
      svgOverlay.innerHTML = "";
      badgeOverlay.innerHTML = "";
      return;
    }

    const width = container.clientWidth || 640;
    const height = container.clientHeight || 480;

    const visibleItems: LeaderItem[] = [];

    for (const [id, object] of objects) {
      if (!object.group.visible) continue;
      const partSpec = scene.parts.find((p) => p.id === id);
      if (!partSpec || !partSpec.drawn) continue;

      projVec.set(partSpec.anchor[0], partSpec.anchor[1], partSpec.anchor[2]);
      projVec.project(camera);

      // Occluded behind camera
      if (projVec.z > 1.0) continue;

      const sx = (projVec.x * 0.5 + 0.5) * width;
      const sy = (-projVec.y * 0.5 + 0.5) * height;

      if (sx < -30 || sx > width + 30 || sy < -30 || sy > height + 30) continue;

      const color = provenanceColor(partSpec.provenance, currentSpec.theme);
      const category = partSpec.category;
      const provenanceWord = partSpec.provenance || "no provenance — drawn as architecture";
      const baseTitle = partSpec.available
        ? `${partSpec.label} — ${partSpec.provenance || "unmeasured"}${partSpec.unit ? ` (${partSpec.unit})` : ""}`
        : `${partSpec.label} — no measurement: ${partSpec.reason ?? "not measured"}`;
      visibleItems.push({
        id,
        label: partSpec.label,
        valueStr: formatValue(partSpec),
        // The tooltip always carries category + provenance, because a
        // single-line badge has no second row to put them in.
        title: [baseTitle, category, partSpec.provenance ? "" : provenanceWord]
          .filter(Boolean)
          .join(" · "),
        color,
        sx,
        sy,
        targetY: sy,
        available: partSpec.available,
        category,
        provenanceWord,
        singleLine: false,
      });
    }

    if (visibleItems.length === 0) {
      svgOverlay.innerHTML = "";
      badgeOverlay.innerHTML = "";
      return;
    }

    // Partition into left and right flanks based on projected 3D anchor X.
    // Invariant (c) of the solver: the partition happens BEFORE the call, so
    // each solver run covers exactly one column and the flanks never mix.
    const leftItems = visibleItems.filter((item) => item.sx < width * 0.5);
    const rightItems = visibleItems.filter((item) => item.sx >= width * 0.5);

    leftItems.sort((a, b) => a.sy - b.sy);
    rightItems.sort((a, b) => a.sy - b.sy);

    // The pure solver returns card TOPS in anchor order: gaps hold (no card
    // overlaps its neighbour), order follows the anchors (leader lines cannot
    // cross within a flank), and a flank too crowded for two-row cards drops
    // wholesale to single-line instead of half of it overlapping.
    const applyFlank = (items: LeaderItem[]): void => {
      if (items.length === 0) return;
      const layout = layoutFlank(items.map((item) => ({ id: item.id, sy: item.sy })), height);
      items.forEach((item, i) => {
        item.targetY = layout.targetY[i];
        item.singleLine = layout.singleLine;
      });
    };

    applyFlank(leftItems);
    applyFlank(rightItems);

    const activeId = pinned ?? hovered;

    let svgContent = "";
    let badgesContent = "";

    const renderItem = (item: LeaderItem, onLeft: boolean) => {
      const isTarget = item.id === activeId;
      const hasActive = activeId !== null;
      const opacity = isTarget ? 1.0 : hasActive ? DIM_OPACITY : 0.85;
      const strokeWidth = isTarget ? 1.75 : 1;
      const strokeDash = isTarget ? "none" : "2 2";
      const lineColor = isTarget ? currentSpec.theme.accent : item.color;

      // The line ends on the badge's *inner* border — the edge facing the cell
      // — at the card's vertical centre (the solver returns card TOPS), a fixed
      // distance from the stage edge, so the badges line up into two columns
      // whatever the parts do. The elbow is the midpoint of the anchor and that
      // terminus: one dog-leg, drawn the same way on both flanks.
      const cardH = item.singleLine ? CARD_H_ONE : CARD_H_TWO;
      const endY = item.targetY + cardH / 2;
      const badgeEdgeX = onLeft ? BADGE_MARGIN + BADGE_WIDTH : width - BADGE_MARGIN - BADGE_WIDTH;
      const elbowX = (item.sx + badgeEdgeX) / 2;

      svgContent += `
        <circle cx="${item.sx.toFixed(1)}" cy="${item.sy.toFixed(1)}" r="2.5" fill="${item.color}" opacity="${opacity}" />
        <circle cx="${item.sx.toFixed(1)}" cy="${item.sy.toFixed(1)}" r="${isTarget ? 6.5 : 5}" fill="none" stroke="${lineColor}" stroke-width="${strokeWidth}" stroke-dasharray="2 2" opacity="${opacity}" />
        <polyline points="${item.sx.toFixed(1)},${item.sy.toFixed(1)} ${elbowX.toFixed(1)},${endY.toFixed(1)} ${badgeEdgeX.toFixed(1)},${endY.toFixed(1)}" fill="none" stroke="${lineColor}" stroke-width="${strokeWidth}" stroke-dasharray="${strokeDash}" opacity="${opacity}" />
      `;
      if (isTarget) {
        // One pulsing target, not nineteen — a pulse everywhere is noise. The
        // ring breathes around the active reticle and touches nothing else.
        const pulseR = 6.5 + 1.4 * Math.sin(pulsePhase * 2 * Math.PI);
        svgContent += `
          <circle cx="${item.sx.toFixed(1)}" cy="${item.sy.toFixed(1)}" r="${pulseR.toFixed(2)}" fill="none" stroke="${currentSpec.theme.accent}" stroke-width="1" opacity="0.9"/>
        `;
      }

      // `box-sizing: border-box` is what makes `width` the *whole* badge, so
      // the border the line meets is exactly at `badgeEdgeX` above rather than
      // a padding-and-border's width past it. Height and the column layout
      // match the solver's own card constants: it reasons with CARD_H_TWO or
      // CARD_H_ONE, and the badge must be exactly that tall or its centre and
      // the line's terminus disagree.
      const badgeStyle = `
        position: absolute;
        top: ${item.targetY.toFixed(1)}px;
        height: ${cardH}px;
        ${onLeft ? `left: ${BADGE_MARGIN}px;` : `right: ${BADGE_MARGIN}px;`}
        box-sizing: border-box;
        width: ${BADGE_WIDTH}px;
        pointer-events: auto;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        justify-content: center;
        gap: 1px;
        padding: 2px 7px;
        font: 500 10.5px/1.3 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: ${isTarget ? "rgba(22, 33, 58, 0.95)" : "rgba(11, 17, 32, 0.82)"};
        border: 1px solid ${isTarget ? currentSpec.theme.accent : "rgba(51, 65, 85, 0.55)"};
        ${onLeft ? `border-left: 2.5px solid ${item.color};` : `border-right: 2.5px solid ${item.color};`}
        border-radius: 4px;
        color: #e2e8f0;
        backdrop-filter: blur(4px);
        box-shadow: ${isTarget ? "0 0 10px rgba(99, 179, 237, 0.35)" : "0 2px 5px rgba(0,0,0,0.35)"};
        opacity: ${opacity};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: opacity 0.12s ease, border-color 0.12s ease;
      `;

      badgesContent += `
        <div class="cell-scene-callout" data-part="${escapeHtml(item.id)}" title="${escapeHtml(item.title)}" style="${badgeStyle}">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;min-width:0">
            <span style="font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; min-width: 0;">${escapeHtml(item.label)}</span>
            <span style="color: ${item.available ? "#94a3b8" : "#64748b"}; font-size: 10px; flex-shrink: 0;">${escapeHtml(item.valueStr)}</span>
          </div>
          ${item.singleLine ? "" : `
          <div style="display:flex;align-items:center;gap:6px;min-width:0">
            <span style="font-size:9px;letter-spacing:0.06em;color:${item.available ? currentSpec.theme.muted : "#64748b"};overflow:hidden;text-overflow:ellipsis">${escapeHtml(item.category)}</span>
            <span title="${escapeHtml(item.provenanceWord)}" style="width:7px;height:7px;border-radius:50%;background:${item.color};flex-shrink:0;margin-left:auto"></span>
          </div>`}
        </div>
      `;
    };

    for (const item of leftItems) renderItem(item, true);
    for (const item of rightItems) renderItem(item, false);

    svgOverlay.innerHTML = svgContent;
    badgeOverlay.innerHTML = badgesContent;
  }

  // ── Resize ────────────────────────────────────────────────────────────
  function resize(): void {
    const width = container.clientWidth || 640;
    const height = container.clientHeight || 480;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    // The chrome SVG is sized in pixels, so a resize re-draws it too.
    applyStageStyle();
    updateLeaderLines();
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
  // The active reticle's breath: one phase for this engine instance, advanced
  // by frame time so the period is wall-clock (1.6 s) rather than frame-count.
  let pulsePhase = 0;
  let lastTickMs = 0;
  function tick(): void {
    if (disposed) return;
    raf = window.requestAnimationFrame(tick);
    const now = performance.now();
    const dtMs = lastTickMs === 0 ? 16 : Math.min(now - lastTickMs, 100);
    lastTickMs = now;
    pulsePhase = (pulsePhase + dtMs / 1600) % 1;
    if (camTween) {
      const t = Math.min(1, (now - camTween.start) / camTween.dur);
      applyPose(poseLerp(camTween.from, camTween.to, t));
      if (t >= 1) camTween = null;
    }
    controls.update();
    renderer.render(scene3, camera);
    updateLeaderLines();
    // The card follows the cursor (a scrub changes every live row) and the
    // selection — one string compare when nothing moved.
    syncDossier();
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
    // `unrollNote` appends, never replaces: a compressed strip's ratio rides
    // beside whatever the data-scaled view already had to say, and is null in
    // wound mode, where old behaviour stays byte-identical.
    state.scaleNote = scene.unrollNote
      ? [scene.scaleNote, scene.unrollNote].filter(Boolean).join(" · ")
      : scene.scaleNote;
    state.measuredCount = scene.timeline.measuredCount;
    state.lastMeasuredCycle = scene.timeline.lastMeasuredCycle;
    state.partCount = scene.parts.length;
    state.drawnParts = scene.parts.filter((part) => part.drawn).length;
    state.projectionLength = Math.max(0, scene.timeline.cycles.length - scene.timeline.measuredCount);
    options.onFrame?.({ ...state });
  }

  function rebuild(spec_: CellSceneSpec): void {
    const built = buildScene(spec_, build);
    paint(built);
    // Frame the cell once per *document*, not once per frame: re-fitting on a
    // scrub or an explode would throw away the orbit the viewer just made, and
    // a new cell is the one moment the old framing is meaningless.
    if (framedSpec !== spec_) {
      framedSpec = spec_;
      fitView(built.bounds.radius, built.bounds.height);
    }
    fitShadow(built.bounds.radius);
    emit();
  }

  // ── Public handle ─────────────────────────────────────────────────────
  const handle: CellSceneHandle = {
    update(spec_, opts = {}) {
      if (disposed) return;
      // A host re-render must not silently drop the reader's palette choice:
      // the spec it passes is the document's own, so re-theme it (cached per
      // source, which is also what keeps the camera's frame-once-per-document
      // guard from mistaking a view toggle for a new cell).
      currentSpec = themed(spec_);
      applyOptions(opts);
      rebuild(currentSpec);
    },
    setCursor(index) {
      if (disposed || !scene) return;
      build.cursor = index;
      rebuild(currentSpec);
    },
    inspect(partId) {
      if (disposed) return;
      pinned = partId;
      applyActiveTreatment();
      // Turn the camera to the inspected part's own side, keeping the distance
      // and height the viewer chose — a click in a list should not throw away
      // their framing — and as a tween rather than a jump, so the click turns
      // the view. Any grab of the controls cancels it mid-flight.
      if (partId && scene) {
        const part = scene.parts.find((candidate) => candidate.id === partId);
        if (part) framePose(poseFacing(part.anchor, { ...currentPose(), targetY: 0 }), 500);
      }
      options.onInspect?.(partId);
      emit();
      updateLeaderLines();
      syncDossier();
    },
    setAnnotations(visible) {
      annotationsVisible = visible;
      updateLeaderLines();
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
      // The same pose a fresh document gets, but tweened: a deliberate return
      // should glide home, not snap — the hand can watch where it went.
      const pose = homePose(lastBounds.radius, lastBounds.height);
      controls.minDistance = pose.distance * 0.5;
      controls.maxDistance = pose.distance * 3.5;
      framePose(pose, 600);
    },
    setTheme(name) {
      themeName = name || null;
      const next = themed(currentSpec);
      // A palette swap is not a new document: hold `framedSpec` so repaint
      // does not re-frame the camera and throw away the viewer's orbit.
      framedSpec = next;
      currentSpec = next;
      applyStageStyle();
      rebuild(currentSpec);
      // The badges, dossier and cards read `currentSpec`/the active palette,
      // so the next frame repaints them from the same swatch as the stage.
      updateLeaderLines();
      syncDossier();
    },
    state: () => ({ ...state }),
    dispose() {
      if (disposed) return;
      disposed = true;
      handle.pause();
      window.cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      renderer.domElement.removeEventListener("click", onPointerClick);
      observer?.disconnect();
      window.removeEventListener("resize", resize);
      for (const object of objects.values()) {
        object.mesh.geometry.dispose();
        object.material.dispose();
      }
      objects.clear();
      floorMesh?.geometry.dispose();
      shadowMesh?.geometry.dispose();
      gaugeMesh?.geometry.dispose();
      gaugeMaterial.dispose();
      controls.dispose();
      environment.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      overlayContainer.remove();
    },
  };

  // A host's options apply to the very first frame, not the second one: the
  // scene it asked for is the scene it gets.
  applyOptions(options);
  currentSpec = themed(currentSpec);
  rebuild(currentSpec);
  raf = window.requestAnimationFrame(tick);
  return handle;
}
