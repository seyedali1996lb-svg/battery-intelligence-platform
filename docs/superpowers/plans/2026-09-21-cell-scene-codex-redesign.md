# Cell Scene Codex Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved spec `docs/superpowers/specs/2026-09-21-cell-scene-codex-redesign-design.md` — 19 anatomy parts, peel + unrolled geometry, pure leader-line solver with two-row callouts, document-carried dossiers with two-tier provenance, dual Codex/Obsidian palettes, and an engine-owned cockpit HUD with hotkeys, camera presets and telemetry.

**Architecture:** One framework-free renderer (`frontend/src/scene/`), one versioned document (`docs/cell_scene.schema.json` produced by `src/cell_scene.py`), four hosts. All five systems live in the scene module (Approach A): new pure modules `annotation.ts`, `dossier.ts`, `tween.ts` plus DOM module `hud.ts`; `geometry.ts` stays DOM-free; `engine.ts` stays the only file that knows three.js. The document remains the only source of every painted colour, size and claim.

**Tech Stack:** TypeScript (strict), three.js 0.186, `node --test` (Node 24), oxlint, Vite 8; Python 3.14, pytest, jsonschema, mkdocs. Shell is PowerShell.

## Global Constraints

Every task's requirements implicitly include this section.

1. **Bijection = 19 parts** across `MESH_PART_IDS` (src/cell_scene.py) ↔ `spec.parts` ↔ schema part-id enum (`minItems`/`maxItems` 19) ↔ `ANATOMY_PART_IDS` (frontend/src/scene/types.ts) ↔ fixture LABELS/TITLES ↔ committed sample.
2. **Default peel draws exactly 285°**: `peelSweepDeg(75/360) === 285`, minimum drawn sweep 90°, `peel = 0.25` → exactly 270° drawn (a quarter of the circumference removed).
3. **Band thresholds identical across palettes**: `theme`, `palettes.codex`, `palettes.obsidian` carry the same `sohBands`/`temperatureBands` `min`/`max`/`label`; only `color` may differ.
4. **Schema is additive-optional only.** `SCENE_SCHEMA_VERSION` stays `1` in all three places (`types.ts`, `cell_scene.py`, schema `"const": 1`). New fields (`category`, `dossier`, `palettes`) are optional properties. Old documents must still validate and render.
5. **The renderer hard-codes no colour.** Every colour the engine paints comes from `spec.theme`/`spec.palettes` (existing contract in theme.ts). Applies to HUD, chrome, dossier card, backdrop, rim glow.
6. **Honesty**: spec rows carry `tag` from `{typical, measured, derived, fitted, refusal}`; live rows only from real series (`temperatureC`, `resistanceNormalized`, `sohPct`, `seiThicknessNm`, `seiSharePct`, `seiPct`, `lamPct`); missing measurements render refusal rows, never invented numbers.
7. **Freshness gate — run after EVERY task** (any change under `frontend/src/scene/` or `src/cell_scene.py` invalidates bundle/sample/manifest hashes):

```powershell
cd frontend; npm run build:scene; cd ..
python scripts/export_scene_sample.py --cell B0005
pytest tests/test_cell_scene.py tests/test_cell_scene_api.py tests/test_cell_scene_bundle.py tests/test_cell_scene_page.py
```

8. **Node gate — run after EVERY task:**

```powershell
cd frontend; npm run test:scene; npm run lint; npm run build; cd ..
```

9. **Final gate (Task 13):** everything in 7+8 plus `pyright` and `mkdocs build --strict`.
10. **Commits**: one conventional commit per task, message given in the task. Never stage `data/relog_severson.log` or `data/rerun_severson_studies.log` (transient, untracked, out of scope).
11. Never alter the two invariants' arithmetic: assembled state (`exploded = 0`) is exactly the declared winding; ribbons tile one turn's advance exactly.

## File Structure

| File | Role | Tasks |
|---|---|---|
| `frontend/src/scene/geometry.ts` | Pure geometry: peel sweep, new placements, unrolled strips, explode keys | 1,2,3 |
| `frontend/src/scene/types.ts` | `ANATOMY_PART_IDS`, `BuildOptions`-adjacent types, `ScenePart.category/dossier`, `CellSceneSpec.palettes`, `SceneTheme` presentation tokens | 2,4,7,10 |
| `frontend/src/scene/materials.ts` | `PART_MATERIALS` entries, `PALETTE_MATERIAL_ADJUST`, `materialFor(id, variant)` | 2,11 |
| `frontend/src/scene/annotation.ts` **(new)** | Pure flank solver, card model, desaturate/dim helpers, chrome SVG string | 4,6,11 |
| `frontend/src/scene/tween.ts` **(new)** | Pure `springStep`, `easeInOut`, `poseLerp`, `poseFacing` | 5 |
| `frontend/src/scene/dossier.ts` **(new)** | Pure `composeDossier(spec, partId, cursor)` two-tier composer | 8 |
| `frontend/src/scene/hud.ts` **(new, DOM)** | Control rail, playback bar, presets, hotkeys, telemetry, theme switch, backdrop, prefs | 12 |
| `frontend/src/scene/theme.ts` | `paletteFor(spec, name)` | 10,11 |
| `frontend/src/scene/engine.ts` | Springs, camera tweens, selection treatment, setTheme, annotation/hud/dossier wiring | 4,6,9,11,12 |
| `frontend/src/scene/index.ts` | Public exports (`composeDossier`, `paletteFor`, `peelSweepDeg`, handle methods) | 1,8,11,12 |
| `frontend/src/scene/fixture.ts` | `LABELS`/`TITLES` for new ids, categories | 2,4 |
| `frontend/src/scene/*.test.ts` | Node tests (`geometry.test.ts`, new `annotation.test.ts`, `tween.test.ts`, `dossier.test.ts`) | all |
| `docs/cell_scene.schema.json` | enum/counts, `category`, `dossier`, `palettes` | 2,4,7,10 |
| `src/cell_scene.py` | `MESH_PART_IDS`, `_PLAIN_TITLE`, cards, `_CATEGORY`, `_DOSSIER_TABLE`, `_editorial`, `_default_palettes()` | 2,4,7,10 |
| `frontend/src/components/CellSceneView.tsx` | Delete `PART_DOSSIERS`; render from document | 9 |
| `frontend/index.html`, `app/static/cell_scene/index.html`, `app/_scene_view.py` | Font `<link>` | 11 |
| `README.md`, `CHANGELOG.md`, `METHODOLOGY.md`, `docs/history.md`, `docs/scene_prompt.md`, `app/static/cell_scene/README.md`, `app/_pages/battery3d.py` | Docs to shipped state | 13 |

---

## PHASE 1 — Geometry: peel, 19 parts, unrolled

### Task 1: `peel` build option (cut-away as a view control)

**Files:**
- Modify: `frontend/src/scene/geometry.ts` (`CELL_GEOMETRY` ~line 420, `BuildOptions` ~line 984, `mergeBuildOptions` ~line 1015, `_cylindricalPlacements` ~line 1217, call site ~line 1578)
- Modify: `frontend/src/scene/index.ts` (export)
- Test: `frontend/src/scene/geometry.test.ts`

**Interfaces:**
- Produces: `DEFAULT_PEEL: number` (= `75/360`), `peelSweepDeg(peel: number): number` (degrees), `BuildOptions.peel: number`, `_cylindricalPlacements(..., sweepRad: number)`.

- [ ] **Step 1: Write the failing tests**

Append to `geometry.test.ts` (it already imports `buildScene`, `makeSpec` and the geometry exports — match the file's existing import block):

```ts
test("the default peel draws exactly the cut-away every previous build drew", () => {
  assert.equal(peelSweepDeg(DEFAULT_PEEL), 285);
});

test("peel 0 closes the can; a quarter peel removes exactly a quarter of the ring", () => {
  assert.equal(peelSweepDeg(0), 360);
  assert.equal(peelSweepDeg(0.25), 270);
  assert.equal(peelSweepDeg(1), 90); // a quarter shell always remains
  assert.equal(peelSweepDeg(Number.NaN), 285); // a garbage value falls back, never NaN
});

test("peel opens the shell and changes nothing inside it", () => {
  const closed = buildScene(makeSpec(), { peel: 0 });
  const open = buildScene(makeSpec(), { peel: 0.25 });
  const inner = (s: Awaited<ReturnType<typeof buildScene>>) =>
    ["cathode_sheet", "anode_sheet", "separator", "mandrel"].map((id) =>
      Array.from(s.parts.find((p) => p.id === id)!.mesh.positions),
    );
  assert.deepEqual(inner(open), inner(closed));
  const canOf = (s: Awaited<ReturnType<typeof buildScene>>) =>
    s.parts.find((p) => p.id === "can")!.mesh;
  assert.ok(vertexCount(canOf(open)) < vertexCount(canOf(closed)), "a smaller sweep draws fewer vertices");
});

test("the wrap opens with the can, because the jacket is skin of the casing", () => {
  const closed = buildScene(makeSpec(), { peel: 0 });
  const open = buildScene(makeSpec(), { peel: 0.25 });
  const wrapV = (s: Awaited<ReturnType<typeof buildScene>>) =>
    vertexCount(s.parts.find((p) => p.id === "wrap")!.mesh);
  assert.ok(wrapV(open) < wrapV(closed));
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm run test:scene`
Expected: FAIL — `peelSweepDeg`/`DEFAULT_PEEL` not exported (or not defined).

- [ ] **Step 3: Implement**

In `geometry.ts`, next to `CELL_GEOMETRY`:

```ts
/** Peel at which the casing is closed. 75/360 reproduces the historic 285° cut. */
export const DEFAULT_PEEL = 75 / 360;

/**
 * Degrees of casing circumference DRAWN at a peel position (view control, never
 * data): peel 0 closes the can, peel 0.25 removes exactly a quarter of the ring
 * ("¼ Peel"), and a quarter shell (90°) always remains — removing the casing
 * entirely remains `casing: "hidden"`'s job. NaN falls back to the default.
 */
export function peelSweepDeg(peel: number): number {
  const p = Number.isFinite(peel) ? Math.min(1, Math.max(0, peel)) : DEFAULT_PEEL;
  return Math.max(90, 360 * (1 - p));
}
```

`BuildOptions` and defaults:

```ts
export interface BuildOptions {
  cursor: number;
  exploded: number;
  casing: "translucent" | "hidden";
  dataScaled: boolean;
  /** 0 = closed can, 0.25 = a quarter of the ring removed, 1 = quarter shell. */
  peel: number;
}

export const DEFAULT_BUILD_OPTIONS: BuildOptions = {
  cursor: 0,
  exploded: 0,
  casing: "translucent",
  dataScaled: false,
  peel: DEFAULT_PEEL,
};
```

`mergeBuildOptions` — add one line inside the returned object (the function's comment says adding a field forces this function to be revisited):

```ts
    peel: patch.peel ?? build.peel,
```

Delete `canSweep: (285 * Math.PI) / 180,` from `CELL_GEOMETRY` (keep its comment updated to point at `peelSweepDeg`). In `_cylindricalPlacements`, add parameter `sweepRad: number` after `drawn`, change the destructure to `const { roll, explode } = CELL_GEOMETRY;`, and replace both `canSweep` uses (casing line ~1231, wrap line ~1315) with `sweepRad`. In `buildScene`, compute once and pass:

```ts
  const sweepRad = (peelSweepDeg(opts.peel) * Math.PI) / 180;
  const placements =
    spec.cell.formFactor === "prismatic"
      ? _prismaticPlacements(exploded, filmThickness, prism, topAssemblyModel(spec))
      : _cylindricalPlacements(exploded, filmThickness, spec, model, drawn, sweepRad);
```

(`_prismaticPlacements` has no casing arc — it takes no sweep.)

In `index.ts` add `peelSweepDeg` and `DEFAULT_PEEL` to the geometry export line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend; npm run test:scene`
Expected: PASS (all, including the four new ones).

- [ ] **Step 5: Freshness gate + commit**

Run: Global Constraint 7 commands (all four suites PASS; sample/manifest re-written).

```powershell
git add frontend/src/scene/geometry.ts frontend/src/scene/index.ts frontend/src/scene/geometry.test.ts app/static/cell_scene/
git commit -m "feat(scene): make the cut-away a peel view control, defaulting to the exact historic 285 degrees"
```

---

### Task 2: Three new anatomy parts → 19, end-to-end

**Files:**
- Modify: `frontend/src/scene/types.ts` (`ANATOMY_PART_IDS` line 19)
- Modify: `frontend/src/scene/geometry.ts` (`CELL_GEOMETRY.explode` ~line 444, `_cylindricalPlacements` return ~line 1297, `_prismaticPlacements` ~line 1433, colour switch ~line 1633)
- Modify: `frontend/src/scene/materials.ts`, `frontend/src/scene/fixture.ts` (`LABELS`, `TITLES` tables)
- Modify: `docs/cell_scene.schema.json` (line 536-537 counts, line 556-573 enum)
- Modify: `src/cell_scene.py` (`MESH_PART_IDS` line 87, `_PLAIN_TITLE` line 514, `_build_parts` end ~line 820)
- Modify: `frontend/src/scene/geometry.test.ts`, possibly `frontend/src/scene/spec.test.ts` (literal `16`/`sixteen` counts)
- Test: `tests/test_cell_scene.py` (bijection/enum tests are data-driven — verify)

**Interfaces:**
- Produces: part ids `"gasket" | "cid_ptc" | "bottom_insulator"`; `CELL_GEOMETRY.explode = { ..., gasket: 0.26, cidPtc: 0.34, bottomInsulator: 0.22 }` (cell units, all multiplied by `exploded`, so zero at rest); placements for both form factors; labels/titles in fixture and Python.

- [ ] **Step 1: Write the failing tests**

Append to `geometry.test.ts`:

```ts
test("the scene draws all nineteen parts and every part has sane geometry", () => {
  const built = buildScene(makeSpec());
  assert.equal(built.parts.length, 19);
  for (const part of built.parts) {
    assert.ok(part.drawn, `${part.id} should be drawn in a cylindrical scene`);
    assert.ok(Number.isFinite(part.mesh.positions[0]), `${part.id} has no geometry`);
  }
});

test("the three new internals stay inside the bore at every explode position", () => {
  const spec = makeSpec();
  for (const t of [0, 0.5, 1]) {
    const built = buildScene(spec, { exploded: t });
    const bore = built.roll.envelopeRadiusMm; // radial envelope the document declares
    for (const id of ["gasket", "cid_ptc", "bottom_insulator"]) {
      const part = built.parts.find((p) => p.id === id)!;
      const r = Math.hypot(part.mesh.positions[0], part.mesh.positions[2]);
      assert.ok(Number.isFinite(r) && r <= bore + 1e-6, `${id} leaves the envelope at explode ${t}`);
    }
  }
});

test("the top stack's axial explode keys are declared numbers", () => {
  const e = CELL_GEOMETRY.explode as unknown as Record<string, unknown>;
  for (const key of ["gasket", "cidPtc", "bottomInsulator", "cap", "vent"]) {
    assert.ok(typeof e[key] === "number", `explode.${key} missing`);
  }
});
```

Update the existing test at `geometry.test.ts:300` — change `16` → `19` and the word "sixteen" → "nineteen" (keep the test's other assertions). Grep both test files and `spec.test.ts` for `16`/`sixteen` part-count literals and update every count assertion (leave prose that talks about history alone).

In Python there is nothing to author: `test_there_is_exactly_one_card_per_anatomy_mesh` and `test_the_schema_enum_lists_exactly_the_mesh_parts_this_builder_emits` are data-driven and will start failing the moment the two sides disagree.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm run test:scene`
Expected: FAIL — `parts.length` is 16, new ids unknown.

- [ ] **Step 3: Implement — renderer side**

`types.ts`, append at the end of `ANATOMY_PART_IDS` (after `"sei_film"`):

```ts
  "gasket",
  "cid_ptc",
  "bottom_insulator",
```

`geometry.ts` — inside `CELL_GEOMETRY.explode`, add:

```ts
    /** Axial lifts at `exploded = 1` for the three interior parts; zero at rest. */
    gasket: 0.26,
    cidPtc: 0.34,
    bottomInsulator: 0.22,
```

In `_cylindricalPlacements`, add three entries to the returned object (after `terminal_neg`, using variables that already exist in scope — `canRadius`, `top`, `topCapY`, `explode`):

```ts
    gasket: {
      // The cap's seal: a flat ring tucked under the cap plate at rest, lifting
      // slightly less than the cap so the two visibly separate as they rise.
      mesh: translateMesh(
        extrudeClosed(ringPoints(canRadius * 0.96, 40), 0.007),
        0,
        0.5 - top.capThickness - 0.004 + explode.gasket * exploded,
        0,
      ),
      anchor: [canRadius * 0.7, 0.5 - top.capThickness + explode.gasket * exploded, 0],
    },
    cid_ptc: {
      // The CID scored disc and its PTC bead, in the boss cavity between the
      // cap plate and the vent — grouped as one part, as the format groups them.
      mesh: translateMesh(
        mergeMeshes([
          extrudeClosed(ringPoints(top.ventRadius * 0.75, 24), 0.006),
          translateMesh(sphereMesh(0.011, 8, 6), top.ventRadius * 0.4, 0, 0),
        ]),
        0,
        0.5 + top.capBossHeight * 0.35 + explode.cidPtc * exploded,
        0,
      ),
      anchor: [-canRadius * 0.55, 0.5 + top.capBossHeight * 0.35 + explode.cidPtc * exploded, 0],
    },
    bottom_insulator: {
      // The floor's insulating disc, lying between the can base and the roll.
      mesh: translateMesh(
        extrudeClosed(ringPoints(canRadius - canThickness, 48), 0.008),
        0,
        -0.484 - explode.bottomInsulator * exploded,
        0,
      ),
      anchor: [(canRadius - canThickness) * 0.6, -0.484 - explode.bottomInsulator * exploded, 0],
    },
```

In `_prismaticPlacements` (flat stack; variables `width`, `depth`, `halfWidth`, `explode` are in scope), add — and deliberately do NOT add `cid_ptc`, so `placement?.mesh ?? emptyMesh()` leaves it `drawn: false` with the card's honest reason:

```ts
    gasket: {
      mesh: translateMesh(boxMesh(width * 0.92, 0.008, depth * 0.92), 0, 0.485 + explode.gasket * exploded, 0),
      anchor: [halfWidth, 0.485 + explode.gasket * exploded, 0],
    },
    bottom_insulator: {
      mesh: translateMesh(boxMesh(width * 0.9, 0.01, depth * 0.9), 0, -0.48 - explode.bottomInsulator * exploded, 0),
      anchor: [-halfWidth, -0.48 - explode.bottomInsulator * exploded, 0],
    },
```

Colour switch in `buildScene` (add before `case "electrolyte"` or anywhere in the switch):

```ts
      case "gasket":
      case "bottom_insulator":
        // Polymer parts: shaded with the separator's own colour so every
        // plastic in the cell reads as plastic at a glance.
        color = spec.theme.separatorColor;
        opacity = 0.9;
        break;
      case "cid_ptc":
        color = spec.theme.metal;
        emissive = 0.15;
        opacity = 0.95;
        break;
```

`materials.ts` — add to `PART_MATERIALS`:

```ts
  /** Polypropylene sealing ring under the cap. */
  gasket: { metalness: 0.0, roughness: 0.6, envMapIntensity: 0.35 },
  /** The CID + PTC unit: passivated steel with a PTC bead. */
  cid_ptc: { metalness: 0.75, roughness: 0.45, envMapIntensity: 0.8 },
  /** Heat-treated polyethylene floor disc. */
  bottom_insulator: { metalness: 0.0, roughness: 0.65, envMapIntensity: 0.3 },
```

`fixture.ts` — add to `LABELS` and `TITLES` tables (they are keyed by `PartId`; match the tables' existing style):

```ts
  gasket: "Gasket",
  cid_ptc: "CID + PTC",
  bottom_insulator: "Bottom insulator",
```

```ts
  gasket: "The gasket — the seal that keeps the top leak-tight",
  cid_ptc: "CID + PTC — the two resettable fuses",
  bottom_insulator: "Bottom insulator — the floor's plastic disc",
```

- [ ] **Step 4: Implement — schema + producer**

`docs/cell_scene.schema.json`: `"minItems": 16` → `19`, `"maxItems": 16` → `19`; append to the `id` enum after `"crimp"`:

```json
            "gasket",
            "cid_ptc",
            "bottom_insulator"
```

`src/cell_scene.py` — `MESH_PART_IDS` gains `..., "gasket", "cid_ptc", "bottom_insulator"` (append inside the tuple). `_PLAIN_TITLE` gains:

```python
    "gasket": "The gasket — the seal that keeps the top leak-tight",
    "cid_ptc": "CID + PTC — the two resettable fuses",
    "bottom_insulator": "Bottom insulator — the floor's plastic disc",
```

At the end of `_build_parts`, immediately before its `return parts`, append three honest architecture cards (no per-part measurement exists — the crimp/wrap precedent: drawn, explained, never zeroed):

```python
    # Architecture without a claim: these three carry no per-part measurement in
    # any cycle summary, so they are reported unavailable WITH a reason and the
    # renderer still draws them — see geometry.test "architecture without a claim".
    for _pid, _label, _meaning in (
        ("gasket", "Gasket",
         "The cap's seal: an insulating ring the crimp compresses to keep the cell leak-tight."),
        ("cid_ptc", "CID + PTC",
         "The two resettable fuses: the CID opens on internal pressure, the PTC limits current on heat."),
        ("bottom_insulator", "Bottom insulator",
         "The floor's plastic disc: it keeps the winding's copper edge off the steel base."),
    ):
        parts.append(_unavailable(
            _pid, _label,
            "no per-part measurement exists in this source's cycle summary — drawn as architecture",
            _meaning,
        ))
```

- [ ] **Step 5: Run renderer tests**

Run: `cd frontend; npm run test:scene`
Expected: PASS. If any remaining literal count assertion fails, update the count to 19 (counts only — never weaken an invariant).

- [ ] **Step 6: Freshness gate + commit**

Run: Global Constraint 7 commands. All four Python suites PASS (bijection/enum tests now agree at 19; sample regenerated with 19 parts).

```powershell
git add frontend/src/scene/ src/cell_scene.py docs/cell_scene.schema.json app/static/cell_scene/
git commit -m "feat(scene): add gasket, CID+PTC and bottom insulator as the 17th-19th anatomy parts"
```

---

### Task 3: `layout: "unrolled"` — flat electrode ribbon view

**Files:**
- Modify: `frontend/src/scene/geometry.ts` (`BuildOptions`, `_cylindricalPlacements`, `BuiltScene` ~line 1110, `buildScene` return)
- Modify: `frontend/src/scene/engine.ts` (compose `scaleNote` — one line in `emit()`)
- Test: `frontend/src/scene/geometry.test.ts`

**Interfaces:**
- Consumes: `peelSweepDeg`/`DEFAULT_PEEL` (Task 1), 19 parts (Task 2).
- Produces: `BuildOptions.layout: "wound" | "unrolled"` (default `"wound"`); `UNROLL_LENGTH: number` (= 1.5 cell units); `BuiltScene.unrollNote: string | null`; `mergeBuildOptions` folds `layout`.

- [ ] **Step 1: Write the failing tests**

```ts
import { UNROLL_LENGTH, DEFAULT_BUILD_OPTIONS } from "./geometry.ts"; // extend the existing import

test("unrolled draws the three ribbons as one flat sandwich, not a spiral", () => {
  const wound = buildScene(makeSpec());
  const flat = buildScene(makeSpec(), { layout: "unrolled" });
  const strip = (s: typeof flat, id: string) => s.parts.find((p) => p.id === id)!;
  for (const id of ["anode_sheet", "separator", "cathode_sheet"]) {
    assert.ok(vertexCount(strip(flat, id).mesh) < vertexCount(strip(wound, id).mesh),
      `${id}: a straight strip has fewer vertices than a 38-turn spiral`);
  }
  // The three strips are parallel slabs: same x-extent, different z bands.
  const extent = (s: typeof flat, id: string) => {
    const pos = strip(s, id).mesh.positions;
    let minX = Infinity, maxX = -Infinity;
    for (let i = 0; i < pos.length; i += 3) { minX = Math.min(minX, pos[i]); maxX = Math.max(maxX, pos[i]); }
    return maxX - minX;
  };
  assert.ok(Math.abs(extent(flat, "anode_sheet") - extent(flat, "cathode_sheet")) < 1e-6);
  assert.ok(extent(flat, "anode_sheet") <= UNROLL_LENGTH + 1e-6);
});

test("the unrolled view admits its compression instead of implying scale", () => {
  const flat = buildScene(makeSpec(), { layout: "unrolled" });
  assert.ok(flat.unrollNote !== null, "unrolled build must print its compression");
  assert.match(flat.unrollNote!, /1:\d+/, "the note carries a ratio");
  const wound = buildScene(makeSpec());
  assert.equal(wound.unrollNote, null);
});

test("unrolling is a view control: assembled wound geometry is untouched by the option's existence", () => {
  assert.equal(DEFAULT_BUILD_OPTIONS.layout, "wound");
  const a = buildScene(makeSpec());
  const b = buildScene(makeSpec(), { layout: "wound" });
  assert.deepEqual(Array.from(a.parts[0].mesh.positions), Array.from(b.parts[0].mesh.positions));
});

test("tabs stay attached to the strip ends when the roll is unrolled", () => {
  const flat = buildScene(makeSpec(), { layout: "unrolled" });
  const tab = flat.parts.find((p) => p.id === "tab_pos")!;
  const pos = tab.mesh.positions;
  let maxX = -Infinity;
  for (let i = 0; i < pos.length; i += 3) maxX = Math.max(maxX, pos[i]);
  assert.ok(maxX > UNROLL_LENGTH * 0.3, "the positive tab rises from the strip's far end");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm run test:scene`
Expected: FAIL — `layout` unknown, `UNROLL_LENGTH` not exported.

- [ ] **Step 3: Implement**

`BuildOptions` gains `layout: "wound" | "unrolled";` with default `layout: "wound"`; `mergeBuildOptions` gains `layout: patch.layout ?? build.layout,`.

```ts
/** Drawn length of the unrolled strip, in cell units. 1.5 ≈ a third of the cell's height. */
export const UNROLL_LENGTH = 1.5;
```

In `_cylindricalPlacements`, thread a boolean through (add parameter `unrolled: boolean` after `sweepRad`; pass `opts.layout === "unrolled"` at the call site). Replace the `ribbon` helper's body:

```ts
  const UNROLL_ORDER = ["anode_sheet", "separator", "cathode_sheet"] as const;
  const ribbon = (id: DrawnRibbon["id"], height: number): Mesh => {
    const lane = drawn.ribbons.find((candidate) => candidate.id === id) as DrawnRibbon;
    if (unrolled) {
      // The flat sandwich: each ribbon as a straight slab at its DECLARED drawn
      // thickness × the stack gain, stacked in winding order with a hairline of
      // air between them — the same thicknesses the spiral drew, laid out.
      const idx = UNROLL_ORDER.indexOf(id);
      const thickness = lane.thickness * EXPLODE_STACK_GAIN;
      const z = (idx - 1) * (thickness + 0.004);
      return translateMesh(boxMesh(UNROLL_LENGTH, height, thickness), 0, 0, z);
    }
    return _rollRibbon(
      {
        innerRadius: drawn.base + lane.offset,
        turns: drawn.spiralTurns,
        pitch: drawn.advance,
        thickness: lane.thickness,
      },
      height,
    );
  };
```

Strip anchors (so leader lines land on the slabs): when `unrolled`, override the three ribbon anchors and both tab anchors in the returned object. Add before `return {`:

```ts
  const stripZ = (id: DrawnRibbon["id"]): number => {
    const idx = UNROLL_ORDER.indexOf(id);
    const lane = drawn.ribbons.find((c) => c.id === id) as DrawnRibbon;
    const thickness = lane.thickness * EXPLODE_STACK_GAIN;
    return (idx - 1) * (thickness + 0.004);
  };
```

Ribbon anchor entries become:

```ts
    anode_sheet: {
      mesh: ribbon("anode_sheet", roll.height),
      anchor: unrolled
        ? [-UNROLL_LENGTH * 0.3, roll.height * 0.15, stripZ("anode_sheet")]
        : [anodeR * Math.cos(THETA_ANODE), roll.height * 0.15, anodeR * Math.sin(THETA_ANODE)],
    },
```

(and the same `unrolled ? ... : ...` shape for `separator` at `x = 0` and `cathode_sheet` at `x = UNROLL_LENGTH * 0.3`, keeping their existing wound expressions in the false branch). Tabs: when `unrolled`, replace each `tabStripMesh(...)` mesh with a flat riser at the strip's far end:

```ts
    tab_pos: {
      mesh: unrolled
        ? translateMesh(boxMesh(0.05, 0.22, 0.008), UNROLL_LENGTH * 0.5, roll.height * 0.6, stripZ("cathode_sheet"))
        : tabStripMesh(rollTopRadius, roll.height / 2, tabTopPos, canRadius, THETA_POS, top.capThickness, -1),
      anchor: [UNROLL_LENGTH * 0.5, roll.height * 0.75, stripZ("cathode_sheet")],
    },
    tab_neg: {
      mesh: unrolled
        ? translateMesh(boxMesh(0.05, 0.22, 0.008), -UNROLL_LENGTH * 0.5, -roll.height * 0.6, stripZ("anode_sheet"))
        : tabStripMesh(rollTopRadius, -roll.height / 2, tabTopNeg, canRadius, THETA_NEG, top.capThickness, 1),
      anchor: [-UNROLL_LENGTH * 0.5, -roll.height * 0.75, stripZ("anode_sheet")],
    },
```

(`sei_film`/`particles`/`electrolyte` keep their wound geometry in unrolled mode — they are described by the strip, not laid out; this is a view control, not a second anatomy.)

Compression note — `BuiltScene` gains field `unrollNote: string | null;`. In `buildScene`'s return (near `scaleNote`):

```ts
    unrollNote:
      opts.layout === "unrolled"
        ? (() => {
            const cellUnits = (model.electrodeLengthM * 1000) / model.unitMm;
            const ratio = Math.max(2, Math.round(cellUnits / UNROLL_LENGTH));
            return `Electrode drawn 1:${ratio} along its length — ${model.electrodeLengthM.toFixed(2)} m compressed to a ${UNROLL_LENGTH} cell-unit strip`;
          })()
        : null,
```

In `engine.ts`'s `emit()`, wherever `state.scaleNote` is assigned, compose:

```ts
    scaleNote: built.unrollNote
      ? [built.scaleNote, built.unrollNote].filter(Boolean).join(" · ") || built.unrollNote
      : built.scaleNote,
```

(Adjust to the exact existing assignment shape — the rule: `unrollNote` appends, never replaces, and is `null` in wound mode so old behaviour is byte-identical.)

`index.ts`: export `UNROLL_LENGTH`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend; npm run test:scene`
Expected: PASS.

- [ ] **Step 5: Freshness gate + commit**

```powershell
git add frontend/src/scene/ app/static/cell_scene/
git commit -m "feat(scene): unroll the ribbons into a flat sandwich that admits its own compression"
```

**Phase 1 ends here:** Global Constraints 7 + 8 all green → mark the phase checkpoint in the spec §10.

---

## PHASE 2 — Annotation: solver, cards, motion

### Task 4: `annotation.ts` flank solver + two-row cards + `category`

**Files:**
- Create: `frontend/src/scene/annotation.ts`, `frontend/src/scene/annotation.test.ts`
- Modify: `frontend/src/scene/engine.ts` (`layoutFlank` ~line 588, badge HTML ~line 662, `LeaderItem` type near line 531)
- Modify: `frontend/src/scene/types.ts` (`ScenePart.category?: string`)
- Modify: `docs/cell_scene.schema.json` (part item properties), `src/cell_scene.py` (`_CATEGORY`, `_editorial`), `frontend/src/scene/fixture.ts` (categories)

**Interfaces:**
- Produces:

```ts
// annotation.ts (pure — no DOM, no three.js)
export interface FlankItem { id: string; sy: number; }
export interface FlankLayout { targetY: number[]; singleLine: boolean; minGap: number; }
export function layoutFlank(items: FlankItem[], stageHeight: number): FlankLayout;
export const CARD_H_TWO = 40, CARD_H_ONE = 20, GAP_TWO = 46, GAP_ONE = 26, CARD_MARGIN = 12;
export function desaturateHex(hex: string, amount: number, neutralHex: string): string; // amount 0..1
export function chromeSvg(opts: { theme: SceneTheme; ornate: boolean; width: number; height: number }): string;
export const CATEGORY_TAXONOMY = ["SHELL","INSULATION","SEAL","SAFETY","TERMINAL","WINDING","ELECTRODE","SEPARATOR","ELECTROLYTE","DEGRADATION"] as const;
export type Category = (typeof CATEGORY_TAXONOMY)[number];
```

- Consumes: `hexToRgb` from `theme.ts` (export it if not already public inside the module — `theme.ts` exports `hexToRgb` already).

- [ ] **Step 1: Write the failing tests** (`annotation.test.ts`, uses `node:test` + `node:assert/strict` like the other scene tests)

```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { layoutFlank, desaturateHex, chromeSvg, CATEGORY_TAXONOMY } from "./annotation.ts";
import { makeSpec } from "./fixture.ts";

const stageH = 560;
const items = (ys: number[]) => ys.map((sy, i) => ({ id: `p${i}`, sy }));

test("cards on one flank never overlap vertically", () => {
  const ys = [30, 34, 40, 200, 205, 500, 505, 540];
  const { targetY, minGap } = layoutFlank(items(ys), stageH);
  for (let i = 1; i < targetY.length; i++) {
    assert.ok(targetY[i] - targetY[i - 1] >= minGap - 1e-9, `gap ${i} collapsed`);
  }
});

test("target order follows anchor order — leader lines cannot cross within a flank", () => {
  const ys = [500, 100, 300, 20];
  const sorted = [...ys].sort((a, b) => a - b);
  const { targetY } = layoutFlank(items(ys), stageH);
  const order = ys.map((sy, i) => ({ sy, ty: targetY[i] })).sort((a, b) => a.sy - b.sy).map((p) => p.ty);
  for (let i = 1; i < order.length; i++) assert.ok(order[i] >= order[i - 1]);
});

test("cards stay inside the stage no matter how the anchors pile up", () => {
  const ys = Array.from({ length: 10 }, (_v, i) => 40 + i); // ten anchors in 10px
  const { targetY, minGap } = layoutFlank(items(ys), stageH);
  const last = targetY[targetY.length - 1];
  assert.ok(last + 60 <= stageH, "two-row cards need their own height below the last card");
  assert.ok(targetY[0] >= 0);
});

test("a flank too crowded for two rows degrades to single-line cards", () => {
  const ys = Array.from({ length: 14 }, (_v, i) => 30 + i * 40);
  const twoRow = layoutFlank(items(ys.slice(0, 6)), stageH);
  assert.equal(twoRow.singleLine, false);
  const crowded = layoutFlank(items(ys), stageH);
  assert.equal(crowded.singleLine, true);
  for (let i = 1; i < crowded.targetY.length; i++) {
    assert.ok(crowded.targetY[i] - crowded.targetY[i - 1] >= crowded.minGap - 1e-9);
  }
});

test("desaturating mixes toward the palette's own neutral, and returns a hex", () => {
  const grey = desaturateHex("#ff0000", 1, "#808080");
  assert.match(grey, /^#[0-9a-f]{6}$/);
  assert.equal(desaturateHex("#ff0000", 0, "#808080"), "#ff0000");
});

test("chrome draws only from the theme it was handed", () => {
  const spec = makeSpec();
  const svg = chromeSvg({ theme: spec.theme, ornate: true, width: 800, height: 600 });
  assert.match(svg, /<svg/);
  assert.ok(!/#fff|white|black/i.test(svg.replace(spec.theme.text, "").replace(spec.theme.muted, "")),
    "no colour literal beyond what the theme itself spells");
});

test("the category taxonomy is exactly the ten words the spec declares", () => {
  assert.equal(CATEGORY_TAXONOMY.length, 10);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm run test:scene` → FAIL, module missing.

- [ ] **Step 3: Implement `annotation.ts`**

```ts
import type { SceneTheme } from "./types.ts";
import { hexToRgb } from "./theme.ts";

export const CATEGORY_TAXONOMY = [
  "SHELL", "INSULATION", "SEAL", "SAFETY", "TERMINAL",
  "WINDING", "ELECTRODE", "SEPARATOR", "ELECTROLYTE", "DEGRADATION",
] as const;
export type Category = (typeof CATEGORY_TAXONOMY)[number];

export const CARD_H_TWO = 40, CARD_H_ONE = 20, GAP_TWO = 46, GAP_ONE = 26, CARD_MARGIN = 12;

export interface FlankItem { id: string; sy: number; }
export interface FlankLayout { targetY: number[]; singleLine: boolean; minGap: number; }

/**
 * Where one flank's callout cards sit, vertically.
 *
 * Contract (all four asserted in annotation.test.ts):
 *  (a) no two cards overlap vertically — consecutive targets differ by minGap;
 *  (b) targetY is monotone in the anchor's sy — two lines between two ordered
 *      pairs cannot cross, so the flank's leader lines never cross;
 *  (c) callers partition items left/right by anchor x, so flanks never mix;
 *  (d) when the flank is too short for its cards, EVERYTHING drops to
 *      single-line mode instead of overlapping.
 */
export function layoutFlank(items: FlankItem[], stageHeight: number): FlankLayout {
  const sorted = [...items].sort((a, b) => a.sy - b.sy);
  const n = sorted.length;
  if (n === 0) return { targetY: [], singleLine: false, minGap: GAP_TWO };

  const space = stageHeight - 2 * CARD_MARGIN;
  let singleLine = n * GAP_TWO > space;
  const minGap = singleLine ? GAP_ONE : GAP_TWO;
  const cardH = singleLine ? CARD_H_ONE : CARD_H_TWO;

  // Forward pass: honour the gap, pulling each card as close to its anchor as
  // the card above allows.
  const targetY: number[] = [];
  let cursor = Math.max(CARD_MARGIN, Math.min(sorted[0].sy - cardH / 2, stageHeight - CARD_MARGIN - cardH));
  for (let i = 0; i < n; i++) {
    const want = Math.max(sorted[i].sy - cardH / 2, i === 0 ? CARD_MARGIN : targetY[i - 1] + minGap);
    cursor = Math.max(want, i === 0 ? CARD_MARGIN : targetY[i - 1] + minGap);
    targetY.push(Math.min(cursor, stageHeight - CARD_MARGIN - cardH));
  }
  // Backward pass: the stack must also fit — pull the tail up while keeping
  // every gap (order is preserved because we only ever decrease a value down to
  // its predecessor's target + minGap).
  for (let i = n - 1; i >= 0; i--) {
    const ceiling = (i === n - 1 ? stageHeight - CARD_MARGIN - cardH : targetY[i + 1] - minGap);
    targetY[i] = Math.min(targetY[i], ceiling);
    if (i > 0) targetY[i] = Math.max(targetY[i], targetY[i - 1] + minGap);
  }
  // If even single-line cannot fit (pathologically short stage), clamp order
  // back into monotone shape by re-spacing from the top.
  if (targetY[n - 1] > stageHeight - CARD_MARGIN) {
    const step = Math.max(1, (stageHeight - 2 * CARD_MARGIN) / Math.max(1, n - 1));
    for (let i = 0; i < n; i++) targetY[i] = Math.min(CARD_MARGIN + i * step, stageHeight - cardH);
    singleLine = true;
  }
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
```

`chromeSvg` — the drafting frame, entirely from tokens (this is the visual chrome; `ornate` selects codex's rose vs obsidian's reticle):

```ts
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
```

- [ ] **Step 4: Run tests, then wire the engine**

Run: `cd frontend; npm run test:scene` → new tests PASS (fix solver math until all four invariants hold; the backward/forward pass above is the reference algorithm).

Engine wiring, in `updateLeaderLines()`:
1. Replace the local `layoutFlank` (line ~588) with the imported pure one: build `FlankItem[]` from the already-filtered flank items (`{ id, sy }`), call `layoutFlank(items, stage.clientHeight)`, read `targetY`/`singleLine` back into the existing drawing loop. Keep `BADGE_MARGIN = 12`, `BADGE_WIDTH = 158` and the existing flank partition by anchor `x` (invariant (c) — comment it).
2. Badge HTML (line ~662): two rows. Row 1 keeps `label` (font-weight 600) + `valueStr` (tabular). Row 2 (omit entirely when `singleLine`):

```ts
        <div style="display:flex;align-items:center;gap:6px;min-width:0">
          <span style="font-size:9px;letter-spacing:0.06em;color:${item.available ? theme.muted : "#64748b"};overflow:hidden;text-overflow:ellipsis">${escapeHtml(item.category || "")}</span>
          <span title="${escapeHtml(item.provenanceWord)}" style="width:7px;height:7px;border-radius:50%;background:${item.provenanceColor};flex-shrink:0;margin-left:auto"></span>
        </div>
```

`LeaderItem` gains `category: string`, `provenanceColor: string`, `provenanceWord: string`, populated from `spec.parts` (`part.category ?? ""`, `theme.provenanceColors[part.provenance ?? ""] ?? theme.provenanceColors[""] ?? "#718096"`, provenance word = `part.provenance || "no provenance — drawn as architecture"`). When `singleLine`, the whole row 2 is skipped and the `title` attribute on the badge must contain category + provenance so hover still exposes them.

- [ ] **Step 5: `category` across the contract**

`types.ts`: `ScenePart` gains `/** Editorial grouping shown on the callout's second row. */ category?: string;`

Schema — in `parts.items.properties` add:

```json
        "category": {
          "type": "string",
          "enum": ["SHELL","INSULATION","SEAL","SAFETY","TERMINAL","WINDING","ELECTRODE","SEPARATOR","ELECTROLYTE","DEGRADATION"],
          "description": "Editorial grouping for the annotation layer. Optional: documents from before it existed render cards without a second row."
        },
```

`cell_scene.py`:

```python
#: Editorial taxonomy for the annotation layer. Colour and claim free: it says
#: what subsystem a part belongs to, never what is true of this cell.
_CATEGORY = {
    "can": "SHELL",
    "wrap": "INSULATION", "gasket": "INSULATION", "bottom_insulator": "INSULATION",
    "crimp": "SEAL",
    "vent": "SAFETY", "cid_ptc": "SAFETY",
    "cap": "TERMINAL", "terminal_pos": "TERMINAL", "terminal_neg": "TERMINAL",
    "mandrel": "WINDING",
    "cathode_sheet": "ELECTRODE", "anode_sheet": "ELECTRODE",
    "tab_pos": "ELECTRODE", "tab_neg": "ELECTRODE",
    "separator": "SEPARATOR",
    "electrolyte": "ELECTROLYTE",
    "particles": "DEGRADATION", "sei_film": "DEGRADATION",
}


def _editorial(part_id: str) -> dict:
    """The part's category (and, later tasks, its dossier) — absent when unknown."""
    cat = _CATEGORY.get(part_id)
    return {"category": cat} if cat else {}
```

In `_part()` and `_unavailable()`, spread it into the returned dict: `return {**{...existing...}, **_editorial(part_id)}` (keep both functions' existing keys untouched).

`fixture.ts` — inside the `ANATOMY_PART_IDS.map` return, add:

```ts
      category: CATEGORIES[id],
```

with a `const CATEGORIES: Record<PartId, string>` table beside `LABELS` mirroring `_CATEGORY` (same ten words).

Python tests — append to `tests/test_cell_scene.py`:

```python
def test_every_part_carries_a_category_from_the_declared_taxonomy():
    from src.cell_scene import _CATEGORY
    scene = _build()
    taxonomy = {"SHELL","INSULATION","SEAL","SAFETY","TERMINAL",
                "WINDING","ELECTRODE","SEPARATOR","ELECTROLYTE","DEGRADATION"}
    assert len(scene["parts"]) == 19
    for part in scene["parts"]:
        assert part["category"] in taxonomy, part["id"]
        assert part["category"] == _CATEGORY[part["id"]]
```

(`_build()` — use whatever module-level scene fixture/helper the file already uses for its data-driven tests; match its call style exactly. If the file builds inline per test, copy the nearest existing test's construction line verbatim.)

JS side — append to `spec.test.ts`:

```ts
test("every part in the sample carries one of the ten declared categories", async () => {
  const sample = JSON.parse(readFileSync(new URL("../../app/static/cell_scene/sample_scene.json", import.meta.url), "utf8"));
  const taxonomy = new Set(["SHELL","INSULATION","SEAL","SAFETY","TERMINAL","WINDING","ELECTRODE","SEPARATOR","ELECTROLYTE","DEGRADATION"]);
  for (const part of sample.parts) assert.ok(taxonomy.has(part.category), `${part.id}: ${part.category}`);
});
```

(Reuse the file's existing sample-loading helper rather than duplicating `readFileSync` if one exists — mirror the neighbouring tests.)

- [ ] **Step 6: Gates + commit**

Run: Node gate, then freshness gate (sample now carries `category`; schema validation passes).

```powershell
git add frontend/src/scene/ src/cell_scene.py docs/cell_scene.schema.json tests/test_cell_scene.py app/static/cell_scene/
git commit -m "feat(scene): solve callout placement with a pure flank solver and give cards a provenance dot"
```

---

### Task 5: `tween.ts` — spring and pose math

**Files:**
- Create: `frontend/src/scene/tween.ts`, `frontend/src/scene/tween.test.ts`

**Interfaces:**

```ts
export interface Spring { value: number; velocity: number; }
export interface SpringSpec { stiffness: number; damping: number; }
export const CRITICAL: SpringSpec;              // { stiffness: 170, damping: 2 * Math.sqrt(170) }
export function springStep(s: Spring, target: number, dtMs: number, spec?: SpringSpec): Spring;
export function springSettled(s: Spring, target: number, eps?: boolean): boolean; // |v-target|<1e-3 && |vel|<1e-3
export function easeInOut(t: number): number;   // 0..1 → smoothstep-squared
export interface CameraPose { azimuth: number; elevation: number; distance: number; targetY: number; }
export function poseLerp(a: CameraPose, b: CameraPose, t: number): CameraPose;
export function poseFacing(anchor: [number, number, number], base: CameraPose): CameraPose;
export const POSE_TO_POSITION: (p: CameraPose, out: { x: number; y: number; z: number }) => void;
```

- [ ] **Step 1: Failing tests** (`tween.test.ts`)

```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { springStep, springSettled, easeInOut, poseLerp, poseFacing, CRITICAL, type Spring } from "./tween.ts";

test("the spring settles at its target in about a third of a second", () => {
  let s: Spring = { value: 0, velocity: 0 };
  let ms = 0;
  while (!springSettled(s, 1) && ms < 2000) { s = springStep(s, 1, 16); ms += 16; }
  assert.ok(springSettled(s, 1), `never settled (t=${ms}ms, v=${s.value})`);
  assert.ok(ms < 700, `settled in ${ms}ms — spec allows ~350ms, hard cap 700`);
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
  for (let t = 0.05; t <= 0.95; t += 0.05) { const v = easeInOut(t); assert.ok(v >= prev); prev = v; }
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
  assert.equal(pose.elevation, 0.4);   // elevation and distance are preserved
  assert.equal(pose.distance, 4);
});

test("CRITICAL really is critically damped", () => {
  assert.ok(Math.abs(CRITICAL.damping - 2 * Math.sqrt(CRITICAL.stiffness)) < 1e-9);
});
```

- [ ] **Step 2:** Run → FAIL (module missing).

- [ ] **Step 3: Implement**

```ts
/**
 * The motion primitives the stage's springs and camera tweens share.
 *
 * Pure numbers only — no three.js, no DOM — so settling behaviour is a thing a
 * test can assert rather than a thing a developer watches happen.
 */
export interface Spring { value: number; velocity: number; }
export interface SpringSpec { stiffness: number; damping: number; }
/** Critically damped: stiffness 170, damping 2√k → no overshoot, ~350 ms settle. */
export const CRITICAL: SpringSpec = { stiffness: 170, damping: 2 * Math.sqrt(170) };

const MAX_DT_MS = 50;

export function springStep(s: Spring, target: number, dtMs: number, spec: SpringSpec = CRITICAL): Spring {
  const dt = Math.min(Math.max(dtMs, 0), MAX_DT_MS) / 1000;
  const accel = spec.stiffness * (target - s.value) - spec.damping * s.velocity;
  return { value: s.value + s.velocity * dt, velocity: s.velocity + accel * dt };
}

export function springSettled(s: Spring, target: number, eps = 1e-3): boolean {
  return Math.abs(s.value - target) < eps && Math.abs(s.velocity) < eps;
}

export function easeInOut(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

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
```

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5:** Commit (no scene-source change that alters geometry — but tween.ts IS a bundle source, so run the freshness gate):

```powershell
git add frontend/src/scene/tween.ts frontend/src/scene/tween.test.ts app/static/cell_scene/
git commit -m "feat(scene): add pure spring, easing and camera-pose primitives"
```

---

### Task 6: Smooth camera framing + pulsing reticle

**Files:**
- Modify: `frontend/src/scene/engine.ts` (`inspect` inside handle ~line 770-800, `tick()` line 696, `updateLeaderLines()` reticle drawing, `fitView` line 297)

**Interfaces:**
- Consumes: `tween.ts` (Task 5), `layoutFlank` (Task 4).
- Produces: engine-internal `camTween: { from: CameraPose; to: CameraPose; start: number; dur: number } | null`, `framePose(to: CameraPose, durMs?: number)`, `currentPose(): CameraPose`; reticle drawing with `pulseRadius`.

- [ ] **Step 1: Manual characterisation (this task's "test" is behavioural and pinned by the existing suite)**

Run `cd frontend; npm run test:scene` FIRST and record PASS baseline. The invariants that must survive: `spec.test.ts` "the renderer mounts the real sample and can scrub its whole timeline" (mounts the engine — a broken tick throws there).

- [ ] **Step 2: Implement camera tween**

Near the top of `mountCellScene` (after `controls` is created):

```ts
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

  controls.addEventListener("start", () => { camTween = null; }); // any user grab cancels
```

In `tick()`, before `renderer.render`:

```ts
    if (camTween) {
      const t = Math.min(1, (performance.now() - camTween.start) / camTween.dur);
      applyPose(poseLerp(camTween.from, camTween.to, t));
      if (t >= 1) camTween = null;
    }
```

Replace the synchronous camera mutation in `handle.inspect(partId)` with:

```ts
    const part = partId ? currentParts.find((p) => p.id === partId) : null; // whatever the closure already resolves anchors from
    if (part) framePose(poseFacing(part.anchor, { ...currentPose(), targetY: 0 }), 500);
```

Keep `inspect`'s existing highlight/pin/`onInspect` semantics — only the camera turn becomes a tween. `fitView` (line 297) keeps its job for resize/reset, but `resetView` should now `framePose` instead of jumping: same target pose it computes today, `dur = 600`.

- [ ] **Step 3: Reticle pulse**

In `updateLeaderLines()`'s reticle drawing (the two concentric circles per anchor), give the ACTIVE part's reticle a third ring whose radius breathes — computed from a module-scope `let pulsePhase = 0` advanced in `tick()`:

```ts
// in tick():
pulsePhase = (pulsePhase + dtMs / 1600) % 1;   // 1.6 s period
// in the reticle draw, for the active item only:
const pr = R + 1.4 * Math.sin(pulsePhase * 2 * Math.PI);
<circle cx=... cy=... r="${pr}" fill="none" stroke="${theme.accent}" stroke-width="1" opacity="0.9"/>
```

Inactive reticles keep exactly today's two static rings. Comment the intent: "one pulsing target, not nineteen — a pulse everywhere is noise."

- [ ] **Step 4: Gates + commit**

Run: Node gate + freshness gate.

```powershell
git add frontend/src/scene/engine.ts app/static/cell_scene/
git commit -m "feat(scene): frame inspected parts with an eased camera tween and pulse the active reticle"
```

---

## PHASE 3 — Dossier

### Task 7: document-carried `dossier` block

**Files:**
- Modify: `docs/cell_scene.schema.json` (parts.items.properties)
- Modify: `src/cell_scene.py` (`_DOSSIER_TABLE`, `_editorial` extension, `_part`/`_unavailable` already spread `_editorial`)
- Test: `tests/test_cell_scene.py`

**Interfaces:**
- Produces: optional per-part object:

```ts
// types.ts
export type DossierTag = "typical" | "measured" | "derived" | "fitted" | "refusal";
export interface DossierSpecRow { label: string; value: string; unit?: string; tag: DossierTag; }
export interface PartDossier {
  latinTitle: string;
  subsystem: string;
  material: string;
  degradation?: string;
  insight?: string;
  specs?: DossierSpecRow[];
}
// ScenePart.dossier?: PartDossier;
```

- [ ] **Step 1: Failing Python tests** (append to `tests/test_cell_scene.py`)

```python
def test_every_part_carries_a_dossier_with_dual_title_and_prose():
    scene = _build()
    for part in scene["parts"]:
        d = part.get("dossier")
        assert d, f"{part['id']} has no dossier"
        assert d["latinTitle"].isupper(), part["id"]
        assert d["subsystem"] and d["material"]
        assert d.get("insight"), part["id"]


def test_every_spec_row_carries_a_tag_from_the_declared_vocabulary():
    scene = _build()
    vocab = {"typical", "measured", "derived", "fitted", "refusal"}
    rows = [r for p in scene["parts"] for r in p["dossier"].get("specs", [])]
    assert len(rows) >= 19 * 4, "every part declares at least its four spec rows"
    for row in rows:
        assert row["tag"] in vocab, row
        assert row["label"] and row["value"] is not None


def test_rows_the_platform_cannot_source_are_refusals_not_numbers():
    scene = _build()
    refusals = [r for p in scene["parts"] for r in p["dossier"].get("specs", [])
                if r["tag"] == "refusal"]
    assert refusals, "the impedance/overpotential questions must be answered honestly"
    for row in refusals:
        assert not row["value"].strip().replace(".", "").replace(",", "").isdigit(), row
```

(Use the file's existing scene-construction helper — copy the line every other test uses.)

- [ ] **Step 2:** Run `pytest tests/test_cell_scene.py -k dossier -v` → FAIL.

- [ ] **Step 3: Schema**

In `parts.items.properties`, add (before the closing of `properties`):

```json
        "dossier": {
          "type": "object",
          "description": "The part's technical dossier: dual Latin/engineering title, subsystem prose, and a two-tier spec table. Rows tagged 'typical' are format-level figures, not this cell's datasheet (say so in disclosures); rows tagged 'refusal' answer a question the platform does not measure per part, with words instead of a number. Optional — documents from before it exist render the older panel.",
          "required": ["latinTitle", "subsystem", "material"],
          "additionalProperties": false,
          "properties": {
            "latinTitle": { "type": "string", "minLength": 1 },
            "subsystem": { "type": "string", "minLength": 1 },
            "material": { "type": "string", "minLength": 1 },
            "degradation": { "type": "string" },
            "insight": { "type": "string" },
            "specs": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["label", "value", "tag"],
                "additionalProperties": false,
                "properties": {
                  "label": { "type": "string", "minLength": 1 },
                  "value": { "type": "string", "minLength": 1 },
                  "unit": { "type": "string" },
                  "tag": { "enum": ["typical", "measured", "derived", "fitted", "refusal"] }
                }
              }
            }
          }
        }
```

Also append one disclosure sentence in `_disclosures(...)`'s returned list (find the list near line 1892 and add):

```python
        "Rows tagged 'typical' on part dossiers are format-level figures for cells "
        "of this format, not measurements of this cell; rows tagged 'refusal' are "
        "questions this platform does not measure per part, answered in words.",
```

- [ ] **Step 4: Producer table**

In `cell_scene.py`, add the table (migrate prose from `frontend/src/components/CellSceneView.tsx`'s `PART_DOSSIERS` — keep each `latinTitle`/`subsystem`/`material`/`degradation` string verbatim; write `insight` fresh where React had none; the 16 existing entries plus these 3 new ones). Content definition for all 19:

| id | latinTitle | spec rows (`label = value unit @tag`) |
|---|---|---|
| can | THORAX METALLICUS | Wall = 0.25 mm @typical · Material = nickel-plated steel @typical · Integrity = visual/pressure check @refusal (not instrumented per part) · Corrosion rate = not measured per part @refusal |
| wrap | TUNICA CONTRACTA | Shrink film = 50 µm PVC @typical · Shrink ratio = 2:1 @typical · Flame class = UL94 VMT2 @typical |
| gasket | ANNULUS SIGILLANS | Thickness = 0.5 mm @typical · Material = PP / paper-composite @typical · Compression set = not measured per part @refusal |
| crimp | CORONA COMPRESSA | Roll height = 1.2 mm @typical · Hermeticity = leak-rate check @refusal (not instrumented per part) |
| cap | GALEA TERMINALIS | Plate thickness = 0.3 mm @typical · Boss diameter from topAssembly @typical · Exhaust slots = scored geometry, not a card @refusal |
| vent | VALVULA SALUTIS | Opening pressure = 1.0–1.5 MPa @typical · Burst pressure = 2.0–2.5 MPa @typical · Spring rate = not measured per part @refusal |
| cid_ptc | CUSTOS INTERRUPTOR | CID trip pressure = 1.0–1.6 MPa @typical · PTC resistance @25°C = 0.05–0.5 Ω @typical · Trip current = not measured per part @refusal |
| terminal_pos | POLUS POSITIVUS | Stud diameter from topAssembly @typical · Contact resistance growth = see cell resistance series @derived |
| terminal_neg | POLUS NEGATIVUS | Base = can steel, same wall @typical · Contact resistance growth = see cell resistance series @derived |
| tab_pos | LIGAMENTUM ALUMINII | Foil tab = 6 × 0.1 mm Al @typical · Weld integrity = not measured per part @refusal |
| tab_neg | LIGAMENTUM CUPRI | Foil tab = 6 × 0.1 mm Cu @typical · Weld integrity = not measured per part @refusal |
| mandrel | AXIS SUBSTRACTIONIS | Diameter from physical.cylindrical.mandrelDiameterMm @measured · Material = nickel-plated steel @typical |
| cathode_sheet | STRATUM CATHODICUM | Coating thickness = 60–80 µm/side @typical · Loading = 15–20 mg/cm² @typical · Porosity = 30–35 % @typical · Tortuosity = 2.5–4 τ @typical |
| anode_sheet | STRATUM ANODICUM | Coating thickness = 70 µm @typical · Cu foil = 10 µm @typical · Loading = 3.5–4.0 mg/cm² @typical · Porosity = 25–30 % @typical · Tortuosity = 2–3 τ @typical · Impedance contribution = not measured per part @refusal |
| separator | SEPTUM SEPARANS | Thickness = 16–25 µm @typical · Porosity = 38–45 % @typical · Tortuosity = 3–5 τ @typical · Shutdown temp = 135 °C @typical |
| electrolyte | LIQUIDUM CONDUCTOR | Salt = 1 M LiPF6 in EC:DMC @typical · Conductivity = 10 mS/cm @25 °C @typical · Consumption rate = inferred from SEI fit @fitted |
| particles | PARTICULAE MOBILES | D50 = 4–8 µm @typical · Fraction lost = fitted LAM share @fitted (gate: splitIdentified) · Per-particle count = not measured per part @refusal |
| sei_film | MEMBRANA SEI | Growth law = √n of cycle count @fitted · Thickness at cursor = seiThicknessNm @derived · Local overpotential = not measured per part @refusal |

(Adjust any `id` ↔ `latinTitle` pairing to the EXACT strings already in `PART_DOSSIERS` for the 16 existing ids — migration means verbatim, not re-authored. The three new latinTitles above are final.)

Add to the table: `subsystem` (React's existing strings, verbatim), `material` (verbatim), `degradation` (verbatim), `insight` (one or two sentences; for anode_sei: "SEI growth consumes cyclable lithium under high C-rate cycling; the film's own thickness series is the platform's √n fit, drawn only when the channel split is identified."). Spec rows: encode the table above as dicts.

```python
_DOSSIER_TABLE: dict[str, dict] = {
    "gasket": {
        "latinTitle": "ANNULUS SIGILLANS",
        "subsystem": "Cap Seal & Leak-Tight Closure",
        "material": "Polypropylene / paper-composite flat gasket",
        "degradation": "Compression set with age and heat; leak paths after deep crimp cycling.",
        "insight": "The gasket is the only thing standing between the electrolyte and the atmosphere; it is why a swollen cell is a safety event and not a cosmetic one.",
        "specs": [
            {"label": "Thickness", "value": "0.5", "unit": "mm", "tag": "typical"},
            {"label": "Material", "value": "PP / paper-composite", "unit": "", "tag": "typical"},
            {"label": "Compression set", "value": "not measured per part on this platform", "unit": "", "tag": "refusal"},
        ],
    },
    "cid_ptc": { ... }, "bottom_insulator": { ... },  # same shape, entries per the table above
    # ... then the 16 migrated entries keyed by their existing ids
}
```

`_editorial` grows one line:

```python
def _editorial(part_id: str) -> dict:
    out = {}
    cat = _CATEGORY.get(part_id)
    if cat:
        out["category"] = cat
    dossier = _DOSSIER_TABLE.get(part_id)
    if dossier:
        out["dossier"] = dossier
    return out
```

- [ ] **Step 5:** Run `pytest tests/test_cell_scene.py -k dossier -v` → PASS; then freshness gate (sample now carries dossiers; the schema addition is optional so old docs stay valid).

- [ ] **Step 6: Commit**

```powershell
git add docs/cell_scene.schema.json src/cell_scene.py tests/test_cell_scene.py app/static/cell_scene/
git commit -m "feat(scene): carry every part's dossier in the document, with two-tier provenance tags"
```

---

### Task 8: `dossier.ts` composer with live rows and refusals

**Files:**
- Create: `frontend/src/scene/dossier.ts`, `frontend/src/scene/dossier.test.ts`
- Modify: `frontend/src/scene/index.ts` (export `composeDossier` + types)

**Interfaces:**

```ts
import type { CellSceneSpec, DossierSpecRow } from "./types.ts";
export interface LiveRow extends DossierSpecRow { atCycle: number | null; }
export interface DossierView {
  partId: string;
  label: string;
  latinTitle: string | null;   // null when the document predates dossiers
  subsystem: string | null;
  material: string | null;
  degradation: string | null;
  insight: string | null;
  specs: DossierSpecRow[];     // document rows, tags intact
  live: LiveRow[];             // composed at the cursor, tags intact
  partial: boolean;            // any row had to refuse
}
export function composeDossier(spec: CellSceneSpec, partId: string, cursor: number): DossierView | null;
```

- [ ] **Step 1: Failing tests** (`dossier.test.ts`)

```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { composeDossier } from "./dossier.ts";
import { makeSpec } from "./fixture.ts";

test("a part with a document dossier composes with its spec rows intact", () => {
  const spec = makeSpec();
  const view = composeDossier(spec, "anode_sheet", 10)!;
  assert.ok(view, "dossier exists");
  assert.equal(view.partId, "anode_sheet");
  assert.ok(view.live.length >= 4, "temperature, resistance, SOH and film rows at minimum");
  assert.ok(view.specs.every((r) => ["typical","measured","derived","fitted","refusal"].includes(r.tag)));
});

test("live rows only ever print numbers the document actually carries", () => {
  const spec = makeSpec({ splitIdentified: false });   // use FixtureOptions' real flag name if different — see note
  const view = composeDossier(spec, "sei_film", 50)!;
  const film = view.live.find((r) => r.label === "SEI thickness")!;
  assert.equal(film.tag, "refusal");
  assert.match(film.value, /not identified|refus/i);
  assert.ok(view.partial);
});

test("an unidentified split never prints a thickness, a broken cursor never prints NaN", () => {
  const spec = makeSpec();
  const view = composeDossier(spec, "sei_film", 9999)!; // cursor clamps
  for (const row of [...view.live, ...view.specs]) {
    assert.ok(!/NaN|undefined/.test(`${row.value}${row.label}`));
  }
});

test("a part id with no dossier returns null rather than a hollow card", () => {
  assert.equal(composeDossier(makeSpec(), "no_such_part", 0), null);
});
```

(If `makeSpec`'s real option for an unidentified split is named differently — check `FixtureOptions` — use that name; the fixture's default HAS an identified split, so the refusal test needs the unidentifiable variant that `geometry.test.ts` line 581 already uses: mirror its call exactly.)

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

```ts
import type { CellSceneSpec, DossierSpecRow, PartDossier } from "./types.ts";
import { readingAt, buildTimeline } from "./geometry.ts";

export interface LiveRow extends DossierSpecRow { atCycle: number | null; }
export interface DossierView { /* as declared above */ }

const TAG = { measured: "measured", derived: "derived", fitted: "fitted", projected: "derived" } as const;

const num = (v: number | null, digits = 1): string =>
  v === null || !Number.isFinite(v) ? "no reading" : v.toFixed(digits);

export function composeDossier(spec: CellSceneSpec, partId: string, cursor: number): DossierView | null {
  const part = spec.parts.find((p) => p.id === partId);
  if (!part) return null;
  const timeline = buildTimeline(spec);
  const c = Math.max(0, Math.min(cursor, timeline.cycles.length - 1));
  const cycle = timeline.cycles[c] ?? null;
  const projected = timeline.projected[c] === true;
  const d: PartDossier | undefined = part.dossier;
  const identified = spec.physics.splitIdentified === true;

  const live: LiveRow[] = [];
  const add = (label: string, value: string, unit: string, tag: DossierSpecRow["tag"]) =>
    live.push({ label, value, unit, tag, atCycle: cycle });

  const temp = readingAt(spec.series.temperatureC, c).value;
  add("Temperature", num(temp), "°C", temp === null ? "refusal" : "measured");
  const r = readingAt(spec.series.resistanceNormalized, c).value;
  add("Resistance growth", r === null ? "no reading" : r.toFixed(3), "× initial (derived)", r === null ? "refusal" : "derived");
  const soh = readingAt(spec.series.sohPct, c).value;
  add("State of health", soh === null ? "no reading" : soh.toFixed(1), "%", soh === null ? "refusal" : projected ? "projected" : "measured");
  const sei = identified ? readingAt(spec.series.seiThicknessNm, c).value : null;
  add("SEI thickness", identified ? num(sei, 0) : "split not identified — withheld",
      identified ? "nm" : "", identified && sei !== null ? "derived" : "refusal");
  const share = identified ? readingAt(spec.series.seiSharePct, c).value : null;
  add("Fade attributed to SEI", identified ? num(share) : "split not identified — withheld",
      identified ? "% of fade (fitted)" : "", identified && share !== null ? "fitted" : "refusal");
  add("Impedance contribution", "not measured per part on this platform", "mΩ", "refusal");
  add("Local overpotential", "not measured per part on this platform", "mV", "refusal");

  return {
    partId,
    label: part.label,
    latinTitle: d?.latinTitle ?? null,
    subsystem: d?.subsystem ?? null,
    material: d?.material ?? null,
    degradation: d?.degradation ?? null,
    insight: d?.insight ?? null,
    specs: d?.specs ?? [],
    live,
    partial: live.some((r2) => r2.tag === "refusal"),
  };
}
```

Verify against real field names before finishing: `spec.physics.splitIdentified` (exists), `timeline.projected` (exists — `buildScene` reads `timeline.projected[cursor]`), `spec.series.seiThicknessNm`/`seiSharePct`/`temperatureC`/`resistanceNormalized`/`sohPct` (all required by schema), `readingAt(series, i).value` (exists). `part.dossier` only exists after Task 7's types change — if you run tasks strictly in order, it does.

Export `composeDossier` and the dossier types from `index.ts` alongside the existing type exports.

- [ ] **Step 4:** Run → PASS. Node gate.

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/scene/dossier.ts frontend/src/scene/dossier.test.ts frontend/src/scene/index.ts
git commit -m "feat(scene): compose the dossier's live block from real series and honest refusals"
```

---

### Task 9: engine dossier card, dim/desaturate, Fresnel rim, React migration

**Files:**
- Modify: `frontend/src/scene/engine.ts` (`PartObject` line 135, `setHovered` line 441, `paint` line 362, overlay creation near `badgeOverlay`)
- Modify: `frontend/src/components/CellSceneView.tsx` (delete `PART_DOSSIERS`, render from document via `composeDossier`)

**Interfaces:**
- Consumes: `composeDossier` (Task 8), `desaturateHex` (Task 4), `theme.accent2`.
- Produces: floating `#cell-scene-dossier` card in the stage overlay, shown while a part is pinned/hovered; `PartObject` gains `baseColor: string; baseOpacity: number; rim: { value: number }`.

- [ ] **Step 1: Characterise current hover behaviour**

Run Node gate, record PASS. The hover treatment must be restorable exactly (existing test at geometry level is unaffected — this is engine-side material state).

- [ ] **Step 2: Selection treatment in engine.ts**

Extend `PartObject`:

```ts
interface PartObject {
  id: string;
  group: Group;
  mesh: Mesh;
  material: MeshStandardMaterial;
  baseEmissive: number;
  /** The build's own colour/opacity, so clearing a selection restores exactly. */
  baseColor: string;
  baseOpacity: number;
  /** Uniform the Fresnel rim reads; 0 when this part is not the selection. */
  rim: { value: number };
}
```

In `paint()` where each material is created, add the rim injection immediately after material construction (graceful by construction: if three renames the chunk, `replace` is a no-op — no throw, no rim):

```ts
    const rim = { value: 0 };
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uRimStrength = rim;
      shader.uniforms.uRimColor = { value: new Color(spec.theme.accent2 ?? spec.theme.accent) };
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
```

Record `baseColor: colorHexUsedByBuild`, `baseOpacity: material.opacity` from the built part on the `PartObject`.

`setHovered(id)` becomes (replacing the emissive-only body):

```ts
  function setHovered(id: string | null): void {
    const active = id ?? pinnedId;   // pinned beats hovered, per existing precedence
    for (const part of partObjects) {
      const isActive = part.id === active;
      const hasActive = active !== null;
      part.material.emissiveIntensity = isActive
        ? part.baseEmissive + 0.6
        : hasActive ? part.baseEmissive : part.baseEmissive;
      part.rim.value = isActive ? 1 : 0;
      if (hasActive && !isActive) {
        part.material.opacity = Math.min(part.baseOpacity, 0.15);
        part.material.color.set(desaturateHex(part.baseColor, 0.7, currentSpec.theme.muted));
      } else {
        part.material.opacity = part.baseOpacity;
        part.material.color.set(part.baseColor);
      }
      part.material.needsUpdate = false; // colours are uniforms, not recompiles
    }
    ...existing annotation/badge opacity update, emit() call...
  }
```

(`DIM_OPACITY` for badges/lines stays exactly as is; this adds the MESH falloff your spec asked for — opacity 0.15 + desaturation — while never touching data-driven colours when nothing is active. `pinnedId`/hover precedence: keep whatever names the existing closure uses.)

- [ ] **Step 3: Floating dossier card**

Create alongside `badgeOverlay`:

```ts
  const dossierCard = document.createElement("div");
  dossierCard.id = "cell-scene-dossier";
  dossierCard.style.cssText =
    "position:absolute;top:56px;left:16px;width:300px;max-height:60%;overflow:auto;" +
    "display:none;pointer-events:auto;padding:10px 12px;border-radius:6px;" +
    `background:${currentSpec.theme.panel};border:1px solid ${currentSpec.theme.grid};` +
    `color:${currentSpec.theme.text};font-family:${currentSpec.theme.fonts?.mono ?? "ui-monospace, monospace"};font-size:11px;z-index:3;`;
  overlay.appendChild(dossierCard);
```

Render function (called from `setHovered` when a part is active, hidden when none):

```ts
  function renderDossier(partId: string): void {
    const view = composeDossier(currentSpec, partId, build.cursor);
    if (!view) { dossierCard.style.display = "none"; return; }
    const theme = activePalette();   // Task 11 wires this; before that, currentSpec.theme
    const row = (r: { label: string; value: string; unit: string; tag: string }) =>
      `<div style="display:flex;justify-content:space-between;gap:8px;padding:2px 0">
         <span style="color:${theme.muted}">${escapeHtml(r.label)}</span>
         <span style="text-align:right"><span style="color:${r.tag === "refusal" ? theme.muted : theme.text}">${escapeHtml(r.value)}</span>
         <span style="color:${theme.muted};font-size:9px"> ${escapeHtml(r.unit)}</span>
         <span title="${r.tag}" style="display:inline-block;width:6px;height:6px;border-radius:50%;margin-left:4px;background:${tagColor(r.tag, theme)}"></span></span>
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
      (view.insight ? `<div style="border-top:1px solid ${theme.grid};margin-top:6px;padding-top:6px;color:${theme.muted}">${escapeHtml(view.insight)}</div>` : "");
    dossierCard.style.display = "block";
  }
  const tagColor = (tag: string, theme: SceneTheme) =>
    tag === "refusal" ? theme.muted
    : tag === "typical" ? theme.grid
    : theme.provenanceColors[tag as "measured"] ?? theme.accent;
```

Hide the card in `dispose()`'s DOM teardown (whatever the badge cleanup does, mirror it).

- [ ] **Step 4: React migration**

In `CellSceneView.tsx`: delete the whole `PART_DOSSIERS` const. Where the right panel currently reads `PART_DOSSIERS[inspected]`, call `composeDossier(spec, inspected, frame.cursor)` (import from `../scene`) and render: header `latinTitle // label`, subsystem, material, spec table with the same tag dots (`provenanceColors`-style mapping), live block, degradation, insight — keeping the panel's existing layout classes. Every string now comes from the document; if `dossier` is absent (old API doc), fall back to today's title/meaning/reading rendering — that fallback must still work.

- [ ] **Step 5: Gates + commit**

Run: Node gate (`npm run build` must pass — React is type-checked here) + freshness gate.

```powershell
git add frontend/src/scene/engine.ts frontend/src/components/CellSceneView.tsx app/static/cell_scene/
git commit -m "feat(scene): dossier card in the stage, dim-and-rim selection, React reads the document"
```

**Phase 3 ends here.**

---

## PHASE 4 — Dual palettes

### Task 10: `palettes` in the contract + producer

**Files:**
- Modify: `docs/cell_scene.schema.json` (root `properties` — root is `additionalProperties: false`, so the new key MUST be declared)
- Modify: `frontend/src/scene/types.ts` (`CellSceneSpec.palettes?`, `SceneTheme` presentation tokens)
- Modify: `src/cell_scene.py` (`_CODEX_THEME`, `_OBSIDIAN_THEME`, `_default_palettes()`, emission in `build_cell_scene` ~line 1358)
- Test: `tests/test_cell_scene.py`

**Interfaces:**
- Produces: `CellSceneSpec.palettes?: Record<string, SceneTheme>`; `SceneTheme.name?`, `SceneTheme.variant?: "dark" | "light"`, `SceneTheme.fonts?: { display: string; mono: string }`, `SceneTheme.accent2?: string`; `default_palettes() -> dict`.

- [ ] **Step 1: Failing tests** (append to `tests/test_cell_scene.py`)

```python
def test_every_palette_keeps_the_platform_s_own_band_thresholds():
    scene = _build()
    base = scene["theme"]["sohBands"]
    base_t = scene["theme"]["temperatureBands"]
    assert set(scene.get("palettes", {})) == {"codex", "obsidian"}
    for name, palette in scene["palettes"].items():
        assert [(b["min"], b["max"], b["label"]) for b in palette["sohBands"]] == \
               [(b["min"], b["max"], b["label"]) for b in base], name
        assert [(b["min"], b["max"]) for b in palette["temperatureBands"]] == \
               [(b["min"], b["max"]) for b in base_t], name


def test_every_palette_colour_is_a_hex_and_carries_its_presentation_tokens():
    import re
    scene = _build()
    for name, palette in scene["palettes"].items():
        assert palette["name"] and palette["variant"] in {"dark", "light"}
        assert palette["fonts"]["display"] and palette["fonts"]["mono"]
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", palette["accent2"])
        for key, value in palette.items():
            if isinstance(value, str) and key.endswith(("Color", "")) and value.startswith("#"):
                assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), (name, key, value)


def test_codex_and_obsidian_disagree_only_where_a_palette_may():
    scene = _build()
    codex, obsidian = scene["palettes"]["codex"], scene["palettes"]["obsidian"]
    assert codex["background"] == "#F4EEDA"
    assert obsidian["background"] == "#070A10"
    assert codex["variant"] == "light" and obsidian["variant"] == "dark"
```

- [ ] **Step 2:** Run `pytest tests/test_cell_scene.py -k palette -v` → FAIL.

- [ ] **Step 3: Schema**

Root `properties` gains (root `additionalProperties: false` — required):

```json
    "palettes": {
      "type": "object",
      "description": "Named alternative SceneThemes. setTheme(name) picks one; the document's own `theme` remains the default and still renders when this block is absent (documents from before it existed).",
      "additionalProperties": {
        "type": "object",
        "additionalProperties": true
      }
    },
```

`types.ts` — `CellSceneSpec` gains `palettes?: Record<string, SceneTheme>;`. `SceneTheme` gains:

```ts
  /** Display name for the switcher, e.g. "Codex Atlanticus". */
  name?: string;
  variant?: "dark" | "light";
  fonts?: { display: string; mono: string };
  /** Second accent: brass on codex, imperial gold on obsidian. */
  accent2?: string;
```

- [ ] **Step 4: Producer**

Add beside `_DEFAULT_THEME` (values verbatim from spec §6.2):

```python
#: The two art-direction palettes. Band THRESHOLDS are copied from
#: _DEFAULT_THEME on purpose and tests pin that equality: a palette may
#: repaint the platform's bands for its background, never re-band them.
_CODEX_THEME = {
    "background": "#F4EEDA", "panel": "#E8DFCA", "text": "#2A2118", "muted": "#6B5B45",
    "accent": "#2F4F6F", "accent2": "#8A6A2F", "grid": "#C9BCA0", "metal": "#7D7466",
    "anodeColor": "#A8763F", "cathodeColor": "#5F4B8B", "separatorColor": "#CFC5AD",
    "electrolyteColor": "#4A7FA5", "seiColor": "#7A4F9E",
    "name": "Codex Atlanticus", "variant": "light",
    "fonts": {
        "display": "'Cinzel','EB Garamond',Georgia,serif",
        "mono": "'JetBrains Mono',ui-monospace,Menlo,monospace",
    },
    "sohBands": [
        {"min": 90.0, "max": None, "color": "#2F855A", "label": "Healthy"},
        {"min": 80.0, "max": 90.0, "color": "#B7791F", "label": "Degrading"},
        {"min": 0.0, "max": 80.0, "color": "#C53030", "label": "End of Life"},
    ],
    "temperatureBands": [
        {"min": 0.0, "max": 30.0, "color": "#2B6CB0"},
        {"min": 30.0, "max": 45.0, "color": "#B7791F"},
        {"min": 45.0, "max": None, "color": "#C53030"},
    ],
    "provenanceColors": {
        "measured": "#2F855A", "derived": "#2F4F6F", "fitted": "#6B46C1",
        "projected": "#B7791F", "": "#8A8172",
    },
    "sopFloorPct": 70.0, "powerGaugeMinPct": 40.0, "powerGaugeMaxPct": 110.0,
}

_OBSIDIAN_THEME = {
    "background": "#070A10", "panel": "#0C101A", "text": "#e2e8f0", "muted": "#94a3b8",
    "accent": "#38BDF8", "accent2": "#EAB308", "grid": "#1B2436", "metal": "#cbd5e0",
    "anodeColor": "#C89B6A", "cathodeColor": "#7b6ba8", "separatorColor": "#d9e2ec",
    "electrolyteColor": "#38BDF8", "seiColor": "#A78BFA",
    "name": "Obsidian Aerospace", "variant": "dark",
    "fonts": {
        "display": "'Inter',system-ui,sans-serif",
        "mono": "'JetBrains Mono',ui-monospace,Menlo,monospace",
    },
    "sohBands": [          # the app's own bands, verbatim — same thresholds
        {"min": 90.0, "max": None, "color": "#48bb78", "label": "Healthy"},
        {"min": 80.0, "max": 90.0, "color": "#f6e05e", "label": "Degrading"},
        {"min": 0.0, "max": 80.0, "color": "#fc8181", "label": "End of Life"},
    ],
    "temperatureBands": [
        {"min": 0.0, "max": 30.0, "color": "#4299e1"},
        {"min": 30.0, "max": 45.0, "color": "#ecc94b"},
        {"min": 45.0, "max": None, "color": "#e53e3e"},
    ],
    "provenanceColors": {
        "measured": "#48bb78", "derived": "#63b3ed", "fitted": "#b794f4",
        "projected": "#f6ad55", "": "#718096",
    },
    "sopFloorPct": 70.0, "powerGaugeMinPct": 40.0, "powerGaugeMaxPct": 110.0,
}


def default_palettes() -> dict:
    import copy
    return copy.deepcopy({"codex": _CODEX_THEME, "obsidian": _OBSIDIAN_THEME})
```

In `build_cell_scene`'s return dict, beside `"theme": ...` (line ~1358), add a parameter `palettes: "dict | None" = None` to the signature and emit:

```python
        "palettes": {**default_palettes(), **(palettes or {})},
```

(`default_theme()` itself stays byte-identical — its equivalence tests and the app's own band tests depend on it.)

- [ ] **Step 5:** Run the three new tests → PASS. Freshness gate (sample gains `palettes`).

- [ ] **Step 6: Commit**

```powershell
git add docs/cell_scene.schema.json frontend/src/scene/types.ts src/cell_scene.py tests/test_cell_scene.py app/static/cell_scene/
git commit -m "feat(scene): carry Codex and Obsidian palettes in the document with pinned band thresholds"
```

---

### Task 11: `setTheme`, per-variant lighting/materials, backdrop, chrome, fonts

**Files:**
- Modify: `frontend/src/scene/theme.ts` (`paletteFor`), `frontend/src/scene/materials.ts` (`PALETTE_MATERIAL_ADJUST`, `materialFor`), `frontend/src/scene/engine.ts` (stage style application, chrome overlay, backdrop div, `setTheme`), `frontend/src/scene/index.ts` (exports)
- Modify: `frontend/index.html`, `app/static/cell_scene/index.html`, `app/_scene_view.py` (font links)

**Interfaces:**
- Produces: `paletteFor(spec: CellSceneSpec, name: string | null | undefined): SceneTheme`; `materialFor(id: string, variant: "dark" | "light"): PartMaterial`; handle `setTheme(name: string): void`; mount option `MountOptions.theme?: string`.

- [ ] **Step 1: Failing tests** — extend `spec.test.ts`:

```ts
test("paletteFor falls back to the document's own theme, and unknown names never throw", async () => {
  const { paletteFor } = await import("./theme.ts");
  const sample = loadSample();
  assert.equal(paletteFor(sample, "codex").background, "#F4EEDA");
  assert.equal(paletteFor(sample, "nope"), sample.theme);
  assert.equal(paletteFor({ ...sample, palettes: undefined }, "codex"), sample.theme);
});

test("the light palette asks for softer metals than the dark one, clamped into range", async () => {
  const { materialFor } = await import("./materials.ts");
  const dark = materialFor("can", "dark");
  const light = materialFor("can", "light");
  assert.ok(light.roughness >= dark.roughness);
  assert.ok(light.metalness >= 0 && light.roughness <= 1);
  assert.ok(light.envMapIntensity < dark.envMapIntensity);
});
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

`theme.ts`:

```ts
import type { CellSceneSpec, SceneTheme } from "./types.ts";

/** The palette a stage should paint with: named if the document carries it, else its own theme. */
export function paletteFor(spec: CellSceneSpec, name: string | null | undefined): SceneTheme {
  if (name && spec.palettes && spec.palettes[name]) return spec.palettes[name];
  return spec.theme;
}

/** Per-origin stage preferences. A browser that refuses storage simply keeps none. */
const PREF_KEY = "cell-scene.pref";
export function readPref<T>(key: string): T | null {
  try {
    const raw = window.localStorage.getItem(PREF_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return (key in parsed ? (parsed[key] as T) : null);
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
```

(`readPref`/`writePref` live HERE, in Task 11 — Task 12's HUD imports them from `theme.ts` rather than redeclaring them. Export both from `index.ts` alongside `paletteFor` only if a host asks; the scene modules import them directly.)

`materials.ts`:

```ts
export const PALETTE_MATERIAL_ADJUST: Record<"dark" | "light", { metalness: number; roughness: number; envMapIntensity: number }> = {
  dark: { metalness: 0, roughness: 0, envMapIntensity: 1 },
  light: { metalness: -0.1, roughness: 0.12, envMapIntensity: 0.6 },
};

/** A part's material, adjusted for the palette's light. Both tables are data a test reads. */
export function materialFor(id: string, variant: "dark" | "light" = "dark"): PartMaterial {
  const base = PART_MATERIALS[id] ?? { metalness: 0, roughness: 0.7, envMapIntensity: 0.5 };
  const d = PALETTE_MATERIAL_ADJUST[variant] ?? PALETTE_MATERIAL_ADJUST.dark;
  return {
    metalness: Math.min(1, Math.max(0, base.metalness + d.metalness)),
    roughness: Math.min(1, Math.max(0, base.roughness + d.roughness)),
    envMapIntensity: base.envMapIntensity * d.envMapIntensity,
  };
}
```

(If `materialFor`'s current signature/name differs, adapt: the existing engine call site passes the part id today — give it the variant from a module-scope `let themeVariant: "dark" | "light" = "dark"` that `applyStageStyle` sets.)

Engine — module scope:

```ts
  let themeName: string | null = options.theme ?? readPref("theme");
  const activePalette = () => paletteFor(currentSpec, themeName);
  let themeVariant: "dark" | "light" = "dark";
```

`applyStageStyle()` (called from `paint`, `rebuild`, `setTheme`):

```ts
  function applyStageStyle(): void {
    const theme = activePalette();
    themeVariant = theme.variant ?? "dark";
    scene.background = new Color(theme.background);
    renderer.toneMappingExposure = themeVariant === "light" ? 1.15 : 1.05;
    // Lighting: locate the DirectionalLight (key) and HemisphereLight (fill) built
    // at mount, keep them as keyLight/fillLight, then:
    keyLight.color.set(themeVariant === "light" ? "#fff6e0" : "#ffffff");
    if ("environmentIntensity" in scene) (scene as { environmentIntensity: number }).environmentIntensity =
      themeVariant === "light" ? 0.4 : 0.55;
    fillLight.intensity = themeVariant === "light" ? 0.9 : 0.7;
    backdrop.innerHTML = themeVariant === "light" ? paperGrain(theme) : vignette(theme);
    chromeLayer.innerHTML = chromeSvg({
      theme, ornate: themeVariant === "light",
      width: stage.clientWidth, height: stage.clientHeight,
    });
    dossierCard.style.background = theme.panel;
    dossierCard.style.borderColor = theme.grid;
    dossierCard.style.color = theme.text;
    writePref("theme", themeName ?? "");
  }
```

Backdrop/chrome layers (created beside `badgeOverlay`, backdrop BELOW the canvas or as a pointer-events:none layer above canvas but below badges; chrome above badges):

```ts
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:0";
  const chromeLayer = document.createElement("div");
  chromeLayer.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:2";
```

```ts
function vignette(theme: SceneTheme): string {
  return `background:radial-gradient(ellipse at center, transparent 55%, ${theme.background}cc 100%)`;
}
function paperGrain(theme: SceneTheme): string {
  // Procedural, ~2 KB of logic: a 64×64 noise tile drawn once onto a canvas.
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(64, 64);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 200 + Math.floor(((i * 2654435761) % 55)); // deterministic, no Math.random in a render
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 14;
  }
  ctx.putImageData(img, 0, 0);
  return `background-image:url(${c.toDataURL()});background-repeat:repeat;opacity:0.5;mix-blend-mode:multiply`;
}
```

`handle.setTheme` + `MountOptions.theme`:

```ts
    setTheme(name: string): void {
      themeName = name;
      applyStageStyle();
      rebuild({ ...currentSpec, theme: paletteFor(currentSpec, name) });
    },
```

Careful: `paletteFor(currentSpec, name)` must be computed against the ORIGINAL palettes (pass `currentSpec` un-mutated, as written) and the rebuilt spec's `theme` becomes the painted palette — but keep `palettes` on the spec so switching back works. Also thread `themeVariant` into `materialFor` at the paint call site (Step 3 note above).

Font links — append to `<head>` of `frontend/index.html` and `app/static/cell_scene/index.html` (and inside the HTML string built by `app/_scene_view.py`):

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=EB+Garamond&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
```

The palettes' `fonts` stacks are the offline fallback — no font binaries ship.

- [ ] **Step 4:** Run Node gate (the new `spec.test` additions PASS) + freshness gate.

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/scene/ frontend/index.html app/static/cell_scene/ app/_scene_view.py
git commit -m "feat(scene): switch palettes from the document with per-variant lighting, chrome and grain"
```

**Phase 4 ends here.**

---

## PHASE 5 — Cockpit HUD

### Task 12: `hud.ts` — rail, presets, hotkeys, telemetry, breathe, persistence

**Files:**
- Create: `frontend/src/scene/hud.ts`
- Modify: `frontend/src/scene/engine.ts` (springs for explode/peel, breathe, presets, hotkeys, telemetry, HUD wiring), `frontend/src/scene/index.ts` (handle methods), `frontend/src/scene/geometry.ts` (`mergeBuildOptions` only if not already folded — Tasks 1/3 did it)

**Interfaces:**

```ts
// hud.ts (DOM)
export interface HudState {
  exploded: number; peel: number; layout: "wound" | "unrolled";
  cursor: number; measuredCount: number; cycle: number | null;
  soh: number | null; projected: boolean; playing: boolean;
  annotations: boolean; breathe: boolean; themeName: string | null;
  availablePalettes: string[];
}
export interface HudCallbacks {
  setExploded(v: number): void; setPeel(v: number): void;
  setLayout(l: "wound" | "unrolled"): void;
  toggleBreathe(): void; toggleAnnotations(): void;
  playPause(): boolean; scrub(cursor: number): void;
  preset(name: "iso" | "plan" | "section" | "unrolled"): void;
  setTheme(name: string): void; frameActive(): void;
}
export interface Telemetry { fps: number; calls: number; triangles: number; vertices: number; }
export interface HudHandle {
  update(state: HudState): void;
  setTelemetry(t: Telemetry): void;
  collapsed(): boolean;
  toggleCollapsed(): void;
  dispose(): void;
}
export function createHud(container: HTMLElement, state: HudState, cb: HudCallbacks): HudHandle;
// readPref/writePref are IMPORTED from ./theme.ts (defined in Task 11) — the HUD
// never declares a second copy.
```

- [ ] **Step 1: Build the rail (failing first is structural here — assert via engine tests that exist: `spec.test.ts` mounts the real sample; if `createHud` throws, that test fails)**

Write `createHud` with this exact inventory (one absolutely-positioned rail along the stage bottom, `pointer-events:auto`, all colours from `state`'s palette via a `theme` argument you add to `HudState` — add `theme: SceneTheme` to `HudState` and refresh it in `update`):

```ts
export function createHud(container: HTMLElement, state: HudState, cb: HudCallbacks): HudHandle {
  const theme = state.theme;
  const rail = document.createElement("div");
  rail.className = "cell-scene-hud";
  rail.style.cssText =
    "position:absolute;left:12px;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:8px;" +
    `align-items:center;padding:8px 10px;border-radius:6px;background:${theme.panel}ee;` +
    `border:1px solid ${theme.grid};color:${theme.text};font-family:${theme.fonts?.mono ?? "ui-monospace,monospace"};font-size:11px;z-index:4;`;

  const btn = (label: string, title: string, hotkey: string | null, onClick: () => void) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = hotkey ? `${title}  [${hotkey}]` : title;
    b.style.cssText =
      `background:${theme.grid};color:${theme.text};border:1px solid ${theme.muted}44;border-radius:4px;` +
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
  const slider = (min: number, max: number, step: number, value: number, onInput: (v: number) => void, width = "110px") => {
    const s = document.createElement("input");
    s.type = "range"; s.min = String(min); s.max = String(max); s.step = String(step);
    s.value = String(value);
    s.style.cssText = `width:${width};accent-color:${theme.accent}`;
    s.addEventListener("input", () => onInput(Number(s.value)));
    return s;
  };
```

Groups, in order on the rail:

1. **Explode**: label `EXPLODE` + slider `0…1` step `0.01` → `cb.setExploded` + mm readout `<span data-mm>` (updated in `update()`: `mm = exploded × terminalPos lift mm` computed as `exploded * (CELL_GEOMETRY.explode.terminalPos * physicalMmPerUnit)` where `physicalMmPerUnit = spec.physical?.unitsMmPerCellUnit ?? 65` — pass `mmPerUnit` and `radialMmMax` in `HudState` instead: `HudState.mmMaxAxial`, `mmMaxRadial`, rendered as `+{v*axial|1} mm · {v*radial|2} mm`), buttons `[ASSEMBLED]` (explode 0), `[EXPLODED 100%]` (1), hotkeys none (E covers toggle at engine level).
2. **Peel**: label `PEEL` + slider `0…1` step `0.01` with a visible `¼` detent (a `<span>` tick label positioned at 25%) + button `[CUTAWAY · ¼ PEEL]` → `cb.setPeel(0.25)` (engine treats same-value as toggle back to default — see engine wiring), hotkey `C`.
3. **Breathe**: `[BREATHE]` toggle, hotkey `B`... **your spec lists only `[E][C][A][H][Space]` as badge hotkeys** — render Breathe WITHOUT a badge, clicking only.
4. **Lifecycle**: `[▶]` button hotkey `Space` → `playing = cb.playPause()`; scrub slider `0…measuredCount` step 1 → `cb.scrub`; readout `cyc {cycle} · soh {soh}% {projected ? "PROJECTED" : ""}`.
5. **Layout**: `[WOUND]` / `[UNROLLED]` → `cb.setLayout`.
6. **Camera**: `[ISO] [PLAN] [SECTION] [UNROLLED]` → `cb.preset(...)` (the UNROLLED preset also sets layout — engine side).
7. **Theme** (only when `state.availablePalettes.length > 1`): `[CODEX] [OBSIDIAN]` → `cb.setTheme("codex" | "obsidian")`.
8. **Telemetry**: `<span data-tel>` right-aligned with `flex-grow` — `update()` renders `60 fps · 24 calls · 184k tris · 96k verts`.
9. Collapse chevron `‹` at the rail's right end → toggles `railCollapsed` (persist `writePref("hudCollapsed", true)`), leaving a single `[⋯]` pill that restores it. Hotkey `H` handled at engine level calls `handle.toggle()` — expose `toggleCollapsed(): void` on `HudHandle`.

`update(state)` writes slider positions, mm text, scrub position, playback label, telemetry span; `setTelemetry` stores the last numbers for `update` to render; both must be cheap (they run at 4 Hz from `tick`).

- [ ] **Step 2: Engine wiring — springs, breathe, presets, hotkeys, telemetry**

Inside `mountCellScene` (engine.ts):

```ts
  // --- view springs: every explode/peel change is a target, never a jump ----
  let explodeTarget = build.exploded;
  let peelTarget = build.peel;
  let explodeSpring: Spring = { value: build.exploded, velocity: 0 };
  let peelSpring: Spring = { value: build.peel, velocity: 0 };
  let lastRebuild = 0;
  let breathe = false;
  let breathePhase = 0;
  let lastTick = performance.now();
```

`applyOptions` (line ~181): when the patch carries `exploded`/`peel`, set `explodeTarget`/`peelTarget` (clamped 0…1) instead of assigning `build.*` directly; merge everything else as today. (Add a comment: sliders retarget, the spring chases — this is why buttons animate.)

In `tick()` (line 696), before render:

```ts
    const now = performance.now();
    const dtMs = Math.min(100, now - lastTick);
    lastTick = now;

    explodeSpring = springStep(explodeSpring, explodeTarget, dtMs);
    peelSpring = springStep(peelSpring, peelTarget, dtMs);
    const moved =
      Math.abs(explodeSpring.value - build.exploded) > 1e-3 ||
      Math.abs(peelSpring.value - build.peel) > 1e-3;
    if (moved && now - lastRebuild >= 8) {
      lastRebuild = now;
      build.exploded = explodeSpring.value;
      build.peel = peelSpring.value;
      rebuild(currentSpec);   // existing rebuild — reads build options from its closure (verify name)
    }

    if (breathe) {
      breathePhase += dtMs / 3200;                 // 3.2 s cycle
      partObjects.forEach((p, i) => {
        const s = 1 + 0.012 * Math.sin((breathePhase + i * 0.11) * Math.PI * 2);
        p.group.scale.setScalar(s);
      });
    }
    // camTween application from Task 6 stays here.
    // telemetry sampling every 4th frame:
    frameCount++;
    if (frameCount % 4 === 0) {
      hudHandle?.setTelemetry({
        fps: Math.round(1000 / Math.max(1, emaDt)),
        calls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
        vertices: paintedVertices,   // computed once in paint(): sum of vertexCount per drawn mesh
      });
      hudHandle?.update(hudState());
    }
    renderer.render(scene, camera);
```

(`emaDt` = `emaDt = emaDt * 0.9 + dtMs * 0.1` at tick top. `paintedVertices` computed in `paint()`. `build` here means the live `BuildOptions` object the engine holds — use the actual closure names it uses in `applyOptions`/`rebuild`; the shapes above are exact, the identifiers bend to the file.)

`hudState()` assembles `HudState` from `build`, `state` (FrameState), `currentSpec`, palettes, prefs.

Camera presets:

```ts
  const PRESETS: Record<"iso" | "plan" | "section" | "unrolled", () => void> = {
    iso: () => framePose({ azimuth: Math.PI / 4, elevation: 0.38, distance: fitDistance(), targetY: 0 }, 600),
    plan: () => framePose({ azimuth: Math.PI / 4, elevation: 1.52, distance: fitDistance() * 0.9, targetY: 0 }, 600),
    section: () => {
      explodeTarget = 0.35; peelTarget = 0.25;          // springs animate the disassembly
      framePose({ azimuth: Math.PI * 1.6, elevation: 0.14, distance: fitDistance(), targetY: 0 }, 600);
    },
    unrolled: () => {
      setBuildOption("layout", "unrolled");
      framePose({ azimuth: 0, elevation: 0.21, distance: UNROLL_LENGTH * 1.6, targetY: 0 }, 600);
    },
  };
```

`fitDistance()` = the value `fitView` computes today (extract it: `fitView` = `framePose` with the same distance/elevation, non-animated or animated — your call, `resetView` prefers the 600 ms tween).

Hotkeys — scoped per spec §7:

```ts
  let pointerOverStage = false;
  container.tabIndex = 0;
  container.addEventListener("pointerenter", () => { pointerOverStage = true; });
  container.addEventListener("pointerleave", () => { pointerOverStage = false; });
  container.addEventListener("keydown", (event) => {
    const focused = container.contains(document.activeElement);
    if (!focused && !pointerOverStage) return;
    const onButton = event.target instanceof HTMLElement && event.target.tagName === "BUTTON";
    const key = event.key.toLowerCase();
    const owned = [" ", "e", "c", "a", "h", "escape"];
    if (event.key === " " && onButton) return;   // let the focused button take Space natively
    if (!owned.includes(key)) return;
    event.preventDefault();
    switch (key) {
      case " ": playing = playing ? (pause(), false) : play(); hudHandle?.update(hudState()); break;
      case "e": explodeTarget = explodeTarget > 0.5 ? 0 : 1; break;
      case "c": peelTarget = Math.abs(peelTarget - 0.25) < 1e-6 ? DEFAULT_PEEL : 0.25; break;
      case "a": setAnnotations(!annotationsVisible); break;
      case "h": hudHandle?.toggleCollapsed(); break;
      case "escape": inspect(null); break;
    }
  });
```

Handle additions (on `CellSceneHandle`, wired in the returned object):

```ts
    setTheme(name: string): void { /* Task 11 body */ },
    framePreset(name: "iso" | "plan" | "section" | "unrolled"): void { PRESETS[name](); },
    telemetry(): Telemetry { return lastTelemetry; },
```

`index.ts`: export `Telemetry` type; `MountOptions` already extended with `theme?: string` in Task 11.

HUD creation (after overlays exist):

```ts
  const hudHandle = createHud(container, hudState(), {
    setExploded: (v) => { explodeTarget = Math.max(0, Math.min(1, v)); },
    setPeel: (v) => { peelTarget = Math.max(0, Math.min(1, v)); },
    setLayout: (l) => { rebuildWith({ layout: l }); },          // merge + rebuild, immediate (no spring needed)
    toggleBreathe: () => { breathe = !breathe; if (!breathe) partObjects.forEach((p) => p.group.scale.setScalar(1)); },
    toggleAnnotations: () => setAnnotations(!annotationsVisible),
    playPause: () => (playing ? (pause(), false) : (play(), true)),
    scrub: (c) => setCursor(c),
    preset: (name) => PRESETS[name](),
    setTheme: (name) => themeHandle(name),
    frameActive: () => { if (pinnedId) inspect(pinnedId); },
  });
  if (readPref("hudCollapsed") === true) hudHandle.toggleCollapsed();
```

Dispose: `hudHandle.dispose()` alongside the existing listener teardown (also remove the `keydown`/`pointerenter`/`pointerleave` listeners — mirror how `resize` removal is done).

`setBreathe`/`setPeel` via handle are NOT required (HUD-only), but `update(spec, options)` must keep working: if the patch carries `layout`, `peel`, `exploded`, route through the same targets/immediates.

- [ ] **Step 3: Verify**

Run: `cd frontend; npm run test:scene` (must include the existing mount test — a throwing `createHud` fails it), `npm run lint`, `npm run build`. Then freshness gate.

Manual characterisation on the standalone page (serve `app/static/cell_scene/index.html` or `npm run preview`): sliders animate (spring), `Space`/`E`/`C`/`A`/`H`/`Esc` work only while the stage has focus/hover, presets tween and a drag cancels them, theme buttons repaint stage+HUD+card together, telemetry numbers move.

- [ ] **Step 4: Commit**

```powershell
git add frontend/src/scene/ app/static/cell_scene/
git commit -m "feat(scene): cockpit HUD with springs, camera presets, scoped hotkeys and telemetry"
```

**Phase 5 ends here.**

---

## PHASE 6 — Artifacts & docs

### Task 13: final verification, docs to shipped state, close out

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `METHODOLOGY.md`, `docs/history.md`, `docs/scene_prompt.md`, `app/static/cell_scene/README.md`, `app/_pages/battery3d.py`
- Artifacts: `app/static/cell_scene/cell_scene.js`, `sample_scene.json`, `manifest.json` (regenerated)

- [ ] **Step 1: Rebuild artifacts from final sources**

```powershell
cd frontend; npm run build:scene; cd ..
python scripts/export_scene_sample.py --cell B0005
```

- [ ] **Step 2: Full verification — record actual numbers**

```powershell
cd frontend; npm run test:scene; npm run lint; npm run build; cd ..
pytest tests/test_cell_scene.py tests/test_cell_scene_api.py tests/test_cell_scene_bundle.py tests/test_cell_scene_page.py
pyright
mkdocs build --strict
```

Expected: all PASS/exit 0. Record the ACTUAL test counts (Node was 76, Python 101 before this work; targets ~110/~125) — the docs below must quote the real numbers, not the targets. If anything fails, fix it in its source (never by weakening an assertion about an invariant in Global Constraints).

- [ ] **Step 3: Docs → shipped state** (quote the numbers from Step 2 everywhere)

- `README.md`: "16 anatomy parts" → "19 anatomy parts"; wherever it describes the annotation/dossier/theme, add the shipped specifics (two-row callouts with provenance dots, peel cut-away, Codex/Obsidian palettes carried in the document, cockpit HUD with scoped hotkeys).
- `CHANGELOG.md`: new entry under today's date covering: 19 parts (gasket/CID+PTC/bottom-insulator), peel as a view control (default unchanged), unrolled layout with printed compression, pure flank solver + two-row callouts, document-carried dossiers with two-tier tags, dual palettes with pinned thresholds, cockpit HUD (presets/hotkeys/telemetry/breathe).
- `docs/history.md`: one new dated row (imperative, present tense, ≤ 14 words): e.g. `| 2026-09-21 | Winds the cell's story into a codex: dossier, palettes, cockpit |`.
- `docs/scene_prompt.md`: extend the contract list with: peel default = historic 285° (test-pinned), `category` taxonomy (ten words), dossier `tag` vocabulary + refusal rule, palettes threshold-equality, hotkey scoping (stage-only), unrolled compression must print. Add mapping rows for the three new part ids. Update any part-count language (16 → 19).
- `app/static/cell_scene/README.md`: quote the NEW raw byte size and gzip size of `cell_scene.js` (measure: `(Get-Item app/static/cell_scene/cell_scene.js).Length` and `gzip -9 -c ... | wc -c`-equivalent in PowerShell — use the same method the README already documents), refresh the release checklist items this work changed (dossier/palette/HUD items become checked behaviours), add one new checklist item: "Hotkeys act only while the stage has focus or hover".
- `app/_pages/battery3d.py`: "all sixteen parts" → "all nineteen parts" (docstring/prose only; AppTest must stay green — Step 2's page suite proves it).
- `METHODOLOGY.md`: update the Node/Python test-count sentences to the Step 2 numbers; add the concentric/peel invariant sentence if counts are mentioned beside it.
- Repo-wide grep for stale counts before committing: `Select-String -Path README.md,CHANGELOG.md,METHODOLOGY.md,docs\*.md,app\_pages\battery3d.py -Pattern "sixteen|16 anatomy|15 cards|74 Node"` — every hit must be either history (leave it) or updated.

- [ ] **Step 4: Re-run gates after doc edits** (docs affect `mkdocs build --strict` and possibly page tests)

```powershell
pytest tests/test_cell_scene_page.py
mkdocs build --strict
```

- [ ] **Step 5: Commit and push**

```powershell
git add README.md CHANGELOG.md METHODOLOGY.md docs/ app/static/cell_scene/ app/_pages/battery3d.py
git commit -m "feat(scene): museum-grade redesign — 19 parts, dual palettes, dossier, cockpit HUD"
git push origin master
```

(If the user has asked for separate commits per docs file, split accordingly; the default here is one close-out commit as the spec's Phase 6 describes.)

---

## Self-Review (executed by the plan's author)

**1. Spec coverage:** §3.1 → Task 1 + 12 (peel, breathe); §3.2 → Task 12 (springs, 8 ms throttle) with the fallback documented in spec §9 (not built — correct, it is a fallback); §3.3 → Task 2 (all three parts, exhaust ports as cap detail = no task, per spec); §3.4 → Task 3; §3.5 → Task 12 (mm readout, quick actions); §4.1/4.2 → Task 4; §4.3 → Task 6; §4.4/4.5 → Tasks 9 and 6; §5 → Tasks 7/8/9; §6 → Tasks 10/11 (incl. font links, chrome, grain, vignette, material adjust, persistence via `readPref`/`writePref` in Task 12 — persistence note: prefs helpers are created in Task 12 but used by Task 11's `setTheme`; **fix: move `readPref`/`writePref` into Task 11's theme.ts-side helpers or implement them at the bottom of theme.ts in Task 11 and import into Task 12** — done: implement `readPref`/`writePref` in `theme.ts` during Task 11 and re-export; Task 12 imports them); §7 → Task 12 (every group, preset parameter table matches spec §7, hotkey table matches spec exactly); §8 → per-task gates + Task 13; §9 → risks addressed inline (throttle, rim no-op fallback, offline fonts, crowded flanks, additive schema, host duplication accepted); §10 → phases map to task groups.

**2. Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to Task N". Two location instructions reference exact line numbers and say what to do at each (not placeholders). The `_DOSSIER_TABLE` migration instruction is content-complete: exact rows for all 19 parts are in the table, with an explicit verbatim-migration rule for React's prose.

**3. Type consistency:** `peelSweepDeg`/`DEFAULT_PEEL` (T1) used by T12 key `C` — consistent; `layoutFlank` signature identical between annotation.ts and the engine call — consistent; `composeDossier(spec, partId, cursor)` matches T8 definition and T9 call; `paletteFor` T11 definition matches T10's `palettes` shape; `HudState` fields referenced by engine (`mmMaxAxial`...) are declared in the Interfaces block — consistent; `springStep`/`Spring` from T5 match T12 usage; `UNROLL_LENGTH` exported in T3, used in T12 preset — consistent; `setTheme` declared in T11 and re-used in T12's handle list — one definition, T11's body.
