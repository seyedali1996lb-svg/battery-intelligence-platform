# The 3D Cell View — the Prompt

This is the standing prompt for the scene: what to build, what "accurate" means,
what "simple" means, and how to tell whether a change made it better. Rewrite it
freely, but keep the three contracts — data, geometry, honesty — because they are
what makes the view trustworthy rather than merely pretty.

---

## The prompt

> Build the 3D view of a battery cell so that a person who has never seen inside
> one can look at it for thirty seconds and correctly answer: *what is this part,
> what is it for, and what do we know about it?* Then make it accurate enough
> that an engineer never has to unlearn anything the picture taught them.
>
> **Shape — the cell is drawn from millimetres, not from taste.**
> Every drawn size comes from the document's `physical` block, in millimetres,
> each field tagged with where it came from: measured, format-standard, derived,
> or declared-schematic. The wound roll fills the can because a real winding has
> nowhere else to go — derive the turn count from the stack pitch and the bore,
> not from how many turns look nice. The top of the cell is a *formed* part: a
> lathed cap plate with its boss, a vent disc sitting in the boss, a button on
> top, the crimp bead rolled proud of the wall, and the heat-shrink jacket
> wrapped to within a millimetre of each end, opened by the same cut-away that
> opens the can. The tabs are leads, not stickers: each leaves the coil's own
> end and rises to the terminal it feeds. Nothing drawn may leave the cell's
> declared envelope at any explode position — not because a clamp says so, but
> because the arithmetic that sizes the parts leaves no room to.
>
> **Light — the materials must name themselves.**
> Nickel-plated steel, aluminium, copper, graphite, polyethylene and liquid
> electrolyte do not look alike, so they must not be shaded alike: a per-part
> material table (metalness, roughness, environment response), filmic tone
> mapping so highlights hold their shape, an environment map so metal reflects
> something, and one shadow-casting key light so the object sits on a floor
> rather than floating in fog. A screenshot should be readable *as materials*
> before a single label is read. The camera frames the part from its own
> bounds — a prismatic cell is not a stretched cylinder and must not be framed
> like one.
>
> **Honesty — the drawing is a claim, so every claim is labelled.**
> Data-driven geometry (the film's nanometres, the gauge, the narrowed
> terminal) is opt-in and annotated. Where a drawn size is exaggerated for
> legibility, the magnification is printed, not implied — a 1,600 nm film drawn
> 300,000× says so on the card. Where the data cannot support a drawing, the
> part is drawn *with its reason*, never silently zeroed. The colour of a part
> is its material's colour; the colour of a *claim* is the app's own band
> tokens, and the two never mix.
>
> **Annotation — the drawing is never covered.** A part's label is not
> painted on the part. Its anchor is projected to the viewport, marked with a
> reticle, and joined by a single dog-leg leader line to a callout badge pinned
> at the stage's own margin — fixed columns left and right, whatever the camera
> does — so the picture stays the picture. Selecting a badge or a mesh opens
> that part's dossier beside the canvas (name, subsystem, material, failure
> mode, camera focus); the hosts show one dossier, not a grid of cards. Text is
> escaped before it is drawn, a refused value shows its reason instead of a
> zero, and everything else that dims does so at one declared opacity.
>
> **Simplicity — one question per view, plain words on every card.**
> The default view answers "what is this?"; the exploded view answers "how is
> it built?"; the data-scaled view answers "what is happening to it?". Part
> cards lead with a plain-language title ("The vent — the cell's safety
> valve"), never with an identifier. Any part a card cannot explain in one
> sentence is either the scene's fault or the sentence's. No control without an
> effect, no number without a unit, no colour without a meaning.
>
> **The test.** Hand the rendered scene to someone new. If they can name the
> casing, the winding, the separator and the vent — and say which part the data
> says is aging — the view is doing its job. If they ask "is that to scale?",
> the answer is already on the screen.

---

## How this maps to the code

| Rule | Where it lives |
| --- | --- |
| Millimetres with provenance | `physical` block in `src/cell_scene.py`; `docs/cell_scene.schema.json` |
| Roll derived from the stack | `rollModel()` + `drawnRoll()`, `frontend/src/scene/geometry.ts` |
| Independent radial offsets on explode (concentric members, envelope by construction) | `explode.radial` → `radialOffsets()` + `drawnRoll()`, same file |
| The core as anatomy (`mandrel`, part 16 of 19) | `MESH_PART_IDS` in `src/cell_scene.py`; placement in `geometry.ts` |
| The gasket under the crimp (part 17 of 19) | `gasket` in `src/cell_scene.py`; `CELL_GEOMETRY.explode.gasket`, `geometry.ts` |
| The CID/PTC safety pair (part 18 of 19) | `cid_ptc` in `src/cell_scene.py`; `CELL_GEOMETRY.explode.cidPtc`, `geometry.ts` |
| The bottom insulator (part 19 of 19) | `bottom_insulator` in `src/cell_scene.py`; `CELL_GEOMETRY.explode.bottomInsulator`, `geometry.ts` |
| Peel default = the historic 285° cut, test-pinned (wound output byte-identical under any view options) | `DEFAULT_PEEL` + `peelSweepDeg()`, `geometry.ts`; asserted in `geometry.test.ts` |
| Unrolled layout prints what the compression cost | `layout: "wound"/"unrolled"` + `unrollNote`, `geometry.ts` |
| Exhaust ports drawn as cap detail, never a part | six scored slots merged into `cap`, `geometry.ts`; the cap dossier's refusal row, `src/cell_scene.py` |
| Leader lines to the stage margin, two-row badges in two fixed columns | flank solver + card heights in `frontend/src/scene/annotation.ts` (`layoutFlank`, `CARD_H_ONE`/`CARD_H_TWO`, `CATEGORY_TAXONOMY`); `BADGE_MARGIN`, `DIM_OPACITY` in `engine.ts` |
| The dossier, carried by the document (one reading surface per part) | `dossier` block + `_DOSSIER_TABLE`, `src/cell_scene.py`; `composeDossier()`, `frontend/src/scene/dossier.ts`; floating card in `engine.ts` |
| Dossier tag vocabulary + refusal rule (`typical`/`measured`/`derived`/`fitted`/`refusal`; ≥ 4 rows per part; reasons, never zeros) | `_DOSSIER_TABLE` + `composeDossier()`, same files |
| Two palettes with pinned band thresholds (colour may change, health may not) | `palettes.codex`/`palettes.obsidian`, `src/cell_scene.py`; threshold-equality test, `tests/test_cell_scene.py`; `paletteFor()`, `theme.ts` |
| Cockpit HUD: presets, transport, telemetry, breathe, collapse | `frontend/src/scene/hud.ts`; springs + `PRESETS` in `engine.ts` |
| Hotkeys act only while the stage has focus or hover | `onStageKey`, `frontend/src/scene/engine.ts` |
| Formed top (cap, boss, vent, button, crimp, wrap) | `topAssemblyModel()` + `latheMesh()` profiles, same file |
| Tabs as leads | `tabStripMesh()`, same file |
| Material table | `frontend/src/scene/materials.ts` |
| Tone mapping, environment, key light | `frontend/src/scene/engine.ts` |
| Bloom on the emissive gauge, tone-mapped exactly once at the end | `RenderPass → UnrealBloomPass → OutputPass` in `frontend/src/scene/engine.ts`; per-palette strength/radius/threshold in `applyStageStyle()`; chain pinned in `render.test.ts` |
| Plain-language titles | `title` on every part, produced in `src/cell_scene.py` |
| Data-scaled geometry as an opt-in, annotated | `dataScaled` build option + `geometryScales` in the document |
| Magnification printed | `physical.film.display` (drawn band + factor) |
| Refusals drawn with reasons | `unavailableReason` / `reason` per part |

## What guards it

* The **schema test** (`spec.test.ts`) reads the committed `sample_scene.json`
  and holds the Python producer and the JS renderer to one document.
* The **shape tests** pin the physics of the drawing: the roll inside its
  envelope at every explode position, the five concentric members' offsets zero
  at rest and ordered at full explode with air between mandrel and anode, the
  bead/jacket/can stacking in that order, declared millimetre sizes equal to the
  drawn ones, tabs attached at both ends, the prismatic stack confined by
  construction.
* The **honesty tests** pin the epistemics: a refused split draws no film
  growth, the fitted share cannot move the drawn film, every part card carries
  a provenance tag, the magnification is printed where the film is drawn.
* The **bundle test** keeps the committed artifacts the hash of their sources.

## The standing backlog (what "10/10" still owes)

1. **Top-assembly millimetres are format-typical, not datasheet** — the block
   says so in `provenance`, but a datasheet-ingest path would move the whole
   top of the cell from "typical" to "this cell".
2. **The screenshot gate** — hover/dim/explode/playback behaviour is still
   checklist-verified; goldens from a headless render would make it a test.
3. **Paint the diagnosis on the object** — the fitted fade already has a
   thickness (the film); the active-material share could darken the anode's
   graphite the same way, making both fitted channels visible on the parts
   they belong to.
4. **Portability** — the document is self-contained and versioned; a single
   offline HTML file with a scene baked in would make a battery passport that
   opens anywhere.
