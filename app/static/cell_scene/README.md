# The 3D cell scene — build artifacts

Four files, all committed on purpose. This directory is what a **browser** loads:
Streamlit serves `app/static/` straight from disk (`server.enableStaticServing`,
`.streamlit/config.toml`), and the API mounts this same directory at `/scene`.
Neither host can run Node, so the renderer has to already exist here.

| File | What it is |
| --- | --- |
| `cell_scene.js` | The whole scene engine, minified, three.js and SVG leader-line overlay included (~647.6 kB, ~167 kB gzipped). Built from `frontend/src/scene/` — one implementation, three hosts. |
| `index.html` | The standalone host: no framework, no build step. Loads the bundle above and a scene document, and does everything the Streamlit page and the React SPA do. |
| `sample_scene.json` | A real `CellSceneSpec` for a real cell, so the page works in a fresh checkout with no data loaded and no API running. Also read by `frontend/src/scene/spec.test.ts`, which is how the Python producer and the JavaScript renderer are held to the same document — including the `physical` block, whose derivations (turn count, pitch, implied electrode length) are recomputed by the renderer and compared against the producer's, and the film's chain, whose nm-per-%-lithium-inventory factor the test re-derives from the tagged assumptions and whose **drawn band the renderer reads out of the document** rather than choosing for itself. |
| `manifest.json` | The hashes that make a stale or hand-edited bundle impossible to commit — **and the cache key**: `bundleSha256[:12]` is what both hosts append to the bundle URL as `?v=`, so shipping a new bundle always ships a new URL and a browser cache can never pin an old renderer. Digests are taken over line-ending-normalized bytes, so the manifest verifies identically on a CRLF (`core.autocrlf`) checkout and on the LF checkout CI hashes. |

## Regenerating (after any change under `frontend/src/scene/`)

```bash
cd frontend
npm ci
npm run test:scene      # node --test src/scene/*.test.ts — the geometry and the
                        # contract with src/cell_scene.py (129 tests)
npm run build:scene     # → ../app/static/cell_scene/cell_scene.js

cd ..
python scripts/export_scene_sample.py --cell B0005   # refreshes the sample + manifest
                                                    # and stamps index.html's <script> with ?v=
python -m pytest tests/test_cell_scene_bundle.py     # or: export_scene_sample.py --check
```

The build is deterministic in the ways that matter (fixed output filename, no
hash in the name) and the manifest records the digest of the bundle **and of
every source file it was built from** — so "I changed the geometry but forgot to
rebuild" fails a test rather than shipping a scene that disagrees with its own
source. The export script also stamps this page's `<script src="cell_scene.js?v=…">`
tag with the manifest's digest and validates the pair in `--check`; the Streamlit
host reads the same `bundleSha256` to build its own URL (`app/_scene_view.py`).
`npm run build` (the SPA) is a separate build of the same sources; it
serves the scene from `frontend/dist` and does not touch this directory.

## Opening the standalone page

* From the Streamlit app: `http://localhost:8501/app/static/cell_scene/index.html`
* From the API: `http://localhost:8000/scene/index.html`
* Any static server: `python -m http.server` in this directory

Query parameters:

| Parameter | Effect |
| --- | --- |
| `?cell=B0005` | Fetch that cell's scene from the REST endpoint (`GET /cells/{id}/scene`) |
| `?api=https://host` | API base URL for `?cell=` (default: same origin) |
| `?token=…` | Bearer token for the REST endpoint — it is authenticated |
| `?spec=./file.json` | Load any URL that serves a `CellSceneSpec` |
| *(nothing)* | Falls back to `sample_scene.json` |

**View state** (the share button is the address bar — these are written back as you
interact, and read on load; precedence is **URL > the stored view of this cell on this
machine > the host's default**, and the loader keys above survive both trips because
one half of the URL says *which* cell and the other *how to look at it*):

| Parameter | Effect |
| --- | --- |
| `?cursor=150` | Life-cursor index (measured + projected cycles); default when unset = last measured cycle |
| `?exploded=0.7` / `?peel=0.4` | Explode / peel position, 0–1 |
| `?layout=unrolled` | Wound or unrolled geometry |
| `?part=anode` | The **pinned** part (a click) — a transient hover never enters the URL |
| `?annotations=0` | Hide the badge/leader-line layer |
| `?theme=codex` | Palette; otherwise the reader's stored choice, else the document's own |

A `file://` double-click cannot `fetch` a sibling file, so the page has no data
path in that mode — serve the directory over HTTP. That is a browser rule, not a
restriction of this page.

## Verifying the render with no compositor

A WebGL canvas whose drawing buffer is not readable cannot be screenshotted — and
in a headless or offscreen webview it may produce no frames at all. The frame is
still readable from inside the page: mount the scene, wait two
`requestAnimationFrame` callbacks (the engine draws in its own), then
`gl.readPixels(...)` on the canvas' `webgl2`/`webgl` context **in the same task**.
The browser clears the buffer at composite time, so reading after a frame in a
later task returns a blank image, which is how a working scene reads as an empty
one. That is how the layout, the explode separation and the per-state differences
were checked during development — the pixels are the evidence, and it is worth
re-running before trusting a change that only the eye is supposed to catch.

## Manual visual checklist

There is no headless browser in this repository's CI, so the numbers, the
geometry and the serialized document are what the automated gates verify. The
things only an eye can catch are below; run through it after any change to
`frontend/src/scene/` or to the theme in `src/cell_scene.py`. Open the harness
page with `?cell=B0005` (or another real cell) and check:

1. **The cell is closed.** Orbit under and behind it: no hole where the casing
   should be, no face that vanishes as you cross behind it. (Casing is drawn as a
   cut-away arc — the *shape* is open, but nothing should flicker or disappear.)
2. **The roll reads as a roll.** At rest it is a *dense* winding — 38 laps of a
   0.175 mm stack, each layer a fraction of a pixel wide, so it should look like
   the tight spiral a jelly roll is and not like a spring with air in it. As you
   drag "Exploded view" the stack magnifies and the lanes open between the five
   concentric members (casing, mandrel, anode, separator, cathode), so with the
   slider at 100% you can count the layers. Nothing may touch or pass through
   the casing at any explode position — that is arithmetic in the model, not a
   clamp, so if it happens the model is wrong.
3. **Drag "Exploded view" to 100%.** The cap, vent and positive terminal lift in
   order; the can and the mandrel core *hold* while the anode, separator and
   cathode peel outward on their own radial offsets; and everything stays inside
   the can.
4. **Scrub the life cursor.** The state gauge on the casing fills as SOH falls;
   the part values and their colours change; particles do **not** teleport. The
   **Anatomy list on the right must move with the scene** — those rows come from
   `handle.parts()`, which resolves each part's reading at the scene's own
   cursor, so a row that keeps its value while the slider moves is the bug that
   API exists to prevent.
5. **Press "Play life".** The gauge drains smoothly and stops at the end of the
   timeline — including at the end of the projection, when there is one — and the
   button returns to "▶ Play life" when it does (the engine reports the end via
   `onPlaybackEnd`; it cannot relabel a button that belongs to the host).
   Pressing it at the very end replays from the first cycle rather than doing
   nothing.
6. **Past "today".** On a cell whose forecast routing allows a projection the
   gauge keeps moving and the cycle label says "projected"; on a cell it refuses,
   the timeline simply ends and the note in the disclosures explains why.
7. **Hover a part.** Its badge and leader line light up at the stage margin and
   every other annotation dims — no label is ever drawn over the meshes. The
   badge's tooltip carries the label, provenance and unit; clicking it (or the
   mesh) inspects the part. Clicking a row in the "Anatomy" list turns the
   camera to that part's side and dims the rest.
8. **Toggle "Data-scaled geometry".** The caption under the canvas prints the
   mapping in use. For the film that means the whole chain and its magnification:
   the nm-per-%-LLI factor, the drawn band, the share of initial capacity the
   band's top represents, and the factor the drawing magnifies by (a cell at
   20% lithium-inventory loss reports ~1650 nm drawn at ~190×). The film's card
   reads in **nanometres** — scrub the cursor and it moves (156 nm at cycle 1 to
   1647 nm at the end of a long record) — and it only grows when the split is
   identified: an unidentified cell prints the *total fade* mapping instead, says
   the thickness is withheld, and leaves the film as architecture.
9. **Toggle "Strip the casing".** The shell goes, the cell stays readable, and
   nothing else changes.
10. **Colours agree with the rest of the app, and a palette never moves a band.** A cell the
    Streamlit UI calls "Degrading" is amber on the casing gauge and in the legend here too.
    Switch palettes from the HUD (Codex ⇄ Obsidian): stage, lights, chrome, dossier card and
    rail repaint together and the choice is remembered — while the SOH band thresholds, what
    counts as "Degrading", stay pinned identical across both.
11. **The film is a thickness before it is a drawing.** The card's number is
    derived from the fitted lithium loss through stated assumptions (Li₂CO₃ at
    its bulk density over the anode's own coated area), and the drawn layer is a
    magnification of it — stated, never left to be read off the picture. If the
    card ever shows a number while the disclosures say the √n channel is not
    identified, or the drawn film moves when you edit `series.seiPct` but not
    `series.seiThicknessNm`, that is a defect: the card and the geometry are
    supposed to be the same number.
12. **It is lit, not just drawn.** The can and the foils should carry a
    reflection (there is a procedural environment behind them), the highlights
    should roll off rather than clip to white, and the cell should cast a soft
    shadow onto the stage. Watch the console: three removes APIs between minor
    versions while still exporting their constants, so a warning here is a real
    defect and there is a test (`src/scene/render.test.ts`) pinning the ones the
    renderer is allowed to name.
13. **The top of the cell is a formed part.** At the rim you should see the
    crimp bead standing slightly proud of the jacket, the cap plate seated over
    the can with its boss raised around the vent, and the button on top of the
    boss — not a stack of flat discs. The two tabs leave the coil's outermost
    turn and bend over the cap to the terminals they feed; at full explode they
    stay attached to both ends. A defect looks like: a jacket covering the
    cut-away, a bead floating free of the can, or a tab that hangs in space.
14. **The annotations live at the margins, in two columns of two rows.** Every badge sits
    12 px from its stage edge, 158 px wide — the value on the first row, its provenance dot
    and unit on the second — and every leader line terminates on the badge's inner border, so
    the left flank and the right flank each read as one column whatever the camera does.
    Orbit the cell: the line should follow its part's anchor with a single dog-leg, never
    cross the drawing, and never end in mid-air. Selecting a part opens that part's dossier —
    carried by the document, floated over the stage — with its Latin name, subsystem,
    material, failure mode and its spec rows tagged typical/measured/derived/fitted; a badge
    or row whose value is refused shows the reason instead of a zero. A defect looks like: a
    badge floating over the can, two badges at the same height, or a dossier that stays blank
    after a click.
15. **Hotkeys act only while the stage has focus or hover.** `Space` plays/pauses the life,
    `E` explodes, `C` cuts a quarter away (and back), `A` toggles the annotations, `H`
    collapses the rail to its `⋯` pill, `Esc` clears the selection — and none of them fire
    while the pointer is elsewhere on the page; a focused rail button takes `Space` itself.
    The explode and peel sliders animate on springs rather than jumping, the HUD's
    millimetre readout tracks the explode, and the telemetry line (fps / calls / triangles /
    vertices) moves while the scene renders.
16. **The gauge glows, and the stage still tells the truth.** The emissive state gauge
    carries a soft bloom — a halo that fades over a few dozen pixels around the band and
    the terminal highlights, never a global haze — while the void stays the palette's own
    colour to the byte (`#0b1120` on Obsidian, `#F4EEDA` on Codex) and the vignette is a
    gradient, never stray text on the stage. Codex's threshold sits deliberately above the
    lit parchment, so its bloom fires only on true >0.9 highlights (none at rest is
    correct); Obsidian's gauge should glow at rest.
17. **The key states numbers, under headings.** The legend beside the stage reads Health /
    Origin / Casing temperature, and *every* band prints its range — including the
    End-of-Life band, which must read `< 80%` (it starts at `min: 0`, the exact case a
    truthiness check used to swallow) and must render as literal text, not vanish into a
    tag the browser invents. An empty section is omitted, not shown as a heading over
    nothing.
18. **The address bar is the share button.** Drag the explode/peel sliders, scrub the
    cursor, click a part, switch palette — the query string gains `exploded=…&cursor=…
    &part=…&theme=…` (coalesced, not per-frame). Copy it into a new tab: the same picture
    reopens. Hovering without clicking must *not* put a part in the URL. Reload lands you
    back where you were even without a URL, via the stored per-cell view.
19. **Badges with no reading collapse.** At a cursor where several parts have none, the
    flanks show one `N unmeasured` count badge rather than a wall of grey rows; clicking it
    (or the HUD toggle) expands them, and the badge's label explains what it will do.
20. **It behaves on a phone and with reduced motion.** Narrow the window (or open on a
    phone): the layout goes single-column, controls stay reachable, and a *tap* on a part
    opens its dossier (touch never hovers, so the press itself is raycast). With
    `prefers-reduced-motion: reduce` set, springs snap, camera poses jump and the target
    reticle holds still — while orbiting, scrubbing and selecting all keep working.
    A screen reader hears the current selection/cycle/SOH announced once, politely, when
    it actually changes.
