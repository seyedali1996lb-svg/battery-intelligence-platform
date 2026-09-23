"""The Streamlit host for the 3D cell scene — the iframe, in one place.

Streamlit cannot host a WebGL canvas itself, so this module builds the one HTML
document it hands to `st.components.v1.html`: the committed scene bundle, the
cell's `CellSceneSpec` as data, and a control strip. It is the *only* place in
this application that knows how the scene is delivered, and it deliberately
renders no numbers of its own — everything on screen is either read from the
spec or painted by the renderer from the spec, so the iframe and the Streamlit
panels beside it cannot disagree.

Two decisions worth stating, because they are the ones a reader would otherwise
have to infer:

* **The controls live inside the iframe.** A Streamlit widget reruns the script,
  which re-sends the component and would reload the canvas — losing the camera
  angle, the explode position and the cursor on every click. So the things that
  change *how you look at the cell* (explode, casing, annotations, data-scaled
  geometry, the life cursor, play) are client-side, and the iframe is rebuilt
  only when the *document* changes (a different cell, a different theme).
* **The bundle is fetched over HTTP, not inlined.** `server.enableStaticServing`
  (`.streamlit/config.toml`) already serves `app/static/` for `theme.css`; the
  same route serves the 578 kB renderer, so the browser fetches it once and
  caches it instead of Streamlit re-sending it on every rerun.
"""

from __future__ import annotations

import json

from cell_scene import default_theme

#: Where the bundle lives on the static route (query-free), relative to the
#: Streamlit app's root. Same convention as `app/static/theme.css` in
#: app/main.py — see the comment there and `.streamlit/config.toml`'s
#: `enableStaticServing`.
BUNDLE_PATH = "app/static/cell_scene/cell_scene.js"


def _bundle_url() -> str:
    """The bundle, cache-busted with the manifest's own hash of it.

    A committed bundle is served from a browser cache that never learns the
    file changed: a screenshot of the deployed scene showed a stale renderer
    (a vignette fix that was already in the bundle had not reached users). The
    hash `scripts/export_scene_sample.py` stamps into `manifest.json` after
    every build is the cache key, so shipping a new bundle always ships a new
    URL. Falls back to the plain path when the manifest is absent, because a
    missing manifest must degrade to "no version", not to a page that will not
    render.
    """
    import pathlib

    manifest_path = pathlib.Path(__file__).resolve().parent / "static" / "cell_scene" / "manifest.json"
    try:
        sha = json.loads(manifest_path.read_text(encoding="utf-8")).get("bundleSha256") or ""
    except Exception:
        sha = ""
    return f"{BUNDLE_PATH}?v={sha[:12]}" if sha else BUNDLE_PATH


BUNDLE_URL = _bundle_url()


def theme_from_app(light_mode: bool = False) -> dict:
    """The scene's theme, restyled to match the Streamlit app's current mode.

    Starts from `cell_scene.default_theme()` — the SOH band colours and
    thresholds the rest of the platform uses — and swaps only the surface
    colours, so a light-mode session gets a legible scene without the health
    semantics moving at all.
    """
    theme = default_theme()
    if not light_mode:
        return theme
    theme.update({
        "background": "#f7fafc",
        "panel": "#ffffff",
        "text": "#1a202c",
        "muted": "#718096",
        "grid": "#cbd5e0",
        "metal": "#4a5568",
    })
    return theme


def _embedded_json(payload: dict) -> str:
    """JSON safe to sit inside a `<script>` tag.

    `</script>` inside a string would end the element early and turn the rest of
    the document into markup — and this payload carries prose (the disclosures)
    that a reviewer can edit without ever thinking about HTML escaping.
    """
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def band_range(minimum: float | None, maximum: float | None, unit: str = "%") -> str:
    """The explicit numeric range of one theme band, in the reader's units.

    The old renderer asked `band.min ? …` — truthiness, so the End-of-Life
    band (`min: 0`) silently lost its range and the key described colours with
    no numbers. A range is the whole point of a key, so every band states it:
    `≥ 90%`, `80–90%`, `< 80%` (and `< 30 °C` … for the casing tint).
    """
    if minimum is None and maximum is None:
        return ""
    if maximum is None:
        return f"≥ {minimum:g}{unit}"
    if not minimum:
        return f"< {maximum:g}{unit}"
    return f"{minimum:g}–{maximum:g}{unit}"


def _esc(text: str) -> str:
    """HTML-escape one fragment — mirrors `legend.ts`'s `esc` exactly.

    Escaping is not optional here: the very range this key must print for the
    End-of-Life band *starts with* `< 80%`, and unescaped that `<` opens an
    element the browser either drops or, worse, invents. The TS half escapes
    the same four characters, so both hosts emit byte-identical rows.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _band_rows(bands: list | None, unit: str, *, swatch: str) -> str:
    rows = []
    for band in bands or []:
        label = band.get("label") or ""
        span = band_range(band.get("min"), band.get("max"), unit)
        text = _esc(" · ".join(part for part in (label, span) if part))
        rows.append(
            f"<span class='band'><i class='{swatch}' style='background:{band.get('color', 'transparent')}'></i>"
            f"{text}</span>"
        )
    return "".join(rows)


def legend_html(theme: dict) -> str:
    """The band and provenance key this host draws beside the stage.

    Mirrors `frontend/src/scene/legend.ts`'s `legendHtml` — the same three
    headings, the same explicit ranges — so the iframe and the standalone
    harness describe identical colours. Reads the theme the meshes are painted
    from, so the key cannot drift from the drawing.
    """
    health = _band_rows(theme.get("sohBands"), "%", swatch="box")
    origin = "".join(
        f"<span class='band'><i class='line' style='background:{colour}'></i>{_esc(name)}</span>"
        for name, colour in (theme.get("provenanceColors") or {}).items()
        if name
    )
    tint = _band_rows(theme.get("temperatureBands"), " °C", swatch="box")
    parts = []
    if health:
        parts.append(f"<div class='legend-head'>Health</div><div class='legend-body'>{health}</div>")
    if origin:
        parts.append(f"<div class='legend-head'>Origin</div><div class='legend-body'>{origin}</div>")
    if tint:
        parts.append(f"<div class='legend-head'>Casing temperature</div><div class='legend-body'>{tint}</div>")
    return "".join(parts)


def scene_iframe_html(spec: dict, *, height: int = 700) -> str:
    """One self-contained HTML document that draws `spec`.

    Raises a plain `ValueError` on a document this host cannot render, because a
    silent blank canvas is the failure mode this whole module exists to avoid.
    """
    if not isinstance(spec, dict) or "schemaVersion" not in spec:
        raise ValueError(
            "scene_iframe_html needs a CellSceneSpec document (see docs/cell_scene.schema.json)"
        )
    theme = spec.get("theme") or {}
    cell = spec.get("cell") or {}
    n_measured = len(spec.get("series", {}).get("cycles", []))
    n_future = len((spec.get("projection") or {}).get("cycles") or []) if spec.get("projection", {}).get("available") else 0
    legend = legend_html(theme)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=EB+Garamond&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  html, body {{ margin:0; padding:0; background:{theme.get('background', '#0b1120')};
                color:{theme.get('text', '#e2e8f0')};
                font:400 12px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif; }}
  .wrap {{ display:grid; grid-template-columns:190px 1fr; gap:10px; padding:8px; height:{height - 24}px; box-sizing:border-box; }}
  .panel {{ background:{theme.get('panel', '#111827')}; border:1px solid {theme.get('grid', '#1f2937')};
            border-radius:10px; padding:10px; }}
  .panel h3 {{ margin:0 0 8px; font-size:10px; letter-spacing:0.08em; text-transform:uppercase;
               color:{theme.get('muted', '#94a3b8')}; font-weight:600; }}
  .controls label {{ display:block; margin-bottom:10px; color:{theme.get('muted', '#94a3b8')}; font-size:11px; }}
  .controls .value {{ float:right; color:{theme.get('text', '#e2e8f0')}; }}
  input[type=range] {{ width:100%; margin-top:5px; accent-color:{theme.get('accent', '#63b3ed')}; }}
  .btns {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px; }}
  button {{ background:transparent; color:{theme.get('text', '#e2e8f0')};
            border:1px solid {theme.get('grid', '#1f2937')}; border-radius:6px;
            padding:4px 8px; font:inherit; font-size:11px; cursor:pointer; }}
  button:hover {{ border-color:{theme.get('accent', '#63b3ed')}; }}
  .toggles label {{ display:flex; align-items:center; gap:6px; margin-bottom:7px; font-size:11px; }}
  .stage {{ position:relative; height:100%; border:1px solid {theme.get('grid', '#1f2937')};
            border-radius:10px; overflow:hidden; background:{theme.get('background', '#0b1120')}; }}
  #canvas {{ position:absolute; inset:0; }}
  .hud {{ position:absolute; left:10px; top:8px; display:flex; gap:10px; align-items:baseline; pointer-events:none; }}
  .hud .soh {{ font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .hud .doc {{ color:{theme.get('muted', '#94a3b8')}; font-size:11px; }}
  .inspected {{ position:absolute; left:10px; bottom:8px; right:10px; font-size:11px;
                color:{theme.get('muted', '#94a3b8')}; pointer-events:none; }}
  .note {{ position:absolute; right:10px; bottom:8px; max-width:46%; text-align:right;
           font-size:10.5px; color:{theme.get('muted', '#94a3b8')}; pointer-events:none; }}
  .legend {{ margin-top:8px; font-size:10.5px; color:{theme.get('muted', '#94a3b8')}; }}
  .legend .legend-head {{ margin-top:8px; font-size:9px; letter-spacing:0.08em; text-transform:uppercase;
                          color:{theme.get('muted', '#94a3b8')}; font-weight:600; }}
  .legend .legend-body {{ margin-top:3px; }}
  .legend .band {{ display:inline-flex; align-items:center; gap:5px; margin-right:9px; }}
  .legend i {{ width:9px; height:9px; border-radius:2px; display:inline-block; }}
  .legend i.line {{ height:2px; width:12px; }}
  .hotkey {{ margin-top:8px; font-size:9.5px; color:{theme.get('muted', '#94a3b8')}; }}
  @media (max-width: 760px) {{
    .wrap {{ grid-template-columns:1fr; grid-template-rows:auto 1fr; height:auto; }}
    .stage {{ height:{max(360, height - 300)}px; }}
  }}
  @media (pointer: coarse) {{
    button {{ padding:7px 10px; }}
    input[type=range] {{ height:24px; }}
  }}
  .fallback {{ padding:24px; font-size:12px; color:{theme.get('muted', '#94a3b8')}; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel controls">
    <h3>Cell controls</h3>
    <div class="btns">
      <button id="play">▶ Play life</button>
      <button id="today">Today</button>
      <button id="reset">Reset view</button>
    </div>
    <div class="btns">
      <button data-pose="iso" title="Isometric view">ISO</button>
      <button data-pose="plan" title="Plan view — top down">Plan</button>
      <button data-pose="section" title="Section view — cut through the axis">Section</button>
    </div>
    <label>Exploded view <span class="value" id="explode-value">0%</span>
      <input id="explode" type="range" min="0" max="100" value="0" /></label>
    <label>Peel layers <span class="value" id="peel-value">0%</span>
      <input id="peel" type="range" min="0" max="100" value="0" /></label>
    <label>Life cursor <span class="value" id="cursor-value">—</span>
      <input id="cursor" type="range" min="0" max="{max(0, n_measured + n_future - 1)}" value="{max(0, n_measured - 1)}" /></label>
    <div class="toggles">
      <label><input id="data-scaled" type="checkbox" /> Data-scaled geometry</label>
      <label><input id="annotations" type="checkbox" checked /> Part annotations</label>
      <label><input id="strip-casing" type="checkbox" /> Strip the casing</label>
    </div>
    <div id="palette" class="btns" style="display:none"></div>
    <div class="legend">{legend}</div>
    <div class="hotkey">Keys: Space play · E explode · C cutaway · A annotations · H hide this rail</div>
  </div>

  <div class="stage">
    <div id="canvas"></div>
    <div class="hud">
      <span class="soh" id="soh">—</span>
      <span class="doc" id="doc">—</span>
    </div>
    <div class="inspected" id="inspected"></div>
    <div class="note" id="note"></div>
    <noscript><div class="fallback">This view needs JavaScript. Every number it shows is on the
      part cards beside it.</div></noscript>
  </div>
</div>
<script src="{BUNDLE_URL}"></script>
<script id="cell-scene-spec" type="application/json">{_embedded_json(spec)}</script>
<script>
(function () {{
  const spec = JSON.parse(document.getElementById("cell-scene-spec").textContent);
  const el = (id) => document.getElementById(id);
  if (!window.CellScene) {{
    el("canvas").innerHTML = "<div class='fallback'>The 3D renderer did not load. It is served from " +
      "{BUNDLE_URL} by Streamlit's static file route (server.enableStaticServing) — every number " +
      "this view would show is on the part cards beside it.</div>";
    return;
  }}
  const setPlayLabel = (playing) => {{ el("play").textContent = playing ? "⏸ Pause" : "▶ Play life"; }};
  // The frame reports the measured length on every tick; Today jumps the cursor
  // to the last measured cycle (never into the projection) from that number.
  // Declared before mount because `onFrame` may fire during the first rebuild.
  let measuredCount = {max(0, n_measured)};
  const mounted = window.CellScene.mount(el("canvas"), spec, {{
    // No cursor pin: when neither the URL nor this cell's stored view names a
    // cursor, the engine's own default is the last measured cycle — *today* —
    // and passing it explicitly would outrank both carriers, reopening every
    // shared or restored view at the end of the record.
    onFrame: (state) => {{
      el("soh").textContent = state.soh === null ? "no reading" : state.soh.toFixed(1) + "%";
      measuredCount = state.measuredCount;
      el("soh").style.color = state.gaugeColor;
      el("doc").textContent = state.gaugeLabel +
        (state.cycle === null ? "" : " · cycle " + Math.round(state.cycle)) +
        (state.projected ? " · " + (state.projectionLabel || "projected") : " · measured");
      el("note").textContent = state.scaleNote || "";
      el("cursor-value").textContent = (state.cursor + 1) + " / " +
        (state.measuredCount + state.projectionLength);
      // The strip mirrors the engine's own frame — exploded, peel, layout —
      // so a change made by a hotkey or the HUD rail shows up here too. One
      // direction only: writing to a slider from here would fire its listener.
      if (state.exploded !== undefined) {{
        const pct = Math.round(state.exploded * 100);
        el("explode").value = String(pct);
        el("explode-value").textContent = pct + "%";
      }}
      if (state.peel !== undefined) {{
        const pct = Math.round(state.peel * 100);
        el("peel").value = String(pct);
        el("peel-value").textContent = pct + "%";
      }}
      // The cursor slider tracks the engine too: a HUD hotkey, playback or a
      // restored view move the timeline without touching this strip, and a
      // slider stuck at its mount position would misreport where the scene is.
      el("cursor").value = String(state.cursor);
      if (state.dataScaled !== undefined) el("data-scaled").checked = state.dataScaled;
      if (state.annotations !== undefined) el("annotations").checked = state.annotations;
      if (state.casing !== undefined) el("strip-casing").checked = state.casing === "hidden";
      el("inspected").textContent = state.inspected
        ? (spec.parts.find((part) => part.id === state.inspected) || {{}}).meaning || ""
        : "Hover a part to read what it is. Drag to orbit, scroll to zoom, right-drag to pan.";
    }},
    // Reaching the end of the cell's life stops playback; the button is the
    // host's, so only the host can put its label back.
    onPlaybackEnd: () => setPlayLabel(false),
  }});
  if (!mounted.handle) {{
    el("canvas").innerHTML = "<div class='fallback'>" +
      (mounted.error || "This browser could not start the 3D view.") +
      "</div>";
    return;
  }}
  const scene = mounted.handle;
  el("explode").addEventListener("input", (event) => {{
    el("explode-value").textContent = event.target.value + "%";
    scene.update(spec, {{ exploded: Number(event.target.value) / 100 }});
  }});
  el("peel").addEventListener("input", (event) => {{
    el("peel-value").textContent = event.target.value + "%";
    scene.update(spec, {{ peel: Number(event.target.value) / 100 }});
  }});
  el("cursor").addEventListener("input", (event) => scene.setCursor(Number(event.target.value)));
  el("data-scaled").addEventListener("change", (event) =>
    scene.update(spec, {{ dataScaled: event.target.checked }}));
  el("strip-casing").addEventListener("change", (event) =>
    scene.update(spec, {{ casing: event.target.checked ? "hidden" : "translucent" }}));
  el("annotations").addEventListener("change", (event) => scene.setAnnotations(event.target.checked));
  el("play").addEventListener("click", () => {{
    if (scene.isPlaying()) {{ scene.pause(); setPlayLabel(false); return; }}
    // play() answers false only when there is no life to play, so the label
    // never claims a playback the engine did not start.
    setPlayLabel(scene.play() === true);
  }});
  el("reset").addEventListener("click", () => scene.resetView());
  el("today").addEventListener("click", () => {{
    const index = Math.max(0, measuredCount - 1);
    scene.setCursor(index);
    el("cursor").value = String(index);
  }});
  document.querySelectorAll("[data-pose]").forEach((button) => {{
    button.addEventListener("click", () => scene.framePreset(button.dataset.pose));
  }});
  // The palette switcher appears only when the document offers a second one —
  // a hidden control that cannot do anything is worse than no control.
  const palettes = Object.keys(spec.palettes || {{}});
  if (palettes.length > 1) {{
    const holder = el("palette");
    holder.style.display = "flex";
    palettes.forEach((name) => {{
      const b = document.createElement("button");
      b.textContent = name;
      b.title = "Palette " + name;
      b.addEventListener("click", () => scene.setTheme(name));
      holder.appendChild(b);
    }});
  }}
}})();
</script>
</body>
</html>"""
