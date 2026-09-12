"""
Per-chemistry model selection.

The platform trains more than one model: one per reference source (NASA
LiCoO2, Severson LFP, the synthetic LiCoO2 fleet), plus — per tenant — one
for that org's own uploaded fleet. Which of them should answer for a given
cell is a *chemistry* question, not a dict-key-order question, and getting it
wrong is silent: several call sites fell back with ``bundles.get(key) or
bundles.get("synth")``, so a cell of an unrecognised source would be scored
by whichever model happened to be constructed first — a synthetic LiCoO2
model answering for an uploaded cell — with nothing in the output saying so.
That is the "one shared model" failure this module exists to end.

select_model_for_cell() replaces the guess with three explicit tiers:

1. **native** — the model trained on this cell's own source. Always preferred:
   its leave-cell-out population contains cells built the way this one was.
2. **chemistry** — a model trained on the SAME chemistry from a different
   source (e.g. an uploaded LFP cell answered by the Severson LFP model).
   Used only when no native model exists, and always flagged
   ``native=False`` so callers must disclose it: it is a chemically-matched
   reference model, **not** a validation on the caller's own cells.
3. **none** — no chemistry-compatible model. Returns a not-found selection and
   the caller withholds. It never answers with a different chemistry's model,
   because cross-chemistry transfer is exactly the failure documented by
   experiment_registry.run_cross_chemistry_study() (NASA -> Severson zero-shot:
   SOH R² ≈ −34, RUL R² ≈ −1.1).

Between several same-chemistry candidates the largest **genuine skill** wins —
soh_r2 minus the trivial linear baseline under identical folds
(batlab.validation.trivial_baseline), NOT the raw R². Raw R² rewards whichever
chemistry happens to age smoothly, which is precisely the confound that
baseline exists to remove: the synthetic LiCoO2 fleet scores R² = 0.998 but
most of that is the shape of an aging curve, while Severson's 0.986 is almost
entirely earned. Ties break toward the larger fold population.

chemistry_accuracy_report() is the companion disclosure the rest of the
platform reads: per-chemistry accuracy stated *separately* — which models
exist for each chemistry, how large their held-out populations are, and each
one's real advantage — so a prediction can be labelled with the accuracy of
the chemistry that produced it instead of one platform-wide number.

Pure logic, no Streamlit, no I/O beyond a best-effort registry lookup for
provenance — usable from app pages, the REST layer, and tests alike.
"""

from __future__ import annotations

from chemistry_profiles import ChemistryProfile

# Last-resort display labels when no cell id is available to ask
# ChemistryProfile about (e.g. a hand-built test bundle).
_SOURCE_LABELS = {
    "nasa": "NASA",
    "severson": "Severson",
    "synth": "Synthetic",
    "upload": "Uploaded",
    "oxford": "Oxford",
}


def _bundle_metrics(bundle) -> dict:
    """The metrics dict inside a model bundle, or {} — never raises."""
    if isinstance(bundle, dict):
        metrics = bundle.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return {}


def _registry_run(run_id) -> dict:
    """Best-effort experiment-registry lookup. Returns {} when the registry is
    unavailable (bare process, no DB) — provenance enrichment, never a hard
    dependency."""
    if not run_id:
        return {}
    try:
        import experiment_registry as reg
        return reg.get_run(reg.PLATFORM_ORG_ID, run_id) or {}
    except Exception:
        return {}


def _sample_cell_id(bundle, key: str, cell_ids_by_key: "dict | None") -> "str | None":
    """One cell id that identifies which chemistry a bundle was trained on."""
    if cell_ids_by_key:
        ids = cell_ids_by_key.get(key) or []
        if ids:
            return ids[0]
    per_cell = _bundle_metrics(bundle).get("lco_per_cell")
    if isinstance(per_cell, dict) and per_cell:
        return next(iter(per_cell))
    if isinstance(bundle, dict):
        fdfs = bundle.get("featured_dfs")
        if isinstance(fdfs, dict) and fdfs:
            return next(iter(fdfs))
    return None


def describe_bundle(key: str, bundle, cell_ids_by_key: "dict | None" = None) -> dict:
    """
    One model descriptor for a bundle: which chemistry and source it was
    trained on, how big its held-out population is, and how much of its R² is
    genuine skill rather than the shape of an aging curve.

    Chemistry is resolved from the run's own registry record first (the
    authoritative dataset/chemistry pair the training call site logged), then
    from a sample cell id, then falls back to the bundle key's conventional
    label. `advantage` is soh_r2 − baseline_soh_r2 (None when the run predates
    the baseline metric) — the honest ranking signal, never raw R².
    """
    metrics = _bundle_metrics(bundle)
    run = _registry_run(metrics.get("experiment_run_id"))

    chemistry = run.get("chemistry")
    dataset = run.get("dataset")
    feature_version = run.get("feature_version")

    per_cell = metrics.get("lco_per_cell")
    per_cell = per_cell if isinstance(per_cell, dict) else {}

    sample = _sample_cell_id(bundle, key, cell_ids_by_key)
    source = None
    if sample is not None:
        try:
            profile = ChemistryProfile.for_cell(sample)
            chemistry = chemistry or profile.short_name
            source = profile.source_label
        except Exception:
            pass
    if source is None:
        source = _SOURCE_LABELS.get(key, key)

    soh_r2 = metrics.get("lco_soh_r2", metrics.get("soh_r2"))
    baseline = metrics.get("baseline_soh_r2")
    advantage = (
        float(soh_r2) - float(baseline)
        if (soh_r2 is not None and baseline is not None) else None
    )
    rul_r2 = metrics.get("lco_rul_r2", metrics.get("rul_r2"))

    return {
        "key":                key,
        "bundle":             bundle,
        "chemistry":          chemistry,
        "dataset":            dataset or key,
        "source":             source,
        "model_label":        f"{source} {chemistry}" if chemistry else str(source),
        "n_cells":            metrics.get("n_cells"),
        "n_folds":            len(per_cell),
        "soh_r2":             soh_r2,
        "baseline_soh_r2":    baseline,
        "advantage":          advantage,
        "rul_r2":             rul_r2,
        "rul_reliable":       bool(metrics.get("rul_reliable", False)),
        # Tier-0 RUL label provenance (v12): what fraction of the RUL
        # evaluation pool carries measured labels, and the closed-form
        # baseline the model must beat. None on pre-v12 bundles.
        "rul_label_coverage": metrics.get("rul_label_coverage"),
        "rul_formula_baseline_r2": metrics.get("rul_formula_baseline_r2"),
        "feature_version":    feature_version or metrics.get("feature_version"),
        "experiment_run_id":  metrics.get("experiment_run_id"),
    }


def describe_bundles(bundles: dict, cell_ids_by_key: "dict | None" = None) -> list[dict]:
    """Describe every non-empty bundle in a {key: bundle} mapping."""
    return [
        describe_bundle(key, bundle, cell_ids_by_key)
        for key, bundle in (bundles or {}).items()
        if bundle
    ]


def _skill_rank(descriptor: dict):
    """
    Sort key placing the most trustworthy model first: models that actually
    have a baseline advantage come first (a model with no baseline can't be
    ranked on skill at all), then larger advantage, then larger population.
    """
    advantage = descriptor.get("advantage")
    return (
        0 if advantage is not None else 1,
        -(float(advantage) if advantage is not None else 0.0),
        -(descriptor.get("n_folds") or 0),
        -(descriptor.get("n_cells") or 0),
    )


def select_model_for_cell(
    cell_id: str,
    bundles: dict,
    cell_ids_by_key: "dict | None" = None,
) -> dict:
    """
    Choose the model that should answer for `cell_id`: its own source's model
    when one exists, else the best same-chemistry model, else nothing.

    Returns
    -------
    {
      "found":          bool,   # a chemistry-compatible model exists
      "bundle":         dict,   # the selected bundle, or {} when not found
      "key":            str | None,
      "selection":      "native" | "chemistry_match" | "none",
      "native":         bool,   # trained on this cell's own source
      "chemistry":      str | None,   # the CELL's chemistry
      "model_label":    str | None,   # the MODEL's source+chemistry
      "accuracy":       dict,   # describe_bundle() of the selected model
      "candidates":     list,   # every same-chemistry descriptor considered
      "n_candidates":   int,
      "reason":         str,    # human sentence, safe to show verbatim
    }

    Never raises: an unresolvable chemistry or an empty bundles mapping is a
    not-found selection, not an exception — a page must not break because
    selection couldn't be resolved.
    """
    profile = None
    try:
        profile = ChemistryProfile.for_cell(cell_id)
    except Exception:
        profile = None

    native_key = getattr(profile, "source_kind", None)
    cell_chemistry = getattr(profile, "short_name", None)

    candidates = describe_bundles(bundles, cell_ids_by_key)

    # ── Tier 1: the cell's own source model ─────────────────────────────────
    for descriptor in candidates:
        if native_key is not None and descriptor["key"] == native_key:
            return {
                "found":        True,
                "bundle":       descriptor["bundle"],
                "key":          descriptor["key"],
                "selection":    "native",
                "native":       True,
                "chemistry":    cell_chemistry,
                "model_label":  descriptor["model_label"],
                "accuracy":     descriptor,
                "candidates":   [descriptor],
                "n_candidates": 1,
                "reason": (
                    f"Answered by the {descriptor['model_label']} model — the model "
                    f"trained on this cell's own source"
                    + (f", validated leave-cell-out on {descriptor['n_folds']} cell(s)"
                       if descriptor["n_folds"] else "")
                    + "."
                ),
            }

    # ── Tier 2: same chemistry, different source ────────────────────────────
    same_chemistry = [
        d for d in candidates
        if cell_chemistry is not None and d["chemistry"] == cell_chemistry
    ]
    if same_chemistry:
        same_chemistry.sort(key=_skill_rank)
        best = same_chemistry[0]
        return {
            "found":        True,
            "bundle":       best["bundle"],
            "key":          best["key"],
            "selection":    "chemistry_match",
            "native":       False,
            "chemistry":    cell_chemistry,
            "model_label":  best["model_label"],
            "accuracy":     best,
            "candidates":   same_chemistry,
            "n_candidates": len(same_chemistry),
            "reason": (
                f"No model was trained on this cell's own source, so it is answered "
                f"by the platform's {cell_chemistry} model — {best['model_label']}"
                + (f", validated on {best['n_folds']} held-out cell(s)" if best["n_folds"] else "")
                + ". This is a chemically-matched reference model, NOT a validation "
                "on this cell's own data."
            ),
        }

    # ── Tier 3: nothing chemistry-compatible — refuse rather than mislead ───
    _chem_txt = cell_chemistry or "an unrecognised"
    return {
        "found":        False,
        "bundle":       {},
        "key":          None,
        "selection":    "none",
        "native":       False,
        "chemistry":    cell_chemistry,
        "model_label":  None,
        "accuracy":     {},
        "candidates":   [],
        "n_candidates": 0,
        "reason": (
            f"No model trained on {_chem_txt} chemistry is available on this "
            "deployment, and this platform does not answer a cell with a "
            "different chemistry's model — cross-chemistry transfer error is "
            "documented in the Benchmark page's cross-chemistry table. "
            "Predictions are withheld."
        ),
    }


def select_bundle_for_cell(
    cell_id: str,
    bundles: dict,
    cell_ids_by_key: "dict | None" = None,
) -> dict:
    """Convenience wrapper returning just the selected bundle ({} when not
    found) — for callers that only need the bundle and whose existing
    not-found handling already covers an empty dict."""
    return select_model_for_cell(cell_id, bundles, cell_ids_by_key)["bundle"]


def chemistry_accuracy_report(
    bundles: dict,
    cell_ids_by_key: "dict | None" = None,
) -> list[dict]:
    """
    Per-chemistry accuracy, stated separately — one row per chemistry that has
    at least one trained model on this deployment.

    This is the disclosure a prediction is labelled with: what accuracy the
    *chemistry* that produced it has earned, and which model the platform
    would pick for a new cell of that chemistry. Several models can share a
    chemistry (NASA and the synthetic fleet are both LiCoO2), which is exactly
    when "one number for the platform" would be misleading.

    Each row:
        {
          "chemistry", "n_models", "n_cells_total", "sources",
          "selected_key", "selected_model_label",   # what selection picks
          "n_folds", "soh_r2", "baseline_soh_r2", "advantage",
          "rul_r2", "rul_reliable", "models": [descriptor, ...],
        }

    Sorted by genuine advantage descending (largest real skill first), so a
    chemistry whose R² is mostly smooth aging curves cannot lead the list.
    """
    by_chemistry: dict = {}
    for descriptor in describe_bundles(bundles, cell_ids_by_key):
        chemistry = descriptor.get("chemistry")
        if not chemistry:
            continue
        by_chemistry.setdefault(chemistry, []).append(descriptor)

    rows = []
    for chemistry, descriptors in by_chemistry.items():
        descriptors.sort(key=_skill_rank)
        best = descriptors[0]
        n_cells_total = sum(d.get("n_cells") or 0 for d in descriptors)
        rows.append({
            "chemistry":            chemistry,
            "n_models":             len(descriptors),
            "n_cells_total":        n_cells_total,
            "sources":              sorted({d["source"] for d in descriptors}),
            "selected_key":         best["key"],
            "selected_model_label": best["model_label"],
            "n_folds":              best["n_folds"],
            "soh_r2":               best["soh_r2"],
            "baseline_soh_r2":      best["baseline_soh_r2"],
            "advantage":            best["advantage"],
            "rul_r2":               best["rul_r2"],
            "rul_reliable":         best["rul_reliable"],
            "models":               descriptors,
        })

    rows.sort(key=lambda r: (
        0 if r["advantage"] is not None else 1,
        -(float(r["advantage"]) if r["advantage"] is not None else 0.0),
    ))
    return rows


def describe_bundle_for_cell(bundle, cell_id: str) -> dict:
    """
    Selection-style record for a bundle a caller was already HANDED, without
    needing the full {key: bundle} mapping.

    Overview and Health receive one bundle from the router rather than the
    whole mapping, but must still be able to say which model answered and
    whether it was chemistry-compatible. This resolves that from the bundle
    itself: its own cells' source (`lco_per_cell` keys, e.g. B0005 -> NASA)
    decides native, and its chemistry decides compatibility. A bundle whose
    chemistry is simply not the cell's is reported as incompatible — the
    caller must not present its number as if it applied.

    Never raises; an unresolvable bundle yields a record with everything None.
    """
    descriptor = describe_bundle("selected", bundle)

    try:
        profile = ChemistryProfile.for_cell(cell_id)
    except Exception:
        profile = None
    cell_chemistry = getattr(profile, "short_name", None)
    cell_source_kind = getattr(profile, "source_kind", None)

    model_chemistry = descriptor.get("chemistry")

    sample = _sample_cell_id(bundle, "selected", None)
    model_source_kind = None
    if sample is not None:
        try:
            model_source_kind = ChemistryProfile.for_cell(sample).source_kind
        except Exception:
            model_source_kind = None

    native = bool(
        cell_source_kind is not None and model_source_kind == cell_source_kind
    )
    if not native and model_source_kind is None and model_chemistry:
        # No cell id to ask — fall back to the weaker chemistry comparison.
        native = (model_chemistry == cell_chemistry)

    compatible = bool(
        native
        or (model_chemistry and cell_chemistry and model_chemistry == cell_chemistry)
        or model_chemistry is None  # unknown chemistry: cannot be shown incompatible
    )

    if native:
        selection_kind = "native"
        reason = (
            f"Answered by the {descriptor['model_label']} model — the model "
            f"trained on this cell's own source"
            + (f", validated leave-cell-out on {descriptor['n_folds']} cell(s)"
               if descriptor["n_folds"] else "")
            + "."
        )
    elif compatible:
        selection_kind = "chemistry_match"
        reason = (
            f"Answered by a {model_chemistry} model ({descriptor['model_label']}) "
            f"from a different source than this cell — a chemically-matched "
            "reference model, NOT a validation on this cell's own data."
        )
    else:
        selection_kind = "incompatible"
        reason = (
            f"The available model was trained on {model_chemistry} chemistry but "
            f"this cell is {cell_chemistry or 'of an unknown chemistry'} — its "
            "numbers do not apply here and must not be presented as if they did."
        )

    return {
        "found":        compatible,
        "bundle":       bundle,
        "key":          descriptor.get("key"),
        "selection":    selection_kind,
        "native":       native,
        "chemistry":    cell_chemistry,
        "model_label":  descriptor.get("model_label"),
        "accuracy":     descriptor,
        "candidates":   [descriptor],
        "n_candidates": 1,
        "compatible":   compatible,
        "reason":       reason,
    }


def per_chemistry_accuracy(
    chemistry: "str | None" = None,
    tenant_org_id: "int | None" = None,
) -> list[dict]:
    """
    Per-chemistry accuracy read from the EXPERIMENT REGISTRY — every logged
    leave-cell-out run, whether or not its model bundle happens to be loaded
    in this process.

    This is the reporting half of the change that makes model selection
    per-chemistry: a prediction is labelled with the accuracy of the chemistry
    that produced it, not one platform-wide number. Several models can cover
    one chemistry (NASA and the synthetic fleet are both LiCoO2), so the rows
    are aggregated per chemistry and the BEST model is reported as the one
    selection would pick — ranked by genuine advantage over the trivial
    baseline, never raw R².

    Returns [] when the registry is empty or unavailable (a fresh deployment
    genuinely has no per-chemistry accuracy to report yet), and a single
    row when `chemistry` is given and found.
    """
    try:
        import experiment_registry as reg
        rows = reg.accuracy_by_source(tenant_org_id=tenant_org_id)
    except Exception:
        return []

    grouped: dict = {}
    for row in rows:
        chem = row.get("chemistry")
        if not chem:
            continue
        score = row.get("advantage")
        grouped.setdefault(chem, []).append((row, score))

    out = []
    for chem, entries in grouped.items():
        entries.sort(key=lambda pair: (
            0 if pair[1] is not None else 1,
            -(float(pair[1]) if pair[1] is not None else 0.0),
        ))
        best = entries[0][0]
        out.append({
            "chemistry":            chem,
            "n_models":             len(entries),
            "n_cells_total":        sum((e[0].get("n_cells") or 0) for e in entries),
            "sources":              sorted({e[0].get("dataset") for e in entries}),
            "selected_key":         best.get("dataset"),
            "selected_model_label": best.get("dataset"),
            "n_folds":              best.get("n_cells"),
            "soh_r2":               best.get("soh_r2"),
            "baseline_soh_r2":      best.get("baseline_soh_r2"),
            "advantage":            best.get("advantage"),
            "rul_r2":               best.get("rul_r2"),
            "rul_reliable":         bool(best.get("rul_reliable")),
            "source":               "experiment_registry",
        })

    out.sort(key=lambda r: (
        0 if r["advantage"] is not None else 1,
        -(float(r["advantage"]) if r["advantage"] is not None else 0.0),
    ))

    if chemistry is None:
        return out
    return [r for r in out if r["chemistry"] == chemistry]


def chemistry_accuracy_line(row: dict) -> str:
    """Plain-text (no markdown) per-chemistry accuracy statement, for UI
    captions rendered through st.caption()/inner HTML. States the chemistry,
    how many models cover it, the population behind the chosen one, its held-out
    R², and its real advantage over the trivial baseline — so a single R² is
    never the whole claim."""
    def _f(v, dec=3):
        return f"{v:.{dec}f}" if v is not None else "—"

    parts = [str(row.get("chemistry") or "—")]
    parts.append(f"{row.get('n_models', 0)} model(s)"
                 + (f" [{', '.join(row.get('sources') or [])}]" if row.get("sources") else ""))
    if row.get("n_folds"):
        parts.append(f"n={row['n_folds']}")
    parts.append(f"R²={_f(row.get('soh_r2'))}")
    if row.get("baseline_soh_r2") is not None and row.get("advantage") is not None:
        parts.append(f"vs trivial baseline {row['advantage']:+.3f}")
    return " · ".join(parts)


def per_chemistry_accuracy_line(
    chemistry: "str | None" = None,
    tenant_org_id: "int | None" = None,
    bundles: "dict | None" = None,
    separator: str = "  |  ",
) -> str:
    """Per-chemistry accuracy as one line (all chemistries, or just one), or
    "" when there is nothing to report. The companion disclosure to a
    prediction: which chemistry it came from and what that chemistry has
    actually earned.

    The experiment registry is preferred — it is the authoritative record of
    every logged run, including models not loaded in this process. `bundles` is
    the fallback for the case the registry legitimately can't cover: a warm
    bundle cache with a fresh/empty registry, where the models exist but no run
    has been logged in this database yet. Without the fallback that case would
    silently report nothing rather than the accuracy of the model that just
    answered.
    """
    rows = per_chemistry_accuracy(chemistry=chemistry, tenant_org_id=tenant_org_id)
    if not rows and bundles:
        rows = chemistry_accuracy_report(bundles, cell_ids_by_key=None)
        if chemistry is not None:
            rows = [r for r in rows if r["chemistry"] == chemistry]
    if not rows:
        return ""
    return separator.join(chemistry_accuracy_line(r) for r in rows)


def chemistry_accuracy_label(row: dict) -> str:
    """One-line per-chemistry accuracy statement for one
    chemistry_accuracy_report() row, e.g.

        "**LiCoO2** · 2 model(s) (NASA, Synthetic) · selected: Synthetic LiCoO2
         n=8 · R²=0.998 · baseline R²=-1.967 → **+2.965**"

    States the chemistry, how many models cover it, which one selection
    picks, that model's held-out population, and its real advantage over the
    trivial baseline — never the raw R² alone."""
    def _f(v, dec=3):
        return f"{v:.{dec}f}" if v is not None else "—"

    parts = [
        f"**{row.get('chemistry') or '—'}**",
        f"{row.get('n_models', 0)} model(s)"
        + (f" ({', '.join(row.get('sources') or [])})" if row.get("sources") else ""),
        f"selected: {row.get('selected_model_label') or '—'}"
        + (f" n={row['n_folds']}" if row.get("n_folds") else ""),
        f"R²={_f(row.get('soh_r2'))}",
    ]
    if row.get("baseline_soh_r2") is not None and row.get("advantage") is not None:
        parts.append(f"baseline R²={_f(row['baseline_soh_r2'])} → **{row['advantage']:+.3f}**")
    return " · ".join(parts)
