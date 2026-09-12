"""
Per-cell provenance for the accuracy numbers shown in-product.

A bare "657 cycles remaining" or "R² = 0.76" invites a generic reading: that
the model is X% accurate, full stop. In reality every such number rests on a
specific, small population — NASA's leave-cell-out (LCO) RUL estimate is
validated on 4 held-out cells; Severson's on 12 — and on a specific chemistry.
The project already gates RUL *per cell* (batlab.validation.lco.
RUL_RELIABLE_FLOOR) precisely because a dataset-wide average can hide a cell
the model genuinely cannot predict; this module applies the same principle to
how those numbers are *labelled*: every accuracy number carries its own fold
count, its own fold R², and its own chemistry source next to it.

This is deliberately one shared resolver rather than per-page formatting, so
Overview, Decision, Health and Fleet cannot drift into describing the same
model with different populations. See docs/history.md's per-cell-vs-dataset-
average reliability gate entry for the bug class this avoids.

Pure logic, no Streamlit — usable from the app pages, the REST layer, and
tests alike.
"""

from __future__ import annotations


def cell_model_provenance(
    bundle: "dict | None",
    cell_id: str,
    selection: "dict | None" = None,
) -> dict:
    """
    Resolve every provenance fact about one cell's accuracy numbers.

    Parameters
    ----------
    bundle : the trained-model bundle for this cell's data source
        (app/_data.py's shape), or None if it isn't available. Never raises
        on a missing/oddly-shaped bundle — a caller rendering a page must not
        break because provenance couldn't be resolved.
    cell_id : the cell whose fold metrics to read.
    selection : optional model_selection.select_model_for_cell() record. When
        given, the provenance states which model actually answered, whether
        it was this cell's own-source model or a chemistry-matched reference
        model, and the per-chemistry accuracy it belongs to — so a prediction
        can never be labelled with one platform-wide number.

    Returns
    -------
    {
      "has_lco":         bool,   # did this bundle run leave-cell-out at all?
      "n_cells":         int | None,   # size of the held-out population
      "fold_rul_r2":     float | None, # this cell's own LCO RUL R²
      "fold_soh_r2":     float | None, # this cell's own LCO SOH R²
      "fold_rul_mae":    float | None, # this cell's own LCO RUL MAE (cycles)
      "baseline_fold_r2": float | None,# trivial cycle_number->SOH baseline, same fold
      "rul_reliable":    bool,   # per-cell reliability (never a dataset average)
      "chemistry":       str | None,   # e.g. "LiCoO2" / "LFP"
      "source":          str | None,   # e.g. "NASA" / "Severson" / "Synthetic"
      # present only when `selection` was supplied:
      "model_label":     str | None,   # the MODEL that answered
      "model_native":    bool | None,  # trained on this cell's own source?
      "selection":       str | None,   # "native" | "chemistry_match" | "none"
      "selection_reason": str | None,
    }

    Notes
    -----
    `n_cells` is the size of the LCO population the model for this cell's
    SOURCE was validated on, not a per-cell count — that population size is
    the honest denominator for "how much trust does this number deserve".
    `rul_reliable` reads `per_cell_rul_reliable` first and only falls back to
    the dataset-level `rul_reliable` when the cell isn't in that map (e.g. a
    cached bundle written before per-cell gating existed) — the aggregate is
    the fallback, never the primary.
    """
    metrics = ((bundle or {}).get("metrics") or {}) if isinstance(bundle, dict) else {}

    per_cell      = metrics.get("lco_per_cell") or {}
    per_cell_ok   = metrics.get("per_cell_rul_reliable") or {}
    baseline_cell = metrics.get("baseline_lco_per_cell") or {}

    fold = per_cell.get(cell_id) or {}
    base = baseline_cell.get(cell_id) or {}

    has_lco = bool(per_cell)

    chemistry = None
    source = None
    try:
        from chemistry_profiles import ChemistryProfile
        profile = ChemistryProfile.for_cell(cell_id)
        chemistry = profile.short_name
        source = profile.source_label
    except Exception:
        pass

    provenance = {
        "has_lco":          has_lco,
        "n_cells":          len(per_cell) if has_lco else None,
        # True when this cell has a fold of its own in the LCO population —
        # distinguishes "no fold" from "a fold whose RUL labels are all
        # formula-extrapolated (fold_rul_r2=None)".
        "has_own_fold":     has_lco and cell_id in per_cell,
        # v12 semantics: fold_rul_r2 is OBSERVED-EOL rows only, and None means
        # this fold has no measured labels at all — its RUL cannot be
        # validated. Callers must render that as "not evaluable", never as 0.
        "fold_rul_r2":      fold.get("rul_r2"),
        "fold_soh_r2":      fold.get("soh_r2"),
        "fold_rul_mae":     fold.get("rul_mae"),
        "baseline_fold_r2": base.get("baseline_soh_r2"),
        "rul_reliable":     bool(per_cell_ok.get(cell_id, metrics.get("rul_reliable", False))),
        # RUL label provenance (Tier-0): how much of this model's RUL
        # evaluation pool is measured, and the formula baseline it must beat.
        "rul_label_coverage": metrics.get("rul_label_coverage"),
        "rul_formula_baseline_r2": metrics.get("rul_formula_baseline_r2"),
        "chemistry":        chemistry,
        "source":           source,
    }

    if selection is not None:
        accuracy = selection.get("accuracy") or {}
        provenance.update({
            "model_label":      selection.get("model_label"),
            "model_native":     selection.get("native"),
            "selection":        selection.get("selection"),
            "selection_reason": selection.get("reason"),
            "model_key":        selection.get("key"),
            "n_candidates":     selection.get("n_candidates"),
            # The population/accuracy of the CHEMISTRY that answered, separate
            # from this cell's own fold — present even when the answering model
            # isn't the cell's own source model.
            "model_n_folds":    accuracy.get("n_folds"),
            "model_soh_r2":     accuracy.get("soh_r2"),
            "model_advantage":  accuracy.get("advantage"),
            "model_rul_r2":     accuracy.get("rul_r2"),
        })
        if not has_lco and accuracy:
            # A chemistry-matched model answered a cell with no fold of its
            # own: report the chemistry's population, honestly labelled as not
            # this cell's own validation.
            provenance["n_cells"] = accuracy.get("n_folds")
            provenance["has_lco"] = bool(accuracy.get("n_folds"))

    return provenance


def provenance_label(
    prov: dict,
    include_chemistry: bool = True,
    include_baseline: bool = False,
) -> str:
    """
    One-line provenance suffix for an accuracy number, e.g.

        "n=4 · R²=0.63 · NASA LiCoO2"
        "n=12 · R²=1.00 · baseline R²=-0.76 · Severson LFP"
        "n=12 · R²=1.00 · Severson LFP model (chemistry-matched)"

    Deliberately few, factual tokens: the population size, this cell's own
    held-out R², and where the data came from. An empty string is returned
    (not a placeholder) when there is no LCO population to describe, so a
    caller can render nothing rather than imply a validation that didn't
    happen.

    When `prov` came from cell_model_provenance(..., selection=...), the model
    that actually answered is named instead of the cell's own source, and a
    chemistry-matched stand-in is flagged: that number belongs to the
    chemistry, not to this cell.
    """
    if prov and prov.get("selection") == "none":
        return "no chemistry-compatible model — prediction withheld"

    if not prov or not prov.get("has_lco"):
        return ""

    parts = [f"n={prov['n_cells']}"]
    if prov.get("has_own_fold") and prov.get("fold_rul_r2") is None:
        # The fold exists but its RUL labels are all formula-extrapolated —
        # say that instead of silently dropping the token (which would read
        # as if the number were merely omitted).
        parts.append("RUL not evaluable (no measured labels)")
    elif prov.get("fold_rul_r2") is not None:
        parts.append(f"R²={prov['fold_rul_r2']:.2f}")
    if include_baseline and prov.get("baseline_fold_r2") is not None:
        parts.append(f"baseline R²={prov['baseline_fold_r2']:.2f}")
    if include_chemistry:
        if prov.get("model_label"):
            suffix = ""
            if prov.get("model_native") is True:
                suffix = " model (own source)"
            elif prov.get("model_native") is False:
                suffix = " model (chemistry-matched)"
            else:
                suffix = " model"
            parts.append(f"{prov['model_label']}{suffix}")
        else:
            src, chem = prov.get("source"), prov.get("chemistry")
            if src and chem:
                parts.append(f"{src} {chem}")
            elif src:
                parts.append(str(src))
    return " · ".join(parts)


def per_cell_reliability_detail(prov: dict, floor: float) -> str:
    """
    Plain-English statement of what the per-cell gate means for this cell,
    used alongside a visible RUL number. States the fold count, this cell's
    own fold R², and the floor — the "this was tested on a cell it never
    trained on, and here is how many cells that population actually is"
    sentence, so a reader cannot mistake the number for a fleet-scale claim.
    """
    if not prov or not prov.get("has_lco"):
        return "No leave-cell-out population is available for this cell's data source."
    n = prov.get("n_cells")
    r2 = prov.get("fold_rul_r2")
    if r2 is None and prov.get("has_own_fold"):
        # v12: the fold exists but carries no observed-EOL rows — every RUL
        # label for this cell is a closed-form extrapolation, so there is
        # nothing measured to validate against. This is the honest statement
        # of exactly what Severson's 0.9994 used to hide.
        coverage = prov.get("rul_label_coverage")
        cov_txt = (
            f"; {coverage * 100:.0f}% of the population's evaluated RUL rows are measured"
            if coverage is not None else ""
        )
        return (
            f"This cell's fold has no observed-EOL rows — its RUL labels are "
            f"formula extrapolations, not measurements, so its RUL cannot be "
            f"validated (population of {n} cell(s){cov_txt}). RUL is withheld."
        )
    if r2 is None:
        if prov.get("model_native") is False:
            # The answering model's population does not contain this cell at
            # all — say so, rather than implying a per-cell gate was applied.
            return (
                f"This cell has no fold of its own in the answered model's "
                f"leave-cell-out population ({n} cell(s), {prov.get('model_label') or '—'}) — "
                "the number comes from a chemically-matched reference model, not "
                "from a validation on these cells."
            )
        return (
            f"RUL is gated per cell over a leave-cell-out population of {n} "
            f"cell(s) (RUL_RELIABLE_FLOOR = {floor:.2f}); this cell has no fold "
            "of its own in that population."
        )
    return (
        f"RUL for this cell is validated leave-cell-out against {n} held-out "
        f"cell(s) of the same source — this cell's own fold R²={r2:.2f} "
        f"(reliable above the {floor:.2f} floor). {n} cell(s) is a thin "
        "population; treat it as directional, not a fleet-scale guarantee."
    )
