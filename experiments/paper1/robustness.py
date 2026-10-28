"""Robustness and reporting blocks added after the methodology audit (2026-10-27). None of them changes a frozen rule or a headline
number; each one shows how much a headline depends on a choice.

The primary correctness target counts a direct answer that is missing (cut off at the token limit, or not parseable) as incorrect: the
probes ask whether the model knows the answer, and the protocol gives it 32 tokens to state one. What the AUROC then measures includes
whether the model completes a short answer. The answered-only blocks condition on having answered instead; that changes the population of
questions, so both are reported with their coverage and neither is the other's refutation.

Per screened cell (cell_diagnostics): the verdict split into design, stability and signal tiers; the verdict when S2 is read as a design
restriction and when S4 judges only the probe's own scale pairs; a monotonicity check of the grades; the verdict under moved thresholds;
and the screen applied to one random half of the items with the measurement on the other half (cross-fitting), three splits, S8 on.

Per run directory (dataset_block): direct answers are separated from format compliance (accuracy among answered items, unfinished
replies at the token limit, and the AUROC when they are counted as correct); the AUROC on answered items only; a bootstrap that
resamples within subjects; the AUROC without duplicate questions; the AUROC on the items the pilot never saw; and, for open questions,
the AUROC under stricter correctness definitions (exact match, committed answers only, single-answer questions only).
"""

import numpy as np
from blocks import auroc_block, r4

from probe_validity import metrics as M
from probe_validity import report
from probe_validity.crossfit import crossfit, subset_cell
from probe_validity.scoring import score_open
from probe_validity.screen import screen
from probe_validity.text import normalize_answer

PILOT_ITEMS = {"mmlu": 500}  # the E10 pilot used the first 500 items of the MMLU manifest, which the sweep repeats
CROSSFIT_SEEDS = (0, 1, 2)
PROBES = ("yesno", "letters", "digits", "words")


def _round(obj):
    """JSON-safe copy: floats rounded, NaN to None, numpy scalars to Python."""
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return r4(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def crossfit_summary(cell, cfg, seeds=CROSSFIT_SEEDS) -> dict:
    """Screen one half, measure the other, swap; for each seed. The held-out AUROC of the folds whose screening half passed, and of the
    others, is what a claim about 'cells that pass' may use; the screening half's own AUROC is shown to expose the selection effect."""
    folds = []
    for seed in seeds:
        for f in crossfit(cell, cfg, seed, with_s8=True):
            folds.append({"seed": seed, **{k: f[k] for k in ("fold", "verdict", "label", "auroc_screen_half", "auroc_heldout")}})

    def mean(rows, key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return r4(np.mean(vals)) if vals else None

    passing = [f for f in folds if f["verdict"] == "PASS"]
    others = [f for f in folds if f["verdict"] != "PASS"]
    return {"n_folds": len(folds), "n_pass": len(passing), "heldout_auroc_when_pass": mean(passing, "auroc_heldout"),
            "screen_half_auroc_when_pass": mean(passing, "auroc_screen_half"), "heldout_auroc_when_not_pass": mean(others, "auroc_heldout"),
            "folds": _round(folds)}


def cell_diagnostics(cell, result: dict, cfg, fresh_from: int | None = None, answered=None) -> dict:
    v = report.variants(result, cell.correct, cell.grade, cell.scale.n, cfg)
    out = {"tiers": report.tiers(result["verdicts"]), "variants": v["overall"], "s4_own_pairs": v["s4_own_pairs"], "monotonicity": _round(v["monotonicity"]),
           "threshold_sensitivity": report.threshold_sensitivity(cell, cfg, result["verdicts"]["S8"]), "crossfit": crossfit_summary(cell, cfg)}
    if answered is not None and not np.all(answered):  # the primary target counts a missing answer as incorrect; this is the same screen on answered items only
        keep = np.flatnonzero(answered)
        sub = screen(subset_cell(cell, keep), cfg, with_s8=True)
        out["answered_only"] = {"coverage": r4(float(np.mean(answered))), "n_items": int(len(keep)), "overall": sub["overall"], "label": sub["label"], "verdicts": sub["verdicts"],
                                "changes_the_verdict": sub["overall"] != result["overall"]}
    if fresh_from:
        idx = np.arange(fresh_from, len(cell.correct))
        sub = screen(subset_cell(cell, idx), cfg, with_s8=True)
        out["items_not_in_pilot"] = {"n": int(len(idx)), "overall": sub["overall"], "label": sub["label"], "verdicts": sub["verdicts"]}
    return out


def _question_key(item) -> str:
    return normalize_answer(item.stem + " " + " ".join(item.options or ()))


def dataset_block(items, direct, direct_rows, grade_by: dict, dataset: str | None, n_boot: int) -> dict:
    n = len(items)
    correct, answered = direct.correct, direct.valid
    unfinished = np.array([r is not None and r.get("finish") == "length" and not answered[i] for i, r in enumerate(direct_rows)])
    out = {"answered": {"rate": r4(answered.mean()), "n": int(answered.sum()), "n_unfinished": int(unfinished.sum()),
                        "accuracy_among_answered": r4(correct[answered].mean()) if answered.any() else None,
                        "accuracy_if_unfinished_were_correct": r4(correct.mean() + unfinished.mean())}}
    strata = np.array([it.group for it in items])
    seen, keep = set(), np.ones(n, dtype=bool)
    for i, it in enumerate(items):
        key = _question_key(it)
        keep[i] = key not in seen
        seen.add(key)
    pilot = PILOT_ITEMS.get(dataset or "", 0)
    out["duplicates"] = {"n_duplicate_items": int((~keep).sum())}
    best = correct.copy()
    best[unfinished] = 1
    probes = {}
    for name, g in grade_by.items():
        probe, variant = name.rsplit("_", 1)
        if probe not in PROBES or variant not in ("raw", "embedded", "stem"):
            continue
        ok = ~np.isnan(g)
        entry = {"answered_only": auroc_block(correct[ok & answered], g[ok & answered], 21, n_boot),
                 "unfinished_counted_correct": auroc_block(best[ok], g[ok], 22, n_boot) if unfinished.any() else None}
        if ok.sum() >= 30 and len(set(correct[ok])) == 2 and len(set(g[ok])) > 1:
            c, s, st = correct[ok], g[ok], strata[ok]
            boot = M.bootstrap_stratified(lambda idx: M.auroc(c[idx], s[idx]), st, n_boot, 23)
            entry["stratified_bootstrap"] = {"auroc": r4(boot["estimate"]), "ci": [r4(boot["lo"]), r4(boot["hi"])]}
        if (~keep).any():
            entry["without_duplicates"] = auroc_block(correct[ok & keep], g[ok & keep], 24, n_boot)
        if pilot:
            tail = np.arange(n) >= pilot
            entry["items_not_in_pilot"] = auroc_block(correct[ok & tail], g[ok & tail], 25, n_boot)
        probes[name] = entry
    out["probes"] = probes
    if items[0].options is None:  # open questions: stricter readings of "correct"
        scored = [score_open(r["text"], it.answers) if r else None for r, it in zip(direct_rows, items)]
        exact = np.array([s.exact if s else 0 for s in scored])
        committed = np.array([int(s.correct and not s.list_like) if s else 0 for s in scored])
        single = np.array([len({normalize_answer(a) for a in it.answers}) == 1 for it in items])
        out["open_answers"] = {"exact_match_accuracy": r4(exact.mean()), "committed_accuracy": r4(committed.mean()), "primary_accuracy": r4(correct.mean()),
                               "list_like_rate": r4(np.mean([s.list_like for s in scored if s])), "single_answer_share": r4(single.mean()), "probes": {}}
        for name, g in grade_by.items():
            if name.endswith("_raw") and name.rsplit("_", 1)[0] in PROBES:
                ok = ~np.isnan(g)
                out["open_answers"]["probes"][name] = {"exact_match": auroc_block(exact[ok], g[ok], 26, n_boot), "committed": auroc_block(committed[ok], g[ok], 27, n_boot),
                                                       "single_answer_items": auroc_block(correct[ok & single], g[ok & single], 28, n_boot)}
    return out
