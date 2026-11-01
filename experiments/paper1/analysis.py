"""Summaries of one run directory (one model on one dataset), built from the stored rows with the shared library code.
Every number that reaches the paper comes out of summarise_directory(); nothing is computed by hand."""

import collections
import json
from pathlib import Path

import numpy as np
from blocks import N_BOOT, auroc_block, gamma_star, paired_auroc_diff, r4, refit_paired_diff  # noqa: F401
from common import ROOT

import legacy_compare
import robustness
from probe_validity import metrics as M
from probe_validity.parse import answer_scale, legacy_graded, legacy_yes, parse_lenient, parse_strict
from probe_validity.prompts import Item
from probe_validity.records import ambiguous_labels, build_cell, direct_outcomes, first_token_score, grades, read_condition
from probe_validity.scales import FACTORIAL, PROBE_SCALES
from probe_validity.scoring import legacy_mmlu_correct, score_open
from probe_validity.screen import load_frozen, screen

PROBE_ORDER = ("yesno", "letters", "digits", "words")
FROZEN = ROOT / "experiments/paper1/frozen/screen_config.json"


def comparator_block(correct, verbal, score, mass, n_levels, multi_token_labels=False, seed=0, n_boot=N_BOOT):
    """The same-prompt first-content-token score against the verbalized grade, on the items where both exist.

    The verbalized grade is missing where the reply is not compliant; the score is missing only where no label log-probabilities
    exist. Comparing AUROCs over different items would confound the readouts with the items they leave out, so the paired
    difference uses the jointly available items, the score's own full-set AUROC is reported next to it, and the score is also
    coarsened to the verbal scale's number of levels, which separates "the readout is worse" from "four labels carry less
    information than a continuous score"."""
    correct, verbal, score = np.asarray(correct), np.asarray(verbal, dtype=float), np.asarray(score, dtype=float)
    have = np.isfinite(score)
    mass_ok = np.asarray(mass, dtype=float)[have]
    out = {"n_items": int(len(correct)), "n_score_available": int(have.sum()), "n_verbal_valid": int(np.isfinite(verbal).sum()),
           "label_mass_mean": r4(mass_ok.mean()) if mass_ok.size else None, "label_mass_p10": r4(np.percentile(mass_ok, 10)) if mass_ok.size else None,
           "approximation": "first token of a multi-token label, not the probability of the full label" if multi_token_labels else "the label is one token",
           "score_all_items": auroc_block(correct[have], score[have], 1, n_boot) if have.sum() >= 30 else None}
    common = have & np.isfinite(verbal)
    out["n_common"] = int(common.sum())
    if common.sum() < 30 or len(set(correct[common])) < 2:
        return out
    c, v, f = correct[common], verbal[common], score[common]
    coarse = M.coarsen_equal_mass(f, n_levels)
    out.update({
        "verbal_on_common": auroc_block(c, v, 2, n_boot), "score_on_common": auroc_block(c, f, 3, n_boot),
        "score_minus_verbal": paired_auroc_diff(c, f, v, 4, n_boot),
        "coarsened_score_on_common": auroc_block(c, coarse, 5, n_boot),
        "coarsened_score_minus_verbal": refit_paired_diff(c, f, v, lambda score, verbal: M.coarsen_equal_mass(score, n_levels), 6, n_boot),
    })
    matched = M.match_marginal(f, v)  # the score forced to use the verbal readout's own label frequencies
    out["usage_matched_score_on_common"] = auroc_block(c, matched, 7, n_boot)
    out["usage_matched_score_minus_verbal"] = refit_paired_diff(c, f, v, M.match_marginal, 8, n_boot)  # refitted in every resample
    return out


def direct_summary(directory, items, name, labels):
    n = len(items)
    rows = read_condition(directory, name, n)
    d = direct_outcomes(rows, items)
    out = {"valid_rate": r4(d.valid.mean()), "accuracy": r4(d.correct.mean()),
           "hit_length_rate": r4(np.mean([r["finish"] == "length" for r in rows if r])),
           "mean_new_tokens": r4(np.mean([r["n_new"] for r in rows if r]))}
    lo, hi = M.wilson_ci(int(d.correct.sum()), len(items))
    out["accuracy_ci"] = [r4(lo), r4(hi)]
    out["accuracy_among_valid"] = r4(d.correct[d.valid].mean()) if d.valid.any() else None  # correctness separated from format compliance
    out["n_unfinished"] = int(sum(1 for i, r in enumerate(rows) if r and r["finish"] == "length" and not d.valid[i]))  # cut at the token limit, no answer
    if items[0].options is not None:
        out["chance_accuracy"] = r4(np.mean([1 / len(it.options) for it in items]))
        out["accuracy_above_chance"] = bool(lo > out["chance_accuracy"])  # False: the model is at the floor on this set
        if len(items[0].options) == 4:  # the legacy scorer reads A-D only: on ten options it would score every E-J answer as wrong
            out["accuracy_legacy_scorer"] = r4(np.mean([legacy_mmlu_correct(r["text"], it.gold) for r, it in zip(rows, items)]))
    else:
        out["accuracy_legacy_inclusion"] = r4(np.mean([score_open(r["text"], it.answers).legacy for r, it in zip(rows, items)]))
        out["hedged_rate"] = r4(np.mean([score_open(r["text"], it.answers).hedged for r, it in zip(rows, items)]))
    return out, d


def summarise_directory(directory: Path, items: list[Item], answer_labels: tuple[str, ...], n_boot: int = N_BOOT) -> dict:
    n = len(items)
    cfg = load_frozen(FROZEN)
    out = {"n": n, "answer_labels": list(answer_labels), "direct": {}, "probes": {}, "screens": {}, "pairs": {}}
    out["direct"]["clean"], direct = direct_summary(directory, items, "direct_clean", answer_labels)
    legacy_direct = None
    if (directory / "direct_legacy.jsonl").exists():
        out["direct"]["legacy_prompt"], legacy_direct = direct_summary(directory, items, "direct_legacy", answer_labels)
        legacy_correct = np.array([legacy_mmlu_correct(r["text"], it.gold) for r, it in zip(read_condition(directory, "direct_legacy", n), items)])
    correct = direct.correct
    is_mcq = items[0].options is not None
    valid_by, score_by, grade_by = {}, {}, {}

    extra = ["letters_reversed_raw", "letters_perm_raw", "letters_flip_order_raw", "letters_flip_labels_raw", "letters_optdigits", "digits_optdigits"]
    for name in [f"{p}_{v}" for p in PROBE_ORDER for v in ("raw", "embedded", "stem")] + extra:
        if not (directory / f"{name}.jsonl").exists():
            continue
        probe, variant = name.rsplit("_", 1)
        scale = PROBE_SCALES[probe]
        rows = read_condition(directory, name, n)
        replies = [r["text"] if r else "" for r in rows]
        strict = [parse_strict(t, scale) for t in replies]
        g = grades(rows, scale)
        valid = ~np.isnan(g)
        valid_by[name], grade_by[name] = valid, g
        lo, hi = M.wilson_ci(int(valid.sum()), n)
        score, mass = first_token_score(rows, scale)
        score_by[name] = score
        usage = M.usage(g[valid], scale.n)
        m = {"n_valid": int(valid.sum()), "valid_rate": r4(valid.mean()), "valid_ci": [r4(lo), r4(hi)],
             "lenient_rate": r4(np.mean([parse_lenient(t, scale).valid for t in replies])),
             "invalid_reasons": dict(collections.Counter(p.reason for p in strict if not p.valid).most_common(5)),
             "usage": {"counts": usage["counts"], "entropy": r4(usage["entropy"]), "top_share": r4(usage["top_share"])},
             "mean_new_tokens": r4(np.mean([r["n_new"] for r in rows if r])),
             "hit_length_rate": r4(np.mean([r["finish"] == "length" for r in rows if r])),
             "on_valid": auroc_block(correct[valid], g[valid], 0, n_boot) if valid.any() else None,
             "first_token": {"label_mass": r4(np.nanmean(mass)), "auroc": auroc_block(correct, score, 1, n_boot),
                             "ambiguous_labels": ambiguous_labels(rows, scale)}}
        m["first_token"]["comparator"] = comparator_block(correct, g, score, mass, scale.n, multi_token_labels=probe == "words", n_boot=n_boot)
        if m["first_token"]["ambiguous_labels"]:
            m["first_token"]["auroc"] = None  # labels share a first token: the baseline would be meaningless
            m["first_token"]["comparator"] = {"unavailable": "labels share a first token"}
        if is_mcq:
            m["mcq_answer_rate"] = r4(np.mean([parse_strict(t, answer_scale(it.option_labels)).valid for t, it in zip(replies, items)]))
        if probe == "yesno":
            yes = np.array([legacy_yes(t) for t in replies])
            m["legacy_yes_rate"] = r4(yes.mean())
            m["dprime_legacy_parser_clean_labels"] = r4(M.dprime_type2(correct, yes))
            if valid.sum() >= 30:
                m["dprime_strict_valid_only"] = r4(M.dprime_type2(correct[valid], (g[valid] == 1).astype(int)))
            if legacy_direct is not None and variant == "embedded":
                m["dprime_legacy_pipeline"] = r4(M.dprime_type2(legacy_correct, yes))  # embedded prompt, legacy parser, legacy scorer
        if probe == "letters":
            lg = [legacy_graded(t) for t in replies]
            lgrade = np.array([x for x, _ in lg], dtype=float)
            m["legacy_parser"] = {"parsed_ok_rate": r4(np.mean([ok for _, ok in lg])),
                                  "agrees_with_strict_on_valid": r4(np.mean(lgrade[valid] == g[valid])) if valid.any() else None,
                                  "auroc_all_items": auroc_block(correct, lgrade, 2, n_boot)}
            both = [i for i in range(n) if valid[i] and direct.label[i] is not None and is_mcq]
            if both:
                conf = [strict[i].label for i in both]
                ans = [direct.label[i] for i in both]
                pc, pa = collections.Counter(conf), collections.Counter(ans)
                m["copy"] = {"rate": r4(np.mean([c == a for c, a in zip(conf, ans)])),
                             "chance": r4(sum(pc[k] * pa[k] for k in pc) / len(both) ** 2), "n": len(both)}
        if is_mcq:  # the earlier draft's association statistic: how strongly the reply label depends on the model's own answer label
            pair = [(strict[i].label, direct.label[i]) for i in range(n) if valid[i] and direct.label[i] is not None]
            if len(pair) >= 30:
                cv = M.cramers_v([a for a, _ in pair], [b for _, b in pair])
                m["association_with_answer_label"] = {"cramers_v": r4(cv["v"]), "cramers_v_corrected": r4(cv["v_corrected"]), "p": cv["p"], "dof": cv["dof"], "n": cv["n"]}
        out["probes"][name] = m

    def gain(better: str, worse: str, seed: int):
        a, b = valid_by[better].astype(float), valid_by[worse].astype(float)
        boot = M.bootstrap(lambda idx: float((a[idx] - b[idx]).mean()), n, n_boot, seed)
        return {"diff": r4(boot["estimate"]), "ci": [r4(boot["lo"]), r4(boot["hi"])]}

    for probe in PROBE_ORDER:
        raw, emb, stem = f"{probe}_raw", f"{probe}_embedded", f"{probe}_stem"
        if raw in valid_by and emb in valid_by:  # instruction capture: removing the embedded answer-format instruction
            out["probes"][raw]["valid_gain_over_embedded"] = gain(raw, emb, 3)
        if raw in valid_by and stem in valid_by:  # option-list capture: removing the lettered options as well
            out["probes"][stem]["valid_gain_over_raw"] = gain(stem, raw, 5)

    meta_path = directory / "meta.json"
    dataset = json.loads(meta_path.read_text()).get("dataset") if meta_path.exists() else None
    fresh_from = robustness.PILOT_ITEMS.get(dataset or "")

    def record(name, cell):
        result = screen(cell, cfg, with_s8=True)
        out["screens"][name] = {"overall": result["overall"], "complete": result["complete"], "label": result["label"], "verdicts": result["verdicts"],
                                "values": {k: _plain(c.value) for k, c in result["checks"].items()},
                                "diagnostics": robustness.cell_diagnostics(cell, result, cfg, fresh_from, direct.valid)}

    for probe in PROBE_ORDER:
        name = f"{probe}_raw"
        if name in valid_by:
            record(name, build_cell(directory, items, probe, answer_labels, direct))

    for probe in PROBE_ORDER:
        name = f"{probe}_stem"
        if name in valid_by:
            record(name, build_cell(directory, items, probe, (), direct, variant="stem"))  # no options shown: nothing to collide with

    out["robustness"] = robustness.dataset_block(items, direct, read_condition(directory, "direct_clean", n), grade_by, dataset, n_boot)
    out["legacy_pipeline"] = legacy_compare.block(directory, items, direct, n_boot)
    out["factorial"] = factorial_block(directory, items, correct, n_boot)
    out["collision_control"] = collision_block(directory, items, direct, n_boot) if is_mcq else None

    raw = [f"{p}_raw" for p in PROBE_ORDER if f"{p}_raw" in grade_by]
    for i, a in enumerate(raw):
        for b in raw[i + 1:]:
            out["pairs"][f"{a}-{b}"] = paired_auroc_diff(correct, grade_by[a], grade_by[b], 4, n_boot)
    return out


def _plain(value):
    """JSON-safe copy of a check's value dict."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return r4(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def position_of(scale_name: str, label: str | None) -> int:
    """Position index (0-based) a confidence label stands for when it is read as an option label: A = 1) = 0."""
    if label is None:
        return -1
    return "ABCD".index(label) if scale_name.startswith("letters") else int(label) - 1


def factorial_block(directory: Path, items: list[Item], correct, n_boot: int = N_BOOT):
    """Label x presentation order. letters (A = certain, certain printed first), letters_flip_order (labels kept, order flipped),
    letters_flip_labels (order kept, labels flipped), letters_reversed (both flipped). A responder that takes the first printed
    line is stable across the label flip and unstable across the order flip; a responder that takes a label token is the reverse."""
    names = {s: f"{s}_raw" for s in FACTORIAL}
    if not all((directory / f"{c}.jsonl").exists() for c in names.values()):
        return None
    n = len(items)
    g, cells = {}, {}
    for scale_name, cond in names.items():
        scale = PROBE_SCALES[scale_name]
        rows = read_condition(directory, cond, n)
        g[scale_name] = grades(rows, scale)
        labels = [parse_strict(r["text"], scale).label if r else None for r in rows]
        valid = [lab for lab in labels if lab is not None]
        first = scale.lines()[0].split(scale.separator)[0]
        valid_mask = ~np.isnan(g[scale_name])
        cells[scale_name] = {
            "n_valid": len(valid), "first_printed_label": first,
            "first_line_rate": r4(valid.count(first) / len(valid)) if valid else None,
            "label_A_rate": r4(valid.count("A") / len(valid)) if valid else None,
            "grade_counts": M.usage(g[scale_name][valid_mask], scale.n)["counts"],
            "auroc": auroc_block(np.asarray(correct)[valid_mask], g[scale_name][valid_mask], 6, n_boot) if valid_mask.any() else None,
        }
    base, contrasts = g["letters"], {}
    for other in FACTORIAL[1:]:
        both = ~np.isnan(base) & ~np.isnan(g[other])
        contrasts[other] = {
            "kappa_with_letters": r4(M.weighted_kappa(base[both].astype(int), g[other][both].astype(int), list(range(4)))) if both.any() else None,
            "same_grade_rate": r4(np.mean(base[both] == g[other][both])) if both.any() else None,
            "share_of_grades_changed": r4(1 - np.mean(base[both] == g[other][both])) if both.any() else None,
            "n_jointly_valid": int(both.sum()),
            "auroc_diff_vs_letters": paired_auroc_diff(correct, g[other], base, 7, n_boot),
        }
    return {"cells": cells, "contrasts": contrasts}


def collision_block(directory: Path, items: list[Item], direct, n_boot: int = N_BOOT):
    """The answer-label control, a 2 x 2 holding each confidence scale fixed: confidence scale (letters, digits) x option labels
    (letters, digits). Collision cells: letters/letters and digits/digits; clean cells: letters/digits and digits/letters.
    "copy_rate" is the answer-confidence agreement: the confidence label stands for the same position as the model's own answer
    (A = 1) = 0). Agreement that remains in the clean cells is position-matched agreement without a shared label; it is reported
    as a fraction of the shared-label agreement (residual_agreement) and is not a causal share. The effects are per scale: they
    can have opposite signs, so the mean over scales must not be read as the effect."""
    cells = {"letters/letters": ("letters", "letters_raw"), "letters/digits": ("letters", "letters_optdigits"),
             "digits/letters": ("digits", "digits_raw"), "digits/digits": ("digits", "digits_optdigits")}
    if not all((directory / f"{c}.jsonl").exists() for _, c in cells.values()):
        return None
    n = len(items)
    answer = np.array([("ABCDEFGHIJ".index(lab) if lab else -1) for lab in direct.label])  # up to ten options; a confidence scale has four levels, so E-J never agree
    correct = direct.correct
    arr = {}
    for key, (scale_name, cond) in cells.items():
        scale = PROBE_SCALES[scale_name]
        rows = read_condition(directory, cond, n)
        conf = np.array([position_of(scale_name, parse_strict(r["text"], scale).label) if r else -1 for r in rows])
        arr[key] = {"grade": grades(rows, scale), "conf": conf}
    common = np.all([(arr[k]["conf"] >= 0) & (answer >= 0) for k in arr], axis=0)  # items valid in all four cells
    out = {"n_common": int(common.sum()), "cells": {}, "effects": {}, "residual_agreement": {}}
    for key, a in arr.items():
        ok = (a["conf"] >= 0) & (answer >= 0)
        pc, pa = np.bincount(a["conf"][ok], minlength=10) / ok.sum(), np.bincount(answer[ok], minlength=10) / ok.sum()
        out["cells"][key] = {
            "n_valid": int((a["conf"] >= 0).sum()), "collision": key.split("/")[0] == key.split("/")[1],
            "copy_rate": r4(np.mean(a["conf"][ok] == answer[ok])), "copy_chance": r4(float(pc @ pa)),
            "auroc": auroc_block(correct[~np.isnan(a["grade"])], a["grade"][~np.isnan(a["grade"])], 8, n_boot),
        }
    for scale, (shared, clean) in {"letters_scale": ("letters/letters", "letters/digits"), "digits_scale": ("digits/digits", "digits/letters")}.items():
        a, b = out["cells"][shared], out["cells"][clean]
        above = a["copy_rate"] - a["copy_chance"]
        # Descriptive, not causal: how much of the agreement between the confidence label and the answer's position is still there
        # when no label is shared. Each ratio uses the cell-level rates, each over the cell's own valid items.
        out["residual_agreement"][scale] = {
            "shared_labels": {"rate": a["copy_rate"], "chance": a["copy_chance"]}, "no_shared_labels": {"rate": b["copy_rate"], "chance": b["copy_chance"]},
            "fraction_of_raw_agreement": r4(b["copy_rate"] / a["copy_rate"]) if a["copy_rate"] > 0 else None,
            "fraction_of_above_chance_agreement": r4((b["copy_rate"] - b["copy_chance"]) / above) if above > 0.02 else None,
            "definition": "(rate - chance) without shared labels / (rate - chance) with shared labels; not the share caused by position",
        }
    if common.sum() < 30:
        return out
    idx = np.flatnonzero(common)
    copy = {k: (arr[k]["conf"][idx] == answer[idx]).astype(float) for k in arr}
    y, g = correct[idx], {k: arr[k]["grade"][idx] for k in arr}

    def effects(sel):
        copy_effect = [copy["letters/letters"][sel].mean() - copy["letters/digits"][sel].mean(),
                       copy["digits/digits"][sel].mean() - copy["digits/letters"][sel].mean()]
        auroc_effect = [M.auroc(y[sel], g["letters/letters"][sel]) - M.auroc(y[sel], g["letters/digits"][sel]),
                        M.auroc(y[sel], g["digits/digits"][sel]) - M.auroc(y[sel], g["digits/letters"][sel])]
        return copy_effect, auroc_effect

    rng = np.random.default_rng(9)
    point = effects(np.arange(len(idx)))
    draws = [effects(rng.integers(0, len(idx), len(idx))) for _ in range(n_boot)]
    for j, name in enumerate(("copy_rate", "auroc")):
        comp = np.array([[d[j][0], d[j][1]] for d in draws])
        mean = comp.mean(axis=1)
        out["effects"][name] = {
            "letters_scale": {"diff": r4(point[j][0]), "ci": [r4(np.percentile(comp[:, 0], 2.5)), r4(np.percentile(comp[:, 0], 97.5))]},
            "digits_scale": {"diff": r4(point[j][1]), "ci": [r4(np.percentile(comp[:, 1], 2.5)), r4(np.percentile(comp[:, 1], 97.5))]},
            "mean": {"diff": r4(np.mean(point[j])), "ci": [r4(np.percentile(mean, 2.5)), r4(np.percentile(mean, 97.5))]},
        }
    return out
