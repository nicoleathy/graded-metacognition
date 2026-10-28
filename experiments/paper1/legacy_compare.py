"""What the earlier draft's pipeline would have reported on the same replies, next to the strict pipeline.

The legacy pipeline counts every reply without "yes" as a "no", reads word-initial a-d as grades and imputes grade 1 when nothing
matches, scores multiple-choice answers on the upper-cased reply (so the article "a" is a choice), and summarises a probe with the
rescaled Goodman-Kruskal gamma (gamma* = (gamma + 1) / 2, 0.5 = chance). The strict pipeline never imputes, scores answers with the
article-safe parser, and reports AUROC, Somers' D and the share of tied pairs. Nothing here is taken from the earlier draft's numbers:
both sides are computed from the stored replies of one run, so a difference is a difference of method.

Two contrasts are kept apart (a review of 27 October 2026 showed they had been mixed):
* metric only: the same compliant replies, the same strict correctness labels and the same items, summarised by rescaled gamma and by AUROC.
  Their difference follows from the tie share t alone: gamma* - AUROC = (AUROC - 0.5) * t / (1 - t).
* full pipeline: the conventional pipeline end to end (legacy answer prompt and scorer, legacy parsing, imputation, gamma*) against the strict one.
  Here the answer prompt, the answer scorer, the confidence parser and the metric all differ, so the gap cannot be attributed to one of them.
"""

import numpy as np
from blocks import auroc_block, gamma_star, r4

from probe_validity.parse import legacy_graded, legacy_yes
from probe_validity.records import grades, read_condition
from probe_validity.scales import PROBE_SCALES
from probe_validity.scoring import legacy_mmlu_correct, score_open

MIN_N = 30


def legacy_correctness(directory, items):
    """Correctness under the legacy scorer: MMLU answers on the legacy direct prompt (A-D read from the upper-cased reply), open
    answers by plain substring. None for ten-option sets: the legacy scorer cannot name E-J."""
    n = len(items)
    if items[0].options is None:
        rows = read_condition(directory, "direct_clean", n)
        return np.array([score_open(r["text"], it.answers).legacy if r else 0 for r, it in zip(rows, items)])
    if len(items[0].options) != 4:
        return None
    name = "direct_legacy" if (directory / "direct_legacy.jsonl").exists() else "direct_clean"
    rows = read_condition(directory, name, n)
    return np.array([legacy_mmlu_correct(r["text"], it.gold) if r else 0 for r, it in zip(rows, items)])


def block(directory, items, direct, n_boot: int) -> dict | None:
    n = len(items)
    legacy_correct = legacy_correctness(directory, items)
    if legacy_correct is None:
        return None
    out = {"accuracy_legacy_pipeline": r4(legacy_correct.mean()), "accuracy_strict": r4(direct.correct.mean())}
    for probe in ("yesno", "letters"):
        variant = "embedded" if (directory / f"{probe}_embedded.jsonl").exists() else "raw"  # the original prompt: embedded on MMLU, plain otherwise
        if not (directory / f"{probe}_{variant}.jsonl").exists():
            continue
        rows = read_condition(directory, f"{probe}_{variant}", n)
        replies = [r["text"] if r else "" for r in rows]
        g = grades(rows, PROBE_SCALES[probe])
        valid = ~np.isnan(g)
        if probe == "yesno":
            legacy_score = np.array([legacy_yes(t) for t in replies], dtype=float)  # a reply without "yes" is a "no", whatever it says
            parsed = valid
        else:
            pairs = [legacy_graded(t) for t in replies]
            legacy_score = np.array([x for x, _ in pairs], dtype=float)  # grade 1 imputed when nothing matches
            parsed = np.array([ok for _, ok in pairs])
        correct = direct.correct
        legacy = {"prompt_variant": variant, "p_positive": r4(legacy_score.mean()) if probe == "yesno" else None, "mean_grade": r4(legacy_score.mean()),
                  "parsed_rate": r4(parsed.mean()), "n_parsed": int(parsed.sum()),
                  "gamma_star_imputed": gamma_star(legacy_correct, legacy_score),
                  "gamma_star_parsed_only": gamma_star(legacy_correct[parsed], legacy_score[parsed]) if parsed.sum() >= MIN_N else None}
        strict = {"valid_rate": r4(valid.mean()), "n_valid": int(valid.sum()),
                  "gamma_star": gamma_star(correct[valid], g[valid]) if valid.sum() >= MIN_N else None,
                  "auroc": auroc_block(correct[valid], g[valid], 11, n_boot) if valid.sum() >= MIN_N else None}
        if probe == "yesno":
            legacy["delta_yes"] = r4(legacy["p_positive"] - legacy_correct.mean())  # the earlier draft's check: positive-response rate minus accuracy
            if valid.any():
                strict["p_positive_valid"] = r4((g[valid] == 1).mean())
                strict["delta_yes"] = r4((g[valid] == 1).mean() - correct[valid].mean())
        a, gs_pipeline = strict["auroc"], legacy["gamma_star_parsed_only"]
        gs_strict = strict["gamma_star"]
        if probe == "yesno":  # the same replies counted the conventional way (non-"yes" is "no") against the strict accuracy: only the imputation differs
            strict["delta_yes_imputed_strict_labels"] = r4(legacy["p_positive"] - correct.mean())
        out[probe] = {"legacy": legacy, "strict": strict,
                      "gamma_star_minus_auroc": r4(gs_strict - a["auroc"]) if a and gs_strict is not None else None,  # metric only: identical replies and labels
                      "gamma_star_minus_auroc_full_pipeline": r4(gs_pipeline - a["auroc"]) if a and gs_pipeline is not None else None,
                      "tie_identity_error": r4(abs((gs_strict - a["auroc"]) - (a["auroc"] - 0.5) * a["p_tie"] / (1 - a["p_tie"]))) if a and gs_strict is not None and a["p_tie"] < 1 else None,
                      "tie_share": a["p_tie"] if a else None}
    return out
