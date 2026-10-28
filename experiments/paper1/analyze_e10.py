"""Summarise E10 runs: valid-reply rates, usage, AUROC on valid replies, the first-token-probability baseline, label
copying, and what the legacy parsers did to the same replies.

    python experiments/paper1/analyze_e10.py                 # every model under runs/e10
    python experiments/paper1/analyze_e10.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import collections
import json
import math

import numpy as np
from common import MANIFESTS, OPTION_LABELS, PROBES, RUNS, e10_conditions, model_slug, split_condition

from probe_validity import metrics as M
from probe_validity.generate import read_jsonl
from probe_validity.loaders import load_manifest, load_mmlu, select
from probe_validity.parse import answer_scale, legacy_graded, legacy_yes, parse_lenient, parse_mcq, parse_strict
from probe_validity.scoring import legacy_mmlu_correct

N_BOOT = 1000


def rows_of(model_dir, name, n):
    by_i = {r["i"]: r for r in read_jsonl(model_dir / f"{name}.jsonl") if r.get("condition") == name}
    if len(by_i) != n:
        raise RuntimeError(f"{model_dir.name}/{name}: {len(by_i)} of {n} rows")
    return [by_i[i] for i in range(n)]


def ci(boot):
    return [round(boot["lo"], 4), round(boot["hi"], 4)]


def auroc_with_ci(correct, score, seed=0):
    if len(set(correct)) < 2 or len(set(score)) < 2:
        return None
    boot = M.bootstrap(lambda idx: M.auroc(correct[idx], score[idx]), len(correct), N_BOOT, seed)
    return {"auroc": round(boot["estimate"], 4), "ci": ci(boot), "gamma": round(M.gamma(correct, score), 4),
            "p_tie": round(M.tie_share(correct, score), 4), "n": len(correct)}


def logit_grade(rows, scale):
    """Expected grade under the first-token label probabilities, renormalised over the scale's labels."""
    lp = np.array([[r["label_lp"][lab] for lab in scale.labels] for r in rows], dtype=float)
    mass = np.exp(lp).sum(1)
    p = np.exp(lp) / np.maximum(mass[:, None], 1e-300)
    grades = np.array([scale.grade_of(lab) for lab in scale.labels], dtype=float)
    return p @ grades, mass


def summarise(model_dir, items):
    n, gold = len(items), [it.gold for it in items]
    out = {"model_dir": model_dir.name, "n": n, "direct": {}, "probes": {}}

    clean = rows_of(model_dir, "direct_clean", n)
    parsed = {name: [parse_mcq(r["text"], OPTION_LABELS, it.options) for r, it in zip(rows_of(model_dir, name, n), items)]  # the option texts, as in the sweep
              for name in ("direct_clean", "direct_legacy")}
    for name, ps in parsed.items():
        rows = rows_of(model_dir, name, n)
        first = [int(p.label == g) for p, g in zip(ps, gold)]
        strict = [int(p.tier == "strict" and p.label == g) for p, g in zip(ps, gold)]
        legacy = [legacy_mmlu_correct(r["text"], g) for r, g in zip(rows, gold)]
        out["direct"][name] = {
            "valid_rate": round(float(np.mean([p.valid for p in ps])), 4),
            "tiers": dict(collections.Counter(p.tier for p in ps)),
            "accuracy_first_label": round(float(np.mean(first)), 4),
            "accuracy_strict_only": round(float(np.mean(strict)), 4),
            "accuracy_legacy_scorer": round(float(np.mean(legacy)), 4),
            "finish_length_rate": round(float(np.mean([r["finish"] == "length" for r in rows])), 4),
            "mean_new_tokens": round(float(np.mean([r["n_new"] for r in rows])), 2),
        }
    correct = np.array([int(p.label == g) for p, g in zip(parsed["direct_clean"], gold)])
    answer = [p.label for p in parsed["direct_clean"]]
    valid_by_cond = {}

    for name in e10_conditions():
        if name.startswith("direct"):
            continue
        probe, variant = split_condition(name)
        scale = PROBES[probe][0]
        rows = rows_of(model_dir, name, n)
        replies = [r["text"] for r in rows]
        strict = [parse_strict(t, scale) for t in replies]
        valid = np.array([p.valid for p in strict])
        valid_by_cond[name] = valid
        grade = np.array([scale.grade_of(p.label) if p.valid else np.nan for p in strict], dtype=float)
        lo, hi = M.wilson_ci(int(valid.sum()), n)
        m = {
            "n_valid": int(valid.sum()), "valid_rate": round(float(valid.mean()), 4), "valid_ci": [round(lo, 4), round(hi, 4)],
            "lenient_rate": round(float(np.mean([parse_lenient(t, scale).valid for t in replies])), 4),
            "invalid_reasons": dict(collections.Counter(p.reason for p in strict if not p.valid).most_common(5)),
            "mcq_answer_rate": round(float(np.mean([parse_strict(t, answer_scale(OPTION_LABELS)).valid for t in replies])), 4),
            "usage": M.usage(grade[valid], scale.n),
            "mean_new_tokens": round(float(np.mean([r["n_new"] for r in rows])), 2),
            "finish_length_rate": round(float(np.mean([r["finish"] == "length" for r in rows])), 4),
        }
        m["usage"]["entropy"] = None if math.isnan(m["usage"]["entropy"]) else round(m["usage"]["entropy"], 4)
        if valid.sum() >= 30:
            m["on_valid"] = auroc_with_ci(correct[valid], grade[valid])
        score, mass = logit_grade(rows, scale)
        m["first_token_label_mass"] = round(float(mass.mean()), 4)
        m["logit_baseline"] = auroc_with_ci(correct, score, seed=1)
        if probe == "yesno":
            yes = np.array([legacy_yes(t) for t in replies])
            m["legacy_yes_rate"] = round(float(yes.mean()), 4)
            m["legacy_dprime_all_items"] = round(M.dprime_type2(correct, yes), 4)
            if valid.sum() >= 30:
                m["strict_dprime_valid_only"] = round(M.dprime_type2(correct[valid], (grade[valid] == 1).astype(int)), 4)
        if probe == "letters":
            legacy = [legacy_graded(t) for t in replies]
            lg = np.array([g for g, _ in legacy], dtype=float)
            m["legacy_parsed_ok_rate"] = round(float(np.mean([ok for _, ok in legacy])), 4)
            m["legacy_grade_equals_strict_on_valid"] = round(float(np.mean(lg[valid] == grade[valid])), 4) if valid.any() else None
            m["legacy_all_items"] = auroc_with_ci(correct, lg, seed=2)
            both = [i for i in range(n) if valid[i] and answer[i] is not None]
            if both:
                conf = [strict[i].label for i in both]
                ans = [answer[i] for i in both]
                pc, pa = collections.Counter(conf), collections.Counter(ans)
                m["copy_rate"] = round(float(np.mean([c == a for c, a in zip(conf, ans)])), 4)
                m["copy_chance"] = round(sum(pc[k] * pa[k] for k in pc) / len(both) ** 2, 4)
                m["copy_n"] = len(both)
        out["probes"][name] = m

    for probe in PROBES:
        raw, emb = valid_by_cond[f"{probe}_raw"].astype(float), valid_by_cond[f"{probe}_embedded"].astype(float)
        boot = M.bootstrap(lambda idx: float((raw[idx] - emb[idx]).mean()), n, N_BOOT, 3)
        out["probes"][f"{probe}_raw"]["valid_gain_over_embedded"] = {"diff": round(boot["estimate"], 4), "ci": ci(boot)}
    return out


def table(summary):
    lines = [f"{summary['model_dir']}  (n={summary['n']})"]
    for name, d in summary["direct"].items():
        lines.append(f"  {name:14} valid {d['valid_rate']:.3f}  acc(first-label) {d['accuracy_first_label']:.3f}  "
                     f"acc(legacy scorer) {d['accuracy_legacy_scorer']:.3f}  tiers {d['tiers']}  hit-length {d['finish_length_rate']:.2f}")
    lines.append(f"  {'condition':17} {'valid':>6} {'lenient':>8} {'answers-MCQ':>11} {'entropy':>8} {'AUROC(valid)':>14} {'AUROC(logit)':>13} {'label mass':>10}")
    for name, m in summary["probes"].items():
        on, lg = m.get("on_valid"), m.get("logit_baseline")
        lines.append(f"  {name:17} {m['valid_rate']:6.3f} {m['lenient_rate']:8.3f} {m['mcq_answer_rate']:11.3f} "
                     f"{(m['usage']['entropy'] if m['usage']['entropy'] is not None else float('nan')):8.3f} "
                     f"{(str(on['auroc']) + ' (n=' + str(on['n']) + ')') if on else '-':>14} "
                     f"{lg['auroc'] if lg else '-':>13} {m['first_token_label_mass']:10.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    items = select(load_mmlu("test"), load_manifest(MANIFESTS / f"mmlu_test_s0_n{args.n}.json"))
    dirs = [RUNS / "e10" / model_slug(args.model)] if args.model else sorted(p for p in (RUNS / "e10").iterdir() if p.is_dir())
    for d in dirs:
        summary = summarise(d, items)
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
        print(table(summary), "\n")


if __name__ == "__main__":
    main()
