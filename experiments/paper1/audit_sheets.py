"""Sheets for the human audit of parser and scorer decisions, and the agreement statistics once two annotators have filled them in.

The program's decision never appears on an annotator sheet. Each dataset gets a blank sheet per annotator and a key file with the
program's decisions. The sample over-represents the cases where a parser is most likely to be wrong: non-strict answer tiers, invalid
replies, short lists and hedges for open answers, and invalid probe replies.

    python experiments/paper1/audit_sheets.py make --per-dataset 200          # writes experiments/paper1/audit/
    python experiments/paper1/audit_sheets.py score --dataset mmlu --a a.csv --b b.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_sweep import items_for
from common import MANIFESTS, ROOT, RUNS
from protocol import DATASET_TABLE

from probe_validity.loaders import load_manifest
from probe_validity.parse import answer_scale, parse_mcq, parse_strict
from probe_validity.records import read_condition
from probe_validity.scales import PROBE_SCALES
from probe_validity.scoring import score_open

OUT = ROOT / "experiments/paper1/audit"
NONE_LABELS = ("NONE", "AMBIGUOUS")


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa between two raters over the same units (any hashable labels); 1 when both always agree and chance agreement is below 1."""
    if len(a) != len(b) or not a:
        raise ValueError("raters must label the same, non-empty set of units")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def candidates(directory: Path, items: list, dataset: str) -> list[dict]:
    """Every direct answer and every content-only probe reply of one run directory, with the program's decision and a flag for risky cases."""
    n, out = len(items), []
    rows = read_condition(directory, "direct_clean", n)
    for i, (row, item) in enumerate(zip(rows, items)):
        if row is None:
            continue
        if item.options is not None:
            p = parse_mcq(row["text"], item.option_labels, item.options)
            out.append({"task": "answer_label", "i": i, "text": row["text"], "decision": p.label or "NONE", "risky": p.tier != "strict", "item": item})
        else:
            sc = score_open(row["text"], item.answers)
            out.append({"task": "answer_correct", "i": i, "text": row["text"], "decision": "yes" if sc.correct else "no", "risky": bool(sc.hedged or sc.list_like or sc.contains != sc.exact), "item": item})
    for probe in ("yesno", "letters", "digits", "words"):
        scale = PROBE_SCALES[probe]
        for i, row in enumerate(read_condition(directory, f"{probe}_raw", n)):
            if row is None:
                continue
            p = parse_strict(row["text"], scale)
            out.append({"task": f"probe_label_{probe}", "i": i, "text": row["text"], "decision": p.label or "NONE", "risky": not p.valid, "item": items[i]})
    return out


def draw(cands: list[dict], k: int, seed: int) -> list[dict]:
    """k units per dataset: half direct-answer decisions and half probe replies (the whole sample goes to one family if the other is empty),
    and inside each family half from the risky cases and half from the rest."""
    rng = np.random.default_rng(seed)
    take = lambda pool, m: [pool[j] for j in rng.permutation(len(pool))[: min(m, len(pool))]]  # noqa: E731
    families = {"answer": [c for c in cands if c.get("task", "").startswith("answer")], "probe": [c for c in cands if not c.get("task", "").startswith("answer")]}
    present = [f for f in families if families[f]]
    out = []
    for f in present:
        quota = k // len(present) if f != present[-1] else k - len(out)
        risky = [c for c in families[f] if c["risky"]]
        plain = [c for c in families[f] if not c["risky"]]
        n_risky = min(quota // 2, len(risky))
        picked = take(risky, n_risky) + take(plain, quota - n_risky)
        out += picked
    return out


def make(per_dataset: int, seed: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_TABLE:
        manifest = load_manifest(MANIFESTS / f"{DATASET_TABLE[dataset][0]}.json")
        pool = []
        dirs = sorted((RUNS / "sweep").glob(f"*/{dataset}"))
        dirs = [d for d in dirs if (d / "DONE").exists()]
        if not dirs:
            continue
        items = items_for(manifest)
        for d in dirs:
            for c in candidates(d, items, dataset):
                c["model"] = d.parent.name
                pool.append(c)
        sample = draw(pool, per_dataset, seed)
        order = np.random.default_rng(seed + 1).permutation(len(sample))
        sample = [sample[j] for j in order]  # the order carries no hint about risk or model
        with (OUT / f"{dataset}_key.csv").open("w", newline="") as fk, (OUT / f"{dataset}_annotator_blank.csv").open("w", newline="") as fa:
            key, blank = csv.writer(fk), csv.writer(fa)
            key.writerow(["unit", "model", "task", "item", "reply", "program_decision", "risky"])
            blank.writerow(["unit", "task", "question", "options_or_accepted_answers", "reply", "your_decision", "notes"])
            for u, c in enumerate(sample):
                it = c["item"]
                if c["task"] == "answer_correct":
                    guide, ask = " | ".join(it.answers), "does the reply commit to an answer matching one accepted answer? yes / no / unclear"
                elif c["task"] == "answer_label":
                    guide = " || ".join(f"{lab}) {opt}" for lab, opt in zip(it.option_labels, it.options))
                    ask = f"which option does the reply commit to? {'/'.join(it.option_labels)} / NONE / AMBIGUOUS"
                else:
                    guide, ask = f"scale labels: {', '.join(PROBE_SCALES[c['task'].split('_')[-1]].labels)}", "which label does the reply start with? the label / NONE / AMBIGUOUS"
                key.writerow([u, c["model"], c["task"], it.qid, c["text"], c["decision"], int(c["risky"])])
                blank.writerow([u, f"{c['task']}: {ask}", it.stem if c["task"].startswith("answer") else "(the reply to a confidence question; the question is not needed)", guide, c["text"], "", ""])
        print(f"{dataset}: {len(sample)} units from {len(dirs)} models -> {OUT / (dataset + '_key.csv')}")


def read_decisions(path: Path) -> dict:
    with path.open(newline="") as f:
        return {int(r["unit"]): r["your_decision"].strip() for r in csv.DictReader(f)}


def score(dataset: str, a_path: Path, b_path: Path) -> dict:
    a, b = read_decisions(a_path), read_decisions(b_path)
    units = sorted(set(a) & set(b))
    with (OUT / f"{dataset}_key.csv").open(newline="") as f:
        key = {int(r["unit"]): r for r in csv.DictReader(f)}
    la, lb = [a[u].lower() for u in units], [b[u].lower() for u in units]
    kappa = cohen_kappa(la, lb)
    both = [u for u in units if a[u].lower() == b[u].lower() and a[u]]  # the program is judged where the two annotators agree
    program = [key[u]["program_decision"].lower() for u in both]
    consensus = [a[u].lower() for u in both]
    by_task = {}
    for u in both:
        t = key[u]["task"].split("_label_")[0]
        by_task.setdefault(t, []).append(key[u]["program_decision"].lower() == a[u].lower())
    result = {"units": len(units), "units_both_agree": len(both), "kappa_annotators": round(kappa, 4),
              "program_agreement_with_consensus": round(float(np.mean([p == c for p, c in zip(program, consensus)])), 4) if both else None,
              "by_task": {t: round(float(np.mean(v)), 4) for t, v in by_task.items()}}
    print(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--per-dataset", type=int, default=200)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--dataset", required=True)
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    args = ap.parse_args()
    make(args.per_dataset, args.seed) if args.cmd == "make" else score(args.dataset, args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
