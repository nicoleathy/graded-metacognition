"""How much does the platform change one model's results? Compare two runs of the same model and dataset (for example Apple MPS with eager attention against a
CUDA GPU with sdpa attention) item by item, on identical items, prompts, model revision and decoding.

    python experiments/paper1/cluster/compare_platforms.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset mmlu \\
        --a experiments/paper1/runs/sweep --b experiments/paper1/runs/platform_check_cuda [--json report.json] [--n-boot 1000] [--no-screens]

This is a calibration, not a result: the second run is made into its own output directory (run_sweep.py --out platform_check_cuda) so that it can never mix with the
runs of the study, where one model is always run on one platform. Both directories are verified first (all rows present, every file's SHA-256 as recorded). The report
has: the settings that must be equal and whether they are, the platform fields that differ, how many replies are identical per condition (greedy conditions; the
sampled decodes of the letters probe draw from each device's own random generator and are not expected to agree item by item), the numerical drift of the first-token
label log-probabilities, paired differences with bootstrap intervals for direct accuracy, compliance and AUROC, and the screen verdicts of both runs side by side.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import MANIFESTS, model_slug  # noqa: E402
from protocol import DATASET_TABLE, conditions_for  # noqa: E402

from probe_validity import metrics as M  # noqa: E402
from probe_validity.loaders import load_manifest  # noqa: E402
from probe_validity.records import direct_outcomes, grades, read_condition  # noqa: E402
from probe_validity.scales import PROBE_SCALES  # noqa: E402

MUST_BE_EQUAL = ("model_id", "revision", "dtype", "chat_template_sha256", "padding_side", "add_special_tokens", "truncation", "generation_config",
                 "generation_config_sampled", "manifest_sha256", "protocol_sha256", "batch_size", "n")
PLATFORM_FIELDS = ("device", "attention", "torch", "transformers", "python")
PROBES = ("yesno", "letters", "digits", "words")


def integrity(directory: Path) -> list[str]:
    """What is wrong with a finished run directory: a condition missing, short of rows, or changed after its hash was recorded (empty when the run is intact)."""
    from verify_runs import expected_rows, verify_directory

    meta = json.loads((directory / "meta.json").read_text())
    return verify_directory(directory, expected_rows(meta["dataset"], meta["n"]))


def compare_meta(a: dict, b: dict) -> dict:
    """The settings that must be equal for the comparison to isolate the platform, and the platform fields as they are."""
    return {"settings_equal": {k: a.get(k) == b.get(k) for k in MUST_BE_EQUAL}, "platform": {k: [a.get(k), b.get(k)] for k in PLATFORM_FIELDS}}


def compare_rows(rows_a: list, rows_b: list) -> dict:
    """Identical replies and first-token log-probability drift over the items that have a row in both runs."""
    pairs = [(x, y) for x, y in zip(rows_a, rows_b) if x is not None and y is not None]
    drift = []
    for x, y in pairs:
        lp_a, lp_b = x.get("label_lp") or {}, y.get("label_lp") or {}
        shared = [k for k in lp_a if k in lp_b]
        if shared:
            drift.append(max(abs(lp_a[k] - lp_b[k]) for k in shared))
    out = {"items": len(pairs), "identical_replies": float(np.mean([x["text"] == y["text"] for x, y in pairs])) if pairs else None}
    if drift:
        out["label_logprob_drift"] = {"median": float(np.median(drift)), "p95": float(np.percentile(drift, 95)), "max": float(np.max(drift))}
    return out


def paired_difference(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """Mean of x minus mean of y over the same items, with a percentile bootstrap interval over items."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return {"diff": None, "ci": None, "n": 0}
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    d = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return {"diff": float(x.mean() - y.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n": n}


def compare_probe(correct_a, grade_a, correct_b, grade_b, n_boot: int, seed: int = 0) -> dict:
    """Compliance, grade agreement and AUROC of one probe on two platforms. The AUROC difference is taken on the items that are compliant in both runs, with each
    run's own correctness labels, and resampled jointly."""
    ok_a, ok_b = ~np.isnan(grade_a), ~np.isnan(grade_b)
    both = ok_a & ok_b
    out = {"compliance": paired_difference(ok_a.astype(float), ok_b.astype(float), n_boot, seed),
           "same_grade_where_both_compliant": float(np.mean(grade_a[both] == grade_b[both])) if both.any() else None}
    ca, cb, ga, gb = (np.asarray(v)[both] for v in (correct_a, correct_b, grade_a, grade_b))
    if both.sum() >= 30 and len(set(ca)) == 2 and len(set(cb)) == 2 and len(set(ga)) > 1 and len(set(gb)) > 1:
        idx = np.random.default_rng(seed).integers(0, len(ca), (n_boot, len(ca)))
        draws = [M.auroc(ca[i], ga[i]) - M.auroc(cb[i], gb[i]) for i in idx if len(set(ca[i])) == 2 and len(set(cb[i])) == 2]
        out["auroc"] = {"a": float(M.auroc(ca, ga)), "b": float(M.auroc(cb, gb)), "diff": float(M.auroc(ca, ga) - M.auroc(cb, gb)),
                        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "n": int(both.sum())}
    return out


def report(model: str, dataset: str, a_root: Path, b_root: Path, n_boot: int, screens: bool) -> dict:
    slug = model_slug(model)
    da, db = a_root / slug / dataset, b_root / slug / dataset
    for d in (da, db):
        if not (d / "DONE").exists():
            raise SystemExit(f"{d} is not a finished run directory")
    broken = {str(d): problems for d in (da, db) if (problems := integrity(d))}
    if broken:
        raise SystemExit(f"a run directory fails verification, so the comparison would mean nothing: {broken}")
    meta_a, meta_b = json.loads((da / "meta.json").read_text()), json.loads((db / "meta.json").read_text())
    manifest = load_manifest(MANIFESTS / f"{meta_a['manifest']}.json")
    from analyze_sweep import items_for

    items = items_for(manifest)
    n = len(items)
    out = {"model": model, "dataset": dataset, "integrity": "both directories complete, every file's SHA-256 as recorded", "meta": compare_meta(meta_a, meta_b),
           "conditions": {}, "probes": {}}
    for cond in conditions_for(dataset):
        out["conditions"][cond.name] = compare_rows(read_condition(da, cond.name, n), read_condition(db, cond.name, n))
    dir_a, dir_b = direct_outcomes(read_condition(da, "direct_clean", n), items), direct_outcomes(read_condition(db, "direct_clean", n), items)
    out["direct"] = {"accuracy": paired_difference(dir_a.correct, dir_b.correct, n_boot), "answered": paired_difference(dir_a.valid, dir_b.valid, n_boot),
                     "same_correctness": float(np.mean(dir_a.correct == dir_b.correct))}
    for probe in PROBES:
        for variant in ("embedded", "raw", "stem"):
            name = f"{probe}_{variant}"
            if (da / f"{name}.jsonl").exists() and (db / f"{name}.jsonl").exists():
                ga = grades(read_condition(da, name, n), PROBE_SCALES[probe])
                gb = grades(read_condition(db, name, n), PROBE_SCALES[probe])
                out["probes"][name] = compare_probe(dir_a.correct, ga, dir_b.correct, gb, n_boot)
    if screens:
        from analysis import summarise_directory

        sa, sb = (summarise_directory(d, items, DATASET_TABLE[dataset][1], max(n_boot // 5, 100)) for d in (da, db))
        out["screens"] = {cell: [sa["screens"][cell]["overall"], sb["screens"][cell]["overall"]] for cell in sa["screens"] if cell in sb["screens"]}
        out["screens_agree"] = [sum(v[0] == v[1] for v in out["screens"].values()), len(out["screens"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="mmlu", choices=list(DATASET_TABLE))
    ap.add_argument("--a", type=Path, required=True, help="runs directory of the first platform, e.g. experiments/paper1/runs/sweep")
    ap.add_argument("--b", type=Path, required=True, help="runs directory of the second platform, e.g. experiments/paper1/runs/platform_check_cuda")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-screens", action="store_true", help="skip the screen verdicts (they need the full per-run analysis)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rep = report(args.model, args.dataset, args.a, args.b, args.n_boot, not args.no_screens)
    bad = [k for k, ok in rep["meta"]["settings_equal"].items() if not ok]
    print(f"{rep['model']} on {rep['dataset']}: {rep['integrity']}")
    print("settings that must be equal: " + ("all equal" if not bad else f"DIFFER: {bad}"))
    print("platform fields (a | b):", {k: v for k, v in rep["meta"]["platform"].items()})
    for kind, keep in (("greedy", lambda k: "sampled" not in k), ("sampled (each device draws from its own random generator, so item-by-item agreement is not expected)", lambda k: "sampled" in k)):
        ident = [c["identical_replies"] for k, c in rep["conditions"].items() if keep(k) and c["identical_replies"] is not None]
        if ident:
            print(f"identical replies per {kind} condition: min {min(ident):.4f}, mean {np.mean(ident):.4f} over {len(ident)} conditions")
    print("direct accuracy a - b:", rep["direct"]["accuracy"], "| same correctness per item:", round(rep["direct"]["same_correctness"], 4))
    for name, p in rep["probes"].items():
        au = p.get("auroc")
        print(f"  {name:16} compliance a-b {p['compliance']['diff']:+.4f} {np.round(p['compliance']['ci'], 4).tolist()}"
              + (f" | AUROC a {au['a']:.4f} b {au['b']:.4f} diff {au['diff']:+.4f} {np.round(au['ci'], 4).tolist()} (n={au['n']})" if au else ""))
    if "screens_agree" in rep:
        print(f"screen verdicts equal in {rep['screens_agree'][0]} of {rep['screens_agree'][1]} cells")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
