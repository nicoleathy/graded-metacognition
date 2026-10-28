"""Summarise every finished run directory (E10 pilot and sweep) into one JSON and print an overview.

    python experiments/paper1/analyze_sweep.py                  # runs/sweep and runs/e10
    python experiments/paper1/analyze_sweep.py --exp e10 --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import time

from analysis import summarise_directory
from common import MANIFESTS, ROOT, RUNS, model_slug
from protocol import DATASET_TABLE
from verify_runs import expected_rows, verify_directory

from probe_validity.loaders import load_items, load_manifest, select
from probe_validity.provenance import code_digest

RESULTS = ROOT / "experiments/paper1/results"
_items = {}


def items_for(manifest: dict):
    """Items of the manifest's dataset at the manifest's recorded Hub revision."""
    dataset = manifest["dataset"]
    if dataset not in _items:
        items, _ = load_items(dataset, manifest["source"]["revision"])  # the revision the manifest recorded, not the current one
        _items[dataset] = items
    return select(_items[dataset], manifest)


def run_dirs(exp: str, model: str | None, datasets: list[str] | None = None, complete_only: bool = True):
    """Run directories to summarise. A directory without its DONE marker is still being written and is skipped."""
    base = RUNS / exp
    if not base.exists():
        return
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if model and model_dir.name != model_slug(model):
            continue
        if (model_dir / "meta.json").exists():  # E10 layout: one dataset per model directory
            candidates = [model_dir]
        else:
            candidates = sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
        for d in candidates:
            if datasets and d.name not in datasets and d is not model_dir:
                continue
            if complete_only and not (d / "DONE").exists():
                continue
            yield model_dir.name, d


def overview(model: str, dataset: str, s: dict) -> str:
    d = s["direct"]["clean"]
    lines = [f"{model} | {dataset} | n={s['n']} | direct acc {d['accuracy']} (valid {d['valid_rate']})"]
    for name, m in s["probes"].items():
        on, ft = m.get("on_valid"), (m.get("first_token") or {}).get("auroc")
        verdict = s["screens"].get(name, {}).get("overall", "")
        lines.append(f"    {name:20} valid {m['valid_rate']:.3f}  AUROC {(on or {}).get('auroc', '-')}  first-token {(ft or {}).get('auroc', '-')}  screen {verdict}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["sweep", "e10"])
    ap.add_argument("--model")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--datasets", nargs="+", help="only these datasets (sweep layout)")
    ap.add_argument("--include-incomplete", action="store_true", help="also summarise directories still being written")
    ap.add_argument("--skip-verify", action="store_true", help="summarise even if a run directory fails the completeness and hash check")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for exp in args.exp:
        summary = {"experiment": exp, "code_sha256": code_digest(ROOT), "n_boot": args.n_boot,
                   "computed": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
        for model, directory in run_dirs(exp, args.model, args.datasets, not args.include_incomplete):
            meta = json.loads((directory / "meta.json").read_text())
            manifest = load_manifest(MANIFESTS / f"{meta['manifest']}.json")
            dataset = manifest["dataset"]
            if exp == "sweep" and not args.skip_verify and (directory / "DONE").exists():
                problems = verify_directory(directory, expected_rows(dataset, meta["n"]))
                if problems:
                    print(f"!!! {model} | {dataset}: not summarised, {len(problems)} problem(s): {problems[:3]} (use --skip-verify to override)", flush=True)
                    continue
            s = summarise_directory(directory, items_for(manifest), DATASET_TABLE[dataset][1], args.n_boot)
            s["run"] = {k: meta.get(k) for k in ("model_id", "revision", "device", "dtype", "attention", "manifest", "manifest_sha256",
                                               "protocol_sha256", "code_sha256", "started")}
            s["complete"] = (directory / "DONE").exists()
            summary["models"].setdefault(model, {})[dataset] = s
            if not args.quiet:
                print(overview(model, dataset, s), flush=True)
        if summary["models"]:
            name = f"{exp}_summary.json" if not args.model else f"{exp}_summary_{model_slug(args.model)}.json"
            (RESULTS / name).write_text(json.dumps(summary, indent=1))
            print(f"wrote {RESULTS / name}")


if __name__ == "__main__":
    main()
