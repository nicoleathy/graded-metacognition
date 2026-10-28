"""E10: is the multiple-choice probe failure an instruction-capture effect? Generates every condition of the 2 x 2
(question string with the answer-format instruction embedded, as in the legacy loader, vs content only) x four probe
formats (Yes/No, letters A-D, digits 1-4, words) on a seeded stratified MMLU sample. Resumable.

    python experiments/paper1/run_e10.py --model Qwen/Qwen2.5-1.5B-Instruct --n 500
"""

import argparse
import json
import time

from common import MANIFESTS, ROOT, RUNS, OPTION_LABELS, PROBES, e10_conditions, model_slug

from probe_validity.generate import Runner, run_condition
from probe_validity.loaders import load_manifest, load_mmlu, select
from probe_validity.prompts import assert_no_format_instruction
from probe_validity.provenance import code_digest, git_state
from probe_validity.scales import label_overlap

MAX_NEW_TOKENS = 32  # as in the legacy evaluator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="run only these conditions")
    args = ap.parse_args()

    manifest = load_manifest(MANIFESTS / f"mmlu_test_s0_n{args.n}.json")
    items = select(load_mmlu("test"), manifest)
    for it in items:
        assert_no_format_instruction(it.content())  # C10: content-only strings carry no format instruction
    for probe, (scale, _) in PROBES.items():
        assert bool(label_overlap(scale, OPTION_LABELS)) == (probe == "letters"), probe  # only the letter scale collides

    out_dir = RUNS / "e10" / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model)
    meta = runner.meta(MAX_NEW_TOKENS)
    meta.update(experiment="e10", n=len(items), manifest=manifest["name"], manifest_sha256=manifest["sha256"],
                batch_size=args.batch_size, git=git_state(ROOT), code_sha256=code_digest(ROOT), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    qids = [it.qid for it in items]
    for name, (build, labels) in e10_conditions().items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        run_condition(runner, name, qids, [build(it) for it in items], out_dir / f"{name}.jsonl",
                      MAX_NEW_TOKENS, args.batch_size, labels)
        print(f"{name}: finished in {time.time() - t0:.0f}s", flush=True)
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
