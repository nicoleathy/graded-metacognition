"""The Paper 1 sweep: every condition of the protocol for one model on the chosen datasets. Resumable; one JSONL per
condition under runs/sweep/<model>/<dataset>/.

    python experiments/paper1/run_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --datasets mmlu triviaqa

Provenance (methodology audit, 2026-10-27): the dataset comes from the Hub revision the manifest recorded, not from the current
one; a directory is resumed only under the settings that wrote its rows; DONE is written only when every condition has all its
rows (never after --only or --limit); and every condition file is bound, by its SHA-256, to the invocation that last completed it.
"""

import argparse
import hashlib
import json
import time

from common import MANIFESTS, ROOT, RUNS, model_slug
from protocol import DATASET_TABLE, SAMPLING, conditions_for, protocol_record

from probe_validity.generate import Runner, read_jsonl, run_condition
from probe_validity.loaders import load_items, load_manifest, select
from probe_validity.prompts import assert_no_format_instruction
from probe_validity.provenance import code_digest, git_state, sha256_of

# Settings that must not change between invocations writing into one run directory: resuming on top of rows made under another
# model revision, decoding configuration, template or software stack would mix runs silently.
COMPARED = ("model_id", "revision", "device", "dtype", "attention", "torch", "transformers", "chat_template_sha256", "padding_side",
            "generation_config", "generation_config_sampled", "manifest_sha256", "batch_size", "n", "limit")  # batch size fixes the batch composition
BOOKKEEPING = {"meta_history.jsonl", "conditions.jsonl"}  # not condition files: a directory holding only these has no rows yet


def last_records(out_dir) -> dict:
    """The last recorded SHA-256 entry of each condition in conditions.jsonl."""
    records, log = {}, out_dir / "conditions.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["condition"]] = rec
    return records


def _norm(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def config_changes(old: dict, new: dict) -> list[str]:
    """Names of the compared settings whose values differ between two invocation records."""
    return [k for k in COMPARED if _norm(old.get(k)) != _norm(new.get(k))]


def rows_present(path, name: str) -> int:
    """Distinct items that have a row in the condition file."""
    return len({r["i"] for r in read_jsonl(path) if r.get("condition") == name}) if path.exists() else 0


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_TABLE), choices=list(DATASET_TABLE))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="run only these conditions")
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--limit", type=int, help="use only the first N manifest items (smoke tests; never for results)")
    ap.add_argument("--allow-config-change", action="store_true", help="resume although the settings differ from the invocation that wrote the rows")
    args = ap.parse_args()

    protocol = protocol_record()
    runner = Runner(args.model)
    for dataset in args.datasets:
        manifest = load_manifest(MANIFESTS / f"{DATASET_TABLE[dataset][0]}.json")
        items, _ = load_items(dataset, manifest["source"]["revision"])  # the revision the manifest recorded, not the current one
        source = manifest["source"]
        items = select(items, manifest)
        if args.limit:
            items = items[: args.limit]
        for it in items:
            if it.options:
                assert_no_format_instruction(it.content())
        out_dir = RUNS / args.out / model_slug(args.model) / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = runner.meta(32)
        meta["generation_config_sampled"] = runner.effective_config(16, SAMPLING)  # used by the letters_sampled* conditions
        meta.update(experiment=args.out, dataset=dataset, n=len(items), limit=args.limit, manifest=manifest["name"], manifest_sha256=manifest["sha256"],
                    source=source, protocol_sha256=protocol["sha256"], batch_size=args.batch_size, git=git_state(ROOT), code_sha256=code_digest(ROOT),
                    started=time.strftime("%Y-%m-%d %H:%M:%S"), only=args.only)
        meta["invocation"] = sha256_of(meta)[:16]
        meta_path = out_dir / "meta.json"
        if meta_path.exists():  # keep the record of earlier invocations: conditions added later run under a newer protocol
            previous = json.loads(meta_path.read_text())
            changed = config_changes(previous, meta) if any(f for f in out_dir.glob("*.jsonl") if f.name not in BOOKKEEPING) else []
            if changed and not args.allow_config_change:
                raise SystemExit(f"{dataset}: {', '.join(changed)} differ from the invocation that wrote the rows in {out_dir}; "
                                 "resuming would mix settings. Pass --allow-config-change to continue anyway.")
            with (out_dir / "meta_history.jsonl").open("a") as fh:
                fh.write(json.dumps(previous) + "\n")
            if (out_dir / "DONE").exists() and (args.only or args.limit or changed):  # this run no longer vouches for a complete, uniform directory
                (out_dir / "DONE").rename(out_dir / f"DONE.superseded.{time.strftime('%Y%m%d-%H%M%S')}")
        meta_path.write_text(json.dumps(meta, indent=1))
        incomplete, recorded = [], last_records(out_dir)
        for cond in conditions_for(dataset):
            sub = items[: cond.subset] if cond.subset else items
            path = out_dir / f"{cond.name}.jsonl"
            if args.only and cond.name not in args.only:
                if rows_present(path, cond.name) < len(sub):
                    incomplete.append(cond.name)
                continue
            t0 = time.time()
            run_condition(runner, cond.name, [it.qid for it in sub], [cond.build(it) for it in sub], path,
                          cond.max_new_tokens, args.batch_size, cond.labels, cond.sampling, cond.seed)
            print(f"{dataset}/{cond.name}: {len(sub)} items in {time.time() - t0:.0f}s", flush=True)
            if rows_present(path, cond.name) < len(sub):
                incomplete.append(cond.name)
            digest = file_sha256(path)
            if recorded.get(cond.name, {}).get("rows_sha256") != digest:  # a no-op resume must not re-stamp rows an earlier invocation wrote
                with (out_dir / "conditions.jsonl").open("a") as fh:  # binds this file's bytes to the invocation that completed it
                    fh.write(json.dumps({"condition": cond.name, "n_items": len(sub), "rows_sha256": digest, "invocation": meta["invocation"],
                                         "code_sha256": meta["code_sha256"], "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        if not incomplete and not args.only and not args.limit:
            (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        else:
            print(f"{dataset}: not marked DONE (incomplete: {incomplete or 'none'}; only={args.only}; limit={args.limit})", flush=True)


if __name__ == "__main__":
    main()
