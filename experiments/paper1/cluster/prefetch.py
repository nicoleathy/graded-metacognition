"""Download what the jobs need, on a machine with internet access (a login node), before any job runs offline.

    python experiments/paper1/cluster/prefetch.py                        # every "ready" row of jobs.tsv
    python experiments/paper1/cluster/prefetch.py --model Qwen/Qwen3-1.7B

Public weights are fetched anonymously (no token is read or sent). A gated model is never touched here: its owner accepts the licence and downloads it
with their own account. The datasets are fetched at the Hub commit that each manifest recorded, so a later upstream change cannot alter a run.
Afterwards set HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 for the jobs.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # before huggingface_hub is imported: never send a stored or environment token

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import MANIFESTS  # noqa: E402
from protocol import DATASET_TABLE  # noqa: E402

from probe_validity.loaders import load_items, load_manifest  # noqa: E402

SKIP = ["*.bin", "*.pt", "*.pth", "*.onnx", "*.gguf", "*.msgpack", "*.h5", "original/*", "*.ot"]  # the runner reads safetensors only


def jobs(only: str | None) -> list[tuple[str, list[str]]]:
    rows = []
    for line in (HERE / "jobs.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            model, datasets, _gb, status, *_ = line.split("\t")
            if status == "ready" and (only is None or model == only):
                rows.append((model, datasets.split()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    todo = jobs(args.model)
    if not todo:
        sys.exit("no ready job matches")
    from huggingface_hub import snapshot_download

    for model, _ in todo:
        print(f"=== weights: {model}", flush=True)
        snapshot_download(model, ignore_patterns=SKIP, token=False)
    for dataset in sorted({d for _, ds in todo for d in ds}):
        manifest = load_manifest(MANIFESTS / f"{DATASET_TABLE[dataset][0]}.json")
        items, source = load_items(dataset, manifest["source"]["revision"])
        print(f"=== dataset: {dataset} ({len(items)} items at {source['revision'][:8]})", flush=True)
    print("done: set HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 for the jobs")


if __name__ == "__main__":
    main()
