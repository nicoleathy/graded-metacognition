"""Download the public panel models that are not in the local Hugging Face cache, WITHOUT a token. The runner reads the cache only, so a public
model must be here before the panel queue can run it. Gated models (Llama, Gemma) are never touched: they need the licence accepted and the
owner's permission to use a token.

    python experiments/paper1/download_panel.py            # downloads, one model at a time
    python experiments/paper1/download_panel.py --dry-run  # lists what would be downloaded
"""

import argparse
import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # before huggingface_hub is imported: never send a stored or environment token

import panel  # noqa: E402

SKIP = ["*.bin", "*.pt", "*.pth", "*.onnx", "*.gguf", "*.msgpack", "*.h5", "original/*", "*.ot"]  # the runner reads safetensors only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = panel.to_download()
    print("to download:", ", ".join(m.id for m in todo) or "nothing")
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download

    for m in todo:
        print(f"=== {m.id}", flush=True)
        snapshot_download(m.id, ignore_patterns=SKIP, token=False)  # token=False: anonymous
        print(f"cached: {m.cached}", flush=True)


if __name__ == "__main__":
    main()
