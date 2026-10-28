# Environment used for the Paper 1 runs

| | |
|---|---|
| Machine | Mac Studio, Apple M4 Max, 128 GB unified memory, macOS 15.6.1 |
| Python | 3.12.13 (Homebrew), virtual environment |
| Key packages | torch 2.8.0, transformers 4.56.0, accelerate 1.10.1, datasets 4.0.0, numpy 2.3.2, scipy 1.16.1, pandas 2.3.2, safetensors 0.8.0, tokenizers 0.22.2 |
| Exact versions | `paper1-macos-arm64-freeze.txt` (`pip freeze` of the environment that produced the finished runs) |
| Install spec | `../requirements-paper1.txt` (the pins the code and tests were verified with) |
| Precision and decoding | bfloat16 weights; greedy decoding, at most 32 new tokens for answers; left padding; batch size 16; the model's own chat template |
| Attention | `eager` on Apple MPS (SDPA returns NaN for left-padded rows there); `sdpa` on CUDA (the `Runner` chooses by device) |
| Models | loaded from the local Hugging Face cache only (`local_files_only=True`), at the revision recorded in each run's `meta.json` |

Every run directory records the device, dtype, attention implementation, torch and transformers versions, the model revision, the generation configuration, the protocol hash and the
code state in `meta.json`, and a SHA-256 per condition file in `conditions.jsonl`. A run directory is resumed only if those settings are unchanged
(`experiments/paper1/run_sweep.py`), so **one model is always run on one platform**; a different platform for another model is allowed, and is recorded.

## Rebuilding on a Linux/CUDA cluster

```bash
PYTHON=python3.12 VENV=$HOME/venvs/paper1 experiments/paper1/cluster/setup_env.sh
# only if the GPU driver is too old for the default CUDA build of torch 2.8.0:
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 experiments/paper1/cluster/setup_env.sh
```

Expect small numerical differences between MPS and CUDA (bfloat16 kernels and the attention implementation differ), which is why the platform is recorded for each model and
why no run directory is ever continued on another machine. Do not upgrade torch or transformers: the tests and the protocol were verified with these versions. The plotting
environment used for manuscript figures is separate and is not needed to run the experiments.
