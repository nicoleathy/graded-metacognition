#!/bin/bash
# Run every dataset of ONE model on this machine, verify the result and pack it. Works under SLURM (see sweep.sbatch) or by hand.
#   experiments/paper1/cluster/run_job.sh Qwen/Qwen3-1.7B "mmlu triviaqa mmlu_pro_eng supergpqa_mfg webquestions"
# Environment: PYTHON (default python, with the pinned environment active), HF_HOME (model and dataset cache), and HF_HUB_OFFLINE=1 after prefetch.py.
set -euo pipefail
MODEL="${1:?model id, e.g. Qwen/Qwen3-1.7B}"
DATASETS="${2:?datasets, e.g. \"mmlu triviaqa\"}"
cd "$(dirname "$0")/../../.."
PY="${PYTHON:-python}"
slug="${MODEL//\//__}"
mkdir -p experiments/paper1/runs/logs
echo "=== $MODEL on $DATASETS | $(date '+%F %T') | host $(hostname) | commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "!!! the working tree has local changes: the run metadata would record a dirty tree; use a clean checkout"; exit 1; fi
nvidia-smi -L 2>/dev/null || true
"$PY" - <<'PY'
import platform, torch, transformers
print("python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"))
assert torch.cuda.is_available(), "no GPU visible: a CPU run would be a different platform and far too slow"
PY
"$PY" -m pytest tests/probe_validity -q -x                     # the sweep must not start if anything fails here
"$PY" experiments/paper1/run_sweep.py --model "$MODEL" --datasets mmlu --limit 16 --out sweep_smoke_cluster   # 16 MMLU items, every condition, scratch directory
# The sweep itself: resumable, refuses to continue a directory written under other settings or on another platform. Never pass --allow-config-change.
"$PY" experiments/paper1/run_sweep.py --model "$MODEL" --datasets $DATASETS --batch-size 16 >> "experiments/paper1/runs/logs/sweep_${slug}.log" 2>&1
"$PY" experiments/paper1/verify_runs.py
bash experiments/paper1/cluster/package_runs.sh "$MODEL"
echo "=== finished $MODEL | $(date '+%F %T')"
