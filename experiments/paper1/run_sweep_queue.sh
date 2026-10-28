#!/bin/bash
# Run the Paper 1 sweep dataset by dataset (every model on MMLU first), keeping the Mac awake. Resumable: rerun the same
# command after an interruption and finished conditions are skipped.
#   experiments/paper1/run_sweep_queue.sh "mmlu mmlu_pro_eng" "Qwen/Qwen2.5-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/venvs/paper1/bin/python"
datasets="$1"; shift
mkdir -p experiments/paper1/runs/logs
for ds in $datasets; do
  for model in "$@"; do
    slug="${model//\//__}"
    echo "=== $ds / $model: $(date '+%F %T')"
    caffeinate -i "$PY" experiments/paper1/run_sweep.py --model "$model" --datasets "$ds" >> "experiments/paper1/runs/logs/sweep_${slug}.log" 2>&1 \
      || echo "!!! $ds / $model failed with status $? (see experiments/paper1/runs/logs/sweep_${slug}.log)"
  done
done
echo "=== queue finished: $(date '+%F %T')"
