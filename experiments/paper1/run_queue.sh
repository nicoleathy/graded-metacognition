#!/bin/bash
# Run the E10 pilot for several models one after the other (resumable: finished conditions are skipped).
#   experiments/paper1/run_queue.sh "Qwen/Qwen2.5-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/venvs/paper1/bin/python"
for model in "$@"; do
  echo "=== $model: $(date '+%F %T')"
  "$PY" experiments/paper1/run_e10.py --model "$model" --n 500 || echo "!!! $model failed with status $?"
done
echo "=== queue finished: $(date '+%F %T')"
