#!/bin/bash
# Run the rest of the Paper 1 panel after the first queue: one model at a time, every dataset of a model before the next model,
# keeping the Mac awake. Models and datasets come from panel.py: public weights first, a gated model only when its weights are
# already cached (nothing here reads a Hugging Face token), the validity-only model last. Each model is smoke-tested on 16 MMLU
# items first; a model whose template or tokenizer breaks is logged and skipped. Resumable: rerun it and finished conditions are skipped.
#   experiments/paper1/run_panel_queue.sh          # start now
#   experiments/paper1/run_panel_queue.sh wait     # wait for run_sweep_queue.sh to finish, then start
set -u
cd "$(dirname "$0")/../.."
PY="$HOME/venvs/paper1/bin/python"
mkdir -p experiments/paper1/runs/logs
if [ "${1:-}" = "wait" ]; then
  while pgrep -f "[r]un_sweep_queue.sh" > /dev/null; do sleep 60; done
  echo "first queue finished: $(date '+%F %T')"
fi
# public models that are not cached yet are fetched without a token (a no-op when download_panel.py already ran); gated ones are never touched
"$PY" experiments/paper1/download_panel.py >> experiments/paper1/runs/logs/download_panel.log 2>&1 || echo "!!! download_panel.py failed (see experiments/paper1/runs/logs/download_panel.log)"
"$PY" experiments/paper1/panel.py --queue | while read -r model datasets; do
  slug="${model//\//__}"
  free_gb=$(df -k "$HOME" | awk 'NR==2 {print int($4/1048576)}')
  if [ "$free_gb" -lt 30 ]; then echo "!!! $model skipped: only ${free_gb} GB free (need 30)"; continue; fi
  echo "=== smoke $model: $(date '+%F %T')"
  if ! "$PY" experiments/paper1/run_sweep.py --model "$model" --datasets mmlu --limit 16 --out sweep_smoke_panel >> "experiments/paper1/runs/logs/smoke_${slug}.log" 2>&1; then
    echo "!!! $model failed its smoke test (see experiments/paper1/runs/logs/smoke_${slug}.log); skipped"; continue
  fi
  "$PY" experiments/paper1/archive_source.py
  echo "=== $model on $datasets: $(date '+%F %T')"
  caffeinate -i "$PY" experiments/paper1/run_sweep.py --model "$model" --datasets $datasets >> "experiments/paper1/runs/logs/sweep_${slug}.log" 2>&1 \
    || echo "!!! $model failed with status $? (see experiments/paper1/runs/logs/sweep_${slug}.log)"
done
echo "=== panel queue finished: $(date '+%F %T')"
