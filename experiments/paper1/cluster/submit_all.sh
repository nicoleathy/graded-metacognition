#!/bin/bash
# Print (default) or run (with --go) the sbatch command of every "ready" row of jobs.tsv. Extra sbatch arguments for all jobs go in P1_SBATCH_ARGS
# (e.g. "--partition=gpu --account=myproject"); P1_SBATCH_ARGS_BIG is added for jobs that need 40 GB or more of GPU memory (e.g. "--constraint=a100-80gb").
set -euo pipefail
cd "$(dirname "$0")/../../.."
go=0; [ "${1:-}" = "--go" ] && go=1
while IFS=$'\t' read -r model datasets gb status _note; do
  case "$model" in ''|'#'*) continue;; esac
  [ "$status" = "ready" ] || { echo "skip $model ($status)"; continue; }
  extra="${P1_SBATCH_ARGS:-}"; [ "$gb" -ge 40 ] && extra="$extra ${P1_SBATCH_ARGS_BIG:-}"
  cmd="sbatch --job-name=p1-${model##*/} $extra experiments/paper1/cluster/sweep.sbatch $model \"$datasets\""
  if [ "$go" = 1 ]; then eval "$cmd"; else echo "$cmd"; fi
done < experiments/paper1/cluster/jobs.tsv
[ "$go" = 1 ] || echo "(dry run: nothing submitted; add --go)"
