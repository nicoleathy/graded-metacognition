#!/bin/bash
# The replication of ONE model: the three wordings, one after the other, on this machine and card (one model, one card type). Works under SLURM (replication.sbatch) or by hand.
#   experiments/paper1/cluster/run_replication.sh Qwen/Qwen3-1.7B                       # all three wordings, the replication datasets
#   experiments/paper1/cluster/run_replication.sh Qwen/Qwen3-1.7B "mmlu" "w2"           # one dataset, one wording (a re-run after a failure; finished directories are skipped)
# Each wording is a run_job.sh call: runs/replication_<wording>/<model>/<dataset>, verified, packed as p1_replication_<wording>_<model>_<time>.tar.gz. The protocol is
# frozen/replication_plan.json; a wording failing does not stop the next one, and the exit code is non-zero if any failed.
set -uo pipefail
MODEL="${1:?model id, e.g. Qwen/Qwen3-1.7B}"
DATASETS="${2:-mmlu mmlu_pro_eng supergpqa_mfg}"
WORDINGS="${3:-w1 w2 w3}"
cd "$(dirname "$0")/../../.."
bad=0
for w in $WORDINGS; do
  P1_SUITE=replication PROBE_WORDING="$w" bash experiments/paper1/cluster/run_job.sh "$MODEL" "$DATASETS" "replication_$w" || { echo "!!! $MODEL wording $w failed"; bad=1; }
done
exit $bad
