#!/bin/bash
# SLURM wrapper around run_replication.sh: one model per job, all three wordings on the same card. Submit from the repository root.
#   sbatch --partition=<gpu partition> --account=<account> --gres=gpu:1 --mem=64G --time=24:00:00 experiments/paper1/cluster/replication.sbatch Qwen/Qwen3-1.7B
# submit_replication.sh does this for every "ready" row of replication_jobs.tsv. Never split one model's replication over different GPU types or machines.
#SBATCH --job-name=p1-replication
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=p1-replication-%x-%j.out
set -euo pipefail
MODEL="${1:-${MODEL:-}}"
: "${MODEL:?give the model id as the first argument}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
source "${P1_VENV:-$HOME/venvs/paper1}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1            # weights and datasets were fetched beforehand by prefetch.py
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1                   # no token is ever used by a job
export PYTHONUNBUFFERED=1
exec bash experiments/paper1/cluster/run_replication.sh "$MODEL" "${2:-mmlu mmlu_pro_eng supergpqa_mfg}" "${3:-w1 w2 w3}"
