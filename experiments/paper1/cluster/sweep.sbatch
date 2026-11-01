#!/bin/bash
# SLURM wrapper around run_job.sh: one model per job. Submit from the repository root. Resource lines can be overridden on the command line:
#   sbatch --partition=<gpu partition> --account=<account> --gres=gpu:1 --mem=64G --time=12:00:00 \
#          experiments/paper1/cluster/sweep.sbatch Qwen/Qwen3-1.7B "mmlu triviaqa mmlu_pro_eng supergpqa_mfg webquestions"
# submit_all.sh does this for every "ready" row of jobs.tsv. Never split one model's datasets over different GPU types or machines.
#SBATCH --job-name=p1-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=p1-sweep-%x-%j.out
set -euo pipefail
MODEL="${1:-${MODEL:-}}"; DATASETS="${2:-${DATASETS:-}}"; OUTDIR="${3:-${OUTDIR:-sweep}}"      # arguments (preferred) or environment variables
: "${MODEL:?give the model id as the first argument}"
: "${DATASETS:?give the datasets as the second argument, quoted}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
source "${P1_VENV:-$HOME/venvs/paper1}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1            # weights and datasets were fetched beforehand by prefetch.py
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1                   # no token is ever used by a job
export PYTHONUNBUFFERED=1
exec bash experiments/paper1/cluster/run_job.sh "$MODEL" "$DATASETS" "$OUTDIR"
