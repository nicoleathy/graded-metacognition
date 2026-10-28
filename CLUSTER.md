# Running the outstanding Paper 1 jobs on a cluster

Status when this file was written (28 October 2026, afternoon). Six models are finished on the Mac Studio (Qwen2.5-1.5B, 3B and 7B, Qwen3-4B, Llama-3.1-8B, Mistral-7B-v0.3; five datasets
each). **Qwen3-8B is still running on the Mac and finishes there: do not run it on the cluster.** The Mac queue has been stopped, so nothing else will start there.
The cluster takes everything that has not been started (`experiments/paper1/cluster/jobs.tsv`):

| Model | Datasets | GPU memory | Status |
|---|---|---|---|
| Qwen/Qwen3-1.7B | mmlu triviaqa mmlu_pro_eng supergpqa_mfg webquestions | 16 GB | **ready** |
| Qwen/Qwen2.5-14B-Instruct | mmlu supergpqa_mfg (validity-only run) | 40 GB or more | **ready** |
| meta-llama/Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, google/gemma-2-2b-it, gemma-2-9b-it | all five | 16 to 24 GB | blocked: gated models; the owner accepts each licence and downloads the weights with their own Hugging Face token (nothing in this repository reads a token) |

## Rules that keep the study valid

1. **One model, one platform.** All conditions and datasets of a model run on one machine and one GPU type. A run directory is resumed only if its recorded settings (device, dtype,
   attention, torch, transformers, batch size, model revision, generation configuration) are unchanged; the run script refuses otherwise. Never pass `--allow-config-change`, and never copy a
   partly finished directory between machines.
2. **Do not change the code or the frozen files** (`experiments/paper1/frozen/screen_config.json`, `protocol.json`; the tests fail if the code drifts from them). Run from a clean checkout of this
   branch; `run_job.sh` refuses a dirty working tree and records the commit hash in every run's `meta.json`.
3. Keep the defaults: greedy decoding, batch size 16, bfloat16. Do not add sampling, change prompts, or reorder items.
4. No tokens in the repository, in job scripts or in logs. Jobs run offline (`HF_HUB_OFFLINE=1`); the cluster owner fetches weights beforehand.
5. Do not analyse or summarise on the cluster for the paper, and do not commit run outputs (`experiments/paper1/runs/` is git-ignored). The results come back as a checksummed package (below).

## Steps

```bash
git clone -b paper1-validity https://github.com/nicoleathy/graded-metacognition.git && cd graded-metacognition   # then: git log -1 (note the commit)

# 1. environment, once, on a node with internet access (Python 3.12; see environment/README.md)
PYTHON=python3.12 VENV=$HOME/venvs/paper1 experiments/paper1/cluster/setup_env.sh
source $HOME/venvs/paper1/bin/activate
export HF_HOME=<scratch>/hf_cache                        # weights are large: Qwen2.5-14B is about 30 GB

# 2. weights (public ones, anonymously) and the datasets at the manifest revisions, still with internet access
python experiments/paper1/cluster/prefetch.py

# 3. submit: prints the sbatch commands; add --go to submit. Extra arguments for your site go in P1_SBATCH_ARGS
P1_SBATCH_ARGS="--partition=<gpu> --account=<account>" P1_SBATCH_ARGS_BIG="--constraint=<80gb gpu>" \
  bash experiments/paper1/cluster/submit_all.sh            # review the commands
  bash experiments/paper1/cluster/submit_all.sh --go

# without SLURM: run_job.sh does everything for one model on the current GPU machine
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash experiments/paper1/cluster/run_job.sh Qwen/Qwen3-1.7B "mmlu triviaqa mmlu_pro_eng supergpqa_mfg webquestions"
```

Each job: records the platform; runs `pytest tests/probe_validity`; smoke-tests the model on 16 MMLU items (every condition, scratch directory); runs the sweep (resumable: if a job is killed, submit it
again and finished conditions are skipped); verifies every run directory (all rows present, every file's SHA-256 as recorded); and writes
`experiments/paper1/cluster/packages/p1_runs_<model>_<time>.tar.gz` with a `.sha256` file. A finished dataset has a `DONE` marker in its directory. Progress: `python experiments/paper1/status.py`.

## Bringing the results back to the Mac

```bash
# on the cluster: copy experiments/paper1/cluster/packages/p1_runs_*.tar.gz and the .sha256 files to the Mac, then on the Mac, in the repository:
shasum -a 256 -c p1_runs_<model>_<time>.tar.gz.sha256
tar -xzf p1_runs_<model>_<time>.tar.gz -C experiments/paper1/runs/sweep/
python experiments/paper1/verify_runs.py                  # must print only "ok" lines
```

The analysis, the tables and the manuscript are then regenerated on the Mac.

## Resources

Not measured on the cluster. For reference, the Mac Studio (M4 Max) needed these hours of generation for all five datasets: Qwen2.5-1.5B 2.1, Qwen2.5-3B 2.9, Qwen3-4B 6.3, Qwen2.5-7B 6.5,
Llama-3.1-8B 8.8, Mistral-7B 10.0. A single modern GPU should be several times faster; the default `--time=12:00:00` in `sweep.sbatch` is generous, so override it for your queue
(`sbatch --time=...`). Memory: bfloat16 weights plus a batch of 16 short prompts, so a 16 GB GPU is enough up to about 7B, 24 GB for the 9B model, 40 GB or more for the 14B model.

## If something fails

- `no GPU visible`: the job landed on a node without a visible GPU; request one (`--gres=gpu:1`).
- Hub or dataset errors under `HF_HUB_OFFLINE=1`: the prefetch did not complete for that model or dataset; rerun `prefetch.py` on a node with internet access.
- `NaN logits`: report it with the log; do not work around it by changing the attention implementation silently, because that changes the platform record.
- Anything about a changed setting on resume: the directory was written under other settings; start that model's directory afresh rather than overriding the check.
