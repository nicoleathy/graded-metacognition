# Running the outstanding Paper 1 jobs on a cluster

Status when this file was written (28 October 2026, afternoon). Six models are finished on the Mac Studio (Qwen2.5-1.5B, 3B and 7B, Qwen3-4B, Llama-3.1-8B, Mistral-7B-v0.3; five datasets
each). **Qwen3-8B is still running on the Mac and finishes there: do not run it on the cluster.** The Mac queue has been stopped, so nothing else will start there.
The cluster takes everything that has not been started (`experiments/paper1/cluster/jobs.tsv`):

| Model | Datasets | GPU memory | Status |
|---|---|---|---|
| Qwen/Qwen3-1.7B | mmlu triviaqa mmlu_pro_eng supergpqa_mfg webquestions | 16 GB | done (ingested on the Mac) |
| Qwen/Qwen2.5-14B-Instruct | mmlu supergpqa_mfg (validity-only run) | 40 GB or more | done (ingested on the Mac) |
| meta-llama/Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, google/gemma-2-2b-it, gemma-2-9b-it | all five | 16 to 24 GB | **ready_gated**: gated models; the owner (who has accepted the licences) downloads the weights to `HF_HOME` with their own Hugging Face token first (nothing in this repository reads a token): see "Gated weights" below. Gemma-2 caps its attention scores (`attn_logit_softcapping` 50), which the sdpa path of the pinned transformers (4.56.0) ignores, so the runner picks eager attention for it on any device (`default_attention` in `probe_validity/generate.py`); the run records `attention: eager` |

Update, 1 November: the first two rows are finished and ingested on the Mac, and the platform calibration run (below) was done. The next cluster jobs are the four gated models (weights first: "Gated weights" below).

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

## Which GPUs

bfloat16 must be native: Ampere or newer (RTX 3090, A5000, A40, A100, L40/L40S, H100, H200). Do not use V100, RTX 8000 or RTX 2080 cards, which have no native bfloat16. A model keeps one card type for all its datasets (the run records the card in `meta.json` as `gpu`, and the package note records the host). The 1B to 3B models and Gemma-2-2B fit a 24 GB card; Gemma-2-9B and Qwen2.5-14B need 40 GB or more at the fixed batch size of 16. Do not lower the batch size to make a model fit: it is part of the recorded settings.

## Gated weights (Llama-3.2-1B and 3B, Gemma-2-2B and 9B)

On a login node with internet access, as the account that accepted the licences. The token is entered by its owner, is never written into this repository, job scripts or logs, and is removed afterwards:

```bash
export HF_HOME=<scratch>/hf_cache                         # the same cache the jobs use
hf auth login                                             # paste the token when asked
for m in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct google/gemma-2-2b-it google/gemma-2-9b-it; do
  hf download "$m" --exclude "*.bin" "*.pt" "*.pth" "*.onnx" "*.gguf" "*.msgpack" "*.h5" "original/*" "*.ot"     # safetensors only, as the runner reads
done
hf auth logout
```

Then `bash experiments/paper1/cluster/submit_all.sh` (review) and `--go`; jobs run offline as before. The four models need about 2.5, 6.4, 5.3 and 18.5 GB.

## Platform calibration run (one extra directory; never `runs/sweep`)

The study keeps each model on one platform. To measure what the platform itself changes, one model that ran on the Mac is run again here, into its own directory, on the same items,
prompts, decoding, dtype, batch size and **model revision**. It is a calibration, not part of the study: nothing from it is added to `runs/sweep/`.

The model is `Qwen/Qwen2.5-1.5B-Instruct`, dataset `mmlu` (2,000 items), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (chat-template SHA-256 starts `cd8e9439f057`). The runner finds a model
through the cache's `refs/main`, so that file must name this revision, and the run's `meta.json` must then record it.

```bash
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1                    # public weights, fetched anonymously
REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', revision='$REV', token=False, ignore_patterns=['*.bin','*.pt','*.pth','*.onnx','*.gguf','*.msgpack','*.h5','original/*','*.ot'])"
D=$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct; mkdir -p $D/refs && printf %s $REV > $D/refs/main
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash experiments/paper1/cluster/run_job.sh Qwen/Qwen2.5-1.5B-Instruct mmlu platform_check_cuda
```

The job writes `experiments/paper1/cluster/packages/p1_calibration_platform_check_cuda_<model>_<time>.tar.gz` and a `.sha256` file. On the Mac:

```bash
shasum -a 256 -c p1_calibration_platform_check_cuda_<model>_<time>.tar.gz.sha256
mkdir -p experiments/paper1/runs/platform_check_cuda && tar -xzf p1_calibration_platform_check_cuda_<model>_<time>.tar.gz -C experiments/paper1/runs/platform_check_cuda/
python experiments/paper1/cluster/compare_platforms.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset mmlu \
  --a experiments/paper1/runs/sweep --b experiments/paper1/runs/platform_check_cuda --json platform_check.json
```

The comparison first verifies both directories (all rows present, every file's SHA-256 as recorded). It then reports whether the settings that must be equal are, which platform fields differ,
how many replies are identical per condition, the drift of the first-token label log-probabilities, paired differences (accuracy, compliance, AUROC) with bootstrap intervals, and the screen
verdicts of both runs side by side. The sampled decodes of the letters probe draw from each device's own random generator, so they are not expected to agree item by item.

## Resources

Not measured on the cluster. For reference, the Mac Studio (M4 Max) needed these hours of generation for all five datasets: Qwen2.5-1.5B 2.1, Qwen2.5-3B 2.9, Qwen3-4B 6.3, Qwen2.5-7B 6.5,
Llama-3.1-8B 8.8, Mistral-7B 10.0. A single modern GPU should be several times faster; the default `--time=12:00:00` in `sweep.sbatch` is generous, so override it for your queue
(`sbatch --time=...`). Memory: bfloat16 weights plus a batch of 16 short prompts, so a 16 GB GPU is enough up to about 7B, 24 GB for the 9B model, 40 GB or more for the 14B model.

## If something fails

- `no GPU visible`: the job landed on a node without a visible GPU; request one (`--gres=gpu:1`).
- Hub or dataset errors under `HF_HUB_OFFLINE=1`: the prefetch did not complete for that model or dataset; rerun `prefetch.py` on a node with internet access.
- `NaN logits`: report it with the log; do not work around it by changing the attention implementation silently, because that changes the platform record.
- Anything about a changed setting on resume: the directory was written under other settings; start that model's directory afresh rather than overriding the check.

## Replication (added 1 November 2026)

The study is finished (13 models, five datasets, packages ingested). The replication repeats its interventions where they were not run, and is the next cluster work. It does not touch the
study's runs: `runs/sweep` and the frozen protocol stay as they are, and `run_sweep.py` refuses to run the sweep under another wording or to write a replication into `runs/sweep`.

**What it runs.** The plan (its hypotheses and decision rules) is frozen with the study's own records, which are not in this repository, before any replication output exists; what the cluster needs is stated here and in the code. Per model: the three multiple-choice sets
(`mmlu`, `mmlu_pro_eng`, `supergpqa_mfg`) under three wordings of the confidence probe (`PROBE_WORDING` = `w1` the study's text, `w2` and `w3` two paraphrases), each with the embedded
instruction, the stem-only prompt, the answer-label control and the label x order design (18 conditions plus the direct answer, no sampled decodes). About 1.2 times the generation of one study sweep per
model, as three directories: `runs/replication_w1`, `runs/replication_w2`, `runs/replication_w3`.

**How.** One job per model; the job runs the three wordings one after the other, so a model stays on one card.

```bash
bash experiments/paper1/cluster/submit_replication.sh            # dry run: prints one sbatch line per ready model of replication_jobs.tsv
bash experiments/paper1/cluster/submit_replication.sh --go
# or by hand, one model:  sbatch --gres=gpu:1 --time=24:00:00 experiments/paper1/cluster/replication.sbatch Qwen/Qwen3-1.7B
# one dataset or wording again after a failure (finished directories are skipped):  experiments/paper1/cluster/run_replication.sh Qwen/Qwen3-1.7B "mmlu" "w2"
```

The rules above apply unchanged (clean checkout of the pushed commit, no `--allow-config-change`, defaults, offline, no analysis on the cluster). Specific to the replication:

- **One card type per model for all three wordings.** For the six models whose study run was on the cluster (Qwen3-1.7B and Qwen2.5-14B on an A100-SXM4-40GB; Llama-3.2-1B and 3B and Gemma-2-2B on an RTX A5000; Gemma-2-9B on an
  H100 NVL) prefer the card of the study run. The other seven models ran on the Mac in the study: any single card type is fine, and their replication is a different platform, so only contrasts inside the replication are compared.
- **Weights.** The seven Mac models (and the gated Llama-3.1-8B and Mistral-7B-v0.3) are probably not in the cluster's `HF_HOME`: fetch them first (`prefetch.py`; the gated ones by their owner), as for the study.
- `run_job.sh` verifies each directory (`verify_runs.py --exp replication_<wording>` must print only `ok` lines) and `package_runs.sh` writes `p1_replication_<wording>_<model>_<time>.tar.gz` with a `.sha256`. Publish the packages to the
  runs repository as before, with a README row (card, host, attention, commit) per package.
- Order of the models in `replication_jobs.tsv` is the priority. Use as many GPUs as the queue allows: the models are independent jobs.
