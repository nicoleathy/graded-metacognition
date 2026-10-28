# Paper 1: validity of verbalized-confidence probes

Code for measuring whether a verbalized-confidence probe measures anything, before reading its score. Library code is in
[`probe_validity/`](../../probe_validity), tests in [`tests/probe_validity/`](../../tests/probe_validity). The legacy `esma/`
package is untouched.

## Setup

```bash
python3.12 -m venv ~/venvs/paper1 && source ~/venvs/paper1/bin/activate
pip install -r requirements-paper1.txt
python -m pytest tests -q          # the whole suite; a sweep must not start if anything fails
```

The library never sends a Hugging Face token implicitly (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`); gated models need an explicit opt-in.

## What is frozen, and where

| File | Content |
|---|---|
| `frozen/screen_config.json` | thresholds of the eight-check screen, validated on synthetic probes and frozen before any real data was screened |
| `frozen/protocol.json` | every condition of the sweep (prompts hashed on a reference item, decoding, subsets, scales) |
| `manifests/*.json` | the exact item ids of each dataset, with the Hugging Face dataset revision and a SHA-256 |
| `results/screen_validation.json` | 7 simulators x 1,000 seeds: the screen's verdict equals the expected verdict in every run |

Tests fail if the code drifts from the frozen files. Every run directory records the model revision, the full generation
configuration, the protocol hash and a SHA-256 of the source code.

## Commands

```bash
python experiments/paper1/make_manifests.py            # download, audit, write manifests (once)
python experiments/paper1/validate_screen.py           # synthetic validation of the screen; writes the frozen config
python experiments/paper1/freeze_protocol.py           # writes frozen/protocol.json
python experiments/paper1/run_e10.py --model Qwen/Qwen2.5-7B-Instruct --n 500      # the 2 x 4 pilot on 500 MMLU items
python experiments/paper1/run_sweep.py --model Qwen/Qwen2.5-7B-Instruct            # every condition of the protocol, resumable
python experiments/paper1/analyze_sweep.py             # summaries of every run directory -> results/*.json
```

Runs are resumable: batches are fixed by prompt length, a restart redoes only batches with a missing row, and a line cut off
by an interrupted write is ignored. All conditions of one model must run on one machine; the run metadata records the device,
dtype and attention implementation.

## Platform notes

* Apple MPS: scaled-dot-product attention returns NaN for left-padded rows, so the runner uses eager attention there and raises
  on any NaN logit instead of reading it as a reply.
* transformers 4.56: a `generation_config` object passed to `generate()` has fields equal to the library defaults replaced by the
  model's Hub values (Qwen2.5 would sample with a 1.05 repetition penalty). The runner resets the model's own generation config and
  passes the decoding settings as arguments; a test checks that greedy decoding equals the argmax of the raw logits.

## Conditions

For MMLU: a clean and a legacy direct prompt; four probe formats (Yes/No, letters A-D, digits 1-4, words) each with the
answer-format instruction embedded in the question string (as the legacy loader built it) and content-only; two relabellings of
the letter scale; five sampled decodes of the letter probe. The other datasets (MMLU-Pro engineering, a SuperGPQA mechanical,
materials-processing and management-science slice, TriviaQA, WebQuestions) have no embedded variant.

## Audit additions (2026-10-27)

The methodology audit of 27 October 2026 (a private review) and the review of the earlier
draft led to these additions. None changes a frozen file: `frozen/screen_config.json` and `frozen/protocol.json` are as they were.

| File | Purpose |
|---|---|
| `panel.py` | the model panel (twelve core models and the 14B validity-only run), their order, and which need a licence: `python experiments/paper1/panel.py` |
| `run_panel_queue.sh`, `after_first_queue.sh` | run the rest of the panel, one model at a time, after the first queue; each model is smoke-tested on 16 items; a gated model runs only when its weights are already cached |
| `analysis.py` + `robustness.py` + `legacy_compare.py` | every summary now carries: verdicts by tier, the S2/S4 variants, monotonicity, threshold sensitivity, cross-fitting, answered-only and unfinished-reply AUROCs, stratified bootstrap, duplicate and pilot-overlap sensitivity, open-answer sensitivity, Cramér's V, and the conventional-versus-strict Yes/No comparison |
| `rankings.py` | does the conventional statistic order models differently from the strict AUROC? (Kendall tau, models that move) |
| `screen_power.py` -> `results/screen_power.json` | power and false alarms of the frozen screen over signal strength, sample size, overlapping labels, relabelling instability and skewed usage |
| `audit_sheets.py` -> `audit/` | `make` writes, per dataset, a blank annotator sheet (100 answer decisions and 100 probe replies; risky cases over-represented; the program's decision is never shown) and a key file; `score` computes Cohen's kappa between two annotators and the program's agreement with their consensus. Regenerate when more datasets finish |
| `download_panel.py` | downloads the public panel models that are not cached, anonymously (the runner reads the cache only; gated models are never touched and no token is read) |
| `verify_runs.py`, `bind_existing.py` | `verify_runs.py` checks every finished directory (all rows present, every file's SHA-256 as recorded) and `analyze_sweep.py` refuses a directory that fails it (`--skip-verify` overrides); `bind_existing.py` adds retroactive hashes to directories written before per-condition hashing |
| `check_open_scorer.py` | every item's first accepted answer, written four plain ways, must score correct on the real TriviaQA and WebQuestions manifests (0 failures now; 0.45% and 0.30% of items failed before the review fixes). Run before interpreting open-question results |
| `archive_source.py` -> `archive/source_<digest>.tar.gz` | a retrievable copy of the exact source, named by the code digest every run records (git-ignored; nothing is committed) |

Provenance guards in `run_sweep.py`: the dataset is loaded at the revision its manifest recorded; a directory is resumed only if the
item ids, prompt hashes and settings (model revision, decoding, template, library versions) equal those that wrote its rows
(`--allow-config-change` overrides, deliberately); `DONE` is written only when every condition has all its rows, never after
`--only` or `--limit`; `conditions.jsonl` binds each condition file's SHA-256 to the invocation that completed it.

Re-analysis after any change to the parser or scorer needs no new generation:

```bash
python experiments/paper1/analyze_sweep.py --exp sweep --n-boot 1000   # all finished directories -> results/sweep_summary.json
```
