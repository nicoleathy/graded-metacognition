# How Confident, Not Just Whether: Graded Metacognition in LLMs

This repository contains the code for *"How Confident, Not Just Whether: Graded Metacognitive Training for Language Models"*.

We extend [ESMA](https://arxiv.org/abs/2602.02605) (Evolution Strategy for Metacognitive Alignment) with graded confidence probes and rewards, enabling LLMs to express calibrated multi-level confidence rather than binary Yes/No self-knowledge.

## Key Results

- **Binary d' overstates metacognitive ability** — graded gamma correlation reveals that models with strong binary d' (0.87) have poor calibrated confidence (γ = 0.35)
- **Graded ESMA improves gamma by 84%** (0.21 → 0.39) on a 3B model while simultaneously improving accuracy (35.6% → 52.0%)
- **Rescues anti-calibrated models** — 1.5B model goes from γ = −0.05 to γ = 0.33
- **Cross-probe generalization** — training with A–D probes also improves FOK (1–5) performance
- **Domain-specific** — metacognitive gains do not transfer from factual knowledge to math reasoning

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/esma-graded.git
cd esma-graded
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, transformers, accelerate, scipy, datasets.

## Quick Start

### 1. Baseline evaluation (no training)

```bash
python scripts/evaluate_qa.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset trivia_qa \
    --meta-type graded \
    --num-samples 2000
```

Supports `--meta-type binary`, `graded`, `fok`, or `numeric`.

### 2. Train with graded ESMA

```bash
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --reward-type graded \
    --meta-type graded \
    --sigma 1e-3 \
    --alpha 5e-4 \
    --num-iterations 750 \
    --population-size 32 \
    --output-dir outputs/qwen3b-graded-esma
```

For binary ESMA comparison, use `--reward-type esma --meta-type binary`.

### 3. Transfer evaluation

```bash
python scripts/evaluate_transfer.py \
    --model outputs/qwen3b-graded-esma/checkpoints/qwen3b-graded-esma_iter600 \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --datasets trivia_qa gsm8k mmlu \
    --meta-types binary graded fok \
    --num-samples 1000
```

## Project Structure

```
esma/
├── prompt.py          # Meta-question templates (binary, graded, FOK, JOL, numeric)
├── metric.py          # d'_type2, gamma correlation, AUROC, calibration error
├── reward.py          # Binary and graded reward functions
├── evolution.py       # ES weight perturbation (unchanged from ESMA)
├── dataset.py         # ESDataset + GradedESDataset
├── utils.py           # Logging, seeding
└── data/
    ├── __init__.py    # Dataset registry
    ├── gsm8k.py       # GSM8K loader
    └── mmlu.py        # MMLU loader
scripts/
├── train_es.py        # Main training loop (supports --meta-type flag)
├── evaluate_qa.py     # Single-dataset evaluation
└── evaluate_transfer.py  # Cross-dataset transfer evaluation
```

## What We Changed from ESMA

The [original ESMA codebase](https://github.com/cosmoquester/ESMA) uses binary Yes/No probes and d'\_type2. We made the following modifications:

| Component | Original ESMA | Our Extension |
|-----------|--------------|---------------|
| **Probes** | Binary Yes/No | + Graded (A–D), FOK (1–5), JOL (1–5), Numeric (1–10) |
| **Metrics** | d'\_type2, alignment | + Gamma correlation, Type 2 AUROC, calibration error, grade separation |
| **Reward** | Discrete {0, 1, 2} | Continuous [0, 2] proportional to confidence-correctness alignment |
| **Datasets** | TriviaQA, FreebaseQA, NQ | + GSM8K, MMLU |
| **Training** | Binary parsing only | Conditional graded/binary routing via `--meta-type` |

**Unchanged:** `evolution.py` (ES mechanism is reward-agnostic), `utils.py`, all original data loaders.

## Metacognitive Probes

| Probe | Scale | Example Response | Parsed As |
|-------|-------|-----------------|-----------|
| Binary | Yes / No | "Yes" | 1 |
| Graded | A / B / C / D | "A" (certain) | 3 |
| FOK | 1–5 | "4" (think I know) | 4 |
| Numeric | 1–10 | "7" | 7 |

## Graded Reward Function

```
r(c, g) = 1 + g/g_max    if correct (c=1)
         = 1 - g/g_max    if incorrect (c=0)
```

Correct + high confidence → 2.0 (best). Incorrect + high confidence → 0.0 (worst). This incentivizes proportional calibration rather than binary discrimination.

## Metrics

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **d'\_type2** | Binary metacognitive sensitivity (SDT) | −∞ to +∞ |
| **Gamma (γ)** | Rank correlation between confidence and accuracy | −1 to +1 |
| **Type 2 AUROC** | Threshold-invariant metacognitive sensitivity | 0 to 1 |
| **Grade separation** | Mean confidence difference (correct − incorrect) | −g\_max to +g\_max |
| **Calibration error** | Accuracy deviation from expected per confidence level | 0 to 1 |

## Reproducing Paper Results

**Step 1: Baselines** (Table 1 — ~20 min per model on 1 GPU)
```bash
for model in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
    for meta in binary graded fok; do
        python scripts/evaluate_qa.py --model $model --meta-type $meta --num-samples 2000
    done
done
```

**Step 2: Training** (Figure 2 — ~13 hours per run on 4× A100)
```bash
# 3B Graded ESMA
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --reward-type graded --meta-type graded \
    --num-iterations 750 --output-dir outputs/qwen3b-graded-esma

# 3B Binary ESMA (comparison)
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --reward-type esma --meta-type binary \
    --num-iterations 750 --output-dir outputs/qwen3b-binary-esma

# 1.5B Graded ESMA
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --reward-type graded --meta-type graded \
    --num-iterations 750 --output-dir outputs/qwen1.5b-graded-esma
```

**Step 3: Transfer** (Figure 3 — ~2 hours on 1 GPU)
```bash
python scripts/evaluate_transfer.py \
    --model outputs/qwen3b-graded-esma/checkpoints/qwen3b-graded-esma_iter600 \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --datasets trivia_qa gsm8k mmlu --meta-types binary graded fok
```

## Citation

## ```bibtex
## @article{your_citation_2026,
##  title={How Confident, Not Just Whether: Graded Metacognitive Training for Language Models},
##  author={Your Name},
##  journal={arXiv preprint},
##  year={2026}
## }
## ```

## Acknowledgments

This work builds on [ESMA](https://arxiv.org/abs/2602.02605) by Park et al. (2026). We thank the authors for releasing their codebase.
