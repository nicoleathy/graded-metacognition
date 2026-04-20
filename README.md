# How Confident, Not Just Whether: Graded Metacognition in LLMs

Code for *"How Confident, Not Just Whether: Graded Metacognitive Training for Language Models"*. We extend [ESMA](https://arxiv.org/abs/2602.02605) with graded confidence probes and rewards, enabling LLMs to express calibrated multi-level confidence rather than binary Yes/No self-knowledge.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Baseline evaluation
python scripts/evaluate_qa.py --model Qwen/Qwen2.5-3B-Instruct \
    --dataset trivia_qa --meta-type graded --num-samples 2000

# 2. Train graded ESMA
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-3B-Instruct --reward-type graded --meta-type graded \
    --sigma 1e-3 --alpha 5e-4 --num-iterations 750 \
    --output-dir outputs/qwen3b-graded-esma

# 3. Transfer evaluation
python scripts/evaluate_transfer.py \
    --model outputs/qwen3b-graded-esma/checkpoints/qwen3b-graded-esma_iter750 \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --datasets trivia_qa gsm8k mmlu --meta-types binary graded fok \
    --extract-logits --save-details
```

## What We Changed from ESMA

| Component | Original ESMA | Our Extension |
|-----------|--------------|---------------|
| Probes | Binary Yes/No | + Graded (A–D), FOK (1–5), JOL (1–5) |
| Metrics | d'\_type2 | + Gamma, Type 2 AUROC, calibration error |
| Reward | Discrete {0, 1, 2} | Continuous [0, 2] proportional to calibration |
| Datasets | TriviaQA | + GSM8K, MMLU |

Unchanged: `evolution.py` (ES is reward-agnostic), `utils.py`, all original data loaders.

## Project Structure

```
esma/
├── prompt.py       # Binary + graded meta-question templates
├── metric.py       # Goodman-Kruskal gamma, d', AUROC, calibration error, logit confidence
├── reward.py       # Binary and graded reward functions
├── evolution.py    # ES perturbation (unchanged)
├── dataset.py      # ESDataset + GradedESDataset
└── data/           # TriviaQA, GSM8K, MMLU loaders
scripts/
├── train_es.py          # Training (--meta-type binary|graded|fok)
├── evaluate_qa.py       # Single-dataset evaluation
└── evaluate_transfer.py # Cross-dataset transfer evaluation (--extract-logits for implicit confidence)
```

## Acknowledgments

Builds on [ESMA](https://arxiv.org/abs/2602.02605) by Park et al. (2026).
