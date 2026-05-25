# Graded Metacognition in LLMs

This repository extends [ESMA](https://arxiv.org/abs/2602.02605) from binary Yes/No self-knowledge probes to graded confidence probes and rewards. Instead of asking whether a model knows an answer, we evaluate and train models to express multi-level confidence over a four-point A-D scale.

The paper shows that binary metacognitive probes can collapse to near-constant responses, especially out-of-domain, while graded probes expose calibration structure that binary metrics can hide.

## Key Contributions

- Graded confidence probes using a four-level A-D scale.
- A continuous graded ESMA reward:

  ```text
  r(c,g) = 1 + g/g_max, if c = 1
  r(c,g) = 1 - g/g_max, if c = 0
  ```

- Probe-health diagnostics for binary collapse using `p_yes`.
- Bias-aware metacognition metrics:
  - Goodman-Kruskal gamma
  - Type-2 AUROC
  - Grade separation
  - Verbal-logit correlation
  - Logit-based Type-2 AUROC
- Transfer evaluation on TriviaQA, MMLU, and GSM8K.
- Cross-family baseline evaluation across Qwen2.5, Qwen3, Llama-3.2, Mistral, and Gemma-2.

## Quick Start

```bash
pip install -r requirements.txt
```

### 1. Baseline evaluation

```bash
python scripts/evaluate_qa.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset trivia_qa \
    --meta-type graded \
    --num-samples 2000
```

### 2. Train graded ESMA

```bash
accelerate launch --num_processes=4 scripts/train_es.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --reward-type graded \
    --meta-type graded \
    --sigma 1e-3 \
    --alpha 5e-4 \
    --num-iterations 750 \
    --output-dir outputs/qwen2.5-3b-graded-esma
```

### 3. Transfer evaluation

```bash
python scripts/evaluate_transfer.py \
    --model outputs/qwen2.5-3b-graded-esma/checkpoints/qwen2.5-3b-graded-esma_iter750 \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --datasets trivia_qa gsm8k mmlu \
    --meta-types binary graded \
    --extract-logits \
    --save-details
```

## What We Changed from ESMA

| Component | Original ESMA | This Work |
|---|---|---|
| Metacognitive probe | Binary Yes/No | Binary + graded A-D confidence |
| Reward | Discrete binary reward | Continuous confidence-calibration reward |
| Main metric | `d'_type2` | `p_yes`, gamma, Type-2 AUROC, grade separation |
| Internal alignment | Not the main focus | Verbal-logit correlation and logit AUROC |
| Evaluation domains | TriviaQA | TriviaQA, MMLU, GSM8K |
| Cross-family validation | Not the main focus | Qwen2.5, Qwen3, Llama-3.2, Mistral, Gemma-2 |

The evolution-strategy update remains reward-agnostic; the main changes are in the probe format, reward function, metrics, and transfer evaluation.

## Experimental Setup

The main training experiments use Qwen2.5-Instruct at 1.5B and 3B. Baseline evaluation also includes Qwen2.5-7B and cross-family models from Qwen3, Llama-3.2, Mistral, and Gemma-2.

Training uses TriviaQA with:

- Population size: 32
- sigma: `1e-3`
- alpha: `5e-4`
- Iterations: 750
- Training samples: 2,000
- Hardware: 4 x A100 GPUs

Transfer evaluation uses TriviaQA, GSM8K, and MMLU. GSM8K is capped at 1,319 examples.

## Project Structure

```text
esma-graded/
├── prompt.py       # Binary and graded meta-question templates
├── metric.py       # d', p_yes, gamma, AUROC, grade separation, logit confidence
├── reward.py       # Binary and graded reward functions
├── evolution.py    # Evolution-strategy perturbation and update
├── dataset.py      # ESDataset and GradedESDataset
└── data/           # TriviaQA, GSM8K, and MMLU loaders

scripts/
├── train_es.py          # ESMA training
├── evaluate_qa.py       # Single-dataset evaluation
└── evaluate_transfer.py # Cross-dataset transfer and logit extraction
```

## Important Caveats

- Graded ESMA training was only run on Qwen2.5-Instruct at 1.5B and 3B.
- Cross-family results are baseline evaluations only; cross-family fine-tuning remains untested.
- GSM8K accuracy is low under the keyword-only protocol, so GSM8K results should be interpreted as calibration-stress results rather than strong reasoning-transfer claims.
- MMLU is affected by A-D answer-label collision because the confidence scale shares labels with the multiple-choice answer space.
- Trained graded models can overuse endpoint grades, so high gamma may reflect decisive separation rather than balanced use of the full scale.
- Most non-3B-graded configurations use a single seed.

## AI Assistance Disclosure

Generative AI assistants were used during paper and code preparation for prose editing, LaTeX/table formatting, and code debugging. The scientific content, experimental design, analyses, result interpretation, and final manuscript decisions are the authors' own.

## Acknowledgments

This repository builds on [ESMA](https://arxiv.org/abs/2602.02605) by Park et al. (2026).
