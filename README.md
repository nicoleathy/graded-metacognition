# Graded Metacognition in LLMs

This repository extends [ESMA](https://arxiv.org/abs/2602.02605) from binary Yes/No self-knowledge to graded A-D confidence probes and rewards.

## Quick Start

```bash
pip install -r requirements.txt
```

### Baseline evaluation

```bash
python scripts/evaluate_qa.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset trivia_qa \
  --meta-type graded \
  --num-samples 2000
```

### Train graded ESMA

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

### Transfer evaluation

```bash
python scripts/evaluate_transfer.py \
  --model outputs/qwen2.5-3b-graded-esma/checkpoints/qwen2.5-3b-graded-esma_iter750 \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --datasets trivia_qa gsm8k mmlu \
  --meta-types binary graded \
  --extract-logits \
  --save-details
```

## Main Changes from ESMA

| Component | ESMA | This Work |
|---|---|---|
| Probe | Binary Yes/No | Binary + graded A-D |
| Reward | Discrete binary reward | Continuous graded reward |
| Metrics | `d'_type2` | `p_yes`, gamma, AUROC, grade separation |
| Evaluation | TriviaQA | TriviaQA, GSM8K, MMLU |

## Caveats

Graded ESMA training was run only on Qwen2.5-Instruct 1.5B and 3B. Cross-family experiments are baseline-only. GSM8K results should be interpreted cautiously because keyword-only accuracy is low. MMLU is affected by A-D answer-label collision.

## Acknowledgments

This repository builds on [ESMA](https://arxiv.org/abs/2602.02605) by Park et al. (2026). Generative AI assistants were used for prose editing, LaTeX/table formatting, and code debugging; all scientific content and final decisions are the authors' own.
