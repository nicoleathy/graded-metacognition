"""Evaluate metacognitive transfer: does ESMA trained on TriviaQA
generalize to reasoning tasks (GSM8K, MMLU)?

Usage:
    # Compare ESMA model vs baseline on multiple benchmarks and meta types
    python scripts/evaluate_transfer.py \
        --model path/to/esma-trained-model \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --datasets trivia_qa gsm8k mmlu \
        --meta-types binary graded fok \
        --num-samples 1000

    # Quick single-model evaluation
    python scripts/evaluate_transfer.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --datasets trivia_qa \
        --meta-types binary graded
"""

import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from esma.data import (
    load_gsm8k_meta,
    load_mmlu_meta,
    load_trivia_qa_meta,
    load_freebase_qa_meta,
    load_nq_open_meta,
    load_web_questions_meta,
)
from esma.dataset import ESDataset
from esma.metric import (
    IGNORE_VALUE,
    correctness_by_inclusion,
    graded_calibration_error,
    graded_meta_metrics,
    meta_metrics,
    metacognitive_resolution,
    multi_threshold_d_prime,
    parse_graded_response,
    parse_numeric_response,
    type2_auroc,
    type2_d_prime,
)
from esma.prompt import (
    DIRECT_QA_PROMPT,
    FOK_META_QA_PROMPT,
    GRADED_META_QA_PROMPT,
    META_PROMPT_TYPES,
    META_QA_PROMPT,
    NUMERIC_META_QA_PROMPT,
)
from esma.utils import get_logger, seed_everything

torch.set_grad_enabled(False)

# fmt: off
parser = argparse.ArgumentParser(description="Evaluate metacognitive transfer across benchmarks")
parser.add_argument("--model", type=str, required=True, help="Primary model to evaluate (e.g. ESMA-trained)")
parser.add_argument("--base-model", type=str, help="Baseline model for comparison (e.g. original pre-ESMA)")
parser.add_argument("--datasets", nargs="+", default=["trivia_qa"], choices=["trivia_qa", "gsm8k", "mmlu", "freebase_qa", "nq_open", "web_questions"], help="Datasets to evaluate on")  # noqa: E501
parser.add_argument("--meta-types", nargs="+", default=["binary", "graded"], choices=list(META_PROMPT_TYPES.keys()), help="Meta prompt types to evaluate")  # noqa: E501
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per dataset")
parser.add_argument("--max-input-length", type=int, default=256, help="Maximum input length")
parser.add_argument("--max-direct-tokens", type=int, default=64, help="Max new tokens for direct answers")
parser.add_argument("--max-meta-tokens", type=int, default=16, help="Max new tokens for meta answers")
parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--output-dir", type=str, default="transfer_results", help="Output directory")
parser.add_argument("--save-details", action="store_true", help="Save per-example details to TSV")
# fmt: on


DATASET_LOADERS = {
    "trivia_qa": lambda ns: load_trivia_qa_meta(split="validation", num_samples=ns),
    "gsm8k": lambda ns: load_gsm8k_meta(split="test", num_samples=ns),
    "mmlu": lambda ns: load_mmlu_meta(split="test", num_samples=ns),
    "freebase_qa": lambda ns: load_freebase_qa_meta(split="test", num_samples=ns),
    "nq_open": lambda ns: load_nq_open_meta(split="validation", num_samples=ns),
    "web_questions": lambda ns: load_web_questions_meta(split="test", num_samples=ns),
}


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data,
    meta_type: str,
    args,
    logger,
) -> tuple[dict, list[dict]]:
    """Evaluate a single model on a single dataset with a single meta type.

    Returns:
        Tuple of (aggregate_metrics_dict, per_example_details_list)
    """
    meta_prompt = META_PROMPT_TYPES[meta_type]

    dataset = ESDataset(
        data,
        tokenizer,
        max_length=args.max_input_length,
        prompt=DIRECT_QA_PROMPT,
        meta_prompt=meta_prompt,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ESDataset.pad_collate_fn,
    )

    all_direct_outputs = []
    all_meta_outputs = []
    all_answers = []
    all_questions = []
    all_question_ids = []

    for batch in tqdm(loader, desc=f"  {meta_type}", leave=False):
        # Direct question
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_direct_tokens,
        )
        generated = outputs[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Meta question
        meta_ids = batch["meta_input_ids"].to(model.device)
        meta_mask = batch["meta_attention_mask"].to(model.device)
        meta_out = model.generate(
            input_ids=meta_ids,
            attention_mask=meta_mask,
            max_new_tokens=args.max_meta_tokens,
        )
        meta_gen = meta_out[:, meta_ids.shape[1] :]
        meta_decoded = tokenizer.batch_decode(meta_gen, skip_special_tokens=True)

        all_direct_outputs.extend(decoded)
        all_meta_outputs.extend(meta_decoded)
        all_answers.extend(batch["answers"])
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["question_id"])

    # Compute metrics
    if meta_type == "binary":
        correctness, yes_list, yes_failures, no_failures, alignments = meta_metrics(
            all_direct_outputs, all_meta_outputs, all_answers
        )
        accuracy = sum(correctness) / len(correctness)
        yes_ratio = sum(yes_list) / len(yes_list)
        d_prime = type2_d_prime(correctness, yes_list)
        alignment_score = sum(alignments) / len(alignments)

        yfr_items = [1 - c for c, y in zip(correctness, yes_list) if y == 1]
        nfr_items = [c for c, y in zip(correctness, yes_list) if y == 0]
        yfr = sum(yfr_items) / len(yfr_items) if yfr_items else 0.0
        nfr = sum(nfr_items) / len(nfr_items) if nfr_items else 0.0

        results = {
            "accuracy": accuracy,
            "d_prime_type2": d_prime,
            "raw_alignment": alignment_score,
            "yes_ratio": yes_ratio,
            "yes_failure_ratio": yfr,
            "no_failure_ratio": nfr,
        }
        meta_values = yes_list
    else:
        graded_results, correctness, grades = graded_meta_metrics(
            all_direct_outputs, all_meta_outputs, all_answers, meta_type=meta_type
        )
        results = graded_results
        meta_values = grades

    # Build per-example details
    details = []
    for i in range(len(all_questions)):
        details.append({
            "question_id": all_question_ids[i],
            "question": all_questions[i],
            "ground_truths": str(all_answers[i]),
            "prediction": all_direct_outputs[i],
            "meta_output": all_meta_outputs[i],
            "correct": correctness[i],
            "meta_value": meta_values[i],
        })

    return results, details


def main(args):
    logger = get_logger(__name__)
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup models to evaluate
    models_to_eval = {"primary": args.model}
    if args.base_model:
        models_to_eval["baseline"] = args.base_model

    all_results = {}

    for model_label, model_path in models_to_eval.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model: {model_label} ({model_path})")
        logger.info(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")
        model.eval()

        for dataset_name in args.datasets:
            logger.info(f"\nDataset: {dataset_name}")

            data = DATASET_LOADERS[dataset_name](args.num_samples)
            logger.info(f"  Loaded {len(data)} samples")

            for meta_type in args.meta_types:
                key = f"{model_label}/{dataset_name}/{meta_type}"
                logger.info(f"  Evaluating: {key}")

                results, details = evaluate_model_on_dataset(
                    model, tokenizer, data, meta_type, args, logger
                )

                all_results[key] = results
                logger.info(f"  Results: {json.dumps(results, indent=2)}")

                # Save per-example details if requested
                if args.save_details:
                    detail_path = os.path.join(
                        args.output_dir,
                        f"{model_label}_{dataset_name}_{meta_type}_details.tsv",
                    )
                    with open(detail_path, "w", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=details[0].keys(), delimiter="\t")
                        writer.writeheader()
                        writer.writerows(details)
                    logger.info(f"  Details saved to: {detail_path}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # Save aggregate results
    output_path = os.path.join(args.output_dir, "transfer_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {output_path}")

    # Print comparison table if we have both models
    if args.base_model:
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*80}")
        for dataset_name in args.datasets:
            for meta_type in args.meta_types:
                base_key = f"baseline/{dataset_name}/{meta_type}"
                esma_key = f"primary/{dataset_name}/{meta_type}"
                if base_key in all_results and esma_key in all_results:
                    logger.info(f"\n{dataset_name} ({meta_type}):")
                    for metric in all_results[base_key]:
                        base_val = all_results[base_key][metric]
                        esma_val = all_results[esma_key][metric]
                        if isinstance(base_val, float):
                            delta = esma_val - base_val
                            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                            logger.info(
                                f"  {metric:30s}: {base_val:8.4f} → {esma_val:8.4f} ({arrow}{abs(delta):.4f})"
                            )


if __name__ == "__main__":
    main(parser.parse_args())
