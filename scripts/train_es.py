import argparse
import itertools
import logging
import os

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from esma.data import load_trivia_qa_meta
from esma.dataset import ESDataset
from esma.evolution import apply_evolution
from esma.metric import (
    IGNORE_VALUE,
    correctness_by_inclusion,
    graded_alignment,
    graded_d_prime,
    meta_metrics,
    metacognitive_resolution,
    parse_graded_response,
    parse_numeric_response,
    type2_d_prime,
)
from esma.prompt import META_PROMPT_TYPES, META_QA_PROMPT
from esma.reward import REWARD_TYPE_TO_FUNCTION
from esma.utils import get_logger, seed_everything

torch.set_grad_enabled(False)

# fmt: off
parser = argparse.ArgumentParser(description="Train TriviaQA ES")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")

g = parser.add_argument_group("Data")
g.add_argument("--max-input-length", type=int, default=128, help="Maximum input length")
g.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens")
g.add_argument("--num-samples", type=int, help="Number of samples to load")
g.add_argument("--num-val-samples", type=int, help="Number of samples to load for validation")

g = parser.add_argument_group("Evolution")
g.add_argument("--sigma", type=float, default=1e-3, help="Sigma")
g.add_argument("--alpha", type=float, default=5e-4, help="Alpha")
g.add_argument("--num-iterations", type=int, default=300, help="Number of iterations")
g.add_argument("--population-size", type=int, default=32, help="Population size")
g.add_argument("--num-data-per-iteration", "-n", type=int, default=256, help="Number of data per iteration")
g.add_argument("--reward-type", type=str, default="multilevel", choices=REWARD_TYPE_TO_FUNCTION.keys(), help="Reward type")  # noqa: E501

g = parser.add_argument_group("Metacognition")
g.add_argument("--meta-type", type=str, default="binary", choices=META_PROMPT_TYPES.keys(), help="Metacognition prompt type: binary (Yes/No), graded (ABCD), fok (1-5), numeric (1-10)")  # noqa: E501

g = parser.add_argument_group("Experiment Settings")
g.add_argument("--batch-size", type=int, default=256, help="Batch size")
g.add_argument("--num-workers", type=int, default=os.cpu_count() // 2, help="Number of workers")
g.add_argument("--seed", type=int, default=42, help="Random seed")
g.add_argument("--output-dir", type=str, help="Output directory")
g.add_argument("--model-save-interval", type=int, default=60, help="Model save interval")
g.add_argument("--evaluate-interval", type=int, default=60, help="Evaluate interval")
g.add_argument("--wandb-run-name", type=str, help="Wandb run name")
g.add_argument("--wandb-project", type=str, default="meta-cognition", help="Wandb project")
g.add_argument("--wandb-entity", type=str, default="cosmoquester", help="Wandb entity")
# fmt: on


def is_graded(meta_type: str) -> bool:
    """Check if the meta type uses graded (non-binary) responses."""
    return meta_type in ("graded", "fok", "jol", "numeric")


def parse_meta_outputs(meta_decoded_outputs: list[str], meta_type: str) -> list[int]:
    """Parse meta outputs based on meta type.

    Returns:
        For binary: list of 0/1 (yes/no)
        For graded: list of grade values (0-3 for ABCD, 1-5 for FOK, 1-10 for numeric)
    """
    if meta_type == "graded":
        return parse_graded_response(meta_decoded_outputs)
    elif meta_type in ("fok", "jol"):
        return parse_numeric_response(meta_decoded_outputs, scale=5)
    elif meta_type == "numeric":
        return parse_numeric_response(meta_decoded_outputs, scale=10)
    else:
        # binary: return 0/1 for yes/no
        from esma.metric import meta_yes
        return meta_yes(meta_decoded_outputs)


def compute_metrics_from_parsed(
    direct_correctness: list[int],
    meta_values: list[int],
    meta_type: str,
    reward_type: str,
    keep_length: bool = True,
) -> dict:
    """Compute metrics from parsed correctness and meta values.

    Handles both binary and graded meta types.
    """
    if is_graded(meta_type):
        # For graded: binarize at midpoint for d' and alignment
        if meta_type == "graded":
            threshold = 2  # B or higher = "Yes"
        elif meta_type in ("fok", "jol"):
            threshold = 3  # 3 or higher = "Yes"
        else:  # numeric
            threshold = 6  # 6 or higher = "Yes"

        binarized_yes = [1 if g >= threshold else 0 for g in meta_values]
        d_prime = type2_d_prime(direct_correctness, binarized_yes)
        alignment = graded_alignment(direct_correctness, meta_values, threshold)
        gamma = metacognitive_resolution(direct_correctness, meta_values)

        return {
            "d_prime_type2": d_prime,
            "meta_alignments": alignment,
            "gamma_correlation": gamma,
            "mean_grade": float(np.mean(meta_values)),
        }
    else:
        # Binary path
        from esma.metric import meta_alignment, meta_wrong_no, meta_wrong_yes

        yes_failures = meta_wrong_yes(direct_correctness, meta_values, keep_length)
        no_failures = meta_wrong_no(direct_correctness, meta_values, keep_length)
        alignments = meta_alignment(direct_correctness, meta_values)

        return {
            "d_prime_type2": type2_d_prime(direct_correctness, meta_values),
            "yes": meta_values,
            "yes_failures": yes_failures,
            "no_failures": no_failures,
            "meta_alignments": alignments,
        }


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    max_new_tokens: int,
    reward_type: str,
    meta_type: str = "binary",
) -> dict[str, torch.Tensor]:
    all_direct_correctness = []
    all_meta_values = []
    all_meta_alignments = []
    all_rewards = []

    # Extra binary-only lists
    all_yes_failures = []
    all_no_failures = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        meta_input_ids = batch["meta_input_ids"].to(model.device)
        meta_attention_mask = batch["meta_attention_mask"].to(model.device)
        meta_outputs = model.generate(
            input_ids=meta_input_ids,
            attention_mask=meta_attention_mask,
            max_new_tokens=max_new_tokens,
        )
        meta_generated_tokens = meta_outputs[:, meta_input_ids.shape[1] :]
        meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

        direct_correctness = correctness_by_inclusion(decoded_outputs, batch["answers"])
        meta_values = parse_meta_outputs(meta_decoded_outputs, meta_type)

        metrics = compute_metrics_from_parsed(
            direct_correctness, meta_values, meta_type, reward_type, keep_length=True
        )

        all_direct_correctness.extend(direct_correctness)
        all_meta_values.extend(meta_values)
        all_meta_alignments.extend(
            metrics["meta_alignments"] if isinstance(metrics["meta_alignments"], list) else [metrics["meta_alignments"]]
        )
        all_rewards.extend(REWARD_TYPE_TO_FUNCTION[reward_type](direct_correctness, meta_values))

        if not is_graded(meta_type):
            all_yes_failures.extend(metrics.get("yes_failures", []))
            all_no_failures.extend(metrics.get("no_failures", []))

    device = model.device
    result = {
        "rewards": torch.tensor(all_rewards, dtype=torch.float32, device=device),
        "direct_correctness": torch.tensor(all_direct_correctness, dtype=torch.float32, device=device),
        "meta_alignments": torch.tensor(all_meta_alignments, dtype=torch.float32, device=device),
        "d_prime_type2": torch.tensor(
            type2_d_prime(
                all_direct_correctness,
                all_meta_values if not is_graded(meta_type) else [1 if g >= 2 else 0 for g in all_meta_values],
            ),
            dtype=torch.float32,
            device=device,
        ),
    }

    if is_graded(meta_type):
        result["mean_grade"] = torch.tensor(float(np.mean(all_meta_values)), dtype=torch.float32, device=device)
        result["gamma_correlation"] = torch.tensor(
            metacognitive_resolution(all_direct_correctness, all_meta_values), dtype=torch.float32, device=device
        )
    else:
        result["yes"] = torch.tensor(all_meta_values, dtype=torch.float32, device=device)
        result["yes_failures"] = torch.tensor(all_yes_failures, dtype=torch.float32, device=device)
        result["no_failures"] = torch.tensor(all_no_failures, dtype=torch.float32, device=device)

    return result


def single_iteration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    iteration_batch: dict,
    accelerator: Accelerator,
    local_seeds: np.ndarray,
    sigma: float,
    batch_size: int,
    max_new_tokens: int,
    reward_type: str,
    meta_type: str = "binary",
) -> tuple[list[float], dict[str, torch.Tensor]]:
    all_direct_correctness = []
    all_meta_values = []
    all_meta_alignments = []
    rewards = []

    # Extra binary-only lists
    all_yes_failures = []
    all_no_failures = []

    for seed in local_seeds:
        apply_evolution(model, seed, absolute_scale=sigma)

        seed_rewards = []
        for i in range(0, len(iteration_batch["answers"]), batch_size):
            batch = {k: v[i : i + batch_size] for k, v in iteration_batch.items()}
            batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_side="left").to(
                accelerator.device
            )
            batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_side="left").to(
                accelerator.device
            )

            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            generated_tokens = outputs[:, batch["input_ids"].shape[1] :]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            batch["meta_input_ids"] = pad_sequence(batch["meta_input_ids"], batch_first=True, padding_side="left").to(
                accelerator.device
            )
            batch["meta_attention_mask"] = pad_sequence(
                batch["meta_attention_mask"], batch_first=True, padding_side="left"
            ).to(accelerator.device)
            meta_outputs = model.generate(
                input_ids=batch["meta_input_ids"],
                attention_mask=batch["meta_attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            meta_generated_tokens = meta_outputs[:, batch["meta_input_ids"].shape[1] :]
            meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

            # Parse correctness and meta values
            direct_correctness = correctness_by_inclusion(decoded_outputs, batch["answers"])
            meta_values = parse_meta_outputs(meta_decoded_outputs, meta_type)

            metrics = compute_metrics_from_parsed(
                direct_correctness, meta_values, meta_type, reward_type, keep_length=True
            )

            all_direct_correctness.extend(direct_correctness)
            all_meta_values.extend(meta_values)
            all_meta_alignments.extend(
                metrics["meta_alignments"]
                if isinstance(metrics["meta_alignments"], list)
                else [metrics["meta_alignments"]]
            )

            if not is_graded(meta_type):
                all_yes_failures.extend(metrics.get("yes_failures", []))
                all_no_failures.extend(metrics.get("no_failures", []))

            seed_rewards.extend(REWARD_TYPE_TO_FUNCTION[reward_type](direct_correctness, meta_values))

        rewards.append(np.mean(seed_rewards))
        apply_evolution(model, seed, absolute_scale=sigma, reverse=True)

    device = accelerator.device
    result = {
        "direct_correctness": torch.tensor(all_direct_correctness, dtype=torch.float32, device=device),
        "meta_alignments": torch.tensor(all_meta_alignments, dtype=torch.float32, device=device),
        "d_prime_type2": torch.tensor(
            type2_d_prime(
                all_direct_correctness,
                all_meta_values if not is_graded(meta_type) else [1 if g >= 2 else 0 for g in all_meta_values],
            ),
            dtype=torch.float32,
            device=device,
        ),
    }

    if is_graded(meta_type):
        result["mean_grade"] = torch.tensor(float(np.mean(all_meta_values)), dtype=torch.float32, device=device)
        result["gamma_correlation"] = torch.tensor(
            metacognitive_resolution(all_direct_correctness, all_meta_values), dtype=torch.float32, device=device
        )
    else:
        result["yes"] = torch.tensor(all_meta_values, dtype=torch.float32, device=device)
        result["yes_failures"] = torch.tensor(all_yes_failures, dtype=torch.float32, device=device)
        result["no_failures"] = torch.tensor(all_no_failures, dtype=torch.float32, device=device)

    return rewards, result


def main(args):
    accelerator = Accelerator()
    logger = get_logger(__name__)

    logger.info(f"[+] Accelerator device: {accelerator.device}")
    if not accelerator.is_main_process:
        logger.setLevel(logging.CRITICAL)
    logger.info(f"[+] Accelerator num_processes: {accelerator.num_processes}")
    logger.info(f"[+] Meta type: {args.meta_type}")
    logger.info(f"[+] Reward type: {args.reward_type}")

    if args.output_dir is not None and args.wandb_run_name is None:
        args.wandb_run_name = os.path.basename(args.output_dir)
    if args.output_dir is None and args.wandb_run_name is not None:
        args.output_dir = os.path.join("outputs", args.wandb_run_name)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"[+] Output directory: {args.output_dir}")
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        basename = os.path.basename(args.output_dir)
    else:
        checkpoint_dir = None
    if args.wandb_run_name is not None and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.output_dir,
        )
        run.config.update(vars(args))
    else:
        run = None

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    # Select meta prompt based on meta_type
    meta_prompt = META_PROMPT_TYPES[args.meta_type]
    logger.info(f"[+] Using meta prompt type: {args.meta_type}")

    logger.info("[+] Loading TriviaQA dataset...")
    train_data = load_trivia_qa_meta(split="train", num_samples=args.num_samples)
    val_data = load_trivia_qa_meta(split="validation", num_samples=args.num_val_samples)
    logger.info(f"[+] Total samples: {len(train_data)}")

    logger.info(f"[+] Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"[+] Tokenized dataset: {len(train_data)}")

    train_dataset = ESDataset(
        train_data,
        tokenizer,
        max_length=args.max_input_length,
        meta_prompt=meta_prompt,
    )
    val_dataset = ESDataset(
        val_data,
        tokenizer,
        max_length=args.max_input_length,
        meta_prompt=meta_prompt,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.num_data_per_iteration,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ESDataset.simple_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ESDataset.pad_collate_fn,
    )
    infinite_loader = itertools.chain.from_iterable(itertools.repeat(train_loader))
    val_loader = accelerator.prepare(val_loader)

    logger.info(f"[+] Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    model.eval()
    model.to(accelerator.device)
    logger.info("[+] Loaded model successfully")

    local_population_size = args.population_size // accelerator.num_processes
    population_seed_gen = np.random.RandomState(args.seed + accelerator.process_index)
    logger.info("Starting training...")
    for iteration, iteration_batch in enumerate(infinite_loader, start=1):
        if iteration > args.num_iterations:
            break

        local_seeds = population_seed_gen.randint(0, 1000000, local_population_size)
        rewards, metrics = single_iteration(
            model,
            tokenizer,
            iteration_batch,
            accelerator,
            local_seeds,
            args.sigma,
            args.batch_size,
            args.max_new_tokens,
            args.reward_type,
            args.meta_type,
        )

        local_seeds_tensor = torch.tensor(local_seeds, dtype=torch.long, device=accelerator.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
        all_seeds = accelerator.gather(local_seeds_tensor)
        all_rewards = accelerator.gather(rewards_tensor)
        all_metrics = {"train/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
        all_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_metrics.items()}
        if accelerator.is_main_process:
            avg_reward = all_rewards.mean().item()
            all_metrics["train/rewards"] = avg_reward
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()])
            logger.info(f"[+] Iteration {iteration:03d} {metric_str}")
            if run is not None:
                run.log(all_metrics, step=iteration)

            if iteration % args.model_save_interval == 0 and checkpoint_dir is not None:
                save_path = os.path.join(checkpoint_dir, f"{basename}_iter{iteration:03d}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"[+] Model saved to {save_path}")
        normalized_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        apply_evolution(
            model,
            all_seeds,
            absolute_scale=args.alpha,
            relative_scales=normalized_rewards,
        )

        if iteration % args.evaluate_interval == 0:
            metrics = evaluate_model(
                model, tokenizer, val_loader, args.max_new_tokens, args.reward_type, args.meta_type
            )
            all_val_metrics = {"val/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
            all_val_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_val_metrics.items()}
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_val_metrics.items()])
            logger.info(f"[+] Validation Iteration {iteration:03d} {metric_str}")
            if run is not None and accelerator.is_main_process:
                run.log(all_val_metrics, step=iteration)

    test_data = load_trivia_qa_meta(split="test", num_samples=100)
    test_dataset = ESDataset(
        test_data,
        tokenizer,
        max_length=args.max_input_length,
        meta_prompt=meta_prompt,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ESDataset.pad_collate_fn,
    )
    test_loader = accelerator.prepare(test_loader)
    test_metrics = evaluate_model(model, tokenizer, test_loader, args.max_new_tokens, args.reward_type, args.meta_type)
    all_test_metrics = {"test/" + k: accelerator.gather(v) for k, v in sorted(test_metrics.items())}
    all_test_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_test_metrics.items()}
    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_test_metrics.items()])
    logger.info(f"[+] Test {metric_str}")
    if run is not None and accelerator.is_main_process:
        run.log(all_test_metrics, step=args.num_iterations)


if __name__ == "__main__":
    main(parser.parse_args())
