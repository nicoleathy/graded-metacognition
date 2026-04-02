"""Supervised Fine-Tuning (SFT) for Language Models.

This script implements standard SFT training using gradient-based optimization.
Supports distributed training via Accelerate and logging via Wandb.
"""

import argparse
import logging
import os

import torch
import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from esma.data import load_fictional_qa_meta, load_trivia_qa_meta
from esma.dataset import SFTMetaDataset
from esma.metric import correctness_by_inclusion
from esma.prompt import DIRECT_QA_PROMPT, META_QA_PROMPT
from esma.utils import get_logger, seed_everything

# fmt: off
parser = argparse.ArgumentParser(description="SFT Training for Language Models")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")

g = parser.add_argument_group("Data")
g.add_argument("--dataset", type=str, default="fictional_qa", choices=["trivia_qa", "fictional_qa"])
g.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
g.add_argument("--num-samples", type=int, help="Number of training samples to load")
g.add_argument("--num-val-samples", type=int, help="Number of validation samples to load")
g.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens for inference")

g = parser.add_argument_group("Training")
g.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
g.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
g.add_argument("--accumulation", type=int, default=1, help="Gradient accumulation steps")
g.add_argument("--learning-rate", "-lr", type=float, default=2e-5, help="Learning rate")
g.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
g.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
g.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")

g = parser.add_argument_group("Experiment Settings")
g.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
g.add_argument("--seed", type=int, default=42, help="Random seed")
g.add_argument("--output-dir", type=str, help="Output directory")
g.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
g.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
g.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
g.add_argument("--wandb-run-name", type=str, help="Wandb run name")
g.add_argument("--wandb-project", type=str, default="meta-cognition-sft", help="Wandb project")
g.add_argument("--wandb-entity", type=str, default="cosmoquester", help="Wandb entity")
# fmt: on


def run_batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: dict,
    max_new_tokens: int,
) -> list[int]:
    """Run inference on a batch of direct questions and return correctness.

    Args:
        model: Model to run inference with (should be unwrapped)
        tokenizer: Tokenizer for decoding
        batch: Batch dict with direct_input_ids, direct_attention_mask, answers
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of correctness values (1 for correct, 0 for incorrect)
    """
    model.eval()
    with torch.no_grad():
        input_ids = batch["direct_input_ids"].to(model.device)
        attention_mask = batch["direct_attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Get correctness
        correctness = correctness_by_inclusion(decoded_outputs, batch["answers"])
    model.train()
    return correctness


def select_meta_inputs(batch: dict, correctness: list[int], tokenizer: AutoTokenizer) -> dict:
    """Select meta inputs based on correctness and prepare for training.

    Args:
        batch: Batch dict with meta_input_ids [B, 2, L] and meta_attention_mask [B, 2, L]
            where index 0=No, 1=Yes
        correctness: List of correctness values (1 for Yes, 0 for No)
        tokenizer: Tokenizer for padding

    Returns:
        Dict with input_ids, attention_mask, labels for training
    """
    B = len(correctness)
    correctness_tensor = torch.tensor(correctness, dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)

    # Select using: meta_input_ids[range(B), correctness]
    # meta_input_ids is [B, 2, L], so we get [B, L]
    selected_input_ids = batch["meta_input_ids"][batch_indices, correctness_tensor]
    selected_attention_mask = batch["meta_attention_mask"][batch_indices, correctness_tensor]

    # Create labels from input_ids (for causal LM training)
    # Similar to SFTDataset: clone input_ids and mask padding tokens
    labels = selected_input_ids.clone()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Mask padding tokens (they're at the start due to left padding)
    labels[selected_input_ids == pad_token_id] = -100

    return {
        "input_ids": selected_input_ids,
        "attention_mask": selected_attention_mask,
        "labels": labels,
    }


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    accelerator: Accelerator,
    max_new_tokens: int,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    # Unwrap model for inference (needed for generate method)
    unwrapped_model = accelerator.unwrap_model(model)

    with torch.no_grad():
        for batch in val_loader:
            # Run inference on direct questions to get correctness
            correctness = run_batch_inference(unwrapped_model, tokenizer, batch, max_new_tokens)

            # Select meta inputs based on correctness
            training_batch = select_meta_inputs(batch, correctness, tokenizer)
            training_batch = {k: v.to(model.device) for k, v in training_batch.items()}

            outputs = model(
                input_ids=training_batch["input_ids"],
                attention_mask=training_batch["attention_mask"],
                labels=training_batch["labels"],
            )
            # Count non-padding tokens
            labels = training_batch["labels"]
            valid_mask = labels != -100
            num_tokens = valid_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

            # Token-level accuracy
            predictions = outputs.logits.argmax(dim=-1)
            # Shift predictions to align with labels (causal LM)
            shift_predictions = predictions[:, :-1]
            shift_labels = labels[:, 1:]
            shift_mask = shift_labels != -100
            correct = ((shift_predictions == shift_labels) & shift_mask).sum().item()
            total_correct += correct

    # Gather across processes
    total_loss_tensor = torch.tensor([total_loss], device=accelerator.device)
    total_tokens_tensor = torch.tensor([total_tokens], device=accelerator.device)
    total_correct_tensor = torch.tensor([total_correct], device=accelerator.device)

    gathered_loss = accelerator.gather(total_loss_tensor).sum().item()
    gathered_tokens = accelerator.gather(total_tokens_tensor).sum().item()
    gathered_correct = accelerator.gather(total_correct_tensor).sum().item()

    avg_loss = gathered_loss / gathered_tokens if gathered_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = gathered_correct / gathered_tokens if gathered_tokens > 0 else 0.0

    model.train()
    return {"val/loss": avg_loss, "val/perplexity": perplexity, "val/accuracy": accuracy}


def main(args):
    logger = get_logger(__name__)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation,
        log_with="wandb" if args.wandb_run_name else None,
    )

    if not accelerator.is_main_process:
        logger.setLevel(logging.CRITICAL)

    logger.info(f"[+] Accelerator device: {accelerator.device}")
    logger.info(f"[+] Num processes: {accelerator.num_processes}")
    logger.info(f"[+] Gradient accumulation steps: {args.accumulation}")

    # Setup output directory
    if args.output_dir is not None and args.wandb_run_name is None:
        args.wandb_run_name = os.path.basename(args.output_dir)
    if args.output_dir is None and args.wandb_run_name is not None:
        args.output_dir = os.path.join("outputs", args.wandb_run_name)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"[+] Output directory: {args.output_dir}")
    else:
        checkpoint_dir = None

    # Initialize Wandb
    if args.wandb_run_name is not None and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.output_dir,
            config=vars(args),
        )
    else:
        run = None

    # Set seed
    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    # Load tokenizer
    logger.info(f"[+] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load dataset
    logger.info(f"[+] Loading {args.dataset} dataset...")
    if args.dataset == "trivia_qa":
        train_data = load_trivia_qa_meta(split="train", num_samples=args.num_samples)
        val_data = load_trivia_qa_meta(split="validation", num_samples=args.num_val_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "fictional_qa":
        train_data = load_fictional_qa_meta(split="train", num_samples=args.num_samples)
        val_data = load_fictional_qa_meta(split="validation", num_samples=args.num_val_samples)
        prompt = DIRECT_QA_PROMPT
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"[+] Train samples: {len(train_data)}")
    logger.info(f"[+] Validation samples: {len(val_data)}")

    # Load model first for inference
    logger.info(f"[+] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
    )
    logger.info(f"[+] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create SFTMetaDataset that provides both direct and meta inputs
    logger.info("[+] Creating SFTMetaDataset...")

    train_meta_dataset = SFTMetaDataset(
        train_data,
        tokenizer,
        max_length=args.max_length,
        direct_prompt=prompt,
        meta_prompt=META_QA_PROMPT,
    )
    val_meta_dataset = SFTMetaDataset(
        val_data,
        tokenizer,
        max_length=args.max_length,
        direct_prompt=prompt,
        meta_prompt=META_QA_PROMPT,
    )

    # Create training data loaders
    train_loader = DataLoader(
        train_meta_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_meta_dataset.sft_meta_collate_fn,
    )
    val_loader = DataLoader(
        val_meta_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_meta_dataset.sft_meta_collate_fn,
    )

    # Set model to training mode
    model.train()

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // args.accumulation
    total_training_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    logger.info(f"[+] Total training steps: {total_training_steps}")
    logger.info(f"[+] Warmup steps: {warmup_steps}")

    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Unwrap model for inference (needed for generate method)
    unwrapped_model = accelerator.unwrap_model(model)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    logger.info("[+] Starting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"[+] Epoch {epoch}/{args.epochs}")

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        epoch_steps = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            # Run inference on direct questions to get correctness
            correctness = run_batch_inference(unwrapped_model, tokenizer, batch, args.max_new_tokens)

            # Select meta inputs based on correctness
            training_batch = select_meta_inputs(batch, correctness, tokenizer)
            training_batch = {k: v.to(model.device) for k, v in training_batch.items()}

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=training_batch["input_ids"],
                    attention_mask=training_batch["attention_mask"],
                    labels=training_batch["labels"],
                )
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

            # Token-level accuracy for training
            with torch.no_grad():
                labels = training_batch["labels"]
                predictions = outputs.logits.argmax(dim=-1)
                shift_predictions = predictions[:, :-1]
                shift_labels = labels[:, 1:]
                shift_mask = shift_labels != -100
                epoch_correct += ((shift_predictions == shift_labels) & shift_mask).sum().item()
                epoch_tokens += shift_mask.sum().item()

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    train_accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
                    current_lr = scheduler.get_last_lr()[0]

                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "acc": f"{train_accuracy:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )

                    if run is not None and accelerator.is_main_process:
                        run.log(
                            {
                                "train/loss": loss.item(),
                                "train/accuracy": train_accuracy,
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )

                # Evaluation
                if global_step % args.eval_steps == 0:
                    val_metrics = evaluate(model, tokenizer, val_loader, accelerator, args.max_new_tokens)

                    if accelerator.is_main_process:
                        logger.info(
                            f"[+] Step {global_step}: "
                            f"val_loss={val_metrics['val/loss']:.4f}, "
                            f"val_ppl={val_metrics['val/perplexity']:.2f}, "
                            f"val_acc={val_metrics['val/accuracy']:.4f}"
                        )

                        if run is not None:
                            run.log(val_metrics, step=global_step)

                        # Save best model
                        if val_metrics["val/loss"] < best_val_loss and checkpoint_dir is not None:
                            best_val_loss = val_metrics["val/loss"]
                            save_path = os.path.join(checkpoint_dir, "best")
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            logger.info(f"[+] Best model saved to {save_path}")

                # Save checkpoint
                if global_step % args.save_steps == 0 and checkpoint_dir is not None:
                    if accelerator.is_main_process:
                        save_path = os.path.join(checkpoint_dir, f"step_{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"[+] Checkpoint saved to {save_path}")

        # Save epoch model
        if checkpoint_dir is not None and accelerator.is_main_process:
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"[+] Epoch {epoch} model saved to {save_path}")

        # End of epoch evaluation
        val_metrics = evaluate(model, tokenizer, val_loader, accelerator, args.max_new_tokens)
        avg_epoch_loss = epoch_loss / epoch_steps

        train_accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
        if accelerator.is_main_process:
            logger.info(
                f"[+] Epoch {epoch} complete: "
                f"train_loss={avg_epoch_loss:.4f}, "
                f"train_acc={train_accuracy:.4f}, "
                f"val_loss={val_metrics['val/loss']:.4f}, "
                f"val_ppl={val_metrics['val/perplexity']:.2f}, "
                f"val_acc={val_metrics['val/accuracy']:.4f}"
            )

            if run is not None:
                run.log(
                    {
                        "epoch/train_loss": avg_epoch_loss,
                        "epoch/train_accuracy": train_accuracy,
                        **val_metrics,
                    },
                    step=global_step,
                )

    # Save final model
    if checkpoint_dir is not None and accelerator.is_main_process:
        save_path = os.path.join(checkpoint_dir, "final")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"[+] Final model saved to {save_path}")

    # Final evaluation
    val_metrics = evaluate(model, tokenizer, val_loader, accelerator, args.max_new_tokens)
    if accelerator.is_main_process:
        logger.info(
            f"[+] Training complete! "
            f"Final val_loss={val_metrics['val/loss']:.4f}, "
            f"val_ppl={val_metrics['val/perplexity']:.2f}, "
            f"val_acc={val_metrics['val/accuracy']:.4f}"
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(parser.parse_args())
