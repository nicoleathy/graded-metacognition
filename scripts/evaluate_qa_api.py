import argparse
import csv
import os
import sys
from multiprocessing import Pool

from openai import OpenAI
from tqdm import tqdm

from esma.data import META_DATASETS, load_fictional_qa_meta, load_trivia_qa_meta
from esma.metric import IGNORE_VALUE, meta_metrics, type2_d_prime
from esma.prompt import DIRECT_QA_PROMPT, META_QA_PROMPT
from esma.utils import get_logger, seed_everything

# Format: (input_price_per_1M, output_price_per_1M)
OPENAI_PRICING = {
    "gemini-3-flash-preview": (0.50, 3.00),
    "gpt-5.2-2025-12-11": (1.75, 14.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
}

parser = argparse.ArgumentParser(description="Evaluate OpenAI API on QA datasets and save to TSV")
parser.add_argument(
    "--model", type=str, default="gpt-5-nano-2025-08-07", choices=OPENAI_PRICING.keys(), help="OpenAI model ID"
)
parser.add_argument(
    "--dataset", type=str, default="trivia_qa", choices=META_DATASETS.keys(), help="Dataset to evaluate"
)
parser.add_argument("--start", type=int, help="Start index to evaluate")
parser.add_argument("--split", type=str, default="validation", help="Split to evaluate")
parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate (0 for all)")
parser.add_argument("--output-path", type=str, help="Output TSV file path")
parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens for generation")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num-parallels", type=int, default=1, help="Number of parallel processes for API calls")


def call_openai_api(client: OpenAI, model: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """Call OpenAI API with a single prompt. Returns (content, usage_dict)."""
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="none",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        content = ""
        print(f"Error calling OpenAI API for prompt: {prompt}", file=sys.stderr)
        print(response, file=sys.stderr)
    else:
        content = content.strip()

    # Extract usage information
    usage = response.usage
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }

    return content, usage_dict


def process_item(args_tuple):
    """Process a single item for multiprocessing - only makes API calls."""
    item, model, prompt_template, meta_prompt_template, max_tokens = args_tuple

    # Create a new client for each process
    client = OpenAI()

    # Prepare prompts
    direct_prompt = prompt_template.format(question=item["question"])
    meta_prompt = meta_prompt_template.format(question=item["question"])

    # Call OpenAI API for direct answer
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    try:
        decoded_output, usage = call_openai_api(client, model, direct_prompt, max_tokens)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]
    except Exception as e:
        decoded_output = ""
        print(f"[-] Error calling OpenAI API for direct prompt: {e}", file=sys.stderr)
        print(direct_prompt, file=sys.stderr)

    # Call OpenAI API for meta answer
    try:
        meta_decoded_output, usage = call_openai_api(client, model, meta_prompt, max_tokens)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]
    except Exception as e:
        meta_decoded_output = ""
        print(f"[-] Error calling OpenAI API for meta prompt: {e}", file=sys.stderr)
        print(meta_prompt, file=sys.stderr)

    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "ground_truths": item["answers"],
        "prediction": decoded_output,
        "meta_answer": meta_decoded_output,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


def main(args):
    logger = get_logger(__name__)

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    # Initialize OpenAI client
    client = OpenAI()
    logger.info(f"[+] Using OpenAI model: {args.model}")

    # Load dataset
    if args.dataset == "trivia_qa":
        logger.info("[+] Loading TriviaQA dataset...")
        data = load_trivia_qa_meta(split=args.split, num_samples=args.num_samples)
        if args.start is not None:
            data = data.select(range(args.start, len(data)))
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "fictional_qa":
        logger.info("[+] Loading FictionalQA dataset...")
        data = load_fictional_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info(f"[+] Total samples to evaluate: {len(data)}")
    logger.info(f"[+] Using {args.num_parallels} parallel processes")

    if args.output_path is None:
        base_model = args.model.replace("/", "_")
        os.makedirs("eval_outputs", exist_ok=True)
        args.output_path = f"eval_outputs/{args.dataset}_{base_model}_{args.split}_{args.num_samples}_{args.start}.tsv"

    # Prepare arguments for multiprocessing
    process_args = [(item, args.model, prompt, META_QA_PROMPT, args.max_tokens) for item in data]

    # Process items in parallel or sequentially - only API calls
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    if args.num_parallels > 1:
        with Pool(processes=args.num_parallels) as pool:
            api_results = list(tqdm(pool.imap(process_item, process_args), total=len(process_args), desc="Evaluating"))
    else:
        # Sequential processing (original behavior)
        api_results = []
        for item in tqdm(data, desc="Evaluating"):
            # Prepare prompts for direct and meta questions
            direct_prompt = prompt.format(question=item["question"])
            meta_prompt = META_QA_PROMPT.format(question=item["question"])

            item_prompt_tokens = 0
            item_completion_tokens = 0
            item_total_tokens = 0

            # Call OpenAI API for direct answer
            try:
                decoded_output, usage = call_openai_api(client, args.model, direct_prompt, args.max_tokens)
                item_prompt_tokens += usage["prompt_tokens"]
                item_completion_tokens += usage["completion_tokens"]
                item_total_tokens += usage["total_tokens"]
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
            except Exception as e:
                logger.warning(f"[-] Error calling API for direct prompt: {e}")
                decoded_output = ""

            # Call OpenAI API for meta answer
            try:
                meta_decoded_output, usage = call_openai_api(client, args.model, meta_prompt, args.max_tokens)
                item_prompt_tokens += usage["prompt_tokens"]
                item_completion_tokens += usage["completion_tokens"]
                item_total_tokens += usage["total_tokens"]
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
            except Exception as e:
                logger.warning(f"Error calling API for meta prompt: {e}")
                meta_decoded_output = ""

            api_results.append(
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "ground_truths": item["answers"],
                    "prediction": decoded_output,
                    "meta_answer": meta_decoded_output,
                    "prompt_tokens": item_prompt_tokens,
                    "completion_tokens": item_completion_tokens,
                    "total_tokens": item_total_tokens,
                }
            )

    # Aggregate token counts from parallel processing
    if args.num_parallels > 1:
        for result in api_results:
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)
            total_tokens += result.get("total_tokens", 0)

    # Extract API results
    all_question_ids = [r["question_id"] for r in api_results]
    all_questions = [r["question"] for r in api_results]
    all_ground_truths = [r["ground_truths"] for r in api_results]
    all_predictions = [r["prediction"] for r in api_results]
    all_meta_answers = [r["meta_answer"] for r in api_results]

    # Compute metrics in main process
    all_direct_correctness, all_yes, all_yes_failures, all_no_failures, all_meta_alignments = meta_metrics(
        all_predictions, all_meta_answers, all_ground_truths, keep_length=True
    )

    # Write results to TSV
    with open(args.output_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "question_id",
                "question",
                "ground_truths",
                "prediction",
                "meta_answer",
                "direct_correctness",
                "yes",
                "yes_failures",
                "no_failures",
                "meta_alignments",
            ]
        )
        for (
            question_id,
            question,
            ground_truths,
            prediction,
            meta_answer,
            direct_correctness,
            yes,
            yes_failures,
            no_failures,
            meta_alignments,
        ) in zip(
            all_question_ids,
            all_questions,
            all_ground_truths,
            all_predictions,
            all_meta_answers,
            all_direct_correctness,
            all_yes,
            all_yes_failures,
            all_no_failures,
            all_meta_alignments,
        ):
            writer.writerow(
                [
                    question_id,
                    question,
                    str(ground_truths),
                    prediction,
                    meta_answer,
                    direct_correctness,
                    yes,
                    yes_failures,
                    no_failures,
                    meta_alignments,
                ]
            )
    logger.info(f"[+] Results saved to: {args.output_path}")

    # Calculate and display token usage and cost
    logger.info("[+] Token usage:")
    logger.info(f"    Input tokens:  {total_prompt_tokens:,}")
    logger.info(f"    Output tokens: {total_completion_tokens:,}")
    logger.info(f"    Total tokens:  {total_tokens:,}")

    # Calculate cost
    model_key = args.model
    # Try to find matching pricing (handle model variants)
    pricing = None
    for key in OPENAI_PRICING:
        if key in model_key or model_key in key:
            pricing = OPENAI_PRICING[key]
            break

    if pricing:
        input_price_per_1M, output_price_per_1M = pricing
        input_cost = (total_prompt_tokens / 1_000_000) * input_price_per_1M
        output_cost = (total_completion_tokens / 1_000_000) * output_price_per_1M
        total_cost = input_cost + output_cost
        logger.info("[+] Estimated cost:")
        logger.info(f"    Input cost:  ${input_cost:.4f}")
        logger.info(f"    Output cost: ${output_cost:.4f}")
        logger.info(f"    Total cost:  ${total_cost:.4f}")
    else:
        logger.warning(f"[!] Pricing not available for model '{args.model}'. Please check OpenAI pricing page.")

    logger.info(f"[+] Exact match accuracy: {sum(all_direct_correctness) / len(all_direct_correctness):.2%}")
    logger.info(f"[+] Yes rate: {sum(all_yes) / len(all_yes):.2%}")

    all_yes_failures = [v for v in all_yes_failures if v != IGNORE_VALUE]
    all_no_failures = [v for v in all_no_failures if v != IGNORE_VALUE]
    if len(all_yes_failures) > 0:
        logger.info(f"[+] Yes failures rate: {sum(all_yes_failures) / len(all_yes_failures):.2%}")
    else:
        logger.info("[-] All meta answers are No")
    if len(all_no_failures) > 0:
        logger.info(f"[+] No failures rate: {sum(all_no_failures) / len(all_no_failures):.2%}")
    else:
        logger.info("[-] All meta answers are Yes")
    logger.info(f"[+] Meta alignments: {sum(all_meta_alignments) / len(all_meta_alignments):.2%}")
    logger.info(f"[+] Type 2 d-prime: {type2_d_prime(all_direct_correctness, all_yes):.2f}")


if __name__ == "__main__":
    main(parser.parse_args())
