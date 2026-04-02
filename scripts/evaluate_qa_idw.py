import argparse
import csv
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from esma.data import META_DATASETS, load_fictional_qa_meta, load_trivia_qa_meta
from esma.dataset import ESDataset
from esma.metric import IGNORE_VALUE, meta_metrics
from esma.prompt import DIRECT_QA_WITH_IDW_PROMPT
from esma.utils import get_logger, seed_everything

parser = argparse.ArgumentParser(description="Evaluate LLM on TriviaQA and save to TSV")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")
parser.add_argument(
    "--dataset", type=str, default="trivia_qa", choices=META_DATASETS.keys(), help="Dataset to evaluate"
)
parser.add_argument("--split", type=str, default="validation", help="Split to evaluate")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate (0 for all)")
parser.add_argument("--output-path", type=str, help="Output TSV file path")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum length of the input text")
parser.add_argument("--max-output-length", type=int, default=32, help="Maximum length of the output text")
parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")


def main(args):
    logger = get_logger(__name__)  # noqa: F821

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    logger.info(f"[+] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    if args.dataset == "trivia_qa":
        logger.info("[+] Loading TriviaQA dataset...")
        data = load_trivia_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_WITH_IDW_PROMPT
    elif args.dataset == "fictional_qa":
        logger.info("[+] Loading FictionalQA dataset...")
        data = load_fictional_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_WITH_IDW_PROMPT
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    dataset = ESDataset(data, tokenizer, max_length=args.max_input_length, prompt=prompt)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ESDataset.pad_collate_fn,
    )
    logger.info(f"[+] Total samples to evaluate: {len(data)}")

    if args.output_path is None:
        base_model = args.model.strip("/").split("/")[-1]
        os.makedirs("eval_outputs", exist_ok=True)
        args.output_path = f"eval_outputs/{args.dataset}_{base_model}_{args.split}_{args.num_samples}_idw.tsv"

    all_question_ids = []
    all_questions = []
    all_ground_truths = []
    all_predictions = []
    all_meta_answers = []
    all_direct_correctness = []
    all_yes = []
    all_yes_failures = []
    all_no_failures = []
    all_meta_alignments = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_output_length,
        )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        meta_decoded_outputs = ["no" if "i don't know" in output.lower() else "yes" for output in decoded_outputs]

        direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
            decoded_outputs, meta_decoded_outputs, batch["answers"], keep_length=True
        )

        all_question_ids.extend(batch["question_id"])
        all_questions.extend(batch["question"])
        all_ground_truths.extend(batch["answers"])
        all_predictions.extend(decoded_outputs)
        all_meta_answers.extend(meta_decoded_outputs)
        all_direct_correctness.extend(direct_correctness)
        all_yes.extend(yes)
        all_yes_failures.extend(yes_failures)
        all_no_failures.extend(no_failures)
        all_meta_alignments.extend(meta_alignments)

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
    logger.info(f"[+] Exact match accuracy: {sum(all_direct_correctness) / len(all_direct_correctness):.2%}")
    logger.info(f"[+] Yes rate: {sum(all_yes) / len(all_yes):.2%}")
    logger.info(f"[+] Meta alignments: {sum(all_meta_alignments) / len(all_meta_alignments):.2%}")

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


if __name__ == "__main__":
    main(parser.parse_args())
