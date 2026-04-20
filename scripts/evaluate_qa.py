import argparse
import csv
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from esma.data import (
    META_DATASETS,
    load_fictional_qa_meta,
    load_freebase_qa_meta,
    load_gsm8k_meta,
    load_mkqa_meta,
    load_mmlu_meta,
    load_nq_open_meta,
    load_trivia_qa_meta,
    load_web_questions_meta,
)
from esma.dataset import ESDataset
from esma.metric import (
    IGNORE_VALUE,
    correctness_by_inclusion,
    correctness_gsm8k,
    correctness_mmlu,
    graded_meta_metrics,
    meta_metrics,
    multi_threshold_d_prime,
    parse_graded_response,
    parse_numeric_response,
    type2_d_prime,
)
from esma.prompt import (
    DIRECT_QA_CN_PROMPT,
    DIRECT_QA_ES_PROMPT,
    DIRECT_QA_KO_PROMPT,
    DIRECT_QA_PROMPT,
    META_PROMPT_TYPES,
    META_QA_CN_PROMPT,
    META_QA_ES_PROMPT,
    META_QA_KO_PROMPT,
    META_QA_PROMPT,
)
from esma.utils import get_logger, seed_everything

parser = argparse.ArgumentParser(description="Evaluate LLM on TriviaQA and save to TSV")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")
parser.add_argument(
    "--dataset", type=str, default="trivia_qa", choices=META_DATASETS.keys(), help="Dataset to evaluate"
)
parser.add_argument("--lang", type=str, default="en", help="Language to evaluate for MKQA")
parser.add_argument("--split", type=str, default="validation", help="Split to evaluate")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate (0 for all)")
parser.add_argument("--output-path", type=str, help="Output TSV file path")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum length of the input text")
parser.add_argument("--max-output-length", type=int, default=32, help="Maximum length of the output text")
parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--meta-type",
    type=str,
    default="binary",
    choices=list(META_PROMPT_TYPES.keys()),
    help="Metacognition prompt type",
)


# Map dataset -> correctness function. TriviaQA, NQ-Open, etc. use substring
# inclusion (that's correct for short keyword answers). MMLU and GSM8K need
# strict matching because their formats make substring matching produce
# wildly inflated accuracy.
CORRECTNESS_FNS = {
    "mmlu": correctness_mmlu,
    "gsm8k": correctness_gsm8k,
}


def main(args):
    logger = get_logger(__name__)  # noqa: F821

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")
    logger.info(f"[+] Meta type: {args.meta_type}")

    logger.info(f"[+] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    # Select meta prompt based on meta_type and language
    if args.meta_type == "binary":
        # Use language-specific binary prompts for MKQA
        if args.dataset == "mkqa":
            if args.lang == "ko":
                meta_prompt = META_QA_KO_PROMPT
            elif args.lang.startswith("zh"):
                meta_prompt = META_QA_CN_PROMPT
            elif args.lang == "es":
                meta_prompt = META_QA_ES_PROMPT
            else:
                meta_prompt = META_QA_PROMPT
        else:
            meta_prompt = META_QA_PROMPT
    else:
        # Graded prompts (English only for now)
        meta_prompt = META_PROMPT_TYPES[args.meta_type]

    # Dataset loading
    if args.dataset == "trivia_qa":
        logger.info("[+] Loading TriviaQA dataset...")
        data = load_trivia_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "fictional_qa":
        logger.info("[+] Loading FictionalQA dataset...")
        data = load_fictional_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "nq_open":
        logger.info("[+] Loading NQ-Open dataset...")
        data = load_nq_open_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "web_questions":
        logger.info("[+] Loading WebQuestions dataset...")
        data = load_web_questions_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "freebase_qa":
        logger.info("[+] Loading FreebaseQA dataset...")
        data = load_freebase_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "gsm8k":
        logger.info("[+] Loading GSM8K dataset...")
        data = load_gsm8k_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "mmlu":
        logger.info("[+] Loading MMLU dataset...")
        data = load_mmlu_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "mkqa":
        logger.info("[+] Loading MKQA dataset...")
        data = load_mkqa_meta(split=args.split, num_samples=args.num_samples, lang=args.lang)
        if args.lang == "ko":
            prompt = DIRECT_QA_KO_PROMPT
        elif args.lang.startswith("zh"):
            prompt = DIRECT_QA_CN_PROMPT
        elif args.lang == "es":
            prompt = DIRECT_QA_ES_PROMPT
        else:
            prompt = DIRECT_QA_PROMPT
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    correctness_fn = CORRECTNESS_FNS.get(args.dataset, correctness_by_inclusion)
    logger.info(f"[+] Using correctness function: {correctness_fn.__name__}")

    dataset = ESDataset(data, tokenizer, max_length=args.max_input_length, prompt=prompt, meta_prompt=meta_prompt)
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
        suffix = f"_{args.meta_type}" if args.meta_type != "binary" else ""
        args.output_path = f"eval_outputs/{args.dataset}_{base_model}_{args.split}_{args.num_samples}{suffix}.tsv"

    all_question_ids = []
    all_questions = []
    all_ground_truths = []
    all_predictions = []
    all_meta_answers = []
    all_direct_correctness = []
    all_meta_values = []

    # Binary-only accumulators
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

        meta_input_ids = batch["meta_input_ids"].to(model.device)
        meta_attention_mask = batch["meta_attention_mask"].to(model.device)
        meta_outputs = model.generate(
            input_ids=meta_input_ids,
            attention_mask=meta_attention_mask,
            max_new_tokens=args.max_output_length,
        )
        meta_generated_tokens = meta_outputs[:, meta_input_ids.shape[1] :]
        meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

        all_question_ids.extend(batch["question_id"])
        all_questions.extend(batch["question"])
        all_ground_truths.extend(batch["answers"])
        all_predictions.extend(decoded_outputs)
        all_meta_answers.extend(meta_decoded_outputs)

        if args.meta_type == "binary":
            direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
                decoded_outputs, meta_decoded_outputs, batch["answers"],
                keep_length=True, lang=args.lang,
                correctness_fn=correctness_fn,
            )
            all_direct_correctness.extend(direct_correctness)
            all_yes.extend(yes)
            all_yes_failures.extend(yes_failures)
            all_no_failures.extend(no_failures)
            all_meta_alignments.extend(meta_alignments)
            all_meta_values.extend(yes)
        else:
            correctness = correctness_fn(decoded_outputs, batch["answers"])
            if args.meta_type == "graded":
                grades = parse_graded_response(meta_decoded_outputs)
            elif args.meta_type in ("fok", "jol"):
                grades = parse_numeric_response(meta_decoded_outputs, scale=5)
            elif args.meta_type == "numeric":
                grades = parse_numeric_response(meta_decoded_outputs, scale=10)
            else:
                grades = parse_graded_response(meta_decoded_outputs)

            all_direct_correctness.extend(correctness)
            all_meta_values.extend(grades)

    # Write output
    with open(args.output_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        if args.meta_type == "binary":
            writer.writerow([
                "question_id", "question", "ground_truths", "prediction",
                "meta_answer", "direct_correctness", "yes",
                "yes_failures", "no_failures", "meta_alignments",
            ])
            for row in zip(
                all_question_ids, all_questions, all_ground_truths,
                all_predictions, all_meta_answers, all_direct_correctness,
                all_yes, all_yes_failures, all_no_failures, all_meta_alignments,
            ):
                writer.writerow([str(v) for v in row])
        else:
            writer.writerow([
                "question_id", "question", "ground_truths", "prediction",
                "meta_answer", "direct_correctness", "meta_grade",
            ])
            for row in zip(
                all_question_ids, all_questions, all_ground_truths,
                all_predictions, all_meta_answers, all_direct_correctness,
                all_meta_values,
            ):
                writer.writerow([str(v) for v in row])

    logger.info(f"[+] Results saved to: {args.output_path}")
    logger.info(f"[+] Exact match accuracy: {sum(all_direct_correctness) / len(all_direct_correctness):.2%}")

    if args.meta_type == "binary":
        logger.info(f"[+] Yes rate: {sum(all_yes) / len(all_yes):.2%}")
        clean_yf = [v for v in all_yes_failures if v != IGNORE_VALUE]
        clean_nf = [v for v in all_no_failures if v != IGNORE_VALUE]
        if clean_yf:
            logger.info(f"[+] Yes failures rate: {sum(clean_yf) / len(clean_yf):.2%}")
        else:
            logger.info("[-] All meta answers are No")
        if clean_nf:
            logger.info(f"[+] No failures rate: {sum(clean_nf) / len(clean_nf):.2%}")
        else:
            logger.info("[-] All meta answers are Yes")
        logger.info(f"[+] Meta alignments: {sum(all_meta_alignments) / len(all_meta_alignments):.2%}")
        logger.info(f"[+] Type 2 d-prime: {type2_d_prime(all_direct_correctness, all_yes):.2f}")
    else:
        from esma.metric import metacognitive_resolution, graded_calibration_error, type2_auroc

        logger.info(f"[+] Mean grade: {sum(all_meta_values) / len(all_meta_values):.2f}")
        correct_grades = [g for c, g in zip(all_direct_correctness, all_meta_values) if c == 1]
        incorrect_grades = [g for c, g in zip(all_direct_correctness, all_meta_values) if c == 0]
        if correct_grades:
            logger.info(f"[+] Mean grade (correct): {sum(correct_grades) / len(correct_grades):.2f}")
        if incorrect_grades:
            logger.info(f"[+] Mean grade (incorrect): {sum(incorrect_grades) / len(incorrect_grades):.2f}")
        logger.info(f"[+] Grade separation: {(sum(correct_grades)/max(len(correct_grades),1)) - (sum(incorrect_grades)/max(len(incorrect_grades),1)):.2f}")  # noqa: E501
        # Both gamma (Goodman-Kruskal, paper-reported) and Kendall's tau-b (the
        # previous implementation). They can differ meaningfully when there are
        # many tied pairs, which is typical for 4-level A-D scales.
        logger.info(f"[+] Gamma (Goodman-Kruskal): {metacognitive_resolution(all_direct_correctness, all_meta_values, method='gamma'):.4f}")
        logger.info(f"[+] Kendall's tau-b:         {metacognitive_resolution(all_direct_correctness, all_meta_values, method='kendall'):.4f}")
        logger.info(f"[+] Calibration error: {graded_calibration_error(all_direct_correctness, all_meta_values):.4f}")
        logger.info(f"[+] Type 2 AUROC: {type2_auroc(all_direct_correctness, all_meta_values):.4f}")

        d_primes = multi_threshold_d_prime(
            all_direct_correctness, all_meta_values,
            max_grade=3 if args.meta_type == "graded" else 5 if args.meta_type in ("fok", "jol") else 10,
        )
        for k, v in d_primes.items():
            logger.info(f"[+] {k}: {v:.4f}")


if __name__ == "__main__":
    main(parser.parse_args())
import argparse
import csv
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from esma.data import (
    META_DATASETS,
    load_fictional_qa_meta,
    load_freebase_qa_meta,
    load_gsm8k_meta,
    load_mkqa_meta,
    load_mmlu_meta,
    load_nq_open_meta,
    load_trivia_qa_meta,
    load_web_questions_meta,
)
from esma.dataset import ESDataset
from esma.metric import (
    IGNORE_VALUE,
    graded_meta_metrics,
    meta_metrics,
    multi_threshold_d_prime,
    parse_graded_response,
    parse_numeric_response,
    type2_d_prime,
)
from esma.prompt import (
    DIRECT_QA_CN_PROMPT,
    DIRECT_QA_ES_PROMPT,
    DIRECT_QA_KO_PROMPT,
    DIRECT_QA_PROMPT,
    META_PROMPT_TYPES,
    META_QA_CN_PROMPT,
    META_QA_ES_PROMPT,
    META_QA_KO_PROMPT,
    META_QA_PROMPT,
)
from esma.utils import get_logger, seed_everything

parser = argparse.ArgumentParser(description="Evaluate LLM on TriviaQA and save to TSV")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")
parser.add_argument(
    "--dataset", type=str, default="trivia_qa", choices=META_DATASETS.keys(), help="Dataset to evaluate"
)
parser.add_argument("--lang", type=str, default="en", help="Language to evaluate for MKQA")
parser.add_argument("--split", type=str, default="validation", help="Split to evaluate")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate (0 for all)")
parser.add_argument("--output-path", type=str, help="Output TSV file path")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum length of the input text")
parser.add_argument("--max-output-length", type=int, default=32, help="Maximum length of the output text")
parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--meta-type",
    type=str,
    default="binary",
    choices=list(META_PROMPT_TYPES.keys()),
    help="Metacognition prompt type",
)


def main(args):
    logger = get_logger(__name__)  # noqa: F821

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")
    logger.info(f"[+] Meta type: {args.meta_type}")

    logger.info(f"[+] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    # Select meta prompt based on meta_type and language
    if args.meta_type == "binary":
        # Use language-specific binary prompts for MKQA
        if args.dataset == "mkqa":
            if args.lang == "ko":
                meta_prompt = META_QA_KO_PROMPT
            elif args.lang.startswith("zh"):
                meta_prompt = META_QA_CN_PROMPT
            elif args.lang == "es":
                meta_prompt = META_QA_ES_PROMPT
            else:
                meta_prompt = META_QA_PROMPT
        else:
            meta_prompt = META_QA_PROMPT
    else:
        # Graded prompts (English only for now)
        meta_prompt = META_PROMPT_TYPES[args.meta_type]

    # Dataset loading
    if args.dataset == "trivia_qa":
        logger.info("[+] Loading TriviaQA dataset...")
        data = load_trivia_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "fictional_qa":
        logger.info("[+] Loading FictionalQA dataset...")
        data = load_fictional_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "nq_open":
        logger.info("[+] Loading NQ-Open dataset...")
        data = load_nq_open_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "web_questions":
        logger.info("[+] Loading WebQuestions dataset...")
        data = load_web_questions_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "freebase_qa":
        logger.info("[+] Loading FreebaseQA dataset...")
        data = load_freebase_qa_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "gsm8k":
        logger.info("[+] Loading GSM8K dataset...")
        data = load_gsm8k_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "mmlu":
        logger.info("[+] Loading MMLU dataset...")
        data = load_mmlu_meta(split=args.split, num_samples=args.num_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "mkqa":
        logger.info("[+] Loading MKQA dataset...")
        data = load_mkqa_meta(split=args.split, num_samples=args.num_samples, lang=args.lang)
        if args.lang == "ko":
            prompt = DIRECT_QA_KO_PROMPT
        elif args.lang.startswith("zh"):
            prompt = DIRECT_QA_CN_PROMPT
        elif args.lang == "es":
            prompt = DIRECT_QA_ES_PROMPT
        else:
            prompt = DIRECT_QA_PROMPT
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    dataset = ESDataset(data, tokenizer, max_length=args.max_input_length, prompt=prompt, meta_prompt=meta_prompt)
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
        suffix = f"_{args.meta_type}" if args.meta_type != "binary" else ""
        args.output_path = f"eval_outputs/{args.dataset}_{base_model}_{args.split}_{args.num_samples}{suffix}.tsv"

    all_question_ids = []
    all_questions = []
    all_ground_truths = []
    all_predictions = []
    all_meta_answers = []
    all_direct_correctness = []
    all_meta_values = []

    # Binary-only accumulators
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

        meta_input_ids = batch["meta_input_ids"].to(model.device)
        meta_attention_mask = batch["meta_attention_mask"].to(model.device)
        meta_outputs = model.generate(
            input_ids=meta_input_ids,
            attention_mask=meta_attention_mask,
            max_new_tokens=args.max_output_length,
        )
        meta_generated_tokens = meta_outputs[:, meta_input_ids.shape[1] :]
        meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

        all_question_ids.extend(batch["question_id"])
        all_questions.extend(batch["question"])
        all_ground_truths.extend(batch["answers"])
        all_predictions.extend(decoded_outputs)
        all_meta_answers.extend(meta_decoded_outputs)

        if args.meta_type == "binary":
            direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
                decoded_outputs, meta_decoded_outputs, batch["answers"], keep_length=True, lang=args.lang
            )
            all_direct_correctness.extend(direct_correctness)
            all_yes.extend(yes)
            all_yes_failures.extend(yes_failures)
            all_no_failures.extend(no_failures)
            all_meta_alignments.extend(meta_alignments)
            all_meta_values.extend(yes)
        else:
            from esma.metric import correctness_by_inclusion

            correctness = correctness_by_inclusion(decoded_outputs, batch["answers"])
            if args.meta_type == "graded":
                grades = parse_graded_response(meta_decoded_outputs)
            elif args.meta_type in ("fok", "jol"):
                grades = parse_numeric_response(meta_decoded_outputs, scale=5)
            elif args.meta_type == "numeric":
                grades = parse_numeric_response(meta_decoded_outputs, scale=10)
            else:
                grades = parse_graded_response(meta_decoded_outputs)

            all_direct_correctness.extend(correctness)
            all_meta_values.extend(grades)

    # Write output
    with open(args.output_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        if args.meta_type == "binary":
            writer.writerow([
                "question_id", "question", "ground_truths", "prediction",
                "meta_answer", "direct_correctness", "yes",
                "yes_failures", "no_failures", "meta_alignments",
            ])
            for row in zip(
                all_question_ids, all_questions, all_ground_truths,
                all_predictions, all_meta_answers, all_direct_correctness,
                all_yes, all_yes_failures, all_no_failures, all_meta_alignments,
            ):
                writer.writerow([str(v) for v in row])
        else:
            writer.writerow([
                "question_id", "question", "ground_truths", "prediction",
                "meta_answer", "direct_correctness", "meta_grade",
            ])
            for row in zip(
                all_question_ids, all_questions, all_ground_truths,
                all_predictions, all_meta_answers, all_direct_correctness,
                all_meta_values,
            ):
                writer.writerow([str(v) for v in row])

    logger.info(f"[+] Results saved to: {args.output_path}")
    logger.info(f"[+] Exact match accuracy: {sum(all_direct_correctness) / len(all_direct_correctness):.2%}")

    if args.meta_type == "binary":
        logger.info(f"[+] Yes rate: {sum(all_yes) / len(all_yes):.2%}")
        clean_yf = [v for v in all_yes_failures if v != IGNORE_VALUE]
        clean_nf = [v for v in all_no_failures if v != IGNORE_VALUE]
        if clean_yf:
            logger.info(f"[+] Yes failures rate: {sum(clean_yf) / len(clean_yf):.2%}")
        else:
            logger.info("[-] All meta answers are No")
        if clean_nf:
            logger.info(f"[+] No failures rate: {sum(clean_nf) / len(clean_nf):.2%}")
        else:
            logger.info("[-] All meta answers are Yes")
        logger.info(f"[+] Meta alignments: {sum(all_meta_alignments) / len(all_meta_alignments):.2%}")
        logger.info(f"[+] Type 2 d-prime: {type2_d_prime(all_direct_correctness, all_yes):.2f}")
    else:
        from esma.metric import metacognitive_resolution, graded_calibration_error, type2_auroc

        logger.info(f"[+] Mean grade: {sum(all_meta_values) / len(all_meta_values):.2f}")
        correct_grades = [g for c, g in zip(all_direct_correctness, all_meta_values) if c == 1]
        incorrect_grades = [g for c, g in zip(all_direct_correctness, all_meta_values) if c == 0]
        if correct_grades:
            logger.info(f"[+] Mean grade (correct): {sum(correct_grades) / len(correct_grades):.2f}")
        if incorrect_grades:
            logger.info(f"[+] Mean grade (incorrect): {sum(incorrect_grades) / len(incorrect_grades):.2f}")
        logger.info(f"[+] Grade separation: {(sum(correct_grades)/max(len(correct_grades),1)) - (sum(incorrect_grades)/max(len(incorrect_grades),1)):.2f}")  # noqa: E501
        logger.info(f"[+] Gamma correlation: {metacognitive_resolution(all_direct_correctness, all_meta_values):.4f}")
        logger.info(f"[+] Calibration error: {graded_calibration_error(all_direct_correctness, all_meta_values):.4f}")
        logger.info(f"[+] Type 2 AUROC: {type2_auroc(all_direct_correctness, all_meta_values):.4f}")

        d_primes = multi_threshold_d_prime(
            all_direct_correctness, all_meta_values,
            max_grade=3 if args.meta_type == "graded" else 5 if args.meta_type in ("fok", "jol") else 10,
        )
        for k, v in d_primes.items():
            logger.info(f"[+] {k}: {v:.4f}")


if __name__ == "__main__":
    main(parser.parse_args())
