import os
import re

from datasets import Dataset, load_dataset


def load_gsm8k(split: str = "test", num_samples: int | None = None) -> Dataset:
    """Load GSM8K dataset for math reasoning evaluation.

    Args:
        split: Split to load (test, train)
        num_samples: Number of samples to load

    Returns:
        Dataset with question, answer, and full solution fields
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_gsm8k_meta(
    split: str = "test", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    """Load GSM8K with standardized meta-evaluation format.

    Extracts the final numeric answer from the '#### N' format used in
    GSM8K solutions. Downstream correctness scoring SHOULD use
    `correctness_gsm8k`, which compares the last numeric token in the
    model output against the gold number (with comma-normalization).

    Substring inclusion is *not* appropriate for GSM8K: with it, "420"
    counts as correct for gold "42", which silently corrupts correctness
    labels and any metacognitive metric computed from them.
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_gsm8k(split, num_samples)

    def process(x, idx):
        answer_match = re.search(r"####\s*(.+)", x["answer"])
        answer = answer_match.group(1).strip() if answer_match else ""
        # Normalize: keep the number only, strip commas
        answer_no_comma = answer.replace(",", "")
        answers = []
        if answer:
            answers.append(answer)
        if answer_no_comma and answer_no_comma != answer:
            answers.append(answer_no_comma)
        return {
            "question_id": f"gsm8k_{idx}",
            "question": x["question"],
            "answers": answers,
        }

    # Drop any rows whose answer couldn't be parsed; they are not useful
    # for metacognition evaluation and will corrupt accuracy statistics.
    processed = dataset.map(
        process,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    processed = processed.filter(lambda r: len(r["answers"]) > 0)
    return processed
