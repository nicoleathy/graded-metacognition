import os

from datasets import Dataset, load_dataset

CHOICES = ["A", "B", "C", "D"]


def load_mmlu(split: str = "test", num_samples: int | None = None, subject: str = "all") -> Dataset:
    """Load MMLU dataset for multi-domain knowledge evaluation.

    Args:
        split: Split to load (test, validation, dev)
        num_samples: Number of samples to load
        subject: MMLU subject to filter by, or "all" for all subjects

    Returns:
        Dataset with multiple-choice questions
    """
    dataset = load_dataset("cais/mmlu", "all", split=split)
    if subject != "all":
        dataset = dataset.filter(lambda x: x["subject"] == subject)
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_mmlu_meta(
    split: str = "test",
    num_samples: int | None = None,
    subject: str = "all",
    num_proc: int | None = None,
) -> Dataset:
    """Load MMLU with standardized meta-evaluation format.

    Formats questions with multiple-choice options and extracts
    the correct answer letter.
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_mmlu(split, num_samples, subject)

    def process(x, idx):
        # Build formatted question with choices
        question = x["question"]
        for j, choice_text in enumerate(x["choices"]):
            question += f"\n{CHOICES[j]}) {choice_text}"

        correct_answer = CHOICES[x["answer"]]
        # Include both the letter and the full text as valid answers
        answers = [correct_answer, x["choices"][x["answer"]]]

        return {
            "question_id": f"mmlu_{idx}",
            "question": question,
            "answers": answers,
            "subject": x.get("subject", "unknown"),
        }

    return dataset.map(
        process,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
