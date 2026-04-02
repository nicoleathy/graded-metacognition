import os

from datasets import Dataset, load_dataset


def load_nq_open(split: str = "validation", num_samples: int | None = None) -> Dataset:
    """Load NQ-Open dataset from HuggingFace.

    https://huggingface.co/datasets/google-research-datasets/nq_open
    NQ-Open is an open domain question answering benchmark derived from Natural Questions.
    The goal is to predict an English answer string for an input English question.
    All questions can be answered using the contents of English Wikipedia.

    Args:
        split: Split to load (validation, train)
        num_samples: Number of samples to load

    Returns:
        Dataset: NQ-Open dataset
            fields:
                - question: Question (string)
                - answer: List of possible answers (list of strings)
    """
    dataset = load_dataset(
        "google-research-datasets/nq_open",
        split=split,
        revision="5dd9790a83002ad084ddeb7c420dc716852c6f28",
    )
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_nq_open_meta(
    split: str = "validation", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    """Load NQ-Open dataset formatted for RL training.

    Args:
        split: Split to load (validation, train)
        num_samples: Number of samples to load
        num_proc: Number of processes for mapping

    Returns:
        Dataset: NQ-Open dataset formatted for RL
            fields:
                - question_id: Question ID (string)
                - question: Question (string)
                - answers: List of answer strings (list of strings)
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_nq_open(split, num_samples)
    return dataset.map(
        lambda x, idx: {
            "question_id": str(idx),
            "question": x["question"],
            "answers": x["answer"],
        },
        num_proc=num_proc,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
