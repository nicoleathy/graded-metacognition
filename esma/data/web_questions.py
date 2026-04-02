import os

from datasets import Dataset, load_dataset


def load_web_questions(split: str = "test", num_samples: int | None = None) -> Dataset:
    """Load WebQuestions dataset from HuggingFace.

    https://huggingface.co/datasets/stanfordnlp/web_questions
    WebQuestions is a question answering dataset containing questions from Google Suggest API
    and answers from Freebase. The dataset contains questions that can be answered using
    Freebase knowledge base.

    Args:
        split: Split to load (train, test)
        num_samples: Number of samples to load

    Returns:
        Dataset: WebQuestions dataset
            fields:
                - url: Freebase URL (string)
                - question: Question (string)
                - answers: List of answer strings (list of strings)
    """
    dataset = load_dataset(
        "stanfordnlp/web_questions",
        split=split,
        revision="0e473cbe21d1e91ec18da343644498be6a3f5454",
    )  # type: ignore
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))  # type: ignore
    return dataset  # type: ignore


def load_web_questions_meta(
    split: str = "test", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    """Load WebQuestions dataset formatted for RL training.

    Args:
        split: Split to load (train, test)
        num_samples: Number of samples to load
        num_proc: Number of processes for mapping

    Returns:
        Dataset: WebQuestions dataset formatted for RL
            fields:
                - question_id: Question ID (string, derived from URL or index)
                - question: Question (string)
                - answers: List of answer strings (list of strings)
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_web_questions(split, num_samples)
    return dataset.map(
        lambda x, idx: {
            "question_id": x.get("url", str(idx)),
            "question": x["question"],
            "answers": x["answers"],
        },
        num_proc=num_proc,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
