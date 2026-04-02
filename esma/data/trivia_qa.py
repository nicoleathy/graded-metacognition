import os

from datasets import Dataset, load_dataset


def load_trivia_qa(split: str = "validation", num_samples: int | None = None) -> Dataset:
    """Load TriviaQA dataset.

    Args:
        split: Split to load (validation, test, train)
        num_samples: Number of samples to load

    Returns:
        Dataset: TriviaQA dataset
            fields:
                - question_id: Question ID (string)
                - question: Question (string)
                - answer: Dictionary with answers
                    - aliases: List of answer aliases (list of strings)
                    - normalized_aliases: Normalized answer (string)
                    - matched_wiki_entity_name: Matched wiki entity name (string)
                    - normalized_matched_wiki_entity_name: Normalized matched wiki entity name (string)
                    - normalized_value: Normalized value (string)
                    - value: Value (string)
                    - type: Type of the answer
    """
    dataset = load_dataset(
        "trivia_qa",
        "rc",
        split=split,
        revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
    )
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_trivia_qa_meta(
    split: str = "validation", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_trivia_qa(split, num_samples)
    return dataset.map(
        lambda x: {
            "question_id": x["question_id"],
            "question": x["question"],
            "answers": x["answer"]["aliases"],
        },
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
