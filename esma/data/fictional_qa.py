import os
import random

from datasets import Dataset, load_dataset


def load_fictional_qa() -> Dataset:
    """Load FictionalQA dataset.

    Returns:
        Dataset: FictionalQA dataset
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
    return load_dataset(
        "tomg-group-umd/fictionalqa",
        "fict_qa",
        split="train",
        revision="e8100f525e9ae97bb9db9e220339ee513114c01e",
    ).filter(lambda x: x["duplicate_relationship"] is None)


def load_fictional_qa_meta(
    split: str = "train", num_samples: int | None = None, num_proc: int | None = None, seed: int = 42
) -> Dataset:
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_fictional_qa()

    event_ids = sorted(set(dataset["event_id"]))
    rand = random.Random(seed)
    rand.shuffle(event_ids)

    if split == "train":
        event_ids = event_ids[: len(event_ids) // 2]
    elif split == "validation":
        event_ids = event_ids[len(event_ids) // 2 :]
    elif split == "all":
        pass
    else:
        raise ValueError(f"Invalid split: {split}")

    dataset = dataset.filter(lambda x: x["event_id"] in event_ids)
    dataset = dataset.map(
        lambda x: {
            "question_id": x["question_id"],
            "question": x["question"],
            "answers": [x["natural_answer"]],
        },
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset
