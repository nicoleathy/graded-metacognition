import json
import os
from pathlib import Path

from datasets import Dataset


def load_freebase_qa(split: str = "test", num_samples: int | None = None) -> Dataset:
    """Load FreebaseQA-eval dataset.

    FreebaseQA is a question answering dataset containing questions and answers
    from Freebase knowledge base. This loader reads the FreebaseQA-eval.json file
    from resource/FreebaseQA-eval.json.

    Args:
        split: Split to load (only "test" is available for this dataset)
        num_samples: Number of samples to load

    Returns:
        Dataset: FreebaseQA dataset
            fields:
                - question_id: Question ID (string)
                - question: Question text (string, from 'RawQuestion' field)
                - processed_question: Processed question text (string, from 'ProcessedQuestion' field)
                - answers: List of answer strings (list of strings, collected from all parses)
                - parses: List of parse information (list of dictionaries)
    """
    if split != "test":
        raise ValueError(f"FreebaseQA-eval only has a 'test' split, got '{split}'")

    # Get the resource directory path
    current_dir = Path(__file__).parent
    resource_file = current_dir.parent / "resource" / "FreebaseQA-eval.json"

    # Load JSON file
    with open(resource_file, encoding="utf-8") as f:
        data = json.load(f)

    # Process questions into dataset format
    processed_data = []
    for q in data.get("Questions", []):
        # Collect all unique answers from all parses
        all_answers = []
        for parse in q.get("Parses", []):
            for answer_obj in parse.get("Answers", []):
                answer_names = answer_obj.get("AnswersName", [])
                for answer_name in answer_names:
                    if answer_name and answer_name not in all_answers:
                        all_answers.append(answer_name)

        processed_data.append(
            {
                "question_id": q.get("Question-ID", ""),
                "question": q.get("RawQuestion", ""),
                "processed_question": q.get("ProcessedQuestion", ""),
                "answers": all_answers if all_answers else [],
                "parses": q.get("Parses", []),
            }
        )

    dataset = Dataset.from_list(processed_data)

    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))

    return dataset


def load_freebase_qa_meta(split: str = "test", num_samples: int | None = None, num_proc: int | None = None) -> Dataset:
    """Load FreebaseQA dataset formatted for RL training.

    Args:
        split: Split to load (only "test" is available for this dataset)
        num_samples: Number of samples to load
        num_proc: Number of processes for mapping

    Returns:
        Dataset: FreebaseQA dataset formatted for RL
            fields:
                - question_id: Question ID (string)
                - question: Question text (string)
                - answers: List of answer strings (list of strings)
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_freebase_qa(split, num_samples)
    return dataset.map(
        lambda x: {
            "question_id": x["question_id"],
            "question": x["question"],
            "answers": x["answers"] if x["answers"] else [],
        },
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
