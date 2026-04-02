import os

from datasets import Dataset, load_dataset


def load_mkqa(split: str = "train", num_samples: int | None = None) -> Dataset:
    """Load MKQA (Multilingual Knowledge Questions & Answers) dataset from HuggingFace.

    https://huggingface.co/datasets/apple/mkqa
    MKQA contains 10,000 queries sampled from the Google Natural Questions dataset.
    For each query we collect new passage-independent answers. These queries and answers
    are then human translated into 25 Non-English languages.

    Args:
        split: Split to load (only "train" is available for this dataset)
        num_samples: Number of samples to load

    Returns:
        Dataset: MKQA dataset
            fields:
                - example_id: Unique example ID (int)
                - query: Original English query (string)
                - queries: Dictionary of queries in all languages (dict)
                - answers: Dictionary of answers in all languages (dict)
    """
    if split != "train":
        raise ValueError(f"MKQA only has a 'train' split, got '{split}'")

    dataset = load_dataset(
        "apple/mkqa",
        split=split,
        revision="d7a2b9681ece319c53f8c2fe850eb4b487cec912",
    )  # type: ignore
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))  # type: ignore
    return dataset  # type: ignore


def load_mkqa_meta(
    split: str = "train", num_samples: int | None = None, num_proc: int | None = None, lang: str = "en"
) -> Dataset:
    """Load MKQA dataset formatted for RL training.

    Args:
        split: Split to load (only "train" is available for this dataset)
        num_samples: Number of samples to load
        num_proc: Number of processes for mapping
        lang: Language code to use (default: "en"). Supported languages include:
            ar, da, de, en, es, fi, fr, he, hu, it, ja, ko, km, ms, nl, no, pl, pt, ru,
            sv, th, tr, vi, zh_cn, zh_hk, zh_tw

    Returns:
        Dataset: MKQA dataset formatted for RL
            fields:
                - question_id: Question ID (string, from example_id)
                - question: Question text (string, in specified language)
                - answers: List of answer strings (list of strings, extracted from answers in specified language)
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_mkqa(split, num_samples)

    def extract_answers(example):
        """Extract answer texts from answers in specified language."""
        lang_answers = example.get("answers", {}).get(lang, [])
        answer_texts = []
        for answer_obj in lang_answers:
            if isinstance(answer_obj, dict):
                # Extract text field from answer object
                text = answer_obj.get("text", "")
                if text:
                    answer_texts.append(text)
                # Also include aliases if available
                aliases = answer_obj.get("aliases", [])
                for alias in aliases:
                    if alias and alias not in answer_texts:
                        answer_texts.append(alias)
        # Get question in specified language, fallback to English query if not available
        queries = example.get("queries", {})
        question = queries.get(lang, example.get("query", ""))
        return {
            "question_id": str(example.get("example_id", "")),
            "question": question,
            "answers": answer_texts if answer_texts else [],
        }

    return dataset.map(
        extract_answers,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
