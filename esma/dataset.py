import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .prompt import DIRECT_QA_PROMPT, META_QA_PROMPT


class ESDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt: str = DIRECT_QA_PROMPT,
        meta_prompt: str | None = None,
    ):
        """Initialize RL dataset.

        Args:
            dataset: TriviaQA dataset having fields:
                - question: Question (string)
                - answers: List of answer aliases (list of strings)
            tokenizer: Tokenizer to tokenize the questions and answers
            max_length: Maximum length of the input text
            prompt: Prompt to use for the questions
            meta_prompt: Prompt to use for the meta questions
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.meta_prompt = meta_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        examples = [
            {
                "role": "user",
                "content": self.prompt.format(question=item["question"]),
            }
        ]
        if self.tokenizer.chat_template is not None:
            examples = self.tokenizer.apply_chat_template(
                examples, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            examples = [example["content"] for example in examples]

        tokens = self.tokenizer(
            examples,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        example = {
            "question_id": item["question_id"],
            "question": item["question"],
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "answers": item["answers"],
        }
        if self.meta_prompt is not None:
            meta_text = self.meta_prompt.format(question=item["question"], answer="")
            if self.tokenizer.chat_template is not None:
                meta_examples = [{"role": "user", "content": meta_text}]
                meta_text = self.tokenizer.apply_chat_template(
                    meta_examples, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            meta_tokens = self.tokenizer(
                meta_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            example["meta_input_ids"] = meta_tokens["input_ids"].squeeze(0)
            example["meta_attention_mask"] = meta_tokens["attention_mask"].squeeze(0)
        return example

    @staticmethod
    def simple_collate_fn(batch: list[dict]) -> list[dict]:
        batched = {
            "question_id": [item["question_id"] for item in batch],
            "input_ids": [item["input_ids"] for item in batch],
            "question": [item["question"] for item in batch],
            "attention_mask": [item["attention_mask"] for item in batch],
            "answers": [item["answers"] for item in batch],
        }
        if "meta_input_ids" in batch[0]:
            batched["meta_input_ids"] = pad_sequence(
                [item["meta_input_ids"] for item in batch],
                batch_first=True,
                padding_side="left",
            )
            batched["meta_attention_mask"] = pad_sequence(
                [item["meta_attention_mask"] for item in batch],
                batch_first=True,
                padding_side="left",
            )
        return batched

    @staticmethod
    def pad_collate_fn(batch: list[dict]) -> dict:
        batched = {
            "question_id": [item["question_id"] for item in batch],
            "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_side="left"),
            "question": [item["question"] for item in batch],
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_side="left",
            ),
            "answers": [item["answers"] for item in batch],
        }
        if "meta_input_ids" in batch[0]:
            batched["meta_input_ids"] = pad_sequence(
                [item["meta_input_ids"] for item in batch],
                batch_first=True,
                padding_side="left",
            )
            batched["meta_attention_mask"] = pad_sequence(
                [item["meta_attention_mask"] for item in batch],
                batch_first=True,
                padding_side="left",
            )
        return batched


class GradedESDataset(ESDataset):
    """Dataset that supports graded metacognition prompts.

    For FOK/graded/numeric prompts: works identically to ESDataset since
    these prompts only need {question}.

    For JOL prompts: the meta question requires the model's answer, so
    meta tokenization must happen at inference time rather than dataset
    construction time. In this case, meta_input_ids is NOT included in
    the dataset items, and must be constructed dynamically after the
    direct question is answered.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt: str = DIRECT_QA_PROMPT,
        meta_prompt: str | None = None,
        meta_type: str = "graded",
    ):
        self.meta_type = meta_type
        # For JOL, don't pass meta_prompt to parent (needs answer at inference time)
        if meta_type == "jol":
            super().__init__(dataset, tokenizer, max_length, prompt, meta_prompt=None)
            self.jol_prompt = meta_prompt  # store for later use
        else:
            super().__init__(dataset, tokenizer, max_length, prompt, meta_prompt)
            self.jol_prompt = None

    def build_jol_meta_inputs(self, questions: list[str], answers: list[str]) -> dict:
        """Build JOL meta question inputs dynamically after getting model answers.

        Args:
            questions: List of questions
            answers: List of model's answers to the direct questions

        Returns:
            Dict with meta_input_ids and meta_attention_mask tensors
        """
        meta_texts = []
        for q, a in zip(questions, answers):
            text = self.jol_prompt.format(question=q, answer=a)
            if self.tokenizer.chat_template is not None:
                examples = [{"role": "user", "content": text}]
                text = self.tokenizer.apply_chat_template(
                    examples, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            meta_texts.append(text)

        tokens = self.tokenizer(
            meta_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            padding_side="left",
        )
        return {
            "meta_input_ids": tokens["input_ids"],
            "meta_attention_mask": tokens["attention_mask"],
        }


class SFTDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt: str = DIRECT_QA_PROMPT,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        examples = [
            {
                "role": "user",
                "content": self.prompt.format(question=item["question"]),
            },
            {
                "role": "assistant",
                "content": self.random.choice(item["answers"]),
            },
        ]
        examples = self.tokenizer.apply_chat_template(examples, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(examples, return_tensors="pt", truncation=True, max_length=self.max_length)
        example = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }
        return example

    def sft_collate_fn(self, batch: list[dict]) -> dict:
        """Collate function for SFT training with proper padding and labels."""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        max_len = max(ids.size(0) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - ids.size(0)
            # Left padding
            padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            padded_mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            # Labels: -100 for padding tokens (ignored in loss)
            label = padded_ids.clone()
            label[:pad_len] = -100

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(labels),
        }


class SFTMetaDataset(Dataset):
    """Dataset for SFT meta training that provides both direct and meta inputs.

    Provides:
    - Direct input_ids/attention_mask for inference
    - Meta input_ids/attention_mask/labels for "Yes" answer
    - Meta input_ids/attention_mask/labels for "No" answer
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        direct_prompt: str = DIRECT_QA_PROMPT,
        meta_prompt: str = META_QA_PROMPT,
    ):
        """Initialize SFT Meta dataset.

        Args:
            dataset: Dataset with fields:
                - question: Question string
                - answers: List of answer aliases (for correctness checking)
            tokenizer: Tokenizer to tokenize the questions and answers
            max_length: Maximum length of the input text
            direct_prompt: Prompt to use for direct questions (inference)
            meta_prompt: Prompt to use for meta questions (training)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direct_prompt = direct_prompt
        self.meta_prompt = meta_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Direct question for inference
        direct_examples = [{"role": "user", "content": self.direct_prompt.format(question=item["question"])}]
        direct_examples = self.tokenizer.apply_chat_template(
            direct_examples, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        direct_tokens = self.tokenizer(
            direct_examples, return_tensors="pt", truncation=True, max_length=self.max_length
        )

        # Meta questions with "No" and "Yes" answers
        meta_tokens = {}
        for answer in ["No", "Yes"]:
            meta_examples = [
                {"role": "user", "content": self.meta_prompt.format(question=item["question"])},
                {"role": "assistant", "content": answer},
            ]
            meta_examples = self.tokenizer.apply_chat_template(
                meta_examples, tokenize=False, add_generation_prompt=True
            )
            meta_tokens[answer.lower()] = self.tokenizer(
                meta_examples, return_tensors="pt", truncation=True, max_length=self.max_length
            )

        return {
            "_dataset_idx": idx,
            "question": item["question"],
            "answers": item["answers"],
            "direct_input_ids": direct_tokens["input_ids"].squeeze(0),
            "direct_attention_mask": direct_tokens["attention_mask"].squeeze(0),
            "meta_no_input_ids": meta_tokens["no"]["input_ids"].squeeze(0),
            "meta_no_attention_mask": meta_tokens["no"]["attention_mask"].squeeze(0),
            "meta_yes_input_ids": meta_tokens["yes"]["input_ids"].squeeze(0),
            "meta_yes_attention_mask": meta_tokens["yes"]["attention_mask"].squeeze(0),
        }

    def sft_meta_collate_fn(self, batch: list[dict]) -> dict:
        """Collate function that returns both direct and meta inputs for batch inference.

        Meta inputs are batched together with shape [B, 2, L] where:
        - B is batch size
        - 2 is for [No, Yes] (index 0 = No, index 1 = Yes)
        - L is sequence length
        """
        direct_input_ids = [item["direct_input_ids"] for item in batch]
        direct_attention_mask = [item["direct_attention_mask"] for item in batch]

        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        pad_token_id = int(pad_token_id)

        all_meta_lengths = []
        for item in batch:
            all_meta_lengths.append(item["meta_no_input_ids"].size(0))
            all_meta_lengths.append(item["meta_yes_input_ids"].size(0))
        max_meta_len = max(all_meta_lengths) if all_meta_lengths else 0

        meta_input_ids_list = []
        meta_attention_mask_list = []

        for item in batch:
            no_ids = item["meta_no_input_ids"]
            no_mask = item["meta_no_attention_mask"]
            no_pad_len = max_meta_len - no_ids.size(0)
            no_padded_ids = torch.cat([torch.full((no_pad_len,), pad_token_id, dtype=no_ids.dtype), no_ids])
            no_padded_mask = torch.cat([torch.zeros(no_pad_len, dtype=no_mask.dtype), no_mask])

            yes_ids = item["meta_yes_input_ids"]
            yes_mask = item["meta_yes_attention_mask"]
            yes_pad_len = max_meta_len - yes_ids.size(0)
            yes_padded_ids = torch.cat([torch.full((yes_pad_len,), pad_token_id, dtype=yes_ids.dtype), yes_ids])
            yes_padded_mask = torch.cat([torch.zeros(yes_pad_len, dtype=yes_mask.dtype), yes_mask])

            # Stack [No, Yes] -> [2, L]
            meta_input_ids_list.append(torch.stack([no_padded_ids, yes_padded_ids]))
            meta_attention_mask_list.append(torch.stack([no_padded_mask, yes_padded_mask]))

        # Stack all items -> [B, 2, L]
        meta_input_ids = torch.stack(meta_input_ids_list)
        meta_attention_mask = torch.stack(meta_attention_mask_list)

        # Pad direct inputs with left padding (consistent with meta inputs)
        max_direct_len = max(ids.size(0) for ids in direct_input_ids) if direct_input_ids else 0
        direct_padded_input_ids = []
        direct_padded_attention_mask = []
        for ids, mask in zip(direct_input_ids, direct_attention_mask):
            pad_len = max_direct_len - ids.size(0)
            padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            padded_mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            direct_padded_input_ids.append(padded_ids)
            direct_padded_attention_mask.append(padded_mask)
        direct_padded_input_ids = torch.stack(direct_padded_input_ids)
        direct_padded_attention_mask = torch.stack(direct_padded_attention_mask)

        questions = [item["question"] for item in batch]
        answers = [item["answers"] for item in batch]
        dataset_indices = [item["_dataset_idx"] for item in batch]

        return {
            "direct_input_ids": direct_padded_input_ids,
            "direct_attention_mask": direct_padded_attention_mask,
            "meta_input_ids": meta_input_ids,
            "meta_attention_mask": meta_attention_mask,
            "question": questions,
            "answers": answers,
            "_dataset_idx": dataset_indices,
        }
