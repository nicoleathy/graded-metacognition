"""Tests for esma.dataset."""

from typing import Any

import torch

from esma.dataset import ESDataset, SFTDataset, SFTMetaDataset
from esma.prompt import META_QA_PROMPT


def _make_fake_dataset(n: int = 3) -> list[dict[str, Any]]:
    """Dataset with question_id, question, answers."""
    return [{"question_id": f"q{i}", "question": f"What is {i}?", "answers": [str(i), f"answer_{i}"]} for i in range(n)]


class _FakeTokenizer:
    """Minimal tokenizer-like object for ESDataset (no chat_template)."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
        if isinstance(text, list):
            text = text[0] if len(text) == 1 else " ".join(text)
        # Simulate tokenization: deterministic from input for reproducible tests
        length = min(len(text.split()) + 2, max_length)
        gen = torch.Generator().manual_seed(hash(text) % (2**32))
        return {
            "input_ids": torch.randint(10, 100, (1, length), generator=gen).squeeze(0),
            "attention_mask": torch.ones(length, dtype=torch.long),
        }


class _FakeTokenizerWithChat(_FakeTokenizer):
    """Tokenizer with chat_template for SFTDataset / SFTMetaDataset."""

    chat_template = "dummy"

    def apply_chat_template(self, examples, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if tokenize:
            raise NotImplementedError
        parts = []
        for ex in examples:
            parts.append(ex.get("content", ""))
        return " ".join(parts) if len(parts) == 1 else parts


class TestESDataset:
    """Tests for ESDataset."""

    def test_len(self):
        data = _make_fake_dataset(5)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64)
        assert len(ds) == 5

    def test_getitem_keys_without_meta(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=None)
        out = ds[0]
        assert "question_id" in out
        assert "question" in out
        assert "input_ids" in out
        assert "attention_mask" in out
        assert "answers" in out
        assert "meta_input_ids" not in out
        assert "meta_attention_mask" not in out

    def test_getitem_keys_with_meta(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=META_QA_PROMPT)
        out = ds[0]
        assert "meta_input_ids" in out
        assert "meta_attention_mask" in out
        assert out["question_id"] == "q0"
        assert out["question"] == "What is 0?"
        assert out["answers"] == ["0", "answer_0"]

    def test_getitem_tensor_shapes(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64)
        out = ds[0]
        assert out["input_ids"].dim() == 1
        assert out["attention_mask"].dim() == 1
        assert out["input_ids"].size(0) == out["attention_mask"].size(0)

    def test_simple_collate_fn_no_meta(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=None)
        batch = [ds[0], ds[1]]
        batched = ESDataset.simple_collate_fn(batch)
        assert len(batched["question_id"]) == 2
        assert len(batched["input_ids"]) == 2
        assert len(batched["question"]) == 2
        assert "meta_input_ids" not in batched

    def test_simple_collate_fn_with_meta(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=META_QA_PROMPT)
        batch = [ds[0], ds[1]]
        batched = ESDataset.simple_collate_fn(batch)
        assert "meta_input_ids" in batched
        assert "meta_attention_mask" in batched

    def test_pad_collate_fn_no_meta(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=None)
        batch = [ds[0], ds[1]]
        batched = ESDataset.pad_collate_fn(batch)
        assert batched["input_ids"].dim() == 2
        assert batched["attention_mask"].dim() == 2
        assert batched["input_ids"].size(0) == 2
        assert "meta_input_ids" not in batched

    def test_pad_collate_fn_with_meta(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizer()
        ds = ESDataset(data, tok, max_length=64, meta_prompt=META_QA_PROMPT)
        batch = [ds[0], ds[1]]
        batched = ESDataset.pad_collate_fn(batch)
        assert batched["meta_input_ids"].dim() == 2
        assert batched["meta_attention_mask"].dim() == 2
        assert batched["meta_input_ids"].size(0) == 2


class TestSFTDataset:
    """Tests for SFTDataset."""

    def test_len(self):
        data = _make_fake_dataset(4)
        tok = _FakeTokenizerWithChat()
        ds = SFTDataset(data, tok, max_length=64)
        assert len(ds) == 4

    def test_getitem_keys(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizerWithChat()
        ds = SFTDataset(data, tok, max_length=64)
        out = ds[0]
        assert "input_ids" in out
        assert "attention_mask" in out
        assert out["input_ids"].dim() == 1

    def test_same_seed_same_output(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizerWithChat()
        ds1 = SFTDataset(data, tok, max_length=64, seed=42)
        ds2 = SFTDataset(data, tok, max_length=64, seed=42)
        out1 = ds1[0]
        out2 = ds2[0]
        assert torch.equal(out1["input_ids"], out2["input_ids"])

    def test_sft_collate_fn(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizerWithChat()
        ds = SFTDataset(data, tok, max_length=64)
        batch = [ds[0], ds[1]]
        batched = ds.sft_collate_fn(batch)
        assert "input_ids" in batched
        assert "attention_mask" in batched
        assert "labels" in batched
        assert batched["input_ids"].dim() == 2
        assert batched["labels"].dim() == 2
        # Labels are -100 at padding positions, non-negative elsewhere
        assert (batched["labels"] == -100).any() or (batched["labels"] >= 0).all()


class TestSFTMetaDataset:
    """Tests for SFTMetaDataset."""

    def test_len(self):
        data = _make_fake_dataset(3)
        tok = _FakeTokenizerWithChat()
        ds = SFTMetaDataset(data, tok, max_length=64)
        assert len(ds) == 3

    def test_getitem_keys(self):
        data = _make_fake_dataset(1)
        tok = _FakeTokenizerWithChat()
        ds = SFTMetaDataset(data, tok, max_length=64)
        out = ds[0]
        assert "direct_input_ids" in out
        assert "direct_attention_mask" in out
        assert "meta_no_input_ids" in out
        assert "meta_no_attention_mask" in out
        assert "meta_yes_input_ids" in out
        assert "meta_yes_attention_mask" in out
        assert "question" in out
        assert "answers" in out
        assert "_dataset_idx" in out

    def test_sft_meta_collate_fn(self):
        data = _make_fake_dataset(2)
        tok = _FakeTokenizerWithChat()
        ds = SFTMetaDataset(data, tok, max_length=64)
        batch = [ds[0], ds[1]]
        batched = ds.sft_meta_collate_fn(batch)
        assert batched["direct_input_ids"].dim() == 2
        assert batched["meta_input_ids"].dim() == 3  # [B, 2, L]
        assert batched["meta_input_ids"].size(0) == 2
        assert batched["meta_input_ids"].size(1) == 2  # No, Yes
        assert len(batched["question"]) == 2
        assert len(batched["answers"]) == 2
        assert len(batched["_dataset_idx"]) == 2
