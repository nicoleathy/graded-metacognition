"""Tests for esma.utils."""

import logging
import random

import numpy as np
import torch

from esma.utils import (
    get_logger,
    normalize_answer,
    remove_articles,
    remove_punc,
    seed_everything,
    white_space_fix,
)


class TestSeedEverything:
    """Tests for seed_everything."""

    def test_seed_reproducibility(self):
        seed_everything(42)
        a = random.random()
        b = np.random.rand()
        c = torch.rand(1).item()

        seed_everything(42)
        assert random.random() == a
        assert np.random.rand() == b
        assert torch.rand(1).item() == c


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        logger = get_logger("test_utils")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_utils"

    def test_logger_has_handler(self):
        logger = get_logger("test_utils_handler")
        assert len(logger.handlers) >= 1

    def test_same_name_returns_same_logger(self):
        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        assert logger1 is logger2


class TestRemoveArticles:
    """Tests for remove_articles."""

    def test_removes_a(self):
        assert remove_articles("a cat") == "  cat"

    def test_removes_an(self):
        assert remove_articles("an apple") == "  apple"

    def test_removes_the(self):
        assert remove_articles("the dog") == "  dog"

    def test_removes_multiple(self):
        # Each article replaced by single space; original spaces preserved
        result = remove_articles("the a an word")
        assert result.strip() == "word"
        assert "the" not in result and "a" not in result and "an" not in result

    def test_whole_word_only(self):
        assert remove_articles("another") == "another"
        assert remove_articles("there") == "there"


class TestWhiteSpaceFix:
    """Tests for white_space_fix."""

    def test_collapses_spaces(self):
        assert white_space_fix("hello    world") == "hello world"

    def test_strips_leading_trailing(self):
        assert white_space_fix("  foo  ") == "foo"

    def test_single_space_between_words(self):
        assert white_space_fix("a  b   c") == "a b c"


class TestRemovePunc:
    """Tests for remove_punc."""

    def test_removes_punctuation(self):
        assert remove_punc("hello!") == "hello"
        assert remove_punc("what?") == "what"

    def test_keeps_letters_and_numbers(self):
        assert remove_punc("a1b2") == "a1b2"

    def test_removes_all_punctuation(self):
        assert remove_punc("h.e,l!l?o") == "hello"


class TestNormalizeAnswer:
    """Tests for normalize_answer."""

    def test_lowercases(self):
        assert normalize_answer("Paris") == "paris"

    def test_removes_articles_and_punc_and_fixes_whitespace(self):
        assert normalize_answer("  The  Paris.  ") == "paris"

    def test_normalize_typical_answer(self):
        assert normalize_answer("The Eiffel Tower!") == "eiffel tower"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_punctuation_and_articles(self):
        assert normalize_answer("a the . , !") == ""
