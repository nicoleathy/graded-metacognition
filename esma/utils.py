import logging
import random
import re
import string
import sys

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Seed everything for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    return white_space_fix(remove_articles(remove_punc(s.lower())))
