"""Text helpers shared by the parsers and scorers."""

import re
import string


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    """SQuAD/TriviaQA answer normalisation (a tested copy of esma.utils.normalize_answer)."""
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def first_span(reply: str) -> str:
    """The first non-empty line of a reply: the model's committed answer, before any continuation."""
    for line in reply.splitlines():
        if line.strip():
            return line.strip()
    return ""


def straighten(text: str) -> str:
    """Curly quotes to ASCII, whitespace collapsed."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


_UNICODE_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2026": "...", "\u00a0": " "})


def unify_punctuation(text: str) -> str:
    """Curly quotes and apostrophes to ASCII, the ellipsis to three dots, and a dash used as punctuation (an em dash, or an en dash with
    spaces around it) to a clause separator, so that the answer scorer's word lists and alias matching see the same characters whichever
    keyboard or model produced the reply."""
    text = text.translate(_UNICODE_PUNCT)
    return re.sub(r"\s*[\u2014\u2015]\s*|(?<=\s)\u2013(?=\s)", " ; ", text)
