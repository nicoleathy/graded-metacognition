"""Prompt builders that keep question content separate from response-format instructions.

The legacy MMLU loader baked "Respond with ONLY the letter (A, B, C, or D)" into the question string,
which was then reused inside every meta prompt. ``variant="embedded"`` reproduces that exactly (so the
effect can be measured); ``variant="raw"`` builds the meta prompt from content only.
"""

from dataclasses import dataclass

from .scales import Scale

# Legacy templates, character for character (tests assert equality with esma.prompt).
DIRECT_OPEN = "Answer the following question with keywords.\nQuestion: {question}\n"
META_YESNO = (
    'Do you know the answer to the following question? If you know and are sure about the answer, '
    'just return "Yes". If you don\'t know the answer or are uncertain, just return "No".\n'
    "Question: {question}\n"
)
GRADED_HEAD = "Question: {question}\n\nHow confident are you that you can correctly answer this question? Return ONLY the {noun}.\n"
MCQ_INSTRUCTION = "Answer the following multiple-choice question. Respond with ONLY the letter ({letters}) of the correct option.\n\n"


@dataclass(frozen=True)
class Item:
    qid: str
    stem: str  # question content only: no response-format instruction
    options: tuple[str, ...] | None = None  # MCQ option texts in order; None for open questions
    answers: tuple[str, ...] = ()  # open QA: acceptable aliases
    gold: str | None = None  # MCQ: gold label, "A".."J"
    group: str = ""  # stratum, e.g. the MMLU subject

    @property
    def option_labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + i) for i in range(len(self.options or ())))

    def content(self, labels: tuple[str, ...] | None = None) -> str:
        """Stem plus labelled options, formatted like the legacy loader ("\nA) text"). `labels` replaces the option labels:
        the answer-label control shows the same options as 1) 2) 3) 4) so that they no longer share labels with a letter scale."""
        if self.options is None:
            return self.stem
        labels = labels or self.option_labels
        if len(labels) != len(self.options):
            raise ValueError(f"{len(labels)} labels for {len(self.options)} options")
        return self.stem + "".join(f"\n{lab}) {opt}" for lab, opt in zip(labels, self.options))

    def legacy_question(self) -> str:
        """The legacy question string: the answer-format instruction is part of it for MCQ."""
        if self.options is None:
            return self.stem
        return mcq_instruction(self.option_labels) + self.content()


def letters_phrase(labels: tuple[str, ...]) -> str:
    """'A, B, C, or D' (legacy wording for four options)."""
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def mcq_instruction(labels: tuple[str, ...]) -> str:
    return MCQ_INSTRUCTION.format(letters=letters_phrase(labels))


def direct_prompt(item: Item) -> str:
    """Direct-answer prompt. MCQ gets one clean instruction; the legacy prompt nested two."""
    if item.options is None:
        return DIRECT_OPEN.format(question=item.stem)
    return mcq_instruction(item.option_labels) + item.content()


def graded_prompt(question: str, scale: Scale) -> str:
    return GRADED_HEAD.format(question=question, noun=scale.noun) + "\n".join(scale.lines()) + "\n"


OPTION_DIGITS = ("1", "2", "3", "4", "5", "6", "7", "8", "9")


def meta_prompt(item: Item, probe: str, variant: str = "raw", scale: Scale | None = None) -> str:
    """Meta (confidence) prompt. probe: 'yesno' or 'graded'. variant: 'embedded' (the legacy question string, with the
    answer-format instruction), 'raw' (stem and lettered options), 'stem' (the question alone, no options shown) or 'optdigits'
    (stem and the same options labelled 1) 2) 3) 4): the answer-label control, which keeps a letter confidence scale fixed
    and removes its overlap with the answer labels)."""
    if variant not in ("raw", "embedded", "stem", "optdigits"):
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "optdigits" and item.options is not None and len(item.options) > len(OPTION_DIGITS):
        raise ValueError("optdigits needs at most nine options")
    question = {
        "embedded": item.legacy_question,
        "raw": item.content,
        "stem": lambda: item.stem,
        "optdigits": lambda: item.content(OPTION_DIGITS[: len(item.options or ())] or None),
    }[variant]()
    if probe == "yesno":
        return META_YESNO.format(question=question)
    if probe == "graded":
        if scale is None:
            raise ValueError("graded probe needs a scale")
        return graded_prompt(question, scale)
    raise ValueError(f"unknown probe {probe!r}")


FORMAT_PHRASES = ("Respond with", "Answer the following", "Return ONLY the letter")


def assert_no_format_instruction(question_text: str) -> None:
    """C10: a raw meta question must not contain a response-format instruction."""
    for phrase in FORMAT_PHRASES:
        if phrase in question_text:
            raise AssertionError(f"response-format instruction leaked into question content: {phrase!r}")
