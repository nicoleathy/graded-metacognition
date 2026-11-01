"""Confidence scales: label sets, descriptors, and descriptor-moving relabellings.

A scale attaches one label to each of four meanings (most confident first). Relabelling keeps the
descriptor text with its meaning and changes only the label, so a probe that reads the meaning (not the
label token) assigns the same semantic grade under every relabelling.
"""

import os
from dataclasses import dataclass

# Descriptor text per meaning, most confident first. w1 is identical to the legacy graded prompt (the study); w2 and w3 are the two pre-specified
# paraphrases of the replication (frozen/replication_plan.json): w2 rewords the descriptors, w3 moves from "know the answer" to "my answer is correct".
WORDINGS = {
    "w1": ("I am certain I know the answer", "I think I probably know", "I'm not sure", "I definitely do not know"),
    "w2": ("I know the answer for sure", "I probably know the answer", "I might not know the answer", "I do not know the answer at all"),
    "w3": ("My answer would almost certainly be correct", "My answer would probably be correct", "My answer might well be wrong", "My answer would almost certainly be wrong"),
}
WORDING = os.environ.get("PROBE_WORDING", "w1")  # read once: one process, one wording; the run metadata records it
if WORDING not in WORDINGS:
    raise ValueError(f"PROBE_WORDING must be one of {sorted(WORDINGS)}, not {WORDING!r}")
DESCRIPTORS = WORDINGS[WORDING]

YESNO_LABELS = ("Yes", "No")


@dataclass(frozen=True)
class Scale:
    name: str
    labels: tuple[str, ...]  # label attached to each meaning, most confident first
    descriptors: tuple[str, ...] = DESCRIPTORS
    noun: str = "letter"  # "Return ONLY the {noun}."
    present: str = "label"  # order of the prompt lines: "label" (sorted by label), "meaning" (most confident first), "reverse_meaning"
    separator: str = ")"  # "A) text" or "CERTAIN: text"

    @property
    def n(self) -> int:
        return len(self.labels)

    def grade_of(self, label: str | None) -> int | None:
        """Ordinal grade (higher = more confident) of a normalised label, or None if it is not on the scale."""
        if label is None or label not in self.labels:
            return None
        return self.n - 1 - self.labels.index(label)

    def lines(self) -> list[str]:
        """The scale as prompt lines, in presentation order."""
        order = list(range(self.n))  # meaning order: 0 is the most confident
        if self.present == "label":
            order.sort(key=lambda i: self.labels[i])
        elif self.present == "reverse_meaning":
            order.reverse()
        elif self.present != "meaning":
            raise ValueError(f"unknown presentation {self.present!r}")
        return [f"{self.labels[i]}{self.separator} {self.descriptors[i]}" for i in order]


# Same presentation as the legacy prompt: A) certain ... D) definitely do not know.
LETTERS = Scale("letters", ("A", "B", "C", "D"))
# Descriptors move with their meaning; only the label changes. Lines stay in alphabetical label order.
LETTERS_REVERSED = Scale("letters_reversed", ("D", "C", "B", "A"))
LETTERS_PERM = Scale("letters_perm", ("C", "A", "D", "B"))  # certain->C, probably->A, unsure->D, none->B
# Digits swapped in for the letters, same presentation order (1 = certain ... 4 = definitely do not know).
DIGITS = Scale("digits", ("1", "2", "3", "4"), noun="number")
WORDS = Scale("words", ("CERTAIN", "PROBABLY", "UNSURE", "NO IDEA"), noun="label", present="meaning", separator=":")

SCALES = {s.name: s for s in (LETTERS, LETTERS_REVERSED, LETTERS_PERM, DIGITS, WORDS)}

# Relabelling by label order moves two things at once: the label that carries a meaning and the line where the meaning is
# printed (LETTERS_REVERSED prints "A) I definitely do not know" first). These two controls pull them apart; with LETTERS and
# LETTERS_REVERSED they form a 2 x 2: labels (A = certain / D = certain) x order (certain printed first / last).
LETTERS_FLIP_ORDER = Scale("letters_flip_order", ("A", "B", "C", "D"), present="reverse_meaning")  # labels keep their meaning, order flipped
LETTERS_FLIP_LABELS = Scale("letters_flip_labels", ("D", "C", "B", "A"), present="meaning")  # order kept, labels flipped
CONTROL_SCALES = {s.name: s for s in (LETTERS_FLIP_ORDER, LETTERS_FLIP_LABELS)}
FACTORIAL = ("letters", "letters_flip_order", "letters_flip_labels", "letters_reversed")

# The binary probe, for parsing only: its prompt is META_YESNO, and it has no descriptors.
YESNO = Scale("yesno", YESNO_LABELS, descriptors=(), noun="word")
PROBE_SCALES = {"yesno": YESNO, **SCALES, **CONTROL_SCALES}


def label_overlap(scale: Scale, answer_labels: tuple[str, ...]) -> tuple[str, ...]:
    """Confidence labels that are also answer labels (case-insensitive). Empty means collision-free (S2)."""
    answers = {a.lower() for a in answer_labels}
    return tuple(lab for lab in scale.labels if lab.lower() in answers)
