"""Reply parsing for confidence probes and multiple-choice answers.

Three parsers, kept apart on purpose:

* strict  - primary. The reply must start with exactly one scale label (markdown, quotes and brackets are
            ignored). Anything else is invalid, and invalid replies are never imputed. (Multiple-choice answers have
            two further tiers: a stated phrase, and a short first line that names the option after a cue word such
            as "option" or "is" and is not part of a formula.)
* lenient - sensitivity analysis only: adds descriptor text and a label found early in the reply.
* legacy  - the original regexes, verbatim, to measure what they did to the numbers.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from .scales import Scale
from .text import first_span, normalize_answer, straighten

OPEN = r"[\s*_`\"'(\[{<]*"
CLOSE = r"[*_`\"')\]}>]*"
DELIM = ".:,;!-\u2013\u2014"


NEGATORS = frozenset("not no never neither nor isnt wasnt arent werent cannot cant dont doesnt didnt except without".split())
# "B is wrong", "B is not correct", "B isn't right": the reply names a label only to reject it
_DENIAL = re.compile(r"^[\s)\].:,;!\-]*(?:(?:is|was|would\s+be)\s+(?:not|wrong|incorrect|false)\b|isn't\b|wasn't\b)", re.I)


@dataclass(frozen=True)
class Parse:
    label: str | None  # the label exactly as spelled on the scale, or None when the reply is invalid
    reason: str = "ok"  # why it is invalid: empty, no_leading_label, off_scale, article, pronoun, multiple, negated, no_label
    tier: str = "none"  # strict, descriptor, lenient (probes); strict, echo, phrase, loose (answers)

    @property
    def valid(self) -> bool:
        return self.label is not None


def _alts(labels: tuple[str, ...]) -> str:
    bodies = (r"\s+".join(re.escape(w) for w in lab.split()) for lab in labels)
    return "|".join(sorted(bodies, key=len, reverse=True))


@lru_cache(maxsize=None)
def _lead_re(labels: tuple[str, ...]) -> re.Pattern:
    return re.compile(rf"^{OPEN}(?P<label>{_alts(labels)})(?P<close>{CLOSE})(?P<rest>.*)$", re.I | re.S)


@lru_cache(maxsize=None)
def _next_re(labels: tuple[str, ...]) -> re.Pattern:
    sep = rf"[\s{re.escape(DELIM)}]*(?:(?:or|and)\s+|[/&]\s*)?"
    # A second label is not followed by a word character or by a formula or unit character: "E= E_0 (i_x ...)", "e^-t" and "49°F" are text, not
    # a label; "B or C?", "B or C..." and "B or C." still list two labels.
    return re.compile(rf"^{sep}{OPEN}(?P<label>{_alts(labels)})(?P<close>{CLOSE})(?![\w=^*+<>|\u00b0])", re.I)


@lru_cache(maxsize=None)
def _phrase_re(labels: tuple[str, ...]) -> re.Pattern:
    head = r"(?i:\b(?:correct\s+(?:answer|option|choice)|best\s+(?:answer|option|choice)|answer)\b\s*(?:is|would\s+be|should\s+be|:|=|-)?\s*)"
    return re.compile(head + r"[(*\[]*(?P<label>" + "|".join(map(re.escape, labels)) + r")(?![\w'])")


@lru_cache(maxsize=None)
def answer_scale(labels: tuple[str, ...]) -> Scale:
    """A label set used only for parsing multiple-choice answers."""
    return Scale("answer", labels, descriptors=())


def _canonical(written: str, scale: Scale) -> str | None:
    """The label of the scale that `written` spells, matched the way the patterns match it (case-insensitively, whitespace collapsed), or
    None. Unicode case folding makes "\u0131" (dotless i) match the label I in a pattern, so a lower-casing comparison would find nothing."""
    key = " ".join(written.split())
    return next((lab for lab in scale.labels if re.fullmatch(re.escape(lab), key, re.I)), None)


def _is_article(written: str, closed: str, rest: str) -> bool:
    """True if a leading "A"/"a" reads as the English article rather than the label: it is followed by a space
    and then a lowercase word ("A good guess is B"), or it is a lowercase "a" followed by any word."""
    if closed or not rest or not rest[0].isspace() or not rest.split():
        return False
    return written == "a" or rest.split(None, 1)[0][0].islower()


PRONOUN_NEXT = frozenset("think believe do don't dont am would will can cannot can't cant have need say guess feel know suppose choose pick go select prefer recommend".split())


def _is_pronoun(written: str, closed: str, rest: str) -> bool:
    """True if a leading "I" reads as the pronoun rather than option I of a ten-option question ("I think the answer is D", "I do not
    know", "I THINK D"): it is unclosed and followed by a space and a lowercase word or a typical verb in any case."""
    if written.lower() != "i" or closed or not rest or not rest[0].isspace() or not rest.split():
        return False
    word = rest.split(None, 1)[0]
    return word[0].islower() or word.lower().strip(".,;:!?") in PRONOUN_NEXT


def _second_label(rest: str, scale: Scale) -> str | None:
    """The label offered right after the first one ("B or C", "B, C", "Yes/No"), or None. The article and the pronoun that open
    the text of an option ("C) a base pair is ...", "D) A jury finding ...") are words, not a second label."""
    nxt = _next_re(scale.labels).match(rest)
    if nxt is None:
        return None
    written, closed, after = nxt.group("label"), nxt.group("close"), rest[nxt.end():]
    label = _canonical(written, scale)
    if label is None or (label == "A" and _is_article(written, closed, after)) or (label == "I" and _is_pronoun(written, closed, after)):
        return None
    return label


def parse_strict(reply: str, scale: Scale) -> Parse:
    """The reply must start with one label of the scale. The label may be wrapped in markdown or brackets and
    may be followed by punctuation, whitespace, or the end of the reply; two different labels in a row
    ("B, C", "Yes/No") are ambiguous."""
    text = straighten(reply)
    if not text:
        return Parse(None, "empty")
    m = _lead_re(scale.labels).match(text)
    if m is None:
        return Parse(None, "no_leading_label")
    written, closed, rest = m.group("label"), m.group("close"), m.group("rest")
    label = _canonical(written, scale)
    if label is None:
        return Parse(None, "no_leading_label")
    if label == "I" and re.match(rf"^{OPEN}i'(?:d|m|ll|ve)\b", text, re.I):
        return Parse(None, "pronoun")  # "I'd say B", "*I'm not sure*": the closing-quote rule would otherwise read the apostrophe as markup
    if rest and not closed and not (rest[0].isspace() or rest[0] in DELIM):
        if rest[0] in "/&" and _second_label(rest, scale) not in (None, label):
            return Parse(None, "multiple")  # "B/C", "Yes/No": two labels joined by a separator
        return Parse(None, "no_leading_label")  # "Because", "Not sure", "Confident", "10", "1/5"
    if label.isdigit() and re.match(r"[.,]\d", rest):
        return Parse(None, "off_scale")  # "1.5", "1,000"
    other = _second_label(rest, scale)
    if other is not None and other != label:
        return Parse(None, "multiple")  # checked before the article rule: "A or B" lists two labels
    if label == "A" and _is_article(written, closed, rest):
        return Parse(None, "article")
    if label == "I" and _is_pronoun(written, closed, rest):
        return Parse(None, "pronoun")
    return Parse(label, "ok", "strict")


# Words that name an option ("option B", "I would say B", "the answer is B") and negators, so that a denial ("not B") is still seen and reported as one.
_LABEL_CUE = re.compile(r"(?:\b(?:option|answer|choice|letter|select|choose|chose|pick|say|guess|think|believe|with|probably|maybe|perhaps|likely|is|be|was|are|not|never|no|neither|nor|except)|\w+n't|:)\s*[(\[*_\"']*$", re.I)
_FORMULA_BEFORE = re.compile(r"[=*+/^<>\u00d7\u00b7\u2212]\s*$")
_FORMULA_AFTER = re.compile(r"^\s*[=*+/^<>\u00d7\u00b7\u2212\u00b0%]")
LOOSE_MAX_WORDS = 14  # a longer first line is a sentence or a derivation, not a short statement of the answer


def _standalone(text: str, scale: Scale) -> list[tuple[int, str]]:
    """Labels that occur as separate tokens in text: (position, label), in order. Single letters must be
    capitals (or written "(b)") so that the article "a" is never read as a label."""
    hits: list[tuple[int, str]] = []
    for lab in scale.labels:  # a pronoun "I" is not option I unless a cue word names it ("option I", "answer is I")
        if len(lab) == 1 and lab.isalpha():
            pats = [rf"(?<![\w'])({re.escape(lab.upper())})(?![\w'])", rf"(?<![\w'])\(({re.escape(lab.lower())})\)(?![\w'])"]
            flags = 0
        elif lab.isdigit():
            pats, flags = [rf"(?<![\w.])({lab})(?!\w|[.,]\d)"], 0
        else:
            pats, flags = [r"(?<![\w'])(" + r"\s+".join(re.escape(w) for w in lab.split()) + r")(?![\w'])"], re.I
        for pat in pats:
            for m in re.finditer(pat, text, flags):
                if lab == "A" and m.group(1) == "A" and _is_article("A", "", text[m.end():]):
                    continue
                if lab == "I" and _is_pronoun(m.group(1), "", text[m.end():]) and not _LABEL_CUE.search(text[: m.start()]):
                    continue
                hits.append((m.start(), lab))
    return sorted(hits)


def parse_lenient(reply: str, scale: Scale, window: int = 12) -> Parse:
    """Sensitivity parser: strict, else the one descriptor text present, else the one label among the first
    `window` words. Contradictory evidence is invalid."""
    strict = parse_strict(reply, scale)
    if strict.valid:
        return strict
    text = straighten(reply)
    if not text:
        return strict
    low = text.lower()
    meanings = [i for i, d in enumerate(scale.descriptors[: scale.n]) if straighten(d).lower() in low]
    if len(meanings) == 1:
        return Parse(scale.labels[meanings[0]], "ok", "descriptor")
    if len(meanings) > 1:
        return Parse(None, "multiple")
    found = {lab for _, lab in _standalone(" ".join(text.split()[:window]), scale)}
    if len(found) == 1:
        return Parse(found.pop(), "ok", "lenient")
    return Parse(None, "multiple" if found else "no_label")


def _negated_before(text: str, start: int) -> bool:
    """A negation word among the two words before position start ("not B", "is not B")."""
    words = [w.replace("'", "").lower() for w in re.findall(r"[A-Za-z']+", text[:start])[-2:]]
    return any(w in NEGATORS for w in words)


def _checked(text: str, start: int, end: int, label: str, labels: tuple[str, ...], tier: str) -> Parse:
    """Accept a label found inside a reply only if no other label is offered right after it and it is not rejected."""
    after = text[end:]
    other = _second_label(after, answer_scale(labels))
    if other is not None and other != label:
        return Parse(None, "multiple")
    if _negated_before(text, start) or _DENIAL.match(after):
        return Parse(None, "negated")
    return Parse(label, "ok", tier)


def _prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _echoes(body: list[str], option: list[str]) -> bool:
    """The reply text restates the option: it starts with the option's words (and may go on), or it is the start of them cut off
    at the token limit. At least min(len(option), 3) leading words must agree."""
    k = _prefix_len(body, option)
    if not option or k < min(len(option), 3):
        return False
    if k == len(option) or k == len(body):
        return True
    return k == len(body) - 1 and option[k].startswith(body[k])  # the last word was cut off


def _echo_parse(text: str, labels: tuple[str, ...], options: tuple[str, ...]) -> Parse | None:
    """A reply that starts with an explicit option marker ("B)", "(B)", "B.", "B:") followed by the text of that same option is an
    answer, even when the text opens with an article, a negation or a subpart label ("B) is not always hate speech"). The text
    must restate the marked option and no other one; otherwise nothing is rescued."""
    m = _lead_re(labels).match(text)
    if m is None or len(options) != len(labels):
        return None
    close, rest = m.group("close"), m.group("rest")
    if not (re.search(r"[)\]]", close) or rest[:1] in (".", ":")):
        return None  # no explicit marker: "B is wrong" stays a denial
    label = _canonical(m.group("label"), answer_scale(labels))
    if label is None:
        return None
    body = normalize_answer(rest).split()
    texts = {lab: normalize_answer(opt).split() for lab, opt in zip(labels, options)}
    if _echoes(body, texts[label]) and not any(_echoes(body, t) for lab, t in texts.items() if lab != label):
        return Parse(label, "ok", "echo")
    return None


def parse_mcq(reply: str, labels: tuple[str, ...], options: tuple[str, ...] | None = None) -> Parse:
    """A multiple-choice answer, tiers in order: strict (the reply starts with the label), phrase ("the answer is B"), loose
    (a short first line that names the option after a cue word, as in "option B" or "I would say B", and is not part of a formula or a unit). Ambiguity ("B or C", two different letters) and rejection ("not B",
    "B is wrong") are never rescued by a looser tier: the reply is invalid and counts as incorrect. With `options` (the option
    texts, aligned with `labels`) one rescue exists: "B) <the text of option B>" is an answer to B whatever the text opens with
    (tier "echo"); it needs the marked option's own text and refuses a reply that restates a different option."""
    labels = tuple(labels)
    scale = answer_scale(labels)
    strict = parse_strict(reply, scale)
    text = straighten(reply)
    opts = tuple(options) if options else None
    if strict.valid:
        rest = _lead_re(labels).match(text).group("rest")
        if _DENIAL.match(rest):
            return (_echo_parse(text, labels, opts) if opts else None) or Parse(None, "negated")
        return strict
    if strict.reason == "multiple":
        return (_echo_parse(text, labels, opts) if opts else None) or strict
    m = _phrase_re(labels).search(text)
    if m:
        return _checked(text, m.start("label"), m.end("label"), m.group("label"), labels, "phrase")
    line = straighten(first_span(reply))
    if len(line.split()) > LOOSE_MAX_WORDS:
        return Parse(None, strict.reason)
    hits = []
    for pos, lab in _standalone(line, scale):
        token = re.match(r"\(?[A-Za-z0-9]+\)?", line[pos:])
        end = pos + (token.end() if token else 1)
        if _LABEL_CUE.search(line[:pos]) and not _FORMULA_BEFORE.search(line[:pos]) and not _FORMULA_AFTER.match(line[end:]):
            hits.append((pos, lab, end))
    if len({lab for _, lab, _ in hits}) > 1:
        return Parse(None, "multiple")
    if hits:
        pos, label, end = hits[0]
        return _checked(line, pos, end, label, labels, "loose")
    return Parse(None, strict.reason)


def mcq_label(reply: str, labels: tuple[str, ...], policy: str = "first_label", options: tuple[str, ...] | None = None) -> str | None:
    """Answer label under a named policy: "first_label" (primary: any tier), "strict" (leading label only),
    or "legacy" (the original regex)."""
    if policy == "legacy":
        return legacy_mmlu_letter(reply)
    p = parse_mcq(reply, labels, options)
    if policy == "strict":
        return p.label if p.tier == "strict" else None
    if policy == "first_label":
        return p.label
    raise ValueError(f"unknown policy {policy!r}")


# --- legacy parsers, reproduced verbatim from esma/metric.py (tests compare them with the originals) ---

_LEGACY_GRADES = {"a": 3, "b": 2, "c": 1, "d": 0}


def legacy_yes(reply: str) -> int:
    """meta_yes: every reply that does not contain "yes" is a "no", whatever it says."""
    return 1 if "yes" in reply.lower() else 0


def legacy_graded(reply: str) -> tuple[int, bool]:
    """parse_graded_response_strict for one reply: (grade, parsed_ok). Matches the word "a" as grade 3 and
    imputes 1 when nothing matches."""
    low = reply.strip().lower()
    match = re.search(r"\b([abcd])\s*[).\]]?", low)
    if match:
        return _LEGACY_GRADES[match.group(1)], True
    if "certain" in low or "sure" in low:
        return 3, False
    if "probably" in low or "think" in low:
        return 2, False
    if "uncertain" in low or "not sure" in low:
        return 1, False
    if "not know" in low or "no idea" in low:
        return 0, False
    return 1, False


def legacy_mmlu_letter(reply: str) -> str | None:
    """correctness_mmlu's extraction: the first standalone A-D in the upper-cased reply (so "a" counts)."""
    match = re.search(r"\b([ABCD])\b", reply.upper())
    return match.group(1) if match else None
