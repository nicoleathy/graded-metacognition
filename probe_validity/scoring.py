"""Correctness of direct answers: multiple-choice letters and short open answers."""

import re
from dataclasses import dataclass

from .parse import legacy_mmlu_letter, mcq_label, parse_mcq
from .text import first_span, normalize_answer, unify_punctuation


@dataclass(frozen=True)
class McqScore:
    correct: int  # 1 only if the parsed label equals the gold label; a reply with no label is incorrect
    label: str | None
    tier: str  # how the label was found: strict, phrase, loose, legacy; "none" when invalid


def score_mcq(reply: str, gold: str, labels: tuple[str, ...], policy: str = "first_label", options: tuple[str, ...] | None = None) -> McqScore:
    label = mcq_label(reply, labels, policy, options)
    if label is None:
        return McqScore(0, None, "none")
    tier = "legacy" if policy == "legacy" else parse_mcq(reply, labels, options).tier
    return McqScore(int(label == gold), label, tier)


def legacy_mmlu_correct(reply: str, gold: str) -> int:
    """correctness_mmlu: the first standalone A-D in the upper-cased reply equals the gold letter."""
    pred = legacy_mmlu_letter(reply)
    return int(pred == gold.strip().upper()) if pred is not None else 0


@dataclass(frozen=True)
class OpenScore:
    correct: int  # primary: the first span equals an alias, or contains one at word boundaries outside a negation, and is not a hedge
    exact: int
    contains: int
    hedged: int
    legacy: int  # the original rule: an alias occurs anywhere in the reply, as a plain substring
    span: str
    list_like: int = 0  # a short comma/semicolon list ("Paris, London"; also "Paris, France"): kept correct, reported for sensitivity


HEDGE_MAX_TOKENS = 10  # a longer span is a sentence: its "or" and "either" are not a list of candidates
LIST_MAX_TOKENS = 5  # three or more comma items are a list of candidates only in a span this short, counting an accepted answer as one token ("Paris, London, Berlin")
_ITEM_SPLIT = re.compile(r"(?<!\d),(?!\d)|;")  # a comma between digits is a thousands separator, not a list
UNCOMMITTED = (("not", "sure"), ("not", "certain"), ("unsure",), ("dont", "know"), ("do", "not", "know"), ("no", "idea"), ("no", "clue"), ("whether",),
               ("either",), ("cannot", "say"), ("cant", "say"))  # words that withhold an answer, at any span length


def _has_phrase(tokens: list[str], phrases) -> bool:
    return any(tuple(tokens[i : i + len(p)]) == p for p in phrases for i in range(len(tokens) - len(p) + 1))


def _mask_aliases(span: str, aliases) -> str:
    """The span with every accepted answer replaced by a placeholder, so that words that belong to an answer ("Trick or Treat", "1,000,000",
    "Washington, D.C.") are never read as a hedge, a list or a negation."""
    for alias in sorted({a.strip() for a in aliases if a.strip()}, key=len, reverse=True):
        span = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", " ALIAS ", span, flags=re.I)
    return span


def is_hedged(span: str) -> bool:
    """A reply that withholds or splits its answer: it says it is not sure / does not know / "whether" (any length), or a short span
    lists alternatives instead of committing: "X or Y", "X / Y", "either ...", or three or more items in a very short comma list. Sentences
    are never hedges for their commas ("Leonardo da Vinci, the Italian polymath, painted ..." has appositives, not alternatives)."""
    tokens = normalize_answer(span).split()
    if _has_phrase(tokens, UNCOMMITTED):
        return True
    if len(tokens) > HEDGE_MAX_TOKENS:
        return False
    low = f" {span.lower()} "
    if " or " in low or " / " in low or " either " in low:
        return True
    return len([part for part in _ITEM_SPLIT.split(span) if part.strip()]) >= 3 and len(tokens) <= LIST_MAX_TOKENS


NEGATORS = frozenset("not no never neither nor isnt wasnt arent werent cannot cant dont doesnt didnt except without unlike hardly barely scarcely".split())
CLAUSE_END = ";,:.!?"
CLAUSE_START = ("but", "however")
RETRACT = frozenset("no nope wait sorry oops correction".split())  # a clause that starts like this takes the previous answer back
_NOT_A_RETRACTION = {("no", "doubt"), ("no", "question"), ("no", "wonder")}


def _tokens_with_clauses(span: str) -> tuple[list[str], list[int]]:
    """normalize_answer(span).split() with the clause each token belongs to. A clause ends after a word that ends with
    punctuation and starts at "but" / "however", so a negation does not reach into the next clause."""
    tokens, clauses, clause = [], [], 0
    for word in span.split():
        norm = normalize_answer(word)
        if norm in CLAUSE_START:
            clause += 1
        for t in norm.split():
            tokens.append(t)
            clauses.append(clause)
        if word[-1] in CLAUSE_END:
            clause += 1
    return tokens, clauses


def _negated(tokens: list[str], clauses: list[int], i: int) -> bool:
    """The alias starting at token i is rejected: a negation word among the two tokens before it in the same clause, or
    "rather than" / "instead of" right before it."""
    before = [tokens[j] for j in range(max(0, i - 2), i) if clauses[j] == clauses[i]]
    return any(t in NEGATORS for t in before) or before in (["rather", "than"], ["instead", "of"])


_COPULA = frozenset("is was are were be would".split())
_REJECTION = frozenset("wrong incorrect false".split())
_ANSWERHOOD = frozenset("answer correct one it choice solution".split())


def _negated_after(tokens: list[str], clauses: list[int], i: int, n: int) -> bool:
    """The alias occupying tokens i..i+n-1 is rejected as the answer right after it, in the same clause: "Paris is not the answer",
    "Paris isn't it", "Paris is wrong". A negation about something else ("Paris is not the capital of Germany") or a "not only"
    ("Paris is not the only answer") does not reject it."""
    after = [tokens[j] for j in range(i + n, min(len(tokens), i + n + 4)) if clauses[j] == clauses[i]]
    while after and after[0] in _COPULA:
        after = after[1:]
    if not after:
        return False
    if after[0] in _REJECTION:
        return True
    if after[0] not in NEGATORS:
        return False
    if len(after) == 1:
        return True
    return "only" not in after[1:3] and any(t in _ANSWERHOOD for t in after[1:3])


def _retracted_after(tokens: list[str], clauses: list[int], i: int, n: int) -> bool:
    """The clause that follows the alias's own clause starts by taking it back ("Paris? No, London.", "Paris - wait, London")."""
    nxt = clauses[i + n - 1] + 1
    for j in range(i + n, len(tokens)):
        if clauses[j] == nxt:
            return tokens[j] in RETRACT and tuple(tokens[j : j + 2]) not in _NOT_A_RETRACTION
        if clauses[j] > nxt:
            return False
    return False


def _contains_tokens(tokens: list[str], clauses: list[int], alias: list[str]) -> bool:
    """The alias occurs at word boundaries at least once outside a negated or retracted context ("Not Paris; London" and "Paris is not the
    answer" do not contain Paris)."""
    n = len(alias)
    return n > 0 and any(tokens[i : i + n] == alias and not _negated(tokens, clauses, i) and not _negated_after(tokens, clauses, i, n)
                         and not _retracted_after(tokens, clauses, i, n) for i in range(len(tokens) - n + 1))


def _list_like(span: str) -> bool:
    parts = [part for part in _ITEM_SPLIT.split(span) if part.strip()]
    sizes = [len(normalize_answer(part).split()) for part in parts]
    return len(parts) >= 2 and all(s <= 4 for s in sizes) and sum(sizes) <= HEDGE_MAX_TOKENS


def score_open(reply: str, aliases: list[str] | tuple[str, ...]) -> OpenScore:
    span = unify_punctuation(first_span(reply))
    accepted = [unify_punctuation(a) for a in aliases]
    tokens, clauses = _tokens_with_clauses(span)
    normalised = [normalize_answer(a).split() for a in accepted]
    exact = int(any(tokens and tokens == alias for alias in normalised))
    contains = int(any(_contains_tokens(tokens, clauses, alias) for alias in normalised))
    masked = _mask_aliases(span, accepted)  # hedges, lists and uncommitted phrases are looked for outside the answers themselves
    hedged = int(is_hedged(masked))
    correct = int(exact or (contains and not hedged))
    legacy = int(any(a.lower() in reply.lower() for a in aliases))
    return OpenScore(correct, exact, contains, hedged, legacy, span, int(_list_like(masked)))


# ---- the sensitivity scorer ---------------------------------------------------------------------------------------------------------------------
# The frozen scorer above is stricter than a careful reader: it looks at the first line only and needs an alias to occur as written. The audit of
# 1 November 2026 found that two raters accepted replies it rejects ("Spanish" for "Spanish Language", an answer on a later line). This scorer
# relaxes exactly three things, chosen as the usual leniencies and not tuned on the audit sample, and is used only for a declared sensitivity
# analysis (OPEN_SCORER=lenient): the primary results are the frozen scorer's. It keeps the refusal to credit a reply that says it does not know and
# the negation and retraction rules, so it is more lenient, not an upper bound on every conceivable reading.
_GENERIC_TAIL = frozenset({"language", "languages"})
_QUALIFIER = re.compile(r"\s*[(\[][^)\]]*[)\]]")  # "True (album)", "Paris [city]"
_COMMON = frozenset("a an the i it is be to of in on and or no yes not true false".split())
LENIENT_PREFIX_MAX_TOKENS = 4  # a reply this short that is part of an accepted answer ("Spanish" for "Spanish Language") is that answer
LENIENT_MIN_COVERAGE = 0.5  # the part must be at least this share of the accepted answer's tokens


def _alias_variants(alias: str) -> list[list[str]]:
    """Token lists the lenient scorer accepts for one alias: as written, without a bracketed qualifier, and without a trailing generic word."""
    base = unify_punctuation(alias)
    found: set[tuple[str, ...]] = set()
    for text in (base, _QUALIFIER.sub("", base)):
        tokens = normalize_answer(text).split()
        if tokens:
            found.add(tuple(tokens))
            if len(tokens) > 1 and tokens[-1] in _GENERIC_TAIL:
                found.add(tuple(tokens[:-1]))
    return [list(v) for v in sorted(found)]


def _usable_anywhere(variant: list[str]) -> bool:
    """Searching the whole reply is safe for a multi-token answer or a single token of at least four letters that is not a common word."""
    return len(variant) > 1 or (len(variant[0]) >= 4 and variant[0] not in _COMMON)


def _is_part_of(reply: list[str], variant: list[str]) -> bool:
    n = len(reply)
    if not (1 <= n <= LENIENT_PREFIX_MAX_TOKENS) or all(tok in _COMMON for tok in reply) or len(variant) < 2 or n / len(variant) < LENIENT_MIN_COVERAGE:
        return False
    return any(variant[i : i + n] == reply for i in range(len(variant) - n + 1))


def score_open_lenient(reply: str, aliases: list[str] | tuple[str, ...]) -> OpenScore:
    """The sensitivity scorer: an alias may occur anywhere in the reply (outside a negation or retraction), an alias is also read without a bracketed
    qualifier and without a trailing "language", and a short reply that is part of an alias counts. A reply that withholds an answer does not."""
    span = unify_punctuation(first_span(reply))
    whole = unify_punctuation(reply)
    accepted = [unify_punctuation(a) for a in aliases]
    strict = score_open(reply, aliases)
    variants = [v for a in accepted for v in _alias_variants(a)]
    tokens, clauses = _tokens_with_clauses(whole)
    contains = int(any(_usable_anywhere(v) and _contains_tokens(tokens, clauses, v) for v in variants))
    first_tokens, first_clauses = _tokens_with_clauses(span)
    contains = int(contains or any(_contains_tokens(first_tokens, first_clauses, v) for v in variants))  # short single-token aliases only in the first span
    part = int(any(_is_part_of(first_tokens, v) for v in variants))
    withheld = int(_has_phrase(normalize_answer(_mask_aliases(whole, accepted)).split(), UNCOMMITTED))
    correct = int(strict.correct or ((contains or part) and not withheld))
    return OpenScore(correct, strict.exact, contains, withheld, strict.legacy, span, 0)
