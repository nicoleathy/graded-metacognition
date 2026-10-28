import pytest

from esma.metric import correctness_by_inclusion
from esma.utils import normalize_answer as legacy_normalize
from probe_validity.scoring import McqScore, is_hedged, legacy_mmlu_correct, score_mcq, score_open
from probe_validity.text import first_span, normalize_answer

LABELS = ("A", "B", "C", "D")


def test_normalize_answer_is_the_legacy_function():
    for s in ["The Eiffel Tower!", "  a  b   c ", "U.S.A.", "An apple, the pear", "", "ça va?", "Hello-World"]:
        assert normalize_answer(s) == legacy_normalize(s)


def test_mcq_scorer_reads_the_first_label():
    assert score_mcq("B", "B", LABELS).correct == 1
    assert score_mcq("C) Paris", "B", LABELS).correct == 0
    assert score_mcq("The answer is B.", "B", LABELS) == McqScore(1, "B", "phrase")
    assert score_mcq("The answer is B.", "B", LABELS).tier == "phrase"
    s = score_mcq("I think a good guess is B", "B", LABELS)
    assert (s.correct, s.label, s.tier) == (1, "B", "loose")
    s = score_mcq("I do not know", "B", LABELS)
    assert (s.correct, s.label, s.tier) == (0, None, "none")
    assert score_mcq("", "A", LABELS).correct == 0


def test_mcq_strict_policy_ignores_everything_but_a_leading_label():
    assert score_mcq("The answer is B.", "B", LABELS, "strict").correct == 0
    assert score_mcq("B", "B", LABELS, "strict").correct == 1


def test_legacy_mmlu_scorer_reads_the_article_as_a_choice():
    assert legacy_mmlu_correct("I think a good guess is B", "B") == 0
    assert legacy_mmlu_correct("I think a good guess is B", "A") == 1  # right for the wrong reason
    assert score_mcq("I think a good guess is B", "A", LABELS).correct == 0


def test_open_scorer_golden_cases():
    assert score_open("Paris", ["Paris"]).correct == 1
    assert score_open("paris.", ["Paris"]).exact == 1
    assert score_open("The Eiffel Tower", ["Eiffel Tower"]).exact == 1
    assert score_open("Paris, France", ["Paris"]).correct == 1  # commits to one answer plus context
    assert score_open("It is Paris", ["Paris"]).correct == 1
    assert score_open("Paris\n\nQuestion: Who wrote Hamlet?\nShakespeare", ["Paris"]).correct == 1
    assert score_open("Parisian", ["Paris"]).correct == 0  # token boundary
    assert score_open("", ["Paris"]).correct == 0


def test_hedged_lists_and_short_aliases_score_wrong():
    hedge = score_open("Paris, London, Rome", ["Paris"])
    assert (hedge.hedged, hedge.contains, hedge.correct, hedge.legacy) == (1, 1, 0, 1)
    assert score_open("Paris or London", ["Paris"]).correct == 0
    assert score_open("Paris / London", ["Paris"]).correct == 0
    short = score_open("1999", ["9"])
    assert (short.correct, short.legacy) == (0, 1)
    assert score_open("The year was 1999", ["1999"]).correct == 1
    assert score_open("Hamlet or Macbeth", ["Hamlet or Macbeth"]).correct == 1  # an exact alias is never a hedge


def test_sentences_are_not_hedges():
    reply = "Robert F. Kennedy was assassinated on June 5, 1968, by Sirhan Sirhan, a Palestinian immigrant."
    assert score_open(reply, ["Sirhan Sirhan"]).correct == 1 and score_open(reply, ["Sirhan Sirhan"]).hedged == 0
    assert score_open("The currency of Israel is the Israeli new shekel.", ["Shekel"]).correct == 1
    assert score_open("It could be Paris or London", ["Paris"]).correct == 0  # short and lists alternatives
    # Known limit, measured by the human audit rather than hidden: a long sentence that hedges is not detected.
    assert score_open("It might be the Eiffel Tower, the Louvre, or the Arc de Triomphe in the city of Paris", ["Louvre"]).correct == 1


def test_legacy_column_equals_the_original_inclusion():
    cases = [("Paris, London", ["paris"]), ("none", ["Rome"]), ("1999", ["9"]), ("", ["x"]), ("Hamlet", ["hamlet", "Macbeth"])]
    for reply, aliases in cases:
        assert score_open(reply, aliases).legacy == correctness_by_inclusion([reply], [aliases])[0]


@pytest.mark.parametrize("reply,span", [("\n\n  Paris \nx", "Paris"), ("", ""), ("   \n  ", ""), ("one", "one")])
def test_first_span(reply, span):
    assert first_span(reply) == span


def test_is_hedged():
    assert is_hedged("Either Paris or London") and is_hedged("a; b; c") and is_hedged("A, B, C, D")
    assert not is_hedged("Paris, France") and not is_hedged("Washington") and not is_hedged("")
    assert not is_hedged("one two three four five six seven eight nine ten eleven, or twelve")  # longer than a list


def test_a_rejected_alias_is_not_credited():
    assert score_open("Not Paris; London.", ["Paris"]).correct == 0  # used to score correct
    assert score_open("Not Paris; London.", ["London"]).correct == 1
    assert score_open("It is not Paris, it is London", ["Paris"]).correct == 0
    assert score_open("It is not Paris, it is London", ["London"]).correct == 1
    assert score_open("Paris, not London", ["London"]).correct == 0
    assert score_open("Paris, not London", ["Paris"]).correct == 1
    assert score_open("Rather than Paris, London", ["Paris"]).correct == 0
    assert score_open("It isn't Paris", ["Paris"]).correct == 0
    assert score_open("No, it is Paris", ["Paris"]).correct == 1  # a leading "No" does not reject what follows
    assert score_open("Paris is not the capital of Germany", ["Paris"]).correct == 1  # the negation comes after the alias
    assert score_open("not Paris", ["not Paris"]).correct == 1  # an exact alias is never rejected


@pytest.mark.parametrize("reply,gold,expected", [
    ("B or C", "B", 0), ("A or B", "B", 0), ("A, B", "A", 0), ("B/C", "B", 0), ("The answer is B or C", "B", 0),
    ("The answer is not B", "B", 0), ("B is wrong", "B", 0), ("Not B", "B", 0), ("I think it's not B but C", "C", 0),
    ("B, not C", "B", 1), ("The answer is B, not C", "B", 1), ("A good guess is B", "B", 1), ("B", "B", 1), ("The answer is C.", "C", 1),
])
def test_multiple_choice_ambiguity_and_rejection(reply, gold, expected):
    assert score_mcq(reply, gold, LABELS).correct == expected


def test_clause_tokenisation_equals_the_global_normalisation():
    from probe_validity.scoring import _tokens_with_clauses

    for span in ["Not Paris; London.", "U.S.A. and the U.K.", "St.Louis, Missouri", "Jean-Paul Sartre", "A tale of two cities", "", "  The  ",
                 "It is not Paris, but London!", "co-operate (sic) 1,000 km", "ça va? oui."]:
        tokens, clauses = _tokens_with_clauses(span)
        assert tokens == normalize_answer(span).split() and len(clauses) == len(tokens), span


# --- methodology audit, 2026-10-27: open answers that name the gold alias but do not commit to it ---

@pytest.mark.parametrize("reply", [
    "Paris is not the answer; London is.",
    "I am not sure whether the correct answer is Paris or London.",
    "I don't know, maybe Paris.",
    "It is either Paris or London, depending on the year and the source you use.",
    "Paris isn't it, it is London",
])
def test_uncommitted_or_rejected_aliases_do_not_score(reply):
    assert score_open(reply, ["Paris"]).correct == 0, reply


@pytest.mark.parametrize("reply", [
    "Paris", "The answer is Paris.", "Paris, not London", "Paris, France", "It is Paris, which is the capital of France",
    "Paris is the capital and the largest city of France, and it is not Lyon or Marseille",
])
def test_committed_answers_still_score(reply):
    assert score_open(reply, ["Paris"]).correct == 1, reply


def test_short_lists_are_flagged_for_the_sensitivity_analysis_but_stay_correct():
    assert score_open("Paris, London", ["Paris"]).list_like == 1
    assert score_open("Paris, France", ["Paris"]).list_like == 1  # the same shape as a committed answer with a qualifier
    assert score_open("Paris, London", ["Paris"]).correct == 1
    assert score_open("Paris", ["Paris"]).list_like == 0
    assert score_open("It is Paris, which is the capital of France", ["Paris"]).list_like == 0
