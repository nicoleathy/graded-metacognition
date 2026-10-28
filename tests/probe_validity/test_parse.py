import random

import pytest

from esma.metric import correctness_mmlu, meta_yes, parse_graded_response_strict
from probe_validity.parse import (
    legacy_graded,
    legacy_mmlu_letter,
    legacy_yes,
    mcq_label,
    parse_lenient,
    parse_mcq,
    parse_strict,
)
from probe_validity.scales import DIGITS, LETTERS, WORDS, YESNO
from probe_validity.scoring import legacy_mmlu_correct

LETTER_GOLDEN = [
    ("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"),
    ("A) I am certain I know the answer", "A"),
    ("D) I definitely do not know", "D"),
    ("(b)", "B"), ("**B**", "B"), ("B.", "B"), ("B - I think I probably know", "B"),
    ("  c\n", "C"), ("\"A\"", "A"), ("[D]", "D"), ("B: I think I probably know", "B"),
    ("A\n\nQuestion: next", "A"), ("a)", "A"), ("a", "A"), ("A, I am certain", "A"),
]
LETTER_INVALID = [
    "I'm not sure", "Not sure", "uncertain", "I am uncertain",
    "I'd say C", "I'd say B", "Confident", "Fairly confident", "Very confident",
    "Based on my knowledge, B", "Probably", "I think I probably know", "Could go either way",
    "High", "Low", "", "   ", "\n", "Because", "Definitely", "Certain",
    "A good guess", "a good guess is B", "B, C", "A or B", "B/C", "A, B, C", "B-C", "Answer: B",
]


@pytest.mark.parametrize("reply,label", LETTER_GOLDEN)
def test_letters_strict_valid(reply, label):
    p = parse_strict(reply, LETTERS)
    assert p.valid and p.label == label and p.tier == "strict"
    assert LETTERS.grade_of(p.label) == "DCBA".index(label)


@pytest.mark.parametrize("reply", LETTER_INVALID)
def test_letters_strict_invalid(reply):
    p = parse_strict(reply, LETTERS)
    assert not p.valid and p.label is None and p.reason != "ok"


def test_yesno_strict():
    for reply, label in [("Yes", "Yes"), ("No", "No"), ("yes.", "Yes"), ("**No**", "No"), ("No, I don't know", "No"),
                         ("Yes, I know the answer.", "Yes"), ("No idea", "No"), ("\"Yes\"", "Yes")]:
        assert parse_strict(reply, YESNO).label == label, reply
    for reply in ["Not sure", "I don't know", "Maybe", "Nope", "None", "Yes/No", "Yes or no", "Yesterday", "", "I think yes"]:
        assert not parse_strict(reply, YESNO).valid, reply


def test_digits_strict():
    for reply, label in [("1", "1"), ("4", "4"), ("3.", "3"), ("2) I'm not sure", "2"), ("**1**", "1"), ("(3)", "3")]:
        assert parse_strict(reply, DIGITS).label == label, reply
    for reply in ["0", "5", "10", "1.5", "1,000", "3%", "1/5", "1 2", "one", "I'd say 3", ""]:
        assert not parse_strict(reply, DIGITS).valid, reply
    assert parse_strict("1.5", DIGITS).reason == "off_scale"


def test_words_strict():
    for reply, label in [("CERTAIN", "CERTAIN"), ("Probably", "PROBABLY"), ("unsure.", "UNSURE"), ("No idea", "NO IDEA"),
                         ("NO IDEA: I definitely do not know", "NO IDEA"), ("**Certain**", "CERTAIN")]:
        assert parse_strict(reply, WORDS).label == label, reply
    for reply in ["I am certain", "Not sure", "Uncertain", "Certainly", "No", "Sure", "Probablyish"]:
        assert not parse_strict(reply, WORDS).valid, reply


def test_article_a_is_never_a_label():
    assert parse_strict("A good guess is B", LETTERS).reason == "article"
    assert parse_strict("a good guess is B", LETTERS).reason == "article"
    assert parse_strict("A is the best", LETTERS).reason == "article"  # ambiguous, so invalid (never imputed)
    assert parse_strict("A I am certain", LETTERS).label == "A"  # capital word after it: the label
    assert parse_strict("A - I am certain", LETTERS).label == "A"
    assert parse_strict("B because I know it", LETTERS).label == "B"  # only the letter A is ambiguous


def test_strict_is_a_pure_prefix_grammar_on_random_word_strings():
    vocab = ("the answer is think I probably know sure maybe because certain definitely a an to and of in it confident "
             "high low unsure not very fairly based on my knowledge uncertain could go either way").split()
    label_words = {"certain", "probably", "unsure"}  # these start a valid reply on the words scale
    rng = random.Random(0)
    for _ in range(3000):
        words = [rng.choice(vocab) for _ in range(rng.randint(2, 12))]
        reply = " ".join(words)
        if rng.random() < 0.3:
            reply = reply.capitalize()
        for scale in (LETTERS, DIGITS, YESNO, WORDS):
            if scale is WORDS and words[0] in label_words:
                continue
            assert not parse_strict(reply, scale).valid, (scale.name, reply)


def test_lenient_adds_descriptors_and_early_labels_only():
    assert parse_lenient("I'd say C", LETTERS).label == "C"
    assert parse_lenient("Based on my knowledge, B", LETTERS).label == "B"
    assert parse_lenient("Answer: B", LETTERS).label == "B"
    assert parse_lenient("I'm not sure", LETTERS).label == "C" and parse_lenient("I'm not sure", LETTERS).tier == "descriptor"
    assert parse_lenient("I think I probably know", LETTERS).label == "B"
    assert parse_lenient("Not sure", LETTERS).valid is False
    assert parse_lenient("B or C", LETTERS).valid is False  # contradictory evidence
    assert parse_lenient("I would say no", YESNO).label == "No"
    assert parse_lenient("Yes and no", YESNO).valid is False
    assert parse_lenient("B", LETTERS).tier == "strict"
    assert parse_lenient("a good guess", LETTERS).valid is False  # the article is not a label


def test_mcq_tiers():
    labels = ("A", "B", "C", "D")
    assert (parse_mcq("B", labels).label, parse_mcq("B", labels).tier) == ("B", "strict")
    assert (parse_mcq("The answer is C.", labels).label, parse_mcq("The answer is C.", labels).tier) == ("C", "phrase")
    assert parse_mcq("The correct answer is (D) Neon", labels).label == "D"
    assert (parse_mcq("I think a good guess is B", labels).label, parse_mcq("I think a good guess is B", labels).tier) == ("B", "loose")
    assert parse_mcq("A good guess is B", labels).label == "B"
    for reply, reason in [("B or C", "multiple"), ("A or B", "multiple"), ("A, B", "multiple"), ("The answer is B or C", "multiple"),
                          ("The answer is not B", "negated"), ("B is wrong", "negated"), ("Not B", "negated")]:
        p = parse_mcq(reply, labels)
        assert (p.label, p.reason) == (None, reason), reply  # never rescued by a looser tier
    assert parse_mcq("B, not C", labels).label == "B" and parse_mcq("The answer is B, not C", labels).label == "B"
    assert parse_mcq("I do not know", labels).label is None
    assert parse_mcq("", labels).label is None
    assert mcq_label("The answer is C.", labels, "strict") is None
    assert mcq_label("C) Paris", labels, "strict") == "C"
    assert parse_mcq("Answer: J", ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")).label == "J"


def test_legacy_yes_matches_the_original_everywhere():
    corpus = ["Yes", "No", "yes.", "I know", "Yesterday", "No, yes", "NO", "Do you know? Yes/No", "", "eyes", "Maybe"]
    for reply in corpus:
        assert legacy_yes(reply) == meta_yes([reply])[0]
    # Substring test: "eyes" and "Yesterday" are "yes", and every reply without "yes" is a "no" (including non-answers).
    assert legacy_yes("eyes") == 1 and legacy_yes("Yesterday") == 1 and legacy_yes("I'm not sure") == 0


def test_legacy_graded_matches_the_original_and_shows_its_defects():
    corpus = ["A", "B) text", "C.", "D", "I'm not sure", "Not sure", "I think a good guess", "certain", "probably", "no idea",
              "I do not know", "", "random words", "I'd say C", "Confident", "a", "Based on my knowledge, B"]
    for reply in corpus:
        grades, ok = parse_graded_response_strict([reply])
        assert legacy_graded(reply) == (grades[0], ok[0]), reply
    # Observed behaviour of the original regex (no boundary after the letter): word-initial a-d is read as a label.
    assert legacy_graded("certain") == (1, True)  # the "c" of "certain" is grade 1, the opposite meaning
    assert legacy_graded("I think a good guess") == (3, True)  # the article "a"
    assert legacy_graded("I am uncertain") == (3, True)  # the "a" of "am"
    assert legacy_graded("I'd say C") == (0, True)  # the "d" of "I'd"
    assert legacy_graded("Not sure") == (3, False)  # fallback finds "sure" before "not sure": certain
    assert legacy_graded("Yes") == (1, False)  # imputed
    assert legacy_graded("random words")[1] is False


def test_legacy_mmlu_scorer_matches_the_original_and_misreads_the_article():
    corpus = ["B", "The answer is C", "I think a good guess is B", "a", "None", "", "D) Neon", "b", "answer: d"]
    for reply in corpus:
        for gold in "ABCD":
            assert legacy_mmlu_correct(reply, gold) == correctness_mmlu([reply], [[gold]])[0]
    assert legacy_mmlu_letter("I think a good guess is B") == "A"
    assert legacy_mmlu_letter("Probably the best choice") is None


# --- methodology audit, 2026-10-27: the article, the pronoun and formula text in direct answers ---

FOUR = ("A", "B", "C", "D")
TEN = tuple("ABCDEFGHIJ")


@pytest.mark.parametrize("reply,labels,label", [
    ("C) a base pair is either inserted or deleted in a gene", FOUR, "C"),  # real Mistral replies, all scored wrong before
    ("B) a budget surplus.", FOUR, "B"),
    ("D) A jury finding beyond a reasonable doubt.", FOUR, "D"),
    ("D) A decrease in the United States trade deficit", FOUR, "D"),
    ("B) a duplication.\n\nThe most common cystic fibrosis mutation is a three-base pair duplication in", FOUR, "B"),
    ("F) E= E_0 (i_x +i_y), B= -B_0i_z", TEN, "F"),  # formula text after a marker is not a second label
    ("H) e^-(1/2)t sin (1/2) t V", TEN, "H"),
])
def test_text_after_an_option_marker_is_not_a_second_label(reply, labels, label):
    assert parse_mcq(reply, labels).label == label


@pytest.mark.parametrize("reply", ["B or C", "B, C", "A, B, C, D", "A or B", "B/C", "B & C", "B) C", "C) or D)"])
def test_two_labels_stay_ambiguous(reply):
    assert parse_mcq(reply, FOUR).reason == "multiple"


def test_the_article_still_cannot_start_an_answer():
    assert parse_strict("A good guess is B", answer_scale_of(FOUR)).reason == "article"
    assert parse_mcq("A good guess is B", FOUR).label == "B"  # the phrase tier reads "is B"? no phrase: loose finds the one capital letter
    assert parse_mcq("A or B", FOUR).reason == "multiple"


def test_the_pronoun_i_is_not_option_i():
    assert parse_mcq("I think the answer is D", TEN).label == "D"
    assert parse_mcq("I think the answer is D", TEN).tier == "phrase"
    for reply in ["I do not know", "i don't know", "I would say it is hard to tell"]:
        assert parse_mcq(reply, TEN).valid is False, reply
    for reply, tier in [("I", "strict"), ("I)", "strict"), ("(I)", "strict"), ("I. Option text", "strict"), ("The answer is I", "phrase"),
                        ("Option I is correct", "loose")]:
        p = parse_mcq(reply, TEN)
        assert (p.label, p.tier) == ("I", tier), reply
    assert parse_mcq("I'd say B", TEN).label == "B"


OPTIONS = ("a thing that thinks", "is not always hate speech, because it does not involve violence", "A fusion reaction is created", "None of the above")


def test_option_echo_rescues_a_marked_restatement_of_the_option():
    reply = "B) is not always hate speech, because it does not involve violence"
    assert parse_mcq(reply, FOUR).reason == "negated"  # without the option texts the denial heuristic cannot tell
    p = parse_mcq(reply, FOUR, OPTIONS)
    assert (p.label, p.tier) == ("B", "echo")
    cut = parse_mcq("B) is not always hate speech, because it does not invo", FOUR, OPTIONS)  # cut off at the token limit
    assert (cut.label, cut.tier) == ("B", "echo")


def test_option_echo_refuses_what_it_cannot_verify():
    assert parse_mcq("B is not correct", FOUR, OPTIONS).reason == "negated"  # no explicit marker
    assert parse_mcq("B) is not correct", FOUR, OPTIONS).reason == "negated"  # the text restates no option
    swapped = ("is wrong", "A fusion reaction is created", "is not always hate speech", "None of the above")
    assert parse_mcq("B) is not always hate speech", FOUR, swapped).reason == "negated"  # restates option C, not B
    assert parse_mcq("B) (a) 1/8, (b) 1/4", FOUR, ("x", "(a) 1/2, (b) 1/4", "y", "z")).reason == "multiple"


def test_the_echo_rescue_does_not_turn_a_wrong_answer_into_a_right_one():
    from probe_validity.scoring import score_mcq
    assert score_mcq("D) A decrease in the United States trade deficit", "A", FOUR).correct == 0
    assert score_mcq("C) a base pair is either inserted or deleted in a gene", "C", FOUR).correct == 1
    assert score_mcq("B) is not always hate speech, because it does not involve violence", "B", FOUR, options=OPTIONS).correct == 1


def answer_scale_of(labels):
    from probe_validity.parse import answer_scale
    return answer_scale(labels)
