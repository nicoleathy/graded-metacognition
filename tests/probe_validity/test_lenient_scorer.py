"""The sensitivity scorer for open answers: it relaxes three things and keeps the refusals, and it is never switched on by accident."""

import pytest

from probe_validity.records import open_scorer
from probe_validity.scoring import score_open, score_open_lenient


@pytest.mark.parametrize("reply,aliases", [
    ("Spanish", ["Spanish Language"]),  # a trailing generic word of the accepted answer
    ("swedish", ["Yiddish Language", "Swedish Language"]),
    ("Capital: Venezuela\nCountry: Venezuela\nAnswer: Caracas", ["Caracas"]),  # the answer is on a later line
    ("Bird, jazz saxophonist, nickname, Charlie Parker", ["Charlie Parker"]),  # the first span is a keyword list that holds the alias
    ("Ringgit", ["Malaysian ringgit"]),  # a short reply that is part of the accepted answer
    ("The word is Paris (France)", ["Paris [city]"]),  # a bracketed qualifier of the alias
])
def test_the_three_relaxations_credit_what_the_frozen_scorer_rejects(reply, aliases):
    assert score_open(reply, aliases).correct == 0 and score_open_lenient(reply, aliases).correct == 1


@pytest.mark.parametrize("reply,aliases", [
    ("I'm not sure, maybe Paris", ["Paris"]),  # a reply that withholds an answer is still not an answer
    ("Not Paris; London", ["Paris"]),  # negation and retraction still count
    ("Berlin", ["Paris"]),
    ("Thomas Jefferson wrote it", ["United States Declaration of Independence"]),  # no spelling or alias guessing
    ("A thing", ["A"]),  # a short or common single-token alias is not searched for in the whole reply
    ("of", ["United States of America"]),  # a common word is never part of an answer
])
def test_refusals_negations_and_wrong_answers_stay_incorrect(reply, aliases):
    assert score_open_lenient(reply, aliases).correct == 0


def test_the_lenient_scorer_never_rejects_what_the_frozen_scorer_accepts():
    for reply, aliases in [("Paris", ["Paris"]), ("It is Paris.", ["Paris"]), ("William Shakespeare", ["William Shakespeare", "Shakespeare"]), ("1999", ["1999"])]:
        assert score_open(reply, aliases).correct == 1 and score_open_lenient(reply, aliases).correct == 1


def test_the_scorer_is_the_frozen_one_unless_the_variable_says_otherwise(monkeypatch):
    monkeypatch.delenv("OPEN_SCORER", raising=False)
    assert open_scorer() is score_open
    monkeypatch.setenv("OPEN_SCORER", "lenient")
    assert open_scorer() is score_open_lenient
    monkeypatch.setenv("OPEN_SCORER", "loose")
    with pytest.raises(ValueError):
        open_scorer()
