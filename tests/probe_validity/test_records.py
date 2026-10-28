import json
import math

import numpy as np

from probe_validity.prompts import Item
from probe_validity.records import ambiguous_labels, build_cell, direct_outcomes, first_token_score, grades, read_condition
from probe_validity.scales import DIGITS, LETTERS, YESNO

OPTS = ("w", "x", "y", "z")


def mcq(i, gold, options=OPTS, group="g"):
    return Item(qid=f"q{i}", stem="stem", options=options, gold=gold, group=group)


def write(directory, name, replies, label_lp=None, only=None):
    with (directory / f"{name}.jsonl").open("w") as fh:
        for i, text in enumerate(replies):
            if only is not None and i not in only:
                continue
            row = {"i": i, "qid": f"q{i}", "condition": name, "text": text, "n_new": 2, "finish": "eos"}
            if label_lp:
                row["label_lp"] = label_lp[i]
            fh.write(json.dumps(row) + "\n")


def test_direct_outcomes_for_multiple_choice(tmp_path):
    items = [mcq(0, "B"), mcq(1, "C"), mcq(2, "A"), mcq(3, "D", options=("w", "x", "y"))]
    lp = [{"A": -3.0, "B": -0.1, "C": -4.0, "D": -5.0}] * 3 + [{"A": -1.0, "B": -1.0, "C": -1.0, "D": -0.5}]
    write(tmp_path, "direct_clean", ["B", "The answer is C.", "I do not know", "A"], lp)
    d = direct_outcomes(read_condition(tmp_path, "direct_clean", 4), items)
    assert d.correct.tolist() == [1, 1, 0, 0] and d.valid.tolist() == [True, True, False, True]
    assert d.label == ["B", "C", None, "A"]
    assert math.isnan(d.reference[2])
    z = math.log(sum(math.exp(v) for v in (-3.0, -0.1, -4.0, -5.0)))
    assert abs(d.reference[0] - (-0.1 - z)) < 1e-9
    z3 = math.log(3 * math.exp(-1.0))  # the third item has three options: D is not one of its labels
    assert abs(d.reference[3] - (-1.0 - z3)) < 1e-9


def test_direct_outcomes_for_open_questions(tmp_path):
    items = [Item(qid=f"q{i}", stem="s", answers=("Paris",)) for i in range(3)]
    path = tmp_path / "direct_clean.jsonl"
    rows = [{"i": 0, "condition": "direct_clean", "text": "Paris", "span_lp": -0.5, "span_n": 1},
            {"i": 1, "condition": "direct_clean", "text": "Paris, London, Rome", "span_lp": -4.0, "span_n": 4},
            {"i": 2, "condition": "direct_clean", "text": "", "span_lp": 0.0, "span_n": 0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    d = direct_outcomes(read_condition(tmp_path, "direct_clean", 3), items)
    assert d.correct.tolist() == [1, 0, 0] and d.valid.tolist() == [True, True, False]
    assert d.reference[0] == -0.5 and d.reference_mean[1] == -1.0 and math.isnan(d.reference[2])


def test_grades_are_semantic_and_invalid_is_nan(tmp_path):
    write(tmp_path, "letters_raw", ["A) sure", "D", "I'd say B", "C"], only={0, 1, 2, 3})
    g = grades(read_condition(tmp_path, "letters_raw", 5), LETTERS)
    assert g[:4].tolist()[:2] == [3.0, 0.0] and math.isnan(g[2]) and g[3] == 1.0 and math.isnan(g[4])  # row 4 is missing
    write(tmp_path, "digits_raw", ["1", "4", "2", "x"])
    assert grades(read_condition(tmp_path, "digits_raw", 4), DIGITS)[:3].tolist() == [3.0, 0.0, 2.0]


def test_first_token_score_is_the_expected_grade(tmp_path):
    lp = [{"Yes": math.log(0.75), "No": math.log(0.05)}, {"Yes": -50.0, "No": -50.0}]
    write(tmp_path, "yesno_raw", ["Yes", "No"], lp)
    score, mass = first_token_score(read_condition(tmp_path, "yesno_raw", 2), YESNO)
    assert abs(score[0] - 0.75 / 0.80) < 1e-9 and abs(mass[0] - 0.80) < 1e-9
    assert abs(score[1] - 0.5) < 1e-9


def test_build_cell_collects_mappings_draws_and_strata(tmp_path):
    n = 6
    items = [mcq(i, "A", group="a" if i < 3 else "b") for i in range(n)]
    write(tmp_path, "direct_clean", ["A", "B", "A", "A", "C", "A"], [{"A": -0.1, "B": -3.0, "C": -3.0, "D": -3.0}] * n)
    write(tmp_path, "letters_raw", ["A", "B", "C", "D", "A", "B"])
    for name, replies in {"letters_reversed_raw": ["D", "C", "B", "A", "D", "C"], "letters_perm_raw": ["C", "A", "D", "B", "C", "A"],
                          "digits_raw": ["1", "2", "3", "4", "1", "2"], "words_raw": ["CERTAIN", "PROBABLY", "UNSURE", "NO IDEA", "CERTAIN", "PROBABLY"]}.items():
        write(tmp_path, name, replies)
    for d in range(2):
        write(tmp_path, f"letters_sampled{d}", ["A", "B", "C"], only={0, 1, 2})
    direct = direct_outcomes(read_condition(tmp_path, "direct_clean", n), items)
    cell = build_cell(tmp_path, items, "letters", ("A", "B", "C", "D"), direct, draws=2)
    assert cell.grade.tolist() == [3, 2, 1, 0, 3, 2]
    assert set(cell.mappings) == {"letters_reversed", "letters_perm", "digits", "words"}
    for mapping in cell.mappings.values():  # the same semantic grades under every label mapping
        assert mapping.tolist() == cell.grade.tolist()
    assert cell.sampled.shape == (n, 2) and np.isnan(cell.sampled[3:]).all() and cell.sampled[:3, 0].tolist() == [3, 2, 1]
    assert cell.strata.tolist() == list("aaabbb") and cell.correct.tolist() == [1, 0, 1, 1, 0, 1]
    assert not np.isnan(cell.reference).any()


def test_labels_that_share_a_first_token_are_flagged(tmp_path):
    from probe_validity.scales import WORDS

    shared = [{"CERTAIN": -1.0, "PROBABLY": -2.0, "UNSURE": -3.0, "NO IDEA": -3.0}] * 3  # UNSURE and NO IDEA start with one token
    write(tmp_path, "words_raw", ["CERTAIN"] * 3, shared)
    assert ambiguous_labels(read_condition(tmp_path, "words_raw", 3), WORDS) == [("UNSURE", "NO IDEA")]
    distinct = [{"CERTAIN": -1.0, "PROBABLY": -2.0, "UNSURE": -3.0, "NO IDEA": -4.0}] * 3
    write(tmp_path, "words_raw", ["CERTAIN"] * 3, distinct)
    assert ambiguous_labels(read_condition(tmp_path, "words_raw", 3), WORDS) == []


def test_stem_cells_use_the_stem_scales_as_their_mappings(tmp_path):
    n = 4
    items = [mcq(i, "A") for i in range(n)]
    write(tmp_path, "direct_clean", ["A", "A", "B", "A"], [{"A": -0.1, "B": -3.0, "C": -3.0, "D": -3.0}] * n)
    write(tmp_path, "letters_stem", ["A", "B", "C", "D"])
    write(tmp_path, "digits_stem", ["1", "2", "3", "4"])
    write(tmp_path, "words_stem", ["CERTAIN", "PROBABLY", "UNSURE", "NO IDEA"])
    write(tmp_path, "letters_perm_raw", ["C", "A", "D", "B"])  # a raw-only scale must not leak into the stem cell
    direct = direct_outcomes(read_condition(tmp_path, "direct_clean", n), items)
    cell = build_cell(tmp_path, items, "letters", (), direct, variant="stem")
    assert set(cell.mappings) == {"digits", "words"} and cell.sampled is None
    assert cell.grade.tolist() == [3, 2, 1, 0] and cell.mappings["digits"].tolist() == [3, 2, 1, 0]
