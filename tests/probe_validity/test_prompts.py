import pytest

from esma import prompt as legacy
from probe_validity.prompts import (
    DIRECT_OPEN,
    FORMAT_PHRASES,
    META_YESNO,
    Item,
    assert_no_format_instruction,
    direct_prompt,
    graded_prompt,
    meta_prompt,
)
from probe_validity.scales import (
    CONTROL_SCALES,
    DESCRIPTORS,
    DIGITS,
    FACTORIAL,
    LETTERS,
    LETTERS_PERM,
    LETTERS_REVERSED,
    SCALES,
    WORDS,
    YESNO,
    label_overlap,
)

STEM = "Which gas do plants absorb?"
OPTIONS = ("Oxygen", "Carbon dioxide", "Helium", "Neon")


def mcq() -> Item:
    return Item(qid="q1", stem=STEM, options=OPTIONS, gold="B", group="biology")


def test_templates_equal_legacy_character_for_character():
    assert DIRECT_OPEN == legacy.DIRECT_QA_PROMPT
    assert META_YESNO == legacy.META_QA_PROMPT
    q = "What is the capital of France?"
    assert graded_prompt(q, LETTERS) == legacy.GRADED_META_QA_PROMPT.format(question=q)


def test_legacy_mmlu_question_string_is_reproduced():
    # esma/data/mmlu.py: instruction + question, then "\nA) text" per option.
    instruction = "Answer the following multiple-choice question. Respond with ONLY the letter (A, B, C, or D) of the correct option.\n\n"
    expected = instruction + STEM + "".join(f"\n{c}) {t}" for c, t in zip("ABCD", OPTIONS))
    assert mcq().legacy_question() == expected


def test_embedded_variant_reproduces_the_leak_and_raw_does_not():
    item = mcq()
    embedded = meta_prompt(item, "yesno", "embedded")
    assert embedded == legacy.META_QA_PROMPT.format(question=item.legacy_question())
    assert "Respond with ONLY the letter" in embedded
    raw = meta_prompt(item, "yesno", "raw")
    assert not any(p in raw for p in FORMAT_PHRASES)
    assert STEM in raw and "B) Carbon dioxide" in raw


def test_assert_no_format_instruction_catches_the_leak():
    with pytest.raises(AssertionError):
        assert_no_format_instruction(mcq().legacy_question())
    assert_no_format_instruction(mcq().content())


def test_direct_prompt_for_mcq_has_one_instruction():
    text = direct_prompt(mcq())
    assert text.count("Respond with ONLY the letter") == 1
    assert "Answer the following question with keywords" not in text


def test_open_item_prompts():
    item = Item(qid="t1", stem="Who wrote Hamlet?", answers=("William Shakespeare",))
    assert direct_prompt(item) == legacy.DIRECT_QA_PROMPT.format(question="Who wrote Hamlet?")
    assert meta_prompt(item, "yesno") == legacy.META_QA_PROMPT.format(question="Who wrote Hamlet?")


def test_letters_scale_lines_equal_legacy():
    legacy_lines = legacy.GRADED_META_QA_PROMPT.split("Return ONLY the letter.\n")[1].strip().split("\n")
    assert LETTERS.lines() == legacy_lines


@pytest.mark.parametrize("scale", list(SCALES.values()), ids=lambda s: s.name)
def test_descriptors_move_with_their_meaning(scale):
    lines = scale.lines()
    assert len(lines) == 4
    for meaning, descriptor in enumerate(DESCRIPTORS):
        label = scale.labels[meaning]
        assert any(line.startswith(label) and line.endswith(descriptor) for line in lines)
        assert scale.grade_of(label) == 3 - meaning


def test_reversed_and_permuted_scales_are_real_relabellings():
    assert LETTERS_REVERSED.lines()[0].startswith("A) I definitely do not know")
    assert LETTERS_REVERSED.grade_of("A") == 0 and LETTERS_REVERSED.grade_of("D") == 3
    assert LETTERS_PERM.grade_of("C") == 3 and LETTERS_PERM.grade_of("B") == 0
    assert DIGITS.grade_of("1") == 3 and WORDS.grade_of("NO IDEA") == 0
    assert LETTERS.grade_of("E") is None and LETTERS.grade_of(None) is None


def test_label_collision_is_a_pure_function_of_the_label_sets():
    answers = ("A", "B", "C", "D")
    assert label_overlap(LETTERS, answers) == ("A", "B", "C", "D")
    assert label_overlap(LETTERS_REVERSED, answers) == ("D", "C", "B", "A")
    assert label_overlap(DIGITS, answers) == () and label_overlap(WORDS, answers) == ()
    assert label_overlap(YESNO, answers) == ()
    assert label_overlap(LETTERS, ()) == ()


def test_graded_prompt_contains_all_descriptors_once():
    text = meta_prompt(mcq(), "graded", "raw", scale=DIGITS)
    for descriptor in DESCRIPTORS:
        assert text.count(descriptor) == 1
    assert "Return ONLY the number." in text


def test_stem_variant_shows_no_options_and_no_instruction():
    item = mcq()
    for probe, scale in (("yesno", None), ("graded", DIGITS)):
        text = meta_prompt(item, probe, "stem", scale)
        assert STEM in text and "Carbon dioxide" not in text and "B)" not in text.replace("B) I think", "")
        assert not any(p in text for p in FORMAT_PHRASES)
    with pytest.raises(ValueError):
        meta_prompt(item, "yesno", "nonsense")


def test_label_and_order_are_crossed_by_the_two_controls():
    from probe_validity.scales import PROBE_SCALES

    cells = {name: PROBE_SCALES[name] for name in FACTORIAL}
    first_meaning = {}
    for name, scale in cells.items():
        lines = scale.lines()
        first_label = lines[0].split(")")[0]
        # the meaning printed on the first line (3 = certain) and the label that carries "certain"
        first_meaning[name] = (scale.grade_of(first_label), scale.labels[0])
        assert len(lines) == 4 and sorted(l.split(") ", 1)[1] for l in lines) == sorted(DESCRIPTORS)  # same four lines, any order
    assert first_meaning == {
        "letters": (3, "A"),  # certain printed first, carried by A
        "letters_flip_order": (0, "A"),  # certain printed last, still carried by A: order changed, labels kept
        "letters_flip_labels": (3, "D"),  # certain printed first, now carried by D: labels changed, order kept
        "letters_reversed": (0, "D"),  # both changed
    }
    # a responder that always takes the first printed line gets a different grade only where the order changed
    assert [cells[n].grade_of(cells[n].lines()[0].split(")")[0]) for n in FACTORIAL] == [3, 0, 3, 0]
    # a responder that always takes the label "A" gets a different grade only where the labels changed
    assert [cells[n].grade_of("A") for n in FACTORIAL] == [3, 3, 0, 0]
    assert set(CONTROL_SCALES) == {"letters_flip_order", "letters_flip_labels"}


def test_existing_scale_lines_are_unchanged_by_the_presentation_option():
    assert WORDS.lines()[0] == "CERTAIN: I am certain I know the answer"
    assert LETTERS_REVERSED.lines()[0] == "A) I definitely do not know"
    assert LETTERS_PERM.lines()[0] == "A) I think I probably know"


def test_answer_label_control_relabels_only_the_options():
    item = mcq()
    letters = meta_prompt(item, "graded", "optdigits", LETTERS)
    assert "1) Oxygen\n2) Carbon dioxide\n3) Helium\n4) Neon" in letters
    assert "A) Oxygen" not in letters and "A) I am certain I know the answer" in letters  # the confidence scale is untouched
    assert not any(p in letters.split("\n\nHow confident")[0] for p in FORMAT_PHRASES)  # the question content carries no instruction
    assert label_overlap(LETTERS, tuple("1234")) == ()  # no shared label left
    digits = meta_prompt(item, "graded", "optdigits", DIGITS)
    assert "1) Oxygen" in digits and "1) I am certain I know the answer" in digits  # digits against digit-labelled options collide
    assert meta_prompt(item, "yesno", "optdigits").count("1) Oxygen") == 1
    assert item.content(("1", "2", "3", "4")).startswith(STEM + "\n1) Oxygen")
    with pytest.raises(ValueError):
        item.content(("1", "2"))
    ten = Item(qid="q", stem="s", options=tuple("abcdefghij"), gold="A")
    assert "\n10) j" in meta_prompt(ten, "yesno", "optdigits")  # the technical sets have up to ten options: labels 1) to 10)
    eleven = Item(qid="q", stem="s", options=tuple("abcdefghijk"), gold="A")
    with pytest.raises(ValueError):
        meta_prompt(eleven, "yesno", "optdigits")


def test_the_wording_changes_the_probe_text_and_nothing_else_and_defaults_to_the_legacy_text():
    """PROBE_WORDING is read once at import, so each wording is checked in its own process: w1 is the legacy prompt, w2 and w3 differ in the
    question of the probe and in the descriptors, and the four meanings, the labels and the response-format sentence stay."""
    import json
    import os
    import subprocess
    import sys

    code = ("import json; from probe_validity.prompts import Item, meta_prompt; from probe_validity.scales import LETTERS, WORDING; "
            "it = Item(qid='q', stem='S', options=('a', 'b', 'c', 'd'), gold='A'); "
            "print(json.dumps({'w': WORDING, 'yn': meta_prompt(it, 'yesno', 'raw'), 'gr': meta_prompt(it, 'graded', 'raw', LETTERS), 'lines': LETTERS.lines()}))")
    out = {}
    for w in ("w1", "w2", "w3"):
        env = {**os.environ, "PROBE_WORDING": w}
        out[w] = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout)
    legacy = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={k: v for k, v in os.environ.items() if k != "PROBE_WORDING"})
    assert json.loads(legacy.stdout)["w"] == "w1"  # no variable: the study's wording
    assert len({out[w]["yn"] for w in out}) == 3 and len({out[w]["gr"] for w in out}) == 3
    assert "I am certain I know the answer" in out["w1"]["gr"] and "I know the answer for sure" in out["w2"]["gr"] and "almost certainly be correct" in out["w3"]["gr"]
    for w in out:  # four lines, the same labels in the same order, the probe's own response-format sentence kept
        assert [ln.split(")")[0] for ln in out[w]["lines"]] == ["A", "B", "C", "D"] and "Return ONLY the letter." in out[w]["gr"]
    bad = subprocess.run([sys.executable, "-c", "import probe_validity.scales"], capture_output=True, text=True, env={**os.environ, "PROBE_WORDING": "w9"})
    assert bad.returncode != 0 and "PROBE_WORDING" in bad.stderr  # an unknown wording is an error, never a silent default
