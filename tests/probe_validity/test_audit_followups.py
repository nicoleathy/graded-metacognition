"""Regression tests for the defects an independent review found after the methodology audit, and pins for behaviour that mutations had left unnoticed."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

from probe_validity import metrics as M
from probe_validity import report
from probe_validity.parse import parse_mcq, parse_strict
from probe_validity.prompts import Item
from probe_validity.records import direct_outcomes
from probe_validity.scales import DIGITS, WORDS, YESNO
from probe_validity.scoring import score_open
from probe_validity.screen import Check, ScreenConfig, screen
from probe_validity.simulators import simulate

FOUR, TEN = tuple("ABCD"), tuple("ABCDEFGHIJ")
CFG = ScreenConfig()
OPTIONS = ("a thing that thinks", "is not always hate speech, because it does not involve violence", "A fusion reaction is created", "None of the above")


# --- parser --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", ["ı", "İyi", "İ", "i̇", "ı think"])
def test_case_folding_lookalikes_of_a_label_never_crash_the_parser(reply):
    assert parse_mcq(reply, TEN).label in (None, "I")  # dotless and dotted capital i fold to the label I in a pattern; nothing may raise
    assert parse_mcq("İyi", TEN).label is None


@pytest.mark.parametrize("reply", ["B or C?", "B or C…", "B or C.", "B or C", "B / C?", "A, B?", "B or C!"])
def test_a_second_label_is_found_whatever_punctuation_ends_it(reply):
    assert parse_mcq(reply, FOUR).reason == "multiple"


def test_the_same_holds_on_every_scale():
    for reply, scale in [("Yes or No?", YESNO), ("2 or 3?", DIGITS), ("CERTAIN or PROBABLY?", WORDS), ("No or Yes…", YESNO)]:
        assert parse_strict(reply, scale).reason == "multiple", reply


@pytest.mark.parametrize("reply", ["*I'm not sure*", "I'M NOT SURE", "I THINK the answer is unclear", "I Think it is hard", "**I do not know**"])
def test_the_pronoun_i_is_not_option_i_in_any_case_or_markdown(reply):
    assert parse_mcq(reply, TEN).valid is False


def test_a_pronoun_before_a_named_option_still_gives_the_option():
    assert parse_mcq("I THINK D", TEN).label == "D" and parse_mcq("I'd say B", TEN).label == "B"


@pytest.mark.parametrize("reply", [
    "To find the conductances (h) of the cylinder we use the equation:", "Q = A * v", "The temperature is 49°F in this case",
    "Let us compute the heat flow through the wall using the formula for a plane wall, where A is the area",
    "ΔS = ∫C_p dT/T where T is the temperature and B is the constant of integration",
])
def test_derivation_text_is_not_an_answer(reply):
    assert parse_mcq(reply, TEN).valid is False


@pytest.mark.parametrize("reply,label", [("Option B is correct", "B"), ("I would go with C", "C"), ("Probably C", "C"), ("The best choice is (D)", "D"), ("It should be D", "D")])
def test_a_short_statement_that_names_an_option_is_an_answer(reply, label):
    assert parse_mcq(reply, TEN).label == label


def test_the_loose_tier_never_resolves_a_denial_or_a_conflict():
    assert parse_mcq("Not B", FOUR).reason == "negated" and parse_mcq("The answer is not B", FOUR).reason == "negated"
    assert parse_mcq("It is either B or C", FOUR).valid is False


def test_the_echo_rescue_needs_three_words_an_explicit_marker_and_no_rival_option():
    assert parse_mcq("B) is not", FOUR, OPTIONS).reason == "negated"  # two matching words are too few
    assert parse_mcq("B is not always hate speech, because it does not invo", FOUR, OPTIONS).reason == "negated"  # no explicit marker: a denial
    rival = ("a", "is not always hate speech", "is not always hate speech or worse", "d")
    assert parse_mcq("B) is not always hate speech", FOUR, rival).reason == "negated"  # the text restates option C as well: nothing is rescued
    assert parse_mcq("B) is not always hate speech, because it does not invo", FOUR, OPTIONS).tier == "echo"  # cut off at the token limit


# --- open answers -------------------------------------------------------------------------------------------------------------------

CASES = [
    ("Leonardo da Vinci, the Italian polymath, painted the Mona Lisa.", ["Leonardo da Vinci"], 1), ("Paris, the capital of France, is on the Seine.", ["Paris"], 1),
    ("It is 1,000,000", ["1,000,000"], 1), ("Washington, D.C., USA", ["Washington, D.C."], 1), ("It's Trick or Treat", ["Trick or Treat"], 1),
    ("I don’t know, maybe Paris.", ["Paris"], 0), ("It isn’t Paris, it is London", ["Paris"], 0), ("The answer is “Paris”.", ["Paris"], 1),
    ("Alexander Fleming discovered penicillin; he had no idea it would save millions.", ["Alexander Fleming"], 0),
    ("Einstein was not right about everything, but he formulated relativity.", ["Einstein"], 1), ("Paris is not the only answer, but it is correct.", ["Paris"], 1),
    ("I am not certain; maybe Paris", ["Paris"], 0), ("Paris is hardly the answer", ["Paris"], 0), ("Paris — no, London", ["Paris"], 0),
    ("Paris? No.", ["Paris"], 0), ("Paris, no doubt", ["Paris"], 1), ("Paris, London, Berlin", ["Paris"], 0), ("Paris, London", ["Paris"], 1),
    ("Paris, France", ["Paris"], 1), ("Paris is not the capital of Germany", ["Paris"], 1), ("Not Paris; London.", ["Paris"], 0),
]


@pytest.mark.parametrize("reply,aliases,expected", CASES)
def test_the_open_scorer_on_cases_from_the_review(reply, aliases, expected):
    assert score_open(reply, aliases).correct == expected, reply


@pytest.mark.parametrize("alias", ["Trick or Treat", "No Country for Old Men", "1,000,000", "Washington, D.C.", "Don’t Stop Believin’", "Not Fade Away", "I Don't Know",
                                   "Hamlet, Prince of Denmark", "Yes or No", "Neither Here Nor There"])
def test_no_rule_fires_on_the_words_of_an_accepted_answer_itself(alias):
    for reply in (alias, f"The answer is {alias}.", f"It's {alias}", f"{alias} is the answer."):
        assert score_open(reply, [alias]).correct == 1, reply


def test_a_clause_that_starts_with_but_ends_a_negation():
    assert score_open("not but London", ["London"]).correct == 1  # without the clause rule the "not" would reach London
    assert score_open("not however London", ["London"]).correct == 1


# --- report, metrics, blocks ----------------------------------------------------------------------------------------------------------

def test_the_newcombe_interval_matches_the_published_example():
    d, lo, hi = report._newcombe(56, 70, 48, 80)  # Newcombe (1998), the Wilson-score method
    assert (round(d, 4), round(lo, 4), round(hi, 4)) == (0.2, 0.0524, 0.3339)


def test_every_sensitivity_move_goes_the_way_it_says():
    higher_is_harder = {"s1_pass", "s1_fail", "s3_min_entropy", "s4_min_kappa", "s5_min_kappa", "s7_min_abs_rho"}
    lower_is_harder = {"s3_max_top_share", "s4_spread_factor"}
    for field, loose, tight in report.SENSITIVITY_MOVES:
        frozen = getattr(CFG, field)
        if field in higher_is_harder:
            assert loose < frozen < tight, field
        elif field in lower_is_harder:
            assert loose > frozen > tight, field
        else:
            assert field == "min_valid" and loose < frozen < tight


def test_the_sensitivity_analysis_carries_the_s8_verdict_into_every_variant():
    cell = simulate("oracle", 0)
    assert screen(cell, CFG)["overall"] == "PASS"
    for s8, expected in (("PASS", "PASS"), ("WARN", "WARN")):
        s = report.threshold_sensitivity(cell, CFG, s8=s8)
        assert s["frozen"] == expected and s["all_loose"] == expected


def test_s4_on_own_pairs_judges_only_the_own_kappa():
    assert report._s4_own(Check("S4", "FAIL", {"own_min_kappa": 0.3}), CFG) == "FAIL"
    assert report._s4_own(Check("S4", "FAIL", {"own_min_kappa": 0.7, "auroc_spread": 9.0, "auroc_half_width": 0.01}), CFG) == "PASS"  # the family's spread does not decide
    assert report._s4_own(Check("S4", "PASS", {"own_min_kappa": None}), CFG) == "FAIL"
    assert report._s4_own(Check("S4", "NA", {}), CFG) == "NA"


def test_the_stratified_bootstrap_keeps_every_stratum_size_fixed():
    strata = np.array([0] * 30 + [1] * 70)
    share = lambda idx: float(np.mean(strata[idx] == 0))  # noqa: E731
    fixed = M.bootstrap_stratified(share, strata, n_boot=60, seed=1)
    assert fixed["lo"] == fixed["hi"] == pytest.approx(0.3)
    free = M.bootstrap(share, len(strata), n_boot=60, seed=1)
    assert free["hi"] > free["lo"]


def test_corrected_cramers_v_removes_the_bias_of_small_tables():
    rng = np.random.default_rng(0)
    draws = [M.cramers_v(rng.integers(0, 10, 324), rng.integers(0, 4, 324)) for _ in range(150)]  # independent variables: a ten-by-four table at n = 324
    assert np.mean([d["v"] for d in draws]) > 0.14 and np.mean([d["v_corrected"] for d in draws]) < 0.06
    big = [M.cramers_v(rng.integers(0, 4, 2000), rng.integers(0, 4, 2000)) for _ in range(50)]
    assert np.mean([d["v"] for d in big]) < 0.06  # the uncorrected V of a small table at large n is already near zero: raw values are not comparable across tables
    assert M.cramers_v(list("ABCD") * 50, list("ABCD") * 50)["v_corrected"] == pytest.approx(1.0, abs=0.01)


def test_refit_paired_diff_refits_the_transform_in_every_resample():
    import blocks

    sizes = []

    def transform(score, verbal):
        sizes.append(len(score))
        return score

    rng = np.random.default_rng(2)
    c = (rng.random(200) < 0.5).astype(int)
    f = c * 0.5 + rng.random(200)
    v = np.digitize(f, [0.5, 1.0]).astype(float)
    out = blocks.refit_paired_diff(c, f, v, transform, seed=0, n_boot=25)
    assert len(sizes) == 26 and out["refit_in_bootstrap"] is True  # the point estimate and one fit per resample


# --- the S7 reference needs a label that is the first token ---------------------------------------------------------------------------

def test_the_answer_likelihood_reference_is_used_only_when_the_label_is_the_first_token():
    item = Item(qid="q", stem="s", options=("w", "x", "y", "z"), gold="B")
    lp = {"A": -3.0, "B": -0.1, "C": -4.0, "D": -5.0}
    rows = [{"text": "B", "label_lp": lp}, {"text": "The answer is B", "label_lp": lp}, {"text": "B) x", "label_lp": lp}]
    d = direct_outcomes(rows, [item] * 3)
    assert np.isfinite(d.reference[0]) and np.isnan(d.reference[1]) and np.isfinite(d.reference[2])  # a phrase answer's first token is not the label


# --- run_sweep.main: DONE, bookkeeping files, batch size, hashes ---------------------------------------------------------------------------

@pytest.fixture()
def sweep(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import run_sweep

    class FakeRunner:
        device = "cpu"

        def __init__(self, model_id):
            pass

        def meta(self, max_new_tokens):
            return {"model_id": "toy", "revision": "r1", "device": FakeRunner.device, "dtype": "f32", "attention": "eager", "torch": "t", "transformers": "x",
                    "chat_template_sha256": "c", "padding_side": "left", "generation_config": {"do_sample": False}}

        def effective_config(self, max_new_tokens, sampling=None):
            return {"do_sample": sampling is not None}

        def plan(self, prompts, batch_size):
            return [list(range(i, min(i + batch_size, len(prompts)))) for i in range(0, len(prompts), batch_size)]

        def generate_batch(self, prompts, max_new_tokens, labels, sampling, seed=0):
            return [{"text": "Yes", "n_new": 1, "finish": "eos"} for _ in prompts]

    items = [Item(qid=f"q{i}", stem=f"question {i}") for i in range(6)]
    conds = [types.SimpleNamespace(name=n, build=lambda it: it.stem, labels=(), max_new_tokens=32, sampling=None, seed=0, subset=None) for n in ("c1", "c2")]
    monkeypatch.setattr(run_sweep, "Runner", FakeRunner)
    monkeypatch.setattr(run_sweep, "load_items", lambda name, revision=None: (items, {"revision": "rev"}))
    monkeypatch.setattr(run_sweep, "load_manifest", lambda path: {"name": "toy", "sha256": "m", "source": {"revision": "rev"}})
    monkeypatch.setattr(run_sweep, "select", lambda items, manifest: items)
    monkeypatch.setattr(run_sweep, "conditions_for", lambda dataset, suite="sweep": conds)
    monkeypatch.setattr(run_sweep, "DATASET_TABLE", {"toy": ("toy_manifest", (), False)})
    monkeypatch.setattr(run_sweep, "protocol_record", lambda: {"sha256": "p"})
    monkeypatch.setattr(run_sweep, "RUNS", tmp_path)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["run_sweep", "--model", "toy", "--datasets", "toy", "--out", "o", *args])
        run_sweep.main()

    return types.SimpleNamespace(run=run, dir=tmp_path / "o" / "toy" / "toy", runner=FakeRunner)


def records(d):
    return [json.loads(line) for line in (d / "conditions.jsonl").read_text().splitlines()]


def test_a_complete_run_is_marked_done_and_a_noop_resume_does_not_restamp_its_rows(sweep):
    sweep.run("--batch-size", "2")
    assert (sweep.dir / "DONE").exists() and [r["condition"] for r in records(sweep.dir)] == ["c1", "c2"]
    first = records(sweep.dir)
    sweep.run("--batch-size", "2")  # nothing to generate
    assert records(sweep.dir) == first and (sweep.dir / "DONE").exists()


def test_a_restricted_rerun_withdraws_the_done_marker_instead_of_vouching_for_the_directory(sweep):
    sweep.run("--batch-size", "2")
    sweep.run("--batch-size", "2", "--only", "c1")
    assert not (sweep.dir / "DONE").exists() and list(sweep.dir.glob("DONE.superseded.*"))
    sweep.run("--batch-size", "2")  # a full rerun restores it
    assert (sweep.dir / "DONE").exists()


def test_resuming_with_another_batch_size_or_a_limit_is_refused(sweep):
    sweep.run("--batch-size", "2")
    with pytest.raises(SystemExit, match="batch_size"):
        sweep.run("--batch-size", "3")
    with pytest.raises(SystemExit, match="limit"):
        sweep.run("--batch-size", "2", "--limit", "2")


def test_bookkeeping_files_alone_do_not_make_a_directory_resumable_under_other_settings(sweep):
    sweep.run("--batch-size", "2", "--only", "none")  # writes meta.json and no rows
    sweep.run("--batch-size", "2", "--only", "none")  # writes meta_history.jsonl too
    assert (sweep.dir / "meta_history.jsonl").exists() and not list(sweep.dir.glob("c*.jsonl"))
    sweep.runner.device = "mps"  # a different device, but nothing was ever generated here
    sweep.run("--batch-size", "2")
    assert (sweep.dir / "DONE").exists()
    sweep.runner.device = "cpu"
