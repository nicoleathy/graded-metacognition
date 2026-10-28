import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/paper1"))

analysis = pytest.importorskip("analysis")
from probe_validity.prompts import Item  # noqa: E402
from probe_validity.records import direct_outcomes, read_condition  # noqa: E402

N = 400
OPTS = ("w", "x", "y", "z")
LETTER_OF = "ABCD"


def write(directory, name, replies):
    with (directory / f"{name}.jsonl").open("w") as fh:
        for i, text in enumerate(replies):
            fh.write(json.dumps({"i": i, "qid": f"q{i}", "condition": name, "text": text, "n_new": 2, "finish": "eos",
                                 "label_lp": {lab: -1.4 for lab in "ABCD"}}) + "\n")


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, N)
    answer = np.where(rng.random(N) < 0.6, gold, rng.integers(0, 4, N))  # the model's own answer: right 60% of the time and more
    items = [Item(qid=f"q{i}", stem="s", options=OPTS, gold=LETTER_OF[gold[i]], group="g") for i in range(N)]
    write(tmp_path, "direct_clean", [LETTER_OF[a] for a in answer])
    return tmp_path, items, answer, gold, rng


def test_collision_control_separates_lexical_copying_from_scale_format(world):
    tmp, items, answer, gold, rng = world
    copy_when_collide = rng.random(N) < 0.8  # a model that echoes its answer label whenever that label is also a confidence label
    random_letter = lambda: LETTER_OF[rng.integers(0, 4)]  # noqa: E731
    write(tmp, "letters_raw", [LETTER_OF[a] if c else random_letter() for a, c in zip(answer, copy_when_collide)])  # letters/letters: collide
    write(tmp, "letters_optdigits", [random_letter() for _ in range(N)])  # letters against digit options: nothing to echo
    write(tmp, "digits_raw", [str(rng.integers(1, 5)) for _ in range(N)])  # digits against letter options: nothing to echo
    write(tmp, "digits_optdigits", [str(a + 1) if c else str(rng.integers(1, 5)) for a, c in zip(answer, copy_when_collide)])  # collide
    direct = direct_outcomes(read_condition(tmp, "direct_clean", N), items)
    block = analysis.collision_block(tmp, items, direct, n_boot=200)
    cells = block["cells"]
    assert cells["letters/letters"]["collision"] and cells["digits/digits"]["collision"]
    assert not cells["letters/digits"]["collision"] and not cells["digits/letters"]["collision"]
    assert cells["letters/letters"]["copy_rate"] > 0.75 and cells["digits/digits"]["copy_rate"] > 0.75
    for clean in ("letters/digits", "digits/letters"):
        assert abs(cells[clean]["copy_rate"] - cells[clean]["copy_chance"]) < 0.08
    effect = block["effects"]["copy_rate"]
    assert 0.5 < effect["mean"]["diff"] < 0.7 and effect["mean"]["ci"][0] > 0.4
    assert effect["letters_scale"]["ci"][0] > 0 and effect["digits_scale"]["ci"][0] > 0
    assert block["n_common"] == N
    # explicit denominators: no echo without shared labels, so nothing remains above chance
    for scale in ("letters_scale", "digits_scale"):
        res = block["residual_agreement"][scale]
        assert res["shared_labels"]["rate"] > 0.75 and abs(res["no_shared_labels"]["rate"] - res["no_shared_labels"]["chance"]) < 0.08
        assert abs(res["fraction_of_above_chance_agreement"]) < 0.2
        assert res["fraction_of_raw_agreement"] == pytest.approx(res["no_shared_labels"]["rate"] / res["shared_labels"]["rate"], abs=1e-3)


def test_residual_agreement_is_reported_as_a_fraction_of_the_above_chance_part(world):
    tmp, items, answer, gold, rng = world
    echo = lambda p: [LETTER_OF[a] if rng.random() < p else LETTER_OF[rng.integers(0, 4)] for a in answer]  # noqa: E731
    write(tmp, "letters_raw", echo(0.8))  # shared labels: 80% echo
    write(tmp, "letters_optdigits", echo(0.4))  # no shared labels, but half of the echo remains
    write(tmp, "digits_raw", [str(rng.integers(1, 5)) for _ in range(N)])
    write(tmp, "digits_optdigits", [str(a + 1) if rng.random() < 0.8 else str(rng.integers(1, 5)) for a in answer])
    block = analysis.collision_block(tmp, items, direct_outcomes(read_condition(tmp, "direct_clean", N), items), n_boot=50)
    res = block["residual_agreement"]["letters_scale"]
    # rate = chance + (1 - chance) * echo with chance 0.25: 0.85 against 0.55 share-of-above-chance 0.5
    assert res["fraction_of_above_chance_agreement"] == pytest.approx((0.4 * 0.75) / (0.8 * 0.75), abs=0.12)
    assert res["fraction_of_raw_agreement"] > res["fraction_of_above_chance_agreement"]  # the raw ratio flatters: chance counts as agreement


def test_collision_control_is_none_without_its_conditions(world):
    tmp, items, answer, gold, rng = world
    write(tmp, "letters_raw", ["A"] * N)
    assert analysis.collision_block(tmp, items, direct_outcomes(read_condition(tmp, "direct_clean", N), items), 50) is None


def test_factorial_attributes_a_first_line_responder_to_position(world):
    tmp, items, answer, gold, rng = world
    from probe_validity.scales import PROBE_SCALES

    for name in ("letters", "letters_flip_order", "letters_flip_labels", "letters_reversed"):
        first = PROBE_SCALES[name].lines()[0].split(")")[0]  # this responder always takes the first printed line
        write(tmp, f"{name}_raw", [first] * N)
    block = analysis.factorial_block(tmp, items, np.zeros(N, dtype=int), n_boot=50)
    assert all(c["first_line_rate"] == 1.0 for c in block["cells"].values())
    # semantic grade of "the first printed line" changes only where the order changed
    assert block["contrasts"]["letters_flip_labels"]["same_grade_rate"] == 1.0
    assert block["contrasts"]["letters_flip_order"]["same_grade_rate"] == 0.0
    assert block["contrasts"]["letters_flip_order"]["share_of_grades_changed"] == 1.0
    assert block["contrasts"]["letters_flip_order"]["n_jointly_valid"] == N
    assert block["contrasts"]["letters_reversed"]["same_grade_rate"] == 0.0


def test_factorial_attributes_a_label_a_responder_to_the_token(world):
    tmp, items, *_ = world
    for name in ("letters", "letters_flip_order", "letters_flip_labels", "letters_reversed"):
        write(tmp, f"{name}_raw", ["A"] * N)
    block = analysis.factorial_block(tmp, items, np.zeros(N, dtype=int), n_boot=50)
    assert all(c["label_A_rate"] == 1.0 for c in block["cells"].values())
    assert block["contrasts"]["letters_flip_order"]["same_grade_rate"] == 1.0  # labels kept: same semantic grade
    assert block["contrasts"]["letters_flip_labels"]["same_grade_rate"] == 0.0
    assert block["contrasts"]["letters_reversed"]["same_grade_rate"] == 0.0


def test_comparator_is_judged_on_the_same_items_and_against_a_coarsened_control():
    rng = np.random.default_rng(1)
    n = 3000
    signal = rng.normal(size=n)
    correct = (rng.random(n) < 1 / (1 + np.exp(-1.5 * signal))).astype(int)
    score = signal + rng.normal(0, 0.3, n)  # a continuous comparator
    verbal = np.digitize(score, [-0.5, 0.5]).astype(float)  # the same information, read out as one of three labels
    invalid = rng.random(n) < 0.4  # 40% of replies are not compliant; those are the harder items, and the comparator keeps them
    invalid &= np.abs(signal) < 0.5
    verbal[invalid] = np.nan
    mass = np.where(invalid, 0.3, 0.95)
    block = analysis.comparator_block(correct, verbal, score, mass, 3, multi_token_labels=True, n_boot=200)
    assert block["n_items"] == n and block["n_score_available"] == n and block["n_verbal_valid"] == int((~invalid).sum())
    assert block["n_common"] == int((~invalid).sum())
    assert block["score_all_items"]["n"] == n and block["verbal_on_common"]["n"] == block["n_common"]
    assert block["approximation"].startswith("first token")
    assert 0.45 < block["label_mass_mean"] < 0.9 and block["label_mass_p10"] <= 0.31
    # the continuous score is better than the three-label readout, but coarsened to three levels the gap is small
    assert block["score_minus_verbal"]["ci"][0] > 0
    assert abs(block["coarsened_score_minus_verbal"]["diff"]) < abs(block["score_minus_verbal"]["diff"])
    assert abs(block["coarsened_score_minus_verbal"]["diff"]) < 0.03


def test_comparator_degrades_gracefully_when_the_verbal_probe_is_unusable():
    n = 100
    out = analysis.comparator_block(np.arange(n) % 2, np.full(n, np.nan), np.linspace(0, 1, n), np.ones(n), 2, n_boot=20)
    assert out["n_common"] == 0 and "score_minus_verbal" not in out and out["score_all_items"]["n"] == n


def test_usage_matched_control_is_zero_for_a_thresholded_score_and_separates_threshold_placement():
    rng = np.random.default_rng(2)
    n = 4000
    score = rng.normal(size=n)
    correct = (rng.random(n) < 1 / (1 + np.exp(-1.5 * score))).astype(int)
    verbal = (score > 1.0).astype(float)  # the same information, thresholded high: "yes" for 16% of the items
    block = analysis.comparator_block(correct, verbal, score, np.ones(n), 2, n_boot=200)
    assert abs(block["usage_matched_score_minus_verbal"]["diff"]) < 0.005  # same ranking, same usage: nothing to find
    assert block["coarsened_score_minus_verbal"]["diff"] > 0.05  # equal-mass bins place the threshold better than the readout does
    assert block["score_minus_verbal"]["diff"] > block["coarsened_score_minus_verbal"]["diff"]


# --- methodology audit, 2026-10-27: the whole summary on a small synthetic run ---

@pytest.fixture()
def run_dir(world):
    """direct answers (60% right), a collapsed Yes/No probe (8 'No' replies, all on wrong answers) and a letters probe that copies the answer."""
    directory, items, answer, gold, rng = world
    wrong = np.flatnonzero(answer != gold)[:8]
    write(directory, "yesno_raw", ["No" if i in set(wrong.tolist()) else "Yes" for i in range(N)])
    write(directory, "letters_raw", [LETTER_OF[a] for a in answer])  # the confidence letter is the model's own answer letter
    return directory, items


def test_the_summary_exposes_the_audit_blocks(run_dir):
    directory, items = run_dir
    s = analysis.summarise_directory(directory, items, ("A", "B", "C", "D"), n_boot=30)
    assert s["direct"]["clean"]["accuracy_among_valid"] == s["direct"]["clean"]["accuracy"]  # every reply is a valid answer here
    assert s["robustness"]["answered"] == {"rate": 1.0, "n": N, "n_unfinished": 0, "accuracy_among_answered": s["direct"]["clean"]["accuracy"],
                                           "accuracy_if_unfinished_were_correct": s["direct"]["clean"]["accuracy"]}
    assert s["probes"]["letters_raw"]["association_with_answer_label"]["cramers_v"] > 0.99  # copying the answer letter
    for name in ("yesno_raw", "letters_raw"):
        d = s["screens"][name]["diagnostics"]
        assert set(d) >= {"tiers", "variants", "monotonicity", "threshold_sensitivity", "crossfit", "s4_own_pairs"}
        assert d["variants"]["frozen"] == s["screens"][name]["overall"]


def test_a_collapsed_yes_no_probe_has_a_high_gamma_star_and_a_chance_auroc(run_dir):
    directory, items = run_dir
    s = analysis.summarise_directory(directory, items, ("A", "B", "C", "D"), n_boot=30)
    yn = s["legacy_pipeline"]["yesno"]
    assert yn["strict"]["gamma_star"] == 1.0  # no discordant pair at all: eight 'No' replies, all on wrong answers
    assert 0.5 < yn["strict"]["auroc"]["auroc"] < 0.56 and yn["tie_share"] > 0.9
    assert yn["gamma_star_minus_auroc"] > 0.4  # the earlier draft's metric calls this probe nearly perfect
    assert yn["strict"]["p_positive_valid"] > 0.97 and yn["strict"]["delta_yes"] > 0.3


def test_unfinished_answers_are_reported_apart_from_wrong_ones(run_dir):
    directory, items = run_dir
    rows = [json.loads(line) for line in (directory / "direct_clean.jsonl").read_text().splitlines()]
    for r in rows[:20]:
        r["text"], r["finish"] = "To solve this we first", "length"  # a derivation cut off at the token limit: no answer
    (directory / "direct_clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = analysis.summarise_directory(directory, items, ("A", "B", "C", "D"), n_boot=30)
    a = s["robustness"]["answered"]
    assert a["n_unfinished"] == 20 and a["rate"] == pytest.approx(1 - 20 / N)
    assert a["accuracy_if_unfinished_were_correct"] == pytest.approx(s["direct"]["clean"]["accuracy"] + 20 / N, abs=1e-3)
    assert s["robustness"]["probes"]["letters_raw"]["unfinished_counted_correct"] is not None


# --- open questions: the same summary on a run directory without options ---

@pytest.fixture()
def open_run(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    items = [Item(qid=f"o{i}", stem="q", answers=(f"ans{i}", f"alias{i}"), group="") for i in range(n)]
    right = rng.random(n) < 0.5
    replies = [f"ans{i}" if r else ("Paris, London" if i % 7 == 0 else f"wrong{i}") for i, r in enumerate(right)]
    replies[3] = f"I am not sure whether it is ans3 or other"  # an uncommitted reply that names the answer
    replies[5] = "ans5 is not the answer; wrong"  # a rejected answer
    write(tmp_path, "direct_clean", replies)
    write(tmp_path, "yesno_raw", ["Yes" if (r or rng.random() < 0.3) else "No" for r in right])
    write(tmp_path, "letters_raw", [LETTER_OF[int(g)] for g in rng.integers(0, 4, n)])
    return tmp_path, items, right


def test_the_summary_of_an_open_question_run_has_the_open_answer_blocks(open_run):
    directory, items, right = open_run
    s = analysis.summarise_directory(directory, items, (), n_boot=20)
    oa = s["robustness"]["open_answers"]
    assert 0 < oa["exact_match_accuracy"] <= oa["primary_accuracy"] + 1e-9 and oa["list_like_rate"] > 0 and oa["single_answer_share"] == 0.0  # two aliases each
    assert set(oa["probes"]) >= {"yesno_raw", "letters_raw"} and oa["probes"]["yesno_raw"]["exact_match"] is not None
    assert s["direct"]["clean"]["hedged_rate"] > 0  # replies 3 and 5 are not answers
    assert s["legacy_pipeline"] is not None and "yesno" in s["legacy_pipeline"]  # the legacy scorer exists for open questions (substring)


def test_replies_that_do_not_commit_are_not_correct(open_run):
    directory, items, right = open_run
    d = analysis.direct_outcomes(analysis.read_condition(directory, "direct_clean", len(items)), items)
    assert d.correct[3] == 0 and d.correct[5] == 0


def test_the_coarsened_and_usage_matched_contrasts_refit_inside_the_bootstrap():
    rng = np.random.default_rng(5)
    n = 600
    correct = (rng.random(n) < 0.5).astype(int)
    score = correct * 0.3 + rng.random(n)  # a continuous score with signal
    verbal = np.digitize(score + rng.normal(0, 0.3, n), [0.4, 0.8, 1.2]).astype(float)  # a four-level reading of a noisier version of it
    block = analysis.comparator_block(correct, verbal, score, np.full(n, 0.9), 4, n_boot=60)
    for key in ("coarsened_score_minus_verbal", "usage_matched_score_minus_verbal"):
        assert block[key]["refit_in_bootstrap"] is True and block[key]["ci"][0] <= block[key]["diff"] <= block[key]["ci"][1]
    from probe_validity import metrics as M

    point = M.auroc(correct, M.match_marginal(score, verbal)) - M.auroc(correct, verbal)
    assert block["usage_matched_score_minus_verbal"]["diff"] == pytest.approx(point, abs=1e-4)  # the point estimate is the same as before


def test_screen_verdicts_are_also_computed_on_answered_items_with_their_coverage(run_dir):
    directory, items = run_dir
    rows = [json.loads(line) for line in (directory / "direct_clean.jsonl").read_text().splitlines()]
    for r in rows[:60]:
        r["text"], r["finish"] = "To solve this we first", "length"
    (directory / "direct_clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = analysis.summarise_directory(directory, items, ("A", "B", "C", "D"), n_boot=20)
    d = s["screens"]["letters_raw"]["diagnostics"]["answered_only"]
    assert d["n_items"] == N - 60 and d["coverage"] == pytest.approx(1 - 60 / N, abs=1e-3) and d["overall"] in ("PASS", "WARN", "FAIL")
    assert "changes_the_verdict" in d


def test_the_metric_only_gap_uses_one_set_of_labels_and_follows_the_tie_identity(run_dir):
    directory, items = run_dir
    s = analysis.summarise_directory(directory, items, ("A", "B", "C", "D"), n_boot=20)
    y = s["legacy_pipeline"]["yesno"]
    assert y["tie_identity_error"] < 1e-3  # gamma* - AUROC = (AUROC - 0.5) * ties / (1 - ties), exactly
    assert y["gamma_star_minus_auroc"] == pytest.approx(y["strict"]["gamma_star"] - y["strict"]["auroc"]["auroc"], abs=1e-4)
    assert "gamma_star_minus_auroc_full_pipeline" in y and "delta_yes_imputed_strict_labels" in y["strict"]
