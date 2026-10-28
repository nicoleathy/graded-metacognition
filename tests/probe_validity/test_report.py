import dataclasses

import numpy as np
import pytest

from probe_validity import report
from probe_validity.scales import LETTERS
from probe_validity.screen import ScreenConfig, screen
from probe_validity.simulators import EXPECTED, OPTIONS, simulate

CFG = ScreenConfig()
RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_combine_is_the_screens_own_decision_rule(name):
    for seed in range(25):
        result = screen(simulate(name, seed), CFG)
        assert report.combine(result["verdicts"]) == result["overall"]


def test_combine_on_hand_built_verdicts():
    ok = {k: "PASS" for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")}
    assert report.combine(ok) == "PASS"
    assert report.combine({**ok, "S7": "WARN"}) == "WARN" and report.combine({**ok, "S6": "NA"}) == "WARN"
    assert report.combine({**ok, "S5": "NA", "S8": "NA"}) == "PASS"  # a partial pass is still a pass here; screen() labels it partial
    assert report.combine({**ok, "S2": "FAIL", "S7": "WARN"}) == "FAIL"


def test_tiers_separate_design_stability_and_signal():
    ok = {k: "PASS" for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")}
    assert report.tiers(ok) == {"design": "PASS", "stability": "PASS", "signal": "PASS"}
    assert report.tiers({**ok, "S2": "FAIL"}) == {"design": "FAIL", "stability": "PASS", "signal": "PASS"}
    assert report.tiers({**ok, "S4": "FAIL", "S5": "NA"})["stability"] == "FAIL"
    assert report.tiers({**ok, "S4": "NA", "S5": "NA"})["stability"] == "NA"
    assert report.tiers({**ok, "S6": "FAIL"})["signal"] == "FAIL" and report.tiers({**ok, "S3": "FAIL"})["signal"] == "FAIL"
    assert report.tiers({**ok, "S7": "WARN"})["signal"] == "WARN" and report.tiers({**ok, "S6": "NA"})["signal"] == "NA"


def _graded(accuracies, n_each):
    grade = np.repeat(np.arange(len(accuracies), dtype=float), n_each)
    rng = np.random.default_rng(0)
    correct = np.concatenate([(rng.random(n_each) < a).astype(int) for a in accuracies])
    return correct, grade


def test_monotonicity_finds_a_credible_reversal_and_ignores_noise():
    correct, grade = _graded([0.10, 0.80, 0.30, 0.90], 300)  # the audit's counterexample: the overall association is still positive
    m = report.monotonicity(correct, grade, 4)
    assert m["monotone"] is False and m["n_reversals"] == 1 and m["largest_reversal"] > 0.3
    assert [s["reversal"] for s in m["steps"]] == [False, True, False]
    ok_correct, ok_grade = _graded([0.3, 0.5, 0.7, 0.9], 300)
    assert report.monotonicity(ok_correct, ok_grade, 4)["monotone"] is True
    flat_correct, flat_grade = _graded([0.5, 0.5, 0.5, 0.5], 60)  # small, equal bins: sampling noise alone is not a reversal
    assert report.monotonicity(flat_correct, flat_grade, 4)["n_reversals"] == 0


def test_monotonicity_skips_levels_with_too_few_replies():
    correct, grade = _graded([0.2, 0.9], 100)
    grade = np.where(grade == 1, 3.0, grade)  # levels 1 and 2 are never used
    m = report.monotonicity(correct, grade, 4)
    assert m["levels_compared"] == 2 and [(s["from"], s["to"]) for s in m["steps"]] == [(0, 3)]


def test_s2_as_a_design_restriction_changes_only_the_overlap_verdict():
    cell = dataclasses.replace(simulate("oracle", 3), scale=LETTERS, answer_labels=OPTIONS)  # a perfect signal on a letter scale over lettered options
    result = screen(cell, CFG)
    assert result["overall"] == "FAIL" and result["verdicts"]["S2"] == "FAIL"
    v = report.variants(result, cell.correct, cell.grade, 4, CFG)
    assert v["overall"]["frozen"] == "FAIL" and v["overall"]["s2_as_design_restriction"] == "PASS"
    assert report.tiers(result["verdicts"]) == {"design": "FAIL", "stability": "PASS", "signal": "PASS"}


def test_loosening_never_lowers_a_verdict_and_tightening_never_raises_it():
    for name in EXPECTED:
        for seed in range(3):
            result = screen(simulate(name, seed), CFG, True)
            s = report.threshold_sensitivity(simulate(name, seed), CFG, result["verdicts"]["S8"])
            assert RANK[s["all_loose"]] >= RANK[s["frozen"]] >= RANK[s["all_tight"]], (name, seed, s)
            assert s["frozen"] == result["overall"]


def test_monotonicity_ignores_replies_without_a_grade():
    correct, grade = _graded([0.3, 0.5, 0.7, 0.9], 200)
    noisy = np.concatenate([grade, np.full(50, np.nan)])
    noisy_correct = np.concatenate([correct, np.zeros(50, dtype=int)])
    assert report.monotonicity(noisy_correct, noisy, 4)["table"] == report.monotonicity(correct, grade, 4)["table"]
