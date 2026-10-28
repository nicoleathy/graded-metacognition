import json
import pathlib

import numpy as np
import pytest

from probe_validity import metrics as M
from probe_validity.scales import DIGITS, LETTERS, WORDS
from probe_validity.screen import Cell, ScreenConfig, load_frozen, s1_parse, s2_labels, s3_usage, s7_convergent, s8_stratum, screen
from probe_validity.simulators import EXPECTED, OPTIONS, simulate

CFG = ScreenConfig()


@pytest.mark.parametrize("name", list(EXPECTED))
def test_screen_matches_the_expected_verdict_in_every_seed(name):
    wrong = []
    for seed in range(150):
        result = screen(simulate(name, seed), CFG)
        got = {"overall": result["overall"], **result["verdicts"]}
        wrong += [(seed, k, v, got[k]) for k, v in EXPECTED[name].items() if got[k] != v]
    assert not wrong, wrong[:5]


def test_oracle_also_passes_with_a_smaller_sample():
    assert all(screen(simulate("oracle", seed, n=250), CFG)["overall"] == "PASS" for seed in range(100))


def test_s2_is_a_pure_function_of_the_label_sets():
    empty = np.array([])
    for scale, expected in [(LETTERS, "FAIL"), (DIGITS, "PASS"), (WORDS, "PASS")]:
        assert s2_labels(Cell(scale, OPTIONS, empty, empty), CFG).verdict == expected
    assert s2_labels(Cell(LETTERS, (), empty, empty), CFG).verdict == "PASS"  # open questions have no answer labels


def test_s1_thresholds():
    def verdict(rate):
        grade = np.where(np.arange(1000) < rate * 1000, 2.0, np.nan)
        return s1_parse(Cell(LETTERS, (), np.zeros(1000), grade), CFG).verdict

    assert [verdict(r) for r in (1.0, 0.95, 0.949, 0.80, 0.799, 0.5)] == ["PASS", "PASS", "WARN", "WARN", "FAIL", "FAIL"]


def test_s3_thresholds():
    def verdict(top_share):
        n = 1000
        k = int(top_share * n)
        grade = np.array([3.0] * k + [2.0] * ((n - k) // 2) + [1.0] * (n - k - (n - k) // 2))
        return s3_usage(Cell(LETTERS, (), np.zeros(n), grade), CFG).verdict

    assert verdict(0.85) == "PASS" and verdict(0.95) == "FAIL"


def test_too_few_valid_replies_is_na_not_pass():
    grade = np.full(40, 2.0)
    result = screen(Cell(LETTERS, (), np.zeros(40), grade), CFG)
    assert result["verdicts"]["S3"] == "NA" and result["verdicts"]["S6"] == "NA"
    assert result["overall"] == "WARN" and result["complete"] is False


def test_missing_comparators_are_reported_not_hidden():
    cell = simulate("oracle", 0)
    cell.mappings, cell.sampled = {}, None
    result = screen(cell, CFG)
    assert result["verdicts"]["S4"] == "NA" and result["verdicts"]["S5"] == "NA"
    assert result["overall"] == "PASS" and result["complete"] is False
    assert {"S4", "S5", "S7", "S8"} <= set(result["not_evaluated"])


def test_s7_flags_a_probe_unrelated_to_the_models_own_likelihoods():
    rng = np.random.default_rng(0)
    cell = simulate("oracle", 1)
    cell.reference = cell.grade + rng.normal(0, 0.5, len(cell.grade))
    assert s7_convergent(cell, CFG).verdict == "PASS"
    cell.reference = rng.normal(size=len(cell.grade))
    assert s7_convergent(cell, CFG).verdict == "WARN"


def test_s8_separates_a_stratum_prior_from_real_signal():
    rng = np.random.default_rng(1)
    n = 3000
    strata = rng.integers(0, 4, n)
    correct = (rng.random(n) < 0.15 + 0.2 * strata).astype(int)
    prior_only = strata.astype(float)  # looks informative overall, knows only the subject
    informative = np.clip(np.round(strata + 2 * correct + rng.normal(0, 0.7, n)), 0, 3)
    assert M.auroc(correct, prior_only) > 0.55
    assert s8_stratum(Cell(LETTERS, (), correct, prior_only, strata=strata), CFG).verdict == "WARN"
    assert s8_stratum(Cell(LETTERS, (), correct, informative, strata=strata), CFG).verdict == "PASS"


def test_config_hash_is_stable_and_sensitive():
    assert ScreenConfig().sha256() == ScreenConfig().sha256()
    assert ScreenConfig(s1_pass=0.9).sha256() != ScreenConfig().sha256()


FROZEN = pathlib.Path(__file__).resolve().parents[2] / "experiments/paper1/frozen/screen_config.json"


def test_code_defaults_equal_the_frozen_thresholds():
    assert load_frozen(FROZEN) == ScreenConfig()  # changing a default requires re-validating and re-freezing


def test_an_edited_frozen_file_is_rejected(tmp_path):
    blob = json.loads(FROZEN.read_text())
    blob["config"]["s1_pass"] = 0.5
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        load_frozen(path)


def test_spec_defaults_describe_a_sound_probe():
    from probe_validity.simulators import Spec, simulate_spec

    results = [screen(simulate_spec(Spec(), seed), CFG)["overall"] for seed in range(100)]
    assert results.count("PASS") >= 98  # a sound probe is not flagged (the screen itself has a small false-alarm rate)


def test_spec_defects_act_as_documented():
    from probe_validity.simulators import Spec, simulate_spec

    cell = simulate_spec(Spec(noncompliance=0.3), 0)
    assert abs(np.isnan(cell.grade).mean() - 0.3) < 0.05
    cell = simulate_spec(Spec(copy_fraction=1.0, overlap=True), 0)
    assert s2_labels(cell, CFG).verdict == "FAIL"
    assert set(cell.mappings) == {"letters_reversed", "digits"}
    # a letter that repeats the answer letter means the opposite under the reversed labels
    assert np.all(cell.grade + cell.mappings["letters_reversed"] == 3.0)
    assert screen(cell, CFG)["verdicts"]["S4"] == "FAIL"
    cell = simulate_spec(Spec(collapse=1.0), 0)
    assert screen(cell, CFG)["verdicts"]["S3"] == "FAIL"
    cell = simulate_spec(Spec(anti_fraction=1.0), 0)
    assert screen(cell, CFG)["verdicts"]["S6"] == "FAIL"
    # same spec and seed give the same cell; another seed does not
    a, b, c = (simulate_spec(Spec(copy_fraction=0.3), s) for s in (1, 1, 2))
    assert np.array_equal(a.grade, b.grade, equal_nan=True) and not np.array_equal(a.grade, c.grade, equal_nan=True)


def test_s4_reports_which_pair_decided_and_the_probes_own_pairs():
    from probe_validity.simulators import Spec, simulate_spec

    result = screen(simulate_spec(Spec(copy_fraction=1.0), 0), CFG)  # letters copy the answer letter; digits do not
    s4 = result["checks"]["S4"].value
    assert result["verdicts"]["S4"] == "FAIL" and s4["scope"] == "scale family"
    assert s4["worst_pair"] in s4["kappas"] and s4["min_kappa"] == min(s4["kappas"].values())
    assert s4["own_min_kappa"] == min(v for k, v in s4["kappas"].items() if k.startswith("own~"))
    # a probe whose own pairs are fine can still inherit a failure from a pair that does not involve it: two mappings that each
    # agree with the probe (kappa about 0.7) but, being corrupted independently, only about 0.5 with each other
    from probe_validity.screen import Cell
    sound = simulate_spec(Spec(n=4000), 1)
    rng = np.random.default_rng(0)
    corrupt = lambda: np.where(rng.random(4000) < 0.25, rng.integers(0, 4, 4000).astype(float), sound.grade)  # noqa: E731
    cell = Cell(sound.scale, (), sound.correct, sound.grade, mappings={"letters_reversed": corrupt(), "digits": corrupt()})
    s4 = screen(cell, CFG)["checks"]["S4"]
    assert s4.verdict == "FAIL" and s4.value["own_min_kappa"] >= 0.6 > s4.value["min_kappa"]
    assert s4.value["worst_pair"] == "letters_reversed~digits"  # the pair that decided does not include the probe


def test_a_pass_with_unevaluated_checks_is_labelled_partial():
    cell = simulate("oracle", 0)
    cell.mappings, cell.sampled = {}, None
    result = screen(cell, CFG)
    assert result["overall"] == "PASS" and result["complete"] is False
    assert result["label"] == "PASS (not evaluated: S4, S5)"
    assert screen(simulate("oracle", 0), CFG)["label"] == "PASS"


def test_crossfit_halves_are_disjoint_stratified_and_deterministic():
    from probe_validity.crossfit import stratified_halves

    strata = np.array(list("aaaaaaabbbbcccccccccc"))
    a, b = stratified_halves(strata, 3)
    assert sorted(a.tolist() + b.tolist()) == list(range(len(strata))) and not set(a) & set(b)
    for g in "abc":
        assert abs(int(np.isin(a, np.flatnonzero(strata == g)).sum()) - int(np.isin(b, np.flatnonzero(strata == g)).sum())) <= 1
    assert all(np.array_equal(x, y) for x, y in zip(stratified_halves(strata, 3), (a, b)))
    assert not np.array_equal(stratified_halves(strata, 4)[0], a)


def test_selecting_on_the_same_items_inflates_auroc_and_crossfitting_removes_it():
    from probe_validity.crossfit import crossfit
    from probe_validity.simulators import Spec, simulate_spec

    # A probe with no signal passes S6 by chance in a few percent of draws. Among those draws the AUROC on the screening items is
    # above chance by construction; on the other half it is not.
    inside, held = [], []
    for seed in range(1500):
        cell = simulate_spec(Spec(n=400, latent_sd=50.0), seed)
        for fold in crossfit(cell, CFG, seed):
            if fold["verdicts"]["S6"] == "PASS":
                inside.append(fold["auroc_screen_half"])
                held.append(fold["auroc_heldout"])
    assert len(inside) > 30
    assert np.mean(inside) - 0.5 > 0.03  # selected on these items: looks informative
    assert abs(np.nanmean(held) - 0.5) < 0.02  # measured on the other half: it is not


def test_s7_is_an_absolute_association_check_and_reports_the_sign():
    rng = np.random.default_rng(0)
    cell = simulate("oracle", 1)
    ref = cell.grade + rng.normal(0, 0.5, len(cell.grade))
    cell.reference, cell.reference_mean = ref, ref / 7
    positive = s7_convergent(cell, CFG)
    assert positive.verdict == "PASS" and positive.value["sign"] == "positive" and positive.value["verdict_if_positive_required"] == "PASS"
    assert positive.value["rho_token_mean"] == pytest.approx(positive.value["rho"])  # a monotone rescaling does not change a rank correlation
    cell.reference = -ref  # strongly negative: passes the absolute check, fails the positive-convergence sensitivity check
    negative = s7_convergent(cell, CFG)
    assert negative.verdict == "PASS" and negative.value["sign"] == "negative" and negative.value["verdict_if_positive_required"] == "WARN"
    cell.reference = rng.normal(size=len(cell.grade))
    assert s7_convergent(cell, CFG).verdict == "WARN"
