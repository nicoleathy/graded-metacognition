import math

import numpy as np
import pytest

from esma.metric import goodman_kruskal_gamma, type2_auroc, type2_auroc_continuous, type2_d_prime
from probe_validity import metrics as M


def brute_force(correct, score):
    conc = disc = tied = 0
    for ci, si in zip(correct, score):
        if not ci:
            continue
        for cj, sj in zip(correct, score):
            if cj:
                continue
            conc += si > sj
            disc += si < sj
            tied += si == sj
    return conc, disc, tied


def random_cell(rng, n=None, levels=4):
    n = n or int(rng.integers(20, 400))
    score = rng.integers(0, levels, n)
    p = 0.25 + 0.15 * score * rng.random()
    correct = (rng.random(n) < p).astype(int)
    return correct, score


def test_pair_counts_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(100):
        correct, score = random_cell(rng, n=int(rng.integers(5, 80)))
        assert tuple(M.pair_counts(correct, score)) == tuple(float(x) for x in brute_force(correct, score))


def test_tie_identity_on_1000_random_datasets():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(1000):
        correct, score = random_cell(rng)
        p_tie = M.tie_share(correct, score)
        if np.isnan(p_tie) or p_tie >= 1:
            continue
        assert abs(M.gamma(correct, score) - (2 * M.auroc(correct, score) - 1) / (1 - p_tie)) < 1e-9
        assert abs(M.somers_d(correct, score) - (2 * M.auroc(correct, score) - 1)) < 1e-12
        checked += 1
    assert checked > 900


def test_agrees_with_the_legacy_metrics():
    rng = np.random.default_rng(2)
    for _ in range(60):
        correct, score = random_cell(rng)
        if correct.min() == correct.max() or score.min() == score.max():
            continue
        assert abs(M.auroc(correct, score) - type2_auroc(correct.tolist(), score.tolist())) < 1e-9
        assert abs(M.auroc(correct, score) - type2_auroc_continuous(correct.tolist(), score.astype(float).tolist())) < 1e-9
        assert abs(M.gamma(correct, score) - goodman_kruskal_gamma(correct.tolist(), score.tolist())) < 1e-9
        yes = (score >= 2).astype(int)
        assert abs(M.dprime_type2(correct, yes) - type2_d_prime(correct.tolist(), yes.tolist())) < 1e-12


def test_undefined_values_are_nan_not_zero():
    correct = [1, 0, 1, 0, 1, 0]
    constant = [2, 2, 2, 2, 2, 2]
    assert M.tie_share(correct, constant) == 1.0
    assert math.isnan(M.gamma(correct, constant))  # the legacy function returns 0.0 here
    assert goodman_kruskal_gamma(correct, constant) == 0.0
    assert M.auroc(correct, constant) == 0.5 and M.somers_d(correct, constant) == 0.0
    assert math.isnan(M.auroc([1, 1, 1], [0, 1, 2])) and math.isnan(M.gamma([0, 0], [1, 2]))
    assert math.isnan(M.dprime_type2([1, 1, 1], [1, 0, 1]))


def test_gamma_can_be_perfect_while_auroc_is_near_chance():
    # Hand computation: 52 correct, 48 incorrect, 2496 pairs. Two correct items score 1, everything else 0.
    # Concordant 2 * 48 = 96, discordant 0, tied 50 * 48 = 2400: gamma = 96 / 96 = 1, AUROC = (96 + 1200) / 2496 = 0.519.
    correct = [1] * 52 + [0] * 48
    score = [1] * 2 + [0] * 50 + [0] * 48
    assert tuple(M.pair_counts(correct, score)) == (96.0, 0.0, 2400.0)
    assert M.gamma(correct, score) == 1.0
    assert M.auroc(correct, score) == pytest.approx(1296 / 2496) and M.tie_share(correct, score) == pytest.approx(2400 / 2496)


def test_auroc_within_strata():
    rng = np.random.default_rng(3)
    n = 4000
    stratum = rng.integers(0, 4, n)
    correct = (rng.random(n) < 0.15 + 0.2 * stratum).astype(int)
    prior_only = stratum.astype(float)  # a score that only knows the stratum
    assert M.auroc(correct, prior_only) > 0.55
    assert abs(M.auroc_within(correct, prior_only, stratum) - 0.5) < 1e-9
    informative = correct + rng.normal(0, 0.8, n)
    assert M.auroc_within(correct, informative, stratum) > 0.6
    one = np.zeros(n, dtype=int)
    assert abs(M.auroc_within(correct, informative, one) - M.auroc(correct, informative)) < 1e-12


def test_delong_matches_bootstrap_standard_error():
    rng = np.random.default_rng(4)
    n = 600
    correct = (rng.random(n) < 0.5).astype(int)
    score = correct * 0.8 + rng.normal(0, 1, n)
    auc, lo, hi = M.delong_ci(correct, score)
    assert abs(auc - M.auroc(correct, score)) < 1e-12
    boot = M.bootstrap(lambda idx: M.auroc(correct[idx], score[idx]), n, n_boot=600, seed=1)
    assert abs((hi - lo) - (boot["hi"] - boot["lo"])) < 0.2 * (hi - lo)
    assert lo < auc < hi


def test_wilson_known_values():
    lo, hi = M.wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 5e-4 and abs(hi - 0.7634) < 5e-4
    lo, hi = M.wilson_ci(0, 20)
    assert lo == 0.0 and abs(hi - 0.1611) < 5e-4
    assert math.isnan(M.wilson_ci(0, 0)[0])


def test_usage_entropy_and_top_share():
    assert M.usage([0, 1, 2, 3] * 10, 4)["entropy"] == pytest.approx(1.0)
    u = M.usage([3] * 20, 4)
    assert u["entropy"] == 0.0 and u["top_share"] == 1.0
    u = M.usage([3, 3, 2, float("nan")], 4)
    assert u["n"] == 3 and u["counts"] == [0, 0, 1, 2]
    assert math.isnan(M.usage([float("nan")], 4)["entropy"])


def test_weighted_kappa():
    a = [0, 1, 2, 3] * 25
    assert M.weighted_kappa(a, a, range(4)) == pytest.approx(1.0)
    assert M.weighted_kappa(a, [3 - x for x in a], range(4)) < -0.9
    rng = np.random.default_rng(5)
    x, y = rng.integers(0, 4, 4000), rng.integers(0, 4, 4000)
    assert abs(M.weighted_kappa(x, y, range(4))) < 0.05
    assert math.isnan(M.weighted_kappa([2] * 10, [2] * 10, range(4)))
    # Hand computation, unweighted 2 x 2 table [[4, 1], [2, 3]]: agreement 0.7, chance 0.5 * 0.6 + 0.5 * 0.4 = 0.5, kappa 0.4.
    a2 = [0] * 5 + [1] * 5
    b2 = [0] * 4 + [1] * 1 + [0] * 2 + [1] * 3
    assert M.weighted_kappa(a2, b2, range(2), "none") == pytest.approx(0.4)


def test_spearman_and_reliability():
    assert M.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(M.spearman([1, 1, 1], [1, 2, 3]))
    rng = np.random.default_rng(6)
    grade = rng.integers(0, 4, 3000)
    correct = (rng.random(3000) < 0.15 + 0.2 * grade).astype(int)
    table = M.reliability(correct, grade, 4)
    accs = [row["accuracy"] for row in table]
    assert accs == sorted(accs) and all(row["lo"] < row["accuracy"] < row["hi"] for row in table)
    assert M.spearman(grade, correct) > 0
    empty = M.reliability([1, 0], [0, 0], 4)
    assert empty[3]["n"] == 0 and math.isnan(empty[3]["accuracy"])


def test_ece_and_brier_references():
    n, bins = 20000, 15
    # A perfectly calibrated forecaster still has sampling error: E[ECE] = sqrt(2/pi) * E[sqrt(p(1-p))] / sqrt(n/bins)
    # with E[sqrt(p(1-p))] = pi/8 for p ~ U(0,1), i.e. 0.0086 here (one draw ranges up to about 0.012).
    expected = np.sqrt(2 / np.pi) * (np.pi / 8) / np.sqrt(n / bins)
    eces = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        p = rng.random(n)
        y = (rng.random(n) < p).astype(int)
        eces.append(M.ece(y, p, bins))
    assert np.mean(eces) == pytest.approx(expected, rel=0.15) and max(eces) < 0.015
    assert M.brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert M.brier(y, p) == pytest.approx(1 / 6, abs=0.01)  # E[p(1-p)] for p ~ U(0,1)
    over = np.clip(p + 0.15, 0, 1)
    assert M.ece(y, over) == pytest.approx(0.15, abs=0.02)


def test_bootstrap_is_seeded_and_counts_nan_draws():
    values = np.arange(50, dtype=float)
    f = lambda idx: float(values[idx].mean())  # noqa: E731
    a, b = M.bootstrap(f, 50, n_boot=200, seed=3), M.bootstrap(f, 50, n_boot=200, seed=3)
    assert a == b and a["lo"] < a["estimate"] < a["hi"] and a["n_nan"] == 0
    g = M.bootstrap(lambda idx: float("nan"), 50, n_boot=10)
    assert g["n_nan"] == 10 and math.isnan(g["lo"])


def test_one_clip_rule_for_dprime():
    import ast
    import pathlib

    root = pathlib.Path(M.__file__).resolve().parents[1]
    found = []
    for path in [*root.glob("probe_validity/*.py"), *root.glob("experiments/paper1/**/*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                calls = {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
                         for c in ast.walk(node) if isinstance(c, ast.Call)}
                if "clip" in calls and "ppf" in calls:  # a d' with a clipping rule
                    found.append((path.name, node.name))
    assert found == [("metrics.py", "dprime_type2")]


@pytest.mark.parametrize("fn", [M.auroc, M.somers_d, M.tie_share, M.gamma, M.pair_counts, M.delong_ci, M.brier, M.ece])
def test_missing_scores_are_rejected_not_ranked(fn):
    for bad in ([None, 0.0, 1.0, 0.0], [float("nan"), 0.0, 1.0, 0.0], [np.inf, 0.0, 1.0, 0.0]):
        with pytest.raises(ValueError, match="finite"):
            fn([1, 0, 1, 0], bad)


def test_missing_scores_cannot_make_a_probe_look_perfect():
    with pytest.raises(ValueError):
        M.auroc([1, 0], [None, 0])  # used to return 1.0: NaN sorted above every real score
    with pytest.raises(ValueError):
        M.auroc_within([1, 0, 1, 0], [None, 0, 1, 0], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        M.spearman([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="0 and 1"):
        M.auroc([1, 2, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        M.auroc([1, None, 0], [0.1, 0.2, 0.3])
    mask = np.array([False, True, True])  # the explicit alternative: select the valid replies, then compute
    assert M.auroc(np.array([1, 1, 0])[mask], np.array([np.nan, 0.5, 0.2])[mask]) == 1.0


def test_ece_does_not_depend_on_the_order_of_identical_observations():
    p = np.full(200, 0.5)
    grouped, alternating = np.array([1] * 100 + [0] * 100), np.array([1, 0] * 100)
    assert M.ece(grouped, p) == M.ece(alternating, p) == pytest.approx(0.0)  # it was 0.48 against 0.025
    rng = np.random.default_rng(0)
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # verbalized probabilities: heavy ties
    prob = rng.choice(levels, 500, p=[0.1, 0.1, 0.2, 0.2, 0.4])
    y = (rng.random(500) < prob - 0.1).astype(int)
    reference = M.ece(y, prob)
    for seed in range(20):
        perm = np.random.default_rng(seed).permutation(500)
        assert M.ece(y[perm], prob[perm]) == pytest.approx(reference, abs=1e-12)
    bins = M.equal_mass_bins(prob, 15)
    assert sorted(i for g in bins for i in g.tolist()) == list(range(500))  # every row in exactly one bin
    for level in levels:  # a probability level never spans two bins
        assert sum(bool((prob[g] == level).any()) for g in bins) == 1


def test_ece_matches_a_hand_computation_without_ties():
    prob = np.arange(1, 13) / 13  # 12 distinct probabilities, 3 bins of 4
    y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    expected = sum(4 / 12 * abs(y[i:i + 4].mean() - prob[i:i + 4].mean()) for i in (0, 4, 8))
    assert M.ece(y, prob, n_bins=3) == pytest.approx(expected)


def test_coarsening_gives_equal_mass_groups_that_keep_ties_together():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert M.coarsen_equal_mass(x, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    tied = M.coarsen_equal_mass(np.array([0.5] * 6 + [0.9, 0.9]), 4)
    assert len(set(tied[:6])) == 1 and len(set(tied[6:])) == 1 and tied[0] < tied[6]
    with pytest.raises(ValueError):
        M.coarsen_equal_mass(np.array([0.1, np.nan]), 2)
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    labels = M.coarsen_equal_mass(z, 4)
    assert np.bincount(labels.astype(int)).tolist() == [1000] * 4 and (np.diff(z[np.argsort(z)]) >= 0).all()
    y = (rng.random(4000) < 1 / (1 + np.exp(-2 * z))).astype(int)
    assert M.auroc(y, z) > M.auroc(y, labels) > 0.5  # coarsening loses information but keeps the order


def test_match_marginal_keeps_the_references_label_frequencies_and_the_scores_order():
    rng = np.random.default_rng(0)
    score = rng.normal(size=2000)
    reference = np.where(rng.random(2000) < 0.97, 1.0, 0.0)  # a readout that says "yes" 97% of the time
    matched = M.match_marginal(score, reference)
    assert abs(matched.mean() - reference.mean()) < 1e-3
    order = np.argsort(score)
    assert (np.diff(matched[order]) >= 0).all()  # a higher score never gets a lower label
    four = rng.integers(0, 4, 2000).astype(float)
    assert np.bincount(M.match_marginal(score, four).astype(int)).tolist() == np.bincount(four.astype(int)).tolist()
    tied = M.match_marginal(np.array([0.5] * 6 + [0.9] * 4), np.array([0.0] * 5 + [1.0] * 5))
    assert len(set(tied[:6])) == 1 and len(set(tied[6:])) == 1  # ties share a label
    with pytest.raises(ValueError):
        M.match_marginal(np.array([0.1, np.nan]), np.array([0.0, 1.0]))


def test_cramers_v_matches_the_textbook_values():
    from probe_validity.metrics import cramers_v

    perfect = cramers_v(list("AABBCC") * 10, list("AABBCC") * 10)
    assert perfect["v"] == pytest.approx(1.0) and perfect["p"] < 1e-10 and perfect["dof"] == 4
    independent = cramers_v(list("ABAB") * 50, list("AABB") * 50)  # a balanced 2 x 2 with no association
    assert independent["v"] == pytest.approx(0.0, abs=1e-12) and independent["p"] == pytest.approx(1.0)
    assert cramers_v(["a", "a", "b", "b"], ["x", "x", "y", "y"])["v"] == pytest.approx(1.0)  # 2 x 2: V equals |phi|
    assert cramers_v(["a", "a"], ["x", "y"])["v"] != cramers_v(["a", "a"], ["x", "y"])["v"]  # a constant variable gives NaN


def test_cramers_v_ignores_label_names_and_handles_unequal_alphabets():
    from probe_validity.metrics import cramers_v

    a = cramers_v(list("ABCD") * 25 + list("AB") * 10, list("ABCD") * 25 + list("AB") * 10)
    b = cramers_v([1, 2, 3, 4] * 25 + [1, 2] * 10, ["w", "x", "y", "z"] * 25 + ["w", "x"] * 10)
    assert a["v"] == pytest.approx(b["v"]) and a["dof"] == 9
    tall = cramers_v(list("ABCD") * 30, list("ABCDEFGHIJ")[:4] * 30)  # fewer columns than 10 options: min(rows, cols) - 1 sets the scale
    assert 0.99 < tall["v"] <= 1.0
    with pytest.raises(ValueError):
        cramers_v([1, 2], [1])
