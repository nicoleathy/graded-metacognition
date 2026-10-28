"""The power study runs every setting with the same run indices 0..SEEDS-1. The runs of different settings are still independent draws, because the generator is
seeded with the run index and a hash of the setting. The per-setting bound on a zero count (1 - 0.05 ** (1 / 300), 0.994%) relies on this; the pooled count over settings is
only a description of those settings either way (experiments/paper1/screen_power.py)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import screen_power  # noqa: E402

from probe_validity.simulators import Spec, simulate_spec  # noqa: E402


def test_one_setting_and_run_index_always_gives_the_same_draw():
    a, b = simulate_spec(Spec(n=500), 7), simulate_spec(Spec(n=500), 7)
    assert np.array_equal(a.correct, b.correct) and np.array_equal(a.grade, b.grade)


def test_the_same_run_index_in_two_settings_is_not_the_same_draw():
    weak, sound = Spec(n=2000, latent_sd=0.5), Spec(n=2000, latent_sd=0.1)
    runs = [(simulate_spec(weak, s), simulate_spec(sound, s)) for s in range(5)]
    for w, s in runs:
        assert not np.array_equal(w.correct, s.correct)  # the correctness labels are not shared between settings
        assert abs(np.corrcoef(w.correct, s.correct)[0, 1]) < 0.1  # and are uncorrelated at n = 2000, as independent draws are
    assert not np.array_equal(runs[0][0].correct, runs[1][0].correct)  # different run indices of one setting differ too


def test_a_setting_that_differs_only_in_n_gets_its_own_stream():
    small, large = simulate_spec(Spec(n=1000), 0), simulate_spec(replace(Spec(n=1000), n=2000), 0)
    assert not np.array_equal(small.correct, large.correct[:1000])  # the sample sizes are separate settings, not nested prefixes of one stream


def test_the_zero_count_bound_is_per_setting():
    assert screen_power.zero_count_upper_bound(300) == 0.00994  # 0.994% for 300 independent runs of one setting
    assert screen_power.zero_count_upper_bound(3000) < screen_power.zero_count_upper_bound(300)  # the pooled figure would be smaller, which is why it is not used as a bound
