"""Seven synthetic probes with known ground truth, used to validate the screen logic before any real data is seen.

Each simulator draws, for n items, the probability p that the model answers correctly, its answer letter, and a latent
feeling z = p + noise. The probe's behaviour defines the cell. EXPECTED lists the verdicts the screen must return in
every run; a screen that disagrees with them in any seed is wrong, whatever any language model does.
"""

import zlib
from dataclasses import dataclass

import numpy as np

from .scales import DIGITS, LETTERS, LETTERS_REVERSED
from .screen import Cell

OPTIONS = ("A", "B", "C", "D")
MAPPINGS = {"letters_reversed": LETTERS_REVERSED, "digits": DIGITS}
CUTS = (0.25, 0.5, 0.75)

EXPECTED = {
    "oracle": {"overall": "PASS", "S1": "PASS", "S2": "PASS", "S3": "PASS", "S4": "PASS", "S5": "PASS", "S6": "PASS"},
    "collapse": {"overall": "FAIL", "S3": "FAIL", "S6": "FAIL"},
    "label_copy": {"overall": "FAIL", "S2": "FAIL", "S4": "FAIL"},
    "label_prior": {"overall": "FAIL", "S4": "FAIL"},
    "noncompliant": {"overall": "FAIL", "S1": "FAIL"},
    "anti_informative": {"overall": "FAIL", "S6": "FAIL"},
    "sampling_noise": {"overall": "FAIL", "S5": "FAIL"},
}


def _grade(x) -> np.ndarray:
    return np.digitize(x, CUTS).astype(float)  # 0 (definitely not) .. 3 (certain)


def _base(rng, n):
    p = rng.beta(2, 2, n)
    return p, (rng.random(n) < p).astype(int), rng.integers(0, 4, n), p + rng.normal(0, 0.10, n)


def _sensible(rng, z, noise=0.04):
    return _grade(z + rng.normal(0, noise, z.shape))


def simulate(name: str, seed: int, n: int = 1000) -> Cell:
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    p, correct, letter, z = _base(rng, n)
    own = _sensible(rng, z)
    maps = {m: _sensible(rng, z) for m in MAPPINGS}
    sampled = np.stack([_sensible(rng, z, 0.06) for _ in range(5)], axis=1)
    cell = dict(scale=LETTERS, answer_labels=(), correct=correct, grade=own, mappings=maps, sampled=sampled)

    if name == "oracle":
        pass
    elif name == "collapse":
        cell.update(grade=np.full(n, 3.0), mappings={m: np.full(n, 3.0) for m in MAPPINGS}, sampled=np.full((n, 5), 3.0))
    elif name == "label_copy":
        # The confidence letter repeats the model's own answer letter, so it says nothing about confidence, and
        # a different label set (digits) is answered sensibly. Under the reversed letters the same letter means the opposite.
        copied = 3.0 - letter
        cell.update(answer_labels=OPTIONS, grade=copied, mappings={"letters_reversed": 3.0 - copied, "digits": maps["digits"]})
    elif name == "label_prior":
        # With probability 0.7 the first-listed label, whatever it means; otherwise a uniformly chosen other label.
        first = rng.random(n) < 0.7
        other = rng.integers(1, 4, n)  # index 1..3 in listing order

        def picked(scale):
            order = sorted(scale.labels)  # lines are listed in label order
            label = np.where(first, order[0], np.array(order)[other])
            return np.array([scale.grade_of(lab) for lab in label], dtype=float)

        cell.update(grade=picked(LETTERS), mappings={m: picked(s) for m, s in MAPPINGS.items()}, sampled=np.stack([picked(LETTERS)] * 5, axis=1))
    elif name == "noncompliant":
        bad = rng.random(n) < 0.4
        cell["grade"] = np.where(bad, np.nan, own)
    elif name == "anti_informative":
        flip = lambda g: 3.0 - g  # noqa: E731
        cell.update(grade=flip(own), mappings={m: flip(g) for m, g in maps.items()}, sampled=flip(sampled))
    elif name == "sampling_noise":
        cell["sampled"] = np.stack([_grade(z + rng.normal(0, 0.6, n)) for _ in range(5)], axis=1)
    else:
        raise KeyError(name)
    return Cell(**cell)


@dataclass(frozen=True)
class Spec:
    """A synthetic probe with graded defects, for measuring the screen's operating characteristics (false alarms on sound
    probes, detection rates at each defect severity). The defaults describe a sound probe."""

    n: int = 1000
    beta: tuple[float, float] = (2.0, 2.0)  # distribution of the model's per-item probability of being right
    latent_sd: float = 0.10  # noise between the model's feeling and that probability: larger means weaker signal
    cuts: tuple[float, float, float] = (0.25, 0.5, 0.75)  # where the feeling is cut into the four grades
    mapping_sd: float = 0.04  # noise added independently under each label mapping
    sample_sd: float = 0.06  # noise added independently to each sampled decode
    noncompliance: float = 0.0  # fraction of replies that cannot be parsed
    copy_fraction: float = 0.0  # fraction of items on which a letter label repeats the model's own answer letter
    prior_weight: float = 0.0  # fraction of items on which the first printed label is taken whatever it means
    collapse: float = 0.0  # fraction of items forced to the top grade
    anti_fraction: float = 0.0  # fraction of items on which the scale is read upside down
    overlap: bool = False  # confidence letters are also answer letters (S2)


def simulate_spec(spec: Spec, seed: int) -> Cell:
    rng = np.random.default_rng([seed, zlib.crc32(repr(spec).encode())])
    n = spec.n
    p = rng.beta(*spec.beta, n)
    correct = (rng.random(n) < p).astype(int)
    letter = rng.integers(0, 4, n)
    z = p + rng.normal(0, spec.latent_sd, n)
    grade = lambda sd: np.digitize(z + rng.normal(0, sd, n), spec.cuts).astype(float)  # noqa: E731
    own, reversed_, digits = grade(spec.mapping_sd), grade(spec.mapping_sd), grade(spec.mapping_sd)
    sampled = np.stack([grade(spec.sample_sd) for _ in range(5)], axis=1)

    copy, pick = rng.random(n) < spec.copy_fraction, rng.random(n) < spec.prior_weight
    top, flip = rng.random(n) < spec.collapse, rng.random(n) < spec.anti_fraction
    nan = rng.random(n) < spec.noncompliance

    def letters(g):  # the letter scale: a copied answer letter A..D is grade 3..0
        g = np.where(copy, 3.0 - letter, g)
        g = np.where(pick, 3.0, g)
        g = np.where(top, 3.0, g)
        return np.where(flip, 3.0 - g, g)

    own, sampled = letters(own), np.stack([letters(sampled[:, j]) for j in range(5)], axis=1)
    reversed_ = np.where(copy, letter.astype(float), reversed_)  # the same letter means the opposite under the reversed labels
    reversed_ = np.where(pick, 0.0, reversed_)  # the first printed label is now "definitely do not know"
    digits = np.where(pick, 3.0, digits)  # digits: the first printed label is still "certain", and nothing can be copied
    reversed_, digits = (np.where(top, 3.0, x) for x in (reversed_, digits))
    reversed_, digits = (np.where(flip, 3.0 - x, x) for x in (reversed_, digits))
    own = np.where(nan, np.nan, own)
    return Cell(LETTERS, OPTIONS if spec.overlap else (), correct, own, mappings={"letters_reversed": reversed_, "digits": digits}, sampled=sampled)
