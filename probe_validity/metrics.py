"""Metrics for confidence probes. numpy and scipy only.

Pairs are always (correct item, incorrect item): C concordant (the correct item has the higher score),
D discordant, T tied. Then AUROC = (C + T/2) / (C + D + T), Somers' D = 2 AUROC - 1, P_tie = T / (C + D + T),
and the Goodman-Kruskal gamma of the 2 x k table is (C - D) / (C + D) = Somers' D / (1 - P_tie).
Undefined values are NaN, never 0.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm, rankdata

NAN = float("nan")


def _arrays(correct, score) -> tuple[np.ndarray, np.ndarray]:
    """Validated pair of arrays. A missing score (None, NaN) is an error, not a value: NaN would sort above every real score and
    a probe with missing replies could look perfect. Select the valid replies with an explicit mask first."""
    c = np.asarray(correct)
    s = np.asarray(score, dtype=float)
    if c.shape != s.shape or c.ndim != 1:
        raise ValueError("correct and score must be 1-D arrays of the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite: mask out invalid replies before computing a metric")
    if c.dtype != bool and not np.isin(c, (0, 1)).all():
        raise ValueError("correct must contain only 0 and 1")
    return c.astype(bool), s


def pair_counts(correct, score) -> tuple[float, float, float]:
    """(C, D, T) over all (correct, incorrect) pairs."""
    c, s = _arrays(correct, score)
    _, inv = np.unique(s, return_inverse=True)
    inv = inv.reshape(-1)
    k = inv.max() + 1 if inv.size else 0
    n_c = np.bincount(inv, weights=c.astype(float), minlength=k)
    n_i = np.bincount(inv, weights=(~c).astype(float), minlength=k)
    below = np.cumsum(n_i) - n_i
    conc = float((n_c * below).sum())
    tied = float((n_c * n_i).sum())
    return conc, float(n_c.sum() * n_i.sum()) - conc - tied, tied


def auroc(correct, score) -> float:
    conc, disc, tied = pair_counts(correct, score)
    total = conc + disc + tied
    return (conc + 0.5 * tied) / total if total else NAN


def somers_d(correct, score) -> float:
    conc, disc, tied = pair_counts(correct, score)
    total = conc + disc + tied
    return (conc - disc) / total if total else NAN


def tie_share(correct, score) -> float:
    conc, disc, tied = pair_counts(correct, score)
    total = conc + disc + tied
    return tied / total if total else NAN


def gamma(correct, score) -> float:
    """Goodman-Kruskal gamma over untied pairs; NaN when every pair is tied."""
    conc, disc, _ = pair_counts(correct, score)
    return (conc - disc) / (conc + disc) if conc + disc else NAN


def auroc_within(correct, score, strata) -> float:
    """AUROC from pairs inside the same stratum only (e.g. subject): removes what a stratum prior can explain."""
    c, s = _arrays(correct, score)
    strata = np.asarray(strata)
    tot_c = tot_d = tot_t = 0.0
    for key in np.unique(strata):
        mask = strata == key
        a, b, t = pair_counts(c[mask], s[mask])
        tot_c, tot_d, tot_t = tot_c + a, tot_d + b, tot_t + t
    total = tot_c + tot_d + tot_t
    return (tot_c + 0.5 * tot_t) / total if total else NAN


def delong_ci(correct, score, alpha: float = 0.05) -> tuple[float, float, float]:
    """(AUROC, lower, upper) with DeLong's variance (fast midrank form of Sun and Xu, 2014)."""
    c, s = _arrays(correct, score)
    pos, neg = s[c], s[~c]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return NAN, NAN, NAN
    tz = rankdata(np.concatenate([pos, neg]))
    tx, ty = rankdata(pos), rankdata(neg)
    auc = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = (np.var(v01, ddof=1) / m if m > 1 else 0.0) + (np.var(v10, ddof=1) / n if n > 1 else 0.0)
    half = norm.ppf(1 - alpha / 2) * np.sqrt(var)
    return float(auc), float(max(0.0, auc - half)), float(min(1.0, auc + half))


def coarsen_equal_mass(score, levels: int) -> np.ndarray:
    """Cut a continuous score into `levels` ordered groups of about equal size; identical scores share a group. Used to ask whether a
    continuous comparator beats a verbal readout only because the readout is limited to a few labels."""
    x = np.asarray(score, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("scores must be finite: mask out missing values first")
    ranks = (rankdata(x) - 0.5) / len(x)  # average rank of ties, in (0, 1)
    return np.minimum((ranks * levels).astype(int), levels - 1).astype(float)


def match_marginal(score, reference) -> np.ndarray:
    """Give a continuous score the label frequencies of a reference ordinal readout: the lowest-scored items receive the lowest label,
    in the reference's proportions. Equal-mass coarsening changes where the labels fall as well as how many there are; this keeps the
    model's own usage (a readout that says "Yes" 97% of the time stays at 97%) and asks only whether the score ranks items
    better than the reference does at that usage. Ties in the score share a label."""
    x, ref = np.asarray(score, dtype=float), np.asarray(reference, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(ref).all()):
        raise ValueError("values must be finite: mask out missing values first")
    levels = np.unique(ref)
    cuts = np.cumsum([(ref == lv).mean() for lv in levels])[:-1]
    ranks = (rankdata(x) - 0.5) / len(x)
    return levels[np.searchsorted(cuts, ranks, side="right")]


def dprime_type2(correct, yes) -> float:
    """z(hit rate) - z(false-alarm rate), rates clipped to [1e-4, 1 - 1e-4] (the one clip rule, as in esma.metric)."""
    c, y = _arrays(correct, yes)
    if c.all() or not c.any():
        return NAN
    hit = float(np.clip(y[c].mean(), 1e-4, 1 - 1e-4))
    fa = float(np.clip(y[~c].mean(), 1e-4, 1 - 1e-4))
    return float(norm.ppf(hit) - norm.ppf(fa))


def wilson_ci(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return NAN, NAN
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def usage(grades, n_levels: int) -> dict:
    """Distribution of valid grades: counts, normalised entropy over n_levels (1 = uniform, 0 = one level), top share."""
    g = np.asarray(grades, dtype=float)
    g = g[~np.isnan(g)].astype(int)
    counts = np.bincount(g, minlength=n_levels)[:n_levels]
    n = int(counts.sum())
    if n == 0:
        return {"n": 0, "counts": counts.tolist(), "entropy": NAN, "top_share": NAN}
    p = counts[counts > 0] / n
    entropy = float(max(0.0, -(p * np.log(p)).sum() / np.log(n_levels)))  # max(): no negative zero for one level
    return {"n": n, "counts": counts.tolist(), "entropy": entropy, "top_share": float(counts.max() / n)}


def weighted_kappa(a, b, levels, weights: str = "quadratic") -> float:
    """Cohen's kappa between two ratings of the same items (weights: quadratic, linear, none). NaN if undefined."""
    a, b = np.asarray(a), np.asarray(b)
    k = len(levels)
    pos = {v: i for i, v in enumerate(levels)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    n = obs.sum()
    if n == 0:
        return NAN
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    i, j = np.indices((k, k))
    if weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "none":
        w = (i != j).astype(float)
    else:
        raise ValueError(f"unknown weights {weights!r}")
    den = float((w * exp).sum())
    return 1.0 - float((w * obs).sum()) / den if den > 0 else NAN


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("values must be finite: mask out invalid replies first")
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return NAN
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def reliability(correct, grades, n_levels: int) -> list[dict]:
    """Accuracy per grade with Wilson 95% intervals; levels never used are listed with n = 0."""
    c, g = _arrays(correct, grades)
    rows = []
    for level in range(n_levels):
        mask = g == level
        n, k = int(mask.sum()), int(c[mask].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"grade": level, "n": n, "accuracy": k / n if n else NAN, "lo": lo, "hi": hi})
    return rows


def brier(correct, prob) -> float:
    c, p = _arrays(correct, prob)
    return float(np.mean((p - c.astype(float)) ** 2))


def equal_mass_bins(prob, n_bins: int) -> list[np.ndarray]:
    """Index groups of roughly equal size, in increasing order of probability, that never separate identical probabilities:
    a bin edge that would fall inside a run of equal values moves to the end of that run."""
    p = np.asarray(prob, dtype=float)
    order = np.argsort(p, kind="stable")
    ps, n = p[order], len(p)
    cuts = [0]
    for b in range(1, n_bins):
        pos = max(int(round(b * n / n_bins)), cuts[-1])
        while 0 < pos < n and ps[pos] == ps[pos - 1]:
            pos += 1
        cuts.append(pos)
    cuts.append(n)
    return [order[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def ece(correct, prob, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins. Identical probabilities share a bin, so the value depends only on
    the multiset of (probability, outcome) pairs, not on the order of the rows. The estimator is biased upwards in finite
    samples (about 0.0086 at n = 20,000 and 15 bins for a perfectly calibrated forecaster); compare with that, not with 0."""
    c, p = _arrays(correct, prob)
    return float(sum(len(g) / len(p) * abs(float(c[g].mean()) - float(p[g].mean())) for g in equal_mass_bins(p, n_bins)))


def bootstrap(fn: Callable[[np.ndarray], float], n: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Item-level bootstrap of fn(index_array). Resamples with replacement; draws that give NaN are counted and dropped.
    For a paired difference, let fn compute both statistics on the same indices."""
    rng = np.random.default_rng(seed)
    point = fn(np.arange(n))
    draws = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=float)
    ok = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.size else (NAN, NAN))
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi), "n_nan": int(draws.size - ok.size)}


def cramers_v(x, y) -> dict:
    """Cramer's V between two categorical variables (any hashable values), with the chi-square test of independence.
    V = sqrt(chi2 / (n * (min(rows, cols) - 1))); 0 = no association, 1 = each value of x determines y. NaN when either variable is constant.
    v_corrected is the bias-corrected V (Bergsma 2013), the one to compare across tables of different size or n."""
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    rows, cols = sorted(set(x), key=str), sorted(set(y), key=str)
    n = len(x)
    if n == 0 or len(rows) < 2 or len(cols) < 2:
        return {"v": NAN, "v_corrected": NAN, "chi2": NAN, "dof": 0, "p": NAN, "n": n}
    table = np.zeros((len(rows), len(cols)))
    ri, ci = {v: i for i, v in enumerate(rows)}, {v: i for i, v in enumerate(cols)}
    for a, b in zip(x, y):
        table[ri[a], ci[b]] += 1
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    dof = (len(rows) - 1) * (len(cols) - 1)
    from scipy.stats import chi2 as chi2_dist

    r, c = len(rows), len(cols)
    # Bergsma (2013): V is biased upward under independence and the bias grows with the table and shrinks with n, so V of a 10 x 4 table at
    # n = 324 and of a 4 x 4 table at n = 2000 are not comparable. The corrected V is.
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1)) if n > 1 else 0.0
    r_c, c_c = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    corrected = float(np.sqrt(phi2 / min(r_c - 1, c_c - 1))) if min(r_c, c_c) > 1 else NAN
    return {"v": float(np.sqrt(chi2 / (n * (min(r, c) - 1)))), "v_corrected": corrected, "chi2": chi2, "dof": dof, "p": float(chi2_dist.sf(chi2, dof)), "n": n}


def bootstrap_stratified(fn: Callable[[np.ndarray], float], strata, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap that resamples items within their strata (subject, template) and keeps every stratum's size, so the interval reflects a
    stratified sample instead of an independent one. With one stratum it is the ordinary item-level bootstrap."""
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == g) for g in sorted(set(strata.tolist()))]
    rng = np.random.default_rng(seed)
    point = fn(np.arange(len(strata)))
    draws = np.array([fn(np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])) for _ in range(n_boot)], dtype=float)
    ok = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.size else (NAN, NAN))
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi), "n_nan": int(draws.size - ok.size)}
