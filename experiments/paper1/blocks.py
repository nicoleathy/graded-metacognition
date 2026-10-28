"""Small statistics blocks shared by the analysis modules: a rounded number, an AUROC with its interval and companions, and a paired
AUROC difference. Kept apart so that analysis.py and the robustness blocks can both use them without importing each other."""

import numpy as np
from common import ROOT  # noqa: F401  (puts the repository on sys.path, as every script of this folder expects)

from probe_validity import metrics as M

N_BOOT = 1000


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def auroc_block(correct, score, seed=0, n_boot=N_BOOT):
    c, s = np.asarray(correct), np.asarray(score, dtype=float)
    ok = ~np.isnan(s)
    c, s = c[ok], s[ok]
    if len(c) < 30 or len(set(c)) < 2 or len(set(s)) < 2:
        return None
    boot = M.bootstrap(lambda idx: M.auroc(c[idx], s[idx]), len(c), n_boot, seed)
    return {"auroc": r4(boot["estimate"]), "ci": [r4(boot["lo"]), r4(boot["hi"])], "somers_d": r4(M.somers_d(c, s)),
            "gamma": r4(M.gamma(c, s)), "p_tie": r4(M.tie_share(c, s)), "n": int(len(c))}


def paired_auroc_diff(correct, a, b, seed=0, n_boot=N_BOOT):
    """AUROC(a) - AUROC(b) on the items where both probes gave a valid reply, with a paired bootstrap interval."""
    c, a, b = np.asarray(correct), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    c, a, b = c[ok], a[ok], b[ok]
    if len(c) < 30 or len(set(c)) < 2:
        return None
    boot = M.bootstrap(lambda idx: M.auroc(c[idx], a[idx]) - M.auroc(c[idx], b[idx]), len(c), n_boot, seed)
    return {"diff": r4(boot["estimate"]), "ci": [r4(boot["lo"]), r4(boot["hi"])], "n_common": int(len(c))}


def gamma_star(correct, score):
    """The earlier draft's metric: Goodman-Kruskal gamma rescaled so that 0.5 is chance, (gamma + 1) / 2; NaN when no pair is untied."""
    g = M.gamma(correct, score)
    return None if g is None or np.isnan(g) else r4((g + 1) / 2)


def refit_paired_diff(correct, score, verbal, transform, seed=0, n_boot=N_BOOT):
    """AUROC(transform(score)) - AUROC(verbal) on the items where both exist. `transform(score, verbal)` is fitted on the data it is given (equal-mass
    coarsening, matching the verbal grade frequencies) and is refitted inside every bootstrap resample, so the interval covers the fitting too. The
    point estimate is the one a transform fitted once on all items gives."""
    c, f, v = np.asarray(correct), np.asarray(score, dtype=float), np.asarray(verbal, dtype=float)
    ok = ~np.isnan(f) & ~np.isnan(v)
    c, f, v = c[ok], f[ok], v[ok]
    if len(c) < 30 or len(set(c)) < 2:
        return None
    boot = M.bootstrap(lambda idx: M.auroc(c[idx], transform(f[idx], v[idx])) - M.auroc(c[idx], v[idx]), len(c), n_boot, seed)
    return {"diff": r4(boot["estimate"]), "ci": [r4(boot["lo"]), r4(boot["hi"])], "n_common": int(len(c)), "refit_in_bootstrap": True}
