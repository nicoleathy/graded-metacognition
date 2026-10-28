"""Screening on one half of the items and measuring on the other.

S6 itself requires an AUROC above chance, so a statement such as "cells that pass the screen discriminate better" is partly true
by selection if the screen and the AUROC use the same items. Whenever passing and failing cells are compared, the screen is
applied to one random half (stratified by subject or template) and the quantity of interest is computed on the other half;
the roles are then swapped.
"""

import numpy as np

from . import metrics as M
from .screen import Cell, ScreenConfig, screen


def subset_cell(cell: Cell, idx) -> Cell:
    """The same cell restricted to the items idx (every aligned array is sliced)."""
    idx = np.asarray(idx)
    return Cell(
        cell.scale, cell.answer_labels, cell.correct[idx], cell.grade[idx],
        mappings={k: v[idx] for k, v in cell.mappings.items()},
        sampled=None if cell.sampled is None else cell.sampled[idx],
        reference=None if cell.reference is None else cell.reference[idx],
        reference_mean=None if cell.reference_mean is None else cell.reference_mean[idx],
        strata=None if cell.strata is None else cell.strata[idx],
    )


def stratified_halves(strata, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint index sets of nearly equal size; every stratum is split as evenly as possible. Deterministic in the seed."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    a, b = [], []
    for g in sorted(set(strata.tolist())):
        idx = rng.permutation(np.flatnonzero(strata == g))
        half = (len(idx) + int(rng.integers(0, 2))) // 2  # an odd stratum gives its extra item to a random side
        a.extend(idx[:half].tolist())
        b.extend(idx[half:].tolist())
    return np.sort(np.array(a, dtype=int)), np.sort(np.array(b, dtype=int))


def heldout_auroc(cell: Cell, idx, min_valid: int = 30) -> float:
    """AUROC of the valid replies among the items idx; NaN when too few or only one outcome."""
    g, y = cell.grade[idx], cell.correct[idx]
    ok = ~np.isnan(g)
    if ok.sum() < min_valid or len(set(y[ok])) < 2:
        return float("nan")
    return float(M.auroc(y[ok], g[ok]))


def crossfit(cell: Cell, cfg: ScreenConfig | None = None, seed: int = 0, with_s8: bool = False) -> list[dict]:
    """Two folds: screen on half A and measure on half B, then the reverse. Each record holds the verdict from the screening
    half, the AUROC on the held-out half, and the AUROC on the screening half (the number a same-items analysis would report)."""
    n = len(cell.correct)
    strata = cell.strata if cell.strata is not None else np.zeros(n, dtype=int)
    a, b = stratified_halves(strata, seed)
    out = []
    for fold, (fit, held) in enumerate(((a, b), (b, a))):
        result = screen(subset_cell(cell, fit), cfg, with_s8)
        out.append({"fold": fold, "verdict": result["overall"], "label": result["label"], "complete": result["complete"],
                    "verdicts": result["verdicts"], "auroc_screen_half": heldout_auroc(cell, fit),
                    "auroc_heldout": heldout_auroc(cell, held), "n_screen": int(len(fit)), "n_heldout": int(len(held))})
    return out
