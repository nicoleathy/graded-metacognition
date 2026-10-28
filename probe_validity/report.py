"""Reporting layers over the frozen screen. Nothing here changes a verdict of screen(): the frozen rules stay as they were, and these
functions show what the verdict is made of (methodology audit, 2026-10-27).

* tiers: S2 is a design restriction (the confidence labels are also answer labels), S4 and S5 are evidence about stability, and S1, S3,
  S6 (with S7, S8 as warnings) are properties of the probe's own replies. The overall FAIL mixes all three.
* monotonicity: S6 requires above-chance discrimination and a positive rank correlation. It does not test that each higher grade is
  more accurate than the one below; this does, with uncertainty.
* variants: the verdict when S2 is read as a design restriction instead of a failure, when S4 judges only the probe's own scale pairs,
  and when S6 also requires monotone grades. Reported next to the frozen verdict, never in place of it.
* threshold_sensitivity: the verdict under each frozen threshold moved a stated amount in each direction.
"""

import dataclasses

import numpy as np

from . import metrics as M
from .screen import CORE, Cell, ScreenConfig, screen

NAN = float("nan")
INDIVIDUAL = ("S1", "S3", "S6")  # the probe's own replies: compliance, usage, positive discrimination


def combine(verdicts: dict) -> str:
    """The screen's decision rule as a function of the check verdicts (the same rule as screen())."""
    if any(verdicts.get(k) == "FAIL" for k in ("S1", "S2", "S3", "S4", "S5", "S6")):
        return "FAIL"
    if any(v == "WARN" for v in verdicts.values()) or any(verdicts.get(k) == "NA" for k in CORE):
        return "WARN"
    return "PASS"


def tiers(verdicts: dict) -> dict:
    """Three separate readings of one cell: design (S2), stability (S4, S5), signal (S1, S3, S6; S7, S8 only warn)."""
    design = verdicts.get("S2", "NA")
    stab = [verdicts.get(k, "NA") for k in ("S4", "S5")]
    stability = "FAIL" if "FAIL" in stab else "NA" if all(v == "NA" for v in stab) else "PASS"
    own = [verdicts.get(k, "NA") for k in INDIVIDUAL]
    if "FAIL" in own:
        signal = "FAIL"
    elif verdicts.get("S6", "NA") == "NA":
        signal = "NA"
    elif "WARN" in own or "WARN" in (verdicts.get("S7"), verdicts.get("S8")):
        signal = "WARN"
    else:
        signal = "PASS"
    return {"design": design, "stability": stability, "signal": signal}


def _newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 of two proportions with its 95% Newcombe (Wilson-based) interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = M.wilson_ci(k1, n1)
    l2, u2 = M.wilson_ci(k2, n2)
    d = p1 - p2
    return d, d - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), d + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def monotonicity(correct, grade, n_levels: int, min_n: int = 30) -> dict:
    """Does accuracy rise with the grade? Adjacent used levels (at least min_n replies each) are compared, higher minus lower; a reversal
    is a difference whose 95% interval lies below zero (the higher grade is credibly less accurate). largest_reversal is the size of
    the biggest one. Counts of replies and accuracies per grade are included."""
    correct, grade = np.asarray(correct), np.asarray(grade, dtype=float)
    ok = ~np.isnan(grade)  # replies without a grade take no part
    c, g = M._arrays(correct[ok], grade[ok])
    table = M.reliability(c, g, n_levels)
    used = [r for r in table if r["n"] >= min_n]
    steps = []
    for lo_row, hi_row in zip(used, used[1:]):
        k_lo, k_hi = int(round(lo_row["accuracy"] * lo_row["n"])), int(round(hi_row["accuracy"] * hi_row["n"]))
        d, lo, hi = _newcombe(k_hi, hi_row["n"], k_lo, lo_row["n"])
        steps.append({"from": lo_row["grade"], "to": hi_row["grade"], "diff": d, "lo": lo, "hi": hi, "reversal": bool(hi < 0)})
    reversals = [s for s in steps if s["reversal"]]
    return {"table": table, "steps": steps, "n_reversals": len(reversals), "monotone": not reversals,
            "largest_reversal": float(max((-s["diff"] for s in reversals), default=0.0)), "levels_compared": len(used)}


def _s4_own(check, cfg: ScreenConfig) -> str:
    """S4 judging only the quadratic weighted kappa of the pairs that include the probe's own scale: the family's other pairs do not decide,
    and neither does the AUROC spread across the family's mappings (a property of the family, not of this probe)."""
    v = check.value
    if check.verdict == "NA":
        return "NA"
    own = v.get("own_min_kappa")
    return "PASS" if own is not None and own >= cfg.s4_min_kappa else "FAIL"


def variants(result: dict, correct, grade, n_levels: int, cfg: ScreenConfig) -> dict:
    """Overall verdicts under alternative readings of the checks, from the stored checks of one screen() result."""
    v = dict(result["verdicts"])
    mono = monotonicity(correct, grade, n_levels)
    s4_own = _s4_own(result["checks"]["S4"], cfg)
    out = {"frozen": result["overall"],
           "s2_as_design_restriction": combine({**v, "S2": "PASS"}),
           "s4_own_pairs_only": combine({**v, "S4": s4_own}),
           "s2_and_s4_relaxed": combine({**v, "S2": "PASS", "S4": s4_own}),
           "s6_requires_monotone": combine({**v, "S6": "FAIL" if v["S6"] == "PASS" and not mono["monotone"] else v["S6"]}),
           "own_replies_only": combine({**v, "S2": "PASS", "S4": "PASS", "S5": "PASS"})}
    return {"overall": out, "s4_own_pairs": s4_own, "monotonicity": mono}


# (field, loosened value, tightened value): each threshold moved by a stated amount in each direction.
SENSITIVITY_MOVES = (
    ("s1_pass", 0.90, 0.99), ("s1_fail", 0.70, 0.90), ("s3_min_entropy", 0.20, 0.40), ("s3_max_top_share", 0.95, 0.85),
    ("s4_min_kappa", 0.50, 0.70), ("s4_spread_factor", 3.0, 1.5), ("s5_min_kappa", 0.50, 0.70), ("s7_min_abs_rho", 0.05, 0.15), ("min_valid", 25, 100),
)


def threshold_sensitivity(cell: Cell, cfg: ScreenConfig, s8: str = "NA") -> dict:
    """Overall verdict of the cell with every threshold loosened or tightened one at a time, and with all moved together. S8 does not
    depend on these thresholds (apart from min_valid), so its verdict from the main screen is carried into every variant."""
    def verdict(c: ScreenConfig) -> str:
        return combine({**screen(cell, c, with_s8=False)["verdicts"], "S8": s8})

    out = {"frozen": verdict(cfg), "single": {}}
    for field, loose, tight in SENSITIVITY_MOVES:
        for tag, value in (("loose", loose), ("tight", tight)):
            out["single"][f"{field}:{tag}"] = verdict(dataclasses.replace(cfg, **{field: value}))
    out["all_loose"] = verdict(dataclasses.replace(cfg, **{f: lo for f, lo, _ in SENSITIVITY_MOVES}))
    out["all_tight"] = verdict(dataclasses.replace(cfg, **{f: hi for f, _, hi in SENSITIVITY_MOVES}))
    out["changed_by_single_move"] = sorted(k for k, v in out["single"].items() if v != out["frozen"])
    return out
