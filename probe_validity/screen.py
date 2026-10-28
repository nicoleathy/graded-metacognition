"""The validity screen: eight checks of one cell (model x probe x dataset x decoding), each with a frozen rule.

S1 parse validity, S2 label disjointness, S3 usage, S4 relabelling invariance, S5 decoding invariance, S6 signal and
monotonicity, S7 convergent validity, S8 stratum-prior control. A check returns PASS, WARN, FAIL or NA (not enough data
or no comparator). The overall verdict is FAIL if S1-S6 contain a FAIL, WARN if anything warns or a core check is NA,
PASS otherwise. Thresholds live in ScreenConfig; they are tuned on the synthetic simulators, then frozen (config hash).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from .provenance import sha256_of
from .scales import Scale, label_overlap

CORE = ("S1", "S2", "S3", "S6")


@dataclass(frozen=True)
class ScreenConfig:
    s1_pass: float = 0.95  # strict parse rate
    s1_fail: float = 0.80
    s3_min_entropy: float = 0.30  # normalised entropy of the grade distribution
    s3_max_top_share: float = 0.90
    s4_min_kappa: float = 0.60  # quadratic weighted kappa between semantic grades under different label mappings
    s4_min_mappings: int = 3
    s4_spread_factor: float = 2.0  # AUROC spread across mappings must stay within this many DeLong half-widths
    s5_min_kappa: float = 0.60
    s7_min_abs_rho: float = 0.10
    min_valid: int = 50  # fewer valid replies than this: the checks that need them are NA
    alpha: float = 0.05

    def sha256(self) -> str:
        return sha256_of(asdict(self))


def load_frozen(path: Path) -> ScreenConfig:
    """The thresholds frozen after validation on the simulators; real cells are screened only with these."""
    blob = json.loads(Path(path).read_text())
    cfg = ScreenConfig(**blob["config"])
    if cfg.sha256() != blob["sha256"]:
        raise ValueError("frozen screen config was edited: its hash does not match")
    return cfg


@dataclass
class Cell:
    scale: Scale
    answer_labels: tuple[str, ...]  # () for open questions
    correct: np.ndarray  # (n,) direct-answer correctness
    grade: np.ndarray  # (n,) strict semantic grade, NaN where the reply is invalid
    mappings: dict[str, np.ndarray] = field(default_factory=dict)  # other label mappings, same items, semantic grades
    sampled: np.ndarray | None = None  # (n, k) semantic grades of k sampled decodes
    reference: np.ndarray | None = None  # (n,) answer log-probability, for S7
    reference_mean: np.ndarray | None = None  # (n,) the same averaged over the answer's tokens (open answers: length-free)
    strata: np.ndarray | None = None  # (n,) subject or template, for S8


@dataclass
class Check:
    name: str
    verdict: str
    value: dict = field(default_factory=dict)
    note: str = ""


def _valid(a) -> np.ndarray:
    return ~np.isnan(np.asarray(a, dtype=float))


def s1_parse(cell: Cell, cfg: ScreenConfig) -> Check:
    rate = float(_valid(cell.grade).mean())
    verdict = "PASS" if rate >= cfg.s1_pass else "FAIL" if rate < cfg.s1_fail else "WARN"
    return Check("S1", verdict, {"valid_rate": rate})


def s2_labels(cell: Cell, cfg: ScreenConfig) -> Check:
    overlap = label_overlap(cell.scale, cell.answer_labels)
    return Check("S2", "FAIL" if overlap else "PASS", {"overlap": list(overlap)})


def s3_usage(cell: Cell, cfg: ScreenConfig) -> Check:
    ok = _valid(cell.grade)
    if ok.sum() < cfg.min_valid:
        return Check("S3", "NA", {"n_valid": int(ok.sum())}, "too few valid replies")
    u = M.usage(cell.grade[ok], cell.scale.n)
    verdict = "PASS" if u["entropy"] >= cfg.s3_min_entropy and u["top_share"] <= cfg.s3_max_top_share else "FAIL"
    return Check("S3", verdict, {k: u[k] for k in ("n", "entropy", "top_share", "counts")})


def s4_relabelling(cell: Cell, cfg: ScreenConfig) -> Check:
    maps = {"own": cell.grade, **cell.mappings}
    if len(maps) < cfg.s4_min_mappings:
        return Check("S4", "NA", {"mappings": len(maps)}, f"needs {cfg.s4_min_mappings} label mappings")
    levels = list(range(cell.scale.n))
    kappas, aurocs, halves = {}, {}, []
    names = list(maps)
    for i, a in enumerate(names):
        ok = _valid(maps[a])
        if ok.sum() >= cfg.min_valid and len(set(cell.correct[ok])) == 2 and len(set(maps[a][ok])) > 1:
            auc, lo, hi = M.delong_ci(cell.correct[ok], maps[a][ok], cfg.alpha)
            aurocs[a] = auc
            halves.append((hi - lo) / 2)
        for b in names[i + 1:]:
            both = ok & _valid(maps[b])
            if both.sum() >= cfg.min_valid:
                kappas[f"{a}~{b}"] = M.weighted_kappa(maps[a][both].astype(int), maps[b][both].astype(int), levels)
    if not kappas:
        return Check("S4", "NA", {}, "too few jointly valid replies")
    clean = {k: (-1.0 if np.isnan(v) else v) for k, v in kappas.items()}
    worst = min(clean.values())
    own = [v for k, v in clean.items() if k.startswith("own~")]
    spread = (max(aurocs.values()) - min(aurocs.values())) if len(aurocs) > 1 else 0.0
    within = len(halves) == 0 or spread <= cfg.s4_spread_factor * max(halves)
    verdict = "PASS" if worst >= cfg.s4_min_kappa and within else "FAIL"
    # S4 judges the whole family of label mappings: the worst pair decides, and it need not involve the probe being screened.
    # worst_pair and own_min_kappa (the worst pair that includes this probe) are diagnostics; neither changes the verdict.
    return Check("S4", verdict, {"min_kappa": float(worst), "worst_pair": min(clean, key=clean.get), "own_min_kappa": float(min(own)) if own else None,
                                 "kappas": kappas, "auroc_spread": float(spread), "auroc_half_width": float(max(halves)) if halves else None,
                                 "scope": "scale family"})


def s5_decoding(cell: Cell, cfg: ScreenConfig) -> Check:
    if cell.sampled is None:
        return Check("S5", "NA", {}, "no sampled decodes")
    levels, kappas = list(range(cell.scale.n)), []
    for j in range(cell.sampled.shape[1]):
        both = _valid(cell.grade) & _valid(cell.sampled[:, j])
        if both.sum() >= cfg.min_valid:
            kappas.append(M.weighted_kappa(cell.grade[both].astype(int), cell.sampled[both, j].astype(int), levels))
    if not kappas:
        return Check("S5", "NA", {}, "too few jointly valid replies")
    mean = float(np.nan_to_num(kappas, nan=-1.0).mean())
    return Check("S5", "PASS" if mean >= cfg.s5_min_kappa else "FAIL", {"mean_kappa": mean, "draws": len(kappas)})


def s6_signal(cell: Cell, cfg: ScreenConfig) -> Check:
    ok = _valid(cell.grade)
    c, g = cell.correct[ok], cell.grade[ok]
    if ok.sum() < cfg.min_valid or len(set(c)) < 2:
        return Check("S6", "NA", {"n_valid": int(ok.sum())}, "too few valid replies or one outcome class")
    auc, lo, hi = M.delong_ci(c, g, cfg.alpha)
    rho = M.spearman(g, c)
    table = M.reliability(c, g, cell.scale.n)
    verdict = "PASS" if lo > 0.5 and rho > 0 else "FAIL"
    return Check("S6", verdict, {"auroc": auc, "lo": lo, "hi": hi, "rho": rho, "reliability": table})


def s7_convergent(cell: Cell, cfg: ScreenConfig) -> Check:
    """Rank association with the model's own answer likelihood. The verdict is an absolute-association check (|rho| >= threshold),
    as frozen: a strongly negative rho passes it. The sign is therefore reported, with the verdict a check that required positive
    convergence would give (sensitivity analysis), and, when stored, rho with the token-averaged likelihood, which does not depend on
    the length of an open answer."""
    if cell.reference is None:
        return Check("S7", "NA", {}, "no answer log-probability")
    ok = _valid(cell.grade) & ~np.isnan(cell.reference)
    if ok.sum() < cfg.min_valid:
        return Check("S7", "NA", {}, "too few valid replies")
    rho = M.spearman(cell.grade[ok], cell.reference[ok])
    value = {"rho": rho, "sign": "positive" if rho > 0 else "negative" if rho < 0 else "none",
             "verdict_if_positive_required": "PASS" if rho >= cfg.s7_min_abs_rho else "WARN"}
    if cell.reference_mean is not None:
        ok_mean = _valid(cell.grade) & ~np.isnan(cell.reference_mean)
        if ok_mean.sum() >= cfg.min_valid:
            value["rho_token_mean"] = M.spearman(cell.grade[ok_mean], cell.reference_mean[ok_mean])
    return Check("S7", "WARN" if not abs(rho) >= cfg.s7_min_abs_rho else "PASS", value)


def s8_stratum(cell: Cell, cfg: ScreenConfig, n_boot: int = 200, seed: int = 0) -> Check:
    if cell.strata is None:
        return Check("S8", "NA", {}, "no strata")
    ok = _valid(cell.grade)
    if ok.sum() < cfg.min_valid:
        return Check("S8", "NA", {}, "too few valid replies")
    c, g, s = cell.correct[ok], cell.grade[ok], cell.strata[ok]
    boot = M.bootstrap(lambda idx: M.auroc_within(c[idx], g[idx], s[idx]), len(c), n_boot, seed, cfg.alpha)
    return Check("S8", "PASS" if boot["lo"] > 0.5 else "WARN", {"auroc_within": boot["estimate"], "lo": boot["lo"], "hi": boot["hi"]})


def screen(cell: Cell, cfg: ScreenConfig | None = None, with_s8: bool = False) -> dict:
    cfg = cfg or ScreenConfig()
    checks = [s1_parse(cell, cfg), s2_labels(cell, cfg), s3_usage(cell, cfg), s4_relabelling(cell, cfg),
              s5_decoding(cell, cfg), s6_signal(cell, cfg), s7_convergent(cell, cfg)]
    checks.append(s8_stratum(cell, cfg) if with_s8 else Check("S8", "NA", {}, "not requested"))
    by_name = {c.name: c for c in checks}
    verdicts = {name: c.verdict for name, c in by_name.items()}
    if any(verdicts[k] == "FAIL" for k in ("S1", "S2", "S3", "S4", "S5", "S6")):
        overall = "FAIL"
    elif any(v == "WARN" for v in verdicts.values()) or any(verdicts[k] == "NA" for k in CORE):
        overall = "WARN"
    else:
        overall = "PASS"
    not_evaluated = [k for k, v in verdicts.items() if v == "NA"]
    missing_core = [k for k in ("S1", "S2", "S3", "S4", "S5", "S6") if k in not_evaluated]
    complete = not missing_core
    # A PASS with checks not evaluated is a partial pass: nothing failed that was looked at, and nothing certifies the rest.
    label = overall if complete else f"{overall} (not evaluated: {', '.join(missing_core)})"
    return {"overall": overall, "complete": complete, "label": label, "verdicts": verdicts, "checks": by_name, "not_evaluated": not_evaluated}
