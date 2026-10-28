"""Power and false-alarm behaviour of the frozen screen over the whole range that matters for the results: signal strength from nearly
chance to strong, the actual sample sizes of the datasets, a sound probe whose labels overlap the answer labels, a probe that is sound
on its own scale but unstable under relabelling, and skewed grade usage. It complements screen_operating_characteristics.py, whose
"zero false alarms" summary covers population AUROC >= 0.70 only (methodology audit, 2026-10-27).

What the numbers can say: a PASS requires demonstrated above-chance discrimination (S6), so a sound probe with a weak signal is often not
passed at small n. That is a failure to establish enough evidence, not evidence that the probe is defective. Zero failures in N runs are
reported with their exact one-sided 95% upper bound, not as a guarantee.

    python experiments/paper1/screen_power.py
"""

import json
import time
from dataclasses import replace

import numpy as np
from common import ROOT
from scipy.stats import beta

from probe_validity import metrics as M
from probe_validity import report
from probe_validity.screen import load_frozen, screen
from probe_validity.simulators import Spec, simulate_spec

SEEDS = 300
OUT = ROOT / "experiments/paper1/results/screen_power.json"
FROZEN = ROOT / "experiments/paper1/frozen/screen_config.json"
SIZES = (324, 500, 969, 1000, 2000)  # SuperGPQA slice, pilot size, MMLU-Pro engineering, WebQuestions, MMLU and TriviaQA
LATENT_SD = (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)  # larger = weaker signal


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def zero_count_upper_bound(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound on a rate when none of n runs failed: 1 - alpha ** (1 / n)."""
    return round(1 - alpha ** (1 / n), 5)


def population_auroc(spec: Spec) -> float:
    big = simulate_spec(replace(spec, n=100_000, noncompliance=0.0, overlap=False), seed=10_000)
    return round(float(M.auroc(big.correct, big.grade)), 4)


def run(spec: Spec, cfg) -> dict:
    overall, signal_pass, own_pass, s2_restricted_pass, failing = {"PASS": 0, "WARN": 0, "FAIL": 0}, 0, 0, 0, {}
    for seed in range(SEEDS):
        cell = simulate_spec(spec, seed)
        result = screen(cell, cfg, with_s8=False)
        overall[result["overall"]] += 1
        for check, verdict in result["verdicts"].items():
            failing[check] = failing.get(check, 0) + (verdict == "FAIL")
        signal_pass += report.tiers(result["verdicts"])["signal"] == "PASS"
        v = report.variants(result, cell.correct, cell.grade, 4, cfg)["overall"]
        own_pass += v["own_replies_only"] == "PASS"
        s2_restricted_pass += v["s2_as_design_restriction"] == "PASS"
    return {"p_pass": overall["PASS"] / SEEDS, "p_warn": overall["WARN"] / SEEDS, "p_fail": overall["FAIL"] / SEEDS, "p_fail_ci": clopper_pearson(overall["FAIL"], SEEDS),
            "p_signal_tier_pass": signal_pass / SEEDS, "p_pass_if_s2_is_a_design_restriction": s2_restricted_pass / SEEDS,
            "p_pass_on_own_replies_only": own_pass / SEEDS, "check_fail_rates": {k: v / SEEDS for k, v in sorted(failing.items())}}


def main() -> None:
    cfg = load_frozen(FROZEN)
    t0 = time.time()
    out = {"seeds_per_setting": SEEDS, "config_sha256": cfg.sha256(), "sizes": SIZES, "curves": {}, "summary": {}}

    curve = []
    for sd in LATENT_SD:
        auroc = population_auroc(Spec(latent_sd=sd))
        for n in SIZES:
            row = run(Spec(n=n, latent_sd=sd), cfg)
            curve.append({"latent_sd": sd, "population_auroc": auroc, "n": n, **row})
        print(f"signal sweep latent_sd={sd}: population AUROC {auroc} ({time.time() - t0:.0f}s)", flush=True)
    out["curves"]["sound_probe_signal_by_n"] = curve

    others = {
        "overlapping_labels_sound_probe": Spec(overlap=True),  # S2 fails by design, whatever the signal
        "format_specific_probe": Spec(mapping_sd=0.35),  # sound on its own scale, unstable under relabelling (S4)
        "top_heavy_usage": Spec(cuts=(0.1, 0.2, 0.35)),  # most replies in the top grade, as models tend to answer
        "bottom_heavy_usage": Spec(cuts=(0.55, 0.75, 0.9)),
    }
    for name, base in others.items():
        rows = []
        for n in (500, 1000, 2000):
            spec = replace(base, n=n)
            rows.append({"n": n, "population_auroc": population_auroc(spec), **run(spec, cfg)})
        out["curves"][name] = rows
        print(f"{name} ({time.time() - t0:.0f}s)", flush=True)

    # Summary: per n, the weakest population AUROC at which a sound probe passes in at least 95% of runs, and where it fails most runs.
    for n in SIZES:
        rows = sorted((r for r in curve if r["n"] == n), key=lambda r: r["population_auroc"])
        out["summary"][f"n={n}"] = {
            "weakest_population_auroc_passing_in_95pct_of_runs": next((r["population_auroc"] for r in rows if r["p_pass"] >= 0.95), None),
            "population_auroc_values_failing_in_over_half_of_runs": [r["population_auroc"] for r in rows if r["p_fail"] > 0.5],
            "false_alarm_rate_at_population_auroc_at_least_0.70": {"max_p_fail": max((r["p_fail"] for r in rows if r["population_auroc"] >= 0.70), default=None)}}
    strong = [r for r in curve if r["population_auroc"] >= 0.70]
    runs = len(strong) * SEEDS
    failures = sum(round(r["p_fail"] * SEEDS) for r in strong)
    out["summary"]["strong_sound_probes"] = {
        "settings": len(strong), "runs": runs, "failures": failures,
        "per_setting": [{"latent_sd": r["latent_sd"], "n": r["n"], "population_auroc": r["population_auroc"], "failures": round(r["p_fail"] * SEEDS), "runs": SEEDS} for r in strong],
        # The bound that is justified: one setting, SEEDS independent runs. Runs do use independent random streams (the generator is seeded with the run index and a
        # hash of the setting, so the same index in two settings gives different draws), but the settings are different distributions, so the pooled count below is a
        # description of these ten settings and not a false-alarm rate of real probes.
        "per_setting_exact_upper_bound_95pct_if_zero": zero_count_upper_bound(SEEDS),
        "pooled_exact_upper_bound_95pct_if_zero": zero_count_upper_bound(runs) if failures == 0 else None,
        "pooled_bound_is_descriptive": "assumes independent runs and describes only these settings"}
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
