"""Operating characteristics of the frozen screen on synthetic probes: how often a sound probe is flagged (false alarms), and how
often a defective probe is caught, as a function of defect severity, sample size and simulation setting. This replaces
"all verdicts equal the expected ones" as the evidence about the thresholds: that check (validate_screen.py) only tests the
implementation on settings with wide margins. The thresholds were not tuned on any of these settings; they are the values
proposed in the audit plan, frozen before this study.

    python experiments/paper1/screen_operating_characteristics.py
"""

import collections
import json
import time
from dataclasses import asdict, replace

import numpy as np
from common import ROOT

from probe_validity import metrics as M
from probe_validity.screen import load_frozen, screen
from probe_validity.simulators import Spec, simulate_spec

SEEDS = 300
OUT = ROOT / "experiments/paper1/results/screen_operating_characteristics.json"
FROZEN = ROOT / "experiments/paper1/frozen/screen_config.json"

DEV = Spec()  # the setting the seven logic-check simulators use
HELD_OUT = {"uniform_p": Spec(beta=(1.0, 1.0)), "hard_items": Spec(beta=(2.0, 5.0)), "other_cuts": Spec(cuts=(0.2, 0.5, 0.8))}
SIZES = (250, 1000, 4000)
# parameter swept, values, the check that should catch it
SWEEPS = {
    "sound_probe_weaker_signal": ("latent_sd", (0.05, 0.1, 0.2, 0.35, 0.5, 0.8), None),
    "noncompliance": ("noncompliance", (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5), "S1"),
    "label_copy": ("copy_fraction", (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0), "S4"),
    "label_prior": ("prior_weight", (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7), "S4"),
    "collapse": ("collapse", (0.0, 0.5, 0.8, 0.9, 0.95, 1.0), "S3"),
    "inverted_scale": ("anti_fraction", (0.0, 0.25, 0.5, 0.75, 1.0), "S6"),
    "sampling_noise": ("sample_sd", (0.06, 0.1, 0.2, 0.4, 0.8), "S5"),
}


def population_auroc(spec: Spec) -> float:
    big = simulate_spec(replace(spec, n=100_000, noncompliance=0.0), seed=10_000)
    return float(M.auroc(big.correct, big.grade))


def run_setting(spec: Spec, cfg, designated: str | None) -> dict:
    overall, failed, designated_fail, designated_flag = collections.Counter(), collections.Counter(), 0, 0
    for seed in range(SEEDS):
        result = screen(simulate_spec(spec, seed), cfg)
        overall[result["overall"]] += 1
        for check, verdict in result["verdicts"].items():
            failed[check] += verdict == "FAIL"
        designated_fail += designated is not None and result["verdicts"][designated] == "FAIL"
        designated_flag += designated is not None and result["verdicts"][designated] in ("FAIL", "WARN")
    lo = lambda k: M.wilson_ci(k, SEEDS)  # noqa: E731
    row = {"p_pass": overall["PASS"] / SEEDS, "p_warn": overall["WARN"] / SEEDS, "p_fail": overall["FAIL"] / SEEDS,
           "p_fail_ci": [round(x, 4) for x in lo(overall["FAIL"])], "check_fail_rates": {k: failed[k] / SEEDS for k in sorted(failed)}}
    if designated:
        row["p_designated_check_fails"] = designated_fail / SEEDS
        row["p_designated_ci"] = [round(x, 4) for x in lo(designated_fail)]
        row["p_designated_check_flags"] = designated_flag / SEEDS  # FAIL or WARN
    return row


def main() -> None:
    cfg = load_frozen(FROZEN)
    t0 = time.time()
    report = {"seeds_per_setting": SEEDS, "config_sha256": cfg.sha256(), "config": asdict(cfg), "settings": {}}
    bases = {"dev": DEV, **HELD_OUT}
    for sweep, (param, values, designated) in SWEEPS.items():
        report["settings"][sweep] = []
        for value in values:
            for base_name, base in bases.items():
                for n in SIZES if base_name == "dev" else (1000,):
                    spec = replace(base, n=n, **{param: value})
                    row = run_setting(spec, cfg, designated)
                    row.update(param=param, value=value, base=base_name, n=n, held_out=base_name != "dev" or n != 1000)
                    if sweep == "sound_probe_weaker_signal":
                        row["population_auroc"] = round(population_auroc(spec), 4)
                    report["settings"][sweep].append(row)
        print(f"{sweep}: {len(report['settings'][sweep])} settings ({time.time() - t0:.0f}s)", flush=True)

    sound = [r for r in report["settings"]["sound_probe_weaker_signal"] if r["population_auroc"] >= 0.70]
    report["summary"] = {
        "sound_probes_with_population_auroc_at_least_0.70": {
            "settings": len(sound), "false_alarm_rate_overall_fail_mean": round(float(np.mean([r["p_fail"] for r in sound])), 4),
            "false_alarm_rate_overall_fail_max": round(float(max(r["p_fail"] for r in sound)), 4),
            "not_pass_rate_mean": round(float(np.mean([1 - r["p_pass"] for r in sound])), 4)},
    }
    for sweep, (param, values, designated) in SWEEPS.items():
        if designated:
            rows = [r for r in report["settings"][sweep] if r["base"] == "dev" and r["n"] == 1000]
            first = next((r["value"] for r in rows if r["p_designated_check_fails"] >= 0.95), None)
            report["summary"][f"{sweep}_smallest_severity_caught_in_95pct_of_runs_dev_n1000"] = first
            flagged = next((r["value"] for r in rows if r["p_designated_check_flags"] >= 0.95), None)
            report["summary"][f"{sweep}_smallest_severity_flagged_warn_or_fail_in_95pct_of_runs_dev_n1000"] = flagged
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
