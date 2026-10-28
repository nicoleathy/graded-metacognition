"""Validate the screen on seven synthetic probes with known ground truth (1,000 seeds each) and write the result.
A single disagreement with the expected verdicts would mean the screen logic is wrong, whatever any model does.

    python experiments/paper1/validate_screen.py
"""

import collections
import json
import time
from dataclasses import asdict

import numpy as np
from common import ROOT

from probe_validity.provenance import git_state
from probe_validity.screen import ScreenConfig, screen
from probe_validity.simulators import EXPECTED, simulate

SEEDS = 1000
OUT = ROOT / "experiments/paper1/results/screen_validation.json"
FROZEN = ROOT / "experiments/paper1/frozen/screen_config.json"


def main() -> None:
    cfg = ScreenConfig()
    report = {"seeds": SEEDS, "n_items": 1000, "config": asdict(cfg), "config_sha256": cfg.sha256(), "simulators": {}}
    for name, expected in EXPECTED.items():
        mismatches, overall = [], collections.Counter()
        margins = collections.defaultdict(list)
        for seed in range(SEEDS):
            result = screen(simulate(name, seed), cfg)
            got = {"overall": result["overall"], **result["verdicts"]}
            overall[result["overall"]] += 1
            mismatches += [{"seed": seed, "check": k, "expected": v, "got": got[k]} for k, v in expected.items() if got[k] != v]
            values = {c: result["checks"][c].value for c in ("S1", "S3", "S4", "S5", "S6")}
            margins["valid_rate"].append(values["S1"].get("valid_rate"))
            margins["entropy"].append(values["S3"].get("entropy"))
            margins["min_kappa_s4"].append(values["S4"].get("min_kappa"))
            margins["mean_kappa_s5"].append(values["S5"].get("mean_kappa"))
            margins["auroc_lower_s6"].append(values["S6"].get("lo"))
        report["simulators"][name] = {
            "expected": expected, "runs": SEEDS, "concordant_runs": SEEDS - len({m["seed"] for m in mismatches}),
            "mismatches": mismatches[:10], "overall_verdicts": dict(overall),
            "margins": {k: {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
                        for k, v in margins.items() if not all(x is None or np.isnan(x) for x in v)},
        }
        print(f"{name:17} concordant {report['simulators'][name]['concordant_runs']}/{SEEDS}  overall {dict(overall)}", flush=True)
    report["all_concordant"] = all(s["concordant_runs"] == SEEDS for s in report["simulators"].values())
    report["git"], report["finished"] = git_state(ROOT), time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(report, indent=1, default=float))
    FROZEN.write_text(json.dumps({"config": asdict(cfg), "sha256": cfg.sha256(),
                                  "note": "Thresholds frozen after validation on synthetic probes and before any real screen."}, indent=1))
    print("all concordant:", report["all_concordant"], "| config sha256", cfg.sha256()[:16])


if __name__ == "__main__":
    main()
