"""Check a run directory before it is summarised: every condition of the protocol is present with all its rows, and every file still has the
SHA-256 recorded when it was completed. Returns the problems instead of raising, so a caller can report them all.

    python experiments/paper1/verify_runs.py            # every finished directory under runs/sweep
    python experiments/paper1/verify_runs.py --exp replication_w2
"""

import argparse

import json
import sys

from common import RUNS
from protocol import conditions_for
from run_sweep import file_sha256, rows_present


def expected_rows(dataset: str, n_items: int, suite: str = "sweep") -> dict[str, int]:
    """Rows each condition must have for a dataset of n_items items."""
    return {c.name: min(c.subset, n_items) if c.subset else n_items for c in conditions_for(dataset, suite)}


def verify_directory(directory, expected: dict[str, int]) -> list[str]:
    problems, records = [], {}
    log = directory / "conditions.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["condition"]] = rec  # the last record of a condition is the one that counts
    else:
        problems.append("no conditions.jsonl: the run predates per-condition hashes (run bind_existing.py)")
    for name, n in expected.items():
        path = directory / f"{name}.jsonl"
        if not path.exists():
            problems.append(f"{name}: file missing")
            continue
        have = rows_present(path, name)
        if have != n:
            problems.append(f"{name}: {have} of {n} rows")
        rec = records.get(name)
        if rec is None and log.exists():
            problems.append(f"{name}: no hash recorded")
        elif rec is not None and rec["rows_sha256"] != file_sha256(path):
            problems.append(f"{name}: the file changed after it was recorded")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="sweep", help="runs/<exp>: sweep or replication_<wording>")
    exp = ap.parse_args().exp
    bad = 0
    for model_dir in sorted(p for p in (RUNS / exp).iterdir() if p.is_dir()):
        for d in sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "DONE").exists()):
            meta = json.loads((d / "meta.json").read_text())
            problems = verify_directory(d, expected_rows(meta["dataset"], meta["n"], meta.get("suite", "sweep")))
            print(f"{'ok ' if not problems else 'BAD'} {d.relative_to(RUNS)}" + ("" if not problems else f": {problems[:3]}"))
            bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
