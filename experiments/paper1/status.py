"""Progress of the sweep: rows written per model and dataset against what the protocol requires.

    python experiments/paper1/status.py
"""

import json
import time

from common import MANIFESTS, RUNS
from protocol import DATASET_TABLE, conditions_for


def n_rows(path) -> int:
    ids = set()
    if path.exists():
        for line in path.read_text().split("\n"):  # not splitlines(): a reply may hold a Unicode line separator
            try:
                ids.add(json.loads(line)["i"])
            except (json.JSONDecodeError, KeyError):
                pass
    return len(ids)


def main() -> None:
    base = RUNS / "sweep"
    if not base.exists():
        print("no sweep runs yet")
        return
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        done = total = 0
        parts = []
        for dataset in DATASET_TABLE:
            d = model_dir / dataset
            n = json.loads((MANIFESTS / f"{DATASET_TABLE[dataset][0]}.json").read_text())["n"]
            want = sum(min(c.subset or n, n) for c in conditions_for(dataset))
            have = sum(n_rows(d / f"{c.name}.jsonl") for c in conditions_for(dataset)) if d.exists() else 0
            done, total = done + have, total + want
            parts.append(f"{dataset} {100 * have / want:.0f}%")
        mtimes = [p.stat().st_mtime for p in model_dir.rglob("*.jsonl")]
        age = (time.time() - max(mtimes)) / 60 if mtimes else float("nan")
        print(f"{model_dir.name:42} {100 * done / total:5.1f}%  ({', '.join(parts)})  last write {age:.0f} min ago")


if __name__ == "__main__":
    main()
