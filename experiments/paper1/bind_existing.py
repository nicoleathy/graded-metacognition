"""Record the SHA-256 of every condition file in the finished run directories that were written before per-condition hashing existed.

The record is marked retroactive: it says what the files contain now, not which invocation wrote them (for those the run metadata and
meta_history.jsonl are the only account). Directories that are still being written (no DONE marker) and directories that already have a
conditions.jsonl are left alone. Nothing is modified or deleted; a new conditions.jsonl is added.

    python experiments/paper1/bind_existing.py
"""

import json
import time

from common import RUNS
from run_sweep import file_sha256, rows_present

from run_sweep import BOOKKEEPING as SKIP  # noqa: E402


def main() -> None:
    bound = 0
    for model_dir in sorted(p for p in (RUNS / "sweep").iterdir() if p.is_dir()):
        for d in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (d / "DONE").exists() or (d / "conditions.jsonl").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            with (d / "conditions.jsonl").open("w") as fh:
                for path in sorted(d.glob("*.jsonl")):
                    if path.name in SKIP:
                        continue
                    fh.write(json.dumps({"condition": path.stem, "n_items": rows_present(path, path.stem), "rows_sha256": file_sha256(path), "invocation": None,
                                         "code_sha256": meta.get("code_sha256"), "retroactive": True, "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            bound += 1
            print(f"bound {d.relative_to(RUNS)}")
    print(f"{bound} directories bound")


if __name__ == "__main__":
    main()
