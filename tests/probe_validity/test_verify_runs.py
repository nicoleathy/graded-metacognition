import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import pytest

pytest.importorskip("transformers")
import verify_runs  # noqa: E402
from run_sweep import file_sha256  # noqa: E402


def write(directory, name, n):
    path = directory / f"{name}.jsonl"
    path.write_text("".join(json.dumps({"i": i, "condition": name}) + "\n" for i in range(n)))
    return path


def record(directory, names):
    with (directory / "conditions.jsonl").open("w") as fh:
        for name in names:
            fh.write(json.dumps({"condition": name, "rows_sha256": file_sha256(directory / f"{name}.jsonl")}) + "\n")


def test_a_complete_recorded_directory_is_clean(tmp_path):
    write(tmp_path, "a", 5), write(tmp_path, "b", 3)
    record(tmp_path, ["a", "b"])
    assert verify_runs.verify_directory(tmp_path, {"a": 5, "b": 3}) == []


def test_missing_short_changed_and_unrecorded_conditions_are_reported(tmp_path):
    write(tmp_path, "a", 5), write(tmp_path, "b", 2), write(tmp_path, "c", 3)
    record(tmp_path, ["a", "b"])
    with (tmp_path / "a.jsonl").open("a") as fh:  # edited after it was recorded
        fh.write(json.dumps({"i": 99, "condition": "a"}) + "\n")
    problems = verify_runs.verify_directory(tmp_path, {"a": 6, "b": 3, "c": 3, "d": 1})
    text = " | ".join(problems)
    assert "b: 2 of 3 rows" in text and "d: file missing" in text and "c: no hash recorded" in text and "a: the file changed" in text


def test_a_directory_without_hashes_says_how_to_fix_it(tmp_path):
    write(tmp_path, "a", 2)
    assert "bind_existing.py" in verify_runs.verify_directory(tmp_path, {"a": 2})[0]


def test_expected_rows_follow_the_protocol_including_subsets():
    rows = verify_runs.expected_rows("mmlu", 2000)
    assert rows["direct_clean"] == 2000 and rows["letters_sampled0"] == 500 and len(rows) == 25
