"""A model's reply may hold a Unicode line separator (U+0085 in a Qwen2.5-1.5B TriviaQA answer). The rows are written with the characters as they are,
so a reader that cuts lines with str.splitlines() splits the record in two, drops both halves and reports a finished condition as one row short:
the directory then never gets its DONE marker, and a resumed run regenerates the item and writes the same unreadable row again."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import run_sweep  # noqa: E402
import status  # noqa: E402

from probe_validity.generate import read_jsonl  # noqa: E402

SEPARATORS = ["\x85", " ", " ", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"]  # everything str.splitlines() treats as a line end besides \n and \r


def write(path, rows, tail=""):
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.write(tail)


def test_a_reply_with_a_unicode_line_separator_is_read_back_as_one_row(tmp_path):
    rows = [{"i": i, "condition": "direct_clean", "text": f"before{sep}after"} for i, sep in enumerate(SEPARATORS)]
    path = tmp_path / "direct_clean.jsonl"
    write(path, rows)
    assert [r["text"] for r in read_jsonl(path)] == [r["text"] for r in rows]  # every separator survives inside its record
    assert run_sweep.rows_present(path, "direct_clean") == len(rows)  # so the condition counts as complete and the directory can be marked DONE
    assert status.n_rows(path) == len(rows)


def test_a_line_cut_short_by_an_interrupted_write_is_still_skipped(tmp_path):
    path = tmp_path / "direct_clean.jsonl"
    write(path, [{"i": 0, "condition": "direct_clean", "text": "ok\x85still ok"}], tail='{"i": 1, "condition": "direct_clean", "te')
    assert [r["i"] for r in read_jsonl(path)] == [0]  # the whole row before the cut is kept, the cut line is dropped and its batch is redone
