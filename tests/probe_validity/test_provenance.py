import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

pytest.importorskip("transformers")
import run_sweep  # noqa: E402

from probe_validity.generate import read_jsonl, run_condition  # noqa: E402


class StubRunner:
    """Just enough of Runner for run_condition: batches of two, one fixed reply per prompt."""

    def plan(self, prompts, batch_size):
        return [list(range(i, min(i + 2, len(prompts)))) for i in range(0, len(prompts), 2)]

    def generate_batch(self, prompts, max_new_tokens, labels, sampling, seed=0):
        return [{"text": "B", "n_new": 1, "finish": "eos"} for _ in prompts]


QIDS, PROMPTS = ["q0", "q1", "q2", "q3"], ["p0", "p1", "p2", "p3"]


def test_a_resume_continues_the_same_rows(tmp_path):
    path = tmp_path / "c.jsonl"
    run_condition(StubRunner(), "c", QIDS[:2], PROMPTS[:2], path, log=lambda *_: None)  # an interrupted run finished the first batch
    rows = run_condition(StubRunner(), "c", QIDS, PROMPTS, path, log=lambda *_: None)
    assert len(rows) == 4 and len(read_jsonl(path)) == 4


@pytest.mark.parametrize("qids,prompts", [(["q0", "OTHER", "q2", "q3"], PROMPTS), (QIDS, ["p0", "CHANGED", "p2", "p3"])])
def test_a_resume_refuses_rows_from_other_items_or_prompts(tmp_path, qids, prompts):
    path = tmp_path / "c.jsonl"
    run_condition(StubRunner(), "c", QIDS, PROMPTS, path, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        run_condition(StubRunner(), "c", qids, prompts, path, log=lambda *_: None)


def test_a_resume_refuses_more_stored_rows_than_the_condition_has_items(tmp_path):
    path = tmp_path / "c.jsonl"
    run_condition(StubRunner(), "c", QIDS, PROMPTS, path, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        run_condition(StubRunner(), "c", QIDS[:2], PROMPTS[:2], path, log=lambda *_: None)


def test_config_changes_names_the_settings_that_differ():
    old = {"model_id": "m", "revision": "r1", "generation_config": {"do_sample": False, "temperature": None}, "torch": "2.8.0", "started": "t0"}
    assert run_sweep.config_changes(old, dict(old, started="t1", code_sha256="x")) == []  # time and code digest may change
    assert run_sweep.config_changes(old, dict(old, revision="r2")) == ["revision"]
    new = dict(old, generation_config={"do_sample": True, "temperature": 0.7})
    assert run_sweep.config_changes(old, new) == ["generation_config"]


def test_rows_present_counts_distinct_items_of_the_named_condition(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"i": 0, "condition": "c"}, {"i": 0, "condition": "c"}, {"i": 1, "condition": "other"}, {"i": 2, "condition": "c"}]))
    assert run_sweep.rows_present(path, "c") == 2
    assert run_sweep.rows_present(tmp_path / "missing.jsonl", "c") == 0
