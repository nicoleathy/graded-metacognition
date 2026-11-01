"""The replication suite: the study's interventions on all three multiple-choice sets under three wordings, in its own directories, without touching the sweep."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "paper1"))

import protocol  # noqa: E402
import run_sweep  # noqa: E402
import verify_runs  # noqa: E402


def names(dataset, suite="sweep"):
    return [c.name for c in protocol.conditions_for(dataset, suite)]


def test_the_replication_has_every_control_on_all_three_multiple_choice_sets():
    needed = {"direct_clean", "yesno_raw", "yesno_embedded", "yesno_stem", "letters_embedded", "digits_stem", "words_embedded", "letters_reversed_raw",
              "letters_flip_order_raw", "letters_flip_labels_raw", "letters_optdigits", "digits_optdigits"}
    assert protocol.REPLICATION_DATASETS == ("mmlu", "mmlu_pro_eng", "supergpqa_mfg")
    for dataset in protocol.REPLICATION_DATASETS:
        have = names(dataset, "replication")
        assert needed <= set(have) and len(have) == len(set(have))
        assert "direct_legacy" not in have and "letters_perm_raw" not in have and not any("sampled" in n for n in have)  # those serve the screen only


def test_the_open_question_sets_are_not_part_of_the_replication():
    for dataset in ("triviaqa", "webquestions"):
        with pytest.raises(ValueError):
            protocol.conditions_for(dataset, "replication")
    with pytest.raises(ValueError):
        protocol.conditions_for("mmlu", "other")


def test_the_sweep_is_not_changed_by_the_suite_argument():
    for dataset in protocol.DATASET_TABLE:
        assert names(dataset) == names(dataset, "sweep")
    assert "letters_embedded" not in names("mmlu_pro_eng")  # the study still has the controls on MMLU only
    assert any("sampled" in n for n in names("mmlu")) and "direct_legacy" in names("mmlu")


def test_expected_rows_follow_the_suite():
    sweep, rep = verify_runs.expected_rows("mmlu", 100), verify_runs.expected_rows("mmlu", 100, "replication")
    assert "letters_sampled0" in sweep and "letters_sampled0" not in rep and rep["letters_embedded"] == 100


def test_each_wording_has_its_own_protocol_hash_and_the_sweep_hash_is_untouched():
    code = "import json; from protocol import protocol_record, replication_record; print(json.dumps({'sweep': protocol_record()['sha256'], 'rep': replication_record()}))"
    out = {}
    for w in ("w1", "w2", "w3"):
        env = {**os.environ, "PROBE_WORDING": w, "PYTHONPATH": str(ROOT / "experiments" / "paper1") + os.pathsep + str(ROOT)}
        out[w] = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env, cwd=ROOT).stdout)
    assert len({out[w]["rep"]["sha256"] for w in out}) == 3 and all(out[w]["rep"]["wording"] == w for w in out)
    frozen = json.loads((ROOT / "experiments/paper1/frozen/protocol.json").read_text())["sha256"]
    assert out["w1"]["sweep"] == frozen  # the replication did not change the frozen record
    assert set(out["w1"]["rep"]["datasets"]) == set(protocol.REPLICATION_DATASETS)


def test_the_resume_guard_treats_old_runs_as_sweep_runs_under_the_legacy_wording():
    old = {"model_id": "m", "n": 10}
    assert run_sweep.config_changes(old, {**old, "suite": "sweep", "wording": "w1"}) == []  # written before the fields existed
    assert run_sweep.config_changes(old, {**old, "suite": "replication", "wording": "w1"}) == ["suite"]
    assert run_sweep.config_changes({**old, "wording": "w2"}, {**old, "wording": "w3"}) == ["wording"]


def test_a_replication_never_writes_into_the_sweep_and_the_sweep_never_runs_under_another_wording(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "--model", "x/y", "--suite", "replication", "--out", "sweep"])
    with pytest.raises(SystemExit, match="own directory"):
        run_sweep.main()
    monkeypatch.setattr(run_sweep, "WORDING", "w2")
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "--model", "x/y"])
    with pytest.raises(SystemExit, match="legacy wording"):
        run_sweep.main()
