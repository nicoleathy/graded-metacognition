import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

audit_sheets = pytest.importorskip("audit_sheets")


def test_cohen_kappa_textbook_values():
    assert audit_sheets.cohen_kappa(list("ABABAB"), list("ABABAB")) == pytest.approx(1.0)
    assert audit_sheets.cohen_kappa(list("AABB"), list("ABAB")) == pytest.approx(0.0)  # agreement at chance level
    assert audit_sheets.cohen_kappa(list("AAAB"), list("AAAA")) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        audit_sheets.cohen_kappa(["a"], ["a", "b"])


def test_the_draw_takes_half_from_risky_cases_and_is_deterministic():
    cands = [{"risky": i % 10 == 0, "id": i} for i in range(1000)]
    a, b = audit_sheets.draw(cands, 100, seed=3), audit_sheets.draw(cands, 100, seed=3)
    assert [c["id"] for c in a] == [c["id"] for c in b] and len(a) == 100
    assert sum(c["risky"] for c in a) == 50  # there are 100 risky cases, so half the sample can come from them
    assert len({c["id"] for c in a}) == 100
    few = [{"risky": i < 3, "id": i} for i in range(500)]
    assert sum(c["risky"] for c in audit_sheets.draw(few, 100, seed=0)) == 3  # never more risky cases than exist


def test_the_draw_balances_answer_decisions_and_probe_replies():
    cands = ([{"task": "answer_label", "risky": i % 5 == 0, "id": f"a{i}"} for i in range(300)]
             + [{"task": "probe_label_yesno", "risky": i % 2 == 0, "id": f"p{i}"} for i in range(900)])
    sample = audit_sheets.draw(cands, 100, seed=1)
    assert len(sample) == 100 and sum(c["task"] == "answer_label" for c in sample) == 50
    assert sum(c["risky"] for c in sample if c["task"] == "answer_label") == 25


def test_the_annotator_sheet_never_shows_the_programs_decision(tmp_path, monkeypatch):
    from probe_validity.prompts import Item
    from probe_validity import records

    monkeypatch.setattr(audit_sheets, "OUT", tmp_path)
    item = Item(qid="q0", stem="Which gas do plants absorb?", options=("Oxygen", "Carbon dioxide", "Helium", "Neon"), gold="B")
    cands = [{"task": "answer_label", "i": 0, "text": "B) Carbon dioxide", "decision": "B", "risky": False, "item": item, "model": "m"}]
    monkeypatch.setattr(audit_sheets, "candidates", lambda d, items, ds: cands)
    monkeypatch.setattr(audit_sheets, "items_for", lambda manifest: [item])
    monkeypatch.setattr(audit_sheets, "DATASET_TABLE", {"mmlu": ("mmlu_test_s0_n2000", (), True)})
    (tmp_path / "sweep").mkdir()
    monkeypatch.setattr(audit_sheets, "RUNS", tmp_path)
    run = tmp_path / "sweep/m/mmlu"
    run.mkdir(parents=True)
    (run / "DONE").write_text("x")
    audit_sheets.make(1, 0)
    blank = list(csv.reader((tmp_path / "mmlu_annotator_blank.csv").open()))
    key = list(csv.reader((tmp_path / "mmlu_key.csv").open()))
    assert "program_decision" in key[0] and "program_decision" not in blank[0]
    assert blank[1][5] == "" and "B" not in [blank[1][5]]  # the decision column is empty
    assert key[1][5] == "B"
