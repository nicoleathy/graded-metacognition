import numpy as np
import pytest

from probe_validity.loaders import make_manifest, save_manifest, load_manifest, select, stratified_order
from probe_validity.prompts import Item


def synthetic_items(sizes: dict[str, int]) -> list[Item]:
    items = []
    for subject, m in sizes.items():
        for j in range(m):
            items.append(Item(qid=f"{subject}_{j}", stem="q", options=("a", "b", "c", "d"), gold="A", group=subject))
    return items


SIZES = {f"s{k:02d}": 100 + 37 * k for k in range(57)}  # 57 subjects of unequal size, as in MMLU


def test_every_prefix_is_proportionally_stratified():
    items = synthetic_items(SIZES)
    groups = np.array([it.group for it in items])
    total = len(items)
    order = stratified_order(groups, seed=0)
    assert sorted(order.tolist()) == list(range(total))
    for n in (200, 500, 2000, 5000):
        counts = {g: int((groups[order[:n]] == g).sum()) for g in SIZES}
        worst = max(abs(counts[g] - n * SIZES[g] / total) for g in SIZES)
        assert worst < 1.6, (n, worst)
    assert all(c >= 1 for c in {g: int((groups[order[:1000]] == g).sum()) for g in SIZES}.values())


def test_the_first_500_are_a_subset_of_the_first_2000():
    items = synthetic_items(SIZES)
    m500, m2000 = (make_manifest(items, n, 0, f"m{n}") for n in (500, 2000))
    assert set(m500["qids"]) <= set(m2000["qids"]) and m2000["qids"][:500] == m500["qids"]


def test_seeded_and_different_across_seeds():
    items = synthetic_items(SIZES)
    a, b, c = (make_manifest(items, 500, s, "m") for s in (0, 0, 1))
    assert a == b and a["qids"] != c["qids"] and a["sha256"] != c["sha256"]


def test_select_detects_edits_and_missing_items(tmp_path):
    items = synthetic_items({"x": 30, "y": 30})
    manifest = make_manifest(items, 10, 0, "m")
    path = tmp_path / "m.json"
    save_manifest(manifest, path)
    chosen = select(items, load_manifest(path))
    assert [it.qid for it in chosen] == manifest["qids"]
    tampered = dict(manifest, qids=manifest["qids"][::-1])
    with pytest.raises(ValueError):
        select(items, tampered)
    with pytest.raises(KeyError):
        select(items[:5], manifest)


def test_mmlu_pro_and_supergpqa_converters_check_the_answer_key():
    from probe_validity.loaders import mmlu_pro_items, supergpqa_items

    pro = [{"question_id": 1, "question": "q", "options": ["a", "b", "c"], "answer": "B", "answer_index": 1, "category": "engineering", "src": "s"},
           {"question_id": 2, "question": "q", "options": ["a", "b"], "answer": "A", "answer_index": 0, "category": "law", "src": "s"}]
    assert [it.qid for it in mmlu_pro_items(pro)] == ["mmlupro_1"] and mmlu_pro_items(pro)[0].gold == "B"
    pro[0]["answer"] = "C"
    with pytest.raises(AssertionError):
        mmlu_pro_items(pro)

    def row(uuid, field, subfield, letter="B"):
        return {"uuid": uuid, "question": "q", "options": ["x", "y", "z"], "answer": "y", "answer_letter": letter,
                "field": field, "subfield": subfield}

    rows = [row("1", "Mechanical Engineering", "Manufacturing Automation"), row("2", "Materials Science and Engineering", "Materials Processing Engineering"),
            row("3", "Management Science and Engineering", "Management Science and Engineering"),
            row("4", "Mechanics", "Rigid Body Mechanics"), row("5", "Materials Science and Engineering", "Materials Physics and Chemistry")]
    assert [it.qid for it in supergpqa_items(rows)] == ["supergpqa_1", "supergpqa_2", "supergpqa_3"]
    with pytest.raises(AssertionError):
        supergpqa_items([row("9", "Mechanical Engineering", "Mechatronic Engineering", letter="C")])


def test_open_qa_converters():
    from probe_validity.loaders import triviaqa_items, webquestions_items

    row = {"question_id": "tc_1", "question": "Who?", "answer": {"aliases": ["A", "B"]}}
    tq = triviaqa_items([row, dict(row), {"question_id": "tc_2", "question": "Where?", "answer": {"aliases": ["C"]}}])
    assert [it.qid for it in tq] == ["triviaqa_validation_tc_1", "triviaqa_validation_tc_2"]  # the repeated row is dropped
    assert tq[0].answers == ("A", "B") and tq[0].options is None
    assert triviaqa_items([row, dict(row, answer={"aliases": ["Z"]})]) == []  # rows that disagree: the id is dropped
    wq = webquestions_items([{"question": "what?", "answers": ["x"]}])
    assert wq[0].answers == ("x",) and wq[0].qid == "webq_test_0"


def test_code_digest_changes_with_the_source(tmp_path):
    from probe_validity.provenance import code_digest

    (tmp_path / "probe_validity").mkdir()
    (tmp_path / "experiments/paper1").mkdir(parents=True)
    (tmp_path / "probe_validity/a.py").write_text("x = 1\n")
    first = code_digest(tmp_path)
    assert first == code_digest(tmp_path)
    (tmp_path / "probe_validity/a.py").write_text("x = 2\n")
    assert code_digest(tmp_path) != first
