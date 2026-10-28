import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/paper1"))

from probe_validity.provenance import sha256_of  # noqa: E402
from probe_validity.scales import label_overlap  # noqa: E402

protocol = pytest.importorskip("protocol")  # experiments/paper1/protocol.py
FROZEN = ROOT / "experiments/paper1/frozen/protocol.json"


def test_protocol_matches_the_frozen_record():
    frozen = json.loads(FROZEN.read_text())
    current = json.loads(json.dumps(protocol.protocol_record(), default=list))
    assert current["sha256"] == frozen["sha256"], "protocol.py changed after it was frozen: re-freeze only before a sweep starts"
    assert current == frozen


def test_conditions_per_dataset():
    counts = {d: len(protocol.conditions_for(d)) for d in protocol.DATASET_TABLE}
    assert counts == {"mmlu": 25, "mmlu_pro_eng": 14, "supergpqa_mfg": 14, "triviaqa": 14, "webquestions": 14}
    mmlu = [c.name for c in protocol.conditions_for("mmlu")]
    assert {"direct_legacy", "yesno_embedded", "letters_embedded", "digits_embedded", "words_embedded",
            "yesno_stem", "letters_stem", "digits_stem", "words_stem",
            "letters_flip_order_raw", "letters_flip_labels_raw", "letters_optdigits", "digits_optdigits"} <= set(mmlu)
    for d in ("mmlu_pro_eng", "supergpqa_mfg", "triviaqa", "webquestions"):
        names = [c.name for c in protocol.conditions_for(d)]
        assert not any("embedded" in n or "stem" in n or "optdigits" in n or n == "direct_legacy" for n in names)  # MMLU-only variants
        assert {"letters_flip_order_raw", "letters_flip_labels_raw"} <= set(names)  # the order controls run everywhere


def test_every_raw_meta_prompt_is_free_of_format_instructions():
    from probe_validity.prompts import assert_no_format_instruction

    for dataset in protocol.DATASET_TABLE:
        item = protocol.REFERENCE_ITEMS[bool(protocol.DATASET_TABLE[dataset][1])]
        for cond in protocol.conditions_for(dataset):
            if cond.name.endswith(("_raw", "_stem", "optdigits")) or "sampled" in cond.name:
                text = cond.build(item)
                body = text.split("\n\nHow confident")[0] if "How confident" in text else text.split("Question:", 1)[1]
                assert_no_format_instruction(body.replace("Question:", ""))


def test_sampled_conditions_use_distinct_seeds_and_the_same_prompt():
    sampled = [c for c in protocol.conditions_for("mmlu") if "sampled" in c.name]
    item = protocol.REFERENCE_ITEMS[True]
    assert len(sampled) == protocol.DRAWS and len({c.seed for c in sampled}) == protocol.DRAWS
    assert len({c.build(item) for c in sampled}) == 1 and all(c.sampling == protocol.SAMPLING for c in sampled)


def test_only_the_letter_scales_collide_with_the_answer_labels():
    from probe_validity.scales import SCALES, YESNO

    answers = ("A", "B", "C", "D")
    colliding = {name for name, s in {**SCALES, "yesno": YESNO}.items() if label_overlap(s, answers)}
    assert colliding == {"letters", "letters_reversed", "letters_perm"}
