"""Write the seeded manifests of every Paper 1 dataset and print an audit of the source data. Run from the repository root:
python experiments/paper1/make_manifests.py"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probe_validity.loaders import load_items, make_manifest, save_manifest  # noqa: E402

OUT = Path(__file__).parent / "manifests"
SEED = 0
# dataset -> manifests to write: (name, n); n=None takes every item
PLAN = {
    "mmlu": [("mmlu_test_s0_n500", 500), ("mmlu_test_s0_n2000", 2000), ("mmlu_test_s0_all", None)],
    "mmlu_pro_eng": [("mmlu_pro_eng_all", None)],
    "supergpqa_mfg": [("supergpqa_mfg_all", None)],
    "triviaqa": [("triviaqa_val_s0_n2000", 2000)],
    "webquestions": [("webquestions_test_s0_n1000", 1000)],
}


def audit(name, items):
    groups = collections.Counter(it.group for it in items)
    opts = collections.Counter(len(it.options) if it.options else 0 for it in items)
    print(f"[{name}] {len(items)} items, {len(groups)} groups, options per item {dict(opts.most_common(5))}, "
          f"duplicate ids {len(items) - len({it.qid for it in items})}, duplicate stems {len(items) - len({it.stem for it in items})}")
    if items[0].options:
        gold = collections.Counter(it.gold for it in items)
        print("   gold letters:", {k: round(v / len(items), 3) for k, v in sorted(gold.items())})
        assert all(it.gold in it.option_labels for it in items)
    else:
        assert all(it.answers for it in items), "an item without an answer alias"
        print("   aliases per item: median", sorted(len(it.answers) for it in items)[len(items) // 2])


def main() -> None:
    for dataset, manifests in PLAN.items():
        items, source = load_items(dataset)
        audit(dataset, items)
        if dataset == "mmlu":
            assert len(items) == 14042 and len({it.group for it in items}) == 57, "unexpected MMLU test split"
        for name, n in manifests:
            manifest = make_manifest(items, n, SEED, name, dataset, source)
            save_manifest(manifest, OUT / f"{name}.json")
            print(f"   wrote {name}: n={manifest['n']} sha256={manifest['sha256'][:16]} revision={source['revision'][:10]}")


if __name__ == "__main__":
    main()
