"""Self-consistency check of the open-answer scorer on the real manifests: every item's first accepted answer, written four plain ways, must
score correct. A failure means a scoring rule fires on the words of an accepted answer itself (a hedge word, a comma, a negator inside a
title), which would put noise into every open-question AUROC. Run it before interpreting open-question results.

    python experiments/paper1/check_open_scorer.py
"""

import sys

from common import MANIFESTS
from protocol import DATASET_TABLE

from probe_validity.loaders import load_items, load_manifest, select
from probe_validity.scoring import score_open

FORMS = ("The answer is {a}.", "{a}", "It's {a}", "{a} is the answer.")


def main() -> int:
    failures = 0
    for dataset in ("triviaqa", "webquestions"):
        manifest = load_manifest(MANIFESTS / f"{DATASET_TABLE[dataset][0]}.json")
        items, _ = load_items(dataset, manifest["source"]["revision"])
        items = select(items, manifest)
        bad = [(form, it.answers[0]) for it in items for form in FORMS if score_open(form.format(a=it.answers[0]), list(it.answers)).correct != 1]
        failures += len(bad)
        print(f"{dataset}: {len(items)} items, {len(bad)} rejected plain restatements of a gold answer" + (f", e.g. {bad[:3]}" if bad else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
