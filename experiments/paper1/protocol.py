"""The Paper 1 protocol: which conditions are generated for each dataset. One table drives the sweep, the analysis and the
frozen protocol record, so a condition cannot exist in one and be missing from another.

Open questions (TriviaQA, WebQuestions) have no embedded variant: their question strings carry no answer-format instruction.
The embedded variants and the legacy direct prompt reproduce the legacy MMLU loader and exist for MMLU only.
"""

from dataclasses import dataclass
from typing import Callable

from common import PROBES, ROOT, Item  # noqa: F401  (ROOT re-exported)

from probe_validity.loaders import LETTERS_10
from probe_validity.prompts import DIRECT_OPEN, direct_prompt, meta_prompt
from probe_validity.provenance import sha256_of
from probe_validity.scales import LETTERS, LETTERS_FLIP_LABELS, LETTERS_FLIP_ORDER, LETTERS_PERM, LETTERS_REVERSED, PROBE_SCALES, SCALES

DRAWS = 5  # sampled decodes per item for the decoding-invariance check
SAMPLING = {"temperature": 0.7, "top_p": 0.8}  # the Hub defaults of Qwen2.5-Instruct, applied to every model
SUBSET = 500  # items used for the sampled decodes (a prefix of the stratified order)

# dataset -> (manifest name, option labels whose first-token log-probabilities are stored, has embedded variants)
DATASET_TABLE = {
    "mmlu": ("mmlu_test_s0_n2000", tuple(LETTERS_10[:4]), True),
    "mmlu_pro_eng": ("mmlu_pro_eng_all", tuple(LETTERS_10), False),
    "supergpqa_mfg": ("supergpqa_mfg_all", tuple(LETTERS_10), False),
    "triviaqa": ("triviaqa_val_s0_n2000", (), False),
    "webquestions": ("webquestions_test_s0_n1000", (), False),
}


@dataclass(frozen=True)
class Condition:
    name: str
    build: Callable[[Item], str]
    labels: tuple[str, ...]  # labels whose first-token log-probabilities are stored
    max_new_tokens: int = 32
    sampling: dict | None = None
    seed: int = 0
    subset: int | None = None  # use only the first `subset` items of the manifest


def _meta(scale_name: str, kind: str, variant: str):
    scale = PROBE_SCALES[scale_name]
    return lambda it: meta_prompt(it, kind, variant, scale)


SUITES = ("sweep", "replication")
# The replication (frozen/replication_plan.json) repeats the study's interventions where they were not run: the embedded instruction, the stem-only
# prompt, the answer-label control and the label x order design on all three multiple-choice sets (the sweep has them on MMLU only), under three
# wordings of the probe (PROBE_WORDING). It has no direct_legacy prompt, no permuted scale and no sampled decodes (those serve the screen only).
REPLICATION_DATASETS = ("mmlu", "mmlu_pro_eng", "supergpqa_mfg")


def conditions_for(dataset: str, suite: str = "sweep") -> list[Condition]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}")
    manifest, option_labels, embedded = DATASET_TABLE[dataset]
    replication = suite == "replication"
    if replication:
        if dataset not in REPLICATION_DATASETS:
            raise ValueError(f"the replication covers {REPLICATION_DATASETS}, not {dataset!r}")
        embedded = True
    conds = [Condition("direct_clean", direct_prompt, option_labels)]
    if embedded and not replication:
        conds.append(Condition("direct_legacy", lambda it: DIRECT_OPEN.format(question=it.legacy_question()), option_labels))
    variants = ("raw", "embedded", "stem") if embedded else ("raw",)  # stem: the question alone, no options shown
    for probe, (scale, kind) in PROBES.items():
        for variant in variants:
            conds.append(Condition(f"{probe}_{variant}", _meta(probe, kind, variant), scale.labels))
    for scale in (LETTERS_REVERSED, LETTERS_PERM, LETTERS_FLIP_ORDER, LETTERS_FLIP_LABELS):
        if replication and scale is LETTERS_PERM:
            continue
        conds.append(Condition(f"{scale.name}_raw", _meta(scale.name, "graded", "raw"), scale.labels))
    if embedded:  # the answer-label control, a 2 x 2 with letters_raw and digits_raw (options labelled by digits)
        conds.append(Condition("letters_optdigits", _meta("letters", "graded", "optdigits"), LETTERS.labels))
        conds.append(Condition("digits_optdigits", _meta("digits", "graded", "optdigits"), PROBE_SCALES["digits"].labels))
    if not replication:
        for draw in range(DRAWS):
            conds.append(Condition(f"letters_sampled{draw}", _meta("letters", "graded", "raw"), LETTERS.labels, 16,
                                   SAMPLING, seed=1000 * (draw + 1), subset=SUBSET))
    return conds


REFERENCE_ITEMS = {  # fixed items whose rendered prompts are hashed, so a template edit changes the protocol hash
    True: Item(qid="ref_mcq", stem="Which gas do plants absorb?", options=("Oxygen", "Carbon dioxide", "Helium", "Neon"), gold="B"),
    False: Item(qid="ref_open", stem="Who wrote Hamlet?", answers=("William Shakespeare",)),
}


def prompt_digest(dataset: str, suite: str = "sweep") -> dict:
    item = REFERENCE_ITEMS[bool(DATASET_TABLE[dataset][1])]
    return {c.name: sha256_of(c.build(item)) for c in conditions_for(dataset, suite)}


VERSION = "2.1"
SUPERSEDES = {"2.0": "fbb472678d98f555961b8035e0cef7a30d375813f5e822edca13901b21a2e22b"}  # 2.1 only adds conditions (added 2026-10-27
# after review: separate label from presentation order, and collision from scale format); no 2.0 condition changed


def protocol_record() -> dict:
    """Everything that defines the sweep, hashed: conditions per dataset, decoding, subsets, label scales."""
    record = {"version": VERSION, "supersedes": SUPERSEDES, "draws": DRAWS, "sampling": SAMPLING, "subset": SUBSET, "datasets": {}, "scales": {
        name: {"labels": s.labels, "descriptors": s.descriptors, "lines": s.lines()} for name, s in SCALES.items()}}
    for dataset, (manifest, labels, embedded) in DATASET_TABLE.items():
        record["datasets"][dataset] = {
            "manifest": manifest, "option_labels": labels,
            "conditions": [{"name": c.name, "max_new_tokens": c.max_new_tokens, "sampling": c.sampling, "seed": c.seed,
                            "subset": c.subset, "labels": c.labels} for c in conditions_for(dataset)],
            "prompt_sha256_on_reference_item": prompt_digest(dataset),
        }
    record["sha256"] = sha256_of(record)
    return record


REPLICATION_VERSION = "R1.0"


def replication_record() -> dict:
    """Everything that defines one wording of the replication, hashed: the wording itself, the conditions per dataset and the prompts on a reference item.
    The sweep's record (protocol_record) is not touched: the frozen protocol hash stays valid."""
    from probe_validity.scales import WORDING

    record = {"version": REPLICATION_VERSION, "suite": "replication", "wording": WORDING, "scales": {
        name: {"labels": s.labels, "descriptors": s.descriptors, "lines": s.lines()} for name, s in SCALES.items()}, "datasets": {}}
    for dataset in REPLICATION_DATASETS:
        manifest, labels, _ = DATASET_TABLE[dataset]
        record["datasets"][dataset] = {
            "manifest": manifest, "option_labels": labels,
            "conditions": [{"name": c.name, "max_new_tokens": c.max_new_tokens, "labels": c.labels} for c in conditions_for(dataset, "replication")],
            "prompt_sha256_on_reference_item": prompt_digest(dataset, "replication"),
        }
    record["sha256"] = sha256_of(record)
    return record
