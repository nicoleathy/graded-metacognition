"""Item loaders and seeded manifests. A manifest is the ordered list of item ids a result was computed on."""

import json
from pathlib import Path

import numpy as np

from .prompts import Item
from .provenance import sha256_of

LETTERS_10 = "ABCDEFGHIJ"


def stratified_order(groups, seed: int) -> np.ndarray:
    """A permutation of range(len(groups)) in which every prefix is close to proportionally stratified by group.

    Inside each group the items are shuffled; the j-th of m items gets the key (j + u) / m with one random offset u per
    group, and all items are sorted by key. A prefix of length n then holds n * m / N items of a group of size m,
    give or take one, so the first 500 and the first 2000 items are both stratified samples and the first 500 is a
    subset of the first 2000.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    keys = np.empty(len(groups))
    for g in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == g)
        keys[rng.permutation(idx)] = (np.arange(len(idx)) + rng.random()) / len(idx)
    return np.argsort(keys, kind="stable")


def make_manifest(items: list[Item], n: int | None, seed: int, name: str, dataset: str = "", source: dict | None = None) -> dict:
    order = stratified_order([it.group for it in items], seed)
    chosen = order if n is None else order[:n]
    qids = [items[i].qid for i in chosen]
    return {"name": name, "dataset": dataset, "source": source or {}, "seed": seed, "n": len(qids), "qids": qids,
            "sha256": sha256_of(qids)}


def select(items: list[Item], manifest: dict) -> list[Item]:
    if sha256_of(manifest["qids"]) != manifest["sha256"]:
        raise ValueError(f"manifest {manifest['name']} was edited: its hash does not match")
    by_id = {it.qid: it for it in items}
    missing = [q for q in manifest["qids"] if q not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} manifest items are not in the dataset, e.g. {missing[:3]}")
    return [by_id[q] for q in manifest["qids"]]


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=0) + "\n")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def hub_source(repo_id: str, split: str, config: str | None = None) -> dict:
    """The exact dataset revision a run uses: the Hub commit hash is recorded in every manifest."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    return {"repo": repo_id, "config": config, "split": split, "revision": info.sha,
            "license": getattr(info.card_data, "license", None)}


def _load(source: dict):
    from datasets import load_dataset

    return load_dataset(source["repo"], source["config"], split=source["split"], revision=source["revision"])


def mmlu_items(rows, split: str = "test") -> list[Item]:
    return [Item(qid=f"mmlu_{split}_{i}", stem=r["question"], options=tuple(r["choices"]), gold=LETTERS_10[r["answer"]],
                 group=r["subject"]) for i, r in enumerate(rows)]


def mmlu_pro_items(rows, category: str = "engineering") -> list[Item]:
    items = []
    for r in rows:
        if r["category"] == category:
            assert LETTERS_10[r["answer_index"]] == r["answer"], r["question_id"]
            items.append(Item(qid=f"mmlupro_{r['question_id']}", stem=r["question"], options=tuple(r["options"]),
                              gold=r["answer"], group=r["src"]))
    return items


# SuperGPQA manufacturing slice, fixed from metadata alone before any model was run: mechanical engineering
# (manufacturing automation, mechatronics), materials processing engineering, and management science and engineering.
SUPERGPQA_MFG_FIELDS = ("Mechanical Engineering", "Management Science and Engineering")
SUPERGPQA_MFG_SUBFIELDS = ("Materials Processing Engineering",)


def supergpqa_items(rows) -> list[Item]:
    items = []
    for r in rows:
        if r["field"] in SUPERGPQA_MFG_FIELDS or r["subfield"] in SUPERGPQA_MFG_SUBFIELDS:
            assert r["options"][LETTERS_10.index(r["answer_letter"])] == r["answer"], r["uuid"]
            items.append(Item(qid=f"supergpqa_{r['uuid']}", stem=r["question"], options=tuple(r["options"]),
                              gold=r["answer_letter"], group=r["subfield"]))
    return items


def triviaqa_items(rows, split: str = "validation") -> list[Item]:
    """One item per question. The validation split has one row per (question, evidence document): 17,944 rows hold
    9,960 question ids, each twice. Repeated rows must agree on the question and the aliases; an id whose rows
    disagree (one id in the current revision, qz_4576) is dropped."""
    first: dict[str, tuple[str, tuple[str, ...]]] = {}
    conflicts = set()
    for r in rows:
        record = (r["question"], tuple(r["answer"]["aliases"]))
        if r["question_id"] not in first:
            first[r["question_id"]] = record
        elif first[r["question_id"]] != record:
            conflicts.add(r["question_id"])
    return [Item(qid=f"triviaqa_{split}_{qid}", stem=q, answers=aliases)
            for qid, (q, aliases) in first.items() if qid not in conflicts]


def webquestions_items(rows, split: str = "test") -> list[Item]:
    return [Item(qid=f"webq_{split}_{i}", stem=r["question"], answers=tuple(r["answers"])) for i, r in enumerate(rows)]


# name -> (hub repo, config, split, rows -> items). The name is what manifests and run directories use.
DATASETS = {
    "mmlu": ("cais/mmlu", "all", "test", mmlu_items),
    "mmlu_pro_eng": ("TIGER-Lab/MMLU-Pro", None, "test", mmlu_pro_items),
    "supergpqa_mfg": ("m-a-p/SuperGPQA", None, "train", supergpqa_items),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation", triviaqa_items),
    "webquestions": ("stanfordnlp/web_questions", None, "test", webquestions_items),
}


def load_items(name: str, revision: str | None = None) -> tuple[list[Item], dict]:
    """(items, source record) of a registered dataset. With `revision` (the Hub commit a manifest recorded) the data come from that
    commit, so a later upstream change cannot alter or block a rerun; without it, from the current commit."""
    repo, config, split, convert = DATASETS[name]
    source = hub_source(repo, split, config) if revision is None else {"repo": repo, "config": config, "split": split, "revision": revision, "license": None}
    return convert(_load(source)), source


def load_mmlu(split: str = "test") -> list[Item]:
    """All MMLU items of a split (cais/mmlu, config "all"), in dataset order, with the subject as the group."""
    from datasets import load_dataset

    return mmlu_items(load_dataset("cais/mmlu", "all", split=split), split)
