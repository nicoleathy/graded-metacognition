"""From stored generation rows to the arrays the metrics and the screen need. Everything is aligned to the manifest order:
position i of every array is item i."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generate import read_jsonl
from .parse import parse_mcq, parse_strict
from .prompts import Item
from .scales import PROBE_SCALES, Scale
from .scoring import score_mcq, score_open, score_open_lenient
from .screen import Cell

MAPPING_SCALES = ("letters", "letters_reversed", "letters_perm", "digits", "words")  # the four-level scales: one meaning each


def open_scorer():
    """The scorer of open answers: the frozen one, or the lenient sensitivity scorer when OPEN_SCORER=lenient is set (read at call time)."""
    name = os.environ.get("OPEN_SCORER", "strict")
    if name not in ("strict", "lenient"):
        raise ValueError(f"OPEN_SCORER must be strict or lenient, not {name!r}")
    return score_open if name == "strict" else score_open_lenient


def read_condition(directory: Path, name: str, n: int) -> list[dict | None]:
    """The rows of one condition by item position; None where an item has no row (a subset condition)."""
    rows: list[dict | None] = [None] * n
    for r in read_jsonl(directory / f"{name}.jsonl"):
        if r.get("condition") == name and r["i"] < n:
            rows[r["i"]] = r
    return rows


@dataclass
class Direct:
    correct: np.ndarray  # (n,) 1 if the committed answer is right; invalid answers are incorrect
    valid: np.ndarray  # (n,) the reply committed to an answer (MCQ: a label was found; open: a non-empty first span)
    label: list  # MCQ: the parsed label or None; open: the first span
    reference: np.ndarray  # (n,) the model's own log-likelihood of its answer (S7); NaN if unavailable
    reference_mean: np.ndarray  # (n,) the same, averaged over the answer's tokens


def direct_outcomes(rows: list[dict | None], items: list[Item]) -> Direct:
    n = len(items)
    correct, valid = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    ref, ref_mean, labels = np.full(n, np.nan), np.full(n, np.nan), [None] * n
    for i, (row, item) in enumerate(zip(rows, items)):
        if row is None:
            continue
        if item.options is not None:
            p = parse_mcq(row["text"], item.option_labels, item.options)
            labels[i], valid[i] = p.label, p.valid
            correct[i] = score_mcq(row["text"], item.gold, item.option_labels, options=item.options).correct
            lp = row.get("label_lp") or {}
            own = [lab for lab in item.option_labels if lab in lp]
            if p.valid and p.tier in ("strict", "echo") and p.label in lp and len(own) > 1:  # elsewhere the first token is not the label
                z = np.logaddexp.reduce([lp[lab] for lab in own])
                ref[i] = ref_mean[i] = lp[p.label] - z  # log of the probability of the chosen label among the item's labels
        else:
            sc = open_scorer()(row["text"], item.answers)
            labels[i], valid[i], correct[i] = sc.span, bool(sc.span), sc.correct
            if "span_lp" in row and row.get("span_n"):
                ref[i], ref_mean[i] = row["span_lp"], row["span_lp"] / row["span_n"]
    return Direct(correct, valid, labels, ref, ref_mean)


def grades(rows: list[dict | None], scale: Scale) -> np.ndarray:
    """Semantic grade (higher = more confident) of each reply under the strict parser; NaN if invalid or missing."""
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            p = parse_strict(row["text"], scale)
            if p.valid:
                out[i] = scale.grade_of(p.label)
    return out


def first_token_score(rows: list[dict | None], scale: Scale) -> tuple[np.ndarray, np.ndarray]:
    """Expected grade under the first-token label probabilities, renormalised over the scale's labels, and the
    probability mass those labels hold. NaN where a row is missing or has no label log-probabilities."""
    score, mass = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    weights = np.array([scale.grade_of(lab) for lab in scale.labels], dtype=float)
    for i, row in enumerate(rows):
        if row is not None and all(lab in row.get("label_lp", {}) for lab in scale.labels):
            p = np.exp([row["label_lp"][lab] for lab in scale.labels])
            mass[i] = p.sum()
            if p.sum() > 0:
                score[i] = float(p @ weights / p.sum())
    return score, mass


def ambiguous_labels(rows: list[dict | None], scale: Scale) -> list[tuple[str, str]]:
    """Pairs of labels whose first-token log-probabilities are identical in every row: they share a first token, so the
    first-token baseline cannot tell them apart and must not be reported for this scale."""
    have = [r for r in rows if r is not None and all(lab in r.get("label_lp", {}) for lab in scale.labels)]
    if not have:
        return []
    pairs = []
    for i, a in enumerate(scale.labels):
        for b in scale.labels[i + 1:]:
            if all(r["label_lp"][a] == r["label_lp"][b] for r in have):
                pairs.append((a, b))
    return pairs


def build_cell(directory: Path, items: list[Item], probe: str, answer_labels: tuple[str, ...], direct: Direct,
               draws: int = 5, variant: str = "raw", with_mappings: bool = True) -> Cell:
    """The screen cell of one probe: its own grades, the relabelled grades of the same model and items, the sampled
    decodes, the answer likelihood, and the item strata."""
    n = len(items)
    scale = PROBE_SCALES[probe]
    cell = Cell(scale, answer_labels, direct.correct, grades(read_condition(directory, f"{probe}_{variant}", n), scale),
                reference=direct.reference, reference_mean=direct.reference_mean, strata=np.array([it.group for it in items]))
    if with_mappings and probe in MAPPING_SCALES:
        # Relabelled grades come from the same prompt variant: raw has five scales (two extra letter relabellings), stem three.
        suffix = "stem" if variant == "stem" else "raw"
        for other in MAPPING_SCALES:
            if other != probe and (directory / f"{other}_{suffix}.jsonl").exists():
                cell.mappings[other] = grades(read_condition(directory, f"{other}_{suffix}", n), PROBE_SCALES[other])
        if probe == "letters" and variant == "raw" and (directory / "letters_sampled0.jsonl").exists():
            draws_rows = [grades(read_condition(directory, f"letters_sampled{d}", n), scale) for d in range(draws)
                          if (directory / f"letters_sampled{d}.jsonl").exists()]
            cell.sampled = np.stack(draws_rows, axis=1)
    return cell
