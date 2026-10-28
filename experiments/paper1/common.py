"""Shared paths and the condition table of the Paper 1 experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_validity.prompts import DIRECT_OPEN, Item, direct_prompt, meta_prompt  # noqa: E402
from probe_validity.scales import DIGITS, LETTERS, WORDS, YESNO  # noqa: E402

MANIFESTS = ROOT / "experiments/paper1/manifests"
RUNS = ROOT / "experiments/paper1/runs"
OPTION_LABELS = ("A", "B", "C", "D")

# Probe name -> (scale used to parse its replies, prompt kind)
PROBES = {"yesno": (YESNO, "yesno"), "letters": (LETTERS, "graded"), "digits": (DIGITS, "graded"), "words": (WORDS, "graded")}


def model_slug(model_id: str) -> str:
    return model_id.strip("/").replace("/", "__")


def e10_conditions() -> dict:
    """The E10 conditions: two direct prompts and the 2 x 2 (embedded instruction vs content only) x four probe formats.
    name -> (prompt builder for one Item, labels whose first-token log-probabilities are stored)."""
    table = {
        "direct_clean": (direct_prompt, OPTION_LABELS),
        "direct_legacy": (lambda it: DIRECT_OPEN.format(question=it.legacy_question()), OPTION_LABELS),
    }
    for probe, (scale, kind) in PROBES.items():
        for variant in ("raw", "embedded", "stem"):
            table[f"{probe}_{variant}"] = (
                lambda it, kind=kind, variant=variant, scale=scale: meta_prompt(it, kind, variant, scale),
                scale.labels,
            )
    return table


def split_condition(name: str) -> tuple[str, str]:
    probe, variant = name.rsplit("_", 1)
    return probe, variant


__all__ = ["ROOT", "MANIFESTS", "RUNS", "OPTION_LABELS", "PROBES", "Item", "model_slug", "e10_conditions", "split_condition"]
