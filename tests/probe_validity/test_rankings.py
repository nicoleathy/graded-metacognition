import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import rankings  # noqa: E402


def world(pairs):
    """pairs: model -> (legacy gamma*, strict AUROC)."""
    return {"models": {m: {"mmlu": {"legacy_pipeline": {"yesno": {"legacy": {"gamma_star_imputed": g}, "strict": {"auroc": {"auroc": a}}}}}} for m, (g, a) in pairs.items()}}


def test_identical_orderings_have_tau_one_and_nobody_moves():
    s = world({"a": (0.9, 0.7), "b": (0.8, 0.6), "c": (0.7, 0.55), "d": (0.6, 0.5), "e": (0.5, 0.45)})
    r = rankings.compare(s, "mmlu", "yesno")
    assert r["kendall_tau"] == 1.0 and r["n_models_that_move"] == 0 and r["top_model_conventional"] == r["top_model_strict"] == "a"


def test_a_reversed_ordering_has_tau_minus_one_and_everybody_moves_but_the_middle():
    s = world({"a": (0.9, 0.45), "b": (0.8, 0.5), "c": (0.7, 0.55), "d": (0.6, 0.6), "e": (0.5, 0.7)})
    r = rankings.compare(s, "mmlu", "yesno")
    assert r["kendall_tau"] == -1.0 and r["n_models_that_move"] == 4 and r["top_model_strict"] == "e" and r["rank_change"]["c"] == 0


def test_too_few_models_give_no_comparison_and_ties_share_a_rank():
    assert rankings.compare(world({"a": (0.9, 0.7), "b": (0.8, 0.6)}), "mmlu", "yesno") is None
    assert rankings.ranks({"a": 0.5, "b": 0.5, "c": 0.1}) == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_the_metric_only_contrast_uses_the_strict_replies_and_not_the_legacy_pipelines_number():
    """Same AUROC for every model, so the metric-only ranking (gamma* on the strict labels) can differ from the pipeline ranking (the legacy pipeline's own gamma*)."""
    models = {}
    for name, (pipeline, metric, auroc) in {"a": (0.9, 0.5, 0.7), "b": (0.8, 0.6, 0.6), "c": (0.7, 0.7, 0.55), "d": (0.6, 0.8, 0.5), "e": (0.5, 0.9, 0.45)}.items():
        models[name] = {"mmlu": {"legacy_pipeline": {"yesno": {"legacy": {"gamma_star_imputed": pipeline}, "strict": {"gamma_star": metric, "auroc": {"auroc": auroc}}}}}}
    pipeline_contrast = rankings.compare({"models": models}, "mmlu", "yesno")
    metric_contrast = rankings.compare({"models": models}, "mmlu", "yesno", metric_only=True)
    assert pipeline_contrast["kendall_tau"] == 1.0 and pipeline_contrast["n_models_that_move"] == 0  # the legacy pipeline's numbers order the models like the AUROC here
    assert metric_contrast["kendall_tau"] == -1.0 and metric_contrast["top_model_conventional"] == "e"  # the statistic alone, on the same replies, orders them the other way
