"""Does the pipeline change which model looks best? Models are ranked by what a conventional pipeline reports (rescaled gamma on every
reply, with replies that name no label imputed) and by what the strict pipeline reports (AUROC on compliant replies), per dataset and probe,
from the summaries of the runs. The comparison is descriptive: with a dozen models the rank correlation has a wide interval, so it is
reported with a bootstrap over models, and the number of models that move is shown next to it.

Two contrasts are kept apart. The default compares the legacy pipeline end to end (its own answer prompt, answer scorer, parsing and imputation) with the strict
pipeline: every difference at once, so a reordering cannot be attributed to the metric. With metric_only=True both sides use the same compliant replies, items and strict
correctness labels and differ only in the statistic (rescaled gamma against AUROC), which isolates the effect of the metric.
"""

import numpy as np
from blocks import r4
from scipy.stats import kendalltau


def ranks(values: dict) -> dict:
    """Rank 1 = highest value; ties share the average rank."""
    keys = list(values)
    order = np.argsort(-np.array([values[k] for k in keys]), kind="stable")
    out = {}
    for pos, i in enumerate(order, 1):
        out[keys[i]] = float(pos)
    for v in {values[k] for k in keys}:  # average the ranks of tied values
        tied = [k for k in keys if values[k] == v]
        if len(tied) > 1:
            mean = float(np.mean([out[k] for k in tied]))
            for k in tied:
                out[k] = mean
    return out


def compare(summary: dict, dataset: str, probe: str, legacy_key: str = "gamma_star_imputed", n_boot: int = 2000, seed: int = 0, metric_only: bool = False) -> dict | None:
    """Kendall's tau between the conventional ranking and the strict ranking over the models with both numbers, and how many models move.
    metric_only: the conventional side is gamma* on the strict replies and labels (the metric alone) instead of the legacy pipeline's own number."""
    legacy, strict = {}, {}
    for model, datasets in summary["models"].items():
        block = ((datasets.get(dataset) or {}).get("legacy_pipeline") or {}).get(probe)
        if not block:
            continue
        a = (block["strict"].get("auroc") or {}).get("auroc")
        g = block["strict"].get("gamma_star") if metric_only else block["legacy"].get(legacy_key)
        if a is not None and g is not None:
            legacy[model], strict[model] = float(g), float(a)
    if len(legacy) < 4:
        return None
    models = sorted(legacy)
    x, y = np.array([legacy[m] for m in models]), np.array([strict[m] for m in models])
    tau = float(kendalltau(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws = [kendalltau(x[idx], y[idx]).statistic for idx in (rng.integers(0, len(models), len(models)) for _ in range(n_boot))]
    draws = np.array([d for d in draws if not np.isnan(d)])
    r_legacy, r_strict = ranks(legacy), ranks(strict)
    moved = {m: r_legacy[m] - r_strict[m] for m in models}
    return {"n_models": len(models), "kendall_tau": r4(tau), "tau_ci": [r4(np.percentile(draws, 2.5)), r4(np.percentile(draws, 97.5))] if draws.size else None,
            "n_models_that_move": int(sum(abs(v) >= 1 for v in moved.values())), "n_models_that_move_by_2_or_more": int(sum(abs(v) >= 2 for v in moved.values())),
            "top_model_conventional": max(legacy, key=legacy.get), "top_model_strict": max(strict, key=strict.get),
            "rank_change": {m: moved[m] for m in models}}
