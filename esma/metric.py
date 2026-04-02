import re

import numpy as np
from scipy.stats import kendalltau, norm

IGNORE_VALUE = -100


# === Original binary metrics (unchanged) ===


def correctness_by_inclusion(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        if any(answer.lower() in output.lower() for answer in answers):
            correctness.append(1)
        else:
            correctness.append(0)
    return correctness


def meta_yes(meta_outputs: list[str]) -> list[int]:
    return [1 if "yes" in output.lower() else 0 for output in meta_outputs]


def meta_yes_ko(meta_outputs: list[str]) -> list[int]:
    return [1 if "예" in output.lower() else 0 for output in meta_outputs]


def meta_yes_cn(meta_outputs: list[str]) -> list[int]:
    return [1 if "是" in output.lower() else 0 for output in meta_outputs]


def meta_yes_es(meta_outputs: list[str]) -> list[int]:
    return [1 if "sí" in output.lower() else 0 for output in meta_outputs]


def meta_wrong_yes(correctness: list[int], yes: list[int], keep_length: bool = False) -> list[int]:
    if not keep_length:
        return [1 - correct for correct, yes in zip(correctness, yes) if yes == 1]
    else:
        return [(IGNORE_VALUE if yes != 1 else 1 - correct) for correct, yes in zip(correctness, yes)]


def meta_wrong_no(correctness: list[int], yes: list[int], keep_length: bool = False) -> list[int]:
    if not keep_length:
        return [correct for correct, yes in zip(correctness, yes) if yes == 0]
    else:
        return [(IGNORE_VALUE if yes != 0 else correct) for correct, yes in zip(correctness, yes)]


def meta_alignment(correctness: list[int], yes: list[int]) -> list[int]:
    return [int(correct == yes) for correct, yes in zip(correctness, yes)]


def type2_d_prime(direct_correctness: list[int], meta_yes: list[int]) -> float:
    hit = [meta_yes[i] for i in range(len(direct_correctness)) if direct_correctness[i] == 1]
    false_alarm = [meta_yes[i] for i in range(len(direct_correctness)) if direct_correctness[i] == 0]

    hit_rate = np.mean(hit)
    false_alarm_rate = np.mean(false_alarm)

    hit_rate = np.clip(hit_rate, 1e-4, 1 - 1e-4)
    false_alarm_rate = np.clip(false_alarm_rate, 1e-4, 1 - 1e-4)

    d_prime_type2 = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    return float(d_prime_type2)


def relative_meta_information(correctness: list[int], yes: list[int]) -> float:
    correctness = np.array(correctness)
    yes = np.array(yes)
    N = correctness.shape[0]

    p_acc = np.mean(correctness)
    if p_acc == 0 or p_acc == 1:
        return 0.0

    h_acc = -(p_acc * np.log2(p_acc) + (1 - p_acc) * np.log2(1 - p_acc))

    counts = np.zeros((2, 2))
    for a, c in zip(correctness, yes):
        counts[a, c] += 1

    p_joint = counts / N
    p_acc_marginal = np.sum(p_joint, axis=1)
    p_conf_marginal = np.sum(p_joint, axis=0)

    mi = 0
    for i in range(2):
        for j in range(2):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_acc_marginal[i] * p_conf_marginal[j]))

    return float(mi / h_acc)


def meta_metrics(
    direct_outputs: list[str],
    meta_outputs: list[str],
    answer_lists: list[list[str]],
    keep_length: bool = False,
    lang: str = "en",
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    direct_correctness = correctness_by_inclusion(direct_outputs, answer_lists)
    if lang == "ko":
        yes = meta_yes_ko(meta_outputs)
    elif lang.startswith("zh"):
        yes = meta_yes_cn(meta_outputs)
    elif lang == "es":
        yes = meta_yes_es(meta_outputs)
    else:
        yes = meta_yes(meta_outputs)
    yes_failures = meta_wrong_yes(direct_correctness, yes, keep_length)
    no_failures = meta_wrong_no(direct_correctness, yes, keep_length)
    meta_alignments = meta_alignment(direct_correctness, yes)
    return direct_correctness, yes, yes_failures, no_failures, meta_alignments


# === NEW: Graded metacognition metrics ===


def parse_graded_response(meta_outputs: list[str]) -> list[int]:
    """Parse ABCD graded responses into ordinal 0-3 scale.

    A=3 (certain know), B=2 (think I know), C=1 (uncertain), D=0 (don't know).
    Falls back to 1 (uncertain) if parsing fails.
    """
    grade_map = {"a": 3, "b": 2, "c": 1, "d": 0}
    results = []
    for output in meta_outputs:
        output_lower = output.strip().lower()
        grade = None
        # Try to find a letter optionally followed by ) or .
        match = re.search(r"\b([abcd])\s*[).\]]?", output_lower)
        if match:
            grade = grade_map[match.group(1)]
        if grade is None:
            # Fallback: look for keywords
            if "certain" in output_lower or "sure" in output_lower:
                grade = 3
            elif "probably" in output_lower or "think" in output_lower:
                grade = 2
            elif "uncertain" in output_lower or "not sure" in output_lower:
                grade = 1
            elif "not know" in output_lower or "no idea" in output_lower:
                grade = 0
            else:
                grade = 1  # default to uncertain
        results.append(grade)
    return results


def parse_numeric_response(meta_outputs: list[str], scale: int = 5) -> list[int]:
    """Parse numeric 1-N responses for FOK/JOL probes.

    Extracts the first valid integer within [1, scale].
    Falls back to midpoint if parsing fails.
    """
    results = []
    for output in meta_outputs:
        nums = re.findall(r"\d+", output.strip())
        parsed = None
        for n in nums:
            val = int(n)
            if 1 <= val <= scale:
                parsed = val
                break
        results.append(parsed if parsed is not None else (scale + 1) // 2)
    return results


def grades_to_binary(grades: list[int], threshold: int) -> list[int]:
    """Convert graded responses to binary yes/no at a given threshold."""
    return [1 if g >= threshold else 0 for g in grades]


def graded_d_prime(correctness: list[int], grades: list[int], threshold: int = 2) -> float:
    """Compute d'_type2 using graded responses binarized at threshold.

    This allows comparison with the original binary d'_type2.
    """
    binarized_yes = grades_to_binary(grades, threshold)
    return type2_d_prime(correctness, binarized_yes)


def multi_threshold_d_prime(correctness: list[int], grades: list[int], max_grade: int = 3) -> dict:
    """Compute d'_type2 at every possible threshold for ROC-like analysis.

    Returns a dict mapping threshold names to d' values.
    """
    results = {}
    for t in range(1, max_grade + 1):
        results[f"d_prime_t{t}"] = graded_d_prime(correctness, grades, threshold=t)
    return results


def graded_alignment(correctness: list[int], grades: list[int], threshold: int = 2) -> list[int]:
    """Alignment where grade >= threshold counts as 'Yes'."""
    return [int((g >= threshold) == (c == 1)) for c, g in zip(correctness, grades)]


def graded_calibration_error(
    correctness: list[int], grades: list[int], num_bins: int | None = None
) -> float:
    """Expected Calibration Error (ECE) adapted for graded metacognition.

    Groups responses by confidence grade, computes |accuracy - expected_accuracy|
    per bin, weighted by bin size. Lower is better.

    If num_bins is None, uses the number of unique grades as bins.
    """
    if not grades:
        return 0.0

    max_grade = max(grades)
    min_grade = min(grades)
    if max_grade == min_grade:
        return 0.0

    unique_grades = sorted(set(grades))
    if num_bins is None:
        num_bins = len(unique_grades)

    total_ece = 0.0
    for g in unique_grades:
        indices = [i for i, gr in enumerate(grades) if gr == g]
        if not indices:
            continue
        bin_acc = np.mean([correctness[i] for i in indices])
        # Linear mapping: grade -> expected accuracy
        expected_acc = (g - min_grade) / (max_grade - min_grade)
        total_ece += len(indices) / len(grades) * abs(bin_acc - expected_acc)
    return float(total_ece)


def metacognitive_resolution(correctness: list[int], grades: list[int]) -> float:
    """Gamma (Goodman-Kruskal) correlation between confidence grades and accuracy.

    This is a standard measure in metacognition psychology research
    (Nelson, 1984; Schwartz & Metcalfe, 2011). Values range from -1 to 1.
    A positive gamma indicates the model's confidence predicts its accuracy.
    """
    if len(set(correctness)) < 2 or len(set(grades)) < 2:
        return 0.0
    tau, _ = kendalltau(grades, correctness)
    return float(tau) if not np.isnan(tau) else 0.0


def type2_auroc(correctness: list[int], grades: list[int]) -> float:
    """Compute Type 2 AUROC from graded confidence scores.

    This is the area under the Type 2 ROC curve, which plots
    P(confidence >= threshold | correct) vs P(confidence >= threshold | incorrect)
    across all possible thresholds. Standard in metacognition research.
    """
    unique_thresholds = sorted(set(grades))
    tpr_list = [1.0]
    fpr_list = [1.0]

    n_correct = sum(correctness)
    n_incorrect = len(correctness) - n_correct

    if n_correct == 0 or n_incorrect == 0:
        return 0.5

    for threshold in unique_thresholds:
        tp = sum(1 for c, g in zip(correctness, grades) if c == 1 and g >= threshold)
        fp = sum(1 for c, g in zip(correctness, grades) if c == 0 and g >= threshold)
        tpr_list.append(tp / n_correct)
        fpr_list.append(fp / n_incorrect)

    tpr_list.append(0.0)
    fpr_list.append(0.0)

    # Sort by FPR for proper AUC computation
    points = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in points]
    tpr_sorted = [p[1] for p in points]

    # np.trapz was removed in NumPy 2.0, renamed to np.trapezoid
    try:
        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    except AttributeError:
        auc = float(np.trapz(tpr_sorted, fpr_sorted))
    return auc


def graded_meta_metrics(
    direct_outputs: list[str],
    meta_outputs: list[str],
    answer_lists: list[list[str]],
    meta_type: str = "graded",
) -> dict:
    """Compute full graded metacognition metrics.

    Args:
        direct_outputs: Model answers to direct questions
        meta_outputs: Model answers to meta questions
        answer_lists: Ground truth answer lists
        meta_type: One of "graded" (ABCD), "fok" (1-5), "numeric" (1-10)

    Returns:
        Dictionary with all computed metrics
    """
    correctness = correctness_by_inclusion(direct_outputs, answer_lists)

    if meta_type == "graded":
        grades = parse_graded_response(meta_outputs)
        max_grade = 3
    elif meta_type in ("fok", "jol"):
        grades = parse_numeric_response(meta_outputs, scale=5)
        max_grade = 5
    elif meta_type == "numeric":
        grades = parse_numeric_response(meta_outputs, scale=10)
        max_grade = 10
    else:
        raise ValueError(f"Unknown meta_type: {meta_type}")

    accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    mean_grade = np.mean(grades) if grades else 0.0

    # Mean grade for correct vs incorrect answers
    correct_grades = [g for c, g in zip(correctness, grades) if c == 1]
    incorrect_grades = [g for c, g in zip(correctness, grades) if c == 0]
    mean_grade_correct = np.mean(correct_grades) if correct_grades else 0.0
    mean_grade_incorrect = np.mean(incorrect_grades) if incorrect_grades else 0.0

    # Grade distribution
    grade_dist = {}
    for g in range(max_grade + 1 if meta_type == "graded" else 1, max_grade + 1):
        grade_dist[f"grade_{g}_ratio"] = sum(1 for x in grades if x == g) / len(grades) if grades else 0.0

    # Core metrics
    results = {
        "accuracy": accuracy,
        "mean_grade": float(mean_grade),
        "mean_grade_correct": float(mean_grade_correct),
        "mean_grade_incorrect": float(mean_grade_incorrect),
        "grade_separation": float(mean_grade_correct - mean_grade_incorrect),
        "gamma_correlation": metacognitive_resolution(correctness, grades),
        "calibration_error": graded_calibration_error(correctness, grades),
        "type2_auroc": type2_auroc(correctness, grades),
    }

    # d' at multiple thresholds
    results.update(multi_threshold_d_prime(correctness, grades, max_grade=max_grade))

    # Grade distribution
    results.update(grade_dist)

    return results, correctness, grades