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


def correctness_mmlu(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    """Strict MMLU correctness: match on the first A/B/C/D letter the model produces.

    MMLU correctness cannot use substring inclusion because the prompt itself
    contains "A)", "B)", "C)", "D)" inside the question, and models frequently
    echo the question. `answer_lists[i][0]` is expected to be the ground-truth
    letter (A/B/C/D).
    """
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        gold_letter = answers[0].strip().upper() if answers else ""
        match = re.search(r"\b([ABCD])\b", output.upper())
        pred = match.group(1) if match else None
        correctness.append(int(pred == gold_letter) if pred is not None else 0)
    return correctness


def correctness_gsm8k(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    """Strict GSM8K correctness: compare the final number the model produces
    against the gold numeric answer (with and without commas).

    Using substring inclusion causes "420" to count as correct when the gold
    answer is "42", which silently corrupts correctness labels.
    """
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        # Extract all numeric tokens from the model's output, take the last one
        # (chain-of-thought-style reasoning typically concludes with the answer).
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", output)
        if not nums:
            correctness.append(0)
            continue
        pred_raw = nums[-1].replace(",", "")
        try:
            pred_val = float(pred_raw)
        except ValueError:
            correctness.append(0)
            continue

        gold_vals = []
        for a in answers:
            a_clean = a.replace(",", "").strip()
            try:
                gold_vals.append(float(a_clean))
            except ValueError:
                continue
        correctness.append(int(any(abs(pred_val - g) < 1e-6 for g in gold_vals)))
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
    correctness_fn=None,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    if correctness_fn is None:
        correctness_fn = correctness_by_inclusion
    direct_correctness = correctness_fn(direct_outputs, answer_lists)
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


def parse_graded_response_strict(meta_outputs: list[str]) -> tuple[list[int], list[bool]]:
    """Same as parse_graded_response but also returns a mask of which responses
    required the fallback/default.

    Returns
    -------
    grades : list[int]
        Parsed grades (with fallback applied where needed).
    parsed_ok : list[bool]
        True iff the grade was extracted from a clean A/B/C/D letter match.
    """
    grade_map = {"a": 3, "b": 2, "c": 1, "d": 0}
    grades: list[int] = []
    parsed_ok: list[bool] = []
    for output in meta_outputs:
        output_lower = output.strip().lower()
        match = re.search(r"\b([abcd])\s*[).\]]?", output_lower)
        if match:
            grades.append(grade_map[match.group(1)])
            parsed_ok.append(True)
            continue
        if "certain" in output_lower or "sure" in output_lower:
            grades.append(3)
        elif "probably" in output_lower or "think" in output_lower:
            grades.append(2)
        elif "uncertain" in output_lower or "not sure" in output_lower:
            grades.append(1)
        elif "not know" in output_lower or "no idea" in output_lower:
            grades.append(0)
        else:
            grades.append(1)
        parsed_ok.append(False)
    return grades, parsed_ok


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


def goodman_kruskal_gamma(correctness: list[int], grades: list[int]) -> float:
    """Goodman-Kruskal gamma: (C - D) / (C + D) over untied pairs only.

    This is the correct operationalization of gamma as used in the
    metacognition literature (Nelson, 1984). Unlike Kendall's tau-b,
    gamma ignores tied pairs entirely, which matters a great deal for
    ordinal scales with few levels (e.g. 4-level A-D) where ties are
    the majority of pairs.

    Returns 0.0 when there are no untied pairs or when either variable
    has only one unique value.
    """
    if len(correctness) != len(grades):
        raise ValueError("correctness and grades must be the same length")
    n = len(correctness)
    if n < 2:
        return 0.0
    if len(set(correctness)) < 2 or len(set(grades)) < 2:
        return 0.0

    c_arr = np.asarray(correctness)
    g_arr = np.asarray(grades)

    concordant = 0
    discordant = 0
    # O(n^2) is fine for n up to ~10k; for larger n, switch to a sort-based
    # inversion-count implementation.
    for i in range(n):
        dg = g_arr[i + 1 :] - g_arr[i]
        dc = c_arr[i + 1 :] - c_arr[i]
        prod = dg * dc
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))

    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


def metacognitive_resolution(
    correctness: list[int], grades: list[int], method: str = "gamma"
) -> float:
    """Rank correlation between confidence grades and accuracy.

    method="gamma" (default): Goodman-Kruskal gamma, the standard measure
        in metacognition research (Nelson, 1984). Ignores tied pairs.
    method="kendall": Kendall's tau-b (the previous implementation),
        retained for backward compatibility.

    Values range from -1 to 1. Positive values mean confidence predicts
    accuracy.
    """
    if len(set(correctness)) < 2 or len(set(grades)) < 2:
        return 0.0
    if method == "gamma":
        return goodman_kruskal_gamma(correctness, grades)
    elif method == "kendall":
        tau, _ = kendalltau(grades, correctness)
        return float(tau) if not np.isnan(tau) else 0.0
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'gamma' or 'kendall'.")


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


def type2_auroc_continuous(correctness: list[int], scores: list[float]) -> float:
    """Type 2 AUROC for *continuous* confidence scores (e.g. logit-based).

    Uses the standard rank-sum (Mann-Whitney U) formulation, which matches
    the definition of AUROC for arbitrary real-valued scores and does not
    require binning.
    """
    c_arr = np.asarray(correctness, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    n_correct = int(np.sum(c_arr == 1))
    n_incorrect = int(np.sum(c_arr == 0))
    if n_correct == 0 or n_incorrect == 0:
        return 0.5

    # AUROC = P(score_correct > score_incorrect) + 0.5 * P(score_correct == score_incorrect)
    s_correct = s_arr[c_arr == 1]
    s_incorrect = s_arr[c_arr == 0]
    greater = 0.0
    equal = 0.0
    # Vectorized pairwise comparison; O(n_c * n_i) memory, fine for n ~= 2k.
    diff = s_correct[:, None] - s_incorrect[None, :]
    greater = float(np.sum(diff > 0))
    equal = float(np.sum(diff == 0))
    return (greater + 0.5 * equal) / (n_correct * n_incorrect)


def expected_confidence_from_logits(
    first_token_logits: np.ndarray,
    option_token_ids: dict[int, list[int]],
    max_grade: int,
) -> float:
    """Compute expected verbalized confidence from first-token logits.

    Given the logits over the vocabulary at the first generated position,
    project onto the probe option tokens (e.g. {A, B, C, D} token IDs),
    renormalize to a proper distribution over options, and return the
    expected grade normalized to [0, 1].

    Parameters
    ----------
    first_token_logits : np.ndarray, shape (vocab_size,)
        Logits for the first generated meta-response token.
    option_token_ids : dict[int, list[int]]
        Mapping from grade value (e.g. 0, 1, 2, 3 for D, C, B, A) to one
        or more candidate token IDs (to handle tokenizer variants such as
        leading-space vs no-leading-space tokens). Probabilities of all
        listed token IDs are summed per grade.
    max_grade : int
        Maximum grade value on the scale (e.g. 3 for A-D, 5 for FOK).

    Returns
    -------
    expected_confidence : float in [0, 1]
        Expected grade / max_grade under the renormalized option distribution.
    """
    logits = np.asarray(first_token_logits, dtype=np.float64)
    # Log-sum-exp stabilization over the union of option tokens.
    all_ids = [tid for ids in option_token_ids.values() for tid in ids]
    if not all_ids:
        return 0.5
    option_logits = logits[all_ids]
    m = np.max(option_logits)
    # Probability mass per grade, summed over tokenizer-variant IDs.
    grade_masses = {}
    for g, ids in option_token_ids.items():
        grade_masses[g] = float(np.sum(np.exp(logits[ids] - m)))
    total = sum(grade_masses.values())
    if total <= 0:
        return 0.5
    expected_grade = 0.0
    for g, mass in grade_masses.items():
        expected_grade += g * (mass / total)
    return expected_grade / max_grade if max_grade > 0 else 0.5


def build_option_token_ids(tokenizer, meta_type: str) -> dict[int, list[int]]:
    """Build a mapping grade -> candidate first-token IDs for a given probe.

    Handles both leading-space (" A") and bare ("A") token variants, which
    differ across tokenizers (e.g. Llama vs Qwen vs Mistral).
    """
    if meta_type == "graded":
        # ABCD -> 3,2,1,0
        letter_to_grade = {"A": 3, "B": 2, "C": 1, "D": 0}
        mapping: dict[int, list[int]] = {}
        for letter, grade in letter_to_grade.items():
            ids = set()
            for variant in (letter, " " + letter, letter + ")", letter + "."):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            mapping[grade] = sorted(ids)
        return mapping
    elif meta_type in ("fok", "jol"):
        mapping = {}
        for n in range(1, 6):
            ids = set()
            for variant in (str(n), " " + str(n)):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            mapping[n] = sorted(ids)
        return mapping
    elif meta_type == "numeric":
        mapping = {}
        for n in range(1, 11):
            ids = set()
            for variant in (str(n), " " + str(n)):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            mapping[n] = sorted(ids)
        return mapping
    else:
        raise ValueError(f"build_option_token_ids: unsupported meta_type {meta_type!r}")


def graded_meta_metrics(
    direct_outputs: list[str],
    meta_outputs: list[str],
    answer_lists: list[list[str]],
    meta_type: str = "graded",
    correctness_fn=None,
    logit_confidences: list[float] | None = None,
) -> tuple[dict, list[int], list[int]]:
    """Compute full graded metacognition metrics.

    Args:
        direct_outputs: Model answers to direct questions
        meta_outputs: Model answers to meta questions
        answer_lists: Ground truth answer lists
        meta_type: One of "graded" (ABCD), "fok" (1-5), "numeric" (1-10)
        correctness_fn: Function to score correctness. Defaults to
            correctness_by_inclusion. Use correctness_mmlu for MMLU and
            correctness_gsm8k for GSM8K.
        logit_confidences: Optional list of float confidences in [0, 1]
            derived from first-token logits, aligned with meta_outputs.
            If provided, additional metrics are reported.

    Returns:
        Tuple of (metrics_dict, correctness_list, grades_list)
    """
    if correctness_fn is None:
        correctness_fn = correctness_by_inclusion
    correctness = correctness_fn(direct_outputs, answer_lists)

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

    # Grade distribution: iterate over the full grade range for the scale.
    # (Previous version had a bug: for "graded" the loop range was empty.)
    grade_dist_start = 0 if meta_type == "graded" else 1
    grade_dist = {}
    for g in range(grade_dist_start, max_grade + 1):
        grade_dist[f"grade_{g}_ratio"] = (
            sum(1 for x in grades if x == g) / len(grades) if grades else 0.0
        )

    # Core metrics
    results = {
        "accuracy": accuracy,
        "mean_grade": float(mean_grade),
        "mean_grade_correct": float(mean_grade_correct),
        "mean_grade_incorrect": float(mean_grade_incorrect),
        "grade_separation": float(mean_grade_correct - mean_grade_incorrect),
        "gamma_correlation": metacognitive_resolution(correctness, grades, method="gamma"),
        "kendall_tau": metacognitive_resolution(correctness, grades, method="kendall"),
        "calibration_error": graded_calibration_error(correctness, grades),
        "type2_auroc": type2_auroc(correctness, grades),
    }

    # d' at multiple thresholds
    results.update(multi_threshold_d_prime(correctness, grades, max_grade=max_grade))

    # Grade distribution
    results.update(grade_dist)

    # Logit-based confidence metrics (implicit confidence)
    if logit_confidences is not None and len(logit_confidences) == len(correctness):
        lc = np.asarray(logit_confidences, dtype=float)
        # Verbalized confidence normalized to [0, 1] for a fair comparison
        verbal_norm = np.asarray(grades, dtype=float) / max_grade if max_grade > 0 else np.zeros_like(lc)
        results["logit_mean_confidence"] = float(np.mean(lc))
        results["logit_mean_confidence_correct"] = (
            float(np.mean(lc[np.asarray(correctness) == 1])) if any(correctness) else 0.0
        )
        results["logit_mean_confidence_incorrect"] = (
            float(np.mean(lc[np.asarray(correctness) == 0])) if any(c == 0 for c in correctness) else 0.0
        )
        results["logit_type2_auroc"] = type2_auroc_continuous(correctness, lc.tolist())
        # Pearson correlation between verbalized (normalized) and logit confidence
        if np.std(lc) > 0 and np.std(verbal_norm) > 0:
            results["verbal_logit_pearson"] = float(np.corrcoef(verbal_norm, lc)[0, 1])
        else:
            results["verbal_logit_pearson"] = 0.0

    return results, correctness, grades
