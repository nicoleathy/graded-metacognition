# === Original binary reward functions ===


def correct_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    return direct_correctness.copy()


def meta_alignment_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    return [int(correct == yes) for correct, yes in zip(direct_correctness, meta_yes)]


def esma_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(2)
            else:
                rewards.append(1)
        else:
            if correct:
                rewards.append(1)
            else:
                rewards.append(0)
    return rewards


# === NEW: Graded reward functions ===


def graded_esma_reward(direct_correctness: list[int], meta_grades: list[int]) -> list[float]:
    """Graded reward that rewards calibrated confidence proportionally.

    When the model is correct, higher confidence = higher reward.
    When the model is incorrect, lower confidence = higher reward.
    This incentivizes the model to calibrate its confidence grades
    to match its actual knowledge state.

    Assumes grades are on a 0-3 scale (ABCD: D=0, C=1, B=2, A=3).
    Reward range: [0.0, 2.0]
    """
    rewards = []
    for correct, grade in zip(direct_correctness, meta_grades):
        max_grade = 3  # for ABCD scale (0-3)
        normalized = grade / max_grade  # 0.0 to 1.0

        if correct:
            # Correct: reward proportional to confidence
            # High confidence + correct = 2.0 (best)
            # Low confidence + correct = 1.0 (missed opportunity)
            rewards.append(1.0 + normalized)
        else:
            # Incorrect: reward inversely proportional to confidence
            # Low confidence + incorrect = 1.0 (good self-awareness)
            # High confidence + incorrect = 0.0 (worst: overconfident and wrong)
            rewards.append(1.0 - normalized)
    return rewards


def graded_fok_reward(direct_correctness: list[int], meta_grades: list[int]) -> list[float]:
    """Graded reward for FOK/JOL probes (1-5 scale).

    Same logic as graded_esma_reward but calibrated for 1-5 scale.
    """
    rewards = []
    for correct, grade in zip(direct_correctness, meta_grades):
        normalized = (grade - 1) / 4  # map 1-5 to 0.0-1.0

        if correct:
            rewards.append(1.0 + normalized)
        else:
            rewards.append(1.0 - normalized)
    return rewards


def graded_numeric_reward(direct_correctness: list[int], meta_grades: list[int]) -> list[float]:
    """Graded reward for numeric 1-10 scale."""
    rewards = []
    for correct, grade in zip(direct_correctness, meta_grades):
        normalized = (grade - 1) / 9  # map 1-10 to 0.0-1.0

        if correct:
            rewards.append(1.0 + normalized)
        else:
            rewards.append(1.0 - normalized)
    return rewards


def graded_quadratic_reward(direct_correctness: list[int], meta_grades: list[int]) -> list[float]:
    """Quadratic scoring rule (Brier-inspired) for graded confidence.

    Uses quadratic penalty to more strongly punish overconfidence.
    Assumes grades on 0-3 scale.
    """
    rewards = []
    for correct, grade in zip(direct_correctness, meta_grades):
        normalized = grade / 3  # 0.0 to 1.0

        # Brier-style: minimize (confidence - correctness)^2
        # Convert to reward: 2 - 2*(confidence - correctness)^2
        brier = (normalized - correct) ** 2
        rewards.append(2.0 * (1.0 - brier))
    return rewards


REWARD_TYPE_TO_FUNCTION = {
    # Original binary
    "correct": correct_reward,
    "alignment": meta_alignment_reward,
    "esma": esma_reward,
    # New graded
    "graded": graded_esma_reward,
    "graded_fok": graded_fok_reward,
    "graded_numeric": graded_numeric_reward,
    "graded_quadratic": graded_quadratic_reward,
}
