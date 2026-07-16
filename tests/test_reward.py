"""Tests for esma.reward."""

from esma.reward import (
    REWARD_TYPE_TO_FUNCTION,
    correct_reward,
    esma_reward,
    meta_alignment_reward,
)


class TestCorrectReward:
    """Tests for correct_reward."""

    def test_returns_copy_of_direct_correctness(self):
        direct_correctness = [1, 0, 1]
        meta_yes = [1, 0, 0]
        result = correct_reward(direct_correctness, meta_yes)
        assert result == [1, 0, 1]
        assert result is not direct_correctness

    def test_ignores_meta_yes(self):
        direct_correctness = [1, 0]
        result = correct_reward(direct_correctness, [0, 1])
        assert result == [1, 0]


class TestMetaAlignmentReward:
    """Tests for meta_alignment_reward."""

    def test_aligned_correct_gives_one(self):
        direct_correctness = [1, 0]
        meta_yes = [1, 0]
        assert meta_alignment_reward(direct_correctness, meta_yes) == [1, 1]

    def test_misaligned_gives_zero(self):
        direct_correctness = [1, 0]
        meta_yes = [0, 1]
        assert meta_alignment_reward(direct_correctness, meta_yes) == [0, 0]

    def test_mixed(self):
        direct_correctness = [1, 0, 1, 0]
        meta_yes = [1, 0, 0, 1]
        assert meta_alignment_reward(direct_correctness, meta_yes) == [1, 1, 0, 0]


class TestEsmaReward:
    """Tests for esma_reward."""

    def test_correct_yes_reward_2(self):
        # correct=1, yes=1 -> 2
        result = esma_reward([1], [1])
        assert result == [2]

    def test_wrong_no_reward_1(self):
        # correct=0, yes=0 -> 1
        result = esma_reward([0], [0])
        assert result == [1]

    def test_correct_but_said_no_reward_1(self):
        # correct=1, yes=0 -> 1
        result = esma_reward([1], [0])
        assert result == [1]

    def test_wrong_but_said_yes_reward_0(self):
        # correct=0, yes=1 -> 0
        result = esma_reward([0], [1])
        assert result == [0]

    def test_all_four_cases(self):
        direct_correctness = [1, 0, 1, 0]
        meta_yes = [1, 0, 0, 1]
        result = esma_reward(direct_correctness, meta_yes)
        assert result == [2, 1, 1, 0]

    def test_length_matches_input(self):
        direct_correctness = [1, 0, 1, 0, 1]
        meta_yes = [1, 1, 0, 0, 1]
        result = esma_reward(direct_correctness, meta_yes)
        assert len(result) == 5


class TestRewardTypeToFunction:
    """Tests for REWARD_TYPE_TO_FUNCTION mapping."""

    def test_has_expected_keys(self):
        assert set(REWARD_TYPE_TO_FUNCTION.keys()) == {"correct", "alignment", "esma"}

    def test_correct_maps_to_correct_reward(self):
        assert REWARD_TYPE_TO_FUNCTION["correct"] is correct_reward

    def test_alignment_maps_to_meta_alignment_reward(self):
        assert REWARD_TYPE_TO_FUNCTION["alignment"] is meta_alignment_reward

    def test_esma_maps_to_esma_reward(self):
        assert REWARD_TYPE_TO_FUNCTION["esma"] is esma_reward

    def test_invocation_works(self):
        direct = [1, 0]
        meta = [1, 0]
        for name, fn in REWARD_TYPE_TO_FUNCTION.items():
            result = fn(direct, meta)
            assert len(result) == 2
            assert all(isinstance(r, int) for r in result)
