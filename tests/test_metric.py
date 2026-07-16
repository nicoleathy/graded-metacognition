"""Tests for esma.metric."""

from esma.metric import (
    IGNORE_VALUE,
    correctness_by_inclusion,
    meta_alignment,
    meta_metrics,
    meta_wrong_no,
    meta_wrong_yes,
    meta_yes,
    meta_yes_cn,
    meta_yes_es,
    meta_yes_ko,
    relative_meta_information,
    type2_d_prime,
)


class TestCorrectnessByInclusion:
    """Tests for correctness_by_inclusion."""

    def test_exact_match(self):
        outputs = ["Paris"]
        answer_lists = [["Paris"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [1]

    def test_answer_in_output(self):
        outputs = ["The capital of France is Paris."]
        answer_lists = [["Paris"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [1]

    def test_case_insensitive(self):
        outputs = ["PARIS"]
        answer_lists = [["Paris"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [1]

    def test_wrong_answer(self):
        outputs = ["London"]
        answer_lists = [["Paris"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [0]

    def test_multiple_answers_first_valid(self):
        outputs = ["Paris"]
        answer_lists = [["Paris", "France"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [1]

    def test_multiple_samples(self):
        outputs = ["Paris", "London", "Berlin"]
        answer_lists = [["Paris"], ["London"], ["Tokyo"]]
        assert correctness_by_inclusion(outputs, answer_lists) == [1, 1, 0]


class TestMetaYes:
    """Tests for meta_yes (English)."""

    def test_yes_detected(self):
        assert meta_yes(["yes"]) == [1]
        assert meta_yes(["Yes"]) == [1]
        assert meta_yes(["YES"]) == [1]

    def test_no_not_detected(self):
        assert meta_yes(["no"]) == [0]
        assert meta_yes(["maybe"]) == [0]

    def test_yes_in_sentence(self):
        assert meta_yes(["I think yes"]) == [1]


class TestMetaYesKo:
    """Tests for meta_yes_ko (Korean)."""

    def test_korean_yes(self):
        assert meta_yes_ko(["예"]) == [1]
        assert meta_yes_ko(["예, 맞습니다"]) == [1]

    def test_no_korean_yes(self):
        assert meta_yes_ko(["아니오"]) == [0]


class TestMetaYesCn:
    """Tests for meta_yes_cn (Chinese)."""

    def test_chinese_yes(self):
        assert meta_yes_cn(["是"]) == [1]

    def test_no_chinese_yes(self):
        assert meta_yes_cn(["否"]) == [0]


class TestMetaYesEs:
    """Tests for meta_yes_es (Spanish)."""

    def test_spanish_yes(self):
        assert meta_yes_es(["sí"]) == [1]

    def test_no_spanish_yes(self):
        assert meta_yes_es(["no"]) == [0]


class TestMetaWrongYes:
    """Tests for meta_wrong_yes."""

    def test_keep_length_false_filters_to_yes_only(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 1, 0, 0]
        result = meta_wrong_yes(correctness, yes, keep_length=False)
        assert result == [0, 1]  # only where yes==1: 1-correct

    def test_keep_length_true_uses_ignore_value(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 1, 0, 0]
        result = meta_wrong_yes(correctness, yes, keep_length=True)
        assert result == [0, 1, IGNORE_VALUE, IGNORE_VALUE]


class TestMetaWrongNo:
    """Tests for meta_wrong_no."""

    def test_keep_length_false_filters_to_no_only(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 1, 0, 0]
        result = meta_wrong_no(correctness, yes, keep_length=False)
        assert result == [1, 0]  # only where yes==0: correct value

    def test_keep_length_true_uses_ignore_value(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 1, 0, 0]
        result = meta_wrong_no(correctness, yes, keep_length=True)
        assert result == [IGNORE_VALUE, IGNORE_VALUE, 1, 0]


class TestMetaAlignment:
    """Tests for meta_alignment."""

    def test_aligned(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 0, 1, 0]
        assert meta_alignment(correctness, yes) == [1, 1, 1, 1]

    def test_misaligned(self):
        correctness = [1, 0]
        yes = [0, 1]
        assert meta_alignment(correctness, yes) == [0, 0]


class TestType2DPrime:
    """Tests for type2_d_prime."""

    def test_perfect_metacognition_positive_d_prime(self):
        # When correct -> yes and wrong -> no, hit rate high, FA low
        direct_correctness = [1, 1, 1, 0, 0, 0]
        meta_yes = [1, 1, 1, 0, 0, 0]
        d = type2_d_prime(direct_correctness, meta_yes)
        assert d > 0

    def test_anti_metacognition_negative_d_prime(self):
        direct_correctness = [1, 1, 0, 0]
        meta_yes = [0, 0, 1, 1]  # say no when correct, yes when wrong
        d = type2_d_prime(direct_correctness, meta_yes)
        assert d < 0

    def test_returns_float(self):
        direct_correctness = [1, 0, 1, 0]
        meta_yes = [1, 0, 0, 1]
        d = type2_d_prime(direct_correctness, meta_yes)
        assert isinstance(d, float)


class TestRelativeMetaInformation:
    """Tests for relative_meta_information."""

    def test_perfect_alignment_nonzero(self):
        correctness = [1, 0, 1, 0]
        yes = [1, 0, 1, 0]
        rmi = relative_meta_information(correctness, yes)
        assert rmi > 0
        assert isinstance(rmi, float)

    def test_zero_when_accuracy_extreme(self):
        # p_acc = 0 or 1 returns 0
        rmi0 = relative_meta_information([0, 0, 0], [0, 0, 0])
        rmi1 = relative_meta_information([1, 1, 1], [1, 1, 1])
        assert rmi0 == 0.0
        assert rmi1 == 0.0

    def test_bounded(self):
        correctness = [1, 0, 1, 0, 1, 0]
        yes = [1, 0, 0, 1, 1, 0]
        rmi = relative_meta_information(correctness, yes)
        assert 0 <= rmi <= 1


class TestMetaMetrics:
    """Tests for meta_metrics."""

    def test_returns_five_lists(self):
        direct = ["Paris", "London"]
        meta = ["yes", "no"]
        answers = [["Paris"], ["Tokyo"]]
        result = meta_metrics(direct, meta, answers, keep_length=True)
        assert len(result) == 5
        direct_correctness, yes, yes_failures, no_failures, alignments = result
        assert len(direct_correctness) == 2
        assert len(yes) == 2
        assert len(yes_failures) == 2
        assert len(no_failures) == 2
        assert len(alignments) == 2

    def test_lang_en_uses_meta_yes(self):
        direct = ["Paris"]
        meta = ["yes"]
        answers = [["Paris"]]
        direct_correctness, yes, _, _, _ = meta_metrics(direct, meta, answers, lang="en")
        assert yes == [1]

    def test_lang_ko_uses_meta_yes_ko(self):
        direct = ["Paris"]
        meta = ["예"]
        answers = [["Paris"]]
        _, yes, _, _, _ = meta_metrics(direct, meta, answers, lang="ko")
        assert yes == [1]

    def test_keep_length_affects_failures_length(self):
        direct = ["Paris", "London"]
        meta = ["yes", "no"]
        answers = [["Paris"], ["London"]]
        _, _, yes_f, no_f, _ = meta_metrics(direct, meta, answers, keep_length=True)
        assert len(yes_f) == 2
        assert len(no_f) == 2
