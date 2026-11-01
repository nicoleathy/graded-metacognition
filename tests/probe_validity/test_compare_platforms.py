"""The platform calibration compares two runs of one model on the same items. These tests give it runs that are identical, runs that differ in known ways, and
settings that differ, and check that what it reports is what was built in."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1/cluster"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import compare_platforms as C  # noqa: E402


def rows(texts, lp=None):
    return [{"i": i, "text": t, "label_lp": ({"A": lp[i][0], "B": lp[i][1]} if lp else {})} for i, t in enumerate(texts)]


def test_identical_runs_have_identical_replies_and_no_drift():
    a = rows(["A", "B", "C", "D"], lp=[(-0.1, -2.3)] * 4)
    out = C.compare_rows(a, [dict(r) for r in a])
    assert out["items"] == 4 and out["identical_replies"] == 1.0
    assert out["label_logprob_drift"] == {"median": 0.0, "p95": 0.0, "max": 0.0}


def test_a_changed_reply_and_a_numerical_drift_are_counted():
    a = rows(["A", "B", "C", "D"], lp=[(-0.10, -2.30)] * 4)
    b = rows(["A", "B", "C", "A"], lp=[(-0.10, -2.30), (-0.10, -2.30), (-0.12, -2.30), (-0.10, -2.50)])
    out = C.compare_rows(a, b)
    assert out["identical_replies"] == 0.75  # one of four replies differs
    assert abs(out["label_logprob_drift"]["max"] - 0.2) < 1e-9  # the largest first-token log-probability difference
    assert C.compare_rows(a + [None], b)["items"] == 4  # an item without a row in one run is not compared


def test_paired_difference_is_zero_for_equal_runs_and_its_interval_excludes_a_large_shift():
    x = np.array([1, 0, 1, 1, 0, 1, 0, 1] * 50)
    same = C.paired_difference(x, x.copy(), 200)
    assert same["diff"] == 0 and same["ci"] == [0.0, 0.0] and same["n"] == 400
    shifted = C.paired_difference(x, np.zeros_like(x), 200)
    assert abs(shifted["diff"] - x.mean()) < 1e-12 and shifted["ci"][0] > 0.4


def test_probe_comparison_reports_compliance_agreement_and_auroc_difference():
    rng = np.random.default_rng(0)
    n = 600
    correct = (rng.random(n) < 0.5).astype(int)
    grade = np.where(rng.random(n) < 0.3 + 0.4 * correct, 1.0, 0.0)  # an informative two-level probe
    same = C.compare_probe(correct, grade, correct, grade.copy(), 100)
    assert same["compliance"]["diff"] == 0 and same["same_grade_where_both_compliant"] == 1.0 and same["auroc"]["diff"] == 0
    other = grade.copy()
    other[:60] = np.nan  # the second platform fails to comply on 10% of items
    flipped = grade.copy()
    flipped[100:200] = 1 - flipped[100:200]  # and answers 100 items with the opposite grade
    out = C.compare_probe(correct, grade, correct, np.where(np.isnan(other), np.nan, flipped), 100)
    assert abs(out["compliance"]["diff"] - 0.1) < 1e-9 and out["compliance"]["ci"][0] > 0
    assert out["same_grade_where_both_compliant"] < 1.0 and out["auroc"]["n"] == n - 60 and out["auroc"]["diff"] > 0  # flipping grades lowers the second AUROC


def test_settings_that_must_be_equal_are_checked_and_the_platform_fields_are_reported_as_they_are():
    a = {"revision": "r1", "dtype": "bfloat16", "batch_size": 16, "device": "mps", "attention": "eager", "torch": "2.8.0"}
    b = {**a, "device": "cuda", "attention": "sdpa", "torch": "2.8.0+cu128"}
    out = C.compare_meta(a, b)
    assert out["settings_equal"]["revision"] and out["settings_equal"]["batch_size"] and all(out["settings_equal"].values())  # absent keys agree (both None)
    assert out["platform"]["device"] == ["mps", "cuda"] and out["platform"]["attention"] == ["eager", "sdpa"]
    assert C.compare_meta(a, {**b, "revision": "r2"})["settings_equal"]["revision"] is False  # another model revision would confound the comparison


def test_a_run_directory_that_is_short_or_was_changed_after_it_was_recorded_fails_the_integrity_check(tmp_path, monkeypatch):
    from run_sweep import file_sha256

    import verify_runs

    monkeypatch.setattr(verify_runs, "expected_rows", lambda dataset, n: {"cond": 2})  # a protocol with one condition of two rows
    (tmp_path / "meta.json").write_text(json.dumps({"dataset": "mmlu", "n": 2}))
    path = tmp_path / "cond.jsonl"
    path.write_text("".join(json.dumps({"i": i, "condition": "cond", "text": "A"}) + "\n" for i in range(2)))
    (tmp_path / "conditions.jsonl").write_text(json.dumps({"condition": "cond", "rows_sha256": file_sha256(path)}) + "\n")
    assert C.integrity(tmp_path) == []
    path.write_text(path.read_text().replace('"A"', '"B"', 1))  # one reply edited after the hash was recorded
    assert C.integrity(tmp_path) == ["cond: the file changed after it was recorded"]
    path.write_text(path.read_text().split("\n")[0] + "\n")  # and one row missing
    assert any("1 of 2 rows" in p for p in C.integrity(tmp_path))
