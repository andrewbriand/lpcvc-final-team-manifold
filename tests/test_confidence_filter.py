import numpy as np
from self_training.confidence_filter import calibrate_thresholds, apply_confidence_filter


def test_calibrate_thresholds():
    rng = np.random.RandomState(42)
    top1_scores = rng.uniform(0.1, 0.5, size=1000).astype(np.float32)
    top2_scores = top1_scores - rng.uniform(0.01, 0.1, size=1000).astype(np.float32)
    thresholds = calibrate_thresholds(top1_scores, top2_scores)
    assert "min_top1_cosine" in thresholds
    assert "min_margin" in thresholds
    assert thresholds["min_top1_cosine"] > 0.1
    assert thresholds["min_margin"] > 0.0


def test_apply_confidence_filter():
    top1_scores = np.array([0.4, 0.2, 0.35, 0.1, 0.5], dtype=np.float32)
    top2_scores = np.array([0.3, 0.19, 0.34, 0.05, 0.2], dtype=np.float32)
    mask = apply_confidence_filter(top1_scores, top2_scores, min_top1_cosine=0.25, min_margin=0.02)
    expected = np.array([True, False, False, False, True])
    np.testing.assert_array_equal(mask, expected)
