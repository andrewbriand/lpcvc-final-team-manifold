import numpy as np
from self_training.diagnostics import compute_retrieval_diagnostics, format_diagnostic_summary, compare_reports


def _make_test_data(n_images=20, n_texts=50, dim=64):
    """Create synthetic retrieval data where each image has 1-2 GT captions."""
    rng = np.random.RandomState(42)
    img_embs = rng.randn(n_images, dim).astype(np.float32)
    txt_embs = rng.randn(n_texts, dim).astype(np.float32)
    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt_embs /= np.linalg.norm(txt_embs, axis=1, keepdims=True)

    # Make GT captions similar to their images (add signal)
    img2txt = []
    for i in range(n_images):
        gt = [i % n_texts]
        if i < n_images // 2:
            gt.append((i + n_images) % n_texts)
        img2txt.append(gt)
        for g in gt:
            txt_embs[g] = img_embs[i] + rng.randn(dim).astype(np.float32) * 0.3
            txt_embs[g] /= np.linalg.norm(txt_embs[g])

    captions = [f"caption number {i} about something" for i in range(n_texts)]
    return img_embs, txt_embs, img2txt, captions


def test_compute_diagnostics_structure():
    img_embs, txt_embs, img2txt, captions = _make_test_data()
    report = compute_retrieval_diagnostics(img_embs, txt_embs, img2txt, captions)

    assert "aggregate" in report
    assert "similarity_stats" in report
    assert "failure_analysis" in report
    assert "caption_analysis" in report
    assert "hardest_queries" in report
    assert "easiest_queries" in report
    assert "per_query" in report

    agg = report["aggregate"]
    assert 0 <= agg["R@1"] <= 1
    assert 0 <= agg["R@10"] <= 1
    assert agg["MedR"] >= 1
    assert agg["n_queries"] == 20


def test_diagnostics_metrics_reasonable():
    img_embs, txt_embs, img2txt, captions = _make_test_data()
    report = compute_retrieval_diagnostics(img_embs, txt_embs, img2txt, captions)

    # With signal injected, R@10 should be decent
    assert report["aggregate"]["R@10"] > 0.3

    # Positive similarities should be higher than negatives on average
    assert report["similarity_stats"]["positive"]["mean"] > report["similarity_stats"]["negative"]["mean"]


def test_format_summary():
    img_embs, txt_embs, img2txt, captions = _make_test_data()
    report = compute_retrieval_diagnostics(img_embs, txt_embs, img2txt, captions)
    summary = format_diagnostic_summary(report)
    assert "RETRIEVAL DIAGNOSTIC SUMMARY" in summary
    assert "R@1:" in summary
    assert "Rank Distribution" in summary
    assert "Failure Modes" in summary


def test_compare_reports():
    img_embs, txt_embs, img2txt, captions = _make_test_data()
    baseline = compute_retrieval_diagnostics(img_embs, txt_embs, img2txt, captions)

    # "Train" by slightly shifting embeddings
    trained_embs = img_embs + np.random.RandomState(99).randn(*img_embs.shape).astype(np.float32) * 0.1
    trained_embs /= np.linalg.norm(trained_embs, axis=1, keepdims=True)
    trained = compute_retrieval_diagnostics(trained_embs, txt_embs, img2txt, captions)

    comparison = compare_reports(baseline, trained)
    assert "COMPARISON" in comparison
    assert "Improved" in comparison
    assert "Regressed" in comparison
