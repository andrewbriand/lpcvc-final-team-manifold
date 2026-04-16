import numpy as np
from self_training.negative_filter import filter_hard_negatives


def test_exact_dedup():
    positive = "a cat sitting on a mat"
    candidates = ["a cat sitting on a mat", "a dog in a park", "a bird flying"]
    candidate_embs = np.random.randn(3, 512).astype(np.float32)
    positive_emb = np.random.randn(512).astype(np.float32)
    result = filter_hard_negatives(positive, positive_emb, candidates, candidate_embs)
    assert "a cat sitting on a mat" not in [r[0] for r in result]


def test_jaccard_near_dedup():
    positive = "a cute cat sitting on a red mat"
    candidates = [
        "a cute cat sitting on a blue mat",
        "a spaceship launching into orbit",
    ]
    candidate_embs = np.random.randn(2, 512).astype(np.float32)
    candidate_embs /= np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    positive_emb = np.random.randn(512).astype(np.float32)
    positive_emb /= np.linalg.norm(positive_emb)
    result = filter_hard_negatives(
        positive, positive_emb, candidates, candidate_embs, jaccard_threshold=0.7,
    )
    captions = [r[0] for r in result]
    assert "a spaceship launching into orbit" in captions


def test_cosine_proximity_filter():
    positive = "a dog"
    positive_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = ["near caption", "far caption"]
    candidate_embs = np.array([
        [0.99, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    candidate_embs /= np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    result = filter_hard_negatives(
        positive, positive_emb, candidates, candidate_embs, neg_cosine_proximity=0.85,
    )
    captions = [r[0] for r in result]
    assert "far caption" in captions
    assert "near caption" not in captions


def test_min_negatives_fallback():
    positive = "a cat"
    positive_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = ["close1", "close2"]
    candidate_embs = np.array([
        [0.99, 0.1, 0.0],
        [0.98, 0.15, 0.0],
    ], dtype=np.float32)
    candidate_embs /= np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    result = filter_hard_negatives(
        positive, positive_emb, candidates, candidate_embs,
        neg_cosine_proximity=0.85, min_negatives=4,
    )
    assert len(result) < 4
