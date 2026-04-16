from __future__ import annotations

import numpy as np


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def filter_hard_negatives(
    positive_caption: str,
    positive_emb: np.ndarray,
    candidate_captions: list[str],
    candidate_embs: np.ndarray,
    jaccard_threshold: float = 0.7,
    neg_cosine_proximity: float = 0.85,
    min_negatives: int = 4,
) -> list[tuple[str, np.ndarray, int]]:
    """Filter hard-negative candidates. Returns (caption, embedding, original_idx) tuples."""
    positive_tokens = set(positive_caption.lower().split())
    results = []

    for i, (caption, emb) in enumerate(zip(candidate_captions, candidate_embs)):
        if caption.strip() == positive_caption.strip():
            continue
        candidate_tokens = set(caption.lower().split())
        if _jaccard(positive_tokens, candidate_tokens) > jaccard_threshold:
            continue
        cos_sim = float(np.dot(positive_emb, emb))
        if cos_sim > neg_cosine_proximity:
            continue
        results.append((caption, emb, i))

    return results
