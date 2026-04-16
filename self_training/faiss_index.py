from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from L2-normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def query_index(
    index: faiss.IndexFlatIP,
    queries: np.ndarray,
    k: int = 17,
    exclude_source_indices: list[int] | None = None,
    caption_embs: np.ndarray | None = None,
    own_caption_embs: np.ndarray | None = None,
    own_cosine_threshold: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Query the index, optionally excluding each query's own source caption.

    Args:
        index: FAISS index to search.
        queries: (N, D) L2-normalized query embeddings.
        k: number of results to retrieve.
        exclude_source_indices: for each query, the index of its own caption in the bank.
        caption_embs: full caption embedding matrix (needed for cosine check).
        own_caption_embs: (N, D) embeddings of each query's own caption.
        own_cosine_threshold: cosine above which a result is considered "own caption".

    Returns:
        scores: (N, K') cosine similarities
        indices: (N, K') caption indices
    """
    scores, indices = index.search(queries.astype(np.float32), k)

    if exclude_source_indices is None:
        return scores, indices

    n = queries.shape[0]
    filtered_scores = []
    filtered_indices = []

    for i in range(n):
        row_scores = []
        row_indices = []
        for j in range(k):
            idx = indices[i, j]
            if idx == -1:
                continue
            if idx == exclude_source_indices[i]:
                continue
            if own_caption_embs is not None and caption_embs is not None:
                cos = float(np.dot(caption_embs[idx], own_caption_embs[i]))
                if cos > own_cosine_threshold:
                    continue
            row_scores.append(scores[i, j])
            row_indices.append(idx)
        filtered_scores.append(row_scores[:k - 1])
        filtered_indices.append(row_indices[:k - 1])

    max_len = max(len(r) for r in filtered_indices) if filtered_indices else 0
    out_scores = np.full((n, max_len), -1.0, dtype=np.float32)
    out_indices = np.full((n, max_len), -1, dtype=np.int64)
    for i in range(n):
        length = len(filtered_indices[i])
        out_scores[i, :length] = filtered_scores[i]
        out_indices[i, :length] = filtered_indices[i]

    return out_scores, out_indices


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))
