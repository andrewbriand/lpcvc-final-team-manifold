import tempfile
from pathlib import Path

import numpy as np

from self_training.faiss_index import build_index, query_index, save_index, load_index


def _make_embeddings(n: int = 100, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(42)
    embs = rng.randn(n, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def test_build_and_query():
    embs = _make_embeddings(100)
    index = build_index(embs)
    query = embs[:3]
    scores, indices = query_index(index, query, k=5)
    assert scores.shape == (3, 5)
    assert indices.shape == (3, 5)
    np.testing.assert_array_equal(indices[:, 0], [0, 1, 2])


def test_own_caption_exclusion():
    embs = _make_embeddings(100)
    index = build_index(embs)
    query = embs[:3]
    source_indices = [0, 1, 2]
    own_caption_embs = embs[:3]
    scores, indices = query_index(
        index, query, k=5,
        exclude_source_indices=source_indices,
        caption_embs=embs,
        own_caption_embs=own_caption_embs,
        own_cosine_threshold=0.99,
    )
    for i in range(3):
        assert indices[i, 0] != source_indices[i]


def test_save_load_index():
    embs = _make_embeddings(50)
    index = build_index(embs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.index"
        save_index(index, path)
        loaded = load_index(path)
        _, idx_orig = query_index(index, embs[:2], k=3)
        _, idx_loaded = query_index(loaded, embs[:2], k=3)
        np.testing.assert_array_equal(idx_orig, idx_loaded)
