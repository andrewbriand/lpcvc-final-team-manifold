"""
Shared retrieval evaluation helpers for the LPCVC harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_embedding_matrix(arr) -> np.ndarray:
    matrix = np.asarray(arr)
    if matrix.dtype == object:
        matrix = np.vstack([np.asarray(x) for x in matrix.tolist()])
    if matrix.ndim == 3 and matrix.shape[1] == 1:
        matrix = matrix[:, 0, :]
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0, :, :]
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.ndim != 2:
        matrix = np.squeeze(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape={matrix.shape}")
    return matrix.astype(np.float32)


def stack_embeddings(raw_output) -> np.ndarray:
    if isinstance(raw_output, list):
        return np.vstack([to_embedding_matrix(x) for x in raw_output])
    return to_embedding_matrix(raw_output)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def similarity_matrix(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    return l2_normalize(img_emb) @ l2_normalize(txt_emb).T


def load_track1_ground_truth(txt_csv: str, img_csv: str) -> tuple[list[int], list[str]]:
    df_img = pd.read_csv(img_csv)
    df_txt = pd.read_csv(txt_csv)
    txt_ids = df_txt.iloc[:, 0].dropna().astype(int).tolist()
    gt_rows = df_img.iloc[:, 1].dropna().astype(str).tolist()
    return txt_ids, gt_rows


def compute_fractional_recall_at_k(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    txt_ids: list[int],
    gt_rows: list[str],
    k: int,
) -> tuple[float, dict[str, int]]:
    sim = similarity_matrix(img_emb, txt_emb)
    txt_id_set = set(txt_ids)

    recalls: list[float] = []
    skipped_missing_gt = 0
    n = min(sim.shape[0], len(gt_rows))
    for i in range(n):
        raw_gt = [int(x) for x in gt_rows[i].split(";") if x.strip().isdigit()]
        valid_gt = [g for g in raw_gt if g in txt_id_set]
        if not valid_gt:
            skipped_missing_gt += 1
            continue

        top_k = np.argsort(-sim[i])[:k]
        pred_ids = [txt_ids[idx] for idx in top_k]
        matched = len(set(pred_ids) & set(valid_gt))
        recalls.append(matched / len(valid_gt))

    if not recalls:
        raise RuntimeError("No valid rows to evaluate after filtering missing GT IDs.")

    diagnostics = {
        "scored_rows": len(recalls),
        "skipped_missing_gt_rows": skipped_missing_gt,
        "image_embeddings": int(img_emb.shape[0]),
        "text_embeddings": int(txt_emb.shape[0]),
    }
    return float(np.mean(recalls)), diagnostics


def evaluate_track1_embeddings(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    txt_csv: str,
    img_csv: str,
    k: int = 10,
) -> tuple[float, dict[str, int]]:
    txt_ids, gt_rows = load_track1_ground_truth(txt_csv, img_csv)
    return compute_fractional_recall_at_k(
        img_emb=to_embedding_matrix(img_emb),
        txt_emb=to_embedding_matrix(txt_emb),
        txt_ids=txt_ids,
        gt_rows=gt_rows,
        k=k,
    )
