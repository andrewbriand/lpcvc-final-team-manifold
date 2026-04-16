from __future__ import annotations

import numpy as np


def calibrate_thresholds(
    top1_scores: np.ndarray,
    top2_scores: np.ndarray,
    top1_percentile: int = 50,
    margin_percentile: int = 40,
) -> dict[str, float]:
    margins = top1_scores - top2_scores
    return {
        "min_top1_cosine": float(np.percentile(top1_scores, top1_percentile)),
        "min_margin": float(np.percentile(margins, margin_percentile)),
    }


def apply_confidence_filter(
    top1_scores: np.ndarray,
    top2_scores: np.ndarray,
    min_top1_cosine: float,
    min_margin: float,
) -> np.ndarray:
    margins = top1_scores - top2_scores
    return (top1_scores >= min_top1_cosine) & (margins >= min_margin)
