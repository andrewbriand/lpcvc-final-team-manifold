from __future__ import annotations

import torch
import torch.nn.functional as F


def self_train_infonce(
    image_embs: torch.Tensor,
    original_text_embs: torch.Tensor,
    pseudo_text_embs: torch.Tensor,
    hard_neg_embs: torch.Tensor,
    logit_scale: float | torch.Tensor = 20.0,
) -> torch.Tensor:
    """InfoNCE loss with two positives (original + pseudo) and hard negatives.

    Args:
        image_embs: (B, D) L2-normalized image embeddings
        original_text_embs: (B, D) original CC12M caption embeddings (stabilizer)
        pseudo_text_embs: (B, D) pseudo-positive caption embeddings
        hard_neg_embs: (B, N, D) hard-negative caption embeddings
        logit_scale: scalar temperature (or learned parameter)

    Returns:
        scalar loss
    """
    B, D = image_embs.shape

    if not isinstance(logit_scale, torch.Tensor):
        logit_scale = torch.tensor(logit_scale, dtype=image_embs.dtype, requires_grad=True)

    logits_orig = (image_embs * original_text_embs).sum(dim=-1) * logit_scale
    logits_pseudo = (image_embs * pseudo_text_embs).sum(dim=-1) * logit_scale
    logits_neg = torch.einsum("bd,bnd->bn", image_embs, hard_neg_embs) * logit_scale

    pos_logits = torch.stack([logits_orig, logits_pseudo], dim=-1)  # (B, 2)
    all_logits = torch.cat([pos_logits, logits_neg], dim=-1)  # (B, 2+N)

    log_sum_exp_all = torch.logsumexp(all_logits, dim=-1)
    log_sum_exp_pos = torch.logsumexp(pos_logits, dim=-1)

    loss = (log_sum_exp_all - log_sum_exp_pos).mean()
    return loss
