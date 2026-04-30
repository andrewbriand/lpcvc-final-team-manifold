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


def kl_anchor_loss(
    student_embs: torch.Tensor,
    teacher_embs: torch.Tensor,
) -> torch.Tensor:
    """KL-divergence anchor loss preventing drift from frozen baseline.

    Penalizes the LoRA'd visual encoder for producing embeddings that
    diverge from what the frozen base model would produce on the same
    images. Uses MSE on L2-normalized embeddings, which is equivalent
    to (2 - 2*cosine_similarity) and thus a smooth proxy for KL on
    the embedding distribution.

    Reference: BYOL (Grill et al. 2020), DINO (Caron et al. 2021)
    use similar representation-level consistency losses to prevent
    collapse in self-supervised training.

    Args:
        student_embs: (B, D) L2-normalized embeddings from LoRA'd model
        teacher_embs: (B, D) L2-normalized embeddings from frozen base

    Returns:
        scalar loss (0 = identical, 2 = orthogonal, 4 = opposite)
    """
    return F.mse_loss(student_embs, teacher_embs)


def combined_loss(
    student_image_embs: torch.Tensor,
    teacher_image_embs: torch.Tensor,
    original_text_embs: torch.Tensor,
    pseudo_text_embs: torch.Tensor,
    hard_neg_embs: torch.Tensor,
    logit_scale: float | torch.Tensor = 20.0,
    anchor_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined InfoNCE + KL anchor loss.

    Returns (total_loss, infonce_loss, anchor_loss) for logging.
    """
    infonce = self_train_infonce(
        student_image_embs, original_text_embs, pseudo_text_embs,
        hard_neg_embs, logit_scale,
    )
    anchor = kl_anchor_loss(student_image_embs, teacher_image_embs)
    total = infonce + anchor_weight * anchor
    return total, infonce, anchor
