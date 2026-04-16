import torch
from self_training.loss import self_train_infonce


def test_infonce_basic():
    torch.manual_seed(42)
    B, D, N_neg = 8, 512, 4
    image_embs = torch.randn(B, D)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    original_embs = torch.randn(B, D)
    original_embs = original_embs / original_embs.norm(dim=-1, keepdim=True)
    pseudo_embs = torch.randn(B, D)
    pseudo_embs = pseudo_embs / pseudo_embs.norm(dim=-1, keepdim=True)
    neg_embs = torch.randn(B, N_neg, D)
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

    loss = self_train_infonce(image_embs, original_embs, pseudo_embs, neg_embs, logit_scale=20.0)
    assert loss.dim() == 0
    assert loss.item() > 0
    assert loss.requires_grad


def test_infonce_gradient_flows():
    torch.manual_seed(42)
    B, D, N = 4, 64, 3
    image_embs = torch.randn(B, D, requires_grad=True)
    original_embs = torch.randn(B, D)
    pseudo_embs = torch.randn(B, D)
    neg_embs = torch.randn(B, N, D)
    loss = self_train_infonce(image_embs, original_embs, pseudo_embs, neg_embs, logit_scale=20.0)
    loss.backward()
    assert image_embs.grad is not None
    assert image_embs.grad.abs().sum() > 0
