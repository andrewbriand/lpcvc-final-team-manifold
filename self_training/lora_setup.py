from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model


VISUAL_STAGE3_TARGETS = [
    f"visual.trunk.stages.3.blocks.{i}.token_mixer.{layer}"
    for i in range(4)
    for layer in ("qkv", "proj")
]


def apply_lora(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Apply LoRA to visual stage-3 attention, freeze everything else.

    Also unfreezes: visual.head (projection), logit_scale.
    """
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=VISUAL_STAGE3_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Unfreeze projection head
    for name, param in model.named_parameters():
        if "visual.head" in name and "lora_" not in name:
            param.requires_grad = True
        if "logit_scale" in name:
            param.requires_grad = True

    return model


def get_trainable_param_names(model: torch.nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]
