import torch
from self_training.lora_setup import apply_lora, get_trainable_param_names


def test_apply_lora_creates_lora_layers():
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP2-S2", pretrained="dfndr2b"
    )
    lora_model = apply_lora(model, rank=8, alpha=16)
    trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in lora_model.named_parameters() if not p.requires_grad]

    # Visual stage-3 qkv and proj should have LoRA adapters
    lora_targets_found = [n for n in trainable if "lora_" in n]
    assert len(lora_targets_found) > 0, "No LoRA parameters found"

    # Expect LoRA on 8 layers: 4 blocks × (qkv + proj)
    qkv_lora = [n for n in lora_targets_found if "qkv" in n]
    proj_lora = [n for n in lora_targets_found if "token_mixer.proj" in n and "qkv" not in n]
    assert len(qkv_lora) == 8, f"Expected 8 qkv LoRA params, got {len(qkv_lora)}"
    assert len(proj_lora) == 8, f"Expected 8 proj LoRA params, got {len(proj_lora)}"

    # Text encoder should be fully frozen
    text_trainable = [n for n in trainable if n.startswith("base_model.model.text.")]
    assert len(text_trainable) == 0, f"Text encoder should be frozen, got: {text_trainable[:5]}"

    # logit_scale should be trainable
    assert any("logit_scale" in n for n in trainable)


def test_get_trainable_param_names():
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP2-S2", pretrained="dfndr2b"
    )
    lora_model = apply_lora(model, rank=8, alpha=16)
    names = get_trainable_param_names(lora_model)
    assert isinstance(names, list)
    assert len(names) > 0
