from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.lora_setup import apply_lora
from validate import load_model


def merge_lora_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    rank: int = 8,
    alpha: int = 16,
    device: str = "cpu",
) -> tuple:
    """Load base model, apply LoRA structure, load checkpoint, merge and unload.

    Returns (model, preprocess, tokenizer) with merged weights.
    """
    model, preprocess, tokenizer = load_model(model_key, device)
    model = apply_lora(model, rank=rank, alpha=alpha)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.merge_and_unload()
    model.eval()
    return model, preprocess, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint and export ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-key", default="mobileclip2_s2")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    print(f"Merging LoRA from {args.checkpoint}...")
    model, preprocess, tokenizer = merge_lora_checkpoint(
        args.model_key, args.checkpoint, args.rank, args.alpha,
    )

    out_dir = args.out_dir or f"exported_onnx_{args.model_key}_selftrained"
    print(f"Exporting ONNX to {out_dir}...")

    import open_clip
    from export_onnx import OpenClipImageEncoder, OpenClipTextEncoder, export_module
    from lpcvc_models import MODELS, build_pipeline_spec

    spec = MODELS[args.model_key]
    pipeline = build_pipeline_spec(args.model_key)
    preprocess_cfg = getattr(model.visual, "preprocess_cfg", {}) or {}

    os.makedirs(out_dir, exist_ok=True)

    image_encoder = OpenClipImageEncoder(
        model,
        native_size=spec.native_size,
        mean=tuple(preprocess_cfg.get("mean", (0.0, 0.0, 0.0))),
        std=tuple(preprocess_cfg.get("std", (1.0, 1.0, 1.0))),
        interpolation=str(preprocess_cfg.get("interpolation", "bilinear")),
    ).eval()
    text_encoder = OpenClipTextEncoder(model).eval()

    dummy_image = torch.rand(1, 3, pipeline.image_shape[2], pipeline.image_shape[3])
    dummy_text = open_clip.get_tokenizer(spec.open_clip_name)(["a photo"]).to(torch.int64)

    export_module(image_encoder, dummy_image,
                  os.path.join(out_dir, "image_encoder.onnx"),
                  "image", "embedding")
    export_module(text_encoder, dummy_text,
                  os.path.join(out_dir, "text_encoder.onnx"),
                  "text", "text_embedding")

    print(f"Done. ONNX exported to {out_dir}/")


if __name__ == "__main__":
    main()
