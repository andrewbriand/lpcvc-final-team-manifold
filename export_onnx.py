"""
Export MobileCLIP encoders to ONNX with LPCVC contract inputs.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from lpcvc_contract import CONTRACT_VERSION, IMAGE_HEIGHT, IMAGE_WIDTH, TEXT_SEQ_LEN
from lpcvc_models import DEFAULT_MODEL, MODELS


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model, native_size: int):
        super().__init__()
        self.visual = clip_model.visual
        self.native_size = native_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.native_size, self.native_size):
            images = F.interpolate(
                images,
                size=(self.native_size, self.native_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.visual(images)
        return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        if not hasattr(clip_model, "text"):
            raise RuntimeError("Expected a MobileCLIP-style model with a .text encoder.")
        self.text_encoder = clip_model.text

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(torch.int64)
        eos_id = 49407
        positions = torch.arange(token_ids.shape[1], device=token_ids.device, dtype=token_ids.dtype)
        is_eos = token_ids == eos_id
        first_eos = is_eos.long().argmax(dim=-1, keepdim=True)
        keep_mask = positions.unsqueeze(0) <= first_eos
        token_ids = token_ids * keep_mask
        features = self.text_encoder(token_ids)
        return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def export_module(
    module: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_name: str,
    output_name: str,
) -> None:
    torch.onnx.export(
        module,
        dummy_input,
        output_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileCLIP encoders to ONNX.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    parser.add_argument("--out-dir", default=None, help="Default: exported_onnx_<model>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]
    out_dir = args.out_dir or f"exported_onnx_{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        spec.open_clip_name,
        pretrained=spec.pretrained,
    )
    model = model.to("cpu").to(torch.float32).eval()

    image_encoder = ImageEncoder(model, native_size=spec.native_size).eval()
    text_encoder = TextEncoder(model).eval()

    dummy_image = torch.rand(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)
    dummy_text = torch.randint(0, 49408, (1, TEXT_SEQ_LEN), dtype=torch.int64)

    with torch.no_grad():
        image_embed = image_encoder(dummy_image)
        text_embed = text_encoder(dummy_text)
    print(f"Smoke image embedding shape: {tuple(image_embed.shape)}")
    print(f"Smoke text embedding shape:  {tuple(text_embed.shape)}")

    image_onnx = os.path.join(out_dir, "image_encoder.onnx")
    text_onnx = os.path.join(out_dir, "text_encoder.onnx")

    export_module(image_encoder, dummy_image, image_onnx, input_name="image", output_name="embedding")
    export_module(text_encoder, dummy_text, text_onnx, input_name="text", output_name="text_embedding")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_key": args.model,
        "open_clip_name": spec.open_clip_name,
        "pretrained": spec.pretrained,
        "native_size": spec.native_size,
        "image_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "text_shape": [1, TEXT_SEQ_LEN],
        "image_onnx": os.path.abspath(image_onnx),
        "text_onnx": os.path.abspath(text_onnx),
    }
    manifest_path = os.path.join(out_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote image ONNX: {image_onnx}")
    print(f"Wrote text ONNX:  {text_onnx}")
    print(f"Wrote manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
