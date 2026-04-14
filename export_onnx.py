"""
Export MobileCLIP encoders to ONNX with LPCVC contract inputs.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from lpcvc_models import DEFAULT_MODEL, MODELS, build_pipeline_spec, pipeline_spec_to_dict


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _resize(images: torch.Tensor, native_size: int, interpolation: str) -> torch.Tensor:
    if images.shape[-2:] == (native_size, native_size):
        return images
    kwargs = {
        "size": (native_size, native_size),
        "mode": interpolation,
    }
    if interpolation in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(images, **kwargs)


class OpenClipImageEncoder(torch.nn.Module):
    def __init__(self, clip_model, native_size: int, mean: tuple[float, ...], std: tuple[float, ...], interpolation: str):
        super().__init__()
        self.visual = clip_model.visual
        self.native_size = native_size
        self.interpolation = interpolation
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _resize(images, self.native_size, self.interpolation)
        images = (images - self.mean) / self.std
        features = self.visual(images)
        return _normalize(features)


class OpenClipTextEncoder(torch.nn.Module):
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
        return _normalize(features)


class SiglipImageEncoder(torch.nn.Module):
    def __init__(self, model, native_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.model = model
        self.native_size = native_size
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _resize(images, self.native_size, "bilinear")
        images = (images - self.mean) / self.std
        features = self.model.get_image_features(pixel_values=images)
        return _normalize(features)


class SiglipTextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(torch.int64)
        # Match the HF SigLIP2 inference path: fixed-length padded input IDs
        # without a hand-built attention mask.
        features = self.model.get_text_features(input_ids=token_ids)
        return _normalize(features)


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
    parser = argparse.ArgumentParser(description="Export retrieval encoders to ONNX.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    parser.add_argument("--out-dir", default=None, help="Default: exported_onnx_<model>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]
    pipeline = build_pipeline_spec(args.model)
    if spec.loader != "open_clip":
        if spec.loader != "siglip_hf":
            raise NotImplementedError(
                f"ONNX export is currently implemented for open_clip and siglip_hf models; got {args.model} ({spec.loader})."
            )
    out_dir = args.out_dir or f"exported_onnx_{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    if spec.loader == "open_clip":
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            spec.open_clip_name,
            pretrained=spec.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(spec.open_clip_name)
        model = model.to("cpu").to(torch.float32).eval()
        preprocess_cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
        image_encoder = OpenClipImageEncoder(
            model,
            native_size=spec.native_size,
            mean=tuple(preprocess_cfg.get("mean", (0.0, 0.0, 0.0))),
            std=tuple(preprocess_cfg.get("std", (1.0, 1.0, 1.0))),
            interpolation=str(preprocess_cfg.get("interpolation", "bilinear")),
        ).eval()
        text_encoder = OpenClipTextEncoder(model).eval()
        dummy_text = tokenizer(["a photo of a cat"])
        if not isinstance(dummy_text, torch.Tensor):
            dummy_text = torch.as_tensor(dummy_text)
        dummy_text = dummy_text.to(torch.int64)
    else:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if not spec.hf_repo:
            raise RuntimeError(f"SigLIP model spec is incomplete: {args.model}")
        model = AutoModel.from_pretrained(spec.hf_repo).to("cpu").to(torch.float32).eval()
        processor = AutoProcessor.from_pretrained(spec.hf_repo)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo)
        image_processor = getattr(processor, "image_processor", processor)
        image_size = getattr(image_processor, "size", {}) or {}
        native_size = int(image_size.get("height", spec.native_size))
        image_encoder = SiglipImageEncoder(
            model,
            native_size=native_size,
            mean=tuple(image_processor.image_mean),
            std=tuple(image_processor.image_std),
        ).eval()
        text_encoder = SiglipTextEncoder(model).eval()
        dummy_text = tokenizer(
            ["a photo of a cat"],
            padding="max_length",
            truncation=True,
            max_length=pipeline.text_shape[1],
            return_tensors="pt",
        )["input_ids"].to(torch.int64)

    dummy_image = torch.rand(1, 3, pipeline.image_shape[2], pipeline.image_shape[3], dtype=torch.float32)

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
        "contract_version": pipeline.submission_contract_version,
        "pipeline_version": pipeline.version,
        "pipeline": pipeline_spec_to_dict(pipeline),
        "model_key": args.model,
        "loader": spec.loader,
        "open_clip_name": spec.open_clip_name,
        "hf_repo": spec.hf_repo,
        "pretrained": spec.pretrained,
        "native_size": spec.native_size,
        "image_shape": list(pipeline.image_shape),
        "text_shape": list(pipeline.text_shape),
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
