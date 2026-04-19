"""Export FG-CLIP2 to ONNX under the LPCVC Track 1 input contract.

Input contract (COMPETITION.md §4.5):
  image: (1, 3, 224, 224) float32 in [0, 1]
  text:  (1, 77) int64 (OpenAI CLIP BPE tokens — NOT FG-CLIP2's native Gemma tokens)

FG-CLIP2 internals:
  - Dynamic patching: HF processor produces (1, N, 768) patch tiles plus
    a (1, N) attention mask and (1, 2) spatial_shapes metadata. We pin
    N=256 (16×16 grid) by resizing 224→256 bilinearly in-graph, matching
    the processor's default behavior for a 224px input at bucket-256.
  - Positional embeddings are stored at 16×16 and dynamically resized via
    F.interpolate(antialias=True). That aa op isn't ONNX-exportable at
    opset 18, so we monkey-patch the embedding forward to skip the
    (identity-at-16×16) resize.
  - Text embedding layer has broken buffers in the released checkpoint;
    _patch_fgclip2_text_embeddings rebuilds them before tracing.

The 77-token int32/int64 text input is fed straight to FG-CLIP2's text
encoder despite vocab/tokenizer mismatch — evaluating what accuracy
survives under the LPCVC contract is the whole point of running this
through the sample set.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from validate import _patch_fgclip2_text_embeddings  # noqa: E402


IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
NATIVE_SIZE = 256
PATCH_SIZE = 16
GRID = NATIVE_SIZE // PATCH_SIZE  # 16
NUM_PATCHES = GRID * GRID  # 256
HIDDEN = PATCH_SIZE * PATCH_SIZE * 3  # 768
CONTRACT_IMAGE_SIZE = 224
CONTRACT_TEXT_LEN = 77


def _patch_vision_embeddings_identity(model: nn.Module) -> None:
    """Replace VisionEmbeddings.forward to skip the aa-interpolate resize.

    Only correct when ``spatial_shapes == position_embedding_size`` (16).
    """
    embeddings = model.vision_model.embeddings
    assert embeddings.position_embedding_size == GRID, (
        f"Expected position_embedding_size == {GRID}, got "
        f"{embeddings.position_embedding_size}; identity-resize shortcut invalid."
    )

    def fixed_forward(self, pixel_values, spatial_shapes=None):
        dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=dtype))
        pos = self.position_embedding.weight.unsqueeze(0)  # (1, 256, 768)
        return patch_embeds + pos

    embeddings.forward = types.MethodType(fixed_forward, embeddings)


class FgClip2ImageEncoder(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            images, size=(NATIVE_SIZE, NATIVE_SIZE),
            mode="bilinear", align_corners=False,
        )
        x = (x - self.mean) / self.std
        x = x.permute(0, 2, 3, 1)
        x = x.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = x.reshape(1, NUM_PATCHES, HIDDEN)
        mask = torch.ones(1, NUM_PATCHES, dtype=torch.int32, device=x.device)
        ss = torch.tensor([[GRID, GRID]], dtype=torch.long, device=x.device)
        feats = self.core.get_image_features(x, mask, ss)
        return feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)


FGCLIP2_SHORT_LEN = 64  # matches config.text_config.max_position_embeddings


class FgClip2TextEncoder(nn.Module):
    """Truncates contract (1, 77) int64 to (1, 64) and uses FG-CLIP2 short mode.

    FG-CLIP2's text encoder supports exactly two lengths: 64 (short mode,
    via position_embedding[64, 768]) or 196 (long mode, hard-coded via
    mask1/mask2 buffers). The LPCVC contract gives us 77 tokens — we
    drop the tail 13 because short captions always fit in 64 anyway.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(torch.int64)[:, :FGCLIP2_SHORT_LEN]
        feats = self.core.get_text_features(input_ids, walk_type="short")
        return feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def export_module(
    module: nn.Module,
    dummy: torch.Tensor,
    out_path: str,
    input_name: str,
    output_name: str,
) -> None:
    torch.onnx.export(
        module, dummy, out_path,
        input_names=[input_name], output_names=[output_name],
        opset_version=18, do_constant_folding=True,
        export_params=True, training=torch.onnx.TrainingMode.EVAL,
        dynamo=False, verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="fgclip2_base",
                        choices=["fgclip2_base", "fgclip2_large", "fgclip2_so400m"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-text", action="store_true",
                        help="Export image encoder only (text wrapper still WIP)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from lpcvc_models import MODELS

    spec = MODELS[args.model_key]
    print(f"Loading {spec.hf_repo} ...")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo, trust_remote_code=True,
    ).to("cpu").to(torch.float32).eval()
    _patch_fgclip2_text_embeddings(model)
    _patch_vision_embeddings_identity(model)

    out_dir = args.out_dir or f"exported_onnx_{args.model_key}"
    os.makedirs(out_dir, exist_ok=True)

    # Image encoder
    image_encoder = FgClip2ImageEncoder(model).eval()
    dummy_image = torch.rand(1, 3, CONTRACT_IMAGE_SIZE, CONTRACT_IMAGE_SIZE)
    image_onnx = os.path.join(out_dir, "image_encoder.onnx")
    print(f"Exporting image encoder → {image_onnx}")
    export_module(image_encoder, dummy_image, image_onnx, "image", "embedding")
    print(f"  size: {os.path.getsize(image_onnx) / 1e6:.1f} MB")

    if args.skip_text:
        print("Skipping text encoder (--skip-text).")
        return

    text_encoder = FgClip2TextEncoder(model).eval()
    dummy_text = torch.zeros(1, CONTRACT_TEXT_LEN, dtype=torch.int64)
    dummy_text[0, 0] = 49406  # CLIP BOS
    dummy_text[0, 1] = 49407  # CLIP EOS
    text_onnx = os.path.join(out_dir, "text_encoder.onnx")
    print(f"Exporting text encoder → {text_onnx}")
    export_module(text_encoder, dummy_text, text_onnx, "text", "text_embedding")
    print(f"  size: {os.path.getsize(text_onnx) / 1e6:.1f} MB")

    print(f"\nDone. ONNX written to {out_dir}/")


if __name__ == "__main__":
    main()
