"""
Shared MobileCLIP model registry.

Keep model choices in one place so export and validation stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

from lpcvc_contract import (
    CONTRACT_VERSION,
    IMAGE_CHANNELS,
    IMAGE_DTYPE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ONNX_TEXT_DTYPE,
    TEXT_DTYPE,
    TEXT_SEQ_LEN,
    TOKENIZER_ID,
)


@dataclass(frozen=True)
class ModelSpec:
    open_clip_name: str | None
    pretrained: str | None
    native_size: int
    notes: str
    loader: str = "open_clip"
    hf_repo: str | None = None
    hf_base_model: str | None = None
    text_max_length: int = 77


@dataclass(frozen=True)
class PipelineSpec:
    version: str
    model_key: str
    submission_contract_version: str | None
    image_shape: tuple[int, int, int, int]
    image_dtype: str
    text_shape: tuple[int, int]
    upload_text_dtype: str
    compile_text_dtype: str
    tokenizer_id: str
    native_size: int
    loader: str
    notes: str


PIPELINE_VERSION = "qai-eval-pipeline-v2"


MODELS: dict[str, ModelSpec] = {
    "mobileclip_s1": ModelSpec("MobileCLIP-S1", "datacompdr", 256, "v1 ~30M, fastest"),
    "mobileclip_s2": ModelSpec("MobileCLIP-S2", "datacompdr", 256, "v1 ~50M"),
    "mobileclip_b": ModelSpec("MobileCLIP-B", "datacompdr", 224, "v1 ~90M"),
    "mobileclip2_s0": ModelSpec("MobileCLIP2-S0", "dfndr2b", 256, "v2 ~45M, fastest"),
    "mobileclip2_s2": ModelSpec("MobileCLIP2-S2", "dfndr2b", 256, "v2 ~55M"),
    "mobileclip2_s3": ModelSpec("MobileCLIP2-S3", "dfndr2b", 256, "v2 ~70M"),
    "mobileclip2_s4": ModelSpec("MobileCLIP2-S4", "dfndr2b", 256, "v2 ~90M, best accuracy"),
    "mobileclip2_b": ModelSpec("MobileCLIP2-B", "dfndr2b", 256, "v2 ~110M"),
    "mobileclip2_l14": ModelSpec("MobileCLIP2-L-14", "dfndr2b", 256, "v2 ~160M, largest"),
    "vit_b16": ModelSpec("ViT-B-16", "openai", 224, "OpenAI CLIP ViT-B/16, ~150M"),
    "vit_l14": ModelSpec("ViT-L-14", "openai", 224, "OpenAI CLIP ViT-L/14, ~428M"),
    "tripletclip_cc12m": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="TripletCLIP ViT-B/32 trained on CC12M",
        loader="tripletclip_hf",
        hf_repo="TripletCLIP/CC12M_TripletCLIP_ViTB12",
        hf_base_model="openai/clip-vit-base-patch32",
        text_max_length=77,
    ),
    "siglip2_base_224": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="Google SigLIP2 base patch16 224",
        loader="siglip_hf",
        hf_repo="google/siglip2-base-patch16-224",
        text_max_length=64,
    ),
    "siglip2_base_256": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=256,
        notes="Google SigLIP2 base patch16 256",
        loader="siglip_hf",
        hf_repo="google/siglip2-base-patch16-256",
        text_max_length=64,
    ),
    "siglip2_base_384": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=384,
        notes="Google SigLIP2 base patch16 384",
        loader="siglip_hf",
        hf_repo="google/siglip2-base-patch16-384",
        text_max_length=64,
    ),
    "siglip2_base_512": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=512,
        notes="Google SigLIP2 base patch16 512",
        loader="siglip_hf",
        hf_repo="google/siglip2-base-patch16-512",
        text_max_length=64,
    ),
    "siglip2_base_patch32_256": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=256,
        notes="Google SigLIP2 base patch32 256",
        loader="siglip_hf",
        hf_repo="google/siglip2-base-patch32-256",
        text_max_length=64,
    ),
    "siglip2_large_256": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=256,
        notes="Google SigLIP2 large patch16 256",
        loader="siglip_hf",
        hf_repo="google/siglip2-large-patch16-256",
        text_max_length=64,
    ),
    "siglip2_large_384": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=384,
        notes="Google SigLIP2 large patch16 384",
        loader="siglip_hf",
        hf_repo="google/siglip2-large-patch16-384",
        text_max_length=64,
    ),
    "siglip2_large_512": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=512,
        notes="Google SigLIP2 large patch16 512",
        loader="siglip_hf",
        hf_repo="google/siglip2-large-patch16-512",
        text_max_length=64,
    ),
    "siglip2_so400m_224": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="Google SigLIP2 so400m patch14 224",
        loader="siglip_hf",
        hf_repo="google/siglip2-so400m-patch14-224",
        text_max_length=64,
    ),
    "siglip2_so400m_256": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=256,
        notes="Google SigLIP2 so400m patch16 256",
        loader="siglip_hf",
        hf_repo="google/siglip2-so400m-patch16-256",
        text_max_length=64,
    ),
    "siglip2_so400m_384": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=384,
        notes="Google SigLIP2 so400m patch16 384",
        loader="siglip_hf",
        hf_repo="google/siglip2-so400m-patch16-384",
        text_max_length=64,
    ),
    "siglip2_so400m_512": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=512,
        notes="Google SigLIP2 so400m patch16 512",
        loader="siglip_hf",
        hf_repo="google/siglip2-so400m-patch16-512",
        text_max_length=64,
    ),
    "siglip2_giant_256": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=256,
        notes="Google SigLIP2 giant 1B patch16 256",
        loader="siglip_hf",
        hf_repo="google/siglip2-giant-opt-patch16-256",
        text_max_length=64,
    ),
    "siglip2_giant_384": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=384,
        notes="Google SigLIP2 giant 1B patch16 384",
        loader="siglip_hf",
        hf_repo="google/siglip2-giant-opt-patch16-384",
        text_max_length=64,
    ),
    "fgclip2_base": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="Qihoo360 FG-CLIP2 base (~0.4B), bilingual fine-grained",
        loader="fgclip2_hf",
        hf_repo="qihoo360/fg-clip2-base",
        text_max_length=196,
    ),
    "fgclip2_large": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="Qihoo360 FG-CLIP2 large (~0.9B)",
        loader="fgclip2_hf",
        hf_repo="qihoo360/fg-clip2-large",
        text_max_length=196,
    ),
    "fgclip2_so400m": ModelSpec(
        open_clip_name=None,
        pretrained=None,
        native_size=224,
        notes="Qihoo360 FG-CLIP2 So400M-scale (~1B)",
        loader="fgclip2_hf",
        hf_repo="qihoo360/fg-clip2-so400m",
        text_max_length=196,
    ),
}

DEFAULT_MODEL = "mobileclip2_s2"


def resolve_tokenizer_id(model_key: str) -> str:
    spec = MODELS[model_key]
    if spec.loader == "siglip_hf":
        if not spec.hf_repo:
            raise RuntimeError(f"SigLIP model spec is missing hf_repo: {model_key}")
        return spec.hf_repo
    if spec.loader == "tripletclip_hf":
        if not spec.hf_base_model:
            raise RuntimeError(f"TripletCLIP model spec is missing hf_base_model: {model_key}")
        return spec.hf_base_model
    if spec.loader == "fgclip2_hf":
        if not spec.hf_repo:
            raise RuntimeError(f"FG-CLIP2 model spec is missing hf_repo: {model_key}")
        return spec.hf_repo
    return TOKENIZER_ID


def build_pipeline_spec(model_key: str) -> PipelineSpec:
    spec = MODELS[model_key]
    tokenizer_id = resolve_tokenizer_id(model_key)
    submission_contract_version = None
    if tokenizer_id == TOKENIZER_ID and spec.text_max_length == TEXT_SEQ_LEN:
        submission_contract_version = CONTRACT_VERSION
    return PipelineSpec(
        version=PIPELINE_VERSION,
        model_key=model_key,
        submission_contract_version=submission_contract_version,
        image_shape=(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        image_dtype=IMAGE_DTYPE,
        text_shape=(1, spec.text_max_length),
        upload_text_dtype=TEXT_DTYPE,
        compile_text_dtype=ONNX_TEXT_DTYPE,
        tokenizer_id=tokenizer_id,
        native_size=spec.native_size,
        loader=spec.loader,
        notes=spec.notes,
    )


def pipeline_spec_to_dict(spec: PipelineSpec) -> dict:
    return {
        "version": spec.version,
        "model_key": spec.model_key,
        "submission_contract_version": spec.submission_contract_version,
        "image_shape": list(spec.image_shape),
        "image_dtype": spec.image_dtype,
        "text_shape": list(spec.text_shape),
        "upload_text_dtype": spec.upload_text_dtype,
        "compile_text_dtype": spec.compile_text_dtype,
        "tokenizer_id": spec.tokenizer_id,
        "native_size": spec.native_size,
        "loader": spec.loader,
        "notes": spec.notes,
    }
