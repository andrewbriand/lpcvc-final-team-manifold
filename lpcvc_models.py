"""
Shared MobileCLIP model registry.

Keep model choices in one place so export and validation stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    open_clip_name: str
    pretrained: str
    native_size: int
    notes: str


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
}

DEFAULT_MODEL = "mobileclip_s2"
