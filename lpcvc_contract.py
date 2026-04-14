"""
Shared LPCVC Track 1 input/output contract.

This module keeps one canonical contract so local validation, QAI uploads,
compile specs, and inference manifests stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "lpcvc-track1-contract-v1"
TOKENIZER_ID = "openai/clip-vit-base-patch32"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_DTYPE = "float32"

TEXT_SEQ_LEN = 77
# Competition/runtime boundary: text input is int32 (1x77).
TEXT_DTYPE = "int32"
# ONNX export / compile-time model input is int64.
ONNX_TEXT_DTYPE = "int64"

# Target device as specified in COMPETITION.md
QAI_DEVICE = "XR2 Gen 2 (Proxy)"
COMPILE_OPTIONS = "--target_runtime qnn_context_binary --truncate_64bit_io"


@dataclass(frozen=True)
class ContractSpec:
    version: str = CONTRACT_VERSION
    image_shape: tuple[int, int, int, int] = (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    image_dtype: str = IMAGE_DTYPE
    text_shape: tuple[int, int] = (1, TEXT_SEQ_LEN)
    text_dtype: str = TEXT_DTYPE
    tokenizer_id: str = TOKENIZER_ID


def get_contract_spec() -> ContractSpec:
    return ContractSpec()


def compile_input_specs() -> dict:
    spec = get_contract_spec()
    return {
        "image": spec.image_shape,
        "text": (spec.text_shape, ONNX_TEXT_DTYPE),
    }
