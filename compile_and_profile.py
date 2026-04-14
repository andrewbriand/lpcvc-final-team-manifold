"""
Compile LPCVC image/text ONNX encoders on QAI Hub.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import onnx
import qai_hub

from lpcvc_contract import COMPILE_OPTIONS, QAI_DEVICE

PROFILE_OPTIONS = "--max_profiler_iterations 100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile LPCVC ONNX encoders on QAI Hub.")
    parser.add_argument("--onnx-dir", default="exported_onnx_mobileclip2_s2")
    parser.add_argument("--device", default=QAI_DEVICE)
    parser.add_argument("--manifest-out", default="manifests/compile_manifest.json")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def load_checked_onnx(path: str, label: str):
    print(f"Loading {label} ONNX: {path}")
    onnx.checker.check_model(path)
    return onnx.load(path, load_external_data=True)


def load_export_manifest(onnx_dir: str) -> dict:
    manifest_path = os.path.join(onnx_dir, "export_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_compile_specs(export_manifest: dict) -> tuple[dict, dict]:
    pipeline = export_manifest.get("pipeline")
    if not pipeline:
        raise ValueError("Export manifest is missing the pipeline specification.")
    image_shape = tuple(pipeline["image_shape"])
    text_shape = tuple(pipeline["text_shape"])
    text_dtype = str(pipeline["compile_text_dtype"])
    return (
        {"image": image_shape, "text": (text_shape, text_dtype)},
        pipeline,
    )


def main() -> None:
    args = parse_args()

    image_onnx_path = os.path.join(args.onnx_dir, "image_encoder.onnx")
    text_onnx_path = os.path.join(args.onnx_dir, "text_encoder.onnx")
    if not os.path.exists(image_onnx_path):
        raise FileNotFoundError(f"Missing image ONNX: {image_onnx_path}")
    if not os.path.exists(text_onnx_path):
        raise FileNotFoundError(f"Missing text ONNX: {text_onnx_path}")

    image_model = load_checked_onnx(image_onnx_path, "image")
    text_model = load_checked_onnx(text_onnx_path, "text")
    export_manifest = load_export_manifest(args.onnx_dir)
    compile_specs, pipeline = resolve_compile_specs(export_manifest)

    device = qai_hub.Device(args.device)
    print(f"Submitting compile jobs on device: {device.name}")
    image_compile_job = qai_hub.submit_compile_job(
        model=image_model,
        device=device,
        input_specs={"image": compile_specs["image"]},
        options=COMPILE_OPTIONS,
    )
    text_compile_job = qai_hub.submit_compile_job(
        model=text_model,
        device=device,
        input_specs={"text": compile_specs["text"]},
        options=COMPILE_OPTIONS,
    )

    print(f"Image compile job ID: {image_compile_job.job_id}")
    print(f"Text compile job ID:  {text_compile_job.job_id}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "contract_version": export_manifest.get("contract_version"),
        "pipeline_version": export_manifest.get("pipeline_version"),
        "model_key": export_manifest.get("model_key"),
        "pipeline": pipeline,
        "compile": {
            "device": device.name,
            "compile_options": COMPILE_OPTIONS,
            "input_specs": {
                "image": list(compile_specs["image"]),
                "text": {
                    "shape": list(compile_specs["text"][0]),
                    "dtype": compile_specs["text"][1],
                },
            },
            "jobs": {
                "image_compile_job_id": image_compile_job.job_id,
                "text_compile_job_id": text_compile_job.job_id,
            },
            "onnx_paths": {
                "image": image_onnx_path,
                "text": text_onnx_path,
            },
        },
        "profile": None,
    }

    if not args.skip_profile:
        print("Waiting for compile jobs before profiling...")
        image_compile_job.wait()
        text_compile_job.wait()
        image_target = qai_hub.get_job(image_compile_job.job_id).get_target_model()
        text_target = qai_hub.get_job(text_compile_job.job_id).get_target_model()

        image_profile_job = qai_hub.submit_profile_job(
            model=image_target,
            device=device,
            options=PROFILE_OPTIONS,
        )
        text_profile_job = qai_hub.submit_profile_job(
            model=text_target,
            device=device,
            options=PROFILE_OPTIONS,
        )
        print(f"Image profile job ID: {image_profile_job.job_id}")
        print(f"Text profile job ID:  {text_profile_job.job_id}")
        manifest["profile"] = {
            "profile_options": PROFILE_OPTIONS,
            "jobs": {
                "image_profile_job_id": image_profile_job.job_id,
                "text_profile_job_id": text_profile_job.job_id,
            },
        }

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote compile manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
