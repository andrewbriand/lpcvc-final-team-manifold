"""
Run QAI Hub inference from compile/upload manifests and score Recall@K.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import qai_hub

from lpcvc_contract import CONTRACT_VERSION, QAI_DEVICE, get_contract_spec
from utils.retrieval_eval import evaluate_track1_embeddings, stack_embeddings


def load_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preflight_contract(compile_manifest: dict, upload_manifest: dict) -> None:
    spec = get_contract_spec()
    expected_image_shape = list(spec.image_shape)
    expected_text_shape = list(spec.text_shape)
    expected_text_dtype = spec.text_dtype

    compile_image = compile_manifest.get("compile", {}).get("input_specs", {}).get("image")
    compile_text = compile_manifest.get("compile", {}).get("input_specs", {}).get("text", {})
    if list(compile_image or []) != expected_image_shape:
        raise ValueError(f"Compile manifest image shape mismatch: {compile_image} != {expected_image_shape}")
    if list(compile_text.get("shape") or []) != expected_text_shape:
        raise ValueError(f"Compile manifest text shape mismatch: {compile_text.get('shape')} != {expected_text_shape}")
    if str(compile_text.get("dtype", "")).lower() != expected_text_dtype:
        raise ValueError(f"Compile manifest text dtype mismatch: {compile_text.get('dtype')} != {expected_text_dtype}")

    upload_contract = upload_manifest.get("contract", {})
    upload_image = upload_contract.get("image_shape")
    upload_text = upload_contract.get("text_shape")
    upload_dtype = upload_contract.get("text_dtype")
    if list(upload_image or []) != expected_image_shape:
        raise ValueError(f"Upload manifest image shape mismatch: {upload_image} != {expected_image_shape}")
    if list(upload_text or []) != expected_text_shape:
        raise ValueError(f"Upload manifest text shape mismatch: {upload_text} != {expected_text_shape}")
    if str(upload_dtype or "").lower() != expected_text_dtype:
        raise ValueError(f"Upload manifest text dtype mismatch: {upload_dtype} != {expected_text_dtype}")

    compile_version = compile_manifest.get("contract_version")
    upload_version = upload_manifest.get("contract_version")
    if compile_version and compile_version != CONTRACT_VERSION:
        raise ValueError(f"Compile manifest contract version mismatch: {compile_version} != {CONTRACT_VERSION}")
    if upload_version and upload_version != CONTRACT_VERSION:
        raise ValueError(f"Upload manifest contract version mismatch: {upload_version} != {CONTRACT_VERSION}")


def resolve_ids(compile_manifest: dict, upload_manifest: dict) -> tuple[str, str, str, str]:
    image_compiled_id = compile_manifest.get("compile", {}).get("jobs", {}).get("image_compile_job_id")
    text_compiled_id = compile_manifest.get("compile", {}).get("jobs", {}).get("text_compile_job_id")
    image_dataset_id = upload_manifest.get("dataset_ids", {}).get("image")
    text_dataset_id = upload_manifest.get("dataset_ids", {}).get("text")

    missing = []
    if not image_compiled_id:
        missing.append("compile.jobs.image_compile_job_id")
    if not text_compiled_id:
        missing.append("compile.jobs.text_compile_job_id")
    if not image_dataset_id:
        missing.append("dataset_ids.image")
    if not text_dataset_id:
        missing.append("dataset_ids.text")
    if missing:
        raise ValueError(f"Missing required manifest fields: {missing}")

    return image_compiled_id, text_compiled_id, image_dataset_id, text_dataset_id


def run_inference_job(model, device, dataset):
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=dataset,
        options="--max_profiler_iterations 1",
    )
    inference_job.wait()
    completed_job = qai_hub.get_job(inference_job.job_id)
    if completed_job.get_status().failure:
        raise RuntimeError(f"Inference failed: {completed_job.get_status().failure}")
    return completed_job


def resolve_target_model(compiled_id: str, modality: str):
    compiled_job = qai_hub.get_job(compiled_id)
    target_model = compiled_job.get_target_model()
    if target_model is not None:
        return target_model

    print(f"{modality} compile target is not ready yet, waiting for compile job {compiled_id}...")
    compiled_job.wait()
    refreshed = qai_hub.get_job(compiled_id)
    status = refreshed.get_status()
    if status.failure:
        raise RuntimeError(f"{modality} compile job failed: {status.failure}")
    target_model = refreshed.get_target_model()
    if target_model is None:
        raise RuntimeError(f"{modality} compile job completed but target model is unavailable.")
    return target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LPCVC inference from compile/upload manifests.")
    parser.add_argument("--compile-manifest", default="manifests/compile_manifest.json")
    parser.add_argument("--upload-manifest", default="manifests/upload_manifest.json")
    parser.add_argument("--img-csv", default="dataset/img_list.csv")
    parser.add_argument("--txt-csv", default="dataset/txt_list.csv")
    parser.add_argument("--device", default=QAI_DEVICE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--manifest-out", default="manifests/inference_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compile_manifest = load_json(args.compile_manifest, "Compile")
    upload_manifest = load_json(args.upload_manifest, "Upload")
    preflight_contract(compile_manifest, upload_manifest)
    image_compiled_id, text_compiled_id, image_dataset_id, text_dataset_id = resolve_ids(
        compile_manifest,
        upload_manifest,
    )

    device = qai_hub.Device(args.device)
    outputs = {}
    job_ids = {}

    tasks = {
        "image": (image_compiled_id, image_dataset_id),
        "text": (text_compiled_id, text_dataset_id),
    }
    for modality, (compiled_id, dataset_id) in tasks.items():
        compiled_model = resolve_target_model(compiled_id, modality)
        input_dataset = qai_hub.get_dataset(dataset_id)

        print(f"Running {modality} inference on {device.name} with model={compiled_model.model_id}")
        completed_job = run_inference_job(compiled_model, device, input_dataset)
        job_ids[f"{modality}_inference_job_id"] = completed_job.job_id
        outputs[modality] = completed_job.download_output_data()["output_0"]

    score, diagnostics = evaluate_track1_embeddings(
        img_emb=stack_embeddings(outputs["image"]),
        txt_emb=stack_embeddings(outputs["text"]),
        txt_csv=args.txt_csv,
        img_csv=args.img_csv,
        k=args.top_k,
    )
    print(f"Recall@{args.top_k}: {score:.6f}")
    print(f"Diagnostics: {diagnostics}")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "device": device.name,
        "inputs": {
            "compile_manifest": args.compile_manifest,
            "upload_manifest": args.upload_manifest,
            "img_csv": args.img_csv,
            "txt_csv": args.txt_csv,
        },
        "resolved_ids": {
            "image_compiled_id": image_compiled_id,
            "text_compiled_id": text_compiled_id,
            "image_dataset_id": image_dataset_id,
            "text_dataset_id": text_dataset_id,
        },
        "jobs": job_ids,
        "metrics": {
            f"recall@{args.top_k}": score,
            **diagnostics,
        },
    }

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Wrote inference manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
