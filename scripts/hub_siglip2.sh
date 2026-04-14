#!/usr/bin/env bash
#
# Convenience wrapper for running SigLIP2 models on QAI Hub.
# Keeps artifacts under exported_onnx_<model>/ and manifests/<model>/,
# and supports resumable per-stage execution for large models.
#
# Usage:
#   ./scripts/hub_siglip2.sh                   # giant, all stages
#   ./scripts/hub_siglip2.sh giant compile     # submit compile jobs only
#   ./scripts/hub_siglip2.sh giant upload      # upload sample dataset only
#   ./scripts/hub_siglip2.sh giant infer       # run inference from existing manifests
#   ./scripts/hub_siglip2.sh base all
#   ./scripts/hub_siglip2.sh giant status
#

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

VARIANT="${1:-giant}"
ACTION="${2:-all}"

case "${VARIANT}" in
  giant)
    MODEL="siglip2_giant_256"
    ;;
  base)
    MODEL="siglip2_base_224"
    ;;
  *)
    echo "Error: unknown SigLIP2 variant '${VARIANT}'" >&2
    echo "Expected one of: giant, base" >&2
    exit 1
    ;;
esac

case "${ACTION}" in
  all|export|compile|upload|infer|status)
    ;;
  *)
    echo "Error: unknown action '${ACTION}'" >&2
    echo "Expected one of: all, export, compile, upload, infer, status" >&2
    exit 1
    ;;
esac

ONNX_DIR="exported_onnx_${MODEL}"
MODEL_MANIFEST_DIR="manifests/${MODEL}"
COMPILE_MANIFEST="${MODEL_MANIFEST_DIR}/compile_manifest.json"
UPLOAD_MANIFEST="${MODEL_MANIFEST_DIR}/upload_manifest.json"
INFERENCE_MANIFEST="${MODEL_MANIFEST_DIR}/inference_manifest.json"

mkdir -p "${MODEL_MANIFEST_DIR}"
export TOKENIZERS_PARALLELISM=false

run_export() {
  echo "=== Export ONNX (${MODEL}) ==="
  "${PYTHON}" export_onnx.py \
    --model "${MODEL}" \
    --out-dir "${ONNX_DIR}"
}

run_compile() {
  echo "=== Submit Compile Jobs (${MODEL}) ==="
  "${PYTHON}" compile_and_profile.py \
    --onnx-dir "${ONNX_DIR}" \
    --manifest-out "${COMPILE_MANIFEST}" \
    --skip-profile
}

run_upload() {
  echo "=== Upload Sample Dataset (${MODEL}) ==="
  "${PYTHON}" upload_dataset.py \
    --model "${MODEL}" \
    --manifest-out "${UPLOAD_MANIFEST}"
}

run_infer() {
  echo "=== Run Inference (${MODEL}) ==="
  "${PYTHON}" inference.py \
    --compile-manifest "${COMPILE_MANIFEST}" \
    --upload-manifest "${UPLOAD_MANIFEST}" \
    --top-k 10 \
    --manifest-out "${INFERENCE_MANIFEST}"
}

show_status() {
  echo "Model: ${MODEL}"
  echo "ONNX dir: ${ONNX_DIR}"
  echo "Compile manifest: ${COMPILE_MANIFEST}"
  echo "Upload manifest: ${UPLOAD_MANIFEST}"
  echo "Inference manifest: ${INFERENCE_MANIFEST}"
  echo ""
  for path in "${COMPILE_MANIFEST}" "${UPLOAD_MANIFEST}" "${INFERENCE_MANIFEST}"; do
    if [[ -f "${path}" ]]; then
      echo "[present] ${path}"
    else
      echo "[missing] ${path}"
    fi
  done
}

case "${ACTION}" in
  export)
    run_export
    ;;
  compile)
    run_compile
    ;;
  upload)
    run_upload
    ;;
  infer)
    run_infer
    ;;
  status)
    show_status
    ;;
  all)
    run_export
    echo ""
    run_compile
    echo ""
    run_upload
    echo ""
    run_infer
    echo ""
    show_status
    ;;
esac
