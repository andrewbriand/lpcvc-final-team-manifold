#!/usr/bin/env bash
#
# Run the full QAI Hub pipeline for a specific model and keep
# manifests/results isolated under manifests/<model>/.
#
# Usage:
#   ./scripts/hub_model.sh mobileclip2_b
#   ./scripts/hub_model.sh vit_b16 --reuse-upload-manifest manifests/upload_manifest.json
#

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/hub_model.sh <model-key> [--reuse-upload-manifest <path>]" >&2
  exit 1
fi

MODEL="$1"
shift

REUSE_UPLOAD_MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse-upload-manifest)
      if [[ $# -lt 2 ]]; then
        echo "Error: --reuse-upload-manifest requires a path" >&2
        exit 1
      fi
      REUSE_UPLOAD_MANIFEST="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ONNX_DIR="exported_onnx_${MODEL}"
MODEL_MANIFEST_DIR="manifests/${MODEL}"
COMPILE_MANIFEST="${MODEL_MANIFEST_DIR}/compile_manifest.json"
UPLOAD_MANIFEST="${MODEL_MANIFEST_DIR}/upload_manifest.json"
INFERENCE_MANIFEST="${MODEL_MANIFEST_DIR}/inference_manifest.json"

mkdir -p "${MODEL_MANIFEST_DIR}"

echo "=== Model: ${MODEL} ==="
echo "=== Step 1/4: Export ONNX ==="
"${PYTHON}" export_onnx.py \
  --model "${MODEL}" \
  --out-dir "${ONNX_DIR}"

echo ""
echo "=== Step 2/4: Compile ==="
"${PYTHON}" compile_and_profile.py \
  --onnx-dir "${ONNX_DIR}" \
  --manifest-out "${COMPILE_MANIFEST}" \
  --skip-profile

echo ""
if [[ -n "${REUSE_UPLOAD_MANIFEST}" ]]; then
  if [[ ! -f "${REUSE_UPLOAD_MANIFEST}" ]]; then
    echo "Error: upload manifest not found at ${REUSE_UPLOAD_MANIFEST}" >&2
    exit 1
  fi
  cp "${REUSE_UPLOAD_MANIFEST}" "${UPLOAD_MANIFEST}"
  echo "=== Step 3/4: Reuse uploaded dataset ==="
  echo "Copied upload manifest from ${REUSE_UPLOAD_MANIFEST}"
else
  echo "=== Step 3/4: Upload dataset ==="
  "${PYTHON}" upload_dataset.py \
    --model "${MODEL}" \
    --manifest-out "${UPLOAD_MANIFEST}"
fi

echo ""
echo "=== Step 4/4: Inference + Recall@10 ==="
"${PYTHON}" inference.py \
  --compile-manifest "${COMPILE_MANIFEST}" \
  --upload-manifest "${UPLOAD_MANIFEST}" \
  --top-k 10 \
  --manifest-out "${INFERENCE_MANIFEST}"

echo ""
echo "Done."
echo "  Compile manifest:   ${COMPILE_MANIFEST}"
echo "  Upload manifest:    ${UPLOAD_MANIFEST}"
echo "  Inference manifest: ${INFERENCE_MANIFEST}"
