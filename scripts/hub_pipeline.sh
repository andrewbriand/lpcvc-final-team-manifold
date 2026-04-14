#!/usr/bin/env bash
#
# Full QAI Hub pipeline: compile → upload → inference → score.
# Requires a QAI Hub account and the qai_hub package.
#
# Usage:
#   ./scripts/hub_pipeline.sh                              # defaults to mobileclip2_s2
#   ./scripts/hub_pipeline.sh --onnx-dir exported_onnx_mobileclip2_s4

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

ONNX_DIR="${1:-exported_onnx_mobileclip2_s2}"
COMPILE_MANIFEST="manifests/compile_manifest.json"
UPLOAD_MANIFEST="manifests/upload_manifest.json"
INFERENCE_MANIFEST="manifests/inference_manifest.json"
MODEL="$("${PYTHON}" - "${ONNX_DIR}" <<'PY'
import json
import os
import sys

manifest_path = os.path.join(sys.argv[1], "export_manifest.json")
if not os.path.exists(manifest_path):
    print("mobileclip2_s2")
    raise SystemExit(0)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(manifest.get("model_key", "mobileclip2_s2"))
PY
)"

echo "=== Step 1/3: Compile ==="
"${PYTHON}" compile_and_profile.py \
  --onnx-dir "${ONNX_DIR}" \
  --manifest-out "${COMPILE_MANIFEST}" \
  --skip-profile

echo ""
echo "=== Step 2/3: Upload dataset ==="
"${PYTHON}" upload_dataset.py \
  --model "${MODEL}" \
  --manifest-out "${UPLOAD_MANIFEST}"

echo ""
echo "=== Step 3/3: Inference + Recall@10 ==="
"${PYTHON}" inference.py \
  --compile-manifest "${COMPILE_MANIFEST}" \
  --upload-manifest "${UPLOAD_MANIFEST}" \
  --top-k 10 \
  --manifest-out "${INFERENCE_MANIFEST}"

echo ""
echo "Done. Results in ${INFERENCE_MANIFEST}"
