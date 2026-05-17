#!/bin/bash
# Stage 1 — variant a1: +CC12M 200K, anchor=1.0, rank=16
# Data-scale lever (clean v3a config).
# Pipeline: train -> export ONNX -> AI Hub compile (--skip-profile) -> HF push.

set -euo pipefail
cd ~/lpcvc
source .venv/bin/activate

export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_PROGRESS_BARS=0
# Load HF token from .env
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

VARIANT="a1_cc12m_anchor1_rank16"
RUN_DIR="schall_runs/stage1_${VARIANT}"
ONNX_DIR="exported_onnx_stage1_${VARIANT}"
COMPILE_MANIFEST_DIR="manifests/stage1_${VARIANT}"
HF_REPO="${HF_REPO:-lpcvc2026-track1-stage-battery}"
HF_SUBPATH="stage1/${VARIANT}"

mkdir -p "$RUN_DIR/logs" "$ONNX_DIR" "$COMPILE_MANIFEST_DIR"

banner() {
  echo
  echo "================================================================"
  echo "  [$VARIANT] $1  ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
  echo "================================================================"
}

# ---- Step 1 — Train
banner "STEP 1 / 4 — Train"
python -m self_training.schall_stage1 \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  --out-dir "$RUN_DIR/artifacts/stage1" \
  --hf-cache-dir hf_cache \
  --epochs 2 \
  --batch-size 128 \
  --anchor-weight 1.0 \
  --no-augment \
  --lora-rank 16 \
  --lora-alpha 32 \
  --cc12m-max-samples 200000 \
  --eval-n-images 5000 \
  2>&1 | tee "$RUN_DIR/logs/01_train.log"

if [[ -d "$RUN_DIR/artifacts/stage1/best" ]]; then
  STAGE1_ADAPTER="$RUN_DIR/artifacts/stage1/best"
else
  STAGE1_ADAPTER="$(ls -td "$RUN_DIR"/artifacts/stage1/epoch_* 2>/dev/null | head -1)"
  if [[ -z "$STAGE1_ADAPTER" ]]; then
    echo "ERROR: training produced no adapter dirs; aborting"
    exit 1
  fi
fi
echo "[$VARIANT] Stage 1 adapter -> $STAGE1_ADAPTER"

# ---- Step 2 — Export merged ONNX
banner "STEP 2 / 4 — Export merged ONNX"
python scripts/export_fgclip2.py \
  --model-key fgclip2_base \
  --out-dir "$ONNX_DIR" \
  --schall-stage1-adapter "$STAGE1_ADAPTER" \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  2>&1 | tee "$RUN_DIR/logs/02_export.log"

# ---- Step 3 — AI Hub compile
banner "STEP 3 / 4 — AI Hub compile"
python compile_and_profile.py \
  --onnx-dir "$ONNX_DIR" \
  --manifest-out "$COMPILE_MANIFEST_DIR/compile_manifest.json" \
  --skip-profile \
  2>&1 | tee "$RUN_DIR/logs/03_compile.log"

IMG_ID="$(python -c "import json; print(json.load(open('$COMPILE_MANIFEST_DIR/compile_manifest.json'))['compile']['jobs']['image_compile_job_id'])")"
TXT_ID="$(python -c "import json; print(json.load(open('$COMPILE_MANIFEST_DIR/compile_manifest.json'))['compile']['jobs']['text_compile_job_id'])")"
echo
echo "==================================================="
echo "  COMPILE JOBS SUBMITTED [$VARIANT]"
echo "  Image: $IMG_ID  https://workbench.aihub.qualcomm.com/jobs/$IMG_ID/"
echo "  Text:  $TXT_ID  https://workbench.aihub.qualcomm.com/jobs/$TXT_ID/"
echo "==================================================="

# ---- Step 4 — HF push (upload under stage1/<variant>/ path)
banner "STEP 4 / 4 — Push artifacts to HF Hub"
python scripts/push_to_hf.py "$HF_REPO" \
  "$RUN_DIR" \
  "$ONNX_DIR" \
  "$COMPILE_MANIFEST_DIR" \
  2>&1 | tee "$RUN_DIR/logs/04_push.log"

banner "PIPELINE COMPLETE [$VARIANT]"
echo "Stage 1 adapter:   $STAGE1_ADAPTER"
echo "ONNX dir:          $ONNX_DIR"
echo "Compile manifest:  $COMPILE_MANIFEST_DIR/compile_manifest.json"
echo "QAI Image job:     $IMG_ID"
echo "QAI Text job:      $TXT_ID"
echo "HF repo:           https://huggingface.co/$HF_REPO"
echo "Logs root:         $RUN_DIR/logs/"
