#!/bin/bash
# Stage 3 — multi-positive InfoNCE.
# Loss-formulation lever: keep v1 hyperparams (rank=16, anchor=1.0, 2 epochs,
# no aug, no CC12M), only change loss to multi-pos with K=5 captions/image.
# Pipeline: train -> export ONNX -> AI Hub compile (--skip-profile) -> HF push.

set -euo pipefail
cd ~/lpcvc
source .venv/bin/activate

export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_PROGRESS_BARS=0
# Load HF token from .env
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

VARIANT="multipos"
RUN_DIR="schall_runs/stage3_${VARIANT}"
ONNX_DIR="exported_onnx_stage3_${VARIANT}"
COMPILE_MANIFEST_DIR="manifests/stage3_${VARIANT}"
HF_REPO="${HF_REPO:-lpcvc2026-track1-stage-battery}"
HF_PREFIX="stage3/${VARIANT}"

mkdir -p "$RUN_DIR/logs" "$ONNX_DIR" "$COMPILE_MANIFEST_DIR"

banner() {
  echo
  echo "================================================================"
  echo "  [stage3/$VARIANT] $1  ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
  echo "================================================================"
}

# ---- Step 1 — Train (multi-positive InfoNCE)
banner "STEP 1 / 4 — Train (multi-positive InfoNCE, K=5)"
python -m self_training.schall_stage1_multipos \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  --out-dir "$RUN_DIR/artifacts/stage1" \
  --hf-cache-dir hf_cache \
  --lora-rank 16 \
  --lora-alpha 32 \
  --epochs 2 \
  --batch-size 128 \
  --anchor-weight 1.0 \
  --no-augment \
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
echo "[stage3/$VARIANT] Stage 1 adapter -> $STAGE1_ADAPTER"

# ---- Step 2 — Export merged ONNX (same export contract as v1: image LoRA
# merged into trunk, retokenizer for the text path)
banner "STEP 2 / 4 — Export merged ONNX"
python scripts/export_fgclip2.py \
  --model-key fgclip2_base \
  --out-dir "$ONNX_DIR" \
  --schall-stage1-adapter "$STAGE1_ADAPTER" \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  2>&1 | tee "$RUN_DIR/logs/02_export.log"

# ---- Step 3 — AI Hub compile (FP16 QNN binary, no INT8, skip profile)
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
echo "  COMPILE JOBS SUBMITTED [stage3/$VARIANT]"
echo "  Image: $IMG_ID  https://workbench.aihub.qualcomm.com/jobs/$IMG_ID/"
echo "  Text:  $TXT_ID  https://workbench.aihub.qualcomm.com/jobs/$TXT_ID/"
echo "==================================================="

# ---- Step 4 — HF push (under stage3/multipos/ prefix)
banner "STEP 4 / 4 — Push artifacts to HF Hub (prefix=$HF_PREFIX)"
python scripts/push_to_hf.py "$HF_REPO" \
  "$RUN_DIR" \
  "$ONNX_DIR" \
  "$COMPILE_MANIFEST_DIR" \
  --prefix "$HF_PREFIX" \
  2>&1 | tee "$RUN_DIR/logs/04_push.log"

banner "PIPELINE COMPLETE [stage3/$VARIANT]"
echo "Stage 1 adapter:   $STAGE1_ADAPTER"
echo "ONNX dir:          $ONNX_DIR"
echo "Compile manifest:  $COMPILE_MANIFEST_DIR/compile_manifest.json"
echo "QAI Image job:     $IMG_ID"
echo "QAI Text job:      $TXT_ID"
echo "HF repo:           https://huggingface.co/$HF_REPO (prefix $HF_PREFIX)"
echo "Logs root:         $RUN_DIR/logs/"
