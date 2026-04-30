#!/bin/bash
# Schall Stage 1 v2 — full autonomous pipeline.
# Steps (sequential, fail-fast): train -> export ONNX -> AI Hub compile -> HF push.
#
# Designed to run inside a tmux session on a GPU box. After the final HF push
# the script exits cleanly, freeing the box to be terminated by the operator.

set -euo pipefail
cd ~/lpcvc
source .venv/bin/activate

export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_PROGRESS_BARS=0

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="schall_runs/run_v2_${TS}"
ONNX_DIR="exported_onnx_fgclip2_schall_v2"
COMPILE_MANIFEST_DIR="manifests/fgclip2_schall_v2"
HF_REPO="${HF_REPO:-lpcvc2026-track1-schall-v2}"

mkdir -p "$RUN_DIR/logs" "$ONNX_DIR" "$COMPILE_MANIFEST_DIR"

banner() {
  echo
  echo "================================================================"
  echo "  $1  ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
  echo "================================================================"
}

# ---------------------------------------------------------------------------
# Step 1 — Train Stage 1 v2
# ---------------------------------------------------------------------------
banner "STEP 1 / 4 — Schall Stage 1 v2 training"
python -m self_training.schall_stage1 \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  --out-dir "$RUN_DIR/artifacts/stage1" \
  --hf-cache-dir hf_cache \
  --epochs 3 \
  --batch-size 128 \
  --anchor-weight 0.5 \
  --cc12m-max-samples 200000 \
  --eval-n-images 5000 \
  2>&1 | tee "$RUN_DIR/logs/01_train.log"

if [[ ! -d "$RUN_DIR/artifacts/stage1/best" ]]; then
  echo "[master_v2] no best/ dir — falling back to last epoch"
  LAST_EPOCH="$(ls -td "$RUN_DIR"/artifacts/stage1/epoch_* 2>/dev/null | head -1)"
  if [[ -z "$LAST_EPOCH" ]]; then
    echo "ERROR: training produced no epoch dirs; aborting"
    exit 1
  fi
  STAGE1_ADAPTER="$LAST_EPOCH"
else
  STAGE1_ADAPTER="$RUN_DIR/artifacts/stage1/best"
fi
echo "[master_v2] Stage 1 adapter -> $STAGE1_ADAPTER"

# ---------------------------------------------------------------------------
# Step 2 — Export merged ONNX
# ---------------------------------------------------------------------------
banner "STEP 2 / 4 — Export merged ONNX"
python scripts/export_fgclip2.py \
  --model-key fgclip2_base \
  --out-dir "$ONNX_DIR" \
  --schall-stage1-adapter "$STAGE1_ADAPTER" \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  2>&1 | tee "$RUN_DIR/logs/02_export.log"

# ---------------------------------------------------------------------------
# Step 3 — AI Hub compile (no INT8, FP16 QNN context binary)
# ---------------------------------------------------------------------------
banner "STEP 3 / 4 — AI Hub compile"
python compile_and_profile.py \
  --onnx-dir "$ONNX_DIR" \
  --manifest-out "$COMPILE_MANIFEST_DIR/compile_manifest.json" \
  --skip-profile \
  2>&1 | tee "$RUN_DIR/logs/03_compile.log"

# Echo the job IDs prominently for the operator
IMG_ID="$(python -c "import json; print(json.load(open('$COMPILE_MANIFEST_DIR/compile_manifest.json'))['compile']['jobs']['image_compile_job_id'])")"
TXT_ID="$(python -c "import json; print(json.load(open('$COMPILE_MANIFEST_DIR/compile_manifest.json'))['compile']['jobs']['text_compile_job_id'])")"
echo
echo "==================================================="
echo "  COMPILE JOBS SUBMITTED"
echo "  Image: $IMG_ID  https://workbench.aihub.qualcomm.com/jobs/$IMG_ID/"
echo "  Text:  $TXT_ID  https://workbench.aihub.qualcomm.com/jobs/$TXT_ID/"
echo "==================================================="

# ---------------------------------------------------------------------------
# Step 4 — Push artifacts to HuggingFace Hub
# ---------------------------------------------------------------------------
banner "STEP 4 / 4 — Push artifacts to HF Hub"
python scripts/push_to_hf.py "$HF_REPO" \
  "$RUN_DIR" \
  "$ONNX_DIR" \
  "$COMPILE_MANIFEST_DIR" \
  2>&1 | tee "$RUN_DIR/logs/04_push.log"

banner "PIPELINE COMPLETE — safe to terminate GPU"
echo "Stage 1 adapter:   $STAGE1_ADAPTER"
echo "ONNX dir:          $ONNX_DIR"
echo "Compile manifest:  $COMPILE_MANIFEST_DIR/compile_manifest.json"
echo "QAI Image job:     $IMG_ID"
echo "QAI Text job:      $TXT_ID"
echo "HF repo:           https://huggingface.co/$HF_REPO"
echo "Logs root:         $RUN_DIR/logs/"
