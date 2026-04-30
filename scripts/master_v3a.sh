#!/bin/bash
# Schall Stage 1 v3a — pure data-scale experiment.
# All v2 changes reverted; only difference vs v1 is +200K CC12M streamed pairs.
# Sequential: train -> export ONNX -> AI Hub compile -> HF push.

set -euo pipefail
cd ~/lpcvc
source .venv/bin/activate

export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_PROGRESS_BARS=0

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="schall_runs/run_v3a_${TS}"
ONNX_DIR="exported_onnx_fgclip2_schall_v3a"
COMPILE_MANIFEST_DIR="manifests/fgclip2_schall_v3a"
HF_REPO="${HF_REPO:-lpcvc2026-track1-schall-v3a}"

mkdir -p "$RUN_DIR/logs" "$ONNX_DIR" "$COMPILE_MANIFEST_DIR"

banner() {
  echo
  echo "================================================================"
  echo "  $1  ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
  echo "================================================================"
}

# ---- Step 1 — Train Stage 1 v3a (v1 config + CC12M 200K)
banner "STEP 1 / 4 — Train (v1 hyperparams + CC12M 200K)"
python -m self_training.schall_stage1 \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  --out-dir "$RUN_DIR/artifacts/stage1" \
  --hf-cache-dir hf_cache \
  --epochs 2 \
  --batch-size 128 \
  --anchor-weight 1.0 \
  --no-augment \
  --cc12m-max-samples 200000 \
  --eval-n-images 5000 \
  2>&1 | tee "$RUN_DIR/logs/01_train.log"

if [[ -d "$RUN_DIR/artifacts/stage1/best" ]]; then
  STAGE1_ADAPTER="$RUN_DIR/artifacts/stage1/best"
else
  STAGE1_ADAPTER="$(ls -td "$RUN_DIR"/artifacts/stage1/epoch_* | head -1)"
fi
echo "[master_v3a] Stage 1 adapter -> $STAGE1_ADAPTER"

# ---- Step 2 — Export merged ONNX
banner "STEP 2 / 4 — Export merged ONNX"
python scripts/export_fgclip2.py \
  --model-key fgclip2_base \
  --out-dir "$ONNX_DIR" \
  --schall-stage1-adapter "$STAGE1_ADAPTER" \
  --retokenizer-checkpoint _a100_artifacts/best.pt \
  2>&1 | tee "$RUN_DIR/logs/02_export.log"

# ---- Step 3 — AI Hub compile (FP16 QNN binary, no INT8)
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
echo "  COMPILE JOBS SUBMITTED"
echo "  Image: $IMG_ID  https://workbench.aihub.qualcomm.com/jobs/$IMG_ID/"
echo "  Text:  $TXT_ID  https://workbench.aihub.qualcomm.com/jobs/$TXT_ID/"
echo "==================================================="

# ---- Step 4 — HF push
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
