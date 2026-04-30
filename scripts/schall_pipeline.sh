#!/bin/bash
# Schall sequential pipeline for FG-CLIP 2 retokenized.
#
# Steps (run in order, fail-fast):
#   0. Empirical bug verification — eval Run #1 best.pt with --fgclip2-fix-resolution.
#      Confirms the diagnosed eval-deployment resolution mismatch and fixes
#      the headline COCO baseline number we expect to beat.
#   1. Stage 1 — image-encoder LoRA finetune at fixed-224 (Schall image-side stage).
#   2. Stage 2 — retokenizer realign against the LoRA'd image embeddings.
#   3. Final eval — full validate.py run combining Stage 1 + Stage 2,
#      with --fgclip2-fix-resolution, on COCO + Flickr + sample.
#
# Each step writes a per-step log to $RUN_DIR/<step>.log. The whole pipeline
# stops on first error.

set -euo pipefail

usage() {
  cat <<EOF
Usage: bash scripts/schall_pipeline.sh \\
  --best-pt PATH                  Run #1 retokenizer best.pt (input)
  --run-dir DIR                   Output dir for all step artifacts
  [--epochs-stage1 N]             Default: 2
  [--epochs-stage2 N]             Default: 2
  [--batch-size-stage1 N]         Default: 128
  [--batch-size-stage2 N]         Default: 256
  [--cc12m-stage1 N]              CC12M streamed pairs for Stage 1 (0=off). Default: 0
  [--cc12m-stage2 N]              CC12M streamed pairs for Stage 2 (0=off). Default: 0
  [--smoke]                       Run all stages in 50-step smoke mode
  [--skip-step0]                  Skip empirical bug verification
  [--no-eval-stages]              Disable per-epoch eval inside stages (use only step 3)
EOF
}

# Defaults
BEST_PT=""
RUN_DIR=""
EPOCHS_S1=2
EPOCHS_S2=2
BS_S1=128
BS_S2=256
CC12M_S1=0
CC12M_S2=0
SMOKE=""
SKIP_STEP0=""
NO_EVAL_STAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --best-pt) BEST_PT="$2"; shift 2;;
    --run-dir) RUN_DIR="$2"; shift 2;;
    --epochs-stage1) EPOCHS_S1="$2"; shift 2;;
    --epochs-stage2) EPOCHS_S2="$2"; shift 2;;
    --batch-size-stage1) BS_S1="$2"; shift 2;;
    --batch-size-stage2) BS_S2="$2"; shift 2;;
    --cc12m-stage1) CC12M_S1="$2"; shift 2;;
    --cc12m-stage2) CC12M_S2="$2"; shift 2;;
    --smoke) SMOKE="--smoke"; shift;;
    --skip-step0) SKIP_STEP0="1"; shift;;
    --no-eval-stages) NO_EVAL_STAGES="--no-eval"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$BEST_PT" || -z "$RUN_DIR" ]]; then
  echo "ERROR: --best-pt and --run-dir are required"
  usage
  exit 1
fi
if [[ ! -f "$BEST_PT" ]]; then
  echo "ERROR: best.pt not found at $BEST_PT"
  exit 1
fi

mkdir -p "$RUN_DIR"
ARTIFACTS="$RUN_DIR/artifacts"
LOGS="$RUN_DIR/logs"
mkdir -p "$ARTIFACTS" "$LOGS"

STAGE1_DIR="$ARTIFACTS/stage1"
STAGE2_DIR="$ARTIFACTS/stage2"

# Print a banner so progress is unambiguous in the GH200 log
banner() {
  local msg="$1"
  echo
  echo "================================================================"
  echo "  $msg  ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
  echo "================================================================"
}

PIPELINE_START="$(date +%s)"

cd "$(dirname "$0")/.."  # repo root
REPO="$(pwd)"
PYTHON="${PYTHON:-python}"

# ----------------------------------------------------------------------------
# Step 0 — Empirical bug verification
# ----------------------------------------------------------------------------
if [[ -z "$SKIP_STEP0" ]]; then
  banner "STEP 0 / 3 — Empirical bug verification on best.pt"
  STEP0_LOG="$LOGS/step0_empirical_bug.log"
  "$PYTHON" validate.py \
    --model fgclip2_base_retokenized \
    --datasets retrieval \
    --retokenizer-checkpoint "$BEST_PT" \
    --fgclip2-fix-resolution \
    --hf-cache-dir hf_cache \
    --batch-size 64 \
    --results-out "$RUN_DIR/step0_results.csv" \
    2>&1 | tee "$STEP0_LOG"
  echo "[step 0] results -> $RUN_DIR/step0_results.csv"
else
  banner "STEP 0 / 3 — SKIPPED"
fi

# ----------------------------------------------------------------------------
# Step 1 — Stage 1 image-encoder LoRA finetune
# ----------------------------------------------------------------------------
banner "STEP 1 / 3 — Schall Stage 1 (image-encoder LoRA finetune)"
STEP1_LOG="$LOGS/step1_stage1.log"
"$PYTHON" -m self_training.schall_stage1 \
  --retokenizer-checkpoint "$BEST_PT" \
  --out-dir "$STAGE1_DIR" \
  --hf-cache-dir hf_cache \
  --epochs "$EPOCHS_S1" \
  --batch-size "$BS_S1" \
  --cc12m-max-samples "$CC12M_S1" \
  $SMOKE \
  $NO_EVAL_STAGES \
  2>&1 | tee "$STEP1_LOG"

# Stage 1 produces $STAGE1_DIR/best/ (peft adapter dir)
if [[ ! -d "$STAGE1_DIR/best" ]]; then
  if [[ -d "$STAGE1_DIR/epoch_${EPOCHS_S1}" ]]; then
    echo "[step 1] no best/ written, fallback to epoch_${EPOCHS_S1}/"
    STAGE1_ADAPTER="$STAGE1_DIR/epoch_${EPOCHS_S1}"
  else
    echo "ERROR: Stage 1 did not produce a usable adapter dir under $STAGE1_DIR"
    exit 1
  fi
else
  STAGE1_ADAPTER="$STAGE1_DIR/best"
fi
echo "[step 1] adapter dir -> $STAGE1_ADAPTER"

# ----------------------------------------------------------------------------
# Step 2 — Stage 2 retokenizer realign
# ----------------------------------------------------------------------------
banner "STEP 2 / 3 — Schall Stage 2 (retokenizer realign)"
STEP2_LOG="$LOGS/step2_stage2.log"
"$PYTHON" -m self_training.schall_stage2 \
  --stage1-adapter-dir "$STAGE1_ADAPTER" \
  --stage1-retokenizer-warmstart "$BEST_PT" \
  --out-dir "$STAGE2_DIR" \
  --hf-cache-dir hf_cache \
  --epochs "$EPOCHS_S2" \
  --batch-size "$BS_S2" \
  --cc12m-max-samples "$CC12M_S2" \
  $SMOKE \
  $NO_EVAL_STAGES \
  2>&1 | tee "$STEP2_LOG"

if [[ -f "$STAGE2_DIR/best.pt" ]]; then
  STAGE2_RETOKENIZER="$STAGE2_DIR/best.pt"
elif [[ -f "$STAGE2_DIR/epoch_${EPOCHS_S2}.pt" ]]; then
  echo "[step 2] no best.pt written, fallback to epoch_${EPOCHS_S2}.pt"
  STAGE2_RETOKENIZER="$STAGE2_DIR/epoch_${EPOCHS_S2}.pt"
else
  echo "ERROR: Stage 2 did not produce a usable retokenizer .pt under $STAGE2_DIR"
  exit 1
fi
echo "[step 2] retokenizer -> $STAGE2_RETOKENIZER"

# ----------------------------------------------------------------------------
# Step 3 — Final eval combining Stage 1 + Stage 2 with the contract fix
# ----------------------------------------------------------------------------
banner "STEP 3 / 3 — Final eval (validate.py, contract preprocess)"
STEP3_LOG="$LOGS/step3_final_eval.log"
"$PYTHON" validate.py \
  --model fgclip2_base_retokenized \
  --datasets retrieval \
  --retokenizer-checkpoint "$STAGE2_RETOKENIZER" \
  --schall-stage1-adapter "$STAGE1_ADAPTER" \
  --fgclip2-fix-resolution \
  --hf-cache-dir hf_cache \
  --batch-size 64 \
  --results-out "$RUN_DIR/step3_results.csv" \
  2>&1 | tee "$STEP3_LOG"
echo "[step 3] results -> $RUN_DIR/step3_results.csv"

PIPELINE_END="$(date +%s)"
ELAPSED=$((PIPELINE_END - PIPELINE_START))

banner "PIPELINE COMPLETE — total ${ELAPSED}s"
echo "Stage 1 adapter:   $STAGE1_ADAPTER"
echo "Stage 2 retokenizer: $STAGE2_RETOKENIZER"
echo "Step 0 results:    $RUN_DIR/step0_results.csv"
echo "Step 3 results:    $RUN_DIR/step3_results.csv"
echo "All logs in:       $LOGS/"
