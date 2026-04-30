#!/usr/bin/env bash
#
# Pull all training/eval artifacts from a remote GPU host before tearing it down.
# Idempotent: re-running it just refreshes any newer files.
#
# Usage:
#   bash scripts/pull-artifacts.sh                          # default: gh200, latest run dir
#   bash scripts/pull-artifacts.sh gh200                    # explicit host alias
#   bash scripts/pull-artifacts.sh gh200 run_20260428       # specific run dir
#   bash scripts/pull-artifacts.sh gh200 ALL                # every run dir
#
# Pulls into _gh200_pull/ at repo root. The host alias must resolve via your
# ~/.ssh/config (we set this up earlier for `gh200`).

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST="${1:-gh200}"
RUN_NAME="${2:-LATEST}"
REMOTE_REPO="${REMOTE_REPO:-~/lpcvc}"
DEST_ROOT="${DEST_ROOT:-_gh200_pull}"

mkdir -p "$DEST_ROOT/schall_runs"
mkdir -p "$DEST_ROOT/exported_onnx"
mkdir -p "$DEST_ROOT/manifests"

# Resolve which schall_runs/<run_*> dirs to pull.
if [[ "$RUN_NAME" == "LATEST" ]]; then
  RUN_NAME="$(ssh "$REMOTE_HOST" "ls -t $REMOTE_REPO/schall_runs/ 2>/dev/null | grep -E '^run_' | head -1")"
  if [[ -z "$RUN_NAME" ]]; then
    echo "[pull] no run_* dirs under $REMOTE_REPO/schall_runs/ on $REMOTE_HOST" >&2
    exit 1
  fi
  echo "[pull] resolved LATEST -> $RUN_NAME"
fi

if [[ "$RUN_NAME" == "ALL" ]]; then
  RUN_PATTERN="schall_runs/run_*"
else
  RUN_PATTERN="schall_runs/$RUN_NAME"
fi

echo "[pull] host=$REMOTE_HOST  remote=$REMOTE_REPO  pattern=$RUN_PATTERN  dest=$DEST_ROOT"

# 1. schall_runs/<runs> — training artifacts, logs, eval CSVs.
echo "[pull] schall_runs ..."
rsync -avz --partial \
  "$REMOTE_HOST:$REMOTE_REPO/$RUN_PATTERN" \
  "$DEST_ROOT/schall_runs/"

# 2. master.log if it exists at the top of schall_runs.
echo "[pull] master.log (if present) ..."
rsync -avz --ignore-missing-args \
  "$REMOTE_HOST:$REMOTE_REPO/schall_runs/master.log" \
  "$DEST_ROOT/schall_runs/" 2>/dev/null || true

# 3. exported_onnx_fgclip2_*  — submitted ONNX dirs (heavy, 600MB-1GB each).
#    Only pulled if they exist; safe to skip with PULL_ONNX=0.
if [[ "${PULL_ONNX:-1}" == "1" ]]; then
  echo "[pull] exported_onnx_fgclip2_* ..."
  rsync -avz --partial \
    --include="exported_onnx_fgclip2_*/" \
    --include="exported_onnx_fgclip2_*/**" \
    --exclude="*" \
    "$REMOTE_HOST:$REMOTE_REPO/" \
    "$DEST_ROOT/exported_onnx/"
fi

# 4. manifests/fgclip2_*  — compile/inference manifests (tiny but critical).
echo "[pull] manifests/fgclip2_* ..."
rsync -avz --partial \
  --include="fgclip2_*/" --include="fgclip2_*/**" --exclude="*" \
  "$REMOTE_HOST:$REMOTE_REPO/manifests/" \
  "$DEST_ROOT/manifests/"

echo
echo "[pull] DONE. Artifacts at: $DEST_ROOT/"
du -sh "$DEST_ROOT"/* 2>/dev/null || true
