#!/usr/bin/env bash
#
# Export ONNX models from an open_clip checkpoint.
#
# Usage:
#   ./scripts/export.sh                          # default: mobileclip_s2
#   ./scripts/export.sh --model mobileclip2_s4   # different model

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

exec "${PYTHON}" export_onnx.py "$@"
