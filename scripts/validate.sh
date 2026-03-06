#!/usr/bin/env bash
#
# Local validation — runs the evaluation harness against one or more datasets.
#
# Usage:
#   ./scripts/validate.sh                          # sample dataset, default model
#   ./scripts/validate.sh --model mobileclip2_s4   # different model
#   ./scripts/validate.sh --datasets quick          # sample + sugarcrepe
#   ./scripts/validate.sh --datasets all            # full matrix
#   ./scripts/validate.sh --disable-cache           # skip embedding cache

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

exec "${PYTHON}" validate.py "$@"
