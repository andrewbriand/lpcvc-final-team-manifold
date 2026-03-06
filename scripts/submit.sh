#!/usr/bin/env bash
#
# Share compile jobs with the LPCVC judges and print the submission form link.
#
# Usage:
#   ./scripts/submit.sh                                          # uses default manifest
#   ./scripts/submit.sh manifests/compile_manifest.json          # explicit path

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Error: no python interpreter found" >&2
  exit 1
fi

MANIFEST="${1:-manifests/compile_manifest.json}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Error: compile manifest not found at ${MANIFEST}" >&2
  echo "Run ./scripts/hub_pipeline.sh first." >&2
  exit 1
fi

"${PYTHON}" -c "
import json, qai_hub
m = json.load(open('${MANIFEST}'))
img_id = m['compile']['jobs']['image_compile_job_id']
txt_id = m['compile']['jobs']['text_compile_job_id']
qai_hub.get_job(img_id).modify_sharing(add_emails=['lowpowervision@gmail.com'])
qai_hub.get_job(txt_id).modify_sharing(add_emails=['lowpowervision@gmail.com'])
print(f'Shared image job: {img_id}')
print(f'Shared text job:  {txt_id}')
print()
print('Now fill out the submission form:')
print('  https://lpcv.ai/2026LPCVC/image-text-retrieval/')
"
