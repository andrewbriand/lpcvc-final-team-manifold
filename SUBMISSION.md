# LPCVC 2026 Track 1 Submission Closeout

Team Manifold placed **3rd** in LPCVC 2026 Track 1: Image-to-Text Retrieval. The official winners page lists:

- Winner: EfficientAI, Beihang University
- 2nd Place: mangojump, Chung Yuan Christian University
- 3rd Place: Team Manifold, Independent Researchers

Official result page: https://lpcv.ai/2026LPCVC/winners/

## Competition Contract

The contract is centralized in `lpcvc_contract.py`.

| Field | Value |
|-------|-------|
| Image input | `float32 (1, 3, 224, 224)` in [0, 1], RGB, CHW |
| Text input | `int32 (1, 77)` at the QNN boundary |
| ONNX text input | Native `int64`; QAI Hub compile uses `--truncate_64bit_io` |
| Tokenizer | `openai/clip-vit-base-patch32` |
| Target device | `XR2 Gen 2 (Proxy)` |
| Compile options | `--target_runtime qnn_context_binary --truncate_64bit_io` |
| Metric | Recall@Top10 |
| Latency gate | image encoder + text encoder < 35 ms |

Do not add an explicit int32-to-int64 Cast node inside the text ONNX graph. QNN handles the int32 boundary conversion through `--truncate_64bit_io`; an in-graph Cast caused earlier QNN inference failures.

## Documented Submission Lineage

The strongest documented lineage in this repo is:

1. FG-CLIP2 base image/text backbone.
2. OpenAI-BPE retokenizer for the competition text contract.
3. Fixed-224 image-side LoRA adaptation following the Schall-style sequential adaptation path.
4. ONNX export with preprocessing and text truncation baked into the graph.
5. QAI Hub compile for `XR2 Gen 2 (Proxy)`.

Primary documented compile jobs:

| Lineage | Image job | Text job | Manifest path |
|---------|-----------|----------|---------------|
| FG-CLIP2 retokenized run 1 | `jp343jqzg` | `j5q7k6wmg` | `manifests/fgclip2_base_retokenized_run1/compile_manifest.json` |
| FG-CLIP2 Schall Stage 1 | `jgzxkozk5` | `jg93ej2lg` | `manifests/fgclip2_schall_stage1/compile_manifest.json` |

Later local/compiled probes exist in `manifests/stage*_multipos*`, but generated manifest directories are intentionally gitignored. Treat those as generated job records, not source files.

## Reproduction Commands

Install:

```bash
python3 -m pip install -r requirements.txt
```

Download the LPCVC sample dataset into `dataset/`.

Run local validation:

```bash
python3 validate.py --model mobileclip2_s2 --datasets sample
```

Export and compile a baseline:

```bash
python3 export_onnx.py --model mobileclip2_s2

python3 compile_and_profile.py \
  --onnx-dir exported_onnx_mobileclip2_s2 \
  --manifest-out manifests/compile_manifest.json \
  --skip-profile
```

Upload sample data and score QAI Hub inference:

```bash
python3 upload_dataset.py \
  --manifest-out manifests/upload_manifest.json

python3 inference.py \
  --compile-manifest manifests/compile_manifest.json \
  --upload-manifest manifests/upload_manifest.json \
  --top-k 10 \
  --manifest-out manifests/inference_manifest.json
```

Share submitted compile jobs with judges:

```bash
python3 - <<'PY'
import json
import qai_hub

manifest_path = "manifests/fgclip2_schall_stage1/compile_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

jobs = manifest["compile"]["jobs"]
for job_id in [jobs["image_compile_job_id"], jobs["text_compile_job_id"]]:
    qai_hub.get_job(job_id).modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print("shared", job_id)
PY
```

## Smoke Test Used For Closeout

Run FAISS tests separately from Torch/WebDataset tests on macOS/Python 3.12 to avoid a native OpenMP runtime conflict.

```bash
python3 -m compileall \
  lpcvc_contract.py lpcvc_models.py export_onnx.py compile_and_profile.py \
  upload_dataset.py inference.py validate.py utils self_training scripts

python3 validate.py --list-datasets

python3 upload_dataset.py \
  --skip-upload \
  --manifest-out /tmp/lpcvc_upload_smoke_manifest.json

python3 -m pytest tests/test_faiss_index.py -q
python3 -m pytest tests -q --ignore=tests/test_faiss_index.py

python3 validate.py \
  --model mobileclip2_s2 \
  --datasets sample \
  --batch-size 8 \
  --results-out /tmp/lpcvc_validate_sample_smoke.csv
```

Closeout smoke result on 2026-05-16:

- FAISS test process: `3 passed`
- non-FAISS test process: `18 passed`
- upload dry-run: 56 images and 211 texts, contract shapes matched
- sample validation smoke: `R@10 = 0.8957`

## Public Lessons

The biggest local lesson was that evaluation must exactly match the deployment contract. Earlier FG-CLIP2 validation paths bypassed the fixed 224x224 contract and inflated local retrieval scores.

The strongest public lesson from the winning EfficientAI repository is activation replacement as a trained reconstruction problem: their README describes replacing GELU MLP activations with ReLU by layer-by-layer knowledge distillation, based on APHQ-ViT. That is a better template for future low-power VLM work than untrained activation swaps.

EfficientAI repository: https://github.com/jn12-29/LPCV-Track1-EfficientAI

## What Is Not In This Repo

The public final repo intentionally excludes:

- private meeting notes and Discord exports
- speculative private-team commentary
- raw datasets and downloaded Hugging Face caches
- credentials, tokens, QAI Hub local state, and machine-specific secrets
- large generated ONNX, checkpoint, and QAI Hub output files

Generated artifacts should be reproduced from scripts or retrieved from the referenced QAI Hub / Hugging Face artifact records.
