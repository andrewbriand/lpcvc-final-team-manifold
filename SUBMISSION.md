# LPCVC 2026 Track 1 Submission Closeout

Team Manifold placed **3rd** in LPCVC 2026 Track 1: Image-to-Text Retrieval. This file documents Team Manifold's implementation, submission lineage, smoke tests, and public artifact boundary.

Official result page: https://lpcv.ai/2026LPCVC/winners/

![Official LPCVC Track 1 result crop showing Team Manifold 3rd Place](assets/lpcvc-2026-track1-team-manifold-result.png)

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

## Methods

FG-CLIP2 does not natively use the `openai/clip-vit-base-patch32` tokenizer required by the LPCVC Track 1 contract. Team Manifold's text path therefore adds a tokenizer-translation wrapper: it accepts the required OpenAI BPE token IDs `(1, 77)`, learns an embedding/adapter representation for the FG-CLIP2 text pathway, truncates to the exported short-mode sequence length, and returns L2-normalized embeddings. The image path keeps the competition input fixed at `(1, 3, 224, 224)`, bakes preprocessing into ONNX, and applies fixed-resolution image-side LoRA adaptation for the documented final submission lineage.

## Ablations and On-Device Latency

Latency values are XR2 Gen 2 Proxy measurements where available.

| Variant | Hidden / expected R@10 | Image ms | Text ms | Total ms | Notes |
|---------|-------------------------|----------|---------|----------|-------|
| MobileCLIP2-S2 baseline | 0.5839 hidden snapshot | 13.638 | 4.307 | 17.945 | early Team Manifold baseline from qualified leaderboard snapshot |
| FG-CLIP2 retokenized run 1 | ~0.61 expected hidden; 0.6429 COCO local before fixed-resolution correction | 18.574 | 5.729 | 24.303 | profile jobs succeeded; under 35 ms latency gate |
| FG-CLIP2 retokenized + Schall Stage 1 image LoRA | 0.6051 hidden final internal closeout | not re-profiled | not re-profiled | not re-profiled | final compile used `--skip-profile`; compile jobs are documented above |

The main ablation lesson was that fixed-resolution evaluation mattered more than broader model changes: earlier FG-CLIP2 local scores were inflated when validation bypassed the deployed `224x224` contract. Stage 1 image-side LoRA improved the hidden score relative to the retokenized baseline, while later combined changes such as web-caption data, weaker anchoring, expanded LoRA scope, and augmentation regressed.

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

Reproduce the Team Manifold submission-family validation once the private retokenizer and LoRA artifacts are restored:

```bash
python3 validate.py \
  --model fgclip2_base_retokenized \
  --retokenizer-checkpoint /path/to/retokenizer/best.pt \
  --schall-stage1-adapter /path/to/schall_stage1/best \
  --fgclip2-fix-resolution \
  --datasets sample
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

The practical closeout lesson is to keep the public repo focused on reproducibility: source code, model contract, exact QAI Hub submission path, and smoke-test evidence.

## Citation

```bibtex
@article{xie2025fgclip2,
  title={FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model},
  author={Xie, Chunyu and Wang, Bin and Kong, Fanjing and Li, Jincheng and Liang, Dawei and Ao, Ji and Leng, Dawei and Yin, Yuhui},
  journal={arXiv preprint arXiv:2510.10921},
  year={2025}
}

@inproceedings{vasu2024mobileclip,
  title={MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  author={Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, Fartash and Vemulapalli, Raviteja and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@article{faghri2025mobileclip2,
  title={MobileCLIP2: Improving Multi-Modal Reinforced Training},
  author={Faghri, Fartash and Vasu, Pavan Kumar Anasosalu and Koc, Cem and Shankar, Vaishaal and Toshev, Alexander and Tuzel, Oncel and Pouransari, Hadi},
  journal={arXiv preprint arXiv:2508.20691},
  year={2025}
}
```

## Contact

For questions about this Team Manifold implementation, open an issue on this repository.

## What Is Not In This Repo

The public final repo intentionally excludes:

- private meeting notes and Discord exports
- speculative private-team commentary
- raw datasets and downloaded Hugging Face caches
- credentials, tokens, QAI Hub local state, and machine-specific secrets
- large generated ONNX, checkpoint, and QAI Hub output files

Generated artifacts should be reproduced from scripts or retrieved from the referenced QAI Hub / Hugging Face artifact records.
