# LPCVC 2026 Track 1 — Image-to-Text Retrieval

Team Manifold's submission harness for [2026 LPCVC Track 1](https://lpcv.ai/2026LPCVC/image-text-retrieval/), the image-to-text retrieval challenge for Qualcomm XR2 Gen 2.

The official [2026 LPCVC results page](https://lpcv.ai/2026LPCVC/winners/) lists Team Manifold as **3rd Place** for Track 1. This repository keeps the reproducible Team Manifold pipeline, local validation harness, QAI Hub export/compile scripts, and final public closeout notes.

![Official LPCVC Track 1 result crop showing Team Manifold 3rd Place](assets/lpcvc-2026-track1-team-manifold-result.png)

## Final submission summary

The final submission line used the competition contract in `lpcvc_contract.py`:

| Field | Value |
|-------|-------|
| Image input | `float32 (1, 3, 224, 224)` in [0, 1], RGB |
| Text input | `int32 (1, 77)` — CLIP token IDs at the QNN boundary |
| Tokenizer | `openai/clip-vit-base-patch32` |
| Target device | XR2 Gen 2 (Proxy) |
| Metric | Recall@Top10 |
| Latency gate | Image + text encoder < 35 ms combined |

The best documented submission lineage in this repo is FG-CLIP2 base with an OpenAI-BPE retokenizer and fixed-224 image-side LoRA adaptation. See [SUBMISSION.md](SUBMISSION.md) for the compile job IDs, artifact policy, smoke-test commands, and closeout notes.

## Methods

Team Manifold adapted FG-CLIP2 to the LPCVC contract by adding a text-side tokenizer-translation wrapper. FG-CLIP2 natively uses a Gemma tokenizer and short-mode text length 64, while LPCVC requires OpenAI CLIP BPE IDs shaped `(B, 77)`. The wrapper replaces FG-CLIP2's native token table with a trainable OpenAI-BPE table `(49408, 768)`, replaces the short-mode position table with a trainable `(64, 768)` table, slices the contract input from 77 to 64 tokens, optionally applies a residual `Linear -> GELU -> Linear` adapter, and then runs the frozen FG-CLIP2 short-mode text trunk. The retokenizer was trained against frozen FG-CLIP2 Gemma-tokenized teacher text features with mean `1 - cosine(student, teacher)`. The final documented image side keeps the deployed `224x224` input contract and applies attention-only image LoRA with contrastive retrieval loss plus a KL anchor to the pre-adaptation image embeddings.

Full reproduction details are in [SUBMISSION.md](SUBMISSION.md).

## Final numbers

These are the documented Team Manifold closeout numbers in this repo. Hidden scores are from the qualified leaderboard / closeout records; latency is XR2 Gen 2 Proxy where a profile artifact exists.

| Variant | Recall@10 | XR2 latency | Status |
|---------|-----------|-------------|--------|
| Official LPCVC result | 3rd place | under 35 ms required | public result page |
| MobileCLIP2-S2 baseline | 0.5839 hidden snapshot | 17.945 ms total: 13.638 image + 4.307 text | baseline only; not the final method |
| FG-CLIP2 retokenized run 1 | 0.6429 COCO local before fixed-resolution correction; 0.586 hidden | 24.303 ms total: 18.574 image + 5.729 text | under latency gate; local score was inflated by resolution mismatch |
| FG-CLIP2 fixed-224 + Schall Stage 1 LoRA | 0.6218 COCO local; 0.6051 hidden closeout, +0.019 over run 1 | not re-profiled in final checkout; same graph family was previously profiled at 24.303 ms before LoRA merge | documented final submission lineage |

See [SUBMISSION.md](SUBMISSION.md#ablation-and-latency-table) for the ablation table, including fixed-resolution correction and rejected Stage 1/Stage 2 variants.

## Reproduce the Team Manifold submission

```bash
python3 -m pip install -r requirements.txt
```

Download the [sample dataset](https://drive.google.com/drive/folders/1tTwrehwLtVtMOTjD5beYDC3lcWIfzS5D) and place it at `dataset/` (images + CSVs).

Restore the private generated artifacts excluded from Git:

- retokenizer checkpoint: `best.pt`
- Schall Stage 1 image LoRA adapter directory: `stage1_best/` or equivalent

Then run the submission-family validation path:

```bash
python3 validate.py \
  --model fgclip2_base_retokenized \
  --retokenizer-checkpoint /path/to/retokenizer/best.pt \
  --schall-stage1-adapter /path/to/schall_stage1/best \
  --fgclip2-fix-resolution \
  --datasets sample
```

For retrieval datasets, this harness reports fractional multi-ground-truth recall so local validation stays closer to the LPCVC competition scorer.

## Sanity check (baseline)

Use this when you only need to verify install, dataset layout, tokenization, and scoring. It evaluates the MobileCLIP2-S2 baseline and does not reproduce Team Manifold's 3rd-place submission.

```bash
./scripts/validate.sh
```

By default this prints Recall@1/5/10 for `mobileclip2_s2` on the LPCVC sample set.

## Smoke test

Run FAISS tests in a separate Python process from the Torch/WebDataset tests. On this macOS/Python 3.12 environment, loading FAISS and Torch-backed dependencies in the same process can trip a native OpenMP runtime conflict.

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

## Validation options

```bash
./scripts/validate.sh --model mobileclip2_s4     # try a different model
./scripts/validate.sh --datasets quick            # sample + sugarcrepe
./scripts/validate.sh --datasets retrieval        # sample + mscoco + flickr30k
./scripts/validate.sh --datasets all              # everything
./scripts/validate.sh --disable-cache             # skip the embedding cache
./scripts/validate.sh --list-datasets             # show available datasets
```

Results are saved as timestamped CSVs in `results/`. To generate comparison charts:

```bash
./scripts/visualize.sh                       # auto-discover all result CSVs
./scripts/visualize.sh results/vit_b16_all.csv results/mobileclip2_b_all.csv  # specific files
```

Charts are saved to `results/charts/` (Recall@1, Recall@10, SugarCrepe accuracy, latency, radar).

### Supported datasets

| Key | Type | Source | Description |
|-----|------|--------|-------------|
| `sample` | retrieval | local | LPCVC sample set (~56 images) |
| `mscoco` | retrieval | HuggingFace | MSCOCO Karpathy 5K |
| `flickr30k` | retrieval | HuggingFace | Flickr30K benchmark |
| `sugarcrepe` | binary | HuggingFace | Hard-negative compositionality |

Groups: `quick` = sample + sugarcrepe, `retrieval` = sample + mscoco + flickr30k, `all` = everything.

### Caching

Preprocessed tensors, tokens, and embeddings are cached in `.cache/validate/` by default. This makes repeat runs near-instant. Disable with `--disable-cache`.

## Available models

| Key | Model | Notes |
|-----|-------|-------|
| `mobileclip_s1` | MobileCLIP-S1 | v1, fastest |
| `mobileclip_s2` | MobileCLIP-S2 | v1 baseline |
| `mobileclip_b` | MobileCLIP-B | v1, larger |
| `mobileclip2_s0` | MobileCLIP2-S0 | v2, smallest |
| `mobileclip2_s2` | MobileCLIP2-S2 | v2 baseline; default sanity-check model |
| `mobileclip2_s3` | MobileCLIP2-S3 | v2, ~70M |
| `mobileclip2_s4` | MobileCLIP2-S4 | v2, strongest |
| `mobileclip2_b` | MobileCLIP2-B | v2, ~110M |
| `mobileclip2_l14` | MobileCLIP2-L-14 | v2, largest |
| `vit_b16` | ViT-B/16 | OpenAI CLIP baseline |
| `vit_l14` | ViT-L/14 | OpenAI CLIP large baseline |
| `tripletclip_cc12m` | TripletCLIP | ViT-B/32 on CC12M (HF) |
| `fgclip2_base` | FG-CLIP2 Base | HF model, native tokenizer; experiment baseline |
| `fgclip2_base_retokenized` | FG-CLIP2 Base + OpenAI-BPE retokenizer | Team Manifold submission-family text wrapper; use with `--retokenizer-checkpoint` |
| `fgclip2_base_retokenized` + `--schall-stage1-adapter` | FG-CLIP2 Base + retokenizer + image LoRA | Documented final submission lineage |
| `siglip2_base_224` | SigLIP2 Base | Google SigLIP2 patch16 224 (HF) |
| `siglip2_giant_256` | SigLIP2 Giant 1B | Google SigLIP2 giant patch16 256 (HF) |

SigLIP2 runs are supported in the local validation harness. They use the model-native Gemma tokenizer with `max_length=64`, so they are useful for comparison experiments but are not drop-in compatible with the LPCVC submission contract in `lpcvc_contract.py`.

## QAI Hub submission pipeline

These steps compile and deploy to Qualcomm XR2 Gen 2 via [QAI Hub](https://aihub.qualcomm.com). Requires a QAI Hub account.

### Submission-family export and compile

Once the private artifacts are restored, export the documented Team Manifold lineage with the FG-CLIP2 retokenized text path and merged Stage 1 image LoRA adapter:

```bash
python3 scripts/export_fgclip2.py \
  --model-key fgclip2_base \
  --retokenizer-checkpoint /path/to/retokenizer/best.pt \
  --schall-stage1-adapter /path/to/schall_stage1/best \
  --out-dir exported_onnx_fgclip2_schall_stage1

python3 compile_and_profile.py \
  --onnx-dir exported_onnx_fgclip2_schall_stage1 \
  --manifest-out manifests/fgclip2_schall_stage1/compile_manifest.json \
  --skip-profile
```

### Baseline scaffold

```bash
# 1. Export ONNX (image + text encoders with preprocessing baked in)
./scripts/export.sh --model mobileclip2_s2

# 2. Compile, upload dataset, run inference, score
./scripts/hub_pipeline.sh

# 3. Share with judges and submit
./scripts/submit.sh
```

Or run each stage individually:

```bash
python3 export_onnx.py --model mobileclip2_s2

python3 compile_and_profile.py \
  --onnx-dir exported_onnx_mobileclip2_s2 \
  --manifest-out manifests/compile_manifest.json \
  --skip-profile

python3 upload_dataset.py \
  --manifest-out manifests/upload_manifest.json

python3 inference.py \
  --compile-manifest manifests/compile_manifest.json \
  --upload-manifest manifests/upload_manifest.json \
  --top-k 10 \
  --manifest-out manifests/inference_manifest.json
```

## Project layout

```
validate.py              # local evaluation harness (main entry point)
visualize_results.py     # generate comparison charts from result CSVs
lpcvc_contract.py        # I/O contract: shapes, dtypes, tokenizer
lpcvc_models.py          # model registry (open_clip names + pretrained tags)
export_onnx.py           # export to ONNX with baked-in preprocessing
compile_and_profile.py   # compile ONNX on QAI Hub
upload_dataset.py        # upload sample data to QAI Hub
inference.py             # run QAI Hub inference + Recall@K
utils/retrieval_eval.py  # shared scoring helpers
scripts/                 # bash wrappers for common workflows
self_training/           # LoRA/retokenizer/self-training experiments
```

### Generated (gitignored)

```
exported_onnx_*/         # ONNX model exports
manifests/               # QAI Hub job manifests
results/                 # validation CSVs
.cache/                  # embedding + preprocessing cache
hf_cache/                # HuggingFace dataset downloads
dataset/                 # local sample data (download separately)
```

## Public artifact boundary

This final repo intentionally excludes private meeting notes, Discord exports, raw datasets, local caches, credentials, checkpoints, and generated ONNX/QAI Hub artifacts. Generated artifacts are either reproducible from the scripts here or referenced by job/artifact IDs in [SUBMISSION.md](SUBMISSION.md).

## Citation

If you use this implementation, cite the upstream model families used by Team Manifold:

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
