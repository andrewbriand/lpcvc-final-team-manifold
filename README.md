# LPCVC 2026 Track 1 — Image-to-Text Retrieval

Lightweight evaluation and submission harness for the [2026 LPCVC Track 1](https://lpcv.ai/2026LPCVC/image-text-retrieval/) competition. Built around MobileCLIP models via `open_clip`.

## Quick start

```bash
pip install -r requirements.txt
```

Download the [sample dataset](https://drive.google.com/drive/folders/1tTwrehwLtVtMOTjD5beYDC3lcWIfzS5D) and place it at `dataset/` (images + CSVs).

Run local validation:

```bash
./scripts/validate.sh
```

That's it. By default this evaluates `mobileclip2_s2` on the LPCVC sample set and prints Recall@1/5/10.
For retrieval datasets, this harness reports fractional multi-ground-truth recall so local validation stays closer to the LPCVC competition scorer.

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
| `mobileclip2_s2` | MobileCLIP2-S2 | v2, current submission default |
| `mobileclip2_s3` | MobileCLIP2-S3 | v2, ~70M |
| `mobileclip2_s4` | MobileCLIP2-S4 | v2, strongest |
| `mobileclip2_b` | MobileCLIP2-B | v2, ~110M |
| `mobileclip2_l14` | MobileCLIP2-L-14 | v2, largest |
| `vit_b16` | ViT-B/16 | OpenAI CLIP baseline |
| `vit_l14` | ViT-L/14 | OpenAI CLIP large baseline |
| `tripletclip_cc12m` | TripletCLIP | ViT-B/32 on CC12M (HF) |
| `siglip2_base_224` | SigLIP2 Base | Google SigLIP2 patch16 224 (HF) |
| `siglip2_giant_256` | SigLIP2 Giant 1B | Google SigLIP2 giant patch16 256 (HF) |

SigLIP2 runs are supported in the local validation harness. They use the model-native Gemma tokenizer with `max_length=64`, so they are useful for comparison experiments but are not drop-in compatible with the LPCVC submission contract in `lpcvc_contract.py`.

## QAI Hub submission pipeline

These steps compile and deploy to Qualcomm XR2 Gen 2 via [QAI Hub](https://aihub.qualcomm.com). Requires a QAI Hub account.

### Step by step

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
python export_onnx.py --model mobileclip2_s2

python compile_and_profile.py \
  --onnx-dir exported_onnx_mobileclip2_s2 \
  --manifest-out manifests/compile_manifest.json \
  --skip-profile

python upload_dataset.py \
  --manifest-out manifests/upload_manifest.json

python inference.py \
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

## Competition contract

| Field | Value |
|-------|-------|
| Image input | `float32 (1, 3, 224, 224)` in [0, 1], RGB |
| Text input | `int32 (1, 77)` — CLIP token IDs |
| Tokenizer | `openai/clip-vit-base-patch32` |
| Target device | XR2 Gen 2 (Proxy) |
| Metric | Recall@Top10 |
| Latency gate | Image + text encoder < 35 ms combined |
