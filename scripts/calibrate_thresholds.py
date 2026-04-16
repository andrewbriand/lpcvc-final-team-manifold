"""Calibrate confidence-filter thresholds from image-caption retrieval scores.

Streams images from the first shard, encodes them, queries the FAISS index,
and uses the resulting score distribution to set min_top1_cosine and min_margin.

Run on GH200 (or any GPU machine).

Usage:
    python scripts/calibrate_thresholds.py \
        --shards-dir data/cc12m_shards \
        --faiss-index data/cc12m_faiss.index \
        --caption-map data/cc12m_captions.npz \
        --model-key mobileclip2_s2 \
        --num-samples 2000 \
        --output logs/self_train/thresholds.json \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.confidence_filter import calibrate_thresholds
from self_training.data_pipeline import create_image_dataset, list_shard_files
from self_training.faiss_index import load_index, query_index
from validate import load_model


def main():
    parser = argparse.ArgumentParser(description="Calibrate confidence-filter thresholds.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/cc12m_shards"))
    parser.add_argument("--faiss-index", type=Path, default=Path("data/cc12m_faiss.index"))
    parser.add_argument("--caption-map", type=Path, default=Path("data/cc12m_captions.npz"))
    parser.add_argument("--model-key", default="mobileclip2_s2")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of images to sample for calibration")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=17)
    parser.add_argument("--top1-percentile", type=int, default=50)
    parser.add_argument("--margin-percentile", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("logs/self_train/thresholds.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load model -----------------------------------------------------
    print(f"Loading model '{args.model_key}' on {args.device}...")
    model, preprocess, _ = load_model(args.model_key, args.device)
    model.eval()

    # ---- 2. Load FAISS index and caption bank ------------------------------
    print("Loading FAISS index...")
    faiss_index = load_index(args.faiss_index)

    print("Loading caption bank...")
    data = np.load(args.caption_map, allow_pickle=True)
    caption_embs = data["embeddings"].astype(np.float32)
    source_keys = list(data["source_keys"])
    sk_to_idx = {k: i for i, k in enumerate(source_keys)}

    # ---- 3. Stream images from first shard ---------------------------------
    shard_paths = list_shard_files(args.shards_dir, num_shards=1)
    if not shard_paths:
        print("No shards found. Check --shards-dir.")
        sys.exit(1)

    dataset = create_image_dataset(shard_paths, shuffle=0)  # no shuffle for reproducibility

    print(f"Collecting top-1/top-2 scores from up to {args.num_samples} images...")
    all_top1 = []
    all_top2 = []
    collected = 0

    # Process in mini-batches
    batch_images = []
    batch_metas = []

    for sample in tqdm(dataset, total=args.num_samples, desc="Calibrating"):
        img, meta = sample
        batch_images.append(preprocess(img))
        batch_metas.append(meta if isinstance(meta, dict) else {})

        if len(batch_images) < args.batch_size:
            if collected + len(batch_images) < args.num_samples:
                continue

        # Encode batch
        image_tensor = torch.stack(batch_images).to(args.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            image_embs_t = model.encode_image(image_tensor)
            image_embs_t = F.normalize(image_embs_t.float(), dim=-1)

        image_embs_np = image_embs_t.cpu().numpy().astype(np.float32)
        B = image_embs_np.shape[0]

        # Build source indices
        source_indices = []
        own_embs = np.zeros((B, caption_embs.shape[1]), dtype=np.float32)
        for i, meta in enumerate(batch_metas):
            key = meta.get("__key__", "")
            idx = sk_to_idx.get(key, -1)
            source_indices.append(idx)
            if idx >= 0:
                own_embs[i] = caption_embs[idx]

        # Query
        scores, _ = query_index(
            faiss_index, image_embs_np,
            k=args.top_k,
            exclude_source_indices=source_indices,
            caption_embs=caption_embs,
            own_caption_embs=own_embs,
            own_cosine_threshold=0.99,
        )

        if scores.shape[1] >= 2:
            all_top1.append(scores[:, 0])
            all_top2.append(scores[:, 1])

        collected += B
        batch_images = []
        batch_metas = []

        if collected >= args.num_samples:
            break

    if not all_top1:
        print("No scores collected. Something went wrong.")
        sys.exit(1)

    top1_scores = np.concatenate(all_top1)
    top2_scores = np.concatenate(all_top2)
    print(f"Collected {len(top1_scores)} score pairs.")

    # ---- 4. Calibrate thresholds -------------------------------------------
    thresholds = calibrate_thresholds(
        top1_scores, top2_scores,
        top1_percentile=args.top1_percentile,
        margin_percentile=args.margin_percentile,
    )

    print(f"Calibrated thresholds:")
    print(f"  min_top1_cosine: {thresholds['min_top1_cosine']:.6f}")
    print(f"  min_margin:      {thresholds['min_margin']:.6f}")

    # Add some distribution stats for reference
    thresholds["_stats"] = {
        "num_samples": int(len(top1_scores)),
        "top1_mean": float(top1_scores.mean()),
        "top1_std": float(top1_scores.std()),
        "top1_p25": float(np.percentile(top1_scores, 25)),
        "top1_p50": float(np.percentile(top1_scores, 50)),
        "top1_p75": float(np.percentile(top1_scores, 75)),
        "margin_mean": float((top1_scores - top2_scores).mean()),
        "margin_std": float((top1_scores - top2_scores).std()),
    }

    with open(args.output, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
