"""Pre-compute FAISS retrieval + filtering for all CC12M images.

Since the text encoder is frozen, FAISS results are identical every epoch.
Pre-computing them once eliminates ~20s/step of redundant work, cutting
total training time by ~70%.

Outputs a directory of .npz shards, each containing:
  - image_indices: indices into the original shard for reconstruction
  - original_embs: (N, D) stabilizer caption embeddings
  - pseudo_embs: (N, D) pseudo-positive caption embeddings
  - hard_neg_embs: (N, n_neg, D) filtered hard-negative embeddings
  - source_keys: list of source key strings (for image loading)

Usage:
    python scripts/precompute_retrieval.py \
        --shards-dir data/cc12m_shards \
        --num-shards 200 \
        --faiss-index data/cc12m_faiss.index \
        --caption-map data/cc12m_captions.npz \
        --output-dir data/precomputed_v3 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.confidence_filter import apply_confidence_filter
from self_training.data_pipeline import list_shard_files
from self_training.faiss_index import load_index, query_index
from self_training.negative_filter import filter_hard_negatives
from validate import load_model


def _chunked_encode(model, tensors, chunk_size=256):
    if tensors.shape[0] <= chunk_size:
        return model.encode_image(tensors)
    chunks = []
    for i in range(0, tensors.shape[0], chunk_size):
        chunks.append(model.encode_image(tensors[i : i + chunk_size]))
    return torch.cat(chunks, dim=0)


def _collate(batch):
    images, metas = [], []
    for img, meta in batch:
        images.append(img)
        metas.append(meta if isinstance(meta, dict) else {})
    return images, metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=200)
    parser.add_argument("--faiss-index", type=Path, required=True)
    parser.add_argument("--caption-map", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="JSON thresholds file. If omitted, uses defaults.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="mobileclip2_s2")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=17)
    parser.add_argument("--min-top1-cosine", type=float, default=0.35)
    parser.add_argument("--min-hard-negatives", type=int, default=2)
    parser.add_argument("--jaccard-threshold", type=float, default=0.7)
    parser.add_argument("--neg-cosine-proximity", type=float, default=0.85)
    parser.add_argument("--own-cosine-threshold", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load thresholds if provided
    if args.thresholds and args.thresholds.exists():
        with open(args.thresholds) as f:
            thresh = json.load(f)
        args.min_top1_cosine = thresh.get("min_top1_cosine", args.min_top1_cosine)
        print(f"Loaded thresholds: top1 >= {args.min_top1_cosine:.4f}")

    # Load model
    print(f"Loading {args.model_key}...")
    model, preprocess, tokenizer = load_model(args.model_key, args.device)
    model.eval()

    # Load caption bank + FAISS
    print("Loading caption bank...")
    data = np.load(args.caption_map, allow_pickle=True)
    caption_embs = data["embeddings"]
    captions = data["captions"].tolist()
    source_keys = data["source_keys"].tolist()
    sk_to_idx = {k: i for i, k in enumerate(source_keys)}
    D = caption_embs.shape[1]

    print("Loading FAISS index...")
    faiss_index = load_index(args.faiss_index)

    # Set up output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stream all shards
    shard_paths = list_shard_files(args.shards_dir, args.num_shards)
    urls = [str(p) for p in shard_paths]
    dataset = (
        wds.WebDataset(urls, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(lambda img: img.convert("RGB"), lambda meta: meta)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        collate_fn=_collate, pin_memory=True,
    )

    total_kept = 0
    total_seen = 0
    shard_idx = 0

    # Accumulate results for saving in chunks
    all_original = []
    all_pseudo = []
    all_hard_neg = []
    all_keys = []
    save_every = 50000  # save a shard every ~50K samples

    print(f"Pre-computing retrieval for {len(shard_paths)} shards...")
    for batch_images, batch_metas in tqdm(loader, desc="Pre-computing"):
        B = len(batch_images)
        total_seen += B

        # Encode images
        image_tensors = torch.stack([preprocess(img) for img in batch_images]).to(args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                image_embs = _chunked_encode(model, image_tensors)
                image_embs = F.normalize(image_embs.float(), dim=-1)
        image_embs_np = image_embs.cpu().numpy().astype(np.float32)

        # Source keys
        batch_source_indices = []
        batch_keys = []
        for meta in batch_metas:
            key = meta.get("__key__", "")
            batch_keys.append(key)
            batch_source_indices.append(sk_to_idx.get(key, -1))

        # Own-caption embeddings
        own_cap = np.zeros((B, D), dtype=np.float32)
        for i, idx in enumerate(batch_source_indices):
            if idx >= 0:
                own_cap[i] = caption_embs[idx]

        # FAISS retrieval
        scores, indices = query_index(
            faiss_index, image_embs_np, k=args.top_k,
            exclude_source_indices=batch_source_indices,
            caption_embs=caption_embs,
            own_caption_embs=own_cap,
            own_cosine_threshold=args.own_cosine_threshold,
        )

        # Confidence filter (single threshold, no margin)
        if scores.shape[1] < 2:
            continue
        top1 = scores[:, 0]
        keep_mask = top1 >= args.min_top1_cosine
        keep_idx = np.where(keep_mask)[0]

        # Per-sample filtering
        for i in keep_idx:
            pp_idx = int(indices[i, 0])
            if pp_idx < 0:
                continue

            pseudo_emb = caption_embs[pp_idx]
            pseudo_caption = captions[pp_idx]

            # Hard negatives
            cand_indices = indices[i, 1:]
            cand_valid = cand_indices[cand_indices >= 0]
            if len(cand_valid) == 0:
                continue

            cand_caps = [captions[int(c)] for c in cand_valid]
            cand_embs = np.stack([caption_embs[int(c)] for c in cand_valid])

            filtered = filter_hard_negatives(
                pseudo_caption, pseudo_emb,
                cand_caps, cand_embs,
                jaccard_threshold=args.jaccard_threshold,
                neg_cosine_proximity=args.neg_cosine_proximity,
                min_negatives=args.min_hard_negatives,
            )

            if len(filtered) < args.min_hard_negatives:
                continue

            # This sample passed — save it
            src_idx = batch_source_indices[i]
            orig_emb = caption_embs[src_idx] if src_idx >= 0 else pseudo_emb

            neg_embs = np.stack([f[1] for f in filtered[:args.min_hard_negatives]])

            all_original.append(orig_emb)
            all_pseudo.append(pseudo_emb)
            all_hard_neg.append(neg_embs)
            all_keys.append(batch_keys[i])
            total_kept += 1

        # Save periodically
        if len(all_original) >= save_every:
            _save_shard(args.output_dir, shard_idx, all_original, all_pseudo,
                        all_hard_neg, all_keys, args.min_hard_negatives)
            shard_idx += 1
            all_original, all_pseudo, all_hard_neg, all_keys = [], [], [], []

    # Save remaining
    if all_original:
        _save_shard(args.output_dir, shard_idx, all_original, all_pseudo,
                    all_hard_neg, all_keys, args.min_hard_negatives)
        shard_idx += 1

    # Save manifest
    manifest = {
        "total_images_seen": total_seen,
        "total_kept": total_kept,
        "retention_rate": total_kept / max(total_seen, 1),
        "num_shards": shard_idx,
        "min_hard_negatives": args.min_hard_negatives,
        "embedding_dim": D,
        "min_top1_cosine": args.min_top1_cosine,
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Kept {total_kept}/{total_seen} samples ({total_kept/max(total_seen,1)*100:.1f}%)")
    print(f"Saved {shard_idx} shards to {args.output_dir}/")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


def _save_shard(output_dir, shard_idx, original, pseudo, hard_neg, keys, n_neg):
    path = output_dir / f"shard_{shard_idx:04d}.npz"
    np.savez_compressed(
        path,
        original_embs=np.stack(original),
        pseudo_embs=np.stack(pseudo),
        hard_neg_embs=np.stack(hard_neg),
        source_keys=np.array(keys, dtype=object),
    )
    print(f"  Saved shard {shard_idx}: {len(original)} samples -> {path}")


if __name__ == "__main__":
    main()
