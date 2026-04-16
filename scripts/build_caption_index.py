"""Encode CC12M captions and build FAISS index.

Run on GH200 (or any GPU machine).

Usage:
    python scripts/build_caption_index.py \
        --shards-dir data/cc12m_shards \
        --num-shards 10 \
        --model-key mobileclip2_s2 \
        --batch-size 512 \
        --output-dir data \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.data_pipeline import extract_captions_from_shards, list_shard_files
from self_training.faiss_index import build_index, save_index
from validate import load_model


def main():
    parser = argparse.ArgumentParser(description="Encode CC12M captions and build FAISS index.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/cc12m_shards"))
    parser.add_argument("--num-shards", type=int, default=None, help="Limit to first N shards (default: all)")
    parser.add_argument("--model-key", default="mobileclip2_s2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Extract captions from shards -----------------------------------
    shard_paths = list_shard_files(args.shards_dir, args.num_shards)
    print(f"Extracting captions from {len(shard_paths)} shards...")
    captions, source_keys = extract_captions_from_shards(shard_paths)
    print(f"Extracted {len(captions)} captions.")

    if len(captions) == 0:
        print("No captions found. Check shard paths.")
        sys.exit(1)

    # ---- 2. Load model (text encoder) --------------------------------------
    print(f"Loading model '{args.model_key}' on {args.device}...")
    model, _, tokenizer = load_model(args.model_key, args.device)
    model.eval()

    # ---- 3. Encode captions in batches -------------------------------------
    print(f"Encoding {len(captions)} captions (batch_size={args.batch_size})...")
    all_embeddings = []

    for start in tqdm(range(0, len(captions), args.batch_size), desc="Encoding"):
        batch_captions = captions[start : start + args.batch_size]
        tokens = tokenizer(batch_captions).to(args.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            text_embs = model.encode_text(tokens)
            text_embs = F.normalize(text_embs.float(), dim=-1)

        all_embeddings.append(text_embs.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # ---- 4. Save caption bank (npz) ----------------------------------------
    npz_path = args.output_dir / "cc12m_captions.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        captions=np.array(captions, dtype=object),
        source_keys=np.array(source_keys, dtype=object),
    )
    print(f"Saved caption bank to {npz_path}")

    # ---- 5. Build and save FAISS index -------------------------------------
    print("Building FAISS index...")
    index = build_index(embeddings)
    index_path = args.output_dir / "cc12m_faiss.index"
    save_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    print("Done.")


if __name__ == "__main__":
    main()
