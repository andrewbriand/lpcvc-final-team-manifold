"""Run exported FG-CLIP2 ONNX on the LPCVC sample set under the contract.

Inputs arrive at the model exactly as the judge would deliver them:
  - Images: PIL → resize 224×224 → float32 / 255 → (1, 3, 224, 224)
  - Text:   OpenAI CLIP BPE tokenizer → pad to 77 → int64 (1, 77)

The ONNX graph internally resizes 224→256 and uses CLIP token IDs to
index FG-CLIP2's 256K-vocab embedding matrix (i.e. reads "wrong" rows).
This script measures the actual R@10 that survives under the contract.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:]


def load_sample(dataset_dir: Path):
    img_rows = _read_csv(dataset_dir / "img_list.csv")
    txt_rows = _read_csv(dataset_dir / "txt_list.csv")

    text_id_to_text = {}
    for row in txt_rows:
        if len(row) < 2:
            continue
        tid, text = row[0].strip(), row[1].strip()
        text_id_to_text[int(tid)] = text
    text_ids_sorted = sorted(text_id_to_text.keys())
    texts = [text_id_to_text[i] for i in text_ids_sorted]
    tid_to_idx = {tid: i for i, tid in enumerate(text_ids_sorted)}

    image_paths: list[Path] = []
    img2txt: list[list[int]] = []
    for row in img_rows:
        if len(row) < 2:
            continue
        fname, tids = row[0].strip(), row[1].strip()
        path = dataset_dir / "images" / fname
        if not path.exists():
            continue
        gt = []
        for token in tids.split(";"):
            token = token.strip()
            if not token:
                continue
            try:
                mapped = tid_to_idx.get(int(token))
            except ValueError:
                continue
            if mapped is not None:
                gt.append(mapped)
        if not gt:
            continue
        image_paths.append(path)
        img2txt.append(gt)
    return image_paths, texts, img2txt


def preprocess_image_contract(img: Image.Image) -> np.ndarray:
    """COMPETITION.md §4.5: resize to 224×224, float32 / 255, (1, 3, 224, 224)."""
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    return arr.astype(np.float32)


def tokenize_contract(texts: list[str]) -> np.ndarray:
    """COMPETITION.md §4.5: CLIPTokenizer openai/clip-vit-base-patch32, max_length=77, int32."""
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tok.add_special_tokens({"cls_token": tok.eos_token})
    out = tok(texts, padding="max_length", truncation=True,
              max_length=77, return_tensors="np")
    return out["input_ids"].astype(np.int64)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=-1, keepdims=True).clip(min=1e-6)
    return arr / n


def recall_at_k(sim: np.ndarray, img2txt: list[list[int]], ks=(1, 5, 10)) -> dict[str, float]:
    out = {}
    n = sim.shape[0]
    for k in ks:
        hit = 0
        top_k = np.argsort(-sim, axis=1)[:, :k]
        for i in range(n):
            if any(t in img2txt[i] for t in top_k[i]):
                hit += 1
        out[f"R@{k}"] = hit / n
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="dataset", type=Path)
    args = parser.parse_args()

    image_paths, texts, img2txt = load_sample(args.dataset)
    print(f"Loaded {len(image_paths)} images, {len(texts)} texts")

    print("Loading ONNX sessions...")
    image_sess = ort.InferenceSession(
        str(args.onnx_dir / "image_encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    text_sess = ort.InferenceSession(
        str(args.onnx_dir / "text_encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # Image embeddings
    img_embs = []
    t0 = time.time()
    for p in image_paths:
        arr = preprocess_image_contract(Image.open(p))
        e = image_sess.run(None, {"image": arr})[0]
        img_embs.append(e[0])
    img_embs = l2_normalize(np.stack(img_embs))
    print(f"Image embeddings: {img_embs.shape}  ({time.time() - t0:.1f}s)")

    # Text embeddings
    t0 = time.time()
    token_ids = tokenize_contract(texts)
    txt_embs = []
    for i in range(len(texts)):
        ids = token_ids[i:i + 1]
        e = text_sess.run(None, {"text": ids})[0]
        txt_embs.append(e[0])
    txt_embs = l2_normalize(np.stack(txt_embs))
    print(f"Text embeddings: {txt_embs.shape}  ({time.time() - t0:.1f}s)")

    # Similarity + R@k
    sim = img_embs @ txt_embs.T
    metrics = recall_at_k(sim, img2txt)
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
