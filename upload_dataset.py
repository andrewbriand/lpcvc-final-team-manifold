"""
Prepare and upload LPCVC sample datasets to QAI Hub.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import qai_hub
from PIL import Image
from transformers import CLIPTokenizer

from lpcvc_contract import (
    CONTRACT_VERSION,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    TEXT_DTYPE,
    TEXT_SEQ_LEN,
    TOKENIZER_ID,
)


def process_image(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[np.newaxis, :]


def read_csv_body(path: str) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return []
    return rows[1:]


def load_images_by_csv_order(img_csv: str, img_dir: str, target_size: tuple[int, int]) -> tuple[list[np.ndarray], list[str]]:
    rows = read_csv_body(img_csv)
    tensors: list[np.ndarray] = []
    paths: list[str] = []
    for row in rows:
        if not row:
            continue
        image_name = row[0].strip()
        if not image_name:
            continue
        image_path = os.path.join(img_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image listed in CSV not found: {image_path}")
        tensors.append(process_image(image_path, target_size))
        paths.append(image_path)
    return tensors, paths


def load_tokenized_prompts(txt_csv: str) -> tuple[list[np.ndarray], list[int], list[str]]:
    rows = read_csv_body(txt_csv)
    text_ids: list[int] = []
    prompts: list[str] = []
    for row in rows:
        if len(row) < 2:
            continue
        text_id_raw = row[0].strip()
        prompt = row[1].strip()
        if not text_id_raw or not prompt:
            continue
        try:
            text_id = int(text_id_raw)
        except ValueError:
            continue
        text_ids.append(text_id)
        prompts.append(prompt)

    if not prompts:
        return [], [], []

    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_ID)
    token_matrix = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=TEXT_SEQ_LEN,
        return_tensors="np",
    )["input_ids"].astype(np.int32)
    tokens = [row[np.newaxis, :] for row in token_matrix]
    return tokens, text_ids, prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload LPCVC sample datasets to QAI Hub.")
    parser.add_argument("--img-dir", default="dataset/images")
    parser.add_argument("--img-csv", default="dataset/img_list.csv")
    parser.add_argument("--txt-csv", default="dataset/txt_list.csv")
    parser.add_argument("--manifest-out", default="manifests/upload_manifest.json")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_tensors, image_paths = load_images_by_csv_order(
        img_csv=args.img_csv,
        img_dir=args.img_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    text_tensors, text_ids, prompts = load_tokenized_prompts(args.txt_csv)

    if not image_tensors:
        raise RuntimeError("No images loaded from img CSV.")
    if not text_tensors:
        raise RuntimeError("No text prompts loaded from txt CSV.")

    print(f"Loaded images: {len(image_tensors)}")
    print(f"Loaded texts:  {len(text_tensors)}")
    print(f"First image shape/dtype: {image_tensors[0].shape} / {image_tensors[0].dtype}")
    print(f"First text shape/dtype:  {text_tensors[0].shape} / {text_tensors[0].dtype}")

    image_dataset_id = None
    text_dataset_id = None
    if not args.skip_upload:
        image_dataset = qai_hub.upload_dataset({"image": image_tensors})
        text_dataset = qai_hub.upload_dataset({"text": text_tensors})
        image_dataset_id = image_dataset.dataset_id
        text_dataset_id = text_dataset.dataset_id
        print(f"Uploaded image dataset ID: {image_dataset_id}")
        print(f"Uploaded text dataset ID:  {text_dataset_id}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "contract": {
            "image_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
            "text_shape": [1, TEXT_SEQ_LEN],
            "text_dtype": TEXT_DTYPE,
            "tokenizer_id": TOKENIZER_ID,
        },
        "inputs": {
            "img_dir": args.img_dir,
            "img_csv": args.img_csv,
            "txt_csv": args.txt_csv,
        },
        "counts": {
            "images": len(image_tensors),
            "texts": len(text_tensors),
        },
        "dataset_ids": {
            "image": image_dataset_id,
            "text": text_dataset_id,
        },
        "preview": {
            "first_image_path": image_paths[0],
            "first_text_id": int(text_ids[0]),
            "first_text": prompts[0],
        },
    }

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote upload manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
