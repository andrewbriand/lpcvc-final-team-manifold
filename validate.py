"""
Lightweight validation harness with optional HF matrix + caching.

Default remains the fast path:
  python validate.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import os
import time
import zipfile
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lpcvc_contract import CONTRACT_VERSION, IMAGE_HEIGHT, IMAGE_WIDTH
from lpcvc_models import DEFAULT_MODEL, MODELS

RESULTS_DIR = "results"
DEFAULT_HF_CACHE = "hf_cache"
DEFAULT_LOCAL_CACHE = ".cache/validate"

# dataset_key -> (task_type, source, description)
DATASETS: dict[str, tuple[str, str, str]] = {
    "sample": ("retrieval", "local", "LPCVC sample dataset"),
    "mscoco": ("retrieval", "hf", "MSCOCO Karpathy 5K retrieval"),
    "flickr30k": ("retrieval", "hf", "Flickr30K retrieval benchmark"),
    "sugarcrepe": ("binary", "hf", "SugarCrepe hard-negative benchmark"),
    "sugarcrepe_pp_swap_att": ("itt", "hf", "SugarCrepe++ ITT benchmark swap_atribute"),
    "sugarcrepe_pp_replace_attribute": ("itt", "hf", "SugarCrepe++ ITT benchmark replace_attribute"),
    "sugarcrepe_pp_replace_object": ("itt", "hf", "SugarCrepe++ ITT benchmark replace_object"),
    "sugarcrepe_pp_replace_relation": ("itt", "hf", "SugarCrepe++ ITT benchmark replace_relation"),
    "sugarcrepe_pp_swap_object": ("itt", "hf", "SugarCrepe++ ITT benchmark swap_object"),
}

GROUPS: dict[str, list[str]] = {
    "quick": ["sample", "sugarcrepe"],
    "retrieval": ["sample", "mscoco", "flickr30k"],
    "all": list(DATASETS.keys()),
}

_SUGARCREPE_PP_IMAGE_CACHE: dict[str, dict] = {}


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def contract_preprocess(preprocess):
    def _wrapped(img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BICUBIC)
        return preprocess(img)

    return _wrapped


def contract_preprocess_image_processor(image_processor):
    def _wrapped(img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BICUBIC)
        return image_processor(images=img, return_tensors="pt")["pixel_values"][0]

    return _wrapped


class TripletCLIPWrapper(torch.nn.Module):
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder({"pixel_values": images})

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=token_ids)
        return outputs.text_embeds


class SigLIPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=images)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        # SigLIP2 HF inference paths use fixed-length padded token IDs without
        # an explicit attention mask; passing a hand-built mask changes the embeddings.
        return self.model.get_text_features(input_ids=token_ids)


def _fgclip2_max_num_patches(image: Image.Image) -> int:
    """Match FG-CLIP2 HF model card (patch stride 16)."""
    w, h = image.size
    max_val = (w // 16) * (h // 16)
    if max_val > 784:
        return 1024
    if max_val > 576:
        return 784
    if max_val > 256:
        return 576
    if max_val > 128:
        return 256
    return 128


def _patch_fgclip2_text_embeddings(model: torch.nn.Module) -> None:
    """Fix broken tensors in the HF FG-CLIP2 text embedding layer.

    The released checkpoint has two issues:
    1. ``position_ids`` is registered with persistent=False and loads as garbage.
    2. ``mask1`` / ``mask2`` are plain attributes created as ``torch.zeros()``
       during ``__init__`` but never saved to the checkpoint, so they arrive
       as meta tensors after ``from_pretrained``.
    All three must be reconstructed before any forward pass.
    """
    emb = model.text_model.embeddings
    dev = next(model.parameters()).device
    cfg = model.config.text_config
    longtext_len = cfg.longtext_len
    keep_len = cfg.keep_len

    emb.register_buffer(
        "position_ids",
        torch.arange(longtext_len, device=dev, dtype=torch.long).unsqueeze(0),
        persistent=False,
    )

    mask1 = torch.zeros(longtext_len, 1, device=dev)
    mask1[:keep_len, :] = 1
    mask2 = torch.zeros(longtext_len, 1, device=dev)
    mask2[keep_len:, :] = 1
    emb.mask1 = mask1
    emb.mask2 = mask2


class FgClip2Wrapper(torch.nn.Module):
    """HF FG-CLIP2 uses dict inputs for images and for text (walk_type)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.model = core
        self._lpcvc_image_input = "dict"
        self._lpcvc_text_input = "dict"

    def encode_image(self, inputs: dict) -> torch.Tensor:
        return self.model.get_image_features(**inputs)

    def encode_text(self, inputs: dict) -> torch.Tensor:
        return self.model.get_text_features(**inputs, walk_type="long")


def load_model(model_key: str, device: str):
    spec = MODELS[model_key]
    if spec.loader == "open_clip":
        import open_clip

        print(f"Loading {spec.open_clip_name} [{spec.pretrained}] on {device}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.open_clip_name,
            pretrained=spec.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(spec.open_clip_name)
        model = model.to(device).eval()
        return model, contract_preprocess(preprocess), tokenizer

    if spec.loader == "tripletclip_hf":
        from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizerFast

        if not spec.hf_repo or not spec.hf_base_model:
            raise RuntimeError(f"TripletCLIP model spec is incomplete: {model_key}")

        print(f"Loading TripletCLIP [{spec.hf_repo}] on {device}")
        vision_encoder = AutoModel.from_pretrained(
            spec.hf_repo,
            subfolder="vision-encoder",
            trust_remote_code=True,
        )
        text_encoder = AutoModel.from_pretrained(
            spec.hf_repo,
            subfolder="text-encoder",
            trust_remote_code=True,
        )
        image_processor = CLIPImageProcessor.from_pretrained(spec.hf_base_model)
        text_tokenizer = CLIPTokenizerFast.from_pretrained(spec.hf_base_model)

        def preprocess(img: Image.Image) -> torch.Tensor:
            return image_processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]

        def tokenizer(texts: list[str]) -> torch.Tensor:
            return text_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=spec.text_max_length,
                return_tensors="pt",
            )["input_ids"]

        model = TripletCLIPWrapper(vision_encoder, text_encoder).to(device).eval()
        return model, preprocess, tokenizer

    if spec.loader == "siglip_hf":
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if not spec.hf_repo:
            raise RuntimeError(f"SigLIP model spec is incomplete: {model_key}")

        print(f"Loading SigLIP [{spec.hf_repo}] on {device}")
        base_model = AutoModel.from_pretrained(spec.hf_repo).to(device).eval()
        processor = AutoProcessor.from_pretrained(spec.hf_repo)
        tokenizer_impl = AutoTokenizer.from_pretrained(spec.hf_repo)

        def tokenizer(texts: list[str]) -> torch.Tensor:
            return tokenizer_impl(
                texts,
                padding="max_length",
                truncation=True,
                max_length=spec.text_max_length,
                return_tensors="pt",
            )["input_ids"]

        model = SigLIPWrapper(base_model).to(device).eval()
        preprocess = contract_preprocess_image_processor(processor.image_processor)
        return model, preprocess, tokenizer

    if spec.loader == "fgclip2_hf":
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        if not spec.hf_repo:
            raise RuntimeError(f"FG-CLIP2 model spec is incomplete: {model_key}")

        print(f"Loading FG-CLIP2 [{spec.hf_repo}] on {device}")
        base = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo,
            trust_remote_code=True,
        ).to(device).eval()
        _patch_fgclip2_text_embeddings(base)
        image_processor = AutoImageProcessor.from_pretrained(spec.hf_repo, trust_remote_code=True)
        tokenizer_impl = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=True)

        def preprocess(img: Image.Image):
            pil = img.convert("RGB")
            m = _fgclip2_max_num_patches(pil)
            return image_processor(images=pil, max_num_patches=m, return_tensors="pt")

        def tokenizer(texts: list[str]):
            lowered = [t.lower() for t in texts]
            return tokenizer_impl(
                lowered,
                padding="max_length",
                truncation=True,
                max_length=spec.text_max_length,
                return_tensors="pt",
            )

        return FgClip2Wrapper(base), preprocess, tokenizer

    raise ValueError(f"Unsupported model loader: {spec.loader}")


def _read_csv_body(path: str) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:] if len(rows) > 1 else []


def load_sample(img_csv: str, txt_csv: str, img_dir: str) -> tuple[list[str], list[str], list[list[int]]]:
    txt_rows = _read_csv_body(txt_csv)
    img_rows = _read_csv_body(img_csv)

    text_ids: list[int] = []
    prompts: list[str] = []
    for row in txt_rows:
        if len(row) < 2:
            continue
        text_id_raw, prompt = row[0].strip(), row[1].strip()
        if not text_id_raw or not prompt:
            continue
        try:
            text_id = int(text_id_raw)
        except ValueError:
            continue
        text_ids.append(text_id)
        prompts.append(prompt)

    if not prompts:
        raise RuntimeError(f"No prompts loaded from {txt_csv}")

    text_id_to_idx = {text_id: idx for idx, text_id in enumerate(text_ids)}
    images: list[str] = []
    img2txt: list[list[int]] = []

    for row in img_rows:
        if len(row) < 2:
            continue
        image_name, gt_raw = row[0].strip(), row[1].strip()
        if not image_name or not gt_raw:
            continue

        image_path = os.path.join(img_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image listed in CSV not found: {image_path}")

        gt_indices: list[int] = []
        for token in gt_raw.split(";"):
            token = token.strip()
            if not token:
                continue
            try:
                text_id = int(token)
            except ValueError:
                continue
            mapped = text_id_to_idx.get(text_id)
            if mapped is not None:
                gt_indices.append(mapped)

        if gt_indices:
            images.append(image_path)
            img2txt.append(gt_indices)

    if not images:
        raise RuntimeError("No valid sample items were loaded.")
    return images, prompts, img2txt


def load_mscoco(cache_dir: str):
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        "nlphuji/mscoco_2014_5k_test_image_text_retrieval",
        "test_5k_mscoco_2014.csv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    zip_path = hf_hub_download(
        "nlphuji/mscoco_2014_5k_test_image_text_retrieval",
        "images_mscoco_2014_5k_test.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    images, captions, img2txt, cap_idx = [], [], [], 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f, zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  mscoco", leave=False):
            filename = (row.get("filename") or "").strip()
            filepath = (row.get("filepath") or "").strip()
            raw_caps = (row.get("raw") or "").strip()
            if not filename or not raw_caps:
                continue

            try:
                caps = ast.literal_eval(raw_caps)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(caps, list) or not caps:
                continue

            member = None
            for candidate in (
                filename,
                os.path.join(filepath, filename) if filepath else "",
                os.path.join("images_mscoco_2014_5k_test", filepath, filename) if filepath else "",
                os.path.join("images_mscoco_2014_5k_test", filename),
            ):
                normalized = candidate.replace("\\", "/")
                if normalized and normalized in names:
                    member = normalized
                    break
            if member is None:
                raise FileNotFoundError(f"MSCOCO image not found in archive: {filepath}/{filename}")

            with zf.open(member) as img_f:
                img = Image.open(io.BytesIO(img_f.read())).convert("RGB")

            images.append(img)
            captions.extend(str(cap).strip() for cap in caps if str(cap).strip())
            img2txt.append(list(range(cap_idx, cap_idx + len(caps))))
            cap_idx += len(caps)
    return images, captions, img2txt


def load_flickr30k(cache_dir: str):
    from datasets import load_dataset

    ds = load_dataset("clip-benchmark/wds_flickr30k", split="test", cache_dir=cache_dir)
    images, captions, img2txt = [], [], []
    for ex in tqdm(ds, desc="  flickr30k", leave=False):
        img = ex.get("jpg")
        if img is None:
            continue
        cap = ex.get("txt") or ""
        if isinstance(cap, list):
            cap = cap[0]
        cap_idx = len(captions)
        images.append(img.convert("RGB"))
        captions.append(cap)
        img2txt.append([cap_idx])
    return images, captions, img2txt


def load_sugarcrepe(cache_dir: str):
    from datasets import load_dataset

    ds = load_dataset("mteb/SUGARCREPE_fmt", split="train", cache_dir=cache_dir)
    images, pos_caps, neg_caps = [], [], []
    for ex in tqdm(ds, desc="  sugarcrepe", leave=False):
        img = ex.get("images")
        pos = ex.get("caption") or ""
        neg = ex.get("negative_caption") or ""
        if img is None or not pos or not neg:
            continue
        images.append(img.convert("RGB"))
        pos_caps.append(pos)
        neg_caps.append(neg)
    return images, pos_caps, neg_caps

def load_sugarcrepe_pp(cache_dir: str, subset):
    from datasets import load_dataset

    ds = load_dataset("Aman-J/SugarCrepe_pp", subset, split="train", cache_dir=cache_dir)
    img_name_to_image = _SUGARCREPE_PP_IMAGE_CACHE.get(cache_dir)
    if img_name_to_image is None:
        ds_ms_coco = load_dataset("lmms-lab/COCO-Caption2017", split="val", cache_dir=cache_dir)
        img_name_to_image = {}
        for ex in ds_ms_coco:
            img_name_to_image[ex.get("question_id")] = ex.get("image")
        _SUGARCREPE_PP_IMAGE_CACHE[cache_dir] = img_name_to_image
    images, pos_caps, pos_caps2, neg_caps = [], [], [], []
    for ex in tqdm(ds, desc="  sugarcrepe", leave=False):
        img_filename = ex.get("filename")
        img = img_name_to_image[img_filename]
        pos = ex.get("caption") or ""
        pos2 = ex.get("caption2") or ""
        neg = ex.get("negative_caption") or ""
        if img_filename is None or img is None or not pos or not neg or not pos2:
            continue
        images.append(img.convert("RGB"))
        pos_caps.append(pos)
        pos_caps2.append(pos2)
        neg_caps.append(neg)
    return images, pos_caps, pos_caps2, neg_caps


def resolve_datasets(raw: str) -> list[str]:
    key = raw.strip().lower()
    if key in GROUPS:
        return GROUPS[key]
    keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
    bad = [k for k in keys if k not in DATASETS]
    if bad:
        raise ValueError(f"Unknown dataset keys: {bad}")
    return keys


def list_datasets() -> None:
    print("\nDatasets:")
    for key, (task, source, desc) in DATASETS.items():
        print(f"  - {key:<10} task={task:<9} source={source:<5} {desc}")
    print("\nGroups:")
    for key, value in GROUPS.items():
        print(f"  - {key}: {', '.join(value)}")
    print()


class CacheManager:
    def __init__(self, root_dir: str, enabled: bool, model_sig: str, preprocess_sig: str, token_sig: str):
        self.enabled = enabled
        self.model_sig = model_sig
        self.preprocess_sig = preprocess_sig
        self.token_sig = token_sig
        self.prep_dir = os.path.join(root_dir, "prep")
        self.tok_dir = os.path.join(root_dir, "tok")
        self.img_emb_dir = os.path.join(root_dir, "img_emb")
        self.txt_emb_dir = os.path.join(root_dir, "txt_emb")
        if enabled:
            os.makedirs(self.prep_dir, exist_ok=True)
            os.makedirs(self.tok_dir, exist_ok=True)
            os.makedirs(self.img_emb_dir, exist_ok=True)
            os.makedirs(self.txt_emb_dir, exist_ok=True)

    @staticmethod
    def _hash(*parts: str) -> str:
        h = hashlib.sha1()
        for part in parts:
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        return h.hexdigest()

    @staticmethod
    def image_identity(img) -> str | None:
        if not isinstance(img, str):
            if isinstance(img, Image.Image):
                rgb = img.convert("RGB")
                digest = hashlib.sha1(rgb.tobytes()).hexdigest()
                return f"pil:{rgb.width}x{rgb.height}:{digest}"
            return None
        path = os.path.abspath(img)
        if not os.path.exists(path):
            return None
        st = os.stat(path)
        return f"{path}:{st.st_size}:{st.st_mtime_ns}"

    def _load(self, path: str):
        if not self.enabled or not os.path.exists(path):
            return None
        return np.load(path)

    def _save(self, path: str, arr: np.ndarray) -> None:
        if self.enabled:
            np.save(path, arr)

    def load_preprocessed(self, image_id: str):
        key = self._hash("prep", CONTRACT_VERSION, self.preprocess_sig, image_id)
        return self._load(os.path.join(self.prep_dir, f"{key}.npy"))

    def save_preprocessed(self, image_id: str, arr: np.ndarray) -> None:
        key = self._hash("prep", CONTRACT_VERSION, self.preprocess_sig, image_id)
        self._save(os.path.join(self.prep_dir, f"{key}.npy"), arr.astype(np.float32))

    def load_tokens(self, text: str):
        key = self._hash("tok", self.token_sig, text)
        return self._load(os.path.join(self.tok_dir, f"{key}.npy"))

    def save_tokens(self, text: str, arr: np.ndarray) -> None:
        key = self._hash("tok", self.token_sig, text)
        self._save(os.path.join(self.tok_dir, f"{key}.npy"), arr.astype(np.int64))

    def load_image_embedding(self, image_id: str):
        key = self._hash("img", CONTRACT_VERSION, self.model_sig, image_id)
        return self._load(os.path.join(self.img_emb_dir, f"{key}.npy"))

    def save_image_embedding(self, image_id: str, arr: np.ndarray) -> None:
        key = self._hash("img", CONTRACT_VERSION, self.model_sig, image_id)
        self._save(os.path.join(self.img_emb_dir, f"{key}.npy"), arr.astype(np.float32))

    def load_text_embedding(self, text: str):
        key = self._hash("txt", CONTRACT_VERSION, self.model_sig, text)
        return self._load(os.path.join(self.txt_emb_dir, f"{key}.npy"))

    def save_text_embedding(self, text: str, arr: np.ndarray) -> None:
        key = self._hash("txt", CONTRACT_VERSION, self.model_sig, text)
        self._save(os.path.join(self.txt_emb_dir, f"{key}.npy"), arr.astype(np.float32))


@torch.no_grad()
def _embed_images_fgclip2(model, preprocess, images: list, device: str, batch_size: int, cache: CacheManager | None):
    out: list[np.ndarray | None] = [None] * len(images)
    misses: list[int] = []
    image_ids: list[str | None] = [None] * len(images)

    for idx, image in enumerate(images):
        img_id = cache.image_identity(image) if cache else None
        image_ids[idx] = img_id
        if cache and img_id:
            emb = cache.load_image_embedding(img_id)
            if emb is not None:
                out[idx] = emb
                continue
        misses.append(idx)

    measured_ms = 0.0
    for i in tqdm(range(0, len(misses), batch_size), desc="  imgs", leave=False):
        chunk = misses[i : i + batch_size]
        for idx in chunk:
            item = images[idx]
            if isinstance(item, str):
                with Image.open(item) as img_f:
                    pil = img_f.convert("RGB")
            else:
                pil = item.convert("RGB")
            img_id = cache.image_identity(images[idx]) if cache else None
            image_ids[idx] = img_id
            inputs = preprocess(pil)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            sync_device(device)
            t0 = time.perf_counter()
            feat = model.encode_image(inputs)
            sync_device(device)
            measured_ms += (time.perf_counter() - t0) * 1000.0
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            row = feat.detach().float().cpu().numpy().astype(np.float32)[0]
            out[idx] = row
            if cache and img_id:
                cache.save_image_embedding(img_id, row)

    if any(x is None for x in out):
        raise RuntimeError("Missing image embeddings in output.")
    return np.vstack(out), measured_ms / max(len(images), 1)


@torch.no_grad()
def _embed_texts_fgclip2(model, tokenizer, texts: list[str], device: str, batch_size: int, cache: CacheManager | None):
    out: list[np.ndarray | None] = [None] * len(texts)
    misses: list[int] = []
    for idx, text in enumerate(texts):
        if cache:
            emb = cache.load_text_embedding(text.strip())
            if emb is not None:
                out[idx] = emb
                continue
        misses.append(idx)

    measured_ms = 0.0
    for i in tqdm(range(0, len(misses), batch_size), desc="  text", leave=False):
        chunk = misses[i : i + batch_size]
        batch_enc = tokenizer([texts[idx] for idx in chunk])
        batch_enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_enc.items()}
        sync_device(device)
        t0 = time.perf_counter()
        features = model.encode_text(batch_enc)
        sync_device(device)
        measured_ms += (time.perf_counter() - t0) * 1000.0
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        arr = features.cpu().numpy().astype(np.float32)
        for j, idx in enumerate(chunk):
            out[idx] = arr[j]
            if cache:
                cache.save_text_embedding(texts[idx].strip(), arr[j])

    if any(x is None for x in out):
        raise RuntimeError("Missing text embeddings in output.")
    return np.vstack(out), measured_ms / max(len(texts), 1)


@torch.no_grad()
def embed_images(model, preprocess, images: list, device: str, batch_size: int, cache: CacheManager | None):
    if getattr(model, "_lpcvc_image_input", None) == "dict":
        return _embed_images_fgclip2(model, preprocess, images, device, batch_size, cache)
    out: list[np.ndarray | None] = [None] * len(images)
    misses: list[int] = []
    image_ids: list[str | None] = [None] * len(images)

    def load_tensor(idx: int):
        item = images[idx]
        img_id = image_ids[idx]
        cached = cache.load_preprocessed(img_id) if (cache and img_id) else None
        if cached is not None:
            return torch.from_numpy(cached), img_id

        if isinstance(item, str):
            with Image.open(item) as img:
                tensor = preprocess(img).cpu()
        else:
            tensor = preprocess(item).cpu()

        if cache and img_id:
            cache.save_preprocessed(img_id, tensor.numpy())
        return tensor, img_id

    for idx, image in enumerate(images):
        img_id = cache.image_identity(image) if cache else None
        image_ids[idx] = img_id
        if cache and img_id:
            emb = cache.load_image_embedding(img_id)
            if emb is not None:
                out[idx] = emb
                continue
        misses.append(idx)

    measured_ms = 0.0
    for i in tqdm(range(0, len(misses), batch_size), desc="  imgs", leave=False):
        chunk = misses[i : i + batch_size]
        tensors: list[torch.Tensor] = []
        for idx in chunk:
            tensor, img_id = load_tensor(idx)
            tensors.append(tensor)
            image_ids[idx] = img_id
        batch = torch.stack(tensors).to(device)
        sync_device(device)
        t0 = time.perf_counter()
        features = model.encode_image(batch)
        sync_device(device)
        measured_ms += (time.perf_counter() - t0) * 1000.0
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        arr = features.cpu().numpy().astype(np.float32)
        for j, idx in enumerate(chunk):
            out[idx] = arr[j]
            if cache and image_ids[idx]:
                cache.save_image_embedding(image_ids[idx], arr[j])

    if any(x is None for x in out):
        raise RuntimeError("Missing image embeddings in output.")
    return np.vstack(out), measured_ms / max(len(images), 1)


@torch.no_grad()
def embed_texts(model, tokenizer, texts: list[str], device: str, batch_size: int, cache: CacheManager | None):
    if getattr(model, "_lpcvc_text_input", None) == "dict":
        return _embed_texts_fgclip2(model, tokenizer, texts, device, batch_size, cache)
    out: list[np.ndarray | None] = [None] * len(texts)
    misses: list[int] = []

    def tokenize_one(text: str) -> np.ndarray:
        cached = cache.load_tokens(text) if cache else None
        if cached is not None:
            return cached.astype(np.int64)
        toks = tokenizer([text])
        if isinstance(toks, torch.Tensor):
            arr = toks.cpu().numpy().astype(np.int64)[0]
        else:
            arr = np.asarray(toks).astype(np.int64)[0]
        if cache:
            cache.save_tokens(text, arr)
        return arr

    for idx, text in enumerate(texts):
        key = text.strip()
        if cache:
            emb = cache.load_text_embedding(key)
            if emb is not None:
                out[idx] = emb
                continue
        misses.append(idx)

    measured_ms = 0.0
    for i in tqdm(range(0, len(misses), batch_size), desc="  text", leave=False):
        chunk = misses[i : i + batch_size]
        toks = np.stack([tokenize_one(texts[idx]) for idx in chunk], axis=0).astype(np.int64)
        batch = torch.from_numpy(toks).to(device)
        sync_device(device)
        t0 = time.perf_counter()
        features = model.encode_text(batch)
        sync_device(device)
        measured_ms += (time.perf_counter() - t0) * 1000.0
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        arr = features.cpu().numpy().astype(np.float32)
        for j, idx in enumerate(chunk):
            out[idx] = arr[j]
            if cache:
                cache.save_text_embedding(texts[idx].strip(), arr[j])

    if any(x is None for x in out):
        raise RuntimeError("Missing text embeddings in output.")
    return np.vstack(out), measured_ms / max(len(texts), 1)


def recall_at_k(image_embeddings: np.ndarray, text_embeddings: np.ndarray, img2txt: list[list[int]], ks=(1, 5, 10)):
    sim = image_embeddings @ text_embeddings.T
    out: dict[str, float] = {}
    for k in ks:
        score = 0.0
        for i, gt_indices in enumerate(img2txt):
            hits = 0
            topk = np.argsort(-sim[i])[:k]
            for gt in gt_indices:
                if gt in topk:
                    hits += 1
            score += hits / max(len(gt_indices), 1)
        out[f"R@{k}"] = score / max(len(img2txt), 1)
    return out


def binary_accuracy(img_embs: np.ndarray, pos_embs: np.ndarray, neg_embs: np.ndarray):
    wins = np.einsum("ij,ij->i", img_embs, pos_embs) > np.einsum("ij,ij->i", img_embs, neg_embs)
    return {"accuracy": float(wins.mean())}

def itt_accuracy(img_embs: np.ndarray, pos_embs: np.ndarray, pos_embs2: np.ndarray, neg_embs: np.ndarray):  
    wins = np.einsum("ij,ij->i", img_embs, pos_embs) > np.einsum("ij,ij->i", img_embs, neg_embs)
    wins2 = np.einsum("ij,ij->i", img_embs, pos_embs2) > np.einsum("ij,ij->i", img_embs, neg_embs)
    wins_final = np.logical_and(wins, wins2)
    return {"accuracy": float(wins_final.mean())}


def eval_retrieval(model, preprocess, tokenizer, payload, device, batch_size, cache):
    images, captions, img2txt = payload
    img_embs, img_ms = embed_images(model, preprocess, images, device, batch_size, cache)
    txt_embs, txt_ms = embed_texts(model, tokenizer, captions, device, batch_size, cache)
    scores = recall_at_k(img_embs, txt_embs, img2txt)
    scores["img_ms"] = img_ms
    scores["txt_ms"] = txt_ms
    return scores


def eval_binary(model, preprocess, tokenizer, payload, device, batch_size, cache):
    images, pos_caps, neg_caps = payload
    img_embs, img_ms = embed_images(model, preprocess, images, device, batch_size, cache)
    cap_embs, txt_ms = embed_texts(model, tokenizer, pos_caps + neg_caps, device, batch_size, cache)
    n = len(pos_caps)
    scores = binary_accuracy(img_embs, cap_embs[:n], cap_embs[n:])
    scores["img_ms"] = img_ms
    scores["txt_ms"] = txt_ms
    return scores

def eval_itt(model, preprocess, tokenizer, payload, device, batch_size, cache):
    images, pos_caps, pos_caps2, neg_caps = payload
    img_embs, img_ms = embed_images(model, preprocess, images, device, batch_size, cache)
    cap_embs, txt_ms = embed_texts(model, tokenizer, pos_caps + pos_caps2 + neg_caps, device, batch_size, cache)
    n = len(pos_caps)
    scores = itt_accuracy(img_embs, cap_embs[:n], cap_embs[n:2*n], cap_embs[2*n:])
    scores["img_ms"] = img_ms
    scores["txt_ms"] = txt_ms
    return scores


def save_results(rows: list[dict], output_path: str | None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = output_path or os.path.join(RESULTS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fields = ["timestamp", "model", "dataset", "metric", "value", "contract_version"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight validator with optional HF matrix and cache.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS.keys()))
    parser.add_argument("--datasets", default="sample", help="Dataset key, group, or comma list")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-csv", default="dataset/img_list.csv")
    parser.add_argument("--txt-csv", default="dataset/txt_list.csv")
    parser.add_argument("--img-dir", default="dataset/images")
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE)
    parser.add_argument("--offline", action="store_true", help="Set HF_DATASETS_OFFLINE=1")
    parser.add_argument("--cache-dir", default=DEFAULT_LOCAL_CACHE, help="Local cache for preprocess/tokens/embeddings")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--results-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        list_datasets()
        return

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    dataset_keys = resolve_datasets(args.datasets)
    device = args.device or auto_device()
    spec = MODELS[args.model]
    model, preprocess, tokenizer = load_model(args.model, device)

    preprocess_sig = hashlib.sha1(f"{repr(preprocess)}|{CONTRACT_VERSION}".encode("utf-8")).hexdigest()[:12]
    token_sig = hashlib.sha1(
        f"{args.model}|{spec.loader}|{spec.hf_repo}|{spec.hf_base_model}|{spec.text_max_length}".encode("utf-8")
    ).hexdigest()[:12]
    model_sig = args.model
    if spec.loader == "siglip_hf":
        model_sig = f"{args.model}|siglip_text_nomask_v1"
    elif spec.loader == "fgclip2_hf":
        model_sig = f"{args.model}|fgclip2_long196_nativeres_v4"
    cache = CacheManager(
        root_dir=args.cache_dir,
        enabled=not args.disable_cache,
        model_sig=model_sig,
        preprocess_sig=preprocess_sig,
        token_sig=token_sig,
    )
    if cache.enabled:
        print(f"Cache enabled: {args.cache_dir}")
    else:
        print("Cache disabled")

    rows: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")

    for key in dataset_keys:
        dtype, _source, description = DATASETS[key]
        print(f"\n== {key} ({dtype}) ==")
        print(f"   {description}")
        try:
            if key == "sample":
                payload = load_sample(args.img_csv, args.txt_csv, args.img_dir)
            elif key == "mscoco":
                payload = load_mscoco(args.hf_cache_dir)
            elif key == "flickr30k":
                payload = load_flickr30k(args.hf_cache_dir)
            elif key == "sugarcrepe":
                payload = load_sugarcrepe(args.hf_cache_dir)
            elif key == "sugarcrepe_pp_swap_att":
                payload = load_sugarcrepe_pp(args.hf_cache_dir, "swap_atribute")
            elif key == "sugarcrepe_pp_replace_attribute":
                payload = load_sugarcrepe_pp(args.hf_cache_dir, "replace_attribute")
            elif key == "sugarcrepe_pp_replace_object":
                payload = load_sugarcrepe_pp(args.hf_cache_dir, "replace_object")
            elif key == "sugarcrepe_pp_replace_relation":
                payload = load_sugarcrepe_pp(args.hf_cache_dir, "replace_relation")
            elif key == "sugarcrepe_pp_swap_object":
                payload = load_sugarcrepe_pp(args.hf_cache_dir, "swap_object")
            else:
                raise RuntimeError(f"Unhandled dataset: {key}")

            if dtype == "retrieval":
                scores = eval_retrieval(model, preprocess, tokenizer, payload, device, args.batch_size, cache)
            elif dtype == "binary":
                scores = eval_binary(model, preprocess, tokenizer, payload, device, args.batch_size, cache)
            elif dtype == "itt":
                scores = eval_itt(model, preprocess, tokenizer, payload, device, args.batch_size, cache)
            else:
                raise RuntimeError(f"Unhandled task type: {dtype}")

            for metric, value in scores.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.4f}")
                rows.append(
                    {
                        "timestamp": now,
                        "model": args.model,
                        "dataset": key,
                        "metric": metric,
                        "value": value,
                        "contract_version": CONTRACT_VERSION,
                    }
                )
        except Exception as exc:
            print(f"   SKIP: {exc}")
            rows.append(
                {
                    "timestamp": now,
                    "model": args.model,
                    "dataset": key,
                    "metric": "error",
                    "value": str(exc),
                    "contract_version": CONTRACT_VERSION,
                }
            )

    out_path = save_results(rows, args.results_out)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
