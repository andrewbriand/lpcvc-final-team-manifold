"""Schall Stage 1 — multi-positive InfoNCE variant.

Identical to schall_stage1.py except for the contrastive loss formulation:
each image carries K=5 positive captions (the dataset's full caption list)
instead of being broken into K (img, cap) pairs. Multi-positive InfoNCE then
sums exp(score) over all K positives in the numerator and over the full
B*K text pool in the denominator. Symmetric direction: each text has 1
positive image, all B images in the denominator.

Reference: SupCon (Khosla et al. 2020). The hypothesis is that COCO/Flickr's
5-cap-per-image structure provides enough redundancy that single-positive
InfoNCE wastes signal — multi-positive should reduce gradient noise.

All other hyperparams (anchor weight 1.0, LoRA rank 16, fixed logit_scale 20.0,
attn-only LoRA, fixed-224 contract, no augmentation, MSE/cosine representation
anchor) are unchanged from v1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lpcvc_contract import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from self_training.loss import kl_anchor_loss  # noqa: E402
from self_training.retokenizer_model import FgClip2Retokenizer  # noqa: E402
from validate import _patch_fgclip2_text_embeddings  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage1MultiposConfig:
    # I/O
    retokenizer_checkpoint: str
    out_dir: str
    hf_cache_dir: str = "hf_cache"
    log_dir: str = "logs/schall_stage1_multipos"

    # Model
    fgclip2_repo: str = "qihoo360/fg-clip2-base"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_vision_path_tokens: tuple[str, ...] = (
        "vision_model",
        "vision_tower",
        "visual",
    )
    lora_attn_module_tokens: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "qkv",
        "proj",
    )

    use_coco: bool = True
    use_flickr: bool = True
    use_cc12m: bool = False
    cc12m_max_samples: int = 0
    seed: int = 0

    # Multi-positive contract: each image carries this many captions.
    # Lists shorter than K are padded by repeating the last caption.
    captions_per_image: int = 5

    batch_size: int = 128
    epochs: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    anchor_weight: float = 1.0
    fixed_logit_scale: float = 20.0
    num_workers: int = 4
    bf16: bool = True
    augment_train: bool = False

    # Eval
    eval_each_epoch: bool = True
    eval_datasets: str = "mscoco,flickr30k"

    # Misc
    log_every: int = 25
    save_every_epoch: bool = True


# ---------------------------------------------------------------------------
# Image processing — identical to v1
# ---------------------------------------------------------------------------
class FgClip2ImageBatcher:
    """Wraps the FG-CLIP 2 image processor for fixed-224 batched training."""

    def __init__(self, image_processor, augment: bool = False):
        self.image_processor = image_processor
        self.augment = augment
        self._train_transform = None
        if augment:
            from torchvision import transforms

            self._train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    (IMAGE_HEIGHT, IMAGE_WIDTH),
                    scale=(0.85, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1
                ),
            ])

    def __call__(self, images: list[Image.Image]) -> dict:
        if self.augment and self._train_transform is not None:
            pils = [self._train_transform(img.convert("RGB")) for img in images]
        else:
            pils = [
                img.convert("RGB").resize(
                    (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BICUBIC
                )
                for img in images
            ]
        out = self.image_processor(
            images=pils, max_num_patches=256, return_tensors="pt"
        )
        return out


# ---------------------------------------------------------------------------
# Multi-positive datasets: emit (img, list[str]) — captions grouped by image
# ---------------------------------------------------------------------------
class ImageMultiCaption(Dataset):
    """Image-with-multiple-captions dataset for multi-positive training.

    Each item is (img, list_of_captions). Caption list length is variable
    in the raw data; the collate fn pads to K.
    """

    def __init__(self, samples: list[tuple]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, caps = self.samples[idx]
        return img.convert("RGB"), list(caps)


def make_collate_multipos(captions_per_image: int):
    """Returns collate_fn that pads each image's caption list to K."""

    def _collate(batch):
        images = [b[0] for b in batch]
        captions_per_img: list[list[str]] = []
        for _, caps in batch:
            caps = [c for c in caps if isinstance(c, str) and c.strip()]
            if not caps:
                caps = [""]
            if len(caps) >= captions_per_image:
                caps = caps[:captions_per_image]
            else:
                # repeat the last caption to pad
                last = caps[-1]
                while len(caps) < captions_per_image:
                    caps.append(last)
            captions_per_img.append(caps)
        return images, captions_per_img

    return _collate


def _coerce_caption_list(cap_field) -> list[str]:
    """Normalize a heterogeneous caption field into a list[str]."""
    out: list[str] = []
    if cap_field is None:
        return out
    if isinstance(cap_field, str):
        if cap_field.strip():
            out.append(cap_field.strip())
        return out
    if isinstance(cap_field, list):
        for c in cap_field:
            if isinstance(c, dict):
                c = c.get("raw") or c.get("caption") or ""
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
        return out
    if isinstance(cap_field, dict):
        c = cap_field.get("raw") or cap_field.get("caption") or ""
        if isinstance(c, str) and c.strip():
            out.append(c.strip())
        return out
    return out


def load_coco_train_multipos(cache_dir: str) -> list[tuple]:
    """COCO train as (img, list[str]) tuples — keeps all 5 captions per image."""
    from datasets import load_dataset

    attempts = [
        ("lmms-lab/COCO-Caption2017", "val"),
        ("lmms-lab/COCO-Caption2017", "test"),
        ("yerevann/coco-karpathy", "train"),
    ]
    last_err = None
    ds = None
    for repo, split in attempts:
        try:
            print(f"[data] trying {repo} split={split} ...")
            ds = load_dataset(repo, split=split, cache_dir=cache_dir)
            break
        except Exception as e:
            last_err = e
            print(f"  failed: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"All COCO train sources failed; last error: {last_err}")

    samples: list[tuple] = []
    cap_count_hist: dict[int, int] = {}
    for ex in tqdm(ds, desc="  coco", leave=False):
        img = ex.get("image") or ex.get("jpg")
        cap_field = (
            ex.get("answer")
            or ex.get("caption")
            or ex.get("captions")
            or ex.get("sentences")
            or ex.get("text")
        )
        if img is None or not cap_field:
            continue
        caps = _coerce_caption_list(cap_field)
        if not caps:
            continue
        cap_count_hist[len(caps)] = cap_count_hist.get(len(caps), 0) + 1
        samples.append((img, caps))
    print(f"[data] coco multipos images: {len(samples):,}")
    print(f"[data] coco caps-per-image histogram: {dict(sorted(cap_count_hist.items()))}")
    return samples


def load_flickr_train_multipos(cache_dir: str) -> list[tuple]:
    """Flickr30K train as (img, list[str]) tuples.

    wds_flickr30k schema (verified): one row per image, `txt` field is a
    SINGLE STRING with 5 captions joined by newline. So we split on '\n'.
    Falls back gracefully for list-shaped or single-string-no-newline schemas.
    """
    from datasets import load_dataset

    print("[data] loading Flickr30K train (multipos) ...")
    try:
        ds = load_dataset(
            "clip-benchmark/wds_flickr30k", split="train", cache_dir=cache_dir
        )
    except Exception as e:
        print(f"[data] flickr30k train load failed: {e}; trying 'val' ...")
        ds = load_dataset(
            "clip-benchmark/wds_flickr30k", split="val", cache_dir=cache_dir
        )

    samples: list[tuple] = []
    # Detect schema from first row.
    first = ds[0]
    cap_field0 = first.get("txt") or first.get("caption") or ""
    schema = "list" if isinstance(cap_field0, list) else (
        "newline" if isinstance(cap_field0, str) and "\n" in cap_field0 else "single"
    )
    print(f"[data] flickr schema detected: {schema}")

    if schema == "single":
        # one row per (img, single_caption) — group by __key__
        groups: dict = {}
        order: list = []
        for ex in tqdm(ds, desc="  flickr-group", leave=False):
            img = ex.get("jpg") or ex.get("image")
            cap = ex.get("txt") or ex.get("caption") or ""
            if img is None or not cap:
                continue
            key = ex.get("__key__") or id(img)
            if key not in groups:
                groups[key] = [img, []]
                order.append(key)
            groups[key][1].extend(_coerce_caption_list(cap))
        for key in order:
            img, caps = groups[key]
            if img is None or not caps:
                continue
            samples.append((img, caps))
    else:
        for ex in tqdm(ds, desc="  flickr", leave=False):
            img = ex.get("jpg") or ex.get("image")
            cap = ex.get("txt") or ex.get("caption") or ""
            if img is None or not cap:
                continue
            if isinstance(cap, list):
                caps = _coerce_caption_list(cap)
            else:
                # newline-joined string -> split into list
                caps = [c.strip() for c in str(cap).split("\n") if c.strip()]
            if not caps:
                continue
            samples.append((img, caps))

    print(f"[data] flickr multipos images: {len(samples):,}")
    if samples:
        lens = [len(c) for _, c in samples]
        print(f"[data] flickr caps-per-image: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.2f}")
    return samples


def load_cc12m_subset_multipos(cache_dir: str, max_samples: int) -> list[tuple]:
    """CC12M streamed: each image has only 1 caption, so list of length 1."""
    from datasets import load_dataset

    print(f"[data] streaming CC12M (multipos, K=1), target {max_samples:,} ...")
    ds = load_dataset(
        "pixparse/cc12m-wds", split="train", streaming=True, cache_dir=cache_dir
    )
    samples: list[tuple] = []
    for ex in ds:
        img = ex.get("jpg") or ex.get("image")
        cap = ex.get("txt") or ""
        if img is None or not cap:
            continue
        samples.append((img, [str(cap).strip()]))
        if len(samples) >= max_samples:
            break
    print(f"[data] cc12m streamed images: {len(samples):,}")
    return samples


# ---------------------------------------------------------------------------
# LoRA targeting — identical to v1
# ---------------------------------------------------------------------------
def find_lora_targets(model: torch.nn.Module, cfg: Stage1MultiposConfig) -> list[str]:
    candidates: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(tok in name for tok in cfg.lora_vision_path_tokens):
            continue
        last = name.split(".")[-1]
        if last not in cfg.lora_attn_module_tokens:
            continue
        candidates.append(name)
    return candidates


# ---------------------------------------------------------------------------
# Multi-positive InfoNCE
# ---------------------------------------------------------------------------
def multipos_infonce(
    image_embs: torch.Tensor,    # (B, D), L2-normalized
    text_embs: torch.Tensor,     # (B*K, D), L2-normalized
    K: int,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric multi-positive InfoNCE.

    image_embs[i] has K positives at text_embs[i*K : (i+1)*K], all other
    text indices are negatives.
    text_embs[j] has 1 positive image at image_embs[j // K], all other
    image indices are negatives.

    Returns (i2t_loss, t2i_loss).
    """
    B = image_embs.shape[0]
    BK = text_embs.shape[0]
    assert BK == B * K, f"shape mismatch B={B} K={K} BK={BK}"

    logits_i2t = logit_scale * (image_embs @ text_embs.t())  # (B, B*K)
    # Mask of positives for each image: shape (B, B*K), True at [i, i*K..(i+1)*K]
    pos_mask = torch.zeros(B, BK, dtype=torch.bool, device=image_embs.device)
    rows = torch.arange(B, device=image_embs.device).unsqueeze(1).expand(B, K)
    cols = torch.arange(K, device=image_embs.device).unsqueeze(0).expand(B, K) \
           + torch.arange(B, device=image_embs.device).unsqueeze(1) * K
    pos_mask[rows, cols] = True

    # log sum exp over all texts (denominator)
    log_denom_i2t = torch.logsumexp(logits_i2t, dim=1)  # (B,)
    # log sum exp over positive texts (numerator)
    neg_inf = torch.finfo(logits_i2t.dtype).min
    pos_logits_i2t = logits_i2t.masked_fill(~pos_mask, neg_inf)
    log_num_i2t = torch.logsumexp(pos_logits_i2t, dim=1)  # (B,)
    loss_i2t = (log_denom_i2t - log_num_i2t).mean()

    # text-to-image: each text has exactly 1 positive image (text j -> image j//K)
    logits_t2i = logits_i2t.t()  # (B*K, B)
    text_labels = torch.arange(BK, device=image_embs.device) // K  # (B*K,)
    loss_t2i = F.cross_entropy(logits_t2i, text_labels)

    return loss_i2t, loss_t2i


# ---------------------------------------------------------------------------
# Eval — identical to v1 (eval is single-caption retrieval anyway)
# ---------------------------------------------------------------------------
def quick_inscript_eval(
    base, retokenizer, image_batcher, device: str, cfg: Stage1MultiposConfig,
    n_max: int = 5000,
) -> dict:
    from validate import load_mscoco

    images, captions, img2txt = load_mscoco(cfg.hf_cache_dir)
    if len(images) > n_max:
        images = images[:n_max]
        cap_keep = set()
        new_img2txt = []
        for gt in img2txt[:n_max]:
            new_img2txt.append(gt)
            cap_keep.update(gt)
        img2txt = new_img2txt
        keep_sorted = sorted(cap_keep)
        idx_remap = {old: new for new, old in enumerate(keep_sorted)}
        captions = [captions[i] for i in keep_sorted]
        img2txt = [[idx_remap[i] for i in gt] for gt in img2txt]

    base.eval()
    img_embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), cfg.batch_size), desc="  eval-img", leave=False):
            chunk = images[i : i + cfg.batch_size]
            inp = image_batcher(chunk)
            inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            feat = base.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            img_embs.append(feat.float().cpu().numpy())
    image_embeddings = np.vstack(img_embs).astype(np.float32)

    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt_embs: list[np.ndarray] = []
    retokenizer.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), cfg.batch_size), desc="  eval-txt", leave=False):
            chunk = captions[i : i + cfg.batch_size]
            ids = tok(chunk, padding="max_length", truncation=True, max_length=77,
                      return_tensors="pt")["input_ids"].to(torch.long).to(device)
            feat = retokenizer(ids)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            txt_embs.append(feat.float().cpu().numpy())
    text_embeddings = np.vstack(txt_embs).astype(np.float32)

    sim = image_embeddings @ text_embeddings.T
    metrics = {}
    for k in (1, 5, 10):
        score = 0.0
        for i, gt in enumerate(img2txt):
            topk = np.argsort(-sim[i])[:k]
            hits = sum(1 for g in gt if g in topk)
            score += hits / max(len(gt), 1)
        metrics[f"R@{k}"] = score / max(len(img2txt), 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retokenizer-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--hf-cache-dir", type=str, default="hf_cache")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--captions-per-image", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--fixed-logit-scale", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-coco", action="store_true")
    parser.add_argument("--no-flickr", action="store_true")
    parser.add_argument("--cc12m-max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-n-images", type=int, default=5000)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick 50-step smoke run for sanity.")
    args = parser.parse_args()

    cfg = Stage1MultiposConfig(
        retokenizer_checkpoint=args.retokenizer_checkpoint,
        out_dir=args.out_dir,
        hf_cache_dir=args.hf_cache_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        captions_per_image=args.captions_per_image,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        anchor_weight=args.anchor_weight,
        fixed_logit_scale=args.fixed_logit_scale,
        num_workers=args.num_workers,
        bf16=not args.no_bf16,
        use_coco=not args.no_coco,
        use_flickr=not args.no_flickr,
        use_cc12m=args.cc12m_max_samples > 0,
        cc12m_max_samples=args.cc12m_max_samples,
        seed=args.seed,
        log_every=args.log_every,
        eval_each_epoch=not args.no_eval,
        augment_train=not args.no_augment,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")
    print(f"[setup] config = {json.dumps(asdict(cfg), default=str, indent=2)}")

    # ---- Model
    from transformers import AutoImageProcessor, AutoModelForCausalLM

    print(f"[load] FG-CLIP 2 base from {cfg.fgclip2_repo} ...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.fgclip2_repo, trust_remote_code=True
    )
    _patch_fgclip2_text_embeddings(base)
    base = base.to(device)

    image_processor = AutoImageProcessor.from_pretrained(
        cfg.fgclip2_repo, trust_remote_code=True
    )
    train_batcher = FgClip2ImageBatcher(image_processor, augment=cfg.augment_train)
    eval_batcher = FgClip2ImageBatcher(image_processor, augment=False)
    print(f"[batcher] train augment={cfg.augment_train}, eval augment=False")

    # ---- Frozen retokenizer
    print(f"[load] retokenizer from {cfg.retokenizer_checkpoint} ...")
    ckpt = torch.load(cfg.retokenizer_checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = (ckpt.get("trainable_state_dict") or {}).get("config", {})
    with_adapter = bool(ckpt_cfg.get("with_adapter", False))
    retokenizer = FgClip2Retokenizer(base, with_adapter=with_adapter).to(device).eval()
    retokenizer.load_trainable_state_dict(ckpt["trainable_state_dict"])
    for p in retokenizer.parameters():
        p.requires_grad = False

    # ---- LoRA
    targets = find_lora_targets(base, cfg)
    if not targets:
        raise RuntimeError(
            "Could not find LoRA target modules. Inspect model.named_modules() "
            "and update Stage1MultiposConfig.lora_*_tokens."
        )
    print(f"[lora] targeting {len(targets)} modules")
    for t in targets[:8]:
        print(f"   - {t}")
    if len(targets) > 8:
        print(f"   ... ({len(targets) - 8} more)")

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    for name, p in base.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False

    # ---- Data
    samples: list[tuple] = []
    if cfg.use_coco:
        samples += load_coco_train_multipos(cfg.hf_cache_dir)
    if cfg.use_flickr:
        samples += load_flickr_train_multipos(cfg.hf_cache_dir)
    if cfg.use_cc12m:
        samples += load_cc12m_subset_multipos(cfg.hf_cache_dir, cfg.cc12m_max_samples)
    if not samples:
        raise RuntimeError("No training images loaded.")
    random.shuffle(samples)
    cap_lens = [len(c) for _, c in samples]
    print(f"[data] total images = {len(samples):,}, "
          f"caps-per-image min={min(cap_lens)} max={max(cap_lens)} "
          f"mean={sum(cap_lens)/len(cap_lens):.2f}, K={cfg.captions_per_image}")

    if args.smoke:
        samples = samples[: cfg.batch_size * 50]
        print(f"[smoke] truncated to {len(samples)} images")

    K = cfg.captions_per_image
    ds = ImageMultiCaption(samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_multipos(K),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---- Tokenizer
    from transformers import CLIPTokenizer
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # ---- Optimizer
    opt_params = [p for p in base.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = max(len(loader) * cfg.epochs, 1)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # ---- Train
    log_path = Path(cfg.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "stage1_multipos_train.jsonl"

    global_step = 0
    best_metric = -1.0
    summary = {
        "config": asdict(cfg),
        "steps": [],
        "evals": [],
        "started_at": time.time(),
    }
    print(f"[train] {total_steps} total steps, batch {cfg.batch_size}, "
          f"K={K}, lr {cfg.lr}, anchor {cfg.anchor_weight}")

    for epoch in range(cfg.epochs):
        base.train()
        epoch_losses = []
        for step, (batch_images, captions_per_img) in enumerate(
            tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        ):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)

            B = len(batch_images)
            inp = train_batcher(batch_images)
            inp = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                   for k, v in inp.items()}

            # Flatten captions: (B, K) -> (B*K,) in row-major order so
            # text_embs[i*K + k] corresponds to image i, caption k.
            flat_captions: list[str] = []
            for caps in captions_per_img:
                flat_captions.extend(caps)
            assert len(flat_captions) == B * K, f"got {len(flat_captions)} expected {B*K}"

            ids = clip_tok(flat_captions, padding="max_length", truncation=True,
                           max_length=77, return_tensors="pt")["input_ids"].to(
                torch.long).to(device, non_blocking=True)

            with torch.no_grad():
                with base.disable_adapter():
                    teacher_img = base.get_image_features(**inp)
                teacher_img = teacher_img / teacher_img.norm(
                    dim=-1, keepdim=True).clamp(min=1e-6)
                text_feat = retokenizer(ids)               # (B*K, D)
                text_feat = text_feat / text_feat.norm(
                    dim=-1, keepdim=True).clamp(min=1e-6)

            amp_dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=(device == "cuda" and cfg.bf16)):
                student_img = base.get_image_features(**inp)
            student_img = student_img.float()
            student_img = student_img / student_img.norm(
                dim=-1, keepdim=True).clamp(min=1e-6)

            loss_i2t, loss_t2i = multipos_infonce(
                student_img, text_feat, K=K, logit_scale=cfg.fixed_logit_scale
            )
            loss_info = 0.5 * (loss_i2t + loss_t2i)
            loss_anchor = kl_anchor_loss(student_img, teacher_img.float())
            loss = loss_info + cfg.anchor_weight * loss_anchor

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=cfg.grad_clip)
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())

            if global_step % cfg.log_every == 0 or global_step == 1:
                msg = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(loss.item(), 4),
                    "loss_info": round(loss_info.item(), 4),
                    "loss_i2t": round(loss_i2t.item(), 4),
                    "loss_t2i": round(loss_t2i.item(), 4),
                    "loss_anchor": round(loss_anchor.item(), 4),
                    "lr": round(optimizer.param_groups[0]["lr"], 6),
                }
                print(f"  {msg}")
                with open(log_file, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                summary["steps"].append(msg)

        # End of epoch
        ep_path = out_dir / f"epoch_{epoch + 1}"
        ep_path.mkdir(parents=True, exist_ok=True)
        base.save_pretrained(str(ep_path))
        print(f"[ckpt] saved LoRA adapter -> {ep_path}")

        ep_summary = {
            "epoch": epoch + 1,
            "step": global_step,
            "mean_loss": float(np.mean(epoch_losses)),
        }

        if cfg.eval_each_epoch:
            print(f"[eval] running in-script COCO {args.eval_n_images}-img eval ...")
            base.eval()
            metrics = quick_inscript_eval(
                base, retokenizer, eval_batcher, device, cfg,
                n_max=args.eval_n_images
            )
            ep_summary["mscoco_eval"] = metrics
            print(f"[eval] COCO {metrics}")
            r10 = metrics.get("R@10", 0.0)
            if r10 > best_metric:
                best_metric = r10
                best_path = out_dir / "best"
                if best_path.exists():
                    import shutil
                    shutil.rmtree(best_path)
                base.save_pretrained(str(best_path))
                ep_summary["new_best"] = True
                print(f"[best] new best COCO R@10 = {r10:.4f} -> {best_path}")

        summary["evals"].append(ep_summary)
        with open(out_dir / "stage1_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    summary["finished_at"] = time.time()
    summary["walltime_s"] = summary["finished_at"] - summary["started_at"]
    summary["best_coco_r10"] = best_metric
    with open(out_dir / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] best COCO R@10 = {best_metric:.4f}")
    print(f"[done] artifacts at {out_dir}")


if __name__ == "__main__":
    main()
