"""Schall Stage 1 — image-encoder LoRA finetune at fixed-224.

Targets the diagnosed eval-deployment resolution mismatch:
  validate.py's fgclip2_*_hf loaders feed variable-resolution images
  (~512x512 effective, up to 1024 patches), but the deployed ONNX takes
  a fixed (1, 3, 224, 224) input. The frozen image trunk degrades at the
  lower resolution; Schall (2024) recommends adapting the image encoder
  first, then realigning text. This script implements Stage 1.

Design:
  - Frozen: retokenizer (loaded from best.pt), text trunk, logit_scale,
    everything outside the LoRA adapters.
  - Trainable: LoRA on FG-CLIP 2 image-trunk attention layers.
  - Data: COCO + Flickr image-caption pairs at fixed 224x224 (the deployed
    contract). CC12M streaming optional; off by default for time.
  - Loss: InfoNCE (image-text contrastive) + KL anchor (frozen teacher
    image embeddings, BYOL/DINO-style consistency).
  - Eval: each epoch, dispatch validate.py with --fgclip2-fix-resolution
    on COCO + Flickr test splits using the merged image encoder.

Output: LoRA adapter saved as `stage1_best/` (peft format) + a small
`stage1_summary.json`. Stage 2 takes this adapter and retrains the
retokenizer against the new image embeddings.
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
class Stage1Config:
    # I/O
    retokenizer_checkpoint: str
    out_dir: str
    hf_cache_dir: str = "hf_cache"
    log_dir: str = "logs/schall_stage1"

    # Model
    fgclip2_repo: str = "qihoo360/fg-clip2-base"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Substring patterns to identify image-trunk attention Linear modules.
    # We scan model.named_modules() and pick any Linear whose name contains
    # ALL of: a vision-side path token AND an attention token.
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

    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    anchor_weight: float = 0.5
    # Letting logit_scale learn drove softmax saturation in earlier self-training
    # runs; pinning it avoids that collapse mode entirely.
    fixed_logit_scale: float = 20.0
    num_workers: int = 4
    bf16: bool = True
    augment_train: bool = True

    # Eval
    eval_each_epoch: bool = True
    eval_datasets: str = "mscoco,flickr30k"

    # Misc
    log_every: int = 25
    save_every_epoch: bool = True


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
class FgClip2ImageBatcher:
    """Wraps the FG-CLIP 2 image processor for fixed-224 batched training.

    Eval mode: deterministic 224x224 BICUBIC resize. Matches deployed contract.
    Train mode (augment=True, v2): random resized crop to 224x224 + horizontal
    flip + mild color jitter. Output shape is identical so downstream
    image_processor sees a uniform batch.
    """

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
        # FG-CLIP 2's processor accepts a list and returns batched tensors
        # when max_num_patches is uniform.
        out = self.image_processor(
            images=pils, max_num_patches=256, return_tensors="pt"
        )
        return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ImageCaptionPair(Dataset):
    """Generic image-caption pair dataset backed by a list of (img, caption).

    Heavy lifting lives in the dataset adapters below. This class just
    indexes into a flat list to keep DataLoader workers simple.
    """

    def __init__(self, samples: list[tuple]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, cap = self.samples[idx]
        return img.convert("RGB"), cap


def _collate_pairs(batch):
    images = [b[0] for b in batch]
    captions = [b[1] for b in batch]
    return images, captions


def load_coco_train(cache_dir: str) -> list[tuple]:
    """Try COCO train splits in order of preference.

    1. yerevann/coco-karpathy train (~113K images, standard CLIP-style split)
    2. lmms-lab/COCO-Caption2017 val (5K — fallback, smaller but always available)
    """
    from datasets import load_dataset

    # Order matters: lmms-lab COCO-Caption2017 SHIPS PIL images so we can use
    # them directly. yerevann/coco-karpathy ships URLs only — no good for
    # offline training. Fall back to it last just in case it has been re-shimmed.
    attempts = [
        ("lmms-lab/COCO-Caption2017", "val"),
        ("lmms-lab/COCO-Caption2017", "test"),
        ("yerevann/coco-karpathy", "train"),
    ]
    last_err = None
    for repo, split in attempts:
        try:
            print(f"[data] trying {repo} split={split} ...")
            ds = load_dataset(repo, split=split, cache_dir=cache_dir)
            break
        except Exception as e:
            last_err = e
            print(f"  failed: {type(e).__name__}: {e}")
    else:
        raise RuntimeError(f"All COCO train sources failed; last error: {last_err}")

    samples: list[tuple] = []
    for ex in tqdm(ds, desc="  coco", leave=False):
        img = ex.get("image") or ex.get("jpg")
        # Different schemas across repos: try common fields.
        cap_field = (
            ex.get("answer")
            or ex.get("caption")
            or ex.get("captions")
            or ex.get("sentences")
            or ex.get("text")
        )
        if img is None or not cap_field:
            continue
        if isinstance(cap_field, list):
            for c in cap_field:
                if isinstance(c, dict):
                    c = c.get("raw") or c.get("caption") or ""
                if isinstance(c, str) and c.strip():
                    samples.append((img, c.strip()))
        elif isinstance(cap_field, str) and cap_field.strip():
            samples.append((img, cap_field.strip()))
    print(f"[data] coco train pairs: {len(samples):,}")
    return samples


def load_flickr_train(cache_dir: str) -> list[tuple]:
    """Flickr30K train split via clip-benchmark/wds_flickr30k."""
    from datasets import load_dataset

    print("[data] loading Flickr30K train ...")
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
    for ex in tqdm(ds, desc="  flickr", leave=False):
        img = ex.get("jpg") or ex.get("image")
        cap = ex.get("txt") or ex.get("caption") or ""
        if img is None or not cap:
            continue
        if isinstance(cap, list):
            for c in cap:
                if isinstance(c, str) and c.strip():
                    samples.append((img, c.strip()))
        else:
            samples.append((img, str(cap).strip()))
    print(f"[data] flickr train pairs: {len(samples):,}")
    return samples


def load_cc12m_subset(cache_dir: str, max_samples: int) -> list[tuple]:
    """Stream pixparse/cc12m-wds, take first max_samples decoded pairs."""
    from datasets import load_dataset

    print(f"[data] streaming CC12M, target {max_samples:,} ...")
    ds = load_dataset(
        "pixparse/cc12m-wds", split="train", streaming=True, cache_dir=cache_dir
    )
    samples: list[tuple] = []
    for ex in ds:
        img = ex.get("jpg") or ex.get("image")
        cap = ex.get("txt") or ""
        if img is None or not cap:
            continue
        samples.append((img, str(cap).strip()))
        if len(samples) >= max_samples:
            break
    print(f"[data] cc12m streamed pairs: {len(samples):,}")
    return samples


# ---------------------------------------------------------------------------
# LoRA targeting
# ---------------------------------------------------------------------------
def find_lora_targets(model: torch.nn.Module, cfg: Stage1Config) -> list[str]:
    """Walk model.named_modules() and pick image-trunk attention Linears."""
    candidates: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        # must live somewhere in the vision path
        if not any(tok in name for tok in cfg.lora_vision_path_tokens):
            continue
        # name's last segment must be an attention-related module
        last = name.split(".")[-1]
        if last not in cfg.lora_attn_module_tokens:
            continue
        candidates.append(name)
    return candidates


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def quick_inscript_eval(
    base, retokenizer, image_batcher, device: str, cfg: Stage1Config,
    n_max: int = 5000,
) -> dict:
    """Lightweight in-script COCO 5K R@10 eval using the live model."""
    from validate import load_mscoco

    images, captions, img2txt = load_mscoco(cfg.hf_cache_dir)
    if len(images) > n_max:
        images = images[:n_max]
        # truncate img2txt accordingly (drop captions whose img got dropped)
        cap_keep = set()
        new_img2txt = []
        for gt in img2txt[:n_max]:
            new_img2txt.append(gt)
            cap_keep.update(gt)
        img2txt = new_img2txt
        # rebuild caption list to keep alignment
        keep_sorted = sorted(cap_keep)
        idx_remap = {old: new for new, old in enumerate(keep_sorted)}
        captions = [captions[i] for i in keep_sorted]
        img2txt = [[idx_remap[i] for i in gt] for gt in img2txt]

    # Image embeddings (LoRA'd model).
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

    # Text embeddings (frozen retokenizer).
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt_embs: list[np.ndarray] = []
    retokenizer.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), cfg.batch_size), desc="  eval-txt", leave=False):
            chunk = captions[i : i + cfg.batch_size]
            ids = tok(chunk, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(torch.long).to(device)
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--fixed-logit-scale", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-coco", action="store_true")
    parser.add_argument("--no-flickr", action="store_true")
    parser.add_argument("--cc12m-max-samples", type=int, default=0,
                        help="If >0, stream this many CC12M pairs in addition to COCO/Flickr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-n-images", type=int, default=5000)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable image augmentation (random resized crop, hflip, color jitter)")
    parser.add_argument("--smoke", action="store_true", help="Quick 50-step smoke run for sanity.")
    args = parser.parse_args()

    cfg = Stage1Config(
        retokenizer_checkpoint=args.retokenizer_checkpoint,
        out_dir=args.out_dir,
        hf_cache_dir=args.hf_cache_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    # ---- Frozen retokenizer (text path)
    print(f"[load] retokenizer from {cfg.retokenizer_checkpoint} ...")
    ckpt = torch.load(cfg.retokenizer_checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = (ckpt.get("trainable_state_dict") or {}).get("config", {})
    with_adapter = bool(ckpt_cfg.get("with_adapter", False))
    retokenizer = FgClip2Retokenizer(base, with_adapter=with_adapter).to(device).eval()
    retokenizer.load_trainable_state_dict(ckpt["trainable_state_dict"])
    for p in retokenizer.parameters():
        p.requires_grad = False

    # ---- Frozen teacher copy of image trunk (for KL anchor)
    # We keep the un-LoRA'd base as the teacher implicitly: PEFT applies
    # adapters in-place but preserves the original weights, so teacher
    # forward = disable adapters via PEFT context manager.

    # ---- Apply LoRA to image trunk attention
    targets = find_lora_targets(base, cfg)
    if not targets:
        raise RuntimeError(
            "Could not find LoRA target modules. Inspect model.named_modules() "
            "and update Stage1Config.lora_*_tokens."
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

    # Freeze everything except LoRA params (PEFT does most of this; we
    # tighten by ensuring NO non-LoRA params have grad).
    for name, p in base.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False

    # ---- Data
    samples: list[tuple] = []
    if cfg.use_coco:
        samples += load_coco_train(cfg.hf_cache_dir)
    if cfg.use_flickr:
        samples += load_flickr_train(cfg.hf_cache_dir)
    if cfg.use_cc12m:
        samples += load_cc12m_subset(cfg.hf_cache_dir, cfg.cc12m_max_samples)
    if not samples:
        raise RuntimeError("No training pairs loaded.")
    random.shuffle(samples)
    print(f"[data] total pairs = {len(samples):,}")

    if args.smoke:
        samples = samples[: cfg.batch_size * 50]
        print(f"[smoke] truncated to {len(samples)} pairs")

    ds = ImageCaptionPair(samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_pairs,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---- Tokenizer (for the retokenizer's text path)
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
    log_file = log_path / "stage1_train.jsonl"

    global_step = 0
    best_metric = -1.0
    summary = {
        "config": asdict(cfg),
        "steps": [],
        "evals": [],
        "started_at": time.time(),
    }
    print(f"[train] {total_steps} total steps, batch {cfg.batch_size}, lr {cfg.lr}, anchor {cfg.anchor_weight}")

    for epoch in range(cfg.epochs):
        base.train()
        epoch_losses = []
        for step, (batch_images, batch_captions) in enumerate(
            tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        ):
            # Set LR
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)

            inp = train_batcher(batch_images)
            inp = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                   for k, v in inp.items()}

            ids = clip_tok(batch_captions, padding="max_length", truncation=True,
                          max_length=77, return_tensors="pt")["input_ids"].to(torch.long).to(device, non_blocking=True)

            with torch.no_grad():
                with base.disable_adapter():
                    teacher_img = base.get_image_features(**inp)
                teacher_img = teacher_img / teacher_img.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                text_feat = retokenizer(ids)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            amp_dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda" and cfg.bf16)):
                student_img = base.get_image_features(**inp)
            student_img = student_img.float()
            student_img = student_img / student_img.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            logit_scale = cfg.fixed_logit_scale
            logits_i2t = logit_scale * student_img @ text_feat.t()
            logits_t2i = logits_i2t.t()
            labels = torch.arange(student_img.shape[0], device=device)
            loss_i2t = F.cross_entropy(logits_i2t, labels)
            loss_t2i = F.cross_entropy(logits_t2i, labels)
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
                base, retokenizer, eval_batcher, device, cfg, n_max=args.eval_n_images
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
