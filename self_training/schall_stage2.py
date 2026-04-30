"""Schall Stage 2 — retokenizer realign against the LoRA'd image encoder.

Reads:
  - Stage 1 LoRA adapter dir (peft format)  -> applied to FG-CLIP 2 base
  - Run #1 retokenizer best.pt              -> warm start

Trains:
  - retokenizer.token_embedding             (49408, hidden)
  - retokenizer.position_embedding          (64, hidden)
  - (optional) retokenizer.adapter

Loss:
  Image-text contrastive (InfoNCE) on COCO + Flickr (+CC12M optional).
  Image features come from the Stage 1 LoRA'd FG-CLIP 2 image trunk
  (FROZEN). Text features come from the trainable retokenizer.

Output:
  out_dir/best.pt — retokenizer checkpoint compatible with
  validate.py's fgclip2_retokenized_hf loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from self_training.retokenizer_model import FgClip2Retokenizer  # noqa: E402
from self_training.schall_stage1 import (  # noqa: E402
    FgClip2ImageBatcher,
    ImageCaptionPair,
    _collate_pairs,
    load_coco_train,
    load_flickr_train,
    load_cc12m_subset,
    quick_inscript_eval,
)
from validate import _patch_fgclip2_text_embeddings  # noqa: E402


@dataclass
class Stage2Config:
    stage1_adapter_dir: str
    stage1_retokenizer_warmstart: str
    out_dir: str
    hf_cache_dir: str = "hf_cache"
    log_dir: str = "logs/schall_stage2"

    fgclip2_repo: str = "qihoo360/fg-clip2-base"

    use_coco: bool = True
    use_flickr: bool = True
    use_cc12m: bool = False
    cc12m_max_samples: int = 0

    batch_size: int = 256
    epochs: int = 2
    lr_token: float = 1e-4
    lr_pos: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 200
    grad_clip: float = 1.0
    fixed_logit_scale: float = 20.0
    num_workers: int = 4
    bf16: bool = True
    seed: int = 0
    log_every: int = 25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-adapter-dir", type=str, required=True,
                        help="Path to Stage 1 PEFT LoRA adapter dir (e.g. .../stage1/best/)")
    parser.add_argument("--stage1-retokenizer-warmstart", type=str, required=True,
                        help="Path to Run #1 retokenizer best.pt — used as warm start for token/pos embeds")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--hf-cache-dir", type=str, default="hf_cache")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr-token", type=float, default=1e-4)
    parser.add_argument("--lr-pos", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = Stage2Config(
        stage1_adapter_dir=args.stage1_adapter_dir,
        stage1_retokenizer_warmstart=args.stage1_retokenizer_warmstart,
        out_dir=args.out_dir,
        hf_cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_token=args.lr_token,
        lr_pos=args.lr_pos,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        fixed_logit_scale=args.fixed_logit_scale,
        num_workers=args.num_workers,
        bf16=not args.no_bf16,
        use_coco=not args.no_coco,
        use_flickr=not args.no_flickr,
        use_cc12m=args.cc12m_max_samples > 0,
        cc12m_max_samples=args.cc12m_max_samples,
        seed=args.seed,
        log_every=args.log_every,
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

    # ---- Model + Stage 1 LoRA
    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForCausalLM, CLIPTokenizer

    print(f"[load] FG-CLIP 2 base ...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.fgclip2_repo, trust_remote_code=True
    )
    _patch_fgclip2_text_embeddings(base)
    base = base.to(device)
    print(f"[load] applying Stage 1 LoRA from {cfg.stage1_adapter_dir} ...")
    base = PeftModel.from_pretrained(base, cfg.stage1_adapter_dir)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    image_processor = AutoImageProcessor.from_pretrained(
        cfg.fgclip2_repo, trust_remote_code=True
    )
    image_batcher = FgClip2ImageBatcher(image_processor)

    # ---- Retokenizer (TRAINABLE), warm-started from Stage 1 input checkpoint
    base_inner = base.get_base_model() if hasattr(base, "get_base_model") else base
    print(f"[load] warm-start retokenizer from {cfg.stage1_retokenizer_warmstart} ...")
    ws = torch.load(cfg.stage1_retokenizer_warmstart, map_location=device, weights_only=False)
    ws_cfg = (ws.get("trainable_state_dict") or {}).get("config", {})
    with_adapter = bool(ws_cfg.get("with_adapter", False))
    retokenizer = FgClip2Retokenizer(base_inner, with_adapter=with_adapter).to(device)
    retokenizer.load_trainable_state_dict(ws["trainable_state_dict"])

    # Trainable parameters: only the new tables (and adapter if present)
    trainable_params: list[torch.nn.Parameter] = []
    pos_params: list[torch.nn.Parameter] = []
    tok_params: list[torch.nn.Parameter] = []
    for name, p in retokenizer.named_parameters():
        if name.startswith("base."):
            p.requires_grad = False
            continue
        p.requires_grad = True
        if name.startswith("token_embedding"):
            tok_params.append(p)
        elif name.startswith("position_embedding"):
            pos_params.append(p)
        else:
            trainable_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": tok_params, "lr": cfg.lr_token, "weight_decay": cfg.weight_decay},
            {"params": pos_params, "lr": cfg.lr_pos, "weight_decay": cfg.weight_decay},
            {"params": trainable_params, "lr": cfg.lr_token, "weight_decay": cfg.weight_decay},
        ]
    )

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

    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    total_steps = max(len(loader) * cfg.epochs, 1)

    def lr_at(step: int, base_lr: float) -> float:
        if step < cfg.warmup_steps:
            return base_lr * step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    log_path = Path(cfg.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "stage2_train.jsonl"

    summary = {
        "config": asdict(cfg),
        "steps": [],
        "evals": [],
        "started_at": time.time(),
    }

    global_step = 0
    best_metric = -1.0

    print(f"[train] {total_steps} steps, batch {cfg.batch_size}, lr_token {cfg.lr_token}, lr_pos {cfg.lr_pos}")

    for epoch in range(cfg.epochs):
        retokenizer.train()
        epoch_losses = []
        from tqdm import tqdm as _tqdm
        for step, (batch_images, batch_captions) in enumerate(
            _tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        ):
            base_lrs = [cfg.lr_token, cfg.lr_pos, cfg.lr_token]
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr_at(global_step, base_lr)

            inp = image_batcher(batch_images)
            inp = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                   for k, v in inp.items()}
            ids = clip_tok(batch_captions, padding="max_length", truncation=True,
                           max_length=77, return_tensors="pt")["input_ids"].to(torch.long).to(device, non_blocking=True)

            # Frozen image features from Stage 1 LoRA'd trunk
            with torch.no_grad():
                amp_dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda" and cfg.bf16)):
                    img_feat = base.get_image_features(**inp)
                img_feat = img_feat.float()
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            # Trainable text features
            txt_feat = retokenizer(ids)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            logit_scale = cfg.fixed_logit_scale
            logits_t2i = logit_scale * txt_feat @ img_feat.t()
            logits_i2t = logits_t2i.t()
            labels = torch.arange(txt_feat.shape[0], device=device)
            loss_t2i = F.cross_entropy(logits_t2i, labels)
            loss_i2t = F.cross_entropy(logits_i2t, labels)
            loss = 0.5 * (loss_t2i + loss_i2t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                tok_params + pos_params + trainable_params,
                max_norm=cfg.grad_clip,
            )
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())

            if global_step % cfg.log_every == 0 or global_step == 1:
                msg = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(loss.item(), 4),
                    "lr_token": round(optimizer.param_groups[0]["lr"], 6),
                    "lr_pos": round(optimizer.param_groups[1]["lr"], 6),
                }
                print(f"  {msg}")
                with open(log_file, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                summary["steps"].append(msg)

        # End of epoch — save retokenizer checkpoint
        ep_ckpt = out_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "trainable_state_dict": retokenizer.trainable_state_dict(),
                "stage": "schall_stage2",
                "epoch": epoch + 1,
                "step": global_step,
                "stage1_adapter_dir": cfg.stage1_adapter_dir,
                "stage1_retokenizer_warmstart": cfg.stage1_retokenizer_warmstart,
            },
            ep_ckpt,
        )
        print(f"[ckpt] saved retokenizer -> {ep_ckpt}")

        ep_summary = {
            "epoch": epoch + 1,
            "step": global_step,
            "mean_loss": float(np.mean(epoch_losses)),
        }

        if not args.no_eval:
            print(f"[eval] in-script COCO {args.eval_n_images}-img eval ...")
            retokenizer.eval()
            base.eval()
            metrics = quick_inscript_eval(
                base, retokenizer, image_batcher, device,
                # Reuse Stage1Config-shaped fields the eval needs
                _Stage1Like(batch_size=cfg.batch_size, hf_cache_dir=cfg.hf_cache_dir),
                n_max=args.eval_n_images,
            )
            ep_summary["mscoco_eval"] = metrics
            print(f"[eval] COCO {metrics}")
            r10 = metrics.get("R@10", 0.0)
            if r10 > best_metric:
                best_metric = r10
                best_path = out_dir / "best.pt"
                torch.save(
                    {
                        "trainable_state_dict": retokenizer.trainable_state_dict(),
                        "stage": "schall_stage2",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "stage1_adapter_dir": cfg.stage1_adapter_dir,
                        "stage1_retokenizer_warmstart": cfg.stage1_retokenizer_warmstart,
                        "coco_R10": r10,
                    },
                    best_path,
                )
                ep_summary["new_best"] = True
                print(f"[best] new best COCO R@10 = {r10:.4f} -> {best_path}")

        summary["evals"].append(ep_summary)
        with open(out_dir / "stage2_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    summary["finished_at"] = time.time()
    summary["walltime_s"] = summary["finished_at"] - summary["started_at"]
    summary["best_coco_r10"] = best_metric
    with open(out_dir / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] best COCO R@10 = {best_metric:.4f}")
    print(f"[done] artifacts at {out_dir}")


# Tiny shim so we can pass a Stage1Config-like object into the shared eval
class _Stage1Like:
    def __init__(self, batch_size: int, hf_cache_dir: str):
        self.batch_size = batch_size
        self.hf_cache_dir = hf_cache_dir


if __name__ == "__main__":
    main()
