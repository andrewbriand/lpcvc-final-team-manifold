"""Fast training loop using pre-computed FAISS retrieval results.

Since the text encoder is frozen, FAISS retrieval produces identical
results every epoch. Pre-computing eliminates ~70% of per-step time
(FAISS queries + filtering), leaving only the GPU forward/backward pass.

Usage:
    python -m self_training.train_precomputed \
        --precomputed-dir data/precomputed_v3 \
        --epochs 15 \
        --batch-size 2048 \
        --lr 5e-5 \
        --device cuda \
        --checkpoint-dir checkpoints/self_train_v4 \
        --log-dir logs/self_train_v4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.config import SelfTrainConfig
from self_training.eval_harness import run_eval_suite
from self_training.logger import TrainingLogger
from self_training.lora_setup import apply_lora
from self_training.loss import combined_loss
from validate import load_model


class PrecomputedDataset(Dataset):
    """Dataset backed by pre-computed retrieval .npz shards.

    Each sample is a tuple of (original_emb, pseudo_emb, hard_neg_embs)
    as numpy arrays. Image tensors are NOT included — the training loop
    loads and encodes images separately via webdataset.
    """

    def __init__(self, precomputed_dir: Path):
        self.precomputed_dir = Path(precomputed_dir)
        manifest_path = self.precomputed_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.n_neg = self.manifest["min_hard_negatives"]
        self.dim = self.manifest["embedding_dim"]

        # Load all shards into memory (they're just embeddings, not images)
        shard_files = sorted(self.precomputed_dir.glob("shard_*.npz"))
        self.original_embs = []
        self.pseudo_embs = []
        self.hard_neg_embs = []
        self.source_keys = []

        for sf in shard_files:
            data = np.load(sf, allow_pickle=True)
            self.original_embs.append(data["original_embs"])
            self.pseudo_embs.append(data["pseudo_embs"])
            self.hard_neg_embs.append(data["hard_neg_embs"])
            self.source_keys.extend(data["source_keys"].tolist())

        self.original_embs = np.concatenate(self.original_embs, axis=0)
        self.pseudo_embs = np.concatenate(self.pseudo_embs, axis=0)
        self.hard_neg_embs = np.concatenate(self.hard_neg_embs, axis=0)

        print(f"Loaded {len(self)} pre-computed samples from "
              f"{len(shard_files)} shards ({self.original_embs.nbytes / 1e9:.1f} GB)")

    def __len__(self):
        return self.original_embs.shape[0]

    def __getitem__(self, idx):
        return (
            self.original_embs[idx],
            self.pseudo_embs[idx],
            self.hard_neg_embs[idx],
            self.source_keys[idx],
        )


def _collate_precomputed(batch):
    originals, pseudos, hard_negs, keys = zip(*batch)
    return (
        np.stack(originals),
        np.stack(pseudos),
        np.stack(hard_negs),
        list(keys),
    )


def _chunked_encode_image(model, image_tensors, chunk_size=256):
    if image_tensors.shape[0] <= chunk_size:
        return model.encode_image(image_tensors)
    chunks = []
    for i in range(0, image_tensors.shape[0], chunk_size):
        chunks.append(model.encode_image(image_tensors[i : i + chunk_size]))
    return torch.cat(chunks, dim=0)


def _cosine_warmup_schedule(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer, lr_schedule):
    for group, lr in zip(optimizer.param_groups, lr_schedule):
        group["lr"] = lr


def make_param_groups(model, config):
    """Build optimizer param groups. logit_scale is frozen (excluded)."""
    lora_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            head_params.append(param)
    return [
        {"params": lora_params, "lr": config.lr},
        {"params": head_params, "lr": config.lr},
    ]


def train_precomputed(config: SelfTrainConfig, precomputed_dir: Path):
    device = config.device
    print(f"Device: {device}")

    # 1. Load model + LoRA
    print("Loading model...")
    model, preprocess, tokenizer = load_model(config.model_key, device)
    print("Applying LoRA...")
    model = apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
    base_model = model.base_model.model

    # 2. Load pre-computed dataset
    print("Loading pre-computed retrieval data...")
    pc_dataset = PrecomputedDataset(precomputed_dir)

    # 3. We still need images for the forward pass. Load them via webdataset
    # but match by source_key. For efficiency, we'll use the pre-computed
    # embeddings for text side and encode images fresh (with LoRA gradients).
    #
    # APPROACH: The pre-computed data contains text embeddings (original, pseudo,
    # hard_neg). The training loop only needs to:
    #   1. Load image by source_key
    #   2. Encode through LoRA'd visual encoder (gradient flows here)
    #   3. Compute loss against pre-computed text embeddings
    #
    # But loading images by key from webdataset is hard (sequential access).
    # SIMPLER APPROACH: Just use the pre-computed text embeddings directly.
    # Images are encoded from the webdataset stream, matched by key.
    #
    # SIMPLEST APPROACH (what we do): Since each pre-computed sample already
    # has its source_key, we iterate through the webdataset, and for each
    # image that has a match in the pre-computed set, we use those text embs.
    # This avoids random access entirely.

    # Build key -> index map for pre-computed data
    key_to_pc_idx = {}
    for i, key in enumerate(pc_dataset.source_keys):
        key_to_pc_idx[key] = i

    print(f"Pre-computed samples: {len(pc_dataset)}")
    print(f"Unique keys: {len(key_to_pc_idx)}")

    # 3b. Frozen teacher for MSE/cosine anchor loss (prevents collapse)
    # Keep a separate copy of the base model's visual encoder (no LoRA)
    # to produce "teacher" embeddings the student should stay close to.
    # Reference: BYOL (Grill 2020), DINO (Caron 2021), EWC-style regularization.
    print("Creating frozen teacher encoder...")
    teacher_model, _, _ = load_model(config.model_key, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 3c. Freeze logit_scale — pretrained value is already well-calibrated.
    # Letting it drift caused loss collapse in v1-v4 (scale → 100, softmax saturates).
    base_model.logit_scale.requires_grad = False
    frozen_logit_scale = base_model.logit_scale.exp().item()
    print(f"Frozen logit_scale: {frozen_logit_scale:.2f}")

    # 4. Optimizer + scheduler (logit_scale excluded)
    param_groups = make_param_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = max(len(pc_dataset) // config.batch_size, 1)
    total_steps = steps_per_epoch * config.max_epochs

    logger = TrainingLogger(config.log_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Anchor weight: {config.anchor_weight}")

    # 5. Training loop — iterate pre-computed data directly
    global_step = 0
    best_metric = -1.0
    patience_counter = 0

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")

        # Shuffle pre-computed indices each epoch
        perm = np.random.permutation(len(pc_dataset))

        # We need actual images for the LoRA forward pass.
        # Stream from webdataset and match keys.
        from self_training.data_pipeline import create_image_dataset, list_shard_files
        shard_paths = list_shard_files(config.cc12m_shards_dir, config.num_shards)
        img_dataset = create_image_dataset(shard_paths, shuffle=0)

        def _img_collate(batch):
            images, metas = [], []
            for img, meta in batch:
                images.append(img)
                metas.append(meta if isinstance(meta, dict) else {})
            return images, metas

        img_loader = DataLoader(
            img_dataset, batch_size=config.batch_size,
            num_workers=4, collate_fn=_img_collate, pin_memory=True,
        )

        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_matched = 0

        for batch_images, batch_metas in tqdm(img_loader, desc=f"Epoch {epoch+1}"):
            # Match images to pre-computed data by key
            matched_img_indices = []
            matched_pc_indices = []
            for i, meta in enumerate(batch_metas):
                key = meta.get("__key__", "")
                if key in key_to_pc_idx:
                    matched_img_indices.append(i)
                    matched_pc_indices.append(key_to_pc_idx[key])

            if len(matched_img_indices) < 16:
                continue

            # Prepare image tensors (only matched ones)
            matched_images = [batch_images[i] for i in matched_img_indices]
            image_tensors = torch.stack([preprocess(img) for img in matched_images]).to(device)

            # Get pre-computed text embeddings
            pc_idx = np.array(matched_pc_indices)
            original_embs_np = pc_dataset.original_embs[pc_idx]
            pseudo_embs_np = pc_dataset.pseudo_embs[pc_idx]
            hard_neg_embs_np = pc_dataset.hard_neg_embs[pc_idx]

            original_t = torch.tensor(original_embs_np, dtype=torch.float32, device=device)
            pseudo_t = torch.tensor(pseudo_embs_np, dtype=torch.float32, device=device)
            hard_neg_t = torch.tensor(hard_neg_embs_np, dtype=torch.float32, device=device)

            # Forward pass: student (LoRA'd) and teacher (frozen) encoders
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Student: gradients flow through LoRA
                student_embs = _chunked_encode_image(base_model, image_tensors)
                student_embs = F.normalize(student_embs.float(), dim=-1)

                # Teacher: frozen baseline, no gradients
                with torch.no_grad():
                    teacher_embs = _chunked_encode_image(teacher_model, image_tensors)
                    teacher_embs = F.normalize(teacher_embs.float(), dim=-1)

            # Combined loss: InfoNCE + MSE/cosine representation anchor
            total_loss, infonce_loss, anchor_loss = combined_loss(
                student_image_embs=student_embs,
                teacher_image_embs=teacher_embs,
                original_text_embs=original_t,
                pseudo_text_embs=pseudo_t,
                hard_neg_embs=hard_neg_t,
                logit_scale=frozen_logit_scale,
                anchor_weight=config.anchor_weight,
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # LR schedule (only 2 groups now — no logit_scale)
            global_step += 1
            base_lrs = [config.lr, config.lr]
            scheduled_lrs = [
                _cosine_warmup_schedule(global_step, config.warmup_steps, total_steps, lr)
                for lr in base_lrs
            ]
            _set_lr(optimizer, scheduled_lrs)

            # Logging
            step_loss = total_loss.item()
            epoch_loss_sum += step_loss
            epoch_steps += 1
            epoch_matched += len(matched_img_indices)
            logger.log_step(global_step, epoch + 1, step_loss, scheduled_lrs[0], len(matched_img_indices))

        # End of epoch
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1} complete. Avg loss: {avg_loss:.4f}, "
              f"steps: {epoch_steps}, matched: {epoch_matched}")

        # Eval
        model.eval()
        with torch.no_grad():
            eval_results = run_eval_suite(
                model, preprocess, tokenizer,
                datasets=config.eval_datasets,
                device=device,
            )

        for ds_key, metrics in eval_results.items():
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {ds_key}: {metric_str}")
        logger.log_eval(global_step, epoch + 1, eval_results)

        # Checkpoint
        ckpt_path = config.checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Primary metric for early stopping: avg of COCO and Flickr R@10
        coco_r10 = eval_results.get("mscoco", {}).get("R@10", 0.0)
        flickr_r10 = eval_results.get("flickr30k", {}).get("R@10", 0.0)
        primary = (coco_r10 + flickr_r10) / 2.0

        if primary > best_metric:
            best_metric = primary
            patience_counter = 0
            torch.save(model.state_dict(), config.checkpoint_dir / "best.pt")
            logger.log_checkpoint(global_step, epoch + 1,
                                  str(config.checkpoint_dir / "best.pt"), primary)
            print(f"New best metric: {primary:.4f} -> saved to best.pt")
        else:
            patience_counter += 1
            print(f"No improvement (patience {patience_counter}/{config.early_stop_patience})")
            if patience_counter >= config.early_stop_patience:
                print("Early stopping triggered.")
                break

    logger.close()
    print(f"\nTraining complete. Best metric: {best_metric:.4f}")
    print(f"Best checkpoint: {config.checkpoint_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Fast training with pre-computed retrieval")
    parser.add_argument("--precomputed-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, default=Path("data/cc12m_shards"))
    parser.add_argument("--num-shards", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-logit-scale", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/self_train_v4"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/self_train_v4"))
    parser.add_argument("--model-key", default="mobileclip2_s2")
    args = parser.parse_args()

    config = SelfTrainConfig(
        model_key=args.model_key,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        cc12m_shards_dir=args.shards_dir,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_logit_scale=args.lr_logit_scale,
        warmup_steps=args.warmup_steps,
        max_epochs=args.epochs,
        first_run_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    train_precomputed(config, args.precomputed_dir)


if __name__ == "__main__":
    main()
