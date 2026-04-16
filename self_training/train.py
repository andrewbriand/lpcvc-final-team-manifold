"""Self-training loop for MobileCLIP2-S2.

Retrieves pseudo-positive and hard-negative captions via FAISS, applies
confidence and hard-negative filtering, and trains the visual encoder
with LoRA + contrastive loss.
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
import webdataset as wds
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from self_training.config import SelfTrainConfig
from self_training.confidence_filter import apply_confidence_filter
from self_training.data_pipeline import create_image_dataset, list_shard_files
from self_training.eval_harness import run_eval_suite
from self_training.faiss_index import load_index, query_index
from self_training.logger import TrainingLogger
from self_training.lora_setup import apply_lora
from self_training.loss import self_train_infonce
from self_training.negative_filter import filter_hard_negatives
from validate import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_caption_bank(config: SelfTrainConfig) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the pre-computed caption bank from an npz file.

    Returns:
        embeddings: (N, D) float32 array of L2-normalised caption embeddings.
        captions: list of N caption strings.
        source_keys: list of N source key strings for own-caption exclusion.
    """
    data = np.load(config.caption_map_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    captions = list(data["captions"])
    source_keys = list(data["source_keys"])
    return embeddings, captions, source_keys


def build_source_key_to_idx(source_keys: list[str]) -> dict[str, int]:
    """Build a dict mapping source_key string -> index in the caption bank."""
    return {k: i for i, k in enumerate(source_keys)}


def make_param_groups(model: torch.nn.Module, config: SelfTrainConfig) -> list[dict]:
    """Create three parameter groups with separate learning rates.

    Groups:
        1. LoRA parameters  -> config.lr
        2. Head parameters   -> config.lr
        3. logit_scale       -> config.lr_logit_scale
    """
    lora_params = []
    head_params = []
    logit_scale_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "logit_scale" in name:
            logit_scale_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            # visual.head and any other unfrozen params
            head_params.append(param)

    return [
        {"params": lora_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": head_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": logit_scale_params, "lr": config.lr_logit_scale, "weight_decay": 0.0},
    ]


def _collate_wds(batch):
    """Custom collation for webdataset (image, meta) tuples."""
    images, metas = [], []
    for img, meta in batch:
        images.append(img)
        metas.append(meta if isinstance(meta, dict) else {})
    return images, metas


def _cosine_warmup_schedule(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Cosine schedule with linear warmup.  Returns a multiplier in [0, 1]."""
    if step < warmup_steps:
        return base_lr * (step / max(warmup_steps, 1))
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, lr_schedule: list[float]) -> None:
    """Set the learning rate for each param group from a list of target LRs."""
    for group, lr in zip(optimizer.param_groups, lr_schedule):
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: SelfTrainConfig) -> None:
    """Run the self-training loop."""
    device = config.device
    print(f"Device: {device}")

    # ---- 1. Load model, apply LoRA ----------------------------------------
    print("Loading model...")
    model, preprocess, tokenizer = load_model(config.model_key, device)
    model = apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
    model.train()

    # Determine how to access the base model (for encode_image / logit_scale)
    base_model = model.base_model.model

    # ---- 2. Load caption bank + FAISS index --------------------------------
    print("Loading caption bank...")
    caption_embs, captions, source_keys = load_caption_bank(config)
    sk_to_idx = build_source_key_to_idx(source_keys)

    print("Loading FAISS index...")
    faiss_index = load_index(config.faiss_index_path)

    # ---- 3. Load calibrated thresholds (if available) ----------------------
    thresholds_path = config.log_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        config.min_top1_cosine = thresholds.get("min_top1_cosine", config.min_top1_cosine)
        config.min_margin = thresholds.get("min_margin", config.min_margin)
        print(f"Loaded calibrated thresholds: top1>={config.min_top1_cosine:.4f}, margin>={config.min_margin:.4f}")
    else:
        print(f"No calibrated thresholds found; using defaults: top1>={config.min_top1_cosine}, margin>={config.min_margin}")

    # ---- 4. Optimizer, scheduler, logger -----------------------------------
    param_groups = make_param_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    shard_paths = list_shard_files(config.cc12m_shards_dir, config.num_shards)
    # Rough estimate: each shard ~10k images
    est_samples_per_epoch = len(shard_paths) * 10_000
    est_steps_per_epoch = max(est_samples_per_epoch // config.batch_size, 1)
    total_steps = est_steps_per_epoch * config.max_epochs

    logger = TrainingLogger(config.log_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Estimated {est_steps_per_epoch} steps/epoch, {total_steps} total steps")
    print(f"Shards: {len(shard_paths)}, batch_size: {config.batch_size}")

    # ---- 5. Training loop --------------------------------------------------
    global_step = 0
    best_metric = -1.0
    patience_counter = 0
    min_sub_batch = 16  # skip step if fewer images pass confidence filter

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")

        dataset = create_image_dataset(shard_paths)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=4,
            collate_fn=_collate_wds,
            pin_memory=True,
        )

        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_images, batch_metas in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # --- a) Preprocess images and encode ---
            image_tensors = torch.stack([preprocess(img) for img in batch_images]).to(device)
            B = image_tensors.shape[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    image_embs = base_model.encode_image(image_tensors)
                    image_embs = F.normalize(image_embs.float(), dim=-1)

            image_embs_np = image_embs.detach().cpu().numpy().astype(np.float32)

            # --- b) Build source key indices for own-caption exclusion ---
            source_indices = []
            for meta in batch_metas:
                key = meta.get("__key__", "")
                idx = sk_to_idx.get(key, -1)
                source_indices.append(idx)

            # Gather own-caption embeddings for cosine exclusion
            own_caption_embs = np.zeros((B, caption_embs.shape[1]), dtype=np.float32)
            for i, idx in enumerate(source_indices):
                if idx >= 0:
                    own_caption_embs[i] = caption_embs[idx]

            # --- c) FAISS retrieval ---
            scores, indices = query_index(
                faiss_index,
                image_embs_np,
                k=config.top_k,
                exclude_source_indices=source_indices,
                caption_embs=caption_embs,
                own_caption_embs=own_caption_embs,
                own_cosine_threshold=config.own_caption_cosine_threshold,
            )

            # --- d) Confidence filter ---
            if scores.shape[1] < 2:
                continue
            top1_scores = scores[:, 0]
            top2_scores = scores[:, 1]
            keep_mask = apply_confidence_filter(
                top1_scores, top2_scores,
                config.min_top1_cosine, config.min_margin,
            )
            keep_indices = np.where(keep_mask)[0]

            if len(keep_indices) < min_sub_batch:
                continue

            # --- e) Build sub-batch of passing images ---
            sub_image_embs_np = image_embs_np[keep_indices]
            sub_scores = scores[keep_indices]
            sub_indices = indices[keep_indices]
            sub_source_indices = [source_indices[i] for i in keep_indices]
            sub_B = len(keep_indices)

            # Pseudo-positive = rank-1 caption
            pseudo_pos_idx = sub_indices[:, 0]  # caption bank index for each image

            # Original caption embeddings (from own caption, if available)
            original_embs_list = []
            pseudo_embs_list = []
            hard_neg_embs_list = []

            valid_mask = []

            for i in range(sub_B):
                pp_idx = int(pseudo_pos_idx[i])
                if pp_idx < 0:
                    valid_mask.append(False)
                    continue

                pseudo_emb = caption_embs[pp_idx]
                pseudo_caption = captions[pp_idx]
                pseudo_embs_list.append(pseudo_emb)

                # Original caption embedding
                src_idx = sub_source_indices[i]
                if src_idx >= 0:
                    original_embs_list.append(caption_embs[src_idx])
                else:
                    # Fall back to pseudo-positive if no original caption
                    original_embs_list.append(pseudo_emb)

                # --- f) Hard-negative filtering ---
                # Candidates = ranks 2..K from retrieval
                cand_bank_indices = sub_indices[i, 1:]
                cand_valid = cand_bank_indices >= 0
                cand_bank_indices_valid = cand_bank_indices[cand_valid]

                if len(cand_bank_indices_valid) == 0:
                    valid_mask.append(False)
                    continue

                cand_captions = [captions[int(ci)] for ci in cand_bank_indices_valid]
                cand_embs_arr = np.stack([caption_embs[int(ci)] for ci in cand_bank_indices_valid])

                filtered = filter_hard_negatives(
                    pseudo_caption, pseudo_emb,
                    cand_captions, cand_embs_arr,
                    jaccard_threshold=config.jaccard_threshold,
                    neg_cosine_proximity=config.neg_cosine_proximity,
                    min_negatives=config.min_hard_negatives,
                )

                if len(filtered) < config.min_hard_negatives:
                    valid_mask.append(False)
                    continue

                neg_embs = np.stack([f[1] for f in filtered[:config.min_hard_negatives]])
                hard_neg_embs_list.append(neg_embs)
                valid_mask.append(True)

            # Rebuild sub-batch with only valid samples
            valid_local_indices = [i for i, v in enumerate(valid_mask) if v]
            if len(valid_local_indices) < min_sub_batch:
                continue

            # Map valid_local_indices back to keep_indices
            final_keep = keep_indices[[
                j for j, v in enumerate(valid_mask) if v
            ]]

            # Build training tensors
            final_B = len(valid_local_indices)
            final_image_embs = image_embs[torch.tensor(final_keep, device=device)]
            final_original_embs = torch.tensor(
                np.stack(original_embs_list), dtype=torch.float32, device=device,
            )
            final_pseudo_embs = torch.tensor(
                np.stack(pseudo_embs_list), dtype=torch.float32, device=device,
            )

            # Pad hard negatives to uniform size
            n_neg = config.min_hard_negatives
            final_hard_neg_embs = torch.tensor(
                np.stack(hard_neg_embs_list), dtype=torch.float32, device=device,
            )  # (final_B, n_neg, D)

            # --- g) Re-encode images through the trainable model ---
            final_image_tensors = image_tensors[torch.tensor(final_keep, device=device)]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                trainable_image_embs = base_model.encode_image(final_image_tensors)
                trainable_image_embs = F.normalize(trainable_image_embs.float(), dim=-1)

            # --- h) Compute loss ---
            logit_scale = base_model.logit_scale.exp().clamp(
                config.logit_scale_clamp[0], math.exp(config.logit_scale_clamp[1]),
            )

            loss = self_train_infonce(
                trainable_image_embs,
                final_original_embs,
                final_pseudo_embs,
                final_hard_neg_embs,
                logit_scale,
            )

            # --- i) Backward + optimizer step ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- j) LR schedule ---
            global_step += 1
            base_lrs = [config.lr, config.lr, config.lr_logit_scale]
            scheduled_lrs = [
                _cosine_warmup_schedule(global_step, config.warmup_steps, total_steps, lr)
                for lr in base_lrs
            ]
            _set_lr(optimizer, scheduled_lrs)

            # --- k) Logging ---
            step_loss = loss.item()
            epoch_loss_sum += step_loss
            epoch_steps += 1
            logger.log_step(global_step, epoch + 1, step_loss, scheduled_lrs[0], final_B)

        # ---- End of epoch: eval + checkpoint --------------------------------
        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}, steps: {epoch_steps}")

        # Run eval
        model.eval()
        print("Running evaluation...")
        eval_results = run_eval_suite(
            base_model, preprocess, tokenizer,
            config.eval_datasets, device,
        )
        logger.log_eval(global_step, epoch + 1, eval_results)

        # Primary metric: average R@10 across retrieval datasets
        r10_values = []
        for ds_key in ("mscoco", "flickr30k", "sample"):
            ds_metrics = eval_results.get(ds_key, {})
            for metric_key in ("R@10", "r10", "R10"):
                if metric_key in ds_metrics:
                    r10_values.append(ds_metrics[metric_key])
                    break
        primary_metric = float(np.mean(r10_values)) if r10_values else avg_epoch_loss * -1

        # Save checkpoint
        ckpt_path = config.checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.log_checkpoint(global_step, epoch + 1, str(ckpt_path), primary_metric)
        print(f"Saved checkpoint: {ckpt_path}")

        # Best model tracking + early stopping
        if primary_metric > best_metric:
            best_metric = primary_metric
            patience_counter = 0
            best_path = config.checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"New best metric: {best_metric:.4f} -> saved to {best_path}")
        else:
            patience_counter += 1
            print(f"No improvement (patience {patience_counter}/{config.early_stop_patience})")

        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.close()
    print(f"\nTraining complete. Best metric: {best_metric:.4f}")
    print(f"Best checkpoint: {config.checkpoint_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Self-training loop for MobileCLIP2-S2")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/cc12m_shards"))
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--faiss-index", type=Path, default=Path("data/cc12m_faiss.index"))
    parser.add_argument("--caption-map", type=Path, default=Path("data/cc12m_captions.npz"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-logit-scale", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/self_train"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/self_train"))
    parser.add_argument("--eval-datasets", nargs="+", default=["sample", "mscoco", "flickr30k", "sugarcrepe"])
    parser.add_argument("--model-key", default="mobileclip2_s2")
    args = parser.parse_args()

    config = SelfTrainConfig(
        model_key=args.model_key,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        cc12m_shards_dir=args.shards_dir,
        num_shards=args.num_shards,
        faiss_index_path=args.faiss_index,
        caption_map_path=args.caption_map,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_logit_scale=args.lr_logit_scale,
        warmup_steps=args.warmup_steps,
        max_epochs=args.epochs,
        first_run_epochs=args.epochs,
        top_k=args.top_k,
        eval_datasets=args.eval_datasets,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    train(config)


if __name__ == "__main__":
    main()
