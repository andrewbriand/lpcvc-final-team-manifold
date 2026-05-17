from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SelfTrainConfig:
    # Model
    model_key: str = "mobileclip2_s2"

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Data
    cc12m_shards_dir: Path = Path("data/cc12m_shards")
    num_shards: int = 10
    faiss_index_path: Path = Path("data/cc12m_faiss.index")
    caption_map_path: Path = Path("data/cc12m_captions.npz")

    # Retrieval
    top_k: int = 17
    own_caption_cosine_threshold: float = 0.99

    # Confidence filter (calibrated offline)
    min_top1_cosine: float = 0.25
    min_margin: float = 0.02
    use_margin_filter: bool = False  # single-threshold preferred (SPF 2025)

    # Hard-negative filter
    jaccard_threshold: float = 0.7
    neg_cosine_proximity: float = 0.85
    min_hard_negatives: int = 2  # lowered from 4 to reduce cascading rejection

    # MSE/cosine representation anchor (prevents collapse)
    anchor_weight: float = 1.0  # weight of anchor loss vs InfoNCE

    # Training
    batch_size: int = 2048
    lr: float = 1e-5  # conservative for self-training with pseudo-labels
    lr_logit_scale: float = 5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_epochs: int = 8
    first_run_epochs: int = 3
    logit_scale_clamp: tuple[float, float] = (0.0, 4.6052)

    # Eval
    eval_datasets: list[str] = field(
        default_factory=lambda: ["sample", "mscoco", "flickr30k", "sugarcrepe"]
    )
    early_stop_patience: int = 2

    # Paths
    checkpoint_dir: Path = Path("checkpoints/self_train")
    log_dir: Path = Path("logs/self_train")
    device: str = "cuda"
