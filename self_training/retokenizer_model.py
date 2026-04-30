"""FG-CLIP 2 retokenizer wrapper.

Wraps a frozen FG-CLIP 2 base model so it can consume LPCVC's contract
input — (1, 77) int64 OpenAI CLIP BPE tokens — instead of the native
Gemma tokenization. The wrapper truncates 77 -> 64 inside the graph,
swaps the Gemma 256K-vocab embedding for a NEW trainable (49408, hidden)
table, and replaces the short-mode position embedding with a NEW
trainable (64, hidden) table. An optional MLP adapter sits between the
embedding sum and the frozen text encoder.

Mirrors the export shape from ``scripts/export_fgclip2.py``
(``FGCLIP2_SHORT_LEN = 64``, ``walk_type="short"``).

Usage:

    base = AutoModelForCausalLM.from_pretrained("qihoo360/fg-clip2-base",
                                                trust_remote_code=True)
    _patch_fgclip2_text_embeddings(base)  # critical, see validate.py
    student = FgClip2Retokenizer(base, with_adapter=False)

    # Smoke training: only embedding+pos params have requires_grad=True.
    out = student(input_ids)  # input_ids: (B, 77) int64
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# Vocab of the OpenAI CLIP BPE tokenizer ("openai/clip-vit-base-patch32").
CLIP_BPE_VOCAB_SIZE = 49408

# FG-CLIP 2's short-mode position-embedding length. Mirrors
# ``scripts/export_fgclip2.py`` (FGCLIP2_SHORT_LEN = 64) and
# ``config.text_config.max_position_embeddings``.
INTERNAL_TEXT_LEN = 64


@dataclass
class RetokenizerConfig:
    vocab_size: int = CLIP_BPE_VOCAB_SIZE
    hidden_dim: int = 768  # overwritten at construct time from FG-CLIP 2 config
    pos_len: int = INTERNAL_TEXT_LEN
    with_adapter: bool = False


class MLPAdapter(nn.Module):
    """Linear -> GeLU -> Linear, hidden_dim throughout.

    Initialized so the module is near-identity at init: first Linear is
    standard, second Linear's weight starts at zero so output equals input
    until training pushes it elsewhere. Avoids destabilizing the frozen
    trunk on step 0.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class FgClip2Retokenizer(nn.Module):
    """Frozen FG-CLIP 2 + new BPE-indexed embedding/position tables.

    Forward pass:
        input_ids (B, 77 or any L >= 64) int64
            -> truncate to (B, 64)
            -> NEW token_embedding (49408, hidden)
            -> + NEW position_embedding (64, hidden)
            -> [optional MLPAdapter]
            -> frozen Fgclip2TextTransformer encoder
            -> final_layer_norm -> head -> last-token pool
            -> (B, hidden)

    Returns L2-normalized text features, matching the export contract.
    """

    def __init__(
        self,
        base_model: nn.Module,
        with_adapter: bool = False,
        normalize_output: bool = True,
    ):
        super().__init__()

        # FG-CLIP2 base text config carries hidden_size and max_position_embeddings.
        text_cfg = base_model.config.text_config
        hidden_dim = int(text_cfg.hidden_size)
        if int(text_cfg.max_position_embeddings) != INTERNAL_TEXT_LEN:
            raise RuntimeError(
                "FG-CLIP 2 short-mode position embedding length "
                f"changed: expected {INTERNAL_TEXT_LEN}, got "
                f"{text_cfg.max_position_embeddings}."
            )

        self.cfg = RetokenizerConfig(
            vocab_size=CLIP_BPE_VOCAB_SIZE,
            hidden_dim=hidden_dim,
            pos_len=INTERNAL_TEXT_LEN,
            with_adapter=with_adapter,
        )

        # New trainable embedding table (49408, hidden_dim).
        self.token_embedding = nn.Embedding(CLIP_BPE_VOCAB_SIZE, hidden_dim)
        # New trainable position embedding (64, hidden_dim).
        self.position_embedding = nn.Embedding(INTERNAL_TEXT_LEN, hidden_dim)

        # Default init: small Gaussian. Lookup-table warm start replaces these.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.adapter: Optional[MLPAdapter] = None
        if with_adapter:
            self.adapter = MLPAdapter(hidden_dim)

        # Hold the frozen base. We deliberately keep it as a submodule so
        # ``.to(device)`` / ``.eval()`` propagate, but freeze every param.
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        self.normalize_output = normalize_output
        self._lpcvc_text_input = "tensor"

    # ------------------------------------------------------------------
    # Lookup-table init (FVT-style warm start).
    # For each CLIP BPE token, decode its surface form, re-tokenize with the
    # Gemma tokenizer, and average those Gemma row embeddings. Tokens whose
    # surface form Gemma can't represent fall back to the unk vector.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_from_lookup_table(
        self,
        clip_tokenizer,  # transformers CLIPTokenizer or open_clip tokenizer-like
        gemma_tokenizer,
        gemma_token_embedding: nn.Embedding,
        verbose: bool = True,
    ) -> dict:
        """Warm-start ``token_embedding`` from FG-CLIP 2's Gemma table.

        Returns a small dict of stats (covered, fallback_unk, mean_subwords).
        """
        gemma_weight = gemma_token_embedding.weight.detach().to(
            self.token_embedding.weight.device
        )
        gemma_dim = gemma_weight.shape[1]
        if gemma_dim != self.token_embedding.weight.shape[1]:
            raise RuntimeError(
                f"hidden_dim mismatch: gemma {gemma_dim} vs new {self.token_embedding.weight.shape[1]}"
            )

        unk_id = getattr(gemma_tokenizer, "unk_token_id", None)
        if unk_id is None:
            unk_id = 0
        unk_vec = gemma_weight[unk_id]

        new_weight = torch.empty_like(self.token_embedding.weight)

        n_covered = 0
        n_unk = 0
        subword_count_sum = 0

        for clip_id in range(CLIP_BPE_VOCAB_SIZE):
            try:
                surface = clip_tokenizer.decode(
                    [clip_id], skip_special_tokens=False
                )
            except Exception:
                surface = ""

            if not surface or surface.strip() == "":
                new_weight[clip_id] = unk_vec
                n_unk += 1
                continue

            try:
                gemma_ids = gemma_tokenizer.encode(surface, add_special_tokens=False)
            except Exception:
                gemma_ids = []

            if len(gemma_ids) == 0:
                new_weight[clip_id] = unk_vec
                n_unk += 1
                continue

            ids_t = torch.tensor(gemma_ids, dtype=torch.long, device=gemma_weight.device)
            new_weight[clip_id] = gemma_weight[ids_t].mean(dim=0)
            n_covered += 1
            subword_count_sum += len(gemma_ids)

        self.token_embedding.weight.copy_(new_weight)

        stats = {
            "covered": n_covered,
            "fallback_unk": n_unk,
            "mean_subwords": (subword_count_sum / max(n_covered, 1)),
        }
        if verbose:
            print(
                f"[lookup-init] covered={n_covered}/{CLIP_BPE_VOCAB_SIZE}, "
                f"unk_fallback={n_unk}, mean_subwords={stats['mean_subwords']:.2f}"
            )
        return stats

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) int. L is allowed to be >= 64; we slice to 64.

        Matches scripts/export_fgclip2.py:114-117 (truncate first 64 tokens).
        """
        ids = input_ids.to(torch.long)[:, :INTERNAL_TEXT_LEN]

        # New embedding table.
        tok = self.token_embedding(ids)  # (B, 64, H)

        # Short-mode positions are simply 0..63.
        pos_ids = torch.arange(
            INTERNAL_TEXT_LEN, device=ids.device, dtype=torch.long
        )
        pos = self.position_embedding(pos_ids)  # (64, H)
        h = tok + pos.unsqueeze(0)              # (B, 64, H)

        if self.adapter is not None:
            h = self.adapter(h)

        # Run the frozen text trunk in short mode by feeding ``inputs_embeds``
        # through the inner text encoder. We bypass the base's
        # Fgclip2TextEmbeddings entirely (we already added position embeds).
        text_model = self.base.text_model  # Fgclip2TextTransformer
        encoder_outputs = text_model.encoder(inputs_embeds=h)
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = text_model.final_layer_norm(last_hidden_state)

        # Short-mode pool: last token, then per-row head.
        pooled = last_hidden_state[:, -1, :]  # (B, H)
        # head is nn.Linear(H, projection_size). Loop matches the upstream
        # walk_short branch faithfully.
        out = torch.cat(
            [text_model.head(pooled[i : i + 1]) for i in range(pooled.shape[0])],
            dim=0,
        )

        if self.normalize_output:
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return out

    # Convenience for validate.py-style harnesses.
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)

    # ------------------------------------------------------------------
    # Save / load helpers — only the trainable retokenizer pieces.
    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> dict:
        sd = {
            "token_embedding": self.token_embedding.state_dict(),
            "position_embedding": self.position_embedding.state_dict(),
            "config": {
                "vocab_size": self.cfg.vocab_size,
                "hidden_dim": self.cfg.hidden_dim,
                "pos_len": self.cfg.pos_len,
                "with_adapter": self.cfg.with_adapter,
            },
        }
        if self.adapter is not None:
            sd["adapter"] = self.adapter.state_dict()
        return sd

    def load_trainable_state_dict(self, sd: dict, strict: bool = True) -> None:
        self.token_embedding.load_state_dict(sd["token_embedding"], strict=strict)
        self.position_embedding.load_state_dict(sd["position_embedding"], strict=strict)
        if "adapter" in sd:
            if self.adapter is None:
                raise RuntimeError(
                    "Checkpoint has adapter weights but model was built with "
                    "with_adapter=False."
                )
            self.adapter.load_state_dict(sd["adapter"], strict=strict)
        elif self.adapter is not None and strict:
            raise RuntimeError(
                "Model expects adapter but checkpoint has none."
            )
