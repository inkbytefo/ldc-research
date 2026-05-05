"""LDC encoder: text -> (initial concept graph C_0, context for cross-attn).

The encoder serves two purposes:
  1. Produce a context tensor (B, L, d) that the diffusion denoiser
     and decoder cross-attend to.
  2. Produce an initial concept graph C_0 (B, K, d_total) via slot
     attention. C_0 is used as the *clean* target for the diffusion
     forward process during training. At inference, the diffusion
     starts from noise and uses only the context — but the encoder
     still computes C_0 for the auxiliary `latent_recon` regularizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.modules import LearnedPositionalEmbedding

from .concept_space import ConceptGraph, ConceptSpaceConfig


@dataclass
class LDCEncoderConfig:
    vocab_size: int
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 64
    pad_id: int = 0


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, key_padding_mask=src_pad_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class LDCEncoder(nn.Module):
    def __init__(self, enc_cfg: LDCEncoderConfig, space_cfg: ConceptSpaceConfig) -> None:
        super().__init__()
        self.enc_cfg = enc_cfg
        self.space_cfg = space_cfg
        d = enc_cfg.d_model

        self.embed = nn.Embedding(enc_cfg.vocab_size, d, padding_idx=enc_cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(enc_cfg.max_len, d)
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(d, enc_cfg.n_heads, enc_cfg.ffn_mult, enc_cfg.dropout)
                for _ in range(enc_cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        # Slot-attention extraction: K learnable queries cross-attend over context.
        self.slot_queries = nn.Parameter(torch.randn(space_cfg.num_slots, d) * 0.02)
        self.slot_attn = nn.MultiheadAttention(d, enc_cfg.n_heads, dropout=enc_cfg.dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(d)
        # Project to concept-space dimension (Phase 1: d_total == d_e == d_model usually).
        self.slot_proj = nn.Linear(d, space_cfg.d_total)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> tuple[ConceptGraph, torch.Tensor, torch.Tensor]:
        """Returns (C_0, context, src_mask).

        src      : (B, L)
        src_mask : (B, L) bool, True = VALID
        C_0      : ConceptGraph with nodes (B, K, d_total)
        context  : (B, L, d_model)
        """
        x = self.embed(src)
        x = self.pos(x)
        src_pad_mask = ~src_mask
        for layer in self.layers:
            x = layer(x, src_pad_mask)
        context = self.final_norm(x)

        B = src.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        slots, _ = self.slot_attn(
            queries, context, context, key_padding_mask=src_pad_mask, need_weights=False
        )
        slots = self.slot_norm(slots)
        nodes = self.slot_proj(slots)
        return ConceptGraph(nodes=nodes, config=self.space_cfg), context, src_mask
