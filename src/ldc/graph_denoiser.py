"""Graph-transformer denoiser for the LDC latent diffusion core.

Given a noised concept graph C_t (B, K, d), a timestep t (B,), and the
encoder context (B, L, d_ctx), predict the noise epsilon (B, K, d).

Per the roadmap (section 1):
    Phase 1: relations are *implicit* — captured by R parallel attention
    head groups with a learnable typed bias matrix `relation_bias`
    (R, K, K). This is the simplest realization of "ilişkiler birinci
    sınıf nesneler" without requiring discrete relation prediction.
    Phase 5+ will replace `relation_bias` with batch-dependent
    relation logits and add hyperbolic-aware distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.modules import SinusoidalTimeEmbedding


@dataclass
class GraphDenoiserConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    num_slots: int = 8
    num_relations: int = 4
    d_concept: int = 64  # node embed dim — projected to d_model internally


class _GraphDenoiserLayer(nn.Module):
    """One layer: typed self-attention over slots + cross-attention over context + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        num_slots: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_relations = num_relations

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        # Per-relation typed attention bias: shared across heads (kept simple).
        # Shape (R, K, K). At forward time we sum over relations to form a (K, K)
        # additive bias that is broadcast over batch and heads.
        self.relation_bias = nn.Parameter(torch.zeros(num_relations, num_slots, num_slots))
        self.relation_gate = nn.Parameter(torch.zeros(num_relations))
        self.drop = nn.Dropout(dropout)

    def _build_attn_bias(self) -> torch.Tensor:
        gates = torch.softmax(self.relation_gate, dim=0).view(-1, 1, 1)
        return (gates * self.relation_bias).sum(dim=0)  # (K, K)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        ctx_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, K, d), context: (B, L, d), ctx_pad_mask: (B, L) True=PAD
        K = x.size(1)
        bias = self._build_attn_bias()
        # Repeat for batch*heads if PyTorch will accept (K, K) — nn.MultiheadAttention
        # expects (L, S) or (N*H, L, S). Use broadcast-friendly (1, 1, K, K) flattened.
        # The cleanest form is to expand to (B*H, K, K). We pull H from self_attn.
        H = self.self_attn.num_heads
        B = x.size(0)
        attn_mask = bias.unsqueeze(0).expand(B * H, K, K)

        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)

        h = self.norm2(x)
        a, _ = self.cross_attn(
            h, context, context, key_padding_mask=ctx_pad_mask, need_weights=False
        )
        x = x + self.drop(a)

        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


class GraphDenoiser(nn.Module):
    """Predicts noise on the (B, K, d_concept) node tensor."""

    def __init__(self, cfg: GraphDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.in_proj = nn.Linear(cfg.d_concept, d)
        self.out_proj = nn.Linear(d, cfg.d_concept)

        self.time_embed = SinusoidalTimeEmbedding(d)
        # Per-slot positional bias (slots are unordered conceptually but we let the
        # model learn to break symmetry if it helps).
        self.slot_pos = nn.Parameter(torch.randn(cfg.num_slots, d) * 0.02)

        self.layers = nn.ModuleList(
            [
                _GraphDenoiserLayer(
                    d, cfg.n_heads, cfg.ffn_mult, cfg.dropout, cfg.num_slots, cfg.num_relations
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

    def forward(
        self,
        x_t: torch.Tensor,  # (B, K, d_concept)
        t: torch.Tensor,  # (B,) long
        context: torch.Tensor,  # (B, L, d_model)
        context_mask: torch.Tensor,  # (B, L) bool, True=VALID
    ) -> torch.Tensor:
        h = self.in_proj(x_t) + self.slot_pos.unsqueeze(0)
        # Add time embedding broadcast over slot dim.
        t_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, d)
        h = h + t_emb
        ctx_pad_mask = ~context_mask
        for layer in self.layers:
            h = layer(h, context, ctx_pad_mask)
        h = self.final_norm(h)
        return self.out_proj(h)
