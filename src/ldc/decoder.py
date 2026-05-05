"""LDC autoregressive decoder.

Cross-attends over the *concept-graph nodes* (B, K, d_concept) — not over
the encoder context. This is the key difference from the baseline: the
decoder only sees the refined latent thoughts, not the raw input tokens.

We do project the concept nodes to d_model (decoder hidden) so the
concept-space dim is decoupled from the decoder hidden size.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import LearnedPositionalEmbedding


@dataclass
class LDCDecoderConfig:
    vocab_size: int
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 128
    pad_id: int = 0
    d_concept: int = 64


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(
            h, h, h, attn_mask=causal, key_padding_mask=tgt_pad_mask, need_weights=False
        )
        x = x + self.drop(a)
        h = self.norm2(x)
        # Concept-graph slots have no padding (fixed K), so no key_padding_mask.
        a, _ = self.cross_attn(h, memory, memory, need_weights=False)
        x = x + self.drop(a)
        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


class LDCDecoder(nn.Module):
    def __init__(self, cfg: LDCDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_len, d)
        self.memory_proj = nn.Linear(cfg.d_concept, d)

        self.layers = nn.ModuleList(
            [_DecoderLayer(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)

    def _project_memory(self, concept_nodes: torch.Tensor) -> torch.Tensor:
        return self.memory_proj(concept_nodes)

    def forward(
        self, tgt_in: torch.Tensor, tgt_in_mask: torch.Tensor, concept_nodes: torch.Tensor
    ) -> torch.Tensor:
        memory = self._project_memory(concept_nodes)
        L = tgt_in.size(1)
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )
        tgt_pad_mask = ~tgt_in_mask
        x = self.embed(tgt_in)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, causal, tgt_pad_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, concept_nodes: torch.Tensor, sos_id: int, eos_id: int, max_len: int
    ) -> torch.Tensor:
        """Greedy decode given a refined concept graph."""
        self.eval()
        memory = self._project_memory(concept_nodes)
        B = concept_nodes.size(0)
        device = concept_nodes.device

        tgt = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            L = tgt.size(1)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            x = self.embed(tgt)
            x = self.pos(x)
            for layer in self.layers:
                x = layer(x, memory, causal, None)
            x = self.final_norm(x)
            logits = self.lm_head(x)
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(
                finished, torch.full_like(next_tok, self.cfg.pad_id), next_tok
            )
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if finished.all():
                break
        return tgt
