"""Vanilla encoder-decoder Transformer baseline.

This is the line LDC v2 must cross. Per the roadmap (Faz 2) we tune this
with care: the parameter count must match LDC v2 within +/-5%, and the
hyperparameters get an HPO sweep before any LDC comparison is made.

We deliberately use plain `nn.MultiheadAttention` building blocks rather
than `nn.Transformer` to keep behaviour identical between this baseline
and the LDC denoiser/decoder (they share the same attention implementation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import LearnedPositionalEmbedding


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 96
    n_heads: int = 4
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_src_len: int = 64
    max_tgt_len: int = 128
    src_pad_id: int = 0
    tgt_pad_id: int = 0
    tie_embeddings: bool = False  # tied src/tgt embed only when vocabs match


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
        # src_pad_mask: True = PAD (matches nn.MultiheadAttention semantics)
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, key_padding_mask=src_pad_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


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
        tgt_pad_mask: torch.Tensor | None,
        src_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(
            h, h, h, attn_mask=causal, key_padding_mask=tgt_pad_mask, need_weights=False
        )
        x = x + self.drop(a)
        h = self.norm2(x)
        a, _ = self.cross_attn(
            h, memory, memory, key_padding_mask=src_pad_mask, need_weights=False
        )
        x = x + self.drop(a)
        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


class TransformerSeq2Seq(nn.Module):
    """Plain encoder-decoder transformer for the baseline runs."""

    config: TransformerConfig

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.src_embed = nn.Embedding(config.src_vocab_size, d, padding_idx=config.src_pad_id)
        self.tgt_embed = nn.Embedding(config.tgt_vocab_size, d, padding_idx=config.tgt_pad_id)
        if config.tie_embeddings:
            assert config.src_vocab_size == config.tgt_vocab_size, "vocab sizes must match to tie"
            self.tgt_embed.weight = self.src_embed.weight

        self.src_pos = LearnedPositionalEmbedding(config.max_src_len, d)
        self.tgt_pos = LearnedPositionalEmbedding(config.max_tgt_len, d)

        self.encoder_layers = nn.ModuleList(
            [
                _EncoderLayer(d, config.n_heads, config.ffn_mult, config.dropout)
                for _ in range(config.n_enc_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                _DecoderLayer(d, config.n_heads, config.ffn_mult, config.dropout)
                for _ in range(config.n_dec_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d)
        self.dec_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, config.tgt_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # src_mask: True = VALID. We need True = PAD for nn.MultiheadAttention.
        src_pad_mask = ~src_mask
        x = self.src_embed(src)
        x = self.src_pos(x)
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask)
        return self.enc_norm(x)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        L = tgt_in.size(1)
        # nn.MultiheadAttention expects mask where True = MASKED OUT.
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )
        tgt_pad_mask = ~tgt_mask
        src_pad_mask = ~src_mask
        x = self.tgt_embed(tgt_in)
        x = self.tgt_pos(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, causal, tgt_pad_mask, src_pad_mask)
        x = self.dec_norm(x)
        return self.lm_head(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Standard teacher-forced training step.

        `tgt` is the full target sequence including <sos>...<eos>. We use
        tgt[:, :-1] as decoder input and tgt[:, 1:] as the next-token target.
        """
        memory = self.encode(src, src_mask)
        tgt_in = tgt[:, :-1]
        tgt_in_mask = tgt_mask[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_out_mask = tgt_mask[:, 1:]

        logits = self.decode(tgt_in, memory, tgt_in_mask, src_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.config.tgt_pad_id,
            reduction="mean",
        )
        # Token-level accuracy (excluding pad)
        with torch.no_grad():
            preds = logits.argmax(-1)
            valid = tgt_out_mask
            correct = ((preds == tgt_out) & valid).sum().float()
            n = valid.sum().clamp_min(1).float()
            tok_acc = correct / n
        return {"loss": loss, "logits": logits, "token_acc": tok_acc}

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy decoding."""
        self.eval()
        memory = self.encode(src, src_mask)
        B = src.size(0)
        device = src.device
        tgt = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        tgt_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        effective_max = min(max_len, self.config.max_tgt_len)
        for _ in range(effective_max - 1):
            logits = self.decode(tgt, memory, tgt_mask, src_mask)
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.config.tgt_pad_id), next_tok)
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
            tgt_mask = torch.cat(
                [tgt_mask, (~finished).unsqueeze(1)], dim=1
            )
            finished = finished | (next_tok == eos_id)
            if finished.all():
                break
        return tgt
