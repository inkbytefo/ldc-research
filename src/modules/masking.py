from __future__ import annotations

import torch


def causal_mask(L: int, device: torch.device | None = None) -> torch.Tensor:
    """Returns an (L, L) bool mask where True = ATTEND, False = MASKED OUT.

    Standard nn.MultiheadAttention `attn_mask` semantics use float -inf or
    bool where True means "ignore". Here we return the *attend* form;
    callers convert as needed.
    """
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))


def length_to_padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """lengths: (B,) -> bool mask (B, max_len) where True = VALID position."""
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange < lengths.unsqueeze(1)
