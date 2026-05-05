from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps integer timestep t to a `d_model`-dimensional embedding via
    sinusoidal frequencies followed by an MLP. Used by the diffusion denoiser.
    """

    def __init__(self, d_model: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 4 * d_model
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integers; returns (B, d_model)
        half = self.d_model // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float, device=device)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.d_model:  # odd d_model
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))
