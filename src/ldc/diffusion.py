"""Gaussian DDPM core, applied to the *node* tensor of a concept graph.

We follow Ho et al. 2020 (DDPM) closely. In Phase 1 we noise only the
Euclidean node embeddings; relations are static (handled by the denoiser).

Conventions
-----------
Timesteps are integers in [0, num_steps - 1] inclusive. t = 0 is the
*least* noisy step (one step from clean), t = num_steps - 1 is the noisiest.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal 2021 cosine schedule."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


class GaussianDiffusion(nn.Module):
    """Stores the noise schedule and exposes q_sample / p_sample."""

    def __init__(self, num_steps: int = 20, schedule: str = "cosine") -> None:
        super().__init__()
        if schedule == "linear":
            betas = linear_beta_schedule(num_steps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False
        )
        # Posterior variance: beta_tilde = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-8)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)
        # 1/sqrt(alpha_t) and beta_t/sqrt(1-alpha_bar_t) for the reverse step formula
        self.register_buffer("sqrt_recip_alphas", torch.rsqrt(alphas), persistent=False)
        self.register_buffer(
            "noise_coef",
            betas / torch.sqrt(1.0 - alphas_cumprod).clamp_min(1e-8),
            persistent=False,
        )

    def _extract(self, buf: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        # buf: (T,), t: (B,) -> (B, 1, 1, ...) reshaped
        out = buf.gather(0, t)
        while out.dim() < len(target_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sa = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        som = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sa * x_0 + som * noise

    def p_sample(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """One reverse step: x_{t-1} given x_t and predicted noise."""
        sra = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        nc = self._extract(self.noise_coef, t, x_t.shape)
        mean = sra * (x_t - nc * noise_pred)

        # Add posterior noise except at t = 0
        var = self._extract(self.posterior_variance, t, x_t.shape)
        nonzero = (t > 0).float()
        while nonzero.dim() < x_t.dim():
            nonzero = nonzero.unsqueeze(-1)
        eps = torch.randn_like(x_t)
        return mean + nonzero * torch.sqrt(var.clamp_min(0.0)) * eps

    def loss(
        self, noise_pred: torch.Tensor, noise_target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(noise_pred, noise_target, reduction=reduction)

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        device: torch.device | str,
    ) -> torch.Tensor:
        """Reverse-process sampling from x_T ~ N(0, I).

        `denoise_fn(x_t, t)` should return the predicted noise. Used in
        unit tests; real model `generate()` calls inline for clarity.
        """
        x = torch.randn(shape, device=device)
        for t_int in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), t_int, dtype=torch.long, device=device)
            noise_pred = denoise_fn(x, t)
            x = self.p_sample(x, t, noise_pred)
        return x
