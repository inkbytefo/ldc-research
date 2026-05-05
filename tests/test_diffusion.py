import torch

from src.ldc.diffusion import (
    GaussianDiffusion,
    cosine_beta_schedule,
    linear_beta_schedule,
)


def test_linear_schedule_monotone_and_bounded():
    betas = linear_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas[1:] >= betas[:-1]).all()
    assert betas.min() > 0
    assert betas.max() < 1


def test_cosine_schedule_clipped():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert betas.min() >= 0
    assert betas.max() <= 0.999


def test_alpha_bar_decreasing_to_near_zero_for_long_schedule():
    """For a long enough schedule, q(x_T) should be ~ N(0, I)."""
    diff = GaussianDiffusion(num_steps=200, schedule="cosine")
    # alpha_bar should decay from ~1 to small values
    assert diff.alphas_cumprod[0].item() > 0.98
    assert diff.alphas_cumprod[-1].item() < 0.05


def test_q_sample_distribution_at_high_t():
    """At max t, q_sample should produce ~ N(0, 1) regardless of x_0."""
    torch.manual_seed(0)
    diff = GaussianDiffusion(num_steps=200, schedule="cosine")
    x_0 = torch.randn(1024, 4, 8) * 5.0  # arbitrary scale
    t = torch.full((1024,), diff.num_steps - 1, dtype=torch.long)
    x_t = diff.q_sample(x_0, t)
    # mean ~ 0, std ~ 1
    assert x_t.mean().abs().item() < 0.1
    assert abs(x_t.std().item() - 1.0) < 0.1


def test_q_sample_at_t0_is_close_to_x0():
    diff = GaussianDiffusion(num_steps=20, schedule="cosine")
    x_0 = torch.randn(8, 4, 16)
    t = torch.zeros(8, dtype=torch.long)
    noise = torch.zeros_like(x_0)
    x_t = diff.q_sample(x_0, t, noise=noise)
    # With zero noise, x_t = sqrt(alpha_bar_0) * x_0
    expected = diff.sqrt_alphas_cumprod[0] * x_0
    assert torch.allclose(x_t, expected, atol=1e-6)


def test_p_sample_shape_preserved():
    diff = GaussianDiffusion(num_steps=10)
    x_t = torch.randn(3, 4, 8)
    t = torch.full((3,), 5, dtype=torch.long)
    pred_noise = torch.randn_like(x_t)
    out = diff.p_sample(x_t, t, pred_noise)
    assert out.shape == x_t.shape


def test_sample_loop_returns_correct_shape():
    diff = GaussianDiffusion(num_steps=5)
    shape = (2, 4, 8)
    out = diff.sample(lambda x, t: torch.zeros_like(x), shape, device="cpu")
    assert out.shape == shape
