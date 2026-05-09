import torch

from src.ldc.graph_denoiser import GraphDenoiser, GraphDenoiserConfig


def test_output_shape_matches_input():
    torch.manual_seed(0)
    cfg = GraphDenoiserConfig(
        d_model=32, n_heads=2, n_layers=2, ffn_mult=2, dropout=0.0,
        num_slots=4, num_relations=2, d_concept=16,
    )
    den = GraphDenoiser(cfg)
    x_t = torch.randn(2, 4, 16)
    t = torch.tensor([0, 3], dtype=torch.long)
    ctx = torch.randn(2, 6, 32)
    ctx_mask = torch.ones(2, 6, dtype=torch.bool)
    out = den(x_t, t, ctx, ctx_mask)
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()


def test_time_conditioning_changes_output():
    torch.manual_seed(0)
    cfg = GraphDenoiserConfig(
        d_model=16, n_heads=2, n_layers=1, ffn_mult=2, dropout=0.0,
        num_slots=4, num_relations=2, d_concept=8,
    )
    den = GraphDenoiser(cfg)
    den.eval()
    x_t = torch.randn(1, 4, 8)
    ctx = torch.randn(1, 5, 16)
    ctx_mask = torch.ones(1, 5, dtype=torch.bool)
    out_a = den(x_t, torch.tensor([0]), ctx, ctx_mask)
    out_b = den(x_t, torch.tensor([7]), ctx, ctx_mask)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_context_mask_affects_output():
    torch.manual_seed(0)
    cfg = GraphDenoiserConfig(
        d_model=16, n_heads=2, n_layers=1, ffn_mult=2, dropout=0.0,
        num_slots=4, num_relations=2, d_concept=8,
    )
    den = GraphDenoiser(cfg)
    den.eval()
    x_t = torch.randn(1, 4, 8)
    t = torch.tensor([0])
    ctx = torch.randn(1, 5, 16)
    full_mask = torch.ones(1, 5, dtype=torch.bool)
    half_mask = torch.tensor([[True, True, True, False, False]])
    out_full = den(x_t, t, ctx, full_mask)
    out_half = den(x_t, t, ctx, half_mask)
    assert not torch.allclose(out_full, out_half)
