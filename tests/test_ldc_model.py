import torch

from src.ldc.concept_space import ConceptSpaceConfig
from src.ldc.decoder import LDCDecoderConfig
from src.ldc.encoder import LDCEncoderConfig
from src.ldc.graph_denoiser import GraphDenoiserConfig
from src.ldc.model import LDCConfig, LDCModel


def _tiny_ldc(vocab: int = 16) -> LDCModel:
    space = ConceptSpaceConfig(num_slots=4, d_euclidean=16, num_relations=2)
    enc = LDCEncoderConfig(
        vocab_size=vocab, d_model=32, n_heads=2, n_layers=1, ffn_mult=2,
        dropout=0.0, max_len=16, pad_id=0,
    )
    dec = LDCDecoderConfig(
        vocab_size=vocab, d_model=32, n_heads=2, n_layers=1, ffn_mult=2,
        dropout=0.0, max_len=16, pad_id=0, d_concept=space.d_total,
    )
    den = GraphDenoiserConfig(
        d_model=32, n_heads=2, n_layers=1, ffn_mult=2, dropout=0.0,
        num_slots=space.num_slots, num_relations=space.num_relations,
        d_concept=space.d_total,
    )
    return LDCModel(LDCConfig(
        encoder=enc, decoder=dec, space=space, denoiser=den,
        num_diffusion_steps=5, schedule="cosine",
    ))


def test_forward_shapes_and_losses_finite():
    torch.manual_seed(0)
    model = _tiny_ldc()
    src = torch.randint(1, 16, (2, 6))
    src_mask = torch.ones(2, 6, dtype=torch.bool)
    tgt = torch.randint(1, 16, (2, 8))
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)
    out = model(src, src_mask, tgt, tgt_mask)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["l_diff"])
    assert torch.isfinite(out["l_lm"])
    assert out["logits"].shape == (2, 7, 16)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = _tiny_ldc()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    src = torch.tensor([[3, 4, 5, 6, 0, 0], [7, 8, 9, 0, 0, 0]])
    src_mask = src != 0
    tgt = torch.tensor([[1, 5, 4, 3, 2, 0, 0], [1, 9, 8, 7, 2, 0, 0]])
    tgt_mask = tgt != 0

    lm_first = None
    for step in range(120):
        out = model(src, src_mask, tgt, tgt_mask)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step == 0:
            lm_first = float(out["l_lm"].item())
    # The LM loss should drop substantially; we don't require full convergence
    # in 120 steps because diffusion adds gradient noise.
    assert float(out["l_lm"].item()) < lm_first * 0.6


def test_refine_returns_concept_graph_with_correct_shape():
    torch.manual_seed(0)
    model = _tiny_ldc()
    src = torch.tensor([[3, 4, 5, 0, 0, 0]])
    src_mask = src != 0
    cg = model.refine(src, src_mask)
    assert cg.shape == (1, 4, 16)
    assert torch.isfinite(cg.nodes).all()


def test_generate_returns_well_formed():
    torch.manual_seed(0)
    model = _tiny_ldc()
    src = torch.tensor([[3, 4, 5, 0, 0, 0]])
    src_mask = src != 0
    out = model.generate(src, src_mask, sos_id=1, eos_id=2, max_len=10, use_clean=True)
    assert out.shape[0] == 1
    assert out[0, 0].item() == 1
    assert out.shape[1] <= 10


def test_seed_reproducibility():
    src = torch.randint(1, 16, (2, 6))
    src_mask = torch.ones(2, 6, dtype=torch.bool)
    tgt = torch.randint(1, 16, (2, 8))
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)

    torch.manual_seed(42)
    m1 = _tiny_ldc()
    torch.manual_seed(42)
    m2 = _tiny_ldc()

    torch.manual_seed(7)
    out1 = m1(src, src_mask, tgt, tgt_mask)
    torch.manual_seed(7)
    out2 = m2(src, src_mask, tgt, tgt_mask)
    assert torch.allclose(out1["loss"], out2["loss"], atol=1e-6)
