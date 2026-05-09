import torch

from src.baselines.transformer import TransformerConfig, TransformerSeq2Seq
from src.utils import count_parameters


def _tiny_config(vocab: int = 16) -> TransformerConfig:
    return TransformerConfig(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=32,
        n_heads=2,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_mult=2,
        dropout=0.0,
        max_src_len=16,
        max_tgt_len=16,
        src_pad_id=0,
        tgt_pad_id=0,
    )


def test_forward_shapes():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = TransformerSeq2Seq(cfg)
    src = torch.randint(1, cfg.src_vocab_size, (2, 6))
    src_mask = torch.ones(2, 6, dtype=torch.bool)
    tgt = torch.randint(1, cfg.tgt_vocab_size, (2, 8))
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)
    out = model(src, src_mask, tgt, tgt_mask)
    assert out["loss"].dim() == 0
    assert out["logits"].shape == (2, 7, cfg.tgt_vocab_size)
    assert 0.0 <= out["token_acc"].item() <= 1.0


def test_overfits_tiny_batch():
    """Two examples, no dropout: training loss should drop near zero quickly."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = TransformerSeq2Seq(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    src = torch.tensor([[3, 4, 5, 6, 0, 0], [7, 8, 9, 0, 0, 0]])
    src_mask = src != 0
    tgt = torch.tensor([[1, 5, 4, 3, 2, 0, 0], [1, 9, 8, 7, 2, 0, 0]])
    tgt_mask = tgt != 0

    losses = []
    for _ in range(80):
        out = model(src, src_mask, tgt, tgt_mask)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))
    assert losses[-1] < 0.5, f"baseline didn't overfit: {losses[-1]}"
    assert losses[-1] < losses[0]


def test_generate_returns_well_formed():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = TransformerSeq2Seq(cfg)
    src = torch.tensor([[3, 4, 5, 0, 0, 0]])
    src_mask = src != 0
    out = model.generate(src, src_mask, sos_id=1, eos_id=2, max_len=10)
    assert out.shape[0] == 1
    assert out[0, 0].item() == 1  # starts with sos
    assert out.shape[1] <= 10


def test_param_count_sane():
    cfg = _tiny_config()
    model = TransformerSeq2Seq(cfg)
    n = count_parameters(model)
    # Sanity: tiny model is well below 1M
    assert 5_000 < n < 200_000
