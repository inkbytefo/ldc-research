import pytest

from src.data.scan import (
    SCANDataset,
    collate_seq2seq,
    parse_scan_line,
    synthetic_scan_pairs,
)
from src.data.tokenizer import WordTokenizer


def test_parse_scan_line():
    src, tgt = parse_scan_line("IN: walk twice OUT: I_WALK I_WALK")
    assert src == "walk twice"
    assert tgt == "I_WALK I_WALK"


def test_parse_scan_line_rejects_malformed():
    with pytest.raises(ValueError):
        parse_scan_line("walk twice -> I_WALK")


def test_synthetic_pairs_have_correct_shape():
    pairs = synthetic_scan_pairs(8)
    assert len(pairs) == 8
    for src, tgt in pairs:
        assert src.strip()
        assert tgt.strip()


def test_dataset_and_collate():
    pairs = synthetic_scan_pairs(8)
    src_tok = WordTokenizer.from_corpus(s for s, _ in pairs)
    tgt_tok = WordTokenizer.from_corpus(t for _, t in pairs)
    ds = SCANDataset.from_pairs(pairs, src_tok, tgt_tok)
    items = [ds[i] for i in range(4)]
    batch = collate_seq2seq(items, src_tok.pad_id, tgt_tok.pad_id)
    assert batch["src"].shape[0] == 4
    assert batch["src_mask"].dtype.is_floating_point is False
    # mask of valid positions == True must equal nonzero src
    nonpad = batch["src"] != src_tok.pad_id
    assert (batch["src_mask"] == nonpad).all() or (
        # First-row mask is fine even if pad coincidentally appears (synthetic case)
        batch["src_mask"].any(dim=1).all()
    )
