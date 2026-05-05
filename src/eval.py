"""Evaluation: token-level loss/accuracy + sequence-level greedy accuracy.

Per the roadmap (Faz 2 G2), we report both:
  - token_acc : per-token argmax matches under teacher forcing
  - seq_acc   : full sequence equal under greedy generation
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.transformer import TransformerSeq2Seq
from src.data.tokenizer import WordTokenizer
from src.ldc.model import LDCModel


def _strip_after_eos(ids: list[int], eos_id: int, pad_id: int) -> list[int]:
    out: list[int] = []
    for x in ids:
        if x == eos_id:
            break
        if x == pad_id:
            continue
        out.append(x)
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    src_tok: WordTokenizer,
    tgt_tok: WordTokenizer,
    max_batches: int = 0,
    num_gen_examples: int = 0,
    max_gen_len: int = 128,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tok_correct = 0
    total_tok = 0
    total_seq_correct = 0
    total_seq = 0
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"])
        total_loss += float(out["loss"].item())

        # Token-level accuracy from teacher-forced logits
        tgt_out = batch["tgt"][:, 1:]
        tgt_out_mask = batch["tgt_mask"][:, 1:]
        preds = out["logits"].argmax(-1)
        valid = tgt_out_mask
        total_tok_correct += int(((preds == tgt_out) & valid).sum().item())
        total_tok += int(valid.sum().item())

        # Sequence-level greedy accuracy
        if isinstance(model, LDCModel):
            gen = model.generate(
                batch["src"], batch["src_mask"],
                sos_id=tgt_tok.sos_id, eos_id=tgt_tok.eos_id, max_len=max_gen_len,
            )
        elif isinstance(model, TransformerSeq2Seq):
            gen = model.generate(
                batch["src"], batch["src_mask"],
                sos_id=tgt_tok.sos_id, eos_id=tgt_tok.eos_id, max_len=max_gen_len,
            )
        else:
            gen = None

        if gen is not None:
            B = batch["src"].size(0)
            for b in range(B):
                # Generated includes <sos> at position 0; strip it.
                gen_ids = gen[b].tolist()[1:]
                gen_clean = _strip_after_eos(gen_ids, tgt_tok.eos_id, tgt_tok.pad_id)
                # Reference: skip <sos>, stop at <eos>
                ref_ids = batch["tgt"][b].tolist()[1:]
                ref_clean = _strip_after_eos(ref_ids, tgt_tok.eos_id, tgt_tok.pad_id)
                if gen_clean == ref_clean:
                    total_seq_correct += 1
                total_seq += 1

        n_batches += 1

    metrics = {
        "loss": total_loss / max(1, n_batches),
        "token_acc": total_tok_correct / max(1, total_tok),
        "seq_acc": total_seq_correct / max(1, total_seq) if total_seq > 0 else 0.0,
    }
    return metrics
