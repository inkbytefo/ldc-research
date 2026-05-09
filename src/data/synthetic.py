from __future__ import annotations

import random
import string

import torch
from torch.utils.data import DataLoader

from src.data.scan import SCANDataset, collate_seq2seq, synthetic_scan_pairs
from src.data.tokenizer import WordTokenizer


def _synthetic_letter_pairs(n: int, n_letters: int, seed: int) -> list[tuple[str, str]]:
    """Letter-reversal task: src='a b c' -> tgt='c b a'"""
    rng = random.Random(seed)
    letters = list(string.ascii_lowercase[:n_letters])
    pairs = []
    for _ in range(n):
        length = rng.randint(2, max(2, n_letters))
        seq = [rng.choice(letters) for _ in range(length)]
        pairs.append((" ".join(seq), " ".join(reversed(seq))))
    return pairs


def build_synthetic_loaders(
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int,
    n_letters: int = 4,
) -> tuple[DataLoader, DataLoader, WordTokenizer]:
    train_pairs = synthetic_scan_pairs(n_train, seed=seed)
    val_pairs = synthetic_scan_pairs(n_val, seed=seed + 1)

    all_pairs = train_pairs + val_pairs
    tok = WordTokenizer.from_corpus(s for pair in all_pairs for s in pair)

    def _make_loader(pairs, shuffle):
        ds = SCANDataset.from_pairs(pairs, tok, tok)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: collate_seq2seq(b, tok.pad_id, tok.pad_id),
        )

    return _make_loader(train_pairs, shuffle=True), _make_loader(val_pairs, shuffle=False), tok
