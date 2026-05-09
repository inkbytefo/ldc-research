from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import WordTokenizer

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_scan_line(line: str) -> tuple[str, str]:
    line = line.strip()
    if "IN:" not in line or "OUT:" not in line:
        raise ValueError(f"Malformed SCAN line: {line!r}")
    in_part, out_part = line.split("OUT:", 1)
    src = in_part.replace("IN:", "").strip()
    tgt = out_part.strip()
    return src, tgt


# ---------------------------------------------------------------------------
# Synthetic pairs (template-based, no external data needed)
# ---------------------------------------------------------------------------

_ACTIONS = ["walk", "run", "jump", "look", "turn"]
_ADVERBS = ["twice", "thrice", "around", "opposite"]
_DIRECTIONS = ["left", "right"]

_ACTION_MAP = {
    "walk": "I_WALK",
    "run": "I_RUN",
    "jump": "I_JUMP",
    "look": "I_LOOK",
    "turn": "I_TURN",
}
_DIR_MAP = {"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"}


def _repeat(cmd: str, n: int) -> str:
    return " ".join([cmd] * n)


def synthetic_scan_pairs(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    while len(pairs) < n:
        action = rng.choice(_ACTIONS)
        out_token = _ACTION_MAP[action]
        variant = rng.random()
        if variant < 0.4:
            src = action
            tgt = out_token
        elif variant < 0.65:
            src = f"{action} twice"
            tgt = _repeat(out_token, 2)
        elif variant < 0.80:
            src = f"{action} thrice"
            tgt = _repeat(out_token, 3)
        elif variant < 0.90:
            d = rng.choice(_DIRECTIONS)
            src = f"turn {d}"
            tgt = _DIR_MAP[d]
        else:
            d = rng.choice(_DIRECTIONS)
            src = f"{action} and turn {d}"
            tgt = f"{out_token} {_DIR_MAP[d]}"
        pairs.append((src, tgt))
    return pairs


# ---------------------------------------------------------------------------
# Dataset & Collate
# ---------------------------------------------------------------------------

class SCANDataset(Dataset):
    def __init__(
        self,
        src_ids: list[list[int]],
        tgt_ids: list[list[int]],
    ) -> None:
        assert len(src_ids) == len(tgt_ids)
        self._src = src_ids
        self._tgt = tgt_ids

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[str, str]],
        src_tok: WordTokenizer,
        tgt_tok: WordTokenizer,
    ) -> "SCANDataset":
        src_ids = [src_tok.encode(s, add_sos=True, add_eos=True) for s, _ in pairs]
        tgt_ids = [tgt_tok.encode(t, add_sos=True, add_eos=True) for _, t in pairs]
        return cls(src_ids, tgt_ids)

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "src_ids": torch.tensor(self._src[idx], dtype=torch.long),
            "tgt_ids": torch.tensor(self._tgt[idx], dtype=torch.long),
        }


def collate_seq2seq(
    items: list[dict[str, Tensor]],
    src_pad_id: int,
    tgt_pad_id: int,
) -> dict[str, Any]:
    src_seqs = [it["src_ids"] for it in items]
    tgt_seqs = [it["tgt_ids"] for it in items]

    src = _pad_sequences(src_seqs, src_pad_id)
    tgt = _pad_sequences(tgt_seqs, tgt_pad_id)
    src_mask = src != src_pad_id
    tgt_mask = tgt != tgt_pad_id

    return {"src": src, "tgt": tgt, "src_mask": src_mask, "tgt_mask": tgt_mask}


def _pad_sequences(seqs: list[Tensor], pad_id: int) -> Tensor:
    max_len = max(s.size(0) for s in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : s.size(0)] = s
    return batch


# ---------------------------------------------------------------------------
# SCAN file loader (requires downloaded SCAN dataset)
# ---------------------------------------------------------------------------

def _load_scan_file(path: Path, src_tok: WordTokenizer | None, tgt_tok: WordTokenizer | None):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(parse_scan_line(line))

    if src_tok is None:
        src_tok = WordTokenizer.from_corpus(s for s, _ in pairs)
    if tgt_tok is None:
        tgt_tok = WordTokenizer.from_corpus(t for _, t in pairs)

    return pairs, src_tok, tgt_tok


def build_scan_loaders(
    split: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    max_src_len: int,
    max_tgt_len: int,
    val_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, WordTokenizer, WordTokenizer]:
    root = Path(data_root)
    train_file = root / f"tasks_train_{split}.txt"
    test_file = root / f"tasks_test_{split}.txt"

    train_pairs, src_tok, tgt_tok = _load_scan_file(train_file, None, None)
    test_pairs, _, _ = _load_scan_file(test_file, src_tok, tgt_tok)

    rng = random.Random(seed)
    rng.shuffle(train_pairs)
    n_val = max(1, int(len(train_pairs) * val_fraction))
    val_pairs = train_pairs[:n_val]
    train_pairs = train_pairs[n_val:]

    def _make_loader(pairs, shuffle):
        ds = SCANDataset.from_pairs(pairs, src_tok, tgt_tok)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda b: collate_seq2seq(b, src_tok.pad_id, tgt_tok.pad_id),
        )

    return (
        _make_loader(train_pairs, shuffle=True),
        _make_loader(val_pairs, shuffle=False),
        _make_loader(test_pairs, shuffle=False),
        src_tok,
        tgt_tok,
    )
