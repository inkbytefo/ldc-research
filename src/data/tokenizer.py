from __future__ import annotations

from typing import Iterable

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

_SPECIALS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class WordTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self._vocab = vocab
        self._token2id: dict[str, int] = {t: i for i, t in enumerate(vocab)}

    @classmethod
    def from_corpus(cls, sentences: Iterable[str]) -> "WordTokenizer":
        words: set[str] = set()
        for s in sentences:
            words.update(s.split())
        vocab = _SPECIALS + sorted(words - set(_SPECIALS))
        return cls(vocab)

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._token2id[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self._token2id[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token2id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token2id[UNK_TOKEN]

    def encode(self, text: str, *, add_sos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self._token2id.get(w, self.unk_id) for w in text.split()]
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i == self.eos_id:
                break
            tok = self._vocab[i] if 0 <= i < len(self._vocab) else UNK_TOKEN
            if tok in (PAD_TOKEN, SOS_TOKEN):
                continue
            tokens.append(tok)
        return " ".join(tokens)
