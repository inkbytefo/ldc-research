from src.data.tokenizer import (
    EOS_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    UNK_TOKEN,
    WordTokenizer,
)


def test_specials_at_known_indices():
    tok = WordTokenizer.from_corpus(["hello world", "foo bar"])
    assert tok.vocab[tok.pad_id] == PAD_TOKEN
    assert tok.vocab[tok.sos_id] == SOS_TOKEN
    assert tok.vocab[tok.eos_id] == EOS_TOKEN
    assert tok.vocab[tok.unk_id] == UNK_TOKEN


def test_encode_decode_roundtrip():
    tok = WordTokenizer.from_corpus(["alpha beta gamma"])
    ids = tok.encode("alpha gamma", add_sos=True, add_eos=True)
    assert ids[0] == tok.sos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "alpha gamma"


def test_unk_for_unseen_token():
    tok = WordTokenizer.from_corpus(["alpha"])
    ids = tok.encode("alpha mystery")
    assert tok.unk_id in ids


def test_decode_stops_at_eos():
    tok = WordTokenizer.from_corpus(["a b c"])
    ids = [tok.sos_id, tok.encode("a")[0], tok.eos_id, tok.encode("b")[0]]
    assert tok.decode(ids) == "a"
