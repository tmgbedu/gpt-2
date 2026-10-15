import hashlib
import json

import numpy as np
import pytest
import torch

from data import encode_documents, file_sha256, get_batch, load_tokens, write_token_file
from test_tokenizer import FIXTURE_VOCAB_SIZE, _python_corpus
from tokenizer import EOT_ID, train_tokenizer


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(_python_corpus(), vocab_size=FIXTURE_VOCAB_SIZE)


DOCS = ["def f(x):\n    return x\n", "class A:\n    pass\n", "x = 1\n"]


# --- encoding ---------------------------------------------------------------


def test_each_document_ends_with_one_eot_and_decodes_back(tok):
    encoded = list(encode_documents(tok, DOCS))
    assert len(encoded) == len(DOCS)
    for ids, text in zip(encoded, DOCS):
        assert ids.dtype == np.uint16
        assert ids[-1] == EOT_ID
        assert (ids[:-1] != EOT_ID).all()
        assert tok.decode(ids[:-1].tolist(), skip_special_tokens=False) == text


def test_special_token_text_in_code_is_not_a_special_token(tok):
    text = "# split on <|endoftext|> or <|user|>\nSEP = '<|endoftext|>'\n"
    (ids,) = encode_documents(tok, [text])
    assert (ids[:-1] >= 9).all()  # none of the 9 special IDs inside the document
    assert tok.decode(ids[:-1].tolist(), skip_special_tokens=False) == text


def test_encoding_does_not_change_the_callers_tokenizer(tok):
    list(encode_documents(tok, DOCS))
    assert tok.encode("<|endoftext|>").ids == [EOT_ID]


# --- writing ----------------------------------------------------------------


def _arrays(*lengths, start=1):
    out, nxt = [], start
    for n in lengths:
        out.append(np.arange(nxt, nxt + n, dtype=np.uint16))
        nxt += n
    return out


def test_writes_tokens_and_metadata(tmp_path):
    path = tmp_path / "train.bin"
    meta = write_token_file(iter(_arrays(3, 4)), path, max_tokens=100, meta={"seed": 7})

    np.testing.assert_array_equal(np.fromfile(path, dtype=np.uint16), np.arange(1, 8))
    saved = json.loads((tmp_path / "train.json").read_text())
    assert saved == meta
    assert saved["seed"] == 7
    assert saved["token_count"] == 7
    assert saved["documents"] == 2
    assert saved["dtype"] == "uint16"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.bin", "train.json"]  # no temp files left


def test_stops_at_the_document_that_reaches_max_tokens(tmp_path):
    path = tmp_path / "train.bin"
    meta = write_token_file(iter(_arrays(4, 4, 4)), path, max_tokens=6, meta={})
    # The 2nd document crosses 6 tokens, so it is kept whole and writing stops there.
    assert meta["token_count"] == 8
    assert meta["documents"] == 2


def test_failure_leaves_no_partial_file_and_keeps_previous_one(tmp_path):
    path = tmp_path / "train.bin"
    write_token_file(iter(_arrays(3)), path, max_tokens=100, meta={"run": 1})

    def broken():
        yield from _arrays(5)
        raise ConnectionError("stream dropped")

    with pytest.raises(ConnectionError):
        write_token_file(broken(), path, max_tokens=100, meta={"run": 2})

    np.testing.assert_array_equal(np.fromfile(path, dtype=np.uint16), np.arange(1, 4))
    assert json.loads((tmp_path / "train.json").read_text())["run"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.bin", "train.json"]


def test_file_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"keera")
    assert file_sha256(p) == hashlib.sha256(b"keera").hexdigest()


# --- loading batches --------------------------------------------------------


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "train.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    return path


def test_load_tokens_is_a_uint16_memmap(token_file):
    tokens = load_tokens(token_file)
    assert isinstance(tokens, np.memmap)
    assert tokens.dtype == np.uint16
    assert len(tokens) == 1000


def test_batch_targets_are_inputs_shifted_by_one(token_file):
    tokens = load_tokens(token_file)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(tokens, batch_size=8, block_size=32, generator=gen)
    assert x.shape == y.shape == (8, 32)
    assert x.dtype == y.dtype == torch.long
    # tokens are 0, 1, 2, ... so every row is consecutive and y = x + 1
    torch.testing.assert_close(y, x + 1)
    torch.testing.assert_close(x[:, 1:], x[:, :-1] + 1)


def test_batches_stay_in_bounds(token_file):
    tokens = load_tokens(token_file)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        x, y = get_batch(tokens, batch_size=16, block_size=999, generator=gen)
        assert x.min() >= 0 and y.max() <= 999


def test_batch_rejects_block_longer_than_data(token_file):
    with pytest.raises(ValueError, match="block_size"):
        get_batch(load_tokens(token_file), batch_size=1, block_size=1000)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_batch_on_mps(token_file):
    x, y = get_batch(load_tokens(token_file), batch_size=2, block_size=16, device="mps")
    assert x.device.type == y.device.type == "mps"
