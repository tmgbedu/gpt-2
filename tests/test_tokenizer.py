import pytest

from tokenizer import (
    SPECIAL_TOKENS,
    compression_stats,
    load_tokenizer,
    select_files,
    train_tokenizer,
)

FIXTURE_VOCAB_SIZE = 400


def _python_corpus() -> list[str]:
    """Small synthetic corpus with plenty of repeated 4/8/12-space indentation."""
    files = []
    for i in range(300):
        files.append(
            f"class Model{i % 11}:\n"
            f"    def func_{i % 37}(self, x, y={i}):\n"
            f"        if x > y:\n"
            f"            return x - y\n"
            f"        for item in range(y):\n"
            f"            x += item\n"
            f"        return x\n"
            "\n"
            f"def helper_{i % 23}(value):\n"
            f"    # compute the result\n"
            f"    result = value * {i % 7} + 1\n"
            f"    return result\n"
        )
    return files


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(_python_corpus(), vocab_size=FIXTURE_VOCAB_SIZE)


def _roundtrip(tok, text: str) -> str:
    return tok.decode(tok.encode(text).ids, skip_special_tokens=False)


def test_vocab_size_is_exact(tok):
    assert tok.get_vocab_size() == FIXTURE_VOCAB_SIZE


def test_training_fails_when_corpus_cannot_reach_vocab_size():
    with pytest.raises(ValueError, match="vocab size"):
        train_tokenizer(["x = 1\n"], vocab_size=FIXTURE_VOCAB_SIZE)


def test_special_tokens_have_the_first_ids(tok):
    assert len(SPECIAL_TOKENS) == 9
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(token) == expected_id


def test_each_special_token_encodes_to_one_id(tok):
    for token in SPECIAL_TOKENS:
        assert tok.encode(token).ids == [tok.token_to_id(token)]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "def f():\n    return 1\n",
        "def f():\n\treturn 1\n",
        "if x:\n    if y:\n\t\tpass\n",
        "line one\r\nline two\r\n",
        "   leading spaces and trailing   \n\n\n",
        "name = 'héllo wörld'  # ünïcode\n",
        "greeting = '你好, 世界 🐍🚀'\n",
        "x = 12345678901234567890\n",
    ],
)
def test_roundtrip_is_exact(tok, text):
    assert _roundtrip(tok, text) == text


@pytest.mark.parametrize(
    "text",
    [
        "# docs mention <|endoftext|> here\nx = 1\n",
        "s = '<|user|>hi<|assistant|>'\n",
        'FIM = "<|fim_prefix|><|fim_suffix|><|fim_middle|>"\n',
    ],
)
def test_roundtrip_keeps_literal_special_tokens(tok, text):
    assert _roundtrip(tok, text) == text


@pytest.mark.parametrize("indent", [" " * 4, " " * 8])
def test_indentation_run_is_one_token(tok, indent):
    text = f"def f(x):\n{indent}return x\n"
    tokens = tok.encode(text).tokens
    # ByteLevel maps a space to "Ġ", so the indent run appears as one token of Ġs.
    assert "Ġ" * len(indent) in tokens


def test_indentation_is_split_from_following_code(tok):
    tokens = tok.encode("if x:\n        return x\n").tokens
    assert "Ġ" * 8 in tokens
    assert not any(t.startswith("Ġ" * 8) and len(t) > 8 for t in tokens)


def test_save_and_load_gives_same_encoding(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    loaded = load_tokenizer(path)
    text = "def f(x):\n    return x  # <|endoftext|>\n"
    assert loaded.encode(text).ids == tok.encode(text).ids


def test_compression_stats(tok):
    texts = ["def f(x):\n    return x\n", "if x:\n        pass\n"]
    stats = compression_stats(tok, texts)
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tok.encode(t).ids) for t in texts)
    assert stats["files"] == 2
    assert stats["chars"] == total_chars
    assert stats["tokens"] == total_tokens
    assert stats["chars_per_token"] == pytest.approx(total_chars / total_tokens)
    assert stats["indent_runs"] == 2
    assert stats["tokens_per_indent_run"] == pytest.approx(1.0)


def test_select_files_skips_oversized_by_utf8_bytes_and_stops_at_limit():
    small = "x = 1\n"
    # 30 characters but 90 UTF-8 bytes.
    multibyte = "你" * 30
    contents = [small, multibyte, small, small, small]
    selected = list(select_files(contents, max_bytes=50, num_files=3))
    assert selected == [small, small, small]
