"""Byte-level BPE tokenizer for Python source code."""

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Order matters: these get ids 0..8.
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]
EOT_ID = 0
PAD_ID = 1

# GPT-4-style split pattern, with one addition at the front: a run of spaces/tabs at the
# start of a line is its own pre-token, so indentation never merges with the code after it.
SPLIT_PATTERN = (
    r"(?:^|(?<=\n))[ \t]+"
    r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_INDENT_RUN = re.compile(r"^[ \t]+(?=\S)", re.MULTILINE)


def _new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    # No normalizer: code must round-trip byte for byte.
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a tokenizer whose vocab is exactly ``vocab_size``, special tokens included."""
    tok = _new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS],
        # All 256 byte symbols, so any Unicode text stays representable.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(
            f"Trained vocab size is {tok.get_vocab_size()}, expected {vocab_size}: "
            "the training corpus has too few distinct pairs to merge."
        )
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def select_files(contents: Iterable[str], max_bytes: int, num_files: int) -> Iterator[str]:
    """Yield up to ``num_files`` files whose UTF-8 size is at most ``max_bytes``."""
    accepted = 0
    for content in contents:
        if accepted >= num_files:
            return
        if len(content.encode("utf-8")) > max_bytes:
            continue
        accepted += 1
        yield content


def compression_stats(tok: Tokenizer, texts: Iterable[str]) -> dict[str, float]:
    """Characters per token overall, and tokens per indentation run."""
    files = chars = tokens = indent_runs = indent_tokens = 0
    for text in texts:
        files += 1
        chars += len(text)
        tokens += len(tok.encode(text).ids)
        for match in _INDENT_RUN.finditer(text):
            indent_runs += 1
            indent_tokens += len(tok.encode(match.group()).ids)
    return {
        "files": files,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "indent_runs": indent_runs,
        "tokens_per_indent_run": indent_tokens / indent_runs if indent_runs else 0.0,
    }
