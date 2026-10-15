import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tokenizer import EOT_ID

DTYPE = np.uint16  # vocab 16,384 < 65,536


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_documents(
    tok: Tokenizer, texts: Iterable[str], batch_size: int = 1000
) -> Iterator[np.ndarray]:
    """Yield one uint16 array per document: its tokens followed by one EOT.

    Special-token strings that appear inside code (e.g. a comment mentioning
    "<|endoftext|>") are encoded as ordinary text; the only boundary IDs come
    from the EOT we append.
    """
    raw = Tokenizer.from_str(tok.to_str())  # private copy, so the caller's tokenizer is unchanged
    raw.encode_special_tokens = True
    texts = iter(texts)
    while batch := list(islice(texts, batch_size)):
        for enc in raw.encode_batch(batch):
            ids = np.empty(len(enc.ids) + 1, dtype=DTYPE)
            ids[:-1] = enc.ids
            ids[-1] = EOT_ID
            yield ids


def _atomic_temp(path: Path, suffix: str):
    return tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=suffix, delete=False
    )


def write_token_file(
    arrays: Iterable[np.ndarray],
    path: str | Path,
    max_tokens: int,
    meta: dict | Callable[[], dict],
) -> dict:
    """Write documents to `path` (.bin) and metadata to the sibling .json.

    Stops after the document that reaches `max_tokens` (documents are never cut).
    Both files are written to temporaries in the same directory and renamed only
    once everything succeeded, so a crash never leaves a partial dataset.
    `meta` may be a function, called after the tokens are written (for counts
    only known at the end).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".json")
    temps: list[str] = []
    try:
        with _atomic_temp(path, ".tmp") as f:
            temps.append(f.name)
            token_count = documents = 0
            for ids in arrays:
                f.write(np.asarray(ids, dtype=DTYPE).tobytes())
                token_count += len(ids)
                documents += 1
                if token_count >= max_tokens:
                    break
        full_meta = {**(meta() if callable(meta) else meta), "token_count": token_count, "documents": documents, "dtype": "uint16"}
        with _atomic_temp(meta_path, ".tmp") as f:
            temps.append(f.name)
            f.write(json.dumps(full_meta, indent=2).encode())
        os.replace(temps[0], path)
        os.replace(temps[1], meta_path)
        return full_meta
    finally:
        for t in temps:
            if os.path.exists(t):
                os.remove(t)


def load_tokens(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=DTYPE, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows from the token stream: x = tokens[i:i+T], y = tokens[i+1:i+T+1]."""
    if block_size >= len(data):
        raise ValueError(f"block_size {block_size} needs more than {len(data)} tokens")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.from_numpy(np.stack([data[i : i + block_size] for i in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block_size] for i in starts]).astype(np.int64))
    return x.to(device), y.to(device)
