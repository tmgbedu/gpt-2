"""Instruction data for fine-tuning: chat-formatted, masked, packed into fixed-length rows.

    uv run python -m data.sft --out-dir data/sft

One example becomes   <|user|> instruction <|end|> <|assistant|> response <|end|>
with a mask that is 1 only on the response and its closing <|end|>: the loss is the mean
of -log p(token) over those positions, so the model learns to answer, not to ask.

Examples are packed whole into rows of `row_len` tokens (block_size + 1, so a row gives
x = row[:-1] and y = row[1:]); the leftover space is <|pad|>, never learned.
Writes <out-dir>/{train,val}_{tokens,mask}.npy and meta.json.
"""

import argparse
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data import file_sha256
from tokenizer import PAD_ID, SPECIAL_TOKENS, load_tokenizer

USER_ID = SPECIAL_TOKENS.index("<|user|>")
ASSISTANT_ID = SPECIAL_TOKENS.index("<|assistant|>")
END_ID = SPECIAL_TOKENS.index("<|end|>")

DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
OPEN_ROWS = 64  # rows still searched for space when packing (first fit over the newest rows)

_plain_copies: dict[int, Tokenizer] = {}


def _plain(tok: Tokenizer) -> Tokenizer:
    """A copy that encodes "<|end|>" written inside text as ordinary characters."""
    if id(tok) not in _plain_copies:
        raw = Tokenizer.from_str(tok.to_str())
        raw.encode_special_tokens = True
        _plain_copies[id(tok)] = raw
    return _plain_copies[id(tok)]


def chat_prompt_ids(tok: Tokenizer, instruction: str) -> list[int]:
    """<|user|> instruction <|end|> <|assistant|>: what the model continues with its answer."""
    return [USER_ID, *_plain(tok).encode(instruction).ids, END_ID, ASSISTANT_ID]


def format_example(tok: Tokenizer, instruction: str, response: str) -> tuple[list[int], list[int]]:
    """Token ids and learn-mask for one instruction/response pair."""
    prompt = chat_prompt_ids(tok, instruction)
    answer = [*_plain(tok).encode(response).ids, END_ID]
    return prompt + answer, [0] * len(prompt) + [1] * len(answer)


def pack_rows(examples: Iterable[tuple[list[int], list[int]]], row_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Put each example whole into the first recent row with room; pad the rest.

    Returns tokens [rows, row_len] (uint16), mask [rows, row_len] (uint8), and how many
    examples were dropped for being longer than a row.
    """
    rows: list[tuple[list[int], list[int]]] = []
    dropped = 0
    for ids, mask in examples:
        if len(ids) > row_len:
            dropped += 1
            continue
        for row_ids, row_mask in rows[-OPEN_ROWS:]:
            if len(row_ids) + len(ids) <= row_len:
                row_ids.extend(ids)
                row_mask.extend(mask)
                break
        else:
            rows.append((list(ids), list(mask)))
    tokens = np.full((len(rows), row_len), PAD_ID, dtype=np.uint16)
    masks = np.zeros((len(rows), row_len), dtype=np.uint8)
    for i, (row_ids, row_mask) in enumerate(rows):
        tokens[i, : len(row_ids)] = row_ids
        masks[i, : len(row_mask)] = row_mask
    return tokens, masks, dropped


def sft_batch(tokens: np.ndarray, mask: np.ndarray, rows: list[int],
              device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """x = row[:-1]; y = row[1:], with -100 (skipped by the loss) wherever the mask is 0."""
    t = torch.from_numpy(np.asarray(tokens[rows], dtype=np.int64))
    m = torch.from_numpy(np.asarray(mask[rows]))
    x, y = t[:, :-1], t[:, 1:].clone()
    y[m[:, 1:] == 0] = -100
    if torch.device(device).type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def _save_npy(array: np.ndarray, path: Path) -> None:
    """Write to a temp file, then rename: never a half-written array."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def prepare_sft(rows: Iterable[dict], tok: Tokenizer, out_dir: str | Path, row_len: int = 1025,
                val_fraction: float = 0.02, seed: int = 1234, settings: dict | None = None) -> dict:
    """Shuffle, split, format and pack instruction rows; write the arrays and meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    order = np.random.default_rng(seed).permutation(len(rows))
    n_val = round(len(rows) * val_fraction)
    meta: dict = {**(settings or {}), "row_len": row_len, "val_fraction": val_fraction, "seed": seed}
    for split, idx in (("val", order[:n_val]), ("train", order[n_val:])):
        examples = (format_example(tok, rows[i]["instruction"], rows[i]["response"]) for i in idx)
        tokens, mask, dropped = pack_rows(examples, row_len)
        _save_npy(tokens, out_dir / f"{split}_tokens.npy")
        _save_npy(mask, out_dir / f"{split}_mask.npy")
        meta[split] = {
            "examples": int(len(idx) - dropped), "dropped_too_long": int(dropped), "rows": int(len(tokens)),
            "tokens": int((tokens != PAD_ID).sum()), "answer_tokens": int(mask.sum()),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_sft(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    return (np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r"),
            np.load(data_dir / f"{split}_mask.npy", mmap_mode="r"))


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--row-len", type=int, default=1025, help="block_size + 1")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    revision = HfApi().dataset_info(DATASET).sha
    rows = load_dataset(DATASET, split="train", revision=revision)
    settings = {"dataset": DATASET, "revision": revision, "tokenizer": str(args.tokenizer),
                "tokenizer_sha256": file_sha256(args.tokenizer),
                "format": "<|user|>{instruction}<|end|><|assistant|>{response}<|end|>"}
    meta = prepare_sft(rows, load_tokenizer(args.tokenizer), args.out_dir, args.row_len,
                       args.val_fraction, args.seed, settings)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
