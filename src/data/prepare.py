"""Tokenize codeparrot into uint16 token files for pretraining.

    uv run python -m data.prepare --train-tokens 10000000 --out-dir data/smoke

Writes <out-dir>/train.bin, val.bin and a .json of metadata next to each.
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from data import encode_documents, file_sha256, write_token_file
from tokenizer import load_tokenizer

TRAIN_DATASET = "codeparrot/codeparrot-clean-train"
VALID_DATASET = "codeparrot/codeparrot-clean-valid"


def prepare(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    tokenizer_path: Path,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
    settings: dict,
) -> dict:
    tok = load_tokenizer(tokenizer_path)
    base = {**settings, "tokenizer": str(tokenizer_path), "tokenizer_sha256": file_sha256(tokenizer_path)}
    out = {}
    # Validation first: it is small, so a broken setup fails fast.
    for split, texts, max_tokens in (("val", valid_texts, val_tokens), ("train", train_texts, train_tokens)):
        skipped = 0  # among files read; encoding reads up to one batch past the last file written

        def small_enough(texts: Iterable[str]) -> Iterator[str]:
            nonlocal skipped
            for text in texts:
                if len(text.encode("utf-8")) > settings["max_bytes"]:
                    skipped += 1
                    continue
                yield text

        out[split] = write_token_file(
            encode_documents(tok, small_enough(texts)),
            out_dir / f"{split}.bin",
            max_tokens,
            lambda: {**base, "split": split, "skipped_too_large": skipped},
        )
    return out


def _revision(name: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(name).sha


def _stream(name: str, revision: str, seed: int | None = None, buffer_size: int = 0) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True, revision=revision)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    return (example["content"] for example in ds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-bytes", type=int, default=100_000, help="skip files over this many UTF-8 bytes")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/smoke"))
    args = parser.parse_args()

    train_rev, valid_rev = _revision(TRAIN_DATASET), _revision(VALID_DATASET)
    settings = {
        "train_dataset": TRAIN_DATASET,
        "train_revision": train_rev,
        "valid_dataset": VALID_DATASET,
        "valid_revision": valid_rev,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "max_bytes": args.max_bytes,
        "fim": None,  # no fill-in-the-middle transform yet
    }
    meta = prepare(
        _stream(TRAIN_DATASET, train_rev, args.seed, args.shuffle_buffer),
        _stream(VALID_DATASET, valid_rev),
        args.tokenizer,
        args.out_dir,
        args.train_tokens,
        args.val_tokens,
        settings,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
