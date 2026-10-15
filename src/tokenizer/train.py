"""Train the tokenizer on codeparrot and evaluate it on held-out files.

    uv run python -m tokenizer.train
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizer import compression_stats, select_files, train_tokenizer

TRAIN_DATASET = "codeparrot/codeparrot-clean-train"
VALID_DATASET = "codeparrot/codeparrot-clean-valid"


def run(
    train_texts: Iterator[str],
    valid_texts: Iterator[str],
    out_dir: Path,
    settings: dict,
) -> dict:
    train_files = 0

    def counted() -> Iterator[str]:
        nonlocal train_files
        for text in train_texts:
            train_files += 1
            yield text

    tok = train_tokenizer(counted(), vocab_size=settings["vocab_size"])
    meta = {"settings": settings, "train_files": train_files}

    # Save before validating, so a failed validation stream doesn't lose the trained tokenizer.
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    config_path = out_dir / "training_config.json"
    config_path.write_text(json.dumps(meta, indent=2) + "\n")

    meta["valid_stats"] = compression_stats(tok, valid_texts)
    config_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _stream(name: str, seed: int | None = None, buffer_size: int = 0) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    return (example["content"] for example in ds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-files", type=int, default=200_000)
    parser.add_argument("--max-bytes", type=int, default=100_000, help="UTF-8 bytes per file")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--valid-files", type=int, default=1_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    settings = {
        "train_dataset": TRAIN_DATASET,
        "valid_dataset": VALID_DATASET,
        "num_files": args.num_files,
        "max_bytes": args.max_bytes,
        "vocab_size": args.vocab_size,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "valid_files": args.valid_files,
    }
    train_texts = select_files(
        _stream(TRAIN_DATASET, args.seed, args.shuffle_buffer), args.max_bytes, args.num_files
    )
    valid_texts = select_files(_stream(VALID_DATASET), args.max_bytes, args.valid_files)

    meta = run(train_texts, valid_texts, args.out, settings)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
