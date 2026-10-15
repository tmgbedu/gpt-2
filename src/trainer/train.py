"""Pretrain Keera on a prepared token dataset. Rerun the same command to resume.

    uv run python -m trainer.train --data-dir data/smoke --out-dir checkpoints/smoke
"""

import argparse
import os
from dataclasses import fields

# Fail with "out of memory" instead of letting macOS swap GPU memory to disk (which
# stalls the GPU). Must be set before the MPS allocator starts.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from model.transformer import ModelConfig
from trainer import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    parser.add_argument("--stop-at", type=int, default=None, help="end this call after this step")
    args = vars(parser.parse_args())
    stop_at = args.pop("stop_at")
    train(TrainConfig(**args), ModelConfig(), stop_at=stop_at)


if __name__ == "__main__":
    main()
