"""Pretrain Keera on a prepared token dataset. Rerun the same command to resume.

    uv run python -m trainer.train --data-dir data/smoke --out-dir checkpoints/smoke

On a machine with 8 NVIDIA GPUs, start one process per GPU with torchrun:

    uv run torchrun --nproc-per-node 8 -m trainer.train --device cuda --precision bf16 ...
"""

import argparse
import os
from dataclasses import fields

# Fail with "out of memory" instead of letting macOS swap GPU memory to disk (which
# stalls the GPU). Must be set before the MPS allocator starts.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import torch
import torch.distributed as dist

from model.transformer import ModelConfig
from trainer import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):  # --compile / --no-compile
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--stop-at", type=int, default=None, help="end this call after this step")
    args = vars(parser.parse_args())
    stop_at = args.pop("stop_at")
    cfg = TrainConfig(**args)

    # torchrun sets WORLD_SIZE (and RANK, LOCAL_RANK, MASTER_ADDR...) for every process it starts.
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        if cfg.device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl" if cfg.device == "cuda" else "gloo")
    try:
        train(cfg, ModelConfig(), stop_at=stop_at)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
