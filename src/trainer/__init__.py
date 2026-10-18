import contextlib
from collections.abc import Callable
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_batch, load_tokens, random_starts, windows
from data.sft import load_sft, sft_batch
from model.transformer import Keera, ModelConfig

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TrainConfig:
    data_dir: str = "data/smoke"
    out_dir: str = "checkpoints/smoke"
    micro_batch: int = 2  # sequences per forward/backward; M3 Pro 18 GB is GPU-bound already at 1
    block_size: int = 1024
    grad_accum: int = 16  # microbatches per optimizer step (32K tokens)
    max_steps: int = 1000  # optimizer steps
    warmup_steps: int = 100
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 100
    keep_every: int = 0  # also keep weights-only step_NNNNNN.pt every N steps; 0 = off
    log_interval: int = 10
    seed: int = 1234
    device: str = "mps"  # or "cuda" (each process uses GPU number LOCAL_RANK) or "cpu"
    precision: str = "fp32"  # or "bf16" autocast; measured no faster on M3 Pro MPS
    compile: bool = False  # torch.compile the model: faster on NVIDIA GPUs after a slow first step
    data_format: str = "tokens"  # "tokens": random windows of train.bin; "sft": masked rows from data.sft
    init_from: str = ""  # start from this checkpoint's weights (fresh optimizer, step 0), e.g. for SFT


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only; RMSNorm weights (vectors) are not decayed."""
    params = list(model.parameters())  # deduplicated, so the tied matrix appears once
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
    )


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for optimizer step `step` (0-based): linear warmup, then cosine to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - 1 - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.max_lr - cfg.min_lr)


def _autocast(device: torch.device, precision: str):
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def forward_backward(model: nn.Module, batches: list[Batch], precision: str) -> float:
    """Accumulate gradients over microbatches; returns the mean loss.

    Each microbatch loss is divided by the number of microbatches, so the summed
    gradient equals the gradient of the mean loss over all of them.

    With several GPUs (`model` wrapped in DDP), the last backward also averages the
    gradient across GPUs; the earlier ones skip that (`no_sync`), since only the
    final sum needs sending. The returned loss is then averaged across GPUs too.
    """
    device = _device(model)
    total = torch.zeros((), device=device)
    for i, (x, y) in enumerate(batches):
        last = i == len(batches) - 1
        sync = contextlib.nullcontext() if last or not isinstance(model, DDP) else model.no_sync()
        with sync:
            with _autocast(device, precision):
                _, loss = model(x, y)
            loss = loss / len(batches)
            loss.backward()
        total += loss.detach()
    if isinstance(model, DDP):
        dist.all_reduce(total)  # sum over GPUs (AVG isn't available on CPU backends)
        total /= dist.get_world_size()
    return total.item()


def rank_share(grad_accum: int, rank: int, world_size: int) -> slice:
    """Which of the step's microbatches this process runs: an equal, contiguous share.

    Every process draws the same list of microbatch positions, then keeps only its own
    share, so N GPUs together see exactly the batch one GPU would have seen alone.
    """
    if grad_accum % world_size:
        raise ValueError(f"grad_accum {grad_accum} must divide evenly between {world_size} GPUs")
    per_rank = grad_accum // world_size
    return slice(rank * per_rank, (rank + 1) * per_rank)


def optimizer_step(model: nn.Module, opt: torch.optim.Optimizer, lr: float, grad_clip: float) -> float:
    """Clip the accumulated gradient, set the lr, update, clear grads. Returns the pre-clip norm."""
    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for group in opt.param_groups:
        group["lr"] = lr
    opt.step()
    opt.zero_grad(set_to_none=True)
    return norm.item()


def fixed_batches(tokens: np.ndarray, n: int, batch_size: int, block_size: int, seed: int) -> list[Batch]:
    """The same validation batches every time, so val losses are comparable across steps."""
    gen = torch.Generator().manual_seed(seed)
    return [get_batch(tokens, batch_size, block_size, generator=gen) for _ in range(n)]


@torch.no_grad()
def evaluate(model: nn.Module, batches: list[Batch], precision: str) -> float:
    was_training = model.training
    model.eval()
    device = _device(model)
    total = 0.0
    for x, y in batches:
        with _autocast(device, precision):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train(was_training)
    return total / len(batches)


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Write to a temp file next to `path`, then rename: never a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _truncate_log(path: Path, last_step: int) -> None:
    """Drop log lines past the checkpoint; those steps will be run again."""
    if not path.exists():
        return
    lines = [ln for ln in path.read_text().splitlines() if json.loads(ln)["step"] <= last_step]
    path.write_text("".join(ln + "\n" for ln in lines))


def _load_data(cfg: TrainConfig, device: torch.device) -> tuple[Callable, Callable, list[Batch]]:
    """(draw, read, val_batches) for the data format.

    draw(generator) picks one microbatch's positions (cheap); read(positions) loads them.
    """
    if cfg.data_format == "tokens":
        train_data = load_tokens(Path(cfg.data_dir) / "train.bin")
        val_data = load_tokens(Path(cfg.data_dir) / "val.bin")
        val = fixed_batches(val_data, cfg.eval_batches, cfg.micro_batch, cfg.block_size, cfg.seed + 1)
        return (lambda gen: random_starts(train_data, cfg.micro_batch, cfg.block_size, gen),
                lambda starts: windows(train_data, starts, cfg.block_size, device), val)
    if cfg.data_format == "sft":
        train_tokens, train_mask = load_sft(cfg.data_dir, "train")
        val_tokens, val_mask = load_sft(cfg.data_dir, "val")
        if train_tokens.shape[1] != cfg.block_size + 1:
            raise ValueError(f"SFT rows hold {train_tokens.shape[1]} tokens; need block_size + 1 = {cfg.block_size + 1}")
        val_gen = torch.Generator().manual_seed(cfg.seed + 1)
        val = [sft_batch(val_tokens, val_mask, torch.randint(len(val_tokens), (cfg.micro_batch,),
                                                              generator=val_gen).tolist())
               for _ in range(cfg.eval_batches)]
        return (lambda gen: torch.randint(len(train_tokens), (cfg.micro_batch,), generator=gen).tolist(),
                lambda rows: sft_batch(train_tokens, train_mask, rows, device), val)
    raise ValueError(f"data_format must be 'tokens' or 'sft', got {cfg.data_format!r}")


def _setup_device(name: str) -> torch.device:
    if name != "cuda":
        return torch.device(name)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmuls on tensor cores (bf16 runs too)
    torch.backends.cudnn.allow_tf32 = True
    return device


def train(cfg: TrainConfig, model_cfg: ModelConfig, stop_at: int | None = None) -> list[dict]:
    """Train until cfg.max_steps, resuming from <out_dir>/checkpoint.pt if it exists.

    `stop_at` ends the call early after that step (used to simulate an interruption).
    Returns one record per optimizer step run in this call.

    Multi-GPU: launch one process per GPU (torchrun) after init_process_group. Each
    process holds a full copy of the model; only process 0 writes logs and checkpoints.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    main_process = rank == 0
    share = rank_share(cfg.grad_accum, rank, world_size)
    device = _setup_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    ckpt_path, log_path = out_dir / "checkpoint.pt", out_dir / "log.jsonl"
    draw, read, val_batches = _load_data(cfg, device)

    torch.manual_seed(cfg.seed)
    model = Keera(model_cfg).to(device)
    opt = make_optimizer(model, cfg)
    batch_gen = torch.Generator().manual_seed(cfg.seed)  # the sampler: its state is our data position
    step = tokens = 0

    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        saved_cfg = {**asdict(TrainConfig()), **ckpt["train_config"]}  # options added later: defaults
        if saved_cfg != asdict(cfg) or ckpt["model_config"] != asdict(model_cfg):
            raise ValueError(f"{ckpt_path} was made with a different config; use a new out_dir")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        batch_gen.set_state(ckpt["batch_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        step, tokens = ckpt["step"], ckpt["tokens"]
        if main_process:
            _truncate_log(log_path, step)
    elif cfg.init_from:
        start = load_checkpoint(cfg.init_from)
        if start["model_config"] != asdict(model_cfg):
            raise ValueError(f"{cfg.init_from} has a different model config: {start['model_config']}")
        model.load_state_dict(start["model"])

    # `model` stays the plain module: its weights are what we save and evaluate.
    # `runner` is what training calls: optionally compiled, and wrapped in DDP across GPUs.
    runner = torch.compile(model) if cfg.compile else model
    if distributed:
        runner = DDP(runner, device_ids=[device.index] if device.type == "cuda" else None)

    tokens_per_step = cfg.grad_accum * cfg.micro_batch * cfg.block_size
    if main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{world_size} process(es) on {device.type} | {tokens_per_step:,} tokens/step | "
              f"{share.stop - share.start} microbatches of {cfg.micro_batch} per process", flush=True)
    history = []
    model.train()

    while step < cfg.max_steps:
        t0 = time.perf_counter()
        # All processes draw the whole step's positions (same generator, same state), so the
        # data order doesn't depend on the number of GPUs; each reads only its own share.
        positions = [draw(batch_gen) for _ in range(cfg.grad_accum)]
        batches = [read(p) for p in positions[share]]
        loss = forward_backward(runner, batches, cfg.precision)
        lr = lr_at(step, cfg)
        grad_norm = optimizer_step(model, opt, lr, cfg.grad_clip)
        step += 1
        tokens += tokens_per_step
        step_seconds = time.perf_counter() - t0
        record = {
            "step": step, "tokens": tokens, "loss": loss, "lr": lr, "grad_norm": grad_norm,
            "step_seconds": step_seconds, "tokens_per_sec": tokens_per_step / step_seconds,
        }

        last = step == cfg.max_steps
        if step % cfg.eval_interval == 0 or last:
            t = time.perf_counter()
            record["val_loss"] = evaluate(model, val_batches, cfg.precision)
            record["eval_seconds"] = time.perf_counter() - t
        if main_process and (step % cfg.checkpoint_interval == 0 or last):
            t = time.perf_counter()
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "tokens": tokens,
                    "batch_rng": batch_gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "train_config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                },
                ckpt_path,
            )
            record["checkpoint_seconds"] = time.perf_counter() - t
        if main_process and cfg.keep_every and step % cfg.keep_every == 0:
            # Weights only (no optimizer): for sampling/evaluating this stage later, not for resuming.
            save_checkpoint(
                {"model": model.state_dict(), "model_config": asdict(model_cfg), "step": step, "tokens": tokens},
                out_dir / f"step_{step:06d}.pt",
            )

        history.append(record)
        if main_process and (step % cfg.log_interval == 0 or "val_loss" in record or last):
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{record['tokens_per_sec']:,.0f} tok/s"
                + (f" | val {record['val_loss']:.4f}" if "val_loss" in record else ""),
                flush=True,
            )
        if stop_at is not None and step >= stop_at:
            break
    return history
