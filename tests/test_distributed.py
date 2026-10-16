"""Multi-GPU (DDP) training, simulated on CPU: each "GPU" is a separate process.

The promise being tested: N processes that each do grad_accum / N microbatches produce
exactly the same training as one process doing all grad_accum microbatches.
"""

import json
import math
import os
import socket
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.transformer import ModelConfig
from test_trainer import TINY, _cfg, data_dir  # noqa: F401  (data_dir is a fixture)
from trainer import TrainConfig, load_checkpoint, rank_share, train


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, backend, cfg, model_cfg, stop_at, result_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        history = train(TrainConfig(**cfg), ModelConfig(**model_cfg), stop_at=stop_at)
        torch.save(history, Path(result_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def run_distributed(cfg, model_cfg, world, tmp_path, stop_at=None, backend="gloo"):
    """Train with `world` processes; returns each rank's history."""
    result_dir = tmp_path / f"results-{_free_port()}"
    result_dir.mkdir()
    mp.spawn(_worker, args=(world, _free_port(), backend, asdict(cfg), asdict(model_cfg), stop_at,
                            str(result_dir)), nprocs=world, join=True)
    return [torch.load(result_dir / f"rank{r}.pt") for r in range(world)]


def _assert_same_training(single, multi):
    assert [r["step"] for r in single] == [r["step"] for r in multi]
    for a, b in zip(single, multi):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-4)
        assert a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-4)
        assert a["lr"] == pytest.approx(b["lr"])
        assert a["tokens"] == b["tokens"]
        if "val_loss" in a:
            assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-4)


def _assert_same_weights(path_a, path_b):
    a, b = load_checkpoint(path_a), load_checkpoint(path_b)
    for k, v in a["model"].items():
        torch.testing.assert_close(v, b["model"][k], rtol=1e-4, atol=1e-6)


# --- splitting the work -------------------------------------------------------


def test_each_rank_gets_an_equal_contiguous_share_of_the_microbatches():
    assert [rank_share(8, r, 4) for r in range(4)] == [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]
    assert rank_share(8, 0, 1) == slice(0, 8)  # one process does everything


def test_grad_accum_must_divide_evenly_between_ranks():
    with pytest.raises(ValueError, match="grad_accum"):
        rank_share(6, 0, 4)


# --- two processes == one process ----------------------------------------------


def test_two_processes_train_exactly_like_one(data_dir, tmp_path):
    single = train(_cfg(data_dir, tmp_path / "single"), TINY)
    histories = run_distributed(_cfg(data_dir, tmp_path / "ddp"), TINY, world=2, tmp_path=tmp_path)

    _assert_same_training(single, histories[0])
    _assert_same_training(single, histories[1])  # every rank sees the same averaged numbers
    _assert_same_weights(tmp_path / "single" / "checkpoint.pt", tmp_path / "ddp" / "checkpoint.pt")


def test_only_rank_zero_writes_files(data_dir, tmp_path):
    run_distributed(_cfg(data_dir, tmp_path / "ddp", keep_every=3), TINY, world=2, tmp_path=tmp_path)
    out = tmp_path / "ddp"
    assert sorted(p.name for p in out.iterdir()) == [
        "checkpoint.pt", "log.jsonl", "step_000003.pt", "step_000006.pt"]
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6]  # once each, not once per rank


def test_two_processes_resume_like_uninterrupted_single_process(data_dir, tmp_path):
    single = train(_cfg(data_dir, tmp_path / "single"), TINY)
    cfg = _cfg(data_dir, tmp_path / "ddp")
    first = run_distributed(cfg, TINY, world=2, tmp_path=tmp_path, stop_at=3)
    second = run_distributed(cfg, TINY, world=2, tmp_path=tmp_path)

    _assert_same_training(single, first[0] + second[0])
    _assert_same_weights(tmp_path / "single" / "checkpoint.pt", tmp_path / "ddp" / "checkpoint.pt")


# --- torch.compile ------------------------------------------------------------


def test_compiled_model_trains_like_eager_and_saves_plain_weights(data_dir, tmp_path):
    eager = train(_cfg(data_dir, tmp_path / "eager", max_steps=3, checkpoint_interval=3), TINY)
    compiled = train(_cfg(data_dir, tmp_path / "compiled", max_steps=3, checkpoint_interval=3,
                          compile=True), TINY)
    for a, b in zip(eager, compiled):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-3)
    ckpt = load_checkpoint(tmp_path / "compiled" / "checkpoint.pt")
    assert not any(k.startswith("_orig_mod.") for k in ckpt["model"])  # loads into a plain Keera


# --- real GPUs (skipped on the Mac; run these first on a rented machine) --------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
needs_2_cuda = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ NVIDIA GPUs")


@needs_cuda
def test_short_bf16_run_on_cuda_learns(data_dir, tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(data_dir, tmp_path / "run", max_steps=60, warmup_steps=5, eval_interval=60,
               checkpoint_interval=60, device="cuda", precision="bf16")
    history = train(cfg, TINY)
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in history)
    assert history[-1]["loss"] < 1.5


@needs_2_cuda
def test_two_gpus_with_nccl_train_like_one(data_dir, tmp_path):
    kw = dict(device="cuda", precision="fp32")
    single = train(_cfg(data_dir, tmp_path / "single", **kw), TINY)
    histories = run_distributed(_cfg(data_dir, tmp_path / "ddp", **kw), TINY, world=2,
                                tmp_path=tmp_path, backend="nccl")
    # GPU kernels sum in a different order than one GPU does, so allow a little more drift.
    for a, b in zip(single, histories[0]):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-3)
