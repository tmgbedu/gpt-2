import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from model.transformer import Keera, ModelConfig
from trainer import (
    TrainConfig,
    evaluate,
    fixed_batches,
    forward_backward,
    load_checkpoint,
    lr_at,
    make_optimizer,
    optimizer_step,
    save_checkpoint,
    train,
)

TINY = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=16)


@pytest.fixture
def data_dir(tmp_path):
    # A learnable stream: short counting runs 0..15 with a little noise.
    rng = np.random.default_rng(0)
    tokens = np.tile(np.arange(16, dtype=np.uint16), 500)
    noise = rng.random(len(tokens)) < 0.05
    tokens[noise] = rng.integers(16, 64, noise.sum())
    d = tmp_path / "data"
    d.mkdir()
    tokens.tofile(d / "train.bin")
    tokens[:2000].tofile(d / "val.bin")
    return d


def _cfg(data_dir, out_dir, **kw):
    base = TrainConfig(
        data_dir=str(data_dir), out_dir=str(out_dir), micro_batch=4, block_size=16,
        grad_accum=2, max_steps=6, warmup_steps=2, max_lr=3e-3, min_lr=3e-4,
        eval_interval=3, eval_batches=2, checkpoint_interval=3, log_interval=1,
        device="cpu", precision="fp32",
    )
    return replace(base, **kw)


# --- optimizer --------------------------------------------------------------


def test_weight_decay_only_on_matrices():
    model = Keera(TINY)
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])  # RMSNorm weights
    norm_ids = {id(m.weight) for m in model.modules() if type(m).__name__ == "RMSNorm"}
    assert {id(p) for p in no_decay["params"]} == norm_ids
    # every parameter exactly once (the tied embedding/output matrix is one parameter)
    all_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(all_ids) == sorted(id(p) for p in model.parameters())


# --- learning-rate schedule -------------------------------------------------


def test_lr_warms_up_then_decays_to_min():
    cfg = TrainConfig(warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-4)
    lrs = [lr_at(s, cfg) for s in range(cfg.max_steps)]
    assert lrs[0] == pytest.approx(1e-4)  # (0+1)/10 of peak
    assert lrs[9] == pytest.approx(1e-3)  # peak at the end of warmup
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:10]))
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))
    assert lrs[60] == pytest.approx((1e-3 + 1e-4) / 2, rel=0.02)  # cosine midpoint
    assert lrs[-1] == pytest.approx(1e-4, rel=0.01)


# --- gradient accumulation and clipping -------------------------------------


def test_accumulated_gradient_equals_one_big_batch():
    torch.manual_seed(0)
    model = Keera(TINY)
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))

    loss = forward_backward(model, [(x[:4], y[:4]), (x[4:], y[4:])], "fp32")
    accumulated = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    _, full = model(x, y)
    full.backward()
    assert loss == pytest.approx(full.item(), rel=1e-5)
    for a, b in zip(accumulated, (p.grad for p in model.parameters())):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_clipping_happens_on_the_accumulated_gradient_before_the_step():
    torch.manual_seed(0)
    model = Keera(TINY)
    opt = make_optimizer(model, TrainConfig())
    for p in model.parameters():
        p.grad = torch.full_like(p, 10.0)
    seen = {}

    def spy_step():
        seen["norm"] = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
        seen["lr"] = [g["lr"] for g in opt.param_groups]

    opt.step = spy_step
    norm = optimizer_step(model, opt, lr=1e-3, grad_clip=1.0)

    assert norm > 1.0  # reports the norm before clipping
    assert seen["norm"] == pytest.approx(1.0, rel=1e-4)
    assert seen["lr"] == [1e-3, 1e-3]
    assert all(p.grad is None for p in model.parameters())  # cleared for the next step


# --- validation -------------------------------------------------------------


def test_fixed_validation_batches_and_no_grad_eval(data_dir):
    tokens = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
    a = fixed_batches(tokens, n=3, batch_size=4, block_size=16, seed=1)
    b = fixed_batches(tokens, n=3, batch_size=4, block_size=16, seed=1)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)

    model = Keera(TINY).train()
    modes = []
    model.register_forward_hook(lambda m, i, o: modes.append(m.training))
    loss1 = evaluate(model, a, "fp32")
    assert modes == [False] * 3  # every validation forward ran in eval mode
    loss2 = evaluate(model, a, "fp32")
    assert loss1 == loss2
    assert model.training  # restored
    assert all(p.grad is None for p in model.parameters())
    assert abs(loss1 - math.log(64)) < 1.0


# --- checkpoints ------------------------------------------------------------


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint({"step": 1}, path)

    def broken_save(obj, f):
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        save_checkpoint({"step": 2}, path)
    monkeypatch.undo()

    assert load_checkpoint(path)["step"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint.pt"]


def test_checkpoint_holds_everything_needed_to_resume(data_dir, tmp_path):
    cfg = _cfg(data_dir, tmp_path / "run", max_steps=3)
    train(cfg, TINY)
    ckpt = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert ckpt["step"] == 3
    assert ckpt["tokens"] == 3 * 2 * 4 * 16  # steps * accum * batch * block
    assert ckpt["train_config"] == cfg.__dict__
    assert ckpt["model_config"] == TINY.__dict__
    for key in ("model", "optimizer", "batch_rng", "torch_rng"):
        assert key in ckpt


def test_resume_matches_uninterrupted_training(data_dir, tmp_path):
    full = train(_cfg(data_dir, tmp_path / "full"), TINY)

    cfg = _cfg(data_dir, tmp_path / "resumed")
    first = train(cfg, TINY, stop_at=3)  # "crash" right after the step-3 checkpoint
    second = train(cfg, TINY)  # a fresh call picks up the checkpoint

    assert [r["step"] for r in first + second] == [1, 2, 3, 4, 5, 6]
    for a, b in zip(full, first + second):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-5)
        assert a["lr"] == pytest.approx(b["lr"])
    final_full = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    final_resumed = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert final_full["tokens"] == final_resumed["tokens"]
    for k, v in final_full["model"].items():
        torch.testing.assert_close(v, final_resumed["model"][k], rtol=1e-5, atol=1e-6)

    log = [json.loads(line) for line in (tmp_path / "resumed" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6]


def test_keeps_weights_only_milestones(data_dir, tmp_path):
    out = tmp_path / "run"
    train(_cfg(data_dir, out, keep_every=2), TINY)
    milestones = sorted(p.name for p in out.glob("step_*.pt"))
    assert milestones == ["step_000002.pt", "step_000004.pt", "step_000006.pt"]

    snap = load_checkpoint(out / "step_000004.pt")
    assert set(snap) == {"model", "model_config", "step", "tokens"}  # no optimizer: 1/3 the size
    assert snap["step"] == 4
    model = Keera(ModelConfig(**snap["model_config"]))
    model.load_state_dict(snap["model"])  # usable for sampling / evaluation


def test_no_milestones_by_default(data_dir, tmp_path):
    train(_cfg(data_dir, tmp_path / "run"), TINY)
    assert list((tmp_path / "run").glob("step_*.pt")) == []


def test_refuses_to_resume_with_a_different_config(data_dir, tmp_path):
    train(_cfg(data_dir, tmp_path / "run", max_steps=3), TINY)
    with pytest.raises(ValueError, match="config"):
        train(_cfg(data_dir, tmp_path / "run", max_steps=3, max_lr=1.0), TINY)


# --- learning -----------------------------------------------------------------


def test_loss_drops_well_below_uniform_guess(data_dir, tmp_path):
    torch.manual_seed(0)
    history = train(_cfg(data_dir, tmp_path / "run", max_steps=60, warmup_steps=5,
                         eval_interval=60, checkpoint_interval=60), TINY)
    assert history[0]["loss"] > math.log(64) - 0.5
    assert history[-1]["loss"] < 1.5
    assert history[-1]["val_loss"] < 1.5


# --- MPS precision ----------------------------------------------------------

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")


@needs_mps
def test_bf16_matches_fp32_within_tolerance_on_mps():
    torch.manual_seed(0)
    model = Keera(TINY).to("mps")
    x = torch.randint(0, 64, (4, 16), device="mps")
    y = torch.randint(0, 64, (4, 16), device="mps")

    grads = {}
    losses = {}
    for precision in ("fp32", "bf16"):
        model.zero_grad()
        losses[precision] = forward_backward(model, [(x, y)], precision)
        grads[precision] = torch.cat([p.grad.flatten() for p in model.parameters()])

    assert losses["bf16"] == pytest.approx(losses["fp32"], rel=0.02)
    assert torch.isfinite(grads["bf16"]).all()
    cos = torch.nn.functional.cosine_similarity(grads["bf16"], grads["fp32"], dim=0)
    assert cos > 0.99


@needs_mps
def test_short_bf16_run_on_mps_learns(data_dir, tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(data_dir, tmp_path / "run", max_steps=60, warmup_steps=5, eval_interval=60,
               checkpoint_interval=60, device="mps", precision="bf16")
    history = train(cfg, TINY)
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in history)
    assert history[-1]["loss"] < 1.5


# --- CLI --------------------------------------------------------------------


def test_cli_help_runs_as_installed_module():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-m", "trainer.train", "--help"],
                            capture_output=True, text=True, env=env, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr
    assert "--grad-accum" in result.stdout
    assert "--compile, --no-compile" in result.stdout  # booleans are flags, not "--compile False"


def test_cli_trains_with_two_processes_under_torchrun(data_dir, tmp_path):
    # The exact launch used on a multi-GPU machine, with CPU processes standing in for GPUs.
    # Uses the real 97.5M model on 16-token windows, so it stays quick.
    out = tmp_path / "run"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"  # gloo picks a broken interface on macOS otherwise
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2",
         "--master-addr", "127.0.0.1", "--master-port", str(port),
         "-m", "trainer.train", "--device", "cpu", "--data-dir", str(data_dir), "--out-dir", str(out),
         "--micro-batch", "1", "--block-size", "16", "--grad-accum", "2", "--max-steps", "2",
         "--warmup-steps", "1", "--eval-interval", "2", "--eval-batches", "1",
         "--checkpoint-interval", "2", "--log-interval", "1"],
        capture_output=True, text=True, env=env, cwd=Path(__file__).parent, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "2 process(es) on cpu" in result.stdout
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2]
    assert load_checkpoint(out / "checkpoint.pt")["step"] == 2


# --- fine-tuning (SFT): packed rows with a learn-mask, starting from a pretrained model ---


@pytest.fixture
def sft_dir(tmp_path):
    # Rows of 17 tokens (block 16 + 1): a "question" of noise, then a counting "answer".
    rng = np.random.default_rng(0)
    d = tmp_path / "sft"
    d.mkdir()
    for split, n in (("train", 64), ("val", 8)):
        tokens = np.tile(np.arange(17, dtype=np.uint16), (n, 1))
        tokens[:, :5] = rng.integers(20, 64, (n, 5))
        mask = np.zeros((n, 17), dtype=np.uint8)
        mask[:, 5:] = 1
        np.save(d / f"{split}_tokens.npy", tokens)
        np.save(d / f"{split}_mask.npy", mask)
    return d


def _sft_cfg(sft_dir, out_dir, **kw):
    return _cfg(sft_dir, out_dir, data_format="sft", **kw)


def test_sft_learns_the_answers(sft_dir, tmp_path):
    torch.manual_seed(0)
    history = train(_sft_cfg(sft_dir, tmp_path / "run", max_steps=40, warmup_steps=5,
                             eval_interval=40, checkpoint_interval=40), TINY)
    assert history[0]["loss"] > math.log(64) - 0.5
    assert history[-1]["val_loss"] < 0.5  # the counting answer is easy once only it is scored


def test_sft_rows_must_be_block_size_plus_one(sft_dir, tmp_path):
    with pytest.raises(ValueError, match="block_size"):
        train(_sft_cfg(sft_dir, tmp_path / "run", block_size=8), TINY)


def test_sft_resume_matches_uninterrupted_training(sft_dir, tmp_path):
    full = train(_sft_cfg(sft_dir, tmp_path / "full"), TINY)
    cfg = _sft_cfg(sft_dir, tmp_path / "resumed")
    resumed = train(cfg, TINY, stop_at=3) + train(cfg, TINY)
    assert [a["loss"] for a in full] == pytest.approx([b["loss"] for b in resumed], rel=1e-5)


def test_init_from_starts_with_the_pretrained_weights_and_a_fresh_optimizer(data_dir, sft_dir, tmp_path):
    train(_cfg(data_dir, tmp_path / "pre"), TINY)
    pre = load_checkpoint(tmp_path / "pre" / "checkpoint.pt")
    # lr 0: nothing moves, so the saved weights must be exactly the pretrained ones.
    history = train(_sft_cfg(sft_dir, tmp_path / "sft", init_from=str(tmp_path / "pre" / "checkpoint.pt"),
                             max_steps=1, max_lr=0.0, min_lr=0.0), TINY)
    assert history[0]["step"] == 1  # step count starts over
    sft = load_checkpoint(tmp_path / "sft" / "checkpoint.pt")
    for k, v in pre["model"].items():
        torch.testing.assert_close(sft["model"][k], v)


def test_init_from_with_a_different_model_shape_is_refused(data_dir, sft_dir, tmp_path):
    train(_cfg(data_dir, tmp_path / "pre", max_steps=1), TINY)
    other = replace(TINY, n_layers=3)
    with pytest.raises(ValueError, match="model config"):
        train(_sft_cfg(sft_dir, tmp_path / "sft", init_from=str(tmp_path / "pre" / "checkpoint.pt")), other)


def test_checkpoints_from_before_the_sft_options_still_resume(data_dir, tmp_path):
    cfg = _cfg(data_dir, tmp_path / "run")
    train(cfg, TINY, stop_at=3)
    path = tmp_path / "run" / "checkpoint.pt"
    ckpt = load_checkpoint(path)
    for key in ("data_format", "init_from"):  # what an older checkpoint looks like
        del ckpt["train_config"][key]
    save_checkpoint(ckpt, path)
    assert [r["step"] for r in train(cfg, TINY)] == [4, 5, 6]
