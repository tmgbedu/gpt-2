import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model.transformer import Keera, ModelConfig
from sample import generate

TINY = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=16)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Keera(TINY).eval()


def test_greedy_picks_the_most_likely_token_each_time(model):
    prompt = [5, 9, 2]
    out = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    ids = torch.tensor([prompt])
    for _ in range(6):
        logits, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert out == ids[0, 3:].tolist()  # only the new tokens are returned


def test_returns_max_new_tokens(model):
    out = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, stop_id=None,
                   generator=torch.Generator().manual_seed(0))
    assert len(out) == 10


def test_stops_after_the_stop_token(model):
    first = generate(model, [1, 2], max_new_tokens=1, temperature=0.0)[0]
    out = generate(model, [1, 2], max_new_tokens=10, temperature=0.0, stop_id=first)
    assert out == [first]  # included, then generation ends


def test_prompts_longer_than_the_context_are_cropped(model):
    out = generate(model, list(range(1, 40)), max_new_tokens=20, temperature=0.0)
    assert len(out) == 20  # would raise max_seq_len otherwise


def test_top_k_only_samples_from_the_k_most_likely(model):
    prompt = [3, 4]
    logits, _ = model(torch.tensor([prompt]))
    top3 = set(logits[0, -1].topk(3).indices.tolist())
    gen = torch.Generator().manual_seed(0)
    firsts = {generate(model, prompt, max_new_tokens=1, temperature=5.0, top_k=3, generator=gen)[0]
              for _ in range(50)}
    assert firsts <= top3


def test_same_seed_same_output(model):
    a = generate(model, [1], 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
    b = generate(model, [1], 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
    assert a == b


def test_cli_help_runs_as_installed_module():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-m", "sample", "--help"],
                            capture_output=True, text=True, env=env, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr
    assert "--prompt" in result.stdout
