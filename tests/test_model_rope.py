import math

import pytest
import torch

from model.rope import apply_rope, rope_cache


def _rotate_at(vec: torch.Tensor, position: int, cos, sin) -> torch.Tensor:
    """Rotate one head vector as if it sat at `position`. Shape [head_dim] -> [head_dim]."""
    seq = torch.zeros(position + 1, vec.shape[-1])
    seq[position] = vec
    return apply_rope(seq[None, None], cos, sin)[0, 0, position]


def test_speeds_follow_the_formula():
    # theta_j = 10000^(-j/32) for head_dim 64; at position 1 the angle equals theta_j.
    cos, sin = rope_cache(seq_len=2, head_dim=64)
    angles = torch.atan2(sin[1, :32], cos[1, :32])
    expected = torch.tensor([10000 ** (-j / 32) for j in range(32)])
    torch.testing.assert_close(angles, expected)


def test_position_zero_is_not_rotated():
    cos, sin = rope_cache(seq_len=4, head_dim=64)
    v = torch.randn(64)
    torch.testing.assert_close(_rotate_at(v, 0, cos, sin), v)


def test_worked_example_single_pair():
    # head_dim 2 -> one pair with theta = 1 radian per position.
    cos, sin = rope_cache(seq_len=4, head_dim=2)
    q = torch.tensor([1.0, 0.0])
    for m in range(4):
        expected = torch.tensor([math.cos(m), math.sin(m)])
        torch.testing.assert_close(_rotate_at(q, m, cos, sin), expected)


def test_rotation_keeps_length():
    cos, sin = rope_cache(seq_len=50, head_dim=64)
    x = torch.randn(2, 12, 50, 64)  # [batch, heads, seq_len, head_dim]
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_score_depends_only_on_distance():
    torch.manual_seed(0)
    cos, sin = rope_cache(seq_len=200, head_dim=64)
    q, k = torch.randn(64), torch.randn(64)

    def score(m, n):
        return _rotate_at(q, m, cos, sin) @ _rotate_at(k, n, cos, sin)

    # "i eat mango" at the start of a file vs. later in it: same distance, same score.
    torch.testing.assert_close(score(2, 0), score(102, 100))
    torch.testing.assert_close(score(2, 1), score(150, 149))
    # Different distance, different score.
    assert not torch.allclose(score(2, 0), score(2, 1))


def test_output_shape_and_longer_cache():
    cos, sin = rope_cache(seq_len=1024, head_dim=64)  # cache covers full context
    x = torch.randn(2, 12, 10, 64)  # but this batch is only 10 tokens long
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    torch.testing.assert_close(y, apply_rope(x, cos[:10], sin[:10]))


def test_keeps_input_dtype():
    cos, sin = rope_cache(seq_len=8, head_dim=64)
    x = torch.randn(1, 1, 8, 64, dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16


def test_head_dim_must_be_even():
    with pytest.raises(ValueError, match="even"):
        rope_cache(seq_len=4, head_dim=63)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_runs_on_mps():
    cos, sin = rope_cache(seq_len=16, head_dim=64, device="mps")
    x = torch.randn(2, 12, 16, 64, device="mps")
    y = apply_rope(x, cos, sin)
    assert y.device.type == "mps"
    torch.testing.assert_close(y.cpu(), apply_rope(x.cpu(), cos.cpu(), sin.cpu()))
