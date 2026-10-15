"""Rotary position embedding (RoPE), Llama layout.

A head vector of size d is split into two halves; dimension j of the first half and
dimension j of the second half form pair j. At position m, pair j is rotated by angle
m * theta_j, with theta_j = base^(-2j/d):

    a' = a cos(m theta_j) - b sin(m theta_j)
    b' = a sin(m theta_j) + b cos(m theta_j)
"""

import torch


def rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of m * theta_j for every position m. Each has shape [seq_len, head_dim]."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even to form pairs, got {head_dim}")
    half = head_dim // 2
    theta = base ** (-torch.arange(half, dtype=torch.float32, device=device) / half)  # [half]
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # [seq_len]
    angles = torch.outer(positions, theta)  # angles[m, j] = m * theta_j
    angles = torch.cat([angles, angles], dim=-1)  # same angle for both halves of each pair
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(a, b) -> (-b, a) for every pair, i.e. a 90-degree turn."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape [..., seq_len, head_dim] by its position. Use for Q and K only."""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(x.dtype)
    sin = sin[:seq_len].to(x.dtype)
    # (a, b) * cos + (-b, a) * sin = (a cos - b sin, a sin + b cos)
    return x * cos + _rotate_half(x) * sin
