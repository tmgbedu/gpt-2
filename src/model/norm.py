import torch
from torch import nn


class RMSNorm(nn.Module):
    """y_i = x_i / RMS(x) * g_i,   where RMS(x) = sqrt(mean_i(x_i^2) + eps).

    Each token's vector (last dimension) is normalized on its own.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # g = (1, ..., 1), learned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)  # (1/d) * sum_i x_i^2, one per token
        rms = torch.sqrt(mean_sq + self.eps)  # RMS(x)
        return x / rms * self.weight  # x_i / RMS(x) * g_i
