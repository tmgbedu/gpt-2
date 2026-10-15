import torch
from torch import nn


class MLP(nn.Module):
    """MLP(y) = GELU(y W1) W2, applied to each token on its own."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)  # expand: 768 -> 3072
        self.act = nn.GELU()  # exact GELU: z * Phi(z)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)  # project back: 3072 -> 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.act(self.w_1(x)))
