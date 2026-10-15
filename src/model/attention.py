import torch
import torch.nn.functional as F
from torch import nn

from model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    head_h = softmax(Q_h K_h^T / sqrt(head_dim) + causal mask) V_h
    out    = [head_1, ..., head_n] W_O
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # 1. Project, then 2. split into heads: [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.w_q(x)), heads(self.w_k(x)), heads(self.w_v(x))

        # 3. Rotate Q and K by position; V is left alone.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # 4. softmax(Q K^T / sqrt(head_dim) + causal mask) V, for all heads at once.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 5. Join heads: [batch, heads, seq, head_dim] -> [batch, seq, d_model], then 6. W_O.
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.w_o(out)
