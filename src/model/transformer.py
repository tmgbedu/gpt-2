import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.norm import RMSNorm
from model.rope import rope_cache


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02


class Block(nn.Module):
    """x = x + Attn(RMSNorm(x));  x = x + MLP(RMSNorm(x))"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Keera(nn.Module):
    """Token IDs -> embedding -> N blocks -> final RMSNorm -> logits (tied with the embedding)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied: one matrix for input and output

        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads, cfg.rope_base)
        # Fixed tables, not weights: move with .to(device) but stay out of checkpoints.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # The two projections that add into the residual stream start smaller, so the
        # stream doesn't grow with depth at init (GPT-2): 2 sublayers per block.
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w_2.weight, mean=0.0, std=residual_std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: [batch, seq_len] token IDs. targets: same shape; -100 marks positions to skip."""
        seq_len = idx.shape[1]
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)  # [batch, seq_len, d_model]
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))  # [batch, seq_len, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss
