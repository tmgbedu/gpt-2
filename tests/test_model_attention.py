import math

import pytest
import torch

from model.attention import CausalSelfAttention
from model.rope import apply_rope, rope_cache

D_MODEL, N_HEADS, HEAD_DIM = 768, 12, 64


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return CausalSelfAttention(D_MODEL, N_HEADS).eval()


@pytest.fixture
def rope():
    return rope_cache(seq_len=32, head_dim=HEAD_DIM)


def test_output_has_same_shape_as_input(attn, rope):
    x = torch.randn(2, 12, D_MODEL)  # [batch, seq_len, d_model]
    assert attn(x, *rope).shape == x.shape


def test_has_four_bias_free_768x768_matrices(attn):
    assert sum(p.numel() for p in attn.parameters()) == 4 * D_MODEL * D_MODEL  # 2,359,296
    for layer in (attn.w_q, attn.w_k, attn.w_v, attn.w_o):
        assert layer.weight.shape == (D_MODEL, D_MODEL)
        assert layer.bias is None


def _by_hand(attn, x, cos, sin):
    """softmax(Q K^T / sqrt(d) + causal mask) V, written out step by step."""
    batch, seq_len, _ = x.shape

    def heads(t):  # [batch, seq, 768] -> [batch, 12, seq, 64]
        return t.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)

    q = heads(x @ attn.w_q.weight.T)
    k = heads(x @ attn.w_k.weight.T)
    v = heads(x @ attn.w_v.weight.T)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)  # [batch, 12, seq, seq]
    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = scores.softmax(dim=-1)  # each row sums to 1
    out = weights @ v  # weighted mix of V

    out = out.transpose(1, 2).reshape(batch, seq_len, D_MODEL)  # join heads
    return out @ attn.w_o.weight.T


def test_matches_attention_written_by_hand(attn, rope):
    x = torch.randn(2, 10, D_MODEL)
    torch.testing.assert_close(attn(x, *rope), _by_hand(attn, x, *rope), atol=1e-5, rtol=1e-4)


def test_no_token_can_see_the_future(attn, rope):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[0, 6] = torch.randn(D_MODEL)  # change token at position 6

    before, after = attn(x, *rope), attn(changed, *rope)
    torch.testing.assert_close(before[0, :6], after[0, :6])  # positions 0..5 unaffected
    assert not torch.allclose(before[0, 6:], after[0, 6:])  # positions 6.. do change


def test_first_token_only_sees_itself(attn, rope):
    # With one visible token, softmax gives it weight 1, so output = W_O(V(x0)).
    x = torch.randn(1, 5, D_MODEL)
    expected = attn.w_o(attn.w_v(x[0, 0]))
    torch.testing.assert_close(attn(x, *rope)[0, 0], expected, atol=1e-5, rtol=1e-4)


def test_d_model_must_split_evenly_into_heads():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(d_model=768, n_heads=10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_runs_on_mps(attn):
    attn = attn.to("mps")
    cos, sin = rope_cache(seq_len=16, head_dim=HEAD_DIM, device="mps")
    x = torch.randn(2, 16, D_MODEL, device="mps")
    y = attn(x, cos, sin)
    assert y.device.type == "mps"
    torch.testing.assert_close(
        y.cpu(), attn.cpu()(x.cpu(), cos.cpu(), sin.cpu()), atol=1e-4, rtol=1e-3
    )
