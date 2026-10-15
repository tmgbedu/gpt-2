import pytest
import torch

from model.norm import RMSNorm


def test_worked_example():
    # x = (2, -1, 3, 1): mean of squares = 15/4, RMS = sqrt(3.75) ≈ 1.936
    norm = RMSNorm(4)
    x = torch.tensor([2.0, -1.0, 3.0, 1.0])
    expected = x / 3.75**0.5
    torch.testing.assert_close(norm(x), expected)


def test_output_has_same_shape_and_rms_one_per_token():
    norm = RMSNorm(768)
    x = torch.randn(2, 12, 768) * 50  # [batch, seq_len, d_model], deliberately large
    y = norm(x)
    assert y.shape == x.shape
    rms_per_token = y.pow(2).mean(dim=-1).sqrt()  # one RMS per token: shape [2, 12]
    torch.testing.assert_close(rms_per_token, torch.ones(2, 12), atol=1e-4, rtol=1e-4)


def test_scale_of_input_does_not_matter():
    norm = RMSNorm(8)
    x = torch.randn(3, 8)
    torch.testing.assert_close(norm(x), norm(100 * x))


def test_weight_g_starts_at_one_and_is_learnable():
    norm = RMSNorm(768)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.shape == (768,)
    torch.testing.assert_close(norm.weight.data, torch.ones(768))


def test_weight_g_scales_each_dimension():
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.0, -1.0]))
    x = torch.tensor([2.0, -1.0, 3.0, 1.0])
    expected = x / 3.75**0.5 * torch.tensor([1.0, 2.0, 0.0, -1.0])
    torch.testing.assert_close(norm(x), expected)


def test_each_token_is_normalized_independently():
    norm = RMSNorm(4)
    a = torch.tensor([2.0, -1.0, 3.0, 1.0])
    b = torch.tensor([0.1, 0.2, 0.3, 0.4])
    both = norm(torch.stack([a, b]))
    torch.testing.assert_close(both[0], norm(a))
    torch.testing.assert_close(both[1], norm(b))


def test_zero_vector_does_not_divide_by_zero():
    y = RMSNorm(4)(torch.zeros(4))
    assert torch.isfinite(y).all()


def test_matches_pytorch_builtin():
    torch.manual_seed(0)
    ours = RMSNorm(768, eps=1e-6)
    builtin = torch.nn.RMSNorm(768, eps=1e-6)
    x = torch.randn(2, 12, 768)
    torch.testing.assert_close(ours(x), builtin(x))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_runs_on_mps():
    norm = RMSNorm(768).to("mps")
    y = norm(torch.randn(2, 12, 768, device="mps"))
    assert y.device.type == "mps"
