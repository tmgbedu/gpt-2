import math

import pytest
import torch

from model.mlp import MLP

D_MODEL, D_FF = 768, 3072


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MLP(D_MODEL, D_FF)


def test_gelu_values_match_the_table(mlp):
    z = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    # GELU(z) = z * Phi(z), Phi = standard normal CDF
    expected = torch.tensor([z_ * 0.5 * (1 + math.erf(z_ / math.sqrt(2))) for z_ in z.tolist()])
    torch.testing.assert_close(mlp.act(z), expected)
    torch.testing.assert_close(
        mlp.act(z), torch.tensor([-0.004, -0.045, -0.159, 0.0, 0.841, 1.954, 2.996]), atol=1e-3, rtol=0
    )


def test_output_has_same_shape_as_input(mlp):
    x = torch.randn(2, 12, D_MODEL)
    assert mlp(x).shape == x.shape


def test_has_two_bias_free_matrices(mlp):
    assert mlp.w_1.weight.shape == (D_FF, D_MODEL)  # 768 -> 3072
    assert mlp.w_2.weight.shape == (D_MODEL, D_FF)  # 3072 -> 768
    assert mlp.w_1.bias is None and mlp.w_2.bias is None
    assert sum(p.numel() for p in mlp.parameters()) == 2 * D_MODEL * D_FF  # 4,718,592


def test_matches_formula(mlp):
    # MLP(y) = GELU(y W1) W2
    y = torch.randn(3, D_MODEL)
    h = y @ mlp.w_1.weight.T
    expected = (h * 0.5 * (1 + torch.erf(h / math.sqrt(2)))) @ mlp.w_2.weight.T
    torch.testing.assert_close(mlp(y), expected)


def test_each_token_is_processed_independently(mlp):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[0, 4] = torch.randn(D_MODEL)
    before, after = mlp(x), mlp(changed)
    others = [i for i in range(10) if i != 4]
    torch.testing.assert_close(before[0, others], after[0, others])
    assert not torch.allclose(before[0, 4], after[0, 4])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_runs_on_mps(mlp):
    x = torch.randn(2, 16, D_MODEL)
    expected = mlp(x)
    y = mlp.to("mps")(x.to("mps"))
    assert y.device.type == "mps"
    torch.testing.assert_close(y.cpu(), expected, atol=1e-4, rtol=1e-3)
