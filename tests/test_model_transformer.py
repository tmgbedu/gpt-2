import math

import pytest
import torch

from model.transformer import Block, Keera, ModelConfig

TINY = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=16)


@pytest.fixture(scope="module")
def full_model():
    torch.manual_seed(0)
    return Keera(ModelConfig()).eval()


def test_default_config_matches_the_design():
    cfg = ModelConfig()
    assert (cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.max_seq_len) == (
        16384, 768, 12, 12, 3072, 1024,
    )


def test_parameter_count(full_model):
    embedding = 16384 * 768  # shared with the output layer, so counted once
    per_layer = 4 * 768 * 768 + 2 * 768 * 3072 + 2 * 768  # attention + MLP + two RMSNorms
    final_norm = 768
    expected = embedding + 12 * per_layer + final_norm
    assert expected == 97_536_768
    assert sum(p.numel() for p in full_model.parameters()) == expected


def test_embedding_and_output_share_one_matrix(full_model):
    assert full_model.lm_head.weight is full_model.tok_emb.weight


def test_logits_shape(full_model):
    idx = torch.randint(0, 16384, (2, 8))
    logits, loss = full_model(idx)
    assert logits.shape == (2, 8, 16384)
    assert loss is None


def test_loss_at_init_is_close_to_uniform_guess(full_model):
    torch.manual_seed(1)
    idx = torch.randint(0, 16384, (2, 64))
    targets = torch.randint(0, 16384, (2, 64))
    _, loss = full_model(idx, targets)
    # An untrained model should be about equally unsure of every token: ln(16384) ≈ 9.70.
    assert math.log(16384) - 0.3 < loss.item() < math.log(16384) + 0.5


def test_block_is_two_residual_steps():
    torch.manual_seed(0)
    model = Keera(TINY)
    block = model.blocks[0]
    x = torch.randn(1, 5, TINY.d_model)
    cos, sin = model.rope_cos, model.rope_sin
    h = x + block.attn(block.norm1(x), cos, sin)
    expected = h + block.mlp(block.norm2(h))
    torch.testing.assert_close(block(x, cos, sin), expected)


def test_no_position_can_see_the_future():
    torch.manual_seed(0)
    model = Keera(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 7] = (idx[0, 7] + 1) % TINY.vocab_size
    before, _ = model(idx)
    after, _ = model(changed)
    torch.testing.assert_close(before[0, :7], after[0, :7])
    assert not torch.allclose(before[0, 7:], after[0, 7:])


def test_residual_projections_start_smaller():
    torch.manual_seed(0)
    model = Keera(ModelConfig(n_layers=12, vocab_size=1024))
    small = 0.02 / math.sqrt(2 * 12)
    for block in model.blocks:
        for w in (block.attn.w_o.weight, block.mlp.w_2.weight):
            assert w.std().item() == pytest.approx(small, rel=0.05)
        assert block.attn.w_q.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert model.tok_emb.weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_rejects_sequences_longer_than_context():
    model = Keera(TINY)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, TINY.max_seq_len + 1, dtype=torch.long))


def test_can_overfit_one_batch():
    # The standard wiring check: a model that can't memorize one batch has a bug.
    torch.manual_seed(0)
    model = Keera(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    targets = torch.randint(0, TINY.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_ignores_targets_marked_minus_100():
    torch.manual_seed(0)
    model = Keera(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    targets = torch.randint(0, TINY.vocab_size, (1, 8))
    masked = targets.clone()
    masked[0, :4] = -100  # e.g. prompt tokens during SFT
    logits, loss = model(idx, masked)
    expected = torch.nn.functional.cross_entropy(logits[0, 4:], targets[0, 4:])
    torch.testing.assert_close(loss, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_runs_on_mps():
    torch.manual_seed(0)
    model = Keera(TINY).to("mps")
    idx = torch.randint(0, TINY.vocab_size, (2, 16), device="mps")
    logits, loss = model(idx, idx)
    assert logits.device.type == "mps"
    loss.backward()
