import torch

from src.models.jev_anchors import anchor_loss, kl_top

OPT = [3, 7]
CTX = [0, 4]


def _logits(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(5, 200, generator=g) * scale


def test_zero_when_model_matches_reference():
    z = _logits(0)
    assert anchor_loss(z, z, decision=2, context=CTX, option_ids=OPT).abs() < 1e-6


def test_positive_when_they_differ():
    assert anchor_loss(_logits(0), _logits(1), decision=2, context=CTX, option_ids=OPT) > 0


def test_grows_with_divergence():
    z0 = _logits(0)
    near = anchor_loss(z0 + 0.01 * _logits(1), z0, 2, CTX, OPT)
    far = anchor_loss(z0 + 1.0 * _logits(1), z0, 2, CTX, OPT)
    assert far > near > 0


def test_kl_top_matches_full_kl_when_k_covers_vocabulary():
    a, b = torch.randn(8), torch.randn(8)
    full = (a.softmax(-1) * (a.log_softmax(-1) - b.log_softmax(-1))).sum()
    assert torch.allclose(kl_top(a, b, k=8), full, atol=1e-6)


def test_option_mass_shift_alone_is_penalized():
    """Moving mass onto the legal options, leaving the rest untouched, must cost something."""
    z0 = _logits(0)
    z = z0.clone()
    z[2, OPT] += 3.0
    assert anchor_loss(z, z0, 2, CTX, OPT) > 0.01
