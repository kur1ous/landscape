import pytest
import torch
from torch import nn

from landscape import normalize_direction, random_direction


def conv_bn_net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 5, 3),
        nn.BatchNorm2d(5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(5 * 6 * 6, 4),
    )


def filter_norms(t):
    return t.reshape(t.shape[0], -1).norm(dim=1)


def test_filter_norms_match_weights():
    model = conv_bn_net()
    d = random_direction(model, seed=1)
    for (name, p), di in zip(model.named_parameters(), d):
        if p.dim() > 1:
            torch.testing.assert_close(filter_norms(di), filter_norms(p.detach()), msg=name)


def test_bias_and_bn_are_zeroed_by_default():
    model = conv_bn_net()
    d = random_direction(model, seed=1)
    seen = 0
    for (name, p), di in zip(model.named_parameters(), d):
        if p.dim() <= 1:
            assert torch.count_nonzero(di) == 0, name
            seen += 1
    assert seen == 4  # conv bias, bn scale, bn shift, linear bias


def test_bias_bn_keep_copies_the_weights():
    model = conv_bn_net()
    d = random_direction(model, seed=1, bias_bn="keep")
    for p, di in zip(model.parameters(), d):
        if p.dim() <= 1:
            torch.testing.assert_close(di, p.detach())


def test_bad_bias_bn_option():
    with pytest.raises(ValueError):
        random_direction(conv_bn_net(), bias_bn="ignore")


def test_seed_controls_direction():
    model = conv_bn_net()
    a, b, c = (random_direction(model, seed=s) for s in (3, 3, 4))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_does_not_touch_model_or_global_rng():
    model = conv_bn_net()
    before = [p.clone() for p in model.parameters()]
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    random_direction(model, seed=9)
    assert torch.equal(torch.rand(3), expected)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_zero_filter_stays_zero():
    w = torch.ones(3, 4)
    w[1] = 0
    (d,) = normalize_direction([torch.randn(3, 4)], [w])
    assert torch.count_nonzero(d[1]) == 0
    assert d[0].norm() == pytest.approx(w[0].norm())


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        normalize_direction([torch.zeros(2, 2)], [torch.zeros(3, 2)])
