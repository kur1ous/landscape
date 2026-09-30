import copy

import numpy as np
import pytest
import torch
from torch import nn

from landscape import Evaluator, along_direction, between_checkpoints, plot_curve, random_direction

CE = nn.CrossEntropyLoss()


def data(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    return [(x, (x.sum(1) > 0).long())]


def mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))


def loss_at_own_weights(model, loader):
    ev = Evaluator(model, CE, loader)
    return ev.loss(ev.theta0)


def test_quadratic_along_direction_is_exact():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    x = torch.tensor([[1.0], [2.0]])
    loader = [(x, x.squeeze(1))]  # perfect fit at alpha = 0
    mse = lambda out, y: ((out.squeeze(1) - y) ** 2).mean()
    curve = along_direction(model, mse, loader, [torch.tensor([[2.0]])], span=(-1, 1), steps=5)
    np.testing.assert_allclose(curve.losses, 4 * curve.alphas**2 * 2.5, atol=1e-6)


def test_batching_does_not_change_the_loss():
    model, (batch,) = mlp(), data()
    x, y = batch
    ragged = [(x[:5], y[:5]), (x[5:20], y[5:20]), (x[20:], y[20:])]
    assert loss_at_own_weights(model, ragged) == pytest.approx(loss_at_own_weights(model, [batch]), rel=1e-6)


def test_eval_mode_is_deterministic_and_state_is_restored():
    model, loader = mlp(), data()
    model.train()
    before = copy.deepcopy(model.state_dict())
    d = random_direction(model, seed=2)
    a = along_direction(model, CE, loader, d, steps=7)
    b = along_direction(model, CE, loader, d, steps=7)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert all(m.training for m in model.modules())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_between_checkpoints_hits_the_endpoints():
    loader = data()
    model, other = mlp(0), mlp(1)
    for (_, buf), (_, other_buf) in zip(model.named_buffers(), other.named_buffers()):
        other_buf.copy_(buf)  # interpolation keeps the model's buffers, so share them
    curve = between_checkpoints(
        model, CE, loader, copy.deepcopy(model.state_dict()), copy.deepcopy(other.state_dict()), steps=11
    )
    assert curve.marks == (0.0, 1.0)
    assert curve.losses[0] == pytest.approx(loss_at_own_weights(model, loader), rel=1e-6)
    assert curve.losses[-1] == pytest.approx(loss_at_own_weights(other, loader), rel=1e-6)


def test_wrong_flat_size_rejected():
    ev = Evaluator(mlp(), CE, data())
    with pytest.raises(ValueError):
        ev.loss(torch.zeros(3))


def test_empty_loader_rejected():
    with pytest.raises(ValueError):
        Evaluator(mlp(), CE, [])


def test_plot_curve_draws_and_saves(tmp_path):
    model = mlp()
    curve = along_direction(model, CE, data(), random_direction(model, seed=0), steps=9)
    ax = plot_curve([curve, curve], labels=["a", "b"])
    assert len(ax.lines) == 4  # a curve and its marker, twice
    ax.figure.savefig(tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000


def test_checkpoint_curve_labels_both_ends():
    model, other = mlp(0), mlp(1)
    curve = between_checkpoints(model, CE, data(), model.state_dict(), other.state_dict(), steps=5)
    ax = plot_curve(curve)
    assert [t.get_text() for t in ax.texts] == ["start", "end"]
