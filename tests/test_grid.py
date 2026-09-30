import numpy as np
import pytest
import torch
from torch import nn

from landscape import Evaluator, along_direction, plane_slice, plot_surface, random_direction

CE = nn.CrossEntropyLoss()


def data(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    return [(x[:25], (x[:25].sum(1) > 0).long()), (x[25:], (x[25:].sum(1) > 0).long())]


def mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))


def dirs(model):
    return random_direction(model, seed=1), random_direction(model, seed=2)


def test_vmapped_grid_matches_point_by_point():
    model, loader = mlp(), data()
    dx, dy = dirs(model)
    fast = plane_slice(model, CE, loader, dx, dy, steps=(7, 5), chunk=8)
    slow = plane_slice(model, CE, loader, dx, dy, steps=(7, 5), vectorize=False)
    np.testing.assert_allclose(fast.losses, slow.losses, rtol=1e-5, atol=1e-6)


def test_shape_axes_and_center():
    model, loader = mlp(), data()
    s = plane_slice(model, CE, loader, *dirs(model), steps=(9, 5))
    assert s.losses.shape == (5, 9)
    assert len(s.alphas) == 9 and len(s.betas) == 5
    ev = Evaluator(model, CE, loader)
    assert s.losses[2, 4] == pytest.approx(ev.loss(ev.theta0), rel=1e-6)


def test_axis_row_agrees_with_1d_curve():
    model, loader = mlp(), data()
    dx, dy = dirs(model)
    s = plane_slice(model, CE, loader, dx, dy, steps=9)
    curve = along_direction(model, CE, loader, dx, steps=9)
    np.testing.assert_allclose(s.losses[4], curve.losses, rtol=1e-5, atol=1e-6)


def test_chunk_size_does_not_matter():
    model, loader = mlp(), data()
    dx, dy = dirs(model)
    a = plane_slice(model, CE, loader, dx, dy, steps=6, chunk=1)
    b = plane_slice(model, CE, loader, dx, dy, steps=6, chunk=1000)
    np.testing.assert_allclose(a.losses, b.losses, rtol=1e-5, atol=1e-6)


def test_untraceable_loss_falls_back_with_warning():
    model, loader = mlp(), data()
    dx, dy = dirs(model)

    def item_loss(out, y):
        return torch.tensor(CE(out, y).item())  # .item() cannot be vmapped

    with pytest.warns(RuntimeWarning, match="point by point"):
        s = plane_slice(model, item_loss, loader, dx, dy, steps=3)
    ref = plane_slice(model, CE, loader, dx, dy, steps=3, vectorize=False)
    np.testing.assert_allclose(s.losses, ref.losses, rtol=1e-5, atol=1e-6)


def test_model_untouched_after_grid():
    model, loader = mlp(), data()
    model.train()
    before = [p.clone() for p in model.parameters()]
    plane_slice(model, CE, loader, *dirs(model), steps=3)
    assert all(m.training for m in model.modules())
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_too_few_steps_rejected():
    model, loader = mlp(), data()
    with pytest.raises(ValueError):
        plane_slice(model, CE, loader, *dirs(model), steps=1)


def test_plot_surface_draws_and_saves(tmp_path):
    model, loader = mlp(), data()
    s = plane_slice(model, CE, loader, *dirs(model), steps=15)
    ax = plot_surface(s)
    assert len(ax.collections) >= 2  # fill and lines
    ax.figure.savefig(tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000
