import warnings

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from landscape import Evaluator, accuracy, along_direction, between_checkpoints, plane_slice


@pytest.fixture
def case():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Dropout(0.5))
    model[1].eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    x = torch.tensor([[3., 0.], [0., 2.], [1., 0.], [0., 1.], [2., 0.]])
    y = torch.tensor([0, 1, 0, 0, 1])
    return model, x, y


@pytest.mark.parametrize("vectorize", [True, False])
def test_accuracy_evaluation_weights_batches_and_preserves_state(case, vectorize):
    model, x, y = case
    before = model[0].weight.detach().clone()
    modes = [m.training for m in model.modules()]
    thetas = torch.stack([before.flatten(), -before.flatten()])
    expected_loss = torch.stack([F.cross_entropy(x, y), F.cross_entropy(-x, y)]).double()
    for size in (2, 5):
        loader = [(x[i:i + size], y[i:i + size]) for i in range(0, len(y), size)]
        ev = Evaluator(model, F.cross_entropy, loader, metric_fn=accuracy)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            loss, metric = ev.evaluate(thetas, vectorize=vectorize)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(metric, torch.tensor([0.6, 0.4], dtype=torch.float64))
        assert metric.device.type == "cpu"
    torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)
    assert [m.training for m in model.modules()] == modes
    ev = Evaluator(model, F.cross_entropy, [(x, y)])
    assert ev.evaluate(thetas, vectorize=vectorize)[1] is None
    torch.testing.assert_close(ev.losses(thetas, vectorize=vectorize), expected_loss)


@pytest.mark.parametrize("vectorize", [True, False])
def test_metric_failure_restores_weights_and_modes(case, vectorize):
    model, x, y = case
    before = model[0].weight.detach().clone()
    modes = [m.training for m in model.modules()]

    def broken(output, target):
        raise LookupError("metric failed")

    ev = Evaluator(model, F.cross_entropy, [(x, y)], metric_fn=broken)
    with pytest.raises(LookupError, match="metric failed"):
        ev.evaluate(-ev.theta0[None], vectorize=vectorize)
    torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)
    assert [m.training for m in model.modules()] == modes


def test_untraceable_metric_falls_back(case):
    model, x, y = case

    def item_metric(output, target):
        return torch.tensor(accuracy(output, target).item())

    ev = Evaluator(model, F.cross_entropy, [(x, y)], metric_fn=item_metric)
    with pytest.warns(RuntimeWarning, match="point by point"):
        loss, metric = ev.evaluate(-ev.theta0[None])
    assert metric.item() == pytest.approx(0.4)
    assert loss.item() == pytest.approx(F.cross_entropy(-x, y).item())
    torch.testing.assert_close(model[0].weight, torch.eye(2), rtol=0, atol=0)
    assert model.training and not model[1].training


@pytest.mark.parametrize("shape,target_shape", [((3,), (3,)), ((3, 2), (3, 1)),
                                                  ((3, 2), (2,)), ((0, 2), (0,))])
def test_accuracy_rejects_invalid_shapes(shape, target_shape):
    with pytest.raises(ValueError, match="accuracy"):
        accuracy(torch.zeros(shape), torch.zeros(target_shape, dtype=torch.long))


@pytest.mark.parametrize("metric_fn", [None, accuracy])
def test_metrics_propagate_to_curves_and_chunked_surfaces(case, metric_fn):
    model, x, y = case
    loader = [(x[:2], y[:2]), (x[2:], y[2:])]
    dx, dy = [-2 * torch.eye(2)], [torch.tensor([[0., 1.], [1., 0.]])]
    curve = along_direction(model, F.cross_entropy, loader, dx, span=(0, 1), steps=3,
                            metric_fn=metric_fn)
    between = between_checkpoints(model, F.cross_entropy, loader, model.state_dict(),
                                  {"0.weight": -torch.eye(2)}, steps=3, metric_fn=metric_fn)
    np.testing.assert_allclose(curve.losses, between.losses)
    for vectorize, chunk in ((True, 2), (False, 20)):
        surface = plane_slice(model, F.cross_entropy, loader, dx, dy, xspan=(0, 1),
                              steps=(3, 5), chunk=chunk, vectorize=vectorize, metric_fn=metric_fn)
        if metric_fn is None:
            assert curve.metric is between.metric is surface.metric is None
        else:
            np.testing.assert_allclose(curve.metric, [0.6, 0.6, 0.4])
            np.testing.assert_allclose(between.metric, curve.metric)
            assert surface.metric.shape == surface.losses.shape == (5, 3)
            expected = [[accuracy(F.linear(x, torch.eye(2) + a * dx[0] + b * dy[0]), y)
                         for a in surface.alphas] for b in surface.betas]
            np.testing.assert_allclose(surface.metric, expected)
